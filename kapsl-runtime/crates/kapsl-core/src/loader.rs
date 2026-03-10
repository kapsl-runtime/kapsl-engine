use flate2::read::GzDecoder;
use fs2::available_space;
use std::collections::HashSet;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use tar::Archive;
use tempfile::TempDir;
use thiserror::Error;

use crate::requirements::HardwareRequirements;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub project_name: String,
    pub framework: String,
    pub version: String,
    pub created_at: String,
    pub model_file: String,
    #[serde(default)]
    pub metadata: Option<serde_yaml::Value>,

    /// Hardware requirements for this model
    #[serde(default)]
    pub hardware_requirements: HardwareRequirements,
}

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML parsing error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Manifest not found in package")]
    ManifestMissing,
    #[error("package {package} references missing model file '{model_file}'")]
    ModelFileMissing {
        package: PathBuf,
        model_file: String,
    },
    #[error(
        "insufficient disk space for model cache at {} (required_copy={}B available={}B reserved_free={}B cache_usage={}B cache_limit={})",
        cache_root.display(),
        required_copy_bytes,
        available_bytes,
        reserved_free_bytes,
        cache_usage_bytes,
        max_cache_bytes.map(|v| format!("{v}B")).unwrap_or_else(|| "none".to_string())
    )]
    InsufficientDiskSpace {
        cache_root: PathBuf,
        required_copy_bytes: u64,
        available_bytes: u64,
        reserved_free_bytes: u64,
        cache_usage_bytes: u64,
        max_cache_bytes: Option<u64>,
    },
}

pub struct PackageLoader {
    _temp_dir: TempDir,
    pub extracted_path: PathBuf,
    pub manifest: Manifest,
    persisted_model_path: PathBuf,
}

impl PackageLoader {
    pub fn load(package_path: &Path) -> Result<Self, LoaderError> {
        let temp_dir = create_package_temp_dir(package_path)?;
        let extracted_path = temp_dir.path().to_path_buf();

        let file = File::open(package_path)?;
        let tar = GzDecoder::new(file);
        let mut archive = Archive::new(tar);

        archive.unpack(&extracted_path)?;

        log::info!("Extracted package to: {}", extracted_path.display());
        for entry in fs::read_dir(&extracted_path)? {
            let entry = entry?;
            let path = entry.path();
            let metadata = fs::metadata(&path)?;
            log::info!(
                "  - {} ({} bytes)",
                path.file_name().unwrap().to_string_lossy(),
                metadata.len()
            );
        }

        let manifest_path = extracted_path.join("metadata.json");
        if !manifest_path.exists() {
            return Err(LoaderError::ManifestMissing);
        }

        let manifest_file = File::open(manifest_path)?;
        let manifest: Manifest = match serde_json::from_reader(manifest_file) {
            Ok(m) => m,
            Err(e) => {
                return Err(LoaderError::Json(e));
            }
        };
        let model_path = extracted_path.join(&manifest.model_file);
        if !model_path.exists() || !model_path.is_file() {
            return Err(LoaderError::ModelFileMissing {
                package: package_path.to_path_buf(),
                model_file: manifest.model_file.clone(),
            });
        }
        let persisted_model_path =
            persist_model_file(package_path, &manifest.model_file, &model_path)?;

        Ok(Self {
            _temp_dir: temp_dir,
            extracted_path,
            manifest,
            persisted_model_path,
        })
    }

    /// Creates a synthetic PackageLoader for a raw model file (e.g. `.gguf`, `.onnx`) that is
    /// not wrapped in a `.aimod` archive. The manifest is inferred from the file's extension.
    pub fn from_raw_file(model_path: &Path) -> Result<Self, LoaderError> {
        let ext = model_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let framework = match ext.as_str() {
            "gguf" => "gguf",
            "onnx" => "onnx",
            _ => "onnx",
        };

        let model_file_name = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model")
            .to_string();

        let manifest = Manifest {
            project_name: model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string(),
            framework: framework.to_string(),
            version: "1.0.0".to_string(),
            created_at: String::new(),
            model_file: model_file_name,
            metadata: None,
            hardware_requirements: HardwareRequirements::default(),
        };

        let extracted_path = model_path
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();

        // A dummy TempDir satisfies the struct contract without touching the actual model file.
        let temp_dir = tempfile::tempdir()?;

        Ok(Self {
            _temp_dir: temp_dir,
            extracted_path,
            manifest,
            persisted_model_path: model_path.to_path_buf(),
        })
    }

    pub fn get_model_path(&self) -> PathBuf {
        self.persisted_model_path.clone()
    }
}

#[derive(Debug, Clone, Copy)]
struct CachePolicy {
    max_cache_bytes: Option<u64>,
    reserved_free_bytes: u64,
}

impl CachePolicy {
    fn from_env() -> Self {
        Self {
            max_cache_bytes: read_bytes_env(
                &[
                    "KAPSL_MODEL_CACHE_MAX_BYTES",
                    "KAPSL_LITE_MODEL_CACHE_MAX_BYTES",
                ],
                &[
                    "KAPSL_MODEL_CACHE_MAX_MIB",
                    "KAPSL_LITE_MODEL_CACHE_MAX_MIB",
                ],
            ),
            reserved_free_bytes: read_bytes_env(
                &[
                    "KAPSL_MODEL_CACHE_RESERVED_FREE_BYTES",
                    "KAPSL_LITE_MODEL_CACHE_RESERVED_FREE_BYTES",
                ],
                &[
                    "KAPSL_MODEL_CACHE_RESERVED_FREE_MIB",
                    "KAPSL_LITE_MODEL_CACHE_RESERVED_FREE_MIB",
                ],
            )
            .unwrap_or(0),
        }
    }
}

#[derive(Debug, Clone)]
struct CacheEntry {
    path: PathBuf,
    bytes: u64,
    modified_ns: u128,
}

#[derive(Debug, Clone)]
struct ModelAsset {
    source_path: PathBuf,
    target_name: String,
}

fn persist_model_file(
    package_path: &Path,
    model_file: &str,
    source_model_path: &Path,
) -> Result<PathBuf, LoaderError> {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    package_path.hash(&mut hasher);
    model_file.hash(&mut hasher);
    if let Ok(meta) = fs::metadata(package_path) {
        meta.len().hash(&mut hasher);
        if let Ok(modified) = meta.modified() {
            if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                duration.as_nanos().hash(&mut hasher);
            }
        }
    }
    let package_hash = hasher.finish();

    let file_name = Path::new(model_file)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("model.bin");
    let cache_root = resolve_model_cache_root(package_path);
    let cache_dir = cache_root.join(format!("{:016x}", package_hash));
    let cache_policy = CachePolicy::from_env();
    persist_model_assets(
        source_model_path,
        &cache_root,
        &cache_dir,
        file_name,
        cache_policy,
    )?;
    Ok(cache_dir.join(file_name))
}

fn resolve_model_cache_root(package_path: &Path) -> PathBuf {
    for key in ["KAPSL_MODEL_CACHE_DIR", "KAPSL_LITE_MODEL_CACHE_DIR"] {
        if let Some(value) = std::env::var_os(key) {
            if !value.is_empty() {
                return PathBuf::from(value);
            }
        }
    }

    if let Some(parent) = package_path.parent() {
        return parent.join(".kapsl-model-cache");
    }

    std::env::temp_dir().join("kapsl-model-cache")
}

fn persist_model_assets(
    source_model_path: &Path,
    cache_root: &Path,
    cache_dir: &Path,
    file_name: &str,
    cache_policy: CachePolicy,
) -> Result<PathBuf, LoaderError> {
    fs::create_dir_all(cache_root)?;
    fs::create_dir_all(cache_dir)?;

    let assets = collect_model_assets(source_model_path, file_name)?;
    let required_copy_bytes = estimate_required_copy_bytes(&assets, cache_dir)?;

    enforce_cache_policy(
        cache_root,
        cache_dir,
        required_copy_bytes,
        cache_policy,
        |path| available_space(path),
    )?;

    for asset in assets {
        copy_if_needed(&asset.source_path, &cache_dir.join(asset.target_name))?;
    }

    Ok(cache_dir.join(file_name))
}

fn collect_model_assets(
    source_model_path: &Path,
    model_target_name: &str,
) -> Result<Vec<ModelAsset>, LoaderError> {
    let mut assets = Vec::new();
    let mut seen_targets = HashSet::new();

    assets.push(ModelAsset {
        source_path: source_model_path.to_path_buf(),
        target_name: model_target_name.to_string(),
    });
    seen_targets.insert(model_target_name.to_string());

    let Some(source_dir) = source_model_path.parent() else {
        return Ok(assets);
    };
    let main_model_name = source_model_path.file_name();

    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.file_name() == main_model_name {
            continue;
        }
        let target_name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if target_name.is_empty() || !seen_targets.insert(target_name.clone()) {
            continue;
        }
        assets.push(ModelAsset {
            source_path: path,
            target_name,
        });
    }

    Ok(assets)
}

fn estimate_required_copy_bytes(
    assets: &[ModelAsset],
    cache_dir: &Path,
) -> Result<u64, LoaderError> {
    let mut required_bytes = 0u64;
    for asset in assets {
        let source_meta = fs::metadata(&asset.source_path)?;
        let target_path = cache_dir.join(&asset.target_name);
        if should_copy_file(source_meta.len(), &target_path)? {
            required_bytes = required_bytes.saturating_add(source_meta.len());
        }
    }
    Ok(required_bytes)
}

fn should_copy_file(source_size: u64, target_path: &Path) -> Result<bool, LoaderError> {
    let metadata = match fs::metadata(target_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(true),
        Err(error) => return Err(LoaderError::Io(error)),
    };

    if !metadata.is_file() {
        return Ok(true);
    }
    Ok(metadata.len() != source_size)
}

fn copy_if_needed(source_path: &Path, target_path: &Path) -> Result<(), LoaderError> {
    let source_size = fs::metadata(source_path)?.len();
    if !should_copy_file(source_size, target_path)? {
        return Ok(());
    }
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(source_path, target_path)?;
    Ok(())
}

fn enforce_cache_policy<F>(
    cache_root: &Path,
    protected_cache_dir: &Path,
    required_copy_bytes: u64,
    cache_policy: CachePolicy,
    mut query_available_space: F,
) -> Result<(), LoaderError>
where
    F: FnMut(&Path) -> std::io::Result<u64>,
{
    if cache_policy.max_cache_bytes.is_none() && cache_policy.reserved_free_bytes == 0 {
        return Ok(());
    }

    fs::create_dir_all(cache_root)?;
    let mut entries = load_cache_entries(cache_root, Some(protected_cache_dir))?;
    let mut cache_usage_bytes = directory_size_bytes(cache_root)?;

    loop {
        let available_bytes = query_available_space(cache_root)?;
        let required_available_bytes = cache_policy
            .reserved_free_bytes
            .saturating_add(required_copy_bytes);
        let free_ok = available_bytes >= required_available_bytes;
        let cap_ok = match cache_policy.max_cache_bytes {
            Some(max_bytes) => cache_usage_bytes.saturating_add(required_copy_bytes) <= max_bytes,
            None => true,
        };

        if free_ok && cap_ok {
            return Ok(());
        }

        let Some(entry) = entries.first().cloned() else {
            return Err(LoaderError::InsufficientDiskSpace {
                cache_root: cache_root.to_path_buf(),
                required_copy_bytes,
                available_bytes,
                reserved_free_bytes: cache_policy.reserved_free_bytes,
                cache_usage_bytes,
                max_cache_bytes: cache_policy.max_cache_bytes,
            });
        };
        entries.remove(0);
        remove_cache_entry(&entry.path)?;
        cache_usage_bytes = cache_usage_bytes.saturating_sub(entry.bytes);
    }
}

fn load_cache_entries(
    cache_root: &Path,
    protected_path: Option<&Path>,
) -> Result<Vec<CacheEntry>, LoaderError> {
    if !cache_root.exists() {
        return Ok(Vec::new());
    }

    let mut entries = Vec::new();
    for entry in fs::read_dir(cache_root)? {
        let entry = entry?;
        let path = entry.path();
        if protected_path.is_some_and(|protected| protected == path) {
            continue;
        }
        let metadata = entry.metadata()?;
        let bytes = if metadata.is_dir() {
            directory_size_bytes(&path)?
        } else if metadata.is_file() {
            metadata.len()
        } else {
            0
        };
        entries.push(CacheEntry {
            path,
            bytes,
            modified_ns: modified_ns(&metadata),
        });
    }

    entries.sort_by(|a, b| {
        a.modified_ns
            .cmp(&b.modified_ns)
            .then_with(|| a.path.cmp(&b.path))
    });
    Ok(entries)
}

fn remove_cache_entry(path: &Path) -> Result<(), LoaderError> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(LoaderError::Io(error)),
    };

    if metadata.is_dir() {
        fs::remove_dir_all(path)?;
    } else {
        fs::remove_file(path)?;
    }
    Ok(())
}

fn directory_size_bytes(path: &Path) -> Result<u64, LoaderError> {
    let metadata = fs::metadata(path)?;
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    if !metadata.is_dir() {
        return Ok(0);
    }

    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        total = total.saturating_add(directory_size_bytes(&entry.path())?);
    }
    Ok(total)
}

fn modified_ns(metadata: &fs::Metadata) -> u128 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn read_bytes_env(byte_keys: &[&str], mib_keys: &[&str]) -> Option<u64> {
    for key in byte_keys {
        if let Some(value) = optional_u64_env(key) {
            return Some(value);
        }
    }
    for key in mib_keys {
        if let Some(value_mib) = optional_u64_env(key) {
            return Some(value_mib.saturating_mul(1024 * 1024));
        }
    }
    None
}

fn optional_u64_env(key: &str) -> Option<u64> {
    let value = std::env::var(key).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<u64>().ok()
}

fn create_package_temp_dir(package_path: &Path) -> Result<TempDir, LoaderError> {
    let mut candidate_roots = Vec::new();

    if let Some(configured_root) = configured_package_temp_root() {
        candidate_roots.push(configured_root);
    }
    if let Some(parent_dir) = package_path.parent() {
        candidate_roots.push(parent_dir.join(".kapsl-package-tmp"));
    }

    for root in candidate_roots {
        if let Err(error) = fs::create_dir_all(&root) {
            log::warn!(
                "Failed to create package temp directory {}: {}",
                root.display(),
                error
            );
            continue;
        }
        match TempDir::new_in(&root) {
            Ok(temp_dir) => return Ok(temp_dir),
            Err(error) => log::warn!(
                "Failed to create package temp dir in {}: {}",
                root.display(),
                error
            ),
        }
    }

    Ok(TempDir::new()?)
}

fn configured_package_temp_root() -> Option<PathBuf> {
    for key in ["KAPSL_PACKAGE_TMP_DIR", "KAPSL_LITE_PACKAGE_TMP_DIR"] {
        let Some(value) = std::env::var_os(key) else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        return Some(PathBuf::from(value));
    }
    None
}

#[path = "loader_tests.rs"]
mod loader_tests;
