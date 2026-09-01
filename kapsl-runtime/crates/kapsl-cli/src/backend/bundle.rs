//! Offline `.kapsl-bundle` creation and activation.
//!
//! Bundles carry the exact signed backend index and pack archives used on the
//! preparation machine. The offline runtime feeds those artifacts back through
//! `BackendManager`; there is no weaker import-only verification path.

use crate::app::BundleCommandArgs;
use crate::backend::{
    backend_cache_root, current_platform, runtime_release_version, safe_extract_tar_gz,
    sha256_file, BackendAccelerator, BackendIndex, BackendManager, BackendPackManifest,
    BackendTarget, LlamaCppBackendPackProfile, OnnxBackendPackProfile, MANAGED_VLLM_PACK_PROFILE,
};
use crate::backend::{
    inspect_serving_manifest, resolve_serving_backend, validate_model_contract,
    validate_serving_backend_declaration, ResolvedServingBackend,
};
use crate::backend::{
    llama_cpp_lazy_packs_supported_for_platform, llama_cpp_pack_profile_for_target,
};
use crate::backend::{onnx_lazy_packs_supported_for_platform, onnx_pack_profile_for_target};
use crate::DynError;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use fs2::FileExt;
use kapsl_hal::device::DeviceInfo;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tar::{Archive, Builder};

const BUNDLE_SCHEMA_VERSION: u32 = 1;
const BUNDLE_MANIFEST_NAME: &str = "bundle-manifest.json";
const BUNDLE_INDEX_PATH: &str = "backend-index.json";
const BUNDLE_INDEX_SIGNATURE_PATH: &str = "signatures/backend-index.json.sig";
const BUNDLE_CHECKSUMS_PATH: &str = "checksums/SHA256SUMS";
const BUNDLE_LICENSES_PATH: &str = "licenses/backend-notices.json";
const BUNDLE_CACHE_ENV: &str = "KAPSL_BUNDLE_CACHE_DIR";
const MAX_BUNDLE_MANIFEST_BYTES: u64 = 2 * 1024 * 1024;
const MAX_BUNDLE_PAYLOAD_BYTES: u64 = 1024 * 1024 * 1024 * 1024;
const BUNDLE_EXTRACTION_OVERHEAD_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct BundleTarget {
    platform: String,
    architecture: String,
    accelerator: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct BundledModel {
    name: String,
    path: String,
    sha256: String,
    bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct BundledBackend {
    backend: String,
    profile: String,
    path: String,
    sha256: String,
    bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct BundleManifest {
    schema_version: u32,
    runtime_version: String,
    created_at_unix_seconds: u64,
    target: BundleTarget,
    total_payload_bytes: u64,
    backend_index_sha256: String,
    backend_index_signature_sha256: String,
    checksums_sha256: String,
    licenses_sha256: String,
    models: Vec<BundledModel>,
    backends: Vec<BundledBackend>,
}

struct PreparedPack {
    manifest: BackendPackManifest,
    archive: PathBuf,
    bundle_path: String,
}

pub(crate) fn execute_bundle_command(args: BundleCommandArgs) -> Result<(), DynError> {
    let detected = DeviceInfo::probe();
    let manager = BackendManager::from_env(false)?;
    create_bundle(&args, &detected, &manager)
}

fn create_bundle(
    args: &BundleCommandArgs,
    detected: &DeviceInfo,
    manager: &BackendManager,
) -> Result<(), DynError> {
    if args.output.exists() {
        return Err(format!(
            "Refusing to overwrite existing bundle {}; choose a new --output path",
            args.output.display()
        )
        .into());
    }
    let output_parent = args.output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_parent)?;

    let (target, target_manifest) = resolve_bundle_target(args.target.as_deref(), detected)?;
    let (index, index_bytes, index_signature) = manager.verified_index_material()?;

    let scratch = tempfile::tempdir_in(output_parent)?;
    let mut models = Vec::with_capacity(args.model.len());
    let mut model_sources = Vec::with_capacity(args.model.len());
    let mut required_packs = BTreeMap::<(String, String), BackendPackManifest>::new();
    let mut used_paths = HashSet::new();

    for (position, model_path) in args.model.iter().enumerate() {
        let absolute = model_path
            .canonicalize()
            .map_err(|error| format!("Invalid model path {}: {error}", model_path.display()))?;
        if !absolute.is_file()
            || absolute
                .extension()
                .and_then(|value| value.to_str())
                .is_none_or(|value| !value.eq_ignore_ascii_case("aimod"))
        {
            return Err(format!(
                "Offline bundles currently accept .aimod package files; got {}",
                absolute.display()
            )
            .into());
        }
        let manifest = inspect_serving_manifest(&absolute)?;
        validate_model_contract(&manifest)?;
        validate_serving_backend_declaration(&manifest)?;
        let decision =
            resolve_serving_backend(&manifest, target.accelerator != BackendAccelerator::Cpu)?;
        if decision.selected == ResolvedServingBackend::Vllm {
            let pack =
                manager.select_pack(&index, "vllm", Some(MANAGED_VLLM_PACK_PROFILE), &target)?;
            required_packs.insert((pack.backend.clone(), pack.profile.clone()), pack);
        }
        if onnx_lazy_packs_supported_for_platform(&target.platform) {
            if let Some(profile) = onnx_pack_profile_for_target(&manifest, target.accelerator)? {
                let pack_target = target_for_onnx_profile(&target, profile);
                let pack =
                    manager.select_pack(&index, "onnx", Some(profile.profile()), &pack_target)?;
                required_packs.insert((pack.backend.clone(), pack.profile.clone()), pack);
            }
        }
        if llama_cpp_lazy_packs_supported_for_platform(&target.platform) {
            if let Some(profile) = llama_cpp_pack_profile_for_target(&manifest, target.accelerator)
            {
                let pack_target = target_for_llama_profile(&target, profile);
                let pack = manager.select_pack(
                    &index,
                    "llama-cpp",
                    Some(profile.profile()),
                    &pack_target,
                )?;
                required_packs.insert((pack.backend.clone(), pack.profile.clone()), pack);
            }
        }

        let file_name = absolute
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("model.aimod");
        let safe_name = sanitize_bundle_name(file_name);
        let mut bundle_path = format!("models/{position:03}-{safe_name}");
        while !used_paths.insert(bundle_path.clone()) {
            bundle_path = format!("models/{position:03}-{}-{safe_name}", used_paths.len());
        }
        let bytes = fs::metadata(&absolute)?.len();
        models.push(BundledModel {
            name: manifest.project_name,
            path: bundle_path,
            sha256: sha256_file(&absolute)?.to_ascii_lowercase(),
            bytes,
        });
        model_sources.push(absolute);
    }

    let mut prepared_packs = Vec::with_capacity(required_packs.len());
    for (position, (_, pack)) in required_packs.into_iter().enumerate() {
        // Use a path owned by the surrounding TempDir rather than keeping an
        // open NamedTempFile handle. Windows otherwise rejects the manager's
        // create/truncate call during cross-platform bundle preparation.
        let archive = scratch.path().join(format!("backend-{position}.tar.gz"));
        manager.fetch_pack_archive(&pack, &archive)?;
        let bundle_path = format!(
            "backends/{}/{}.tar.gz",
            sanitize_bundle_name(&pack.backend),
            sanitize_bundle_name(&pack.profile)
        );
        prepared_packs.push(PreparedPack {
            manifest: pack,
            archive,
            bundle_path,
        });
    }

    let bundled_backends = prepared_packs
        .iter()
        .map(|prepared| BundledBackend {
            backend: prepared.manifest.backend.clone(),
            profile: prepared.manifest.profile.clone(),
            path: prepared.bundle_path.clone(),
            sha256: prepared.manifest.sha256.to_ascii_lowercase(),
            bytes: prepared.manifest.download_bytes,
        })
        .collect::<Vec<_>>();
    let license_bytes = serde_json::to_vec_pretty(
        &prepared_packs
            .iter()
            .map(|prepared| {
                serde_json::json!({
                    "backend": prepared.manifest.backend,
                    "profile": prepared.manifest.profile,
                    "notices": prepared.manifest.licenses,
                })
            })
            .collect::<Vec<_>>(),
    )?;
    let index_signature_bytes = index_signature.as_bytes();

    let mut checksum_rows = Vec::new();
    checksum_rows.push((sha256_bytes(&index_bytes), BUNDLE_INDEX_PATH.to_string()));
    checksum_rows.push((
        sha256_bytes(index_signature_bytes),
        BUNDLE_INDEX_SIGNATURE_PATH.to_string(),
    ));
    checksum_rows.push((
        sha256_bytes(&license_bytes),
        BUNDLE_LICENSES_PATH.to_string(),
    ));
    checksum_rows.extend(
        models
            .iter()
            .map(|model| (model.sha256.clone(), model.path.clone())),
    );
    checksum_rows.extend(
        bundled_backends
            .iter()
            .map(|backend| (backend.sha256.clone(), backend.path.clone())),
    );
    checksum_rows.sort_by(|left, right| left.1.cmp(&right.1));
    let checksum_bytes = checksum_rows
        .iter()
        .map(|(digest, path)| format!("{digest}  {path}\n"))
        .collect::<String>()
        .into_bytes();

    let total_payload_bytes = models
        .iter()
        .map(|model| model.bytes)
        .chain(bundled_backends.iter().map(|backend| backend.bytes))
        .chain([
            index_bytes.len() as u64,
            index_signature_bytes.len() as u64,
            license_bytes.len() as u64,
            checksum_bytes.len() as u64,
        ])
        .fold(0_u64, u64::saturating_add);
    let bundle_manifest = BundleManifest {
        schema_version: BUNDLE_SCHEMA_VERSION,
        runtime_version: manager.runtime_version().to_string(),
        created_at_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        target: target_manifest,
        total_payload_bytes,
        backend_index_sha256: sha256_bytes(&index_bytes),
        backend_index_signature_sha256: sha256_bytes(index_signature_bytes),
        checksums_sha256: sha256_bytes(&checksum_bytes),
        licenses_sha256: sha256_bytes(&license_bytes),
        models,
        backends: bundled_backends,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&bundle_manifest)?;

    let temporary = tempfile::NamedTempFile::new_in(output_parent)?;
    let output_file = temporary.reopen()?;
    let encoder = GzEncoder::new(output_file, Compression::fast());
    let mut archive = Builder::new(encoder);
    append_bytes(&mut archive, BUNDLE_MANIFEST_NAME, &manifest_bytes, 0o644)?;
    append_bytes(&mut archive, BUNDLE_INDEX_PATH, &index_bytes, 0o644)?;
    append_bytes(
        &mut archive,
        BUNDLE_INDEX_SIGNATURE_PATH,
        index_signature_bytes,
        0o644,
    )?;
    append_bytes(&mut archive, BUNDLE_CHECKSUMS_PATH, &checksum_bytes, 0o644)?;
    append_bytes(&mut archive, BUNDLE_LICENSES_PATH, &license_bytes, 0o644)?;
    for (model, source) in bundle_manifest.models.iter().zip(model_sources.iter()) {
        append_file(&mut archive, &model.path, source, 0o644)?;
    }
    for prepared in &prepared_packs {
        append_file(
            &mut archive,
            &prepared.bundle_path,
            &prepared.archive,
            0o644,
        )?;
    }
    archive.finish()?;
    let encoder = archive.into_inner()?;
    let output_file = encoder.finish()?;
    output_file.sync_all()?;
    drop(output_file);
    temporary
        .persist_noclobber(&args.output)
        .map_err(|error| error.error)?;

    println!(
        "Created {} for {} model(s) with {} deduplicated backend pack(s) targeting {}-{}.",
        args.output.display(),
        bundle_manifest.models.len(),
        bundle_manifest.backends.len(),
        bundle_manifest.target.platform,
        bundle_manifest.target.accelerator
    );
    Ok(())
}

/// Resolve any bundle arguments to persistent embedded `.aimod` paths. Bundle
/// caches remain valid for the lifetime of the server and across later runs.
pub(crate) fn expand_run_bundles(
    inputs: &[PathBuf],
    device_info: &DeviceInfo,
) -> Result<Vec<PathBuf>, DynError> {
    let mut resolved = Vec::new();
    for input in inputs {
        if input
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("kapsl-bundle"))
        {
            resolved.extend(activate_bundle(input, device_info)?);
        } else {
            resolved.push(input.clone());
        }
    }
    Ok(resolved)
}

fn activate_bundle(bundle: &Path, device_info: &DeviceInfo) -> Result<Vec<PathBuf>, DynError> {
    let cache_root = bundle_cache_root()?;
    let manager = BackendManager::from_env(true)?;
    activate_bundle_with_manager(bundle, device_info, &cache_root, &manager)
}

fn activate_bundle_with_manager(
    bundle: &Path,
    device_info: &DeviceInfo,
    cache_root: &Path,
    manager: &BackendManager,
) -> Result<Vec<PathBuf>, DynError> {
    let absolute = bundle
        .canonicalize()
        .map_err(|error| format!("Invalid offline bundle path {}: {error}", bundle.display()))?;
    let bundle_digest = sha256_file(&absolute)?;
    fs::create_dir_all(cache_root)?;
    let locks = cache_root.join(".locks");
    fs::create_dir_all(&locks)?;
    let lock_path = locks.join(format!("{bundle_digest}.lock"));
    let process_lock = bundle_process_path_lock(&lock_path);
    let _process_guard = process_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)?;
    lock.lock_exclusive()?;

    let final_root = cache_root.join(&bundle_digest);
    if final_root.is_dir() {
        match validate_extracted_bundle(&final_root, device_info, manager) {
            Ok(models) => {
                let _ = FileExt::unlock(&lock);
                return Ok(models);
            }
            Err(error) => {
                let quarantine =
                    cache_root.join(format!(".invalid-{}-{}", bundle_digest, unique_nonce()));
                fs::rename(&final_root, quarantine).map_err(|rename_error| {
                    format!(
                        "Cached bundle {} is invalid ({error}) and could not be quarantined: {rename_error}",
                        final_root.display()
                    )
                })?;
            }
        }
    }

    let manifest = inspect_bundle_manifest(&absolute)?;
    validate_bundle_manifest_shape(&manifest)?;
    let required_space = manifest
        .total_payload_bytes
        .saturating_add(BUNDLE_EXTRACTION_OVERHEAD_BYTES);
    let available = fs2::available_space(cache_root)?;
    if required_space > available {
        return Err(format!(
            "Insufficient disk space for offline bundle: required {required_space} bytes, available {available} at {}",
            cache_root.display()
        )
        .into());
    }
    let stage = tempfile::Builder::new()
        .prefix(&format!(".bundle-{bundle_digest}-"))
        .tempdir_in(cache_root)?;
    safe_extract_tar_gz(
        &absolute,
        stage.path(),
        required_space.saturating_add(MAX_BUNDLE_MANIFEST_BYTES),
    )?;
    let stage_root = stage.path().to_path_buf();
    let staged_models = validate_extracted_bundle(&stage_root, device_info, manager)?;
    let model_paths = staged_models
        .iter()
        .map(|path| {
            path.strip_prefix(&stage_root)
                .map(Path::to_path_buf)
                .map_err(|error| format!("Resolve staged bundle model path: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    fs::rename(stage.keep(), &final_root)?;
    let _ = FileExt::unlock(&lock);
    Ok(model_paths
        .into_iter()
        .map(|relative| final_root.join(relative))
        .collect())
}

fn validate_extracted_bundle(
    root: &Path,
    device_info: &DeviceInfo,
    manager: &BackendManager,
) -> Result<Vec<PathBuf>, DynError> {
    let manifest = read_bundle_manifest_file(&root.join(BUNDLE_MANIFEST_NAME))?;
    validate_bundle_manifest_shape(&manifest)?;
    if manifest.runtime_version != runtime_release_version() {
        return Err(format!(
            "Offline bundle targets Kapsl {}, but this runtime is {}",
            manifest.runtime_version,
            runtime_release_version()
        )
        .into());
    }
    let target = target_for_bundle_host(&manifest.target, device_info)?;

    let index_path = root.join(BUNDLE_INDEX_PATH);
    let signature_path = root.join(BUNDLE_INDEX_SIGNATURE_PATH);
    verify_digest(
        &root.join(BUNDLE_CHECKSUMS_PATH),
        &manifest.checksums_sha256,
        "checksum inventory",
    )?;
    verify_checksum_inventory(root, &manifest)?;
    verify_bundle_tree(root, &manifest)?;
    let index_bytes = read_bounded(&index_path, 8 * 1024 * 1024)?;
    let signature = String::from_utf8(read_bounded(&signature_path, 64 * 1024)?)?;
    let index: BackendIndex = manager.cache_verified_index_bytes(&index_bytes, signature.trim())?;

    let mut backend_ids = HashSet::new();
    let mut selected_packs = Vec::with_capacity(manifest.backends.len());
    for bundled in &manifest.backends {
        validate_bundle_relative_path(&bundled.path)?;
        if !backend_ids.insert((bundled.backend.clone(), bundled.profile.clone())) {
            return Err(format!(
                "Offline bundle repeats backend {}/{}",
                bundled.backend, bundled.profile
            )
            .into());
        }
        let pack_target = target_for_backend_identity(&target, &bundled.backend, &bundled.profile);
        let pack = manager.select_pack(
            &index,
            &bundled.backend,
            Some(&bundled.profile),
            &pack_target,
        )?;
        if !pack.sha256.eq_ignore_ascii_case(&bundled.sha256)
            || pack.download_bytes != bundled.bytes
        {
            return Err(format!(
                "Bundled backend {}/{} does not match its signed index entry",
                bundled.backend, bundled.profile
            )
            .into());
        }
        let archive_path = root.join(&bundled.path);
        selected_packs.push((pack, archive_path));
    }

    let mut models = Vec::with_capacity(manifest.models.len());
    let mut model_paths = HashSet::new();
    let mut required_backend_ids = HashSet::new();
    for model in &manifest.models {
        validate_bundle_relative_path(&model.path)?;
        if !model_paths.insert(model.path.clone()) {
            return Err(format!("Offline bundle repeats model path {}", model.path).into());
        }
        let path = root.join(&model.path);
        if fs::metadata(&path)?.len() != model.bytes {
            return Err(format!("Bundled model {} has the wrong size", model.name).into());
        }
        // Validate the embedded model contract before exposing its path to the
        // ordinary startup planner.
        let embedded_manifest = inspect_serving_manifest(&path)?;
        validate_model_contract(&embedded_manifest)?;
        validate_serving_backend_declaration(&embedded_manifest)?;
        let decision = resolve_serving_backend(
            &embedded_manifest,
            target.accelerator != BackendAccelerator::Cpu,
        )?;
        if decision.selected == ResolvedServingBackend::Vllm {
            required_backend_ids
                .insert(("vllm".to_string(), MANAGED_VLLM_PACK_PROFILE.to_string()));
        }
        if onnx_lazy_packs_supported_for_platform(&target.platform) {
            if let Some(profile) =
                onnx_pack_profile_for_target(&embedded_manifest, target.accelerator)?
            {
                required_backend_ids.insert(("onnx".to_string(), profile.profile().to_string()));
            }
        }
        if llama_cpp_lazy_packs_supported_for_platform(&target.platform) {
            if let Some(profile) =
                llama_cpp_pack_profile_for_target(&embedded_manifest, target.accelerator)
            {
                required_backend_ids
                    .insert(("llama-cpp".to_string(), profile.profile().to_string()));
            }
        }
        models.push(path);
    }
    if backend_ids != required_backend_ids {
        let missing = required_backend_ids
            .difference(&backend_ids)
            .cloned()
            .collect::<Vec<_>>();
        let unexpected = backend_ids
            .difference(&required_backend_ids)
            .cloned()
            .collect::<Vec<_>>();
        return Err(format!(
            "Offline bundle backend closure does not match its models (missing: {missing:?}, unexpected: {unexpected:?})"
        )
        .into());
    }
    // No pack is installed until every model contract and the exact backend
    // closure have passed verification.
    for (pack, archive_path) in selected_packs {
        manager.install_pack_from_archive(&pack, &archive_path)?;
    }
    Ok(models)
}

fn target_for_onnx_profile(
    target: &BackendTarget,
    profile: OnnxBackendPackProfile,
) -> BackendTarget {
    let mut resolved = target.clone();
    resolved.accelerator = profile.accelerator();
    if profile == OnnxBackendPackProfile::Cpu {
        resolved.cuda_version = None;
        resolved.driver_version = None;
    }
    resolved
}

fn target_for_llama_profile(
    target: &BackendTarget,
    profile: LlamaCppBackendPackProfile,
) -> BackendTarget {
    let mut resolved = target.clone();
    resolved.accelerator = profile.accelerator();
    if profile == LlamaCppBackendPackProfile::Cpu {
        resolved.cuda_version = None;
        resolved.driver_version = None;
    }
    resolved
}

fn target_for_backend_identity(
    target: &BackendTarget,
    backend: &str,
    profile: &str,
) -> BackendTarget {
    if backend == "llama-cpp" {
        let profile = match profile {
            crate::backend::LLAMA_CPP_CPU_PACK_PROFILE => Some(LlamaCppBackendPackProfile::Cpu),
            crate::backend::LLAMA_CPP_CUDA12_PACK_PROFILE => {
                Some(LlamaCppBackendPackProfile::Cuda12)
            }
            _ => None,
        };
        return profile
            .map(|profile| target_for_llama_profile(target, profile))
            .unwrap_or_else(|| target.clone());
    }
    if backend != "onnx" {
        return target.clone();
    }
    let profile = match profile {
        crate::backend::ONNX_CPU_PACK_PROFILE => Some(OnnxBackendPackProfile::Cpu),
        crate::backend::ONNX_CUDA12_PACK_PROFILE => Some(OnnxBackendPackProfile::Cuda12),
        crate::backend::ONNX_TENSORRT10_PACK_PROFILE => Some(OnnxBackendPackProfile::TensorRt10),
        _ => None,
    };
    profile
        .map(|profile| target_for_onnx_profile(target, profile))
        .unwrap_or_else(|| target.clone())
}

fn resolve_bundle_target(
    requested: Option<&str>,
    device_info: &DeviceInfo,
) -> Result<(BackendTarget, BundleTarget), DynError> {
    if let Some(requested) = requested {
        let (platform, accelerator) = [
            ("-tensorrt", BackendAccelerator::TensorRt),
            ("-cuda", BackendAccelerator::Cuda),
            ("-cpu", BackendAccelerator::Cpu),
        ]
        .into_iter()
        .find_map(|(suffix, accelerator)| {
            requested
                .strip_suffix(suffix)
                .map(|platform| (platform.to_string(), accelerator))
        })
        .ok_or_else(|| {
            format!(
                "Invalid bundle target `{requested}`; expected <os>-<architecture>-cpu, -cuda, or -tensorrt"
            )
        })?;
        let architecture = platform
            .rsplit_once('-')
            .map(|(_, architecture)| architecture.to_string())
            .ok_or_else(|| format!("Invalid bundle platform `{platform}`"))?;
        let target = BackendTarget {
            platform: platform.clone(),
            architecture: architecture.clone(),
            accelerator,
            // A cross-target names a runtime family, not a particular host.
            // Actual minimum versions are enforced again on bundle activation.
            cuda_version: (accelerator != BackendAccelerator::Cpu).then(|| "999.0".to_string()),
            driver_version: (accelerator != BackendAccelerator::Cpu).then(|| "9999.0".to_string()),
        };
        return Ok((
            target,
            BundleTarget {
                platform,
                architecture,
                accelerator: accelerator.as_str().to_string(),
            },
        ));
    }

    let target = BackendTarget::current(device_info);
    let manifest = BundleTarget {
        platform: target.platform.clone(),
        architecture: target.architecture.clone(),
        accelerator: target.accelerator.as_str().to_string(),
    };
    Ok((target, manifest))
}

fn target_for_bundle_host(
    declared: &BundleTarget,
    device_info: &DeviceInfo,
) -> Result<BackendTarget, DynError> {
    if declared.platform != current_platform() {
        return Err(format!(
            "Offline bundle targets {}, but this host is {}",
            declared.platform,
            current_platform()
        )
        .into());
    }
    let platform_architecture = declared
        .platform
        .rsplit_once('-')
        .map(|(_, architecture)| architecture)
        .ok_or_else(|| format!("Invalid offline bundle platform `{}`", declared.platform))?;
    if normalize_bundle_architecture(platform_architecture)
        != normalize_bundle_architecture(&declared.architecture)
        || normalize_bundle_architecture(&declared.architecture)
            != normalize_bundle_architecture(std::env::consts::ARCH)
    {
        return Err(format!(
            "Offline bundle architecture {} does not match platform {} or host {}",
            declared.architecture,
            declared.platform,
            std::env::consts::ARCH
        )
        .into());
    }
    let accelerator = match declared.accelerator.as_str() {
        "cpu" => BackendAccelerator::Cpu,
        "cuda" => BackendAccelerator::Cuda,
        "tensorrt" => BackendAccelerator::TensorRt,
        other => return Err(format!("Unsupported bundle accelerator `{other}`").into()),
    };
    if accelerator != BackendAccelerator::Cpu && !device_info.has_cuda {
        return Err(format!(
            "Offline bundle requires {}, but no CUDA device is available",
            declared.accelerator
        )
        .into());
    }
    let mut target = BackendTarget::current(device_info);
    target.accelerator = accelerator;
    target.platform = declared.platform.clone();
    target.architecture = declared.architecture.clone();
    Ok(target)
}

fn normalize_bundle_architecture(value: &str) -> &str {
    match value {
        "amd64" => "x86_64",
        "arm64" => "aarch64",
        other => other,
    }
}

fn inspect_bundle_manifest(path: &Path) -> Result<BundleManifest, DynError> {
    let file = File::open(path)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    for (position, entry) in archive.entries()?.enumerate() {
        if position > 64 {
            break;
        }
        let entry = entry?;
        if entry.path()?.as_ref() == Path::new(BUNDLE_MANIFEST_NAME) {
            if entry.size() > MAX_BUNDLE_MANIFEST_BYTES {
                return Err("Offline bundle manifest exceeds its size limit".into());
            }
            let mut bytes = Vec::new();
            entry
                .take(MAX_BUNDLE_MANIFEST_BYTES + 1)
                .read_to_end(&mut bytes)?;
            return Ok(serde_json::from_slice(&bytes)?);
        }
    }
    Err(format!("{} does not contain {BUNDLE_MANIFEST_NAME}", path.display()).into())
}

fn read_bundle_manifest_file(path: &Path) -> Result<BundleManifest, DynError> {
    Ok(serde_json::from_slice(&read_bounded(
        path,
        MAX_BUNDLE_MANIFEST_BYTES,
    )?)?)
}

fn validate_bundle_manifest_shape(manifest: &BundleManifest) -> Result<(), DynError> {
    if manifest.schema_version != BUNDLE_SCHEMA_VERSION {
        return Err(format!(
            "Unsupported offline bundle schema {}; expected {}",
            manifest.schema_version, BUNDLE_SCHEMA_VERSION
        )
        .into());
    }
    if manifest.models.is_empty() {
        return Err("Offline bundle contains no models".into());
    }
    if manifest.total_payload_bytes == 0 || manifest.total_payload_bytes > MAX_BUNDLE_PAYLOAD_BYTES
    {
        return Err(format!(
            "Offline bundle declares invalid payload size {}",
            manifest.total_payload_bytes
        )
        .into());
    }
    validate_sha256_text(&manifest.backend_index_sha256)?;
    validate_sha256_text(&manifest.backend_index_signature_sha256)?;
    validate_sha256_text(&manifest.checksums_sha256)?;
    validate_sha256_text(&manifest.licenses_sha256)?;
    for model in &manifest.models {
        validate_bundle_relative_path(&model.path)?;
        if !model.path.starts_with("models/")
            || !model.path.to_ascii_lowercase().ends_with(".aimod")
        {
            return Err(format!(
                "Bundled model path must be models/*.aimod: `{}`",
                model.path
            )
            .into());
        }
        validate_sha256_text(&model.sha256)?;
    }
    for backend in &manifest.backends {
        validate_bundle_relative_path(&backend.path)?;
        if !backend.path.starts_with("backends/")
            || !backend.path.to_ascii_lowercase().ends_with(".tar.gz")
        {
            return Err(format!(
                "Bundled backend path must be backends/*.tar.gz: `{}`",
                backend.path
            )
            .into());
        }
        validate_sha256_text(&backend.sha256)?;
    }
    Ok(())
}

fn bundle_cache_root() -> Result<PathBuf, DynError> {
    if let Some(path) = std::env::var_os(BUNDLE_CACHE_ENV) {
        return Ok(PathBuf::from(path));
    }
    let data_root = backend_cache_root()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .ok_or("Could not determine bundle cache directory; set KAPSL_BUNDLE_CACHE_DIR")?;
    Ok(data_root.join("bundles"))
}

fn validate_bundle_relative_path(value: &str) -> Result<(), DynError> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("Invalid path in offline bundle manifest: `{value}`").into());
    }
    Ok(())
}

fn validate_sha256_text(value: &str) -> Result<(), DynError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("Invalid SHA-256 digest in offline bundle: `{value}`").into());
    }
    Ok(())
}

fn verify_digest(path: &Path, expected: &str, label: &str) -> Result<(), DynError> {
    let actual = sha256_file(path)?;
    if actual != expected.to_ascii_lowercase() {
        return Err(format!(
            "Offline bundle {label} failed checksum verification: expected {expected}, got {actual}"
        )
        .into());
    }
    Ok(())
}

fn verify_checksum_inventory(root: &Path, manifest: &BundleManifest) -> Result<(), DynError> {
    let bytes = read_bounded(&root.join(BUNDLE_CHECKSUMS_PATH), 8 * 1024 * 1024)?;
    let text = std::str::from_utf8(&bytes)?;
    let expected = [
        (
            BUNDLE_INDEX_PATH.to_string(),
            manifest.backend_index_sha256.clone(),
        ),
        (
            BUNDLE_INDEX_SIGNATURE_PATH.to_string(),
            manifest.backend_index_signature_sha256.clone(),
        ),
        (
            BUNDLE_LICENSES_PATH.to_string(),
            manifest.licenses_sha256.clone(),
        ),
    ]
    .into_iter()
    .chain(
        manifest
            .models
            .iter()
            .map(|model| (model.path.clone(), model.sha256.clone())),
    )
    .chain(
        manifest
            .backends
            .iter()
            .map(|backend| (backend.path.clone(), backend.sha256.clone())),
    )
    .collect::<BTreeMap<_, _>>();
    let mut seen = HashSet::new();
    for (line_number, line) in text.lines().enumerate() {
        let (digest, path) = line.split_once("  ").ok_or_else(|| {
            format!(
                "Invalid checksum inventory line {} in offline bundle",
                line_number + 1
            )
        })?;
        validate_sha256_text(digest)?;
        validate_bundle_relative_path(path)?;
        if !seen.insert(path.to_string()) {
            return Err(format!("Duplicate checksum inventory path `{path}`").into());
        }
        let expected_digest = expected.get(path).ok_or_else(|| {
            format!("Unexpected checksum inventory path `{path}` in offline bundle")
        })?;
        if !digest.eq_ignore_ascii_case(expected_digest) {
            return Err(format!(
                "Checksum inventory digest for `{path}` does not match the bundle manifest"
            )
            .into());
        }
        verify_digest(&root.join(path), digest, path)?;
    }
    if seen.is_empty() {
        return Err("Offline bundle checksum inventory is empty".into());
    }
    let expected_paths = expected.keys().cloned().collect::<HashSet<_>>();
    if seen != expected_paths {
        let missing = expected_paths
            .difference(&seen)
            .cloned()
            .collect::<Vec<_>>();
        let unexpected = seen
            .difference(&expected_paths)
            .cloned()
            .collect::<Vec<_>>();
        return Err(format!(
            "Offline bundle checksum inventory does not match its manifest (missing: {missing:?}, unexpected: {unexpected:?})"
        )
        .into());
    }
    Ok(())
}

fn verify_bundle_tree(root: &Path, manifest: &BundleManifest) -> Result<(), DynError> {
    let allowed = [
        BUNDLE_MANIFEST_NAME.to_string(),
        BUNDLE_INDEX_PATH.to_string(),
        BUNDLE_INDEX_SIGNATURE_PATH.to_string(),
        BUNDLE_CHECKSUMS_PATH.to_string(),
        BUNDLE_LICENSES_PATH.to_string(),
    ]
    .into_iter()
    .chain(manifest.models.iter().map(|model| model.path.clone()))
    .chain(manifest.backends.iter().map(|backend| backend.path.clone()))
    .map(PathBuf::from)
    .collect::<HashSet<_>>();

    fn visit(root: &Path, path: &Path, allowed: &HashSet<PathBuf>) -> Result<(), DynError> {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            let metadata = fs::symlink_metadata(&entry_path)?;
            if metadata.file_type().is_symlink() {
                return Err(format!(
                    "Offline bundle contains an unsupported symlink: {}",
                    entry_path.display()
                )
                .into());
            }
            if metadata.is_dir() {
                visit(root, &entry_path, allowed)?;
                continue;
            }
            let relative = entry_path.strip_prefix(root)?;
            if !metadata.is_file() || !allowed.contains(relative) {
                return Err(format!(
                    "Offline bundle contains an undeclared entry: {}",
                    relative.display()
                )
                .into());
            }
        }
        Ok(())
    }

    visit(root, root, &allowed)
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, DynError> {
    let file = File::open(path)?;
    if file.metadata()?.len() > limit {
        return Err(format!("{} exceeds its {} byte limit", path.display(), limit).into());
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(format!("{} exceeds its {} byte limit", path.display(), limit).into());
    }
    Ok(bytes)
}

fn append_bytes<W: Write>(
    archive: &mut Builder<W>,
    path: &str,
    bytes: &[u8],
    mode: u32,
) -> std::io::Result<()> {
    let mut header = tar::Header::new_gnu();
    header.set_size(bytes.len() as u64);
    header.set_mode(mode);
    header.set_mtime(0);
    header.set_cksum();
    archive.append_data(&mut header, path, bytes)
}

fn append_file<W: Write>(
    archive: &mut Builder<W>,
    path: &str,
    source: &Path,
    mode: u32,
) -> std::io::Result<()> {
    let mut file = File::open(source)?;
    let mut header = tar::Header::new_gnu();
    header.set_size(file.metadata()?.len());
    header.set_mode(mode);
    header.set_mtime(0);
    header.set_cksum();
    archive.append_data(&mut header, path, &mut file)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sanitize_bundle_name(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() || sanitized == "." || sanitized == ".." {
        "item".to_string()
    } else {
        sanitized
    }
}

fn unique_nonce() -> String {
    format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

fn bundle_process_path_lock(path: &Path) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<BTreeMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    let locks = LOCKS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut locks = locks
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    locks
        .entry(path.to_path_buf())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ServingBackendPolicy;
    use crate::backend::{
        BackendExecutionMode, BackendInstaller, BackendMemoryManifest,
        BACKEND_INDEX_SCHEMA_VERSION, BACKEND_PACK_SCHEMA_VERSION, BACKEND_RUNTIME_ABI,
    };
    use crate::features::packaging::{create_kapsl_package, PackageKapslRequest};
    use base64::engine::general_purpose::STANDARD as BASE64;
    use base64::Engine as _;
    use ed25519_dalek::{Signer, SigningKey};
    use std::sync::Barrier;

    fn sign_index(key: &SigningKey, bytes: &[u8]) -> String {
        let mut message = b"kapsl-backend-index-v1\0".to_vec();
        message.extend_from_slice(bytes);
        format!("ed25519:{}", BASE64.encode(key.sign(&message).to_bytes()))
    }

    fn sign_artifact(key: &SigningKey, digest: &str) -> String {
        let message = format!("kapsl-backend-artifact-v1\0sha256:{digest}");
        format!(
            "ed25519:{}",
            BASE64.encode(key.sign(message.as_bytes()).to_bytes())
        )
    }

    fn bundle_manager_fixture() -> (tempfile::TempDir, BackendManager) {
        let root = tempfile::tempdir().unwrap();
        let artifact = root.path().join("vllm.tar.gz");
        let file = File::create(&artifact).unwrap();
        let encoder = GzEncoder::new(file, Compression::fast());
        let mut archive = Builder::new(encoder);
        let payload = serde_json::json!({
            "schema_version": BACKEND_PACK_SCHEMA_VERSION,
            "backend": "vllm",
            "profile": MANAGED_VLLM_PACK_PROFILE,
            "pack_version": "test",
            "runtime_abi": BACKEND_RUNTIME_ABI,
            "platform": "linux-x86_64",
            "execution_mode": "external",
            "entrypoint": "bin/python"
        });
        append_bytes(
            &mut archive,
            "backend-pack.json",
            &serde_json::to_vec(&payload).unwrap(),
            0o644,
        )
        .unwrap();
        append_bytes(&mut archive, "bin/python", b"#!/bin/sh\n", 0o755).unwrap();
        archive.finish().unwrap();
        archive.into_inner().unwrap().finish().unwrap();

        let signing = SigningKey::from_bytes(&[19_u8; 32]);
        let digest = sha256_file(&artifact).unwrap();
        let vllm_pack = BackendPackManifest {
            schema_version: BACKEND_PACK_SCHEMA_VERSION,
            backend: "vllm".to_string(),
            profile: MANAGED_VLLM_PACK_PROFILE.to_string(),
            pack_version: "test".to_string(),
            runtime_abi: BACKEND_RUNTIME_ABI,
            adapter_abi: None,
            compatible_kapsl: format!("={}", runtime_release_version()),
            platform: "linux-x86_64".to_string(),
            architecture: "x86_64".to_string(),
            accelerator_profile: "cuda".to_string(),
            minimum_cuda: None,
            minimum_driver: None,
            execution_mode: BackendExecutionMode::External,
            kv_mode: None,
            entrypoint: "bin/python".to_string(),
            artifact: format!("file://{}", artifact.display()),
            download_bytes: fs::metadata(&artifact).unwrap().len(),
            installed_bytes: 4096,
            sha256: digest.clone(),
            signature: sign_artifact(&signing, &digest),
            memory: BackendMemoryManifest::default(),
            installer: BackendInstaller::Extract,
            files: BTreeMap::from([("bin/python".to_string(), sha256_bytes(b"#!/bin/sh\n"))]),
            licenses: Vec::new(),
            priority: 100,
        };

        let onnx_artifact = root.path().join("onnx-cpu.tar.gz");
        let file = File::create(&onnx_artifact).unwrap();
        let encoder = GzEncoder::new(file, Compression::fast());
        let mut archive = Builder::new(encoder);
        let onnx_entrypoint = b"fixture ONNX native entrypoint";
        let payload = serde_json::json!({
            "schema_version": BACKEND_PACK_SCHEMA_VERSION,
            "backend": "onnx",
            "profile": crate::backend::ONNX_CPU_PACK_PROFILE,
            "pack_version": "test",
            "runtime_abi": BACKEND_RUNTIME_ABI,
            "platform": "linux-x86_64",
            "execution_mode": "native",
            "entrypoint": "libkapsl_backend_onnx.so"
        });
        append_bytes(
            &mut archive,
            "backend-pack.json",
            &serde_json::to_vec(&payload).unwrap(),
            0o644,
        )
        .unwrap();
        append_bytes(
            &mut archive,
            "libkapsl_backend_onnx.so",
            onnx_entrypoint,
            0o644,
        )
        .unwrap();
        archive.finish().unwrap();
        archive.into_inner().unwrap().finish().unwrap();
        let onnx_digest = sha256_file(&onnx_artifact).unwrap();
        let onnx_pack = BackendPackManifest {
            schema_version: BACKEND_PACK_SCHEMA_VERSION,
            backend: "onnx".to_string(),
            profile: crate::backend::ONNX_CPU_PACK_PROFILE.to_string(),
            pack_version: "test".to_string(),
            runtime_abi: BACKEND_RUNTIME_ABI,
            adapter_abi: None,
            compatible_kapsl: format!("={}", runtime_release_version()),
            platform: "linux-x86_64".to_string(),
            architecture: "x86_64".to_string(),
            accelerator_profile: "cpu".to_string(),
            minimum_cuda: None,
            minimum_driver: None,
            execution_mode: BackendExecutionMode::Native,
            kv_mode: None,
            entrypoint: "libkapsl_backend_onnx.so".to_string(),
            artifact: format!("file://{}", onnx_artifact.display()),
            download_bytes: fs::metadata(&onnx_artifact).unwrap().len(),
            installed_bytes: 4096,
            sha256: onnx_digest.clone(),
            signature: sign_artifact(&signing, &onnx_digest),
            memory: BackendMemoryManifest::default(),
            installer: BackendInstaller::Extract,
            files: BTreeMap::from([(
                "libkapsl_backend_onnx.so".to_string(),
                sha256_bytes(onnx_entrypoint),
            )]),
            licenses: Vec::new(),
            priority: 100,
        };

        let llama_artifact = root.path().join("llama-cpp-cpu.tar.gz");
        let file = File::create(&llama_artifact).unwrap();
        let encoder = GzEncoder::new(file, Compression::fast());
        let mut archive = Builder::new(encoder);
        let llama_entrypoint = b"fixture llama.cpp native entrypoint";
        let payload = serde_json::json!({
            "schema_version": BACKEND_PACK_SCHEMA_VERSION,
            "backend": "llama-cpp",
            "profile": crate::backend::LLAMA_CPP_CPU_PACK_PROFILE,
            "pack_version": "test",
            "runtime_abi": BACKEND_RUNTIME_ABI,
            "platform": "linux-x86_64",
            "execution_mode": "native",
            "kv_mode": "native",
            "entrypoint": "lib/libkapsl_backend_llama_cpp.so"
        });
        append_bytes(
            &mut archive,
            "backend-pack.json",
            &serde_json::to_vec(&payload).unwrap(),
            0o644,
        )
        .unwrap();
        append_bytes(
            &mut archive,
            "lib/libkapsl_backend_llama_cpp.so",
            llama_entrypoint,
            0o644,
        )
        .unwrap();
        archive.finish().unwrap();
        archive.into_inner().unwrap().finish().unwrap();
        let llama_digest = sha256_file(&llama_artifact).unwrap();
        let llama_pack = BackendPackManifest {
            schema_version: BACKEND_PACK_SCHEMA_VERSION,
            backend: "llama-cpp".to_string(),
            profile: crate::backend::LLAMA_CPP_CPU_PACK_PROFILE.to_string(),
            pack_version: "test".to_string(),
            runtime_abi: BACKEND_RUNTIME_ABI,
            adapter_abi: None,
            compatible_kapsl: format!("={}", runtime_release_version()),
            platform: "linux-x86_64".to_string(),
            architecture: "x86_64".to_string(),
            accelerator_profile: "cpu".to_string(),
            minimum_cuda: None,
            minimum_driver: None,
            execution_mode: BackendExecutionMode::Native,
            kv_mode: Some("native".to_string()),
            entrypoint: "lib/libkapsl_backend_llama_cpp.so".to_string(),
            artifact: format!("file://{}", llama_artifact.display()),
            download_bytes: fs::metadata(&llama_artifact).unwrap().len(),
            installed_bytes: 4096,
            sha256: llama_digest.clone(),
            signature: sign_artifact(&signing, &llama_digest),
            memory: BackendMemoryManifest::default(),
            installer: BackendInstaller::Extract,
            files: BTreeMap::from([(
                "lib/libkapsl_backend_llama_cpp.so".to_string(),
                sha256_bytes(llama_entrypoint),
            )]),
            licenses: Vec::new(),
            priority: 100,
        };
        let index = BackendIndex {
            schema_version: BACKEND_INDEX_SCHEMA_VERSION,
            runtime_version: runtime_release_version(),
            generated_at: "2026-08-25T00:00:00Z".to_string(),
            packs: vec![vllm_pack, onnx_pack, llama_pack],
        };
        let index_bytes = serde_json::to_vec_pretty(&index).unwrap();
        let index_path = root.path().join("backend-index.json");
        fs::write(&index_path, &index_bytes).unwrap();
        fs::write(
            root.path().join("backend-index.json.sig"),
            sign_index(&signing, &index_bytes),
        )
        .unwrap();
        let manager = BackendManager::for_test(
            root.path().join("backend-cache"),
            &runtime_release_version(),
            index_path,
            signing.verifying_key(),
            false,
        );
        (root, manager)
    }

    fn package_model(
        root: &Path,
        name: &str,
        safetensors: bool,
        serving_backend: Option<ServingBackendPolicy>,
    ) -> PathBuf {
        let extension = if safetensors { "safetensors" } else { "onnx" };
        let model = root.join(format!("{name}.{extension}"));
        let package = root.join(format!("{name}.aimod"));
        fs::write(&model, format!("dummy {name} weights")).unwrap();
        create_kapsl_package(
            &PackageKapslRequest {
                model_path: model.to_string_lossy().to_string(),
                output_path: Some(package.to_string_lossy().to_string()),
                project_name: Some(name.to_string()),
                framework: None,
                format: safetensors.then(|| "safetensors".to_string()),
                model_type: safetensors.then(|| "causal-lm".to_string()),
                task: safetensors.then(|| "generate".to_string()),
                serving_backend,
                version: Some("1.0.0".to_string()),
                metadata: None,
            },
            false,
        )
        .unwrap();
        package
    }

    fn package_gguf_model(root: &Path, name: &str) -> PathBuf {
        let model = root.join(format!("{name}.gguf"));
        let package = root.join(format!("{name}.aimod"));
        fs::write(&model, format!("dummy {name} GGUF weights")).unwrap();
        create_kapsl_package(
            &PackageKapslRequest {
                model_path: model.to_string_lossy().to_string(),
                output_path: Some(package.to_string_lossy().to_string()),
                project_name: Some(name.to_string()),
                framework: None,
                format: Some("gguf".to_string()),
                model_type: Some("causal-lm".to_string()),
                task: Some("generate".to_string()),
                serving_backend: Some(ServingBackendPolicy::LlamaCpp),
                version: Some("1.0.0".to_string()),
                metadata: None,
            },
            false,
        )
        .unwrap();
        package
    }

    #[test]
    fn parses_cross_platform_targets() {
        let info = DeviceInfo::probe();
        let (target, manifest) = resolve_bundle_target(Some("linux-x86_64-cuda"), &info).unwrap();
        assert_eq!(target.platform, "linux-x86_64");
        assert_eq!(target.architecture, "x86_64");
        assert_eq!(target.accelerator, BackendAccelerator::Cuda);
        assert_eq!(manifest.accelerator, "cuda");
    }

    #[test]
    fn llama_cpp_bundle_identity_resolves_its_exact_accelerator() {
        let base = BackendTarget {
            platform: "linux-x86_64".to_string(),
            architecture: "x86_64".to_string(),
            accelerator: BackendAccelerator::Cuda,
            cuda_version: Some("12.6".to_string()),
            driver_version: Some("560.28.03".to_string()),
        };
        let cpu = target_for_backend_identity(
            &base,
            "llama-cpp",
            crate::backend::LLAMA_CPP_CPU_PACK_PROFILE,
        );
        assert_eq!(cpu.accelerator, BackendAccelerator::Cpu);
        assert_eq!(cpu.cuda_version, None);
        assert_eq!(cpu.driver_version, None);

        let cuda = target_for_backend_identity(
            &base,
            "llama-cpp",
            crate::backend::LLAMA_CPP_CUDA12_PACK_PROFILE,
        );
        assert_eq!(cuda.accelerator, BackendAccelerator::Cuda);
        assert_eq!(cuda.cuda_version.as_deref(), Some("12.6"));
    }

    #[test]
    fn rejects_unsafe_manifest_paths() {
        assert!(validate_bundle_relative_path("models/model.aimod").is_ok());
        assert!(validate_bundle_relative_path("../model.aimod").is_err());
        assert!(validate_bundle_relative_path("/tmp/model.aimod").is_err());
    }

    #[test]
    fn sanitizes_archive_names() {
        assert_eq!(sanitize_bundle_name("my model.aimod"), "my_model.aimod");
        assert_eq!(sanitize_bundle_name("../"), ".._");
    }

    #[test]
    fn single_model_offline_bundle_round_trips_from_cache() {
        let (root, manager) = bundle_manager_fixture();
        let model = package_model(root.path(), "model", false, None);
        let output = root.path().join("model.kapsl-bundle");
        let info = DeviceInfo::probe();
        let args = BundleCommandArgs {
            model: vec![model],
            output: output.clone(),
            target: Some(format!("{}-cpu", current_platform())),
        };
        create_bundle(&args, &info, &manager).unwrap();

        let bundle_cache = root.path().join("bundle-cache");
        let first = activate_bundle_with_manager(&output, &info, &bundle_cache, &manager).unwrap();
        let second = activate_bundle_with_manager(&output, &info, &bundle_cache, &manager).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 1);
        assert!(first[0].is_file());
        assert_eq!(
            inspect_serving_manifest(&first[0]).unwrap().project_name,
            "model"
        );
    }

    #[test]
    fn linux_multi_model_bundle_deduplicates_onnx_cpu_pack() {
        let (root, manager) = bundle_manager_fixture();
        let model_a = package_model(root.path(), "onnx-a", false, None);
        let model_b = package_model(root.path(), "onnx-b", false, None);
        let output = root.path().join("onnx-production.kapsl-bundle");
        create_bundle(
            &BundleCommandArgs {
                model: vec![model_a, model_b],
                output: output.clone(),
                target: Some("linux-x86_64-cpu".to_string()),
            },
            &DeviceInfo::probe(),
            &manager,
        )
        .unwrap();

        let manifest = inspect_bundle_manifest(&output).unwrap();
        assert_eq!(manifest.models.len(), 2);
        assert_eq!(manifest.backends.len(), 1);
        assert_eq!(manifest.backends[0].backend, "onnx");
        assert_eq!(
            manifest.backends[0].profile,
            crate::backend::ONNX_CPU_PACK_PROFILE
        );
    }

    #[test]
    fn linux_multi_model_bundle_deduplicates_llama_cpp_cpu_pack() {
        let (root, manager) = bundle_manager_fixture();
        let model_a = package_gguf_model(root.path(), "gguf-a");
        let model_b = package_gguf_model(root.path(), "gguf-b");
        let output = root.path().join("gguf-production.kapsl-bundle");
        create_bundle(
            &BundleCommandArgs {
                model: vec![model_a, model_b],
                output: output.clone(),
                target: Some("linux-x86_64-cpu".to_string()),
            },
            &DeviceInfo::probe(),
            &manager,
        )
        .unwrap();

        let manifest = inspect_bundle_manifest(&output).unwrap();
        assert_eq!(manifest.models.len(), 2);
        assert_eq!(manifest.backends.len(), 1);
        assert_eq!(manifest.backends[0].backend, "llama-cpp");
        assert_eq!(
            manifest.backends[0].profile,
            crate::backend::LLAMA_CPP_CPU_PACK_PROFILE
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn linux_gguf_offline_bundle_installs_its_llama_cpp_pack() {
        let (root, manager) = bundle_manager_fixture();
        let model = package_gguf_model(root.path(), "offline-gguf");
        let output = root.path().join("offline-gguf.kapsl-bundle");
        let info = DeviceInfo::probe();
        create_bundle(
            &BundleCommandArgs {
                model: vec![model],
                output: output.clone(),
                target: Some("linux-x86_64-cpu".to_string()),
            },
            &info,
            &manager,
        )
        .unwrap();

        let models = activate_bundle_with_manager(
            &output,
            &info,
            &root.path().join("bundle-cache"),
            &manager,
        )
        .unwrap();
        assert_eq!(models.len(), 1);
        let installed = manager.list().unwrap();
        assert_eq!(installed.len(), 1);
        assert_eq!(installed[0].backend, "llama-cpp");
        assert_eq!(
            installed[0].profile,
            crate::backend::LLAMA_CPP_CPU_PACK_PROFILE
        );
        assert!(installed[0].valid);
    }

    #[test]
    fn multi_model_bundle_deduplicates_required_backend_pack() {
        let (root, manager) = bundle_manager_fixture();
        let model_a = package_model(
            root.path(),
            "model-a",
            true,
            Some(ServingBackendPolicy::Vllm),
        );
        let model_b = package_model(
            root.path(),
            "model-b",
            true,
            Some(ServingBackendPolicy::Vllm),
        );
        let output = root.path().join("production.kapsl-bundle");
        let args = BundleCommandArgs {
            model: vec![model_a, model_b],
            output: output.clone(),
            target: Some("linux-x86_64-cuda".to_string()),
        };
        create_bundle(&args, &DeviceInfo::probe(), &manager).unwrap();

        let manifest = inspect_bundle_manifest(&output).unwrap();
        assert_eq!(manifest.models.len(), 2);
        assert_eq!(manifest.backends.len(), 1);
        assert_eq!(manifest.backends[0].backend, "vllm");
        assert_eq!(manifest.backends[0].profile, MANAGED_VLLM_PACK_PROFILE);
    }

    #[test]
    fn concurrent_bundle_activation_produces_one_valid_cache() {
        let (root, manager) = bundle_manager_fixture();
        let model = package_model(root.path(), "concurrent", false, None);
        let output = root.path().join("concurrent.kapsl-bundle");
        let info = DeviceInfo::probe();
        create_bundle(
            &BundleCommandArgs {
                model: vec![model],
                output: output.clone(),
                target: Some(format!("{}-cpu", current_platform())),
            },
            &info,
            &manager,
        )
        .unwrap();

        let cache = root.path().join("concurrent-cache");
        let manager = Arc::new(manager);
        let info = Arc::new(info);
        let barrier = Arc::new(Barrier::new(4));
        let handles = (0..4)
            .map(|_| {
                let output = output.clone();
                let cache = cache.clone();
                let manager = manager.clone();
                let info = info.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    activate_bundle_with_manager(&output, &info, &cache, &manager).unwrap()
                })
            })
            .collect::<Vec<_>>();
        let resolved = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert!(resolved.windows(2).all(|pair| pair[0] == pair[1]));
        assert!(resolved[0][0].is_file());
    }
}
