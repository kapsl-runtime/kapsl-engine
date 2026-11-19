//! Signed backend-pack resolution and installation.
//!
//! A model package may select a backend policy, but it never supplies an
//! executable or URL.  Those deployment details come only from the signed
//! runtime-specific backend index handled here.

use crate::features::http_client;
use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE_NO_PAD as BASE64_URL_SAFE};
use base64::Engine as _;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use flate2::read::GzDecoder;
use fs2::FileExt;
use kapsl_hal::device::{DeviceBackend, DeviceInfo};
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::ffi::OsStr;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tar::Archive;

mod archive;
mod checksums;
mod files;
mod index;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub(crate) use archive::safe_extract_tar_gz;
use archive::*;
pub(crate) use files::sha256_file;
use files::*;
use index::*;
pub(crate) use types::*;
use validate::*;

pub(crate) const BACKEND_INDEX_SCHEMA_VERSION: u32 = 1;
pub(crate) const BACKEND_PACK_SCHEMA_VERSION: u32 = 1;
pub(crate) const BACKEND_RUNTIME_ABI: u32 = 1;
pub(crate) const MANAGED_VLLM_PACK_PROFILE: &str = "cu130-flash-attn";
pub(crate) const STANDARD_NATIVE_ADAPTER_ABI: &str = "kapsl-backend-v1";

const BACKEND_CACHE_ENV: &str = "KAPSL_BACKEND_CACHE_DIR";
const BACKEND_INDEX_URL_ENV: &str = "KAPSL_BACKEND_INDEX_URL";
const BACKEND_INDEX_PATH_ENV: &str = "KAPSL_BACKEND_INDEX_PATH";
const BACKEND_PUBLIC_KEYS_ENV: &str = "KAPSL_BACKEND_PUBLIC_KEYS";
const OFFLINE_ENV: &str = "KAPSL_OFFLINE";
const DEFAULT_DOWNLOAD_BASE_URL: &str = "https://downloads.kapsl.net";
const MAX_INDEX_BYTES: u64 = 8 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const INSTALL_SPACE_OVERHEAD_BYTES: u64 = 256 * 1024 * 1024;
const INSTALLED_SIZE_TOLERANCE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ARCHIVE_ENTRIES: usize = 250_000;
const INSTALL_RECORD_NAME: &str = ".kapsl-backend.json";
const PAYLOAD_MANIFEST_NAME: &str = "backend-pack.json";

#[derive(Debug)]
pub(crate) struct BackendManagerError(String);

impl BackendManagerError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for BackendManagerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BackendManagerError {}

impl From<std::io::Error> for BackendManagerError {
    fn from(error: std::io::Error) -> Self {
        Self::new(error.to_string())
    }
}

pub(crate) type ManagerResult<T> = Result<T, BackendManagerError>;

#[derive(Clone)]
struct BackendManagerConfig {
    cache_root: PathBuf,
    runtime_version: String,
    index_url: String,
    index_path: Option<PathBuf>,
    offline: bool,
    trusted_keys: Vec<VerifyingKey>,
    allow_file_artifacts: bool,
}

#[derive(Clone)]
pub(crate) struct BackendManager {
    config: BackendManagerConfig,
}

struct MaintenanceReadGuard {
    _process: RwLockReadGuard<'static, ()>,
    _file: File,
}

struct MaintenanceWriteGuard {
    _process: RwLockWriteGuard<'static, ()>,
    _file: File,
}

impl BackendManager {
    pub(crate) fn from_env(force_offline: bool) -> ManagerResult<Self> {
        let runtime_version = runtime_release_version();
        let cache_root = backend_cache_root().ok_or_else(|| {
            BackendManagerError::new(
                "could not determine the local data directory for the backend cache; set KAPSL_BACKEND_CACHE_DIR",
            )
        })?;
        let index_url = std::env::var(BACKEND_INDEX_URL_ENV)
            .unwrap_or_else(|_| default_backend_index_url(&runtime_version));
        let index_path = std::env::var_os(BACKEND_INDEX_PATH_ENV).map(PathBuf::from);
        let offline = force_offline || env_flag(OFFLINE_ENV);
        let trusted_keys = trusted_public_keys()?;
        Ok(Self {
            config: BackendManagerConfig {
                cache_root,
                runtime_version,
                index_url,
                index_path,
                offline,
                trusted_keys,
                allow_file_artifacts: false,
            },
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        cache_root: PathBuf,
        runtime_version: &str,
        index_path: PathBuf,
        trusted_key: VerifyingKey,
        offline: bool,
    ) -> Self {
        Self {
            config: BackendManagerConfig {
                cache_root,
                runtime_version: runtime_version.to_string(),
                index_url: String::new(),
                index_path: Some(index_path),
                offline,
                trusted_keys: vec![trusted_key],
                allow_file_artifacts: true,
            },
        }
    }

    pub(crate) fn runtime_version(&self) -> &str {
        &self.config.runtime_version
    }

    pub(crate) fn installed_path(&self, pack: &BackendPackManifest) -> ManagerResult<PathBuf> {
        validate_cache_component("backend", &pack.backend)?;
        validate_cache_component("profile", &pack.profile)?;
        if pack.pack_version.trim().is_empty()
            || pack.platform.trim().is_empty()
            || pack.architecture.trim().is_empty()
        {
            return Err(BackendManagerError::new(
                "backend pack version, platform, and architecture must be non-empty",
            ));
        }
        Ok(self
            .runtime_cache_root()
            .join(&pack.backend)
            .join(&pack.profile))
    }

    /// Resolve one model contract against the verified signed index without
    /// downloading or loading adapter code.
    pub(crate) fn plan_compatible_backend(
        &self,
        requirements: &BackendPackRequirements,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackPlan> {
        let index = self.load_index()?;
        let selection = self.select_compatible_pack(&index, requirements, target)?;
        let pack = selection.manifest;
        let installed = self.installed_pack_is_valid(&pack).unwrap_or(false);
        Ok(BackendPackPlan {
            selected_backend: pack.backend.clone(),
            profile: pack.profile.clone(),
            selection_reason: selection.reason,
            installed,
            download_required: !installed,
            download_bytes: if installed { 0 } else { pack.download_bytes },
            execution_mode: pack.execution_mode.as_str().to_string(),
            manifest: pack,
        })
    }

    pub(crate) fn ensure_vllm(&self, target: &BackendTarget) -> ManagerResult<PathBuf> {
        let plan = self.plan_vllm(target)?;
        self.ensure_pack(&plan.manifest)
    }

    pub(crate) fn ensure_pack(&self, pack: &BackendPackManifest) -> ManagerResult<PathBuf> {
        let _maintenance = self.acquire_maintenance_read()?;
        self.validate_pack_manifest(pack, None)?;
        let final_path = self.installed_path(pack)?;
        let lock_path = self.install_lock_path(pack)?;
        let process_lock = process_path_lock(&lock_path);
        let _process_guard = process_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let lock = lock_file(&lock_path)?;
        lock.lock_exclusive().map_err(|error| {
            BackendManagerError::new(format!(
                "lock backend installation {}: {error}",
                lock_path.display()
            ))
        })?;

        self.cleanup_stale_install_stages(pack)?;
        if self.installed_pack_is_valid(pack)? {
            let _ = fs2::FileExt::unlock(&lock);
            return Ok(final_path);
        }
        if self.config.offline {
            let _ = fs2::FileExt::unlock(&lock);
            return Err(BackendManagerError::new(format!(
                "backend `{}/{}` is not installed and Kapsl is offline; prepare a .kapsl-bundle on a connected machine or run `kapsl backend ensure MODEL` while online",
                pack.backend, pack.profile
            )));
        }

        let staging_parent = self.pack_staging_root(pack)?;
        fs::create_dir_all(&staging_parent)?;
        ensure_available_space(
            &staging_parent,
            pack.download_bytes
                .saturating_mul(2)
                .saturating_add(pack.installed_bytes)
                .saturating_add(INSTALL_SPACE_OVERHEAD_BYTES),
            "download and install backend pack",
        )?;
        let stage = tempfile::Builder::new()
            .prefix("install-")
            .tempdir_in(&staging_parent)
            .map_err(|error| {
                BackendManagerError::new(format!(
                    "create backend staging directory under {}: {error}",
                    staging_parent.display()
                ))
            })?;
        let archive_path = stage.path().join("backend-pack.tar.gz");
        eprintln!(
            "Downloading Kapsl backend {}/{} ({} bytes)...",
            pack.backend, pack.profile, pack.download_bytes
        );
        self.download_artifact(pack, &archive_path)?;
        self.install_verified_archive_locked(pack, &archive_path, &final_path, stage.path())?;
        let _ = fs2::FileExt::unlock(&lock);
        Ok(final_path)
    }

    /// Install an archive that was carried by a verified offline bundle. The
    /// caller supplies the signed index entry; this method re-verifies the
    /// artifact before it can enter the shared cache.
    pub(crate) fn install_pack_from_archive(
        &self,
        pack: &BackendPackManifest,
        archive_path: &Path,
    ) -> ManagerResult<PathBuf> {
        let _maintenance = self.acquire_maintenance_read()?;
        self.validate_pack_manifest(pack, None)?;
        self.verify_artifact_file(pack, archive_path)?;
        let final_path = self.installed_path(pack)?;
        let lock_path = self.install_lock_path(pack)?;
        let process_lock = process_path_lock(&lock_path);
        let _process_guard = process_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let lock = lock_file(&lock_path)?;
        lock.lock_exclusive().map_err(|error| {
            BackendManagerError::new(format!(
                "lock backend installation {}: {error}",
                lock_path.display()
            ))
        })?;
        self.cleanup_stale_install_stages(pack)?;
        if !self.installed_pack_is_valid(pack)? {
            let staging_parent = self.pack_staging_root(pack)?;
            fs::create_dir_all(&staging_parent)?;
            let stage = tempfile::Builder::new()
                .prefix("install-")
                .tempdir_in(&staging_parent)
                .map_err(|error| BackendManagerError::new(error.to_string()))?;
            self.install_verified_archive_locked(pack, archive_path, &final_path, stage.path())?;
        }
        let _ = fs2::FileExt::unlock(&lock);
        Ok(final_path)
    }

    pub(crate) fn fetch_pack_archive(
        &self,
        pack: &BackendPackManifest,
        destination: &Path,
    ) -> ManagerResult<()> {
        if self.config.offline {
            return Err(BackendManagerError::new(format!(
                "cannot download backend `{}/{}` while offline",
                pack.backend, pack.profile
            )));
        }
        self.download_artifact(pack, destination)
    }

    /// Verify a caller-supplied archive against its signed index entry without
    /// installing it. Offline bundle creation uses this for release artifacts
    /// already present on the preparation host.
    pub(crate) fn verify_pack_archive(
        &self,
        pack: &BackendPackManifest,
        archive_path: &Path,
    ) -> ManagerResult<()> {
        self.validate_pack_manifest(pack, None)?;
        self.verify_artifact_file(pack, archive_path)
    }

    pub(crate) fn select_compatible_pack(
        &self,
        index: &BackendIndex,
        requirements: &BackendPackRequirements,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackSelection> {
        let backend_pin = requirements
            .backend_pin
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let preferred_profile = requirements
            .preferred_profile
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let mut rejected = Vec::new();
        let mut candidates = Vec::new();

        for pack in &index.packs {
            if backend_pin.is_some_and(|backend| !pack.backend.eq_ignore_ascii_case(backend)) {
                continue;
            }
            if preferred_profile.is_some_and(|profile| pack.profile != profile) {
                continue;
            }

            let identity = format!("{}/{}", pack.backend, pack.profile);
            if let Err(error) = self.validate_pack_manifest(pack, Some(target)) {
                rejected.push(format!("{identity}: {error}"));
                continue;
            }
            if requirements
                .execution_mode
                .is_some_and(|mode| pack.execution_mode != mode)
            {
                rejected.push(format!(
                    "{identity}: execution mode {} does not match required {}",
                    pack.execution_mode.as_str(),
                    requirements
                        .execution_mode
                        .expect("checked execution mode")
                        .as_str()
                ));
                continue;
            }
            if requirements.adapter_abi.is_some()
                && pack.adapter_abi.as_deref() != requirements.adapter_abi.as_deref()
            {
                rejected.push(format!(
                    "{identity}: adapter ABI {:?} does not match required {:?}",
                    pack.adapter_abi, requirements.adapter_abi
                ));
                continue;
            }
            if let Some(format) = requirements.format.as_deref() {
                if !signed_contract_contains(&pack.formats, format) {
                    rejected.push(format!(
                        "{identity}: format `{format}` is not declared by the signed pack"
                    ));
                    continue;
                }
            }
            if let Some(model_type) = requirements.model_type.as_deref() {
                if !pack.model_types.is_empty()
                    && !signed_contract_contains(&pack.model_types, model_type)
                {
                    rejected.push(format!(
                        "{identity}: model type `{model_type}` is not declared by the signed pack"
                    ));
                    continue;
                }
            }
            if let Some(task) = requirements.task.as_deref() {
                if !signed_contract_contains(&pack.tasks, task) {
                    rejected.push(format!(
                        "{identity}: task `{task}` is not declared by the signed pack"
                    ));
                    continue;
                }
            }
            if let Some(provider) = requirements.execution_provider.as_deref() {
                if !signed_contract_contains(
                    &pack.accelerator_requirements.execution_providers,
                    provider,
                ) {
                    rejected.push(format!(
                        "{identity}: execution provider `{provider}` is not declared by the signed pack"
                    ));
                    continue;
                }
            }
            let missing = pack
                .capabilities
                .missing_required(&requirements.capabilities);
            if !missing.is_empty() {
                rejected.push(format!(
                    "{identity}: missing required capabilities {}",
                    missing.join(", ")
                ));
                continue;
            }
            if let Some(scope) = requirements.allocation_scope.as_deref() {
                if pack.memory_behavior.allocation_scope.as_deref() != Some(scope) {
                    rejected.push(format!(
                        "{identity}: allocation scope {:?} does not match required `{scope}`",
                        pack.memory_behavior.allocation_scope
                    ));
                    continue;
                }
            }
            if requirements.synchronize_before_free && !pack.memory_behavior.synchronize_before_free
            {
                rejected.push(format!(
                    "{identity}: signed memory behavior does not synchronize the device before freeing governed allocations"
                ));
                continue;
            }
            candidates.push(pack.clone());
        }

        candidates.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| compare_pack_versions(&right.pack_version, &left.pack_version))
                .then_with(|| left.backend.cmp(&right.backend))
                .then_with(|| left.profile.cmp(&right.profile))
        });

        let Some(selected) = candidates.first() else {
            let selection = backend_pin
                .map(|backend| format!("explicit backend `{backend}`"))
                .unwrap_or_else(|| "automatic backend selection".to_string());
            return Err(BackendManagerError::new(format!(
                "{selection} found no compatible signed pack for format `{}`, model type `{}`, task `{}`, provider `{}` on {} ({}){}",
                requirements.format.as_deref().unwrap_or("any"),
                requirements.model_type.as_deref().unwrap_or("any"),
                requirements.task.as_deref().unwrap_or("any"),
                requirements.execution_provider.as_deref().unwrap_or("any"),
                target.platform,
                target.accelerator.as_str(),
                if rejected.is_empty() {
                    String::new()
                } else {
                    format!("; rejected candidates: {}", rejected.join("; "))
                }
            )));
        };

        let tied = candidates
            .iter()
            .take_while(|candidate| {
                candidate.priority == selected.priority
                    && compare_pack_versions(&candidate.pack_version, &selected.pack_version)
                        == std::cmp::Ordering::Equal
            })
            .map(|candidate| format!("{}/{}", candidate.backend, candidate.profile))
            .collect::<Vec<_>>();
        if tied.len() > 1 {
            return Err(BackendManagerError::new(format!(
                "backend selection is ambiguous between equally ranked signed packs: {}; use an explicit backend pin or assign distinct signed priorities",
                tied.join(", ")
            )));
        }

        let selection_kind = backend_pin
            .map(|backend| format!("explicit backend pin `{backend}`"))
            .unwrap_or_else(|| "capability-based automatic selection".to_string());
        Ok(BackendPackSelection {
            manifest: selected.clone(),
            reason: format!(
                "{selection_kind} matched signed {}/{} version {} for format `{}`, model type `{}`, task `{}`, provider `{}`, and {} execution",
                selected.backend,
                selected.profile,
                selected.pack_version,
                requirements.format.as_deref().unwrap_or("any"),
                requirements.model_type.as_deref().unwrap_or("any"),
                requirements.task.as_deref().unwrap_or("any"),
                requirements.execution_provider.as_deref().unwrap_or("any"),
                target.accelerator.as_str()
            ),
        })
    }

    pub(crate) fn validate_pack_for_target(
        &self,
        pack: &BackendPackManifest,
        target: &BackendTarget,
    ) -> ManagerResult<()> {
        self.validate_pack_manifest(pack, Some(target))
    }

    pub(crate) fn select_pack(
        &self,
        index: &BackendIndex,
        backend: &str,
        preferred_profile: Option<&str>,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackManifest> {
        let mut rejected = Vec::new();
        let mut candidates = Vec::new();
        for pack in index.packs.iter().filter(|pack| pack.backend == backend) {
            if preferred_profile.is_some_and(|profile| pack.profile != profile) {
                continue;
            }
            match self.validate_pack_manifest(pack, Some(target)) {
                Ok(()) => candidates.push(pack.clone()),
                Err(error) => rejected.push(format!("{}: {error}", pack.profile)),
            }
        }
        candidates.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| compare_pack_versions(&right.pack_version, &left.pack_version))
        });
        candidates.into_iter().next().ok_or_else(|| {
            BackendManagerError::new(format!(
                "no compatible signed backend pack for `{backend}` profile `{}` on {} ({}, CUDA {:?}, driver {:?}){}",
                preferred_profile.unwrap_or("auto"),
                target.platform,
                target.accelerator.as_str(),
                target.cuda_version,
                target.driver_version,
                if rejected.is_empty() {
                    String::new()
                } else {
                    format!("; rejected candidates: {}", rejected.join("; "))
                }
            ))
        })
    }

    pub(crate) fn list(&self) -> ManagerResult<Vec<BackendCacheEntry>> {
        let mut result = Vec::new();
        if !self.config.cache_root.is_dir() {
            return Ok(result);
        }
        // A local install record is not a trust root. Only label a current
        // entry valid when it exactly matches a pack from the cached,
        // signature-verified runtime index.
        let signed_current_packs = self
            .load_index()
            .map(|index| index.packs)
            .unwrap_or_default();
        for runtime_entry in read_dirs(&self.config.cache_root)? {
            let runtime_path = runtime_entry.path();
            if !runtime_path.is_dir() || is_internal_name(&runtime_entry.file_name()) {
                continue;
            }
            let runtime_version = runtime_entry.file_name().to_string_lossy().into_owned();
            for backend_entry in read_dirs(&runtime_path)? {
                let backend_path = backend_entry.path();
                if !backend_path.is_dir() || is_internal_name(&backend_entry.file_name()) {
                    continue;
                }
                for profile_entry in read_dirs(&backend_path)? {
                    let profile_path = profile_entry.path();
                    if !profile_path.is_dir() || is_internal_name(&profile_entry.file_name()) {
                        continue;
                    }
                    let record = read_json_bounded::<InstalledPackRecord>(
                        &profile_path.join(INSTALL_RECORD_NAME),
                        MAX_MANIFEST_BYTES,
                    );
                    let (backend, profile, pack_version, execution_mode, valid) = match record {
                        Ok(record) => {
                            let valid = runtime_version == self.config.runtime_version
                                && signed_current_packs
                                    .iter()
                                    .any(|pack| pack == &record.manifest)
                                && self
                                    .installed_pack_is_valid(&record.manifest)
                                    .unwrap_or(false);
                            (
                                record.manifest.backend,
                                record.manifest.profile,
                                record.manifest.pack_version,
                                record.manifest.execution_mode.as_str().to_string(),
                                valid,
                            )
                        }
                        Err(_) => (
                            backend_entry.file_name().to_string_lossy().into_owned(),
                            profile_entry.file_name().to_string_lossy().into_owned(),
                            "unknown".to_string(),
                            "unknown".to_string(),
                            false,
                        ),
                    };
                    result.push(BackendCacheEntry {
                        runtime_version: runtime_version.clone(),
                        backend,
                        profile,
                        pack_version,
                        execution_mode,
                        installed_bytes: directory_size(&profile_path)?,
                        path: profile_path,
                        valid,
                    });
                }
            }
        }
        result.sort_by(|left, right| {
            left.runtime_version
                .cmp(&right.runtime_version)
                .then_with(|| left.backend.cmp(&right.backend))
                .then_with(|| left.profile.cmp(&right.profile))
        });
        Ok(result)
    }

    /// Remove abandoned staging/quarantine data. With `old_versions`, also
    /// remove cache trees belonging to other runtime versions.
    pub(crate) fn prune(&self, old_versions: bool) -> ManagerResult<(u64, usize)> {
        let _maintenance = self.acquire_maintenance_write()?;
        let mut bytes = 0_u64;
        let mut entries = 0_usize;
        if !self.config.cache_root.is_dir() {
            return Ok((0, 0));
        }
        for runtime_entry in read_dirs(&self.config.cache_root)? {
            let runtime_path = runtime_entry.path();
            if !runtime_path.is_dir() {
                continue;
            }
            let runtime_name = runtime_entry.file_name().to_string_lossy().into_owned();
            if old_versions
                && !is_internal_name(&runtime_entry.file_name())
                && runtime_name != self.config.runtime_version
            {
                bytes = bytes.saturating_add(directory_size(&runtime_path)?);
                fs::remove_dir_all(&runtime_path).map_err(|error| {
                    BackendManagerError::new(format!(
                        "remove old backend cache {}: {error}",
                        runtime_path.display()
                    ))
                })?;
                entries += 1;
                continue;
            }
            for name in [".staging", ".quarantine"] {
                let path = runtime_path.join(name);
                if path.is_dir() {
                    bytes = bytes.saturating_add(directory_size(&path)?);
                    fs::remove_dir_all(&path).map_err(|error| {
                        BackendManagerError::new(format!(
                            "remove stale backend data {}: {error}",
                            path.display()
                        ))
                    })?;
                    entries += 1;
                }
            }
        }
        Ok((bytes, entries))
    }

    fn runtime_cache_root(&self) -> PathBuf {
        self.config.cache_root.join(&self.config.runtime_version)
    }

    fn acquire_maintenance_read(&self) -> ManagerResult<MaintenanceReadGuard> {
        fs::create_dir_all(&self.config.cache_root)?;
        let process = backend_maintenance_lock()
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let file = lock_file(&self.config.cache_root.join(".maintenance.lock"))?;
        fs2::FileExt::lock_shared(&file).map_err(|error| {
            BackendManagerError::new(format!("lock backend cache for installation: {error}"))
        })?;
        Ok(MaintenanceReadGuard {
            _process: process,
            _file: file,
        })
    }

    fn acquire_maintenance_write(&self) -> ManagerResult<MaintenanceWriteGuard> {
        fs::create_dir_all(&self.config.cache_root)?;
        let process = backend_maintenance_lock()
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let file = lock_file(&self.config.cache_root.join(".maintenance.lock"))?;
        file.lock_exclusive().map_err(|error| {
            BackendManagerError::new(format!("lock backend cache for pruning: {error}"))
        })?;
        Ok(MaintenanceWriteGuard {
            _process: process,
            _file: file,
        })
    }

    fn install_lock_path(&self, pack: &BackendPackManifest) -> ManagerResult<PathBuf> {
        validate_cache_component("backend", &pack.backend)?;
        validate_cache_component("profile", &pack.profile)?;
        Ok(self
            .runtime_cache_root()
            .join(".locks")
            .join(&pack.backend)
            .join(format!("{}.lock", pack.profile)))
    }

    fn pack_staging_root(&self, pack: &BackendPackManifest) -> ManagerResult<PathBuf> {
        validate_cache_component("backend", &pack.backend)?;
        validate_cache_component("profile", &pack.profile)?;
        Ok(self
            .runtime_cache_root()
            .join(".staging")
            .join(&pack.backend)
            .join(&pack.profile))
    }

    fn installed_pack_is_valid(&self, expected: &BackendPackManifest) -> ManagerResult<bool> {
        let root = self.installed_path(expected)?;
        if !root.is_dir() {
            return Ok(false);
        }
        let record_path = root.join(INSTALL_RECORD_NAME);
        let record =
            match read_json_bounded::<InstalledPackRecord>(&record_path, MAX_MANIFEST_BYTES) {
                Ok(record) => record,
                Err(_) => return Ok(false),
            };
        if record.manifest != *expected {
            return Ok(false);
        }
        if validate_tree_symlinks(&root, &root).is_err() {
            return Ok(false);
        }
        if !entrypoint_is_usable(&root, expected)? {
            return Ok(false);
        }
        let started = Instant::now();
        let (files, algorithm) = match &expected.files_blake3 {
            Some(files) => (files, checksums::Algorithm::Blake3),
            None => (&expected.files, checksums::Algorithm::Sha256),
        };
        let valid = checksums::verify_installed_files(&root, files, algorithm)?;
        log::debug!(
            "Installed backend pack checksums {}/{} files={} valid={} elapsed_ms={:.3} algorithm={:?}",
            expected.backend,
            expected.profile,
            files.len(),
            valid,
            started.elapsed().as_secs_f64() * 1000.0,
            algorithm
        );
        if !valid {
            return Ok(false);
        }
        for notice in &expected.licenses {
            if notice
                .path
                .as_deref()
                .is_some_and(|relative| !root.join(relative).is_file())
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn download_artifact(
        &self,
        pack: &BackendPackManifest,
        destination: &Path,
    ) -> ManagerResult<()> {
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        if let Some(path) = pack.artifact.strip_prefix("file://") {
            if !self.config.allow_file_artifacts {
                return Err(BackendManagerError::new(
                    "file:// backend artifacts are disabled",
                ));
            }
            fs::copy(Path::new(path), destination).map_err(|error| {
                BackendManagerError::new(format!("copy backend artifact: {error}"))
            })?;
        } else {
            if !pack.artifact.starts_with("https://") {
                return Err(BackendManagerError::new(format!(
                    "backend artifact URL must use HTTPS: {}",
                    pack.artifact
                )));
            }
            let agent = http_client::http_agent_for_transfer();
            let mut response = agent.get(&pack.artifact).call().map_err(|error| {
                BackendManagerError::new(format!(
                    "download backend artifact {}: {}",
                    pack.artifact,
                    http_client::format_remote_http_error(error)
                ))
            })?;
            let file = File::create(destination)?;
            let mut writer = BufWriter::new(file);
            let mut reader = response
                .body_mut()
                .as_reader()
                .take(pack.download_bytes + 1);
            let copied = std::io::copy(&mut reader, &mut writer)?;
            writer.flush()?;
            if copied != pack.download_bytes {
                return Err(BackendManagerError::new(format!(
                    "backend artifact size mismatch: expected {} bytes, received {}",
                    pack.download_bytes, copied
                )));
            }
        }
        self.verify_artifact_file(pack, destination)
    }

    fn verify_artifact_file(
        &self,
        pack: &BackendPackManifest,
        archive_path: &Path,
    ) -> ManagerResult<()> {
        let size = fs::metadata(archive_path)?.len();
        if size != pack.download_bytes {
            return Err(BackendManagerError::new(format!(
                "backend artifact size mismatch: expected {} bytes, got {}",
                pack.download_bytes, size
            )));
        }
        let actual = sha256_file(archive_path)?;
        if actual != pack.sha256.to_ascii_lowercase() {
            return Err(BackendManagerError::new(format!(
                "backend artifact checksum mismatch: expected {}, got {}",
                pack.sha256, actual
            )));
        }
        self.verify_artifact_signature(&actual, &pack.signature)
    }

    fn install_verified_archive_locked(
        &self,
        pack: &BackendPackManifest,
        archive_path: &Path,
        final_path: &Path,
        stage_root: &Path,
    ) -> ManagerResult<()> {
        // Always re-check at the installation boundary. Bundle extraction or a
        // local filesystem race cannot turn prior verification into trust.
        self.verify_artifact_file(pack, archive_path)?;
        ensure_available_space(
            stage_root,
            pack.download_bytes
                .saturating_add(pack.installed_bytes)
                .saturating_add(INSTALL_SPACE_OVERHEAD_BYTES),
            "extract and install backend pack",
        )?;
        let extract_root = stage_root.join("extracted");
        fs::create_dir_all(&extract_root)?;
        safe_extract_tar_gz(
            archive_path,
            &extract_root,
            pack.installed_bytes.saturating_add(pack.download_bytes),
        )?;
        let payload_root = find_payload_root(&extract_root)?;
        let payload: BackendPayloadManifest = read_json_bounded(
            &payload_root.join(PAYLOAD_MANIFEST_NAME),
            MAX_MANIFEST_BYTES,
        )?;
        validate_payload_manifest(pack, &payload)?;

        let install_root = stage_root.join("installed");
        match &pack.installer {
            BackendInstaller::Extract => {
                fs::rename(&payload_root, &install_root).map_err(|error| {
                    BackendManagerError::new(format!(
                        "stage extracted backend payload {}: {error}",
                        payload_root.display()
                    ))
                })?;
            }
            BackendInstaller::Bootstrap { path } => {
                let script = payload_root.join(path);
                if !script.is_file() {
                    return Err(BackendManagerError::new(format!(
                        "backend bootstrap is missing: {}",
                        script.display()
                    )));
                }
                let status = Command::new("bash")
                    .arg(&script)
                    .arg(&payload_root)
                    .arg(&install_root)
                    .stdin(Stdio::null())
                    .status()
                    .map_err(|error| {
                        BackendManagerError::new(format!(
                            "start trusted backend bootstrap {}: {error}",
                            script.display()
                        ))
                    })?;
                if !status.success() {
                    return Err(BackendManagerError::new(format!(
                        "trusted backend bootstrap failed with {}",
                        status
                    )));
                }
                fs::copy(
                    payload_root.join(PAYLOAD_MANIFEST_NAME),
                    install_root.join(PAYLOAD_MANIFEST_NAME),
                )?;
            }
        }

        validate_tree_symlinks(&install_root, &install_root)?;
        let installed_size = directory_size(&install_root)?;
        let installed_size_limit = pack
            .installed_bytes
            .saturating_add((pack.installed_bytes / 20).max(INSTALLED_SIZE_TOLERANCE_BYTES));
        if installed_size > installed_size_limit {
            return Err(BackendManagerError::new(format!(
                "installed backend expanded to {installed_size} bytes, beyond its signed {installed_size_limit} byte allowance"
            )));
        }

        if !entrypoint_is_usable(&install_root, pack)? {
            return Err(BackendManagerError::new(format!(
                "installed backend entrypoint is missing or unusable: {}",
                install_root.join(&pack.entrypoint).display()
            )));
        }
        for (relative, expected) in &pack.files {
            let path = install_root.join(relative);
            if !path.is_file() {
                return Err(BackendManagerError::new(format!(
                    "installed backend is missing checksummed file {}",
                    relative
                )));
            }
            let actual = sha256_file(&path)?;
            if actual != expected.to_ascii_lowercase() {
                return Err(BackendManagerError::new(format!(
                    "installed backend file {} failed checksum verification",
                    relative
                )));
            }
        }
        // Check both signed digest maps before publishing an installation.
        // A malformed extension must fail here, not only on the next load.
        if let Some(files) = &pack.files_blake3 {
            if !checksums::verify_installed_files(
                &install_root,
                files,
                checksums::Algorithm::Blake3,
            )? {
                return Err(BackendManagerError::new(
                    "installed backend failed BLAKE3 verification",
                ));
            }
        }
        for notice in &pack.licenses {
            if let Some(relative) = notice.path.as_deref() {
                let path = install_root.join(relative);
                if !path.is_file() {
                    return Err(BackendManagerError::new(format!(
                        "installed backend is missing license notice {}",
                        relative
                    )));
                }
            }
        }
        let record = InstalledPackRecord {
            manifest: pack.clone(),
            installed_at_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        let record_bytes = serde_json::to_vec_pretty(&record).map_err(|error| {
            BackendManagerError::new(format!("encode installed backend record: {error}"))
        })?;
        atomic_write(&install_root.join(INSTALL_RECORD_NAME), &record_bytes)?;
        sync_tree_security(&install_root)?;

        if let Some(parent) = final_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let quarantine = if final_path.exists() {
            let quarantine_root = self
                .runtime_cache_root()
                .join(".quarantine")
                .join(&pack.backend);
            fs::create_dir_all(&quarantine_root)?;
            let path = quarantine_root.join(format!("{}-{}", pack.profile, unique_nonce()));
            fs::rename(final_path, &path).map_err(|error| {
                BackendManagerError::new(format!(
                    "quarantine incomplete backend {}: {error}",
                    final_path.display()
                ))
            })?;
            Some(path)
        } else {
            None
        };
        if let Err(error) = fs::rename(&install_root, final_path) {
            if let Some(previous) = quarantine.as_ref() {
                let _ = fs::rename(previous, final_path);
            }
            return Err(BackendManagerError::new(format!(
                "atomically activate backend {}: {error}",
                final_path.display()
            )));
        }
        sync_parent(final_path)?;
        if let Some(previous) = quarantine {
            fs::remove_dir_all(previous)?;
        }
        eprintln!(
            "Installed Kapsl backend {}/{} at {}",
            pack.backend,
            pack.profile,
            final_path.display()
        );
        Ok(())
    }

    fn cleanup_stale_install_stages(&self, pack: &BackendPackManifest) -> ManagerResult<()> {
        let root = self.pack_staging_root(pack)?;
        if !root.is_dir() {
            return Ok(());
        }
        for entry in read_dirs(&root)? {
            let path = entry.path();
            if path.is_dir() {
                fs::remove_dir_all(path)?;
            } else {
                fs::remove_file(path)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn runtime_release_version() -> String {
    // The release workflow injects KAPSL_VERSION at compile time. Never accept
    // a process-time override here: cache namespaces, signed indexes, and ABI
    // compatibility must describe the binary that is actually executing.
    option_env!("KAPSL_VERSION")
        .unwrap_or(env!("CARGO_PKG_VERSION"))
        .to_string()
}

pub(crate) fn backend_cache_root() -> Option<PathBuf> {
    std::env::var_os(BACKEND_CACHE_ENV)
        .map(PathBuf::from)
        .or_else(|| dirs::data_local_dir().map(|path| path.join("kapsl/backends")))
}

pub(crate) fn current_platform() -> String {
    format!(
        "{}-{}",
        std::env::consts::OS,
        normalize_architecture(std::env::consts::ARCH)
    )
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn signed_contract_contains(values: &[String], required: &str) -> bool {
    let required = required.trim();
    !required.is_empty()
        && values
            .iter()
            .any(|value| value.trim().eq_ignore_ascii_case(required))
}

fn compare_pack_versions(left: &str, right: &str) -> std::cmp::Ordering {
    match (semver::Version::parse(left), semver::Version::parse(right)) {
        (Ok(left), Ok(right)) => left.cmp(&right),
        (Ok(_), Err(_)) => std::cmp::Ordering::Greater,
        (Err(_), Ok(_)) => std::cmp::Ordering::Less,
        (Err(_), Err(_)) => {
            let left_numeric = numeric_version_components(left);
            let right_numeric = numeric_version_components(right);
            left_numeric
                .cmp(&right_numeric)
                .then_with(|| left.cmp(right))
        }
    }
}

fn ensure_available_space(path: &Path, required: u64, operation: &str) -> ManagerResult<()> {
    let available = fs2::available_space(path).map_err(|error| {
        BackendManagerError::new(format!(
            "inspect available space at {}: {error}",
            path.display()
        ))
    })?;
    if required > available {
        return Err(BackendManagerError::new(format!(
            "insufficient disk space to {operation}: required {required} bytes, available {available} at {}",
            path.display()
        )));
    }
    Ok(())
}

fn backend_maintenance_lock() -> &'static RwLock<()> {
    static LOCK: OnceLock<RwLock<()>> = OnceLock::new();
    LOCK.get_or_init(|| RwLock::new(()))
}

fn entrypoint_is_usable(root: &Path, pack: &BackendPackManifest) -> ManagerResult<bool> {
    let path = root.join(&pack.entrypoint);
    if !path.is_file() {
        return Ok(false);
    }
    #[cfg(unix)]
    if pack.execution_mode == BackendExecutionMode::External {
        use std::os::unix::fs::PermissionsExt;
        if fs::metadata(path)?.permissions().mode() & 0o111 == 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn sync_tree_security(root: &Path) -> ManagerResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fn visit(path: &Path) -> std::io::Result<()> {
            let metadata = fs::symlink_metadata(path)?;
            if metadata.file_type().is_symlink() {
                return Ok(());
            }
            let mut permissions = metadata.permissions();
            // Installed code is user-owned: strip setuid/setgid and prevent a
            // group or another local account from rewriting a verified pack.
            let mode = permissions.mode() & !0o6022;
            permissions.set_mode(mode);
            fs::set_permissions(path, permissions)?;
            if !metadata.is_dir() {
                return Ok(());
            }
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let child = entry.path();
                visit(&child)?;
            }
            Ok(())
        }
        visit(root)?;
    }
    Ok(())
}

fn is_internal_name(name: &OsStr) -> bool {
    name.to_string_lossy().starts_with('.')
}
