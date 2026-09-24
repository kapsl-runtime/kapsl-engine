//! Validation of signed pack manifests against this runtime and target.

use super::*;

pub(super) fn validate_signed_contract_values(
    label: &str,
    values: &[String],
    allow_empty: bool,
) -> ManagerResult<()> {
    if values.is_empty() && !allow_empty {
        return Err(BackendManagerError::new(format!(
            "standard native backend pack must declare non-empty {label}"
        )));
    }
    let mut normalized = HashSet::new();
    for value in values {
        let value = value.trim().to_ascii_lowercase();
        if value.is_empty() {
            return Err(BackendManagerError::new(format!(
                "standard native backend pack contains an empty {label} value"
            )));
        }
        if !normalized.insert(value.clone()) {
            return Err(BackendManagerError::new(format!(
                "standard native backend pack repeats {label} value `{value}`"
            )));
        }
    }
    Ok(())
}

pub(super) fn accelerator_matches(profile: &str, target: BackendAccelerator) -> bool {
    match profile.trim().to_ascii_lowercase().as_str() {
        "cpu" => target == BackendAccelerator::Cpu,
        "cuda" => matches!(
            target,
            BackendAccelerator::Cuda | BackendAccelerator::TensorRt
        ),
        "tensorrt" => target == BackendAccelerator::TensorRt,
        _ => false,
    }
}

pub(super) fn normalize_architecture(value: &str) -> &str {
    match value {
        "amd64" => "x86_64",
        "arm64" => "aarch64",
        other => other,
    }
}

pub(super) fn version_at_least(actual: &str, minimum: &str) -> bool {
    let mut actual = numeric_version_components(actual);
    let mut minimum = numeric_version_components(minimum);
    if actual.is_empty() || minimum.is_empty() {
        return false;
    }
    let length = actual.len().max(minimum.len());
    actual.resize(length, 0);
    minimum.resize(length, 0);
    actual >= minimum
}

pub(super) fn numeric_version_components(value: &str) -> Vec<u64> {
    value
        .split(|character: char| !character.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .take(4)
        .map(|part| part.parse::<u64>().unwrap_or(0))
        .collect()
}

pub(super) fn validate_cache_component(label: &str, value: &str) -> ManagerResult<()> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'+'))
    {
        return Err(BackendManagerError::new(format!(
            "invalid backend {label} `{value}`"
        )));
    }
    Ok(())
}

pub(super) fn validate_relative_pack_path(label: &str, value: &str) -> ManagerResult<()> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(BackendManagerError::new(format!(
            "invalid {label} `{value}`: expected a normalized relative path"
        )));
    }
    Ok(())
}

pub(super) fn validate_sha256(value: &str) -> ManagerResult<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(BackendManagerError::new(format!(
            "invalid SHA-256 digest `{value}`"
        )));
    }
    Ok(())
}

pub(super) fn validate_payload_manifest(
    signed: &BackendPackManifest,
    payload: &BackendPayloadManifest,
) -> ManagerResult<()> {
    let matches = payload.schema_version == signed.schema_version
        && payload.backend == signed.backend
        && payload.profile == signed.profile
        && payload.pack_version == signed.pack_version
        && payload.runtime_abi == signed.runtime_abi
        && payload.adapter_abi == signed.adapter_abi
        && payload.platform == signed.platform
        && payload.execution_mode == signed.execution_mode
        && payload.kv_mode == signed.kv_mode
        && payload.entrypoint == signed.entrypoint;
    if matches {
        Ok(())
    } else {
        Err(BackendManagerError::new(format!(
            "backend payload manifest does not match the signed index entry: payload={payload:?}"
        )))
    }
}

impl BackendManager {
    pub(super) fn validate_pack_manifest(
        &self,
        pack: &BackendPackManifest,
        target: Option<&BackendTarget>,
    ) -> ManagerResult<()> {
        if pack.schema_version != BACKEND_PACK_SCHEMA_VERSION {
            return Err(BackendManagerError::new(format!(
                "pack schema {} is unsupported",
                pack.schema_version
            )));
        }
        if pack.runtime_abi != BACKEND_RUNTIME_ABI {
            return Err(BackendManagerError::new(format!(
                "backend ABI {} does not match runtime ABI {}",
                pack.runtime_abi, BACKEND_RUNTIME_ABI
            )));
        }
        if let Some(adapter_abi) = pack.adapter_abi.as_deref() {
            if pack.execution_mode != BackendExecutionMode::Native {
                return Err(BackendManagerError::new(
                    "only native backend packs may declare adapter_abi",
                ));
            }
            if adapter_abi != STANDARD_NATIVE_ADAPTER_ABI {
                return Err(BackendManagerError::new(format!(
                    "unsupported native adapter ABI `{adapter_abi}`"
                )));
            }
            validate_signed_contract_values("formats", &pack.formats, false)?;
            validate_signed_contract_values("model_types", &pack.model_types, true)?;
            validate_signed_contract_values("tasks", &pack.tasks, false)?;
            if !pack.capabilities.memory_reporting {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} must declare memory_reporting",
                    pack.backend, pack.profile
                )));
            }
            if pack.capabilities.scoped_device_allocator
                && !pack.capabilities.governed_device_allocator
            {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} declares a scoped allocator without a governed device allocator",
                    pack.backend, pack.profile
                )));
            }
            if pack.accelerator_profile != "cpu" && !pack.capabilities.governed_device_allocator {
                return Err(BackendManagerError::new(format!(
                    "standard native accelerator pack {}/{} must use the governed device allocator",
                    pack.backend, pack.profile
                )));
            }
            if pack.capabilities.scoped_device_allocator
                && pack
                    .memory_behavior
                    .allocation_scope
                    .as_deref()
                    .is_none_or(|scope| scope.trim().is_empty())
            {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} must name its scoped allocation contract",
                    pack.backend, pack.profile
                )));
            }
            if pack
                .memory_behavior
                .device_allocation
                .as_deref()
                .is_none_or(|behavior| behavior.trim().is_empty())
            {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} must declare its device allocation behavior",
                    pack.backend, pack.profile
                )));
            }
            if pack.memory_behavior.synchronize_before_free
                && !pack.capabilities.governed_device_allocator
            {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} cannot request host synchronization without a governed device allocator",
                    pack.backend, pack.profile
                )));
            }
            if !pack.memory_behavior.planned_reporting
                || !pack.memory_behavior.live_reporting
                || !pack.memory_behavior.request_reporting
            {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} must declare planned, live, and request memory reporting",
                    pack.backend, pack.profile
                )));
            }
            if pack.accelerator_requirements.kind.as_deref()
                != Some(pack.accelerator_profile.as_str())
            {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} accelerator requirement does not match profile `{}`",
                    pack.backend, pack.profile, pack.accelerator_profile
                )));
            }
            validate_signed_contract_values(
                "execution providers",
                &pack.accelerator_requirements.execution_providers,
                false,
            )?;
            if pack.accelerator_requirements.implicit_cpu_fallback != Some(false) {
                return Err(BackendManagerError::new(format!(
                    "standard native backend pack {}/{} must explicitly disable implicit CPU fallback",
                    pack.backend, pack.profile
                )));
            }
        }
        validate_cache_component("backend", &pack.backend)?;
        validate_cache_component("profile", &pack.profile)?;
        if !matches!(
            pack.accelerator_profile.as_str(),
            "cpu" | "cuda" | "tensorrt"
        ) {
            return Err(BackendManagerError::new(format!(
                "unsupported accelerator profile `{}`",
                pack.accelerator_profile
            )));
        }
        match (pack.backend.as_str(), pack.kv_mode.as_deref()) {
            ("llama-cpp", Some("native" | "shared_pool")) => {}
            ("llama-cpp", None) => {
                return Err(BackendManagerError::new(
                    "llama.cpp backend packs must sign an explicit kv_mode",
                ));
            }
            ("llama-cpp", Some(kv_mode)) => {
                return Err(BackendManagerError::new(format!(
                    "unsupported backend KV mode `{kv_mode}` for llama-cpp"
                )));
            }
            (backend, Some(kv_mode)) => {
                return Err(BackendManagerError::new(format!(
                    "backend {backend} cannot declare KV mode `{kv_mode}`"
                )));
            }
            (_, None) => {}
        }
        validate_relative_pack_path("entrypoint", &pack.entrypoint)?;
        if let BackendInstaller::Bootstrap { path } = &pack.installer {
            validate_relative_pack_path("bootstrap path", path)?;
        }
        validate_sha256(&pack.sha256)?;
        let _ = decode_signature(&pack.signature)?;
        if !(pack.artifact.starts_with("https://")
            || self.config.allow_file_artifacts && pack.artifact.starts_with("file://"))
        {
            return Err(BackendManagerError::new(format!(
                "backend artifact URL must use HTTPS: {}",
                pack.artifact
            )));
        }
        if pack.download_bytes == 0 || pack.installed_bytes == 0 {
            return Err(BackendManagerError::new(
                "backend pack sizes must both be greater than zero",
            ));
        }
        let runtime = Version::parse(&self.config.runtime_version).map_err(|error| {
            BackendManagerError::new(format!(
                "runtime version `{}` is not valid semver: {error}",
                self.config.runtime_version
            ))
        })?;
        let compatibility = VersionReq::parse(&pack.compatible_kapsl).map_err(|error| {
            BackendManagerError::new(format!(
                "invalid compatible_kapsl requirement `{}`: {error}",
                pack.compatible_kapsl
            ))
        })?;
        if !compatibility.matches(&runtime) {
            return Err(BackendManagerError::new(format!(
                "pack requires Kapsl `{}`, runtime is {}",
                pack.compatible_kapsl, runtime
            )));
        }
        for (path, digest) in &pack.files {
            validate_relative_pack_path("file checksum path", path)?;
            validate_sha256(digest)?;
        }
        if let Some(files) = &pack.files_blake3 {
            if !files.keys().eq(pack.files.keys()) {
                return Err(BackendManagerError::new(
                    "BLAKE3 file set must exactly match the signed SHA-256 file set",
                ));
            }
            for digest in files.values() {
                if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    return Err(BackendManagerError::new("invalid BLAKE3 digest"));
                }
            }
        }
        if !pack.files.contains_key(&pack.entrypoint) {
            return Err(BackendManagerError::new(format!(
                "backend entrypoint `{}` requires a signed installed-file checksum",
                pack.entrypoint
            )));
        }
        for notice in &pack.licenses {
            if let Some(path) = notice.path.as_deref() {
                validate_relative_pack_path("license path", path)?;
            }
        }
        for (label, minimum) in [
            ("minimum_cuda", pack.minimum_cuda.as_deref()),
            ("minimum_driver", pack.minimum_driver.as_deref()),
        ] {
            if minimum.is_some_and(|value| numeric_version_components(value).is_empty()) {
                return Err(BackendManagerError::new(format!(
                    "backend {label} must contain a numeric version"
                )));
            }
        }
        if let Some(target) = target {
            if pack.platform != target.platform {
                return Err(BackendManagerError::new(format!(
                    "platform {} does not match {}",
                    pack.platform, target.platform
                )));
            }
            if normalize_architecture(&pack.architecture)
                != normalize_architecture(&target.architecture)
            {
                return Err(BackendManagerError::new(format!(
                    "architecture {} does not match {}",
                    pack.architecture, target.architecture
                )));
            }
            if !accelerator_matches(&pack.accelerator_profile, target.accelerator) {
                return Err(BackendManagerError::new(format!(
                    "accelerator profile {} does not match {}",
                    pack.accelerator_profile,
                    target.accelerator.as_str()
                )));
            }
            if let Some(minimum) = pack.minimum_cuda.as_deref() {
                let actual = target.cuda_version.as_deref().ok_or_else(|| {
                    BackendManagerError::new(format!("CUDA {minimum} or newer is required"))
                })?;
                if !version_at_least(actual, minimum) {
                    return Err(BackendManagerError::new(format!(
                        "CUDA {minimum} or newer is required; detected {actual}"
                    )));
                }
            }
            if let Some(minimum) = pack.minimum_driver.as_deref() {
                let actual = target.driver_version.as_deref().ok_or_else(|| {
                    BackendManagerError::new(format!("driver {minimum} or newer is required"))
                })?;
                if !version_at_least(actual, minimum) {
                    return Err(BackendManagerError::new(format!(
                        "driver {minimum} or newer is required; detected {actual}"
                    )));
                }
            }
        }
        Ok(())
    }
}
