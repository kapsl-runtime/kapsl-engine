//! Loading, caching, and signature verification of the backend index.

use super::*;

pub(super) fn default_backend_index_url(runtime_version: &str) -> String {
    let channel = if runtime_version.contains("-beta.") {
        "runtime/beta"
    } else {
        "runtime"
    };
    let base =
        std::env::var("KAPSL_BASE_URL").unwrap_or_else(|_| DEFAULT_DOWNLOAD_BASE_URL.to_string());
    format!(
        "{}/{channel}/v{runtime_version}/backend-index.json",
        base.trim_end_matches('/')
    )
}

pub(super) fn trusted_public_keys() -> ManagerResult<Vec<VerifyingKey>> {
    let mut encoded = Vec::new();
    if let Some(keys) = option_env!("KAPSL_BACKEND_PUBLIC_KEYS") {
        encoded.push(keys.to_string());
    }
    if let Ok(keys) = std::env::var(BACKEND_PUBLIC_KEYS_ENV) {
        encoded.push(keys);
    }
    let mut result = Vec::new();
    for value in encoded {
        for candidate in value
            .split(|character: char| {
                character == ',' || character == ';' || character.is_whitespace()
            })
            .filter(|candidate| !candidate.is_empty())
        {
            let candidate = candidate.strip_prefix("ed25519:").unwrap_or(candidate);
            let bytes = decode_base64(candidate).map_err(|error| {
                BackendManagerError::new(format!("invalid trusted backend public key: {error}"))
            })?;
            let bytes: [u8; 32] = bytes.try_into().map_err(|_| {
                BackendManagerError::new("trusted Ed25519 public keys must contain 32 bytes")
            })?;
            let key = VerifyingKey::from_bytes(&bytes).map_err(|error| {
                BackendManagerError::new(format!("invalid Ed25519 public key: {error}"))
            })?;
            if !result.contains(&key) {
                result.push(key);
            }
        }
    }
    Ok(result)
}

pub(super) fn decode_signature(encoded: &str) -> ManagerResult<Signature> {
    let encoded = encoded.strip_prefix("ed25519:").unwrap_or(encoded);
    let bytes = decode_base64(encoded).map_err(|error| {
        BackendManagerError::new(format!("invalid Ed25519 signature encoding: {error}"))
    })?;
    Signature::from_slice(&bytes).map_err(|error| {
        BackendManagerError::new(format!("invalid Ed25519 signature length: {error}"))
    })
}

pub(super) fn decode_base64(value: &str) -> Result<Vec<u8>, base64::DecodeError> {
    BASE64
        .decode(value)
        .or_else(|_| BASE64_URL_SAFE.decode(value))
}

impl BackendManager {
    pub(crate) fn load_index(&self) -> ManagerResult<BackendIndex> {
        let index_dir = self.runtime_cache_root().join(".index");
        fs::create_dir_all(&index_dir)?;
        let lock_path = index_dir.join("index.lock");
        let process_lock = process_path_lock(&lock_path);
        let _process_guard = process_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let lock = lock_file(&lock_path)?;
        lock.lock_exclusive().map_err(|error| {
            BackendManagerError::new(format!("lock backend index cache: {error}"))
        })?;

        let cached_index = index_dir.join("backend-index.json");
        let cached_signature = index_dir.join("backend-index.json.sig");
        if cached_index.is_file() && cached_signature.is_file() {
            match self.read_and_verify_index(&cached_index, &cached_signature) {
                Ok(index) => {
                    let _ = fs2::FileExt::unlock(&lock);
                    return Ok(index);
                }
                Err(error) if self.config.offline => {
                    let _ = fs2::FileExt::unlock(&lock);
                    return Err(BackendManagerError::new(format!(
                        "cached backend index is invalid while offline: {error}"
                    )));
                }
                Err(error) => {
                    eprintln!("Ignoring invalid cached backend index: {error}");
                    quarantine_file(&cached_index)?;
                    quarantine_file(&cached_signature)?;
                }
            }
        }

        if self.config.offline {
            let _ = fs2::FileExt::unlock(&lock);
            return Err(BackendManagerError::new(
                "no verified backend index is cached and Kapsl is offline",
            ));
        }

        let (index_bytes, signature_bytes) = self.read_index_source()?;
        let signature_text = std::str::from_utf8(&signature_bytes)
            .map_err(|error| {
                BackendManagerError::new(format!(
                    "backend index signature is not UTF-8 text: {error}"
                ))
            })?
            .trim();
        self.verify_index_signature(&index_bytes, signature_text)?;
        let index = self.decode_and_validate_index(&index_bytes)?;
        atomic_write(&cached_index, &index_bytes)?;
        atomic_write(&cached_signature, signature_text.as_bytes())?;
        let _ = fs2::FileExt::unlock(&lock);
        Ok(index)
    }

    pub(crate) fn verify_index_bytes(
        &self,
        index_bytes: &[u8],
        signature: &str,
    ) -> ManagerResult<BackendIndex> {
        self.verify_index_signature(index_bytes, signature.trim())?;
        self.decode_and_validate_index(index_bytes)
    }

    /// Anchor an index carried by a verified offline bundle into the ordinary
    /// runtime cache. Subsequent lazy-pack validation therefore uses the same
    /// signed trust path without requiring network access. A different valid
    /// index for the same immutable runtime release is rejected.
    pub(crate) fn cache_verified_index_bytes(
        &self,
        index_bytes: &[u8],
        signature: &str,
    ) -> ManagerResult<BackendIndex> {
        let index = self.verify_index_bytes(index_bytes, signature)?;
        let index_dir = self.runtime_cache_root().join(".index");
        fs::create_dir_all(&index_dir)?;
        let lock_path = index_dir.join("index.lock");
        let process_lock = process_path_lock(&lock_path);
        let _process_guard = process_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let lock = lock_file(&lock_path)?;
        lock.lock_exclusive().map_err(|error| {
            BackendManagerError::new(format!("lock backend index cache: {error}"))
        })?;

        let cached_index = index_dir.join("backend-index.json");
        let cached_signature = index_dir.join("backend-index.json.sig");
        if cached_index.is_file() || cached_signature.is_file() {
            if !cached_index.is_file() || !cached_signature.is_file() {
                let _ = fs2::FileExt::unlock(&lock);
                return Err(BackendManagerError::new(
                    "backend index cache is incomplete; refusing to replace it from an offline bundle",
                ));
            }
            let existing_bytes = read_file_bounded(&cached_index, MAX_INDEX_BYTES)?;
            let existing_signature = read_file_bounded(&cached_signature, 64 * 1024)?;
            let existing_signature = std::str::from_utf8(&existing_signature)
                .map_err(|error| BackendManagerError::new(error.to_string()))?
                .trim();
            self.verify_index_bytes(&existing_bytes, existing_signature)?;
            if existing_bytes != index_bytes || existing_signature != signature.trim() {
                let _ = fs2::FileExt::unlock(&lock);
                return Err(BackendManagerError::new(
                    "offline bundle carries a different signed backend index for this immutable runtime release",
                ));
            }
            let _ = fs2::FileExt::unlock(&lock);
            return Ok(index);
        }

        atomic_write(&cached_index, index_bytes)?;
        atomic_write(&cached_signature, signature.trim().as_bytes())?;
        let _ = fs2::FileExt::unlock(&lock);
        Ok(index)
    }

    pub(crate) fn verified_index_material(&self) -> ManagerResult<(BackendIndex, Vec<u8>, String)> {
        let _ = self.load_index()?;
        let index_path = self.runtime_cache_root().join(".index/backend-index.json");
        let signature_path = self
            .runtime_cache_root()
            .join(".index/backend-index.json.sig");
        let bytes = read_file_bounded(&index_path, MAX_INDEX_BYTES)?;
        let signature_bytes = read_file_bounded(&signature_path, 64 * 1024)?;
        let signature = std::str::from_utf8(&signature_bytes)
            .map_err(|error| BackendManagerError::new(error.to_string()))?
            .trim()
            .to_string();
        let index = self.verify_index_bytes(&bytes, &signature)?;
        Ok((index, bytes, signature))
    }

    pub(super) fn read_index_source(&self) -> ManagerResult<(Vec<u8>, Vec<u8>)> {
        if let Some(index_path) = self.config.index_path.as_ref() {
            let bytes = read_file_bounded(index_path, MAX_INDEX_BYTES)?;
            let signature_path = PathBuf::from(format!("{}.sig", index_path.display()));
            let signature = read_file_bounded(&signature_path, 64 * 1024)?;
            return Ok((bytes, signature));
        }
        if !self.config.index_url.starts_with("https://") {
            return Err(BackendManagerError::new(format!(
                "backend index URL must use HTTPS: {}",
                self.config.index_url
            )));
        }
        let bytes = download_bytes(&self.config.index_url, MAX_INDEX_BYTES)?;
        let signature = download_bytes(&format!("{}.sig", self.config.index_url), 64 * 1024)?;
        Ok((bytes, signature))
    }

    pub(super) fn read_and_verify_index(
        &self,
        index_path: &Path,
        signature_path: &Path,
    ) -> ManagerResult<BackendIndex> {
        let bytes = read_file_bounded(index_path, MAX_INDEX_BYTES)?;
        let signature_bytes = read_file_bounded(signature_path, 64 * 1024)?;
        let signature = std::str::from_utf8(&signature_bytes)
            .map_err(|error| BackendManagerError::new(error.to_string()))?
            .trim();
        self.verify_index_bytes(&bytes, signature)
    }

    pub(super) fn decode_and_validate_index(&self, bytes: &[u8]) -> ManagerResult<BackendIndex> {
        let index: BackendIndex = serde_json::from_slice(bytes).map_err(|error| {
            BackendManagerError::new(format!("invalid backend-index.json: {error}"))
        })?;
        if index.schema_version != BACKEND_INDEX_SCHEMA_VERSION {
            return Err(BackendManagerError::new(format!(
                "backend index schema {} is unsupported; expected {}",
                index.schema_version, BACKEND_INDEX_SCHEMA_VERSION
            )));
        }
        if index.runtime_version != self.config.runtime_version {
            return Err(BackendManagerError::new(format!(
                "backend index targets Kapsl {}, but this runtime is {}",
                index.runtime_version, self.config.runtime_version
            )));
        }
        if index.packs.is_empty() {
            return Err(BackendManagerError::new("backend index contains no packs"));
        }
        let mut identities = HashSet::new();
        for pack in &index.packs {
            self.validate_pack_manifest(pack, None)?;
            if !identities.insert((
                pack.backend.clone(),
                pack.profile.clone(),
                pack.pack_version.clone(),
                pack.platform.clone(),
                normalize_architecture(&pack.architecture).to_string(),
                pack.accelerator_profile.clone(),
            )) {
                return Err(BackendManagerError::new(format!(
                    "backend index contains duplicate pack identity {}/{} version {} for {}-{} ({})",
                    pack.backend,
                    pack.profile,
                    pack.pack_version,
                    pack.platform,
                    pack.architecture,
                    pack.accelerator_profile
                )));
            }
        }
        Ok(index)
    }

    pub(super) fn verify_index_signature(&self, bytes: &[u8], encoded: &str) -> ManagerResult<()> {
        let mut message = b"kapsl-backend-index-v1\0".to_vec();
        message.extend_from_slice(bytes);
        self.verify_signature(&message, encoded, "backend index")
    }

    pub(super) fn verify_artifact_signature(
        &self,
        digest: &str,
        encoded: &str,
    ) -> ManagerResult<()> {
        let message = format!("kapsl-backend-artifact-v1\0sha256:{digest}");
        self.verify_signature(message.as_bytes(), encoded, "backend artifact")
    }

    pub(super) fn verify_signature(
        &self,
        message: &[u8],
        encoded: &str,
        label: &str,
    ) -> ManagerResult<()> {
        if self.config.trusted_keys.is_empty() {
            return Err(BackendManagerError::new(format!(
                "cannot verify {label}: this Kapsl build has no trusted backend signing key; official builds must embed KAPSL_BACKEND_PUBLIC_KEYS"
            )));
        }
        let signature = decode_signature(encoded)?;
        if self
            .config
            .trusted_keys
            .iter()
            .any(|key| key.verify(message, &signature).is_ok())
        {
            return Ok(());
        }
        Err(BackendManagerError::new(format!(
            "{label} signature verification failed"
        )))
    }
}
