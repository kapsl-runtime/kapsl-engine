use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use kapsl_rag_sdk::manifest::{ConnectorManifest, ConnectorRuntime as ManifestRuntime};
use kapsl_rag_sdk::types::ConnectorConfig;
use serde::Deserialize;
use serde_json::Value;

use crate::runtime::{
    ConnectorClient, ConnectorRuntime as RuntimeTrait, SidecarConnectorRuntime, WasiPermissions,
    WasmConnectorRuntime,
};

#[derive(thiserror::Error, Debug)]
pub enum ExtensionError {
    #[error("io error: {0}")]
    Io(String),
    #[error("manifest not found in {0}")]
    ManifestMissing(String),
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("extension not installed: {0}")]
    NotInstalled(String),
    #[error("runtime error: {0}")]
    Runtime(String),
}

impl From<io::Error> for ExtensionError {
    fn from(err: io::Error) -> Self {
        ExtensionError::Io(err.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct InstalledExtension {
    pub manifest: ConnectorManifest,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ExtensionRegistry {
    pub root: PathBuf,
}

impl ExtensionRegistry {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn discover(&self) -> Result<Vec<InstalledExtension>, ExtensionError> {
        let mut extensions = Vec::new();
        if !self.root.exists() {
            return Ok(extensions);
        }
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if let Ok(manifest) = load_manifest(&path) {
                extensions.push(InstalledExtension { manifest, path });
            }
        }
        Ok(extensions)
    }

    pub fn install_from_dir(&self, source: &Path) -> Result<InstalledExtension, ExtensionError> {
        let manifest = load_manifest(source)?;
        let target = self.root.join(&manifest.id);
        if target.exists() {
            fs::remove_dir_all(&target)?;
        }
        copy_dir_all(source, &target)?;
        Ok(InstalledExtension {
            manifest,
            path: target,
        })
    }

    pub fn uninstall(&self, extension_id: &str) -> Result<(), ExtensionError> {
        let target = self.root.join(extension_id);
        if !target.exists() {
            return Err(ExtensionError::NotInstalled(extension_id.to_string()));
        }
        fs::remove_dir_all(target)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ExtensionManager {
    pub registry: ExtensionRegistry,
    pub config_root: PathBuf,
}

impl ExtensionManager {
    pub fn new(registry: ExtensionRegistry, config_root: impl Into<PathBuf>) -> Self {
        Self {
            registry,
            config_root: config_root.into(),
        }
    }

    pub fn set_workspace_config(
        &self,
        workspace_id: &str,
        extension_id: &str,
        config: &ConnectorConfig,
    ) -> Result<(), ExtensionError> {
        let dir = self.config_root.join(workspace_id);
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{extension_id}.json"));
        let data = serde_json::to_vec_pretty(config)
            .map_err(|e| ExtensionError::InvalidManifest(e.to_string()))?;
        fs::write(path, data)?;
        Ok(())
    }

    pub fn get_workspace_config(
        &self,
        workspace_id: &str,
        extension_id: &str,
    ) -> Result<Option<ConnectorConfig>, ExtensionError> {
        let path = self
            .config_root
            .join(workspace_id)
            .join(format!("{extension_id}.json"));
        if !path.exists() {
            return Ok(None);
        }
        let data = fs::read_to_string(path)?;
        let config = serde_json::from_str(&data)
            .map_err(|e| ExtensionError::InvalidManifest(e.to_string()))?;
        Ok(Some(config))
    }

    pub fn get_workspace_connector_config(
        &self,
        workspace_id: &str,
        extension_id: &str,
    ) -> Result<Option<ConnectorConfig>, ExtensionError> {
        let config = self.get_workspace_config(workspace_id, extension_id)?;
        Ok(config.map(strip_wasi_block))
    }

    pub fn list_configs(
        &self,
        workspace_id: &str,
    ) -> Result<HashMap<String, ConnectorConfig>, ExtensionError> {
        let mut configs = HashMap::new();
        let dir = self.config_root.join(workspace_id);
        if !dir.exists() {
            return Ok(configs);
        }
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                let data = fs::read_to_string(&path)?;
                let config = serde_json::from_str(&data)
                    .map_err(|e| ExtensionError::InvalidManifest(e.to_string()))?;
                configs.insert(stem.to_string(), config);
            }
        }
        Ok(configs)
    }

    pub fn get_workspace_wasi_permissions(
        &self,
        workspace_id: &str,
        extension_id: &str,
    ) -> Result<WasiPermissions, ExtensionError> {
        let config = self.get_workspace_config(workspace_id, extension_id)?;
        wasi_permissions_from_config(config.as_ref())
    }

    pub fn launch_connector(
        &self,
        workspace_id: &str,
        extension: &InstalledExtension,
    ) -> Result<ConnectorClient<ConnectorRuntimeHandle>, ExtensionError> {
        let entrypoint = resolve_entrypoint(extension)?;
        let runtime = match extension.manifest.runtime {
            ManifestRuntime::Wasm => {
                let permissions =
                    self.get_workspace_wasi_permissions(workspace_id, &extension.manifest.id)?;
                ConnectorRuntimeHandle::Wasm(
                    WasmConnectorRuntime::spawn_with_permissions(&entrypoint, permissions)
                        .map_err(|e| ExtensionError::Runtime(e.to_string()))?,
                )
            }
            ManifestRuntime::Sidecar => {
                let runtime = SidecarConnectorRuntime::spawn(&entrypoint)
                    .map_err(|e| ExtensionError::Runtime(e.to_string()))?;
                ConnectorRuntimeHandle::Sidecar(runtime)
            }
        };
        Ok(ConnectorClient::new(runtime))
    }
}

pub enum ConnectorRuntimeHandle {
    Wasm(WasmConnectorRuntime),
    Sidecar(SidecarConnectorRuntime),
}

impl RuntimeTrait for ConnectorRuntimeHandle {
    fn send(
        &mut self,
        request: kapsl_rag_sdk::protocol::ConnectorRequest,
    ) -> Result<kapsl_rag_sdk::protocol::ConnectorResponse, crate::runtime::RuntimeError> {
        match self {
            ConnectorRuntimeHandle::Wasm(runtime) => runtime.send(request),
            ConnectorRuntimeHandle::Sidecar(runtime) => runtime.send(request),
        }
    }

    fn close(&mut self) -> Result<(), crate::runtime::RuntimeError> {
        match self {
            ConnectorRuntimeHandle::Wasm(runtime) => runtime.close(),
            ConnectorRuntimeHandle::Sidecar(runtime) => runtime.close(),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct WasiConfig {
    #[serde(default)]
    env: HashMap<String, String>,
    #[serde(default)]
    preopen_dirs: Vec<WasiDirConfig>,
}

#[derive(Debug, Deserialize)]
struct WasiDirConfig {
    host_path: String,
    guest_path: String,
    #[serde(default)]
    read_only: bool,
}

fn wasi_permissions_from_config(
    config: Option<&ConnectorConfig>,
) -> Result<WasiPermissions, ExtensionError> {
    let Some(config) = config else {
        return Ok(WasiPermissions::default());
    };
    let obj = match config {
        serde_json::Value::Object(_) => config,
        _ => return Ok(WasiPermissions::default()),
    };

    let wasi_value = obj.get("wasi");
    if wasi_value.is_none() {
        return Ok(WasiPermissions::default());
    }
    let wasi_value = wasi_value.unwrap();
    let parsed: WasiConfig = serde_json::from_value(wasi_value.clone())
        .map_err(|e| ExtensionError::InvalidConfig(e.to_string()))?;

    let mut permissions = WasiPermissions::default();
    for (key, value) in parsed.env {
        validate_env_kv(&key, &value)?;
        permissions = permissions.with_env(key, value);
    }

    for dir in parsed.preopen_dirs {
        validate_host_path(&dir.host_path)?;
        validate_guest_path(&dir.guest_path)?;
        permissions =
            permissions.allow_dir(PathBuf::from(dir.host_path), dir.guest_path, dir.read_only);
    }

    Ok(permissions)
}

fn strip_wasi_block(config: ConnectorConfig) -> ConnectorConfig {
    match config {
        Value::Object(mut map) => {
            map.remove("wasi");
            Value::Object(map)
        }
        other => other,
    }
}

fn validate_env_kv(key: &str, value: &str) -> Result<(), ExtensionError> {
    if key.is_empty() {
        return Err(ExtensionError::InvalidConfig(
            "env key cannot be empty".to_string(),
        ));
    }
    if key.contains('\0') || value.contains('\0') {
        return Err(ExtensionError::InvalidConfig(
            "env key/value cannot contain NUL".to_string(),
        ));
    }
    Ok(())
}

fn validate_guest_path(path: &str) -> Result<(), ExtensionError> {
    if path.is_empty() || !path.starts_with('/') {
        return Err(ExtensionError::InvalidConfig(
            "preopened guest path must be absolute".to_string(),
        ));
    }
    if path.contains('\0') {
        return Err(ExtensionError::InvalidConfig(
            "preopened guest path cannot contain NUL".to_string(),
        ));
    }
    Ok(())
}

fn validate_host_path(path: &str) -> Result<(), ExtensionError> {
    let host_path = Path::new(path);
    if !host_path.is_absolute() {
        return Err(ExtensionError::InvalidConfig(
            "preopened host path must be absolute".to_string(),
        ));
    }
    Ok(())
}

fn load_manifest(dir: &Path) -> Result<ConnectorManifest, ExtensionError> {
    let toml_path = dir.join("rag-extension.toml");
    let json_path = dir.join("rag-extension.json");

    if toml_path.exists() {
        let data = fs::read_to_string(&toml_path)?;
        let manifest =
            toml::from_str(&data).map_err(|e| ExtensionError::InvalidManifest(e.to_string()))?;
        return Ok(manifest);
    }

    if json_path.exists() {
        let data = fs::read_to_string(&json_path)?;
        let manifest = serde_json::from_str(&data)
            .map_err(|e| ExtensionError::InvalidManifest(e.to_string()))?;
        return Ok(manifest);
    }

    Err(ExtensionError::ManifestMissing(dir.display().to_string()))
}

fn resolve_entrypoint(extension: &InstalledExtension) -> Result<PathBuf, ExtensionError> {
    let entry = extension.manifest.entrypoint.as_deref();
    let runtime = &extension.manifest.runtime;
    let default_entry = match runtime {
        ManifestRuntime::Wasm => "connector.wasm",
        ManifestRuntime::Sidecar => "connector",
    };
    let entry = entry.unwrap_or(default_entry);
    let path = Path::new(entry);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        extension.path.join(entry)
    };
    if !resolved.exists() {
        return Err(ExtensionError::InvalidConfig(format!(
            "entrypoint not found: {}",
            resolved.display()
        )));
    }
    Ok(resolved)
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<(), ExtensionError> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}
