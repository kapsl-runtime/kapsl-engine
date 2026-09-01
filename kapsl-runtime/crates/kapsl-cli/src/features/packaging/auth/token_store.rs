use super::auth_base_url_from_remote_url;
use crate::app::{REMOTE_TOKEN_ENV, REMOTE_TOKEN_STORE_PATH_ENV};
use crate::features::auth::format_authorization_header;
use crate::runtime::optional_env_var;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct RemoteTokenStoreFile {
    #[serde(default)]
    tokens_by_remote: HashMap<String, String>,
    #[serde(default)]
    last_remote_url: Option<String>,
}

fn remote_token_store_key(remote_url: &str) -> String {
    auth_base_url_from_remote_url(remote_url).unwrap_or_else(|_| remote_url.trim().to_string())
}

fn resolve_remote_token_store_path() -> PathBuf {
    if let Some(path) = optional_env_var(REMOTE_TOKEN_STORE_PATH_ENV) {
        return PathBuf::from(path);
    }

    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".kapsl").join("remote-token-store.json")
}

fn load_remote_token_store(path: &Path) -> RemoteTokenStoreFile {
    let Ok(raw) = fs::read_to_string(path) else {
        return RemoteTokenStoreFile::default();
    };
    serde_json::from_str(&raw).unwrap_or_default()
}

fn save_remote_token_store(path: &Path, store: &RemoteTokenStoreFile) -> Result<(), String> {
    let parent = path.parent().ok_or_else(|| {
        format!(
            "Invalid token store path (missing parent directory): {}",
            path.display()
        )
    })?;
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "Failed to create token store directory {}: {}",
            parent.display(),
            error
        )
    })?;

    let raw = serde_json::to_string_pretty(store)
        .map_err(|error| format!("Failed to serialize token store: {error}"))?;
    fs::write(path, raw)
        .map_err(|error| format!("Failed to write token store {}: {error}", path.display()))
}

fn read_stored_remote_token_for_remote(remote_url: &str) -> Option<String> {
    let path = resolve_remote_token_store_path();
    let store = load_remote_token_store(&path);
    store
        .tokens_by_remote
        .get(&remote_token_store_key(remote_url))
        .cloned()
}

pub(super) fn read_last_remote_url_from_store() -> Option<String> {
    let store = load_remote_token_store(&resolve_remote_token_store_path());
    store
        .last_remote_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

pub(crate) fn store_remote_token_for_remote(
    remote_url: &str,
    token: &str,
) -> Result<PathBuf, String> {
    let path = resolve_remote_token_store_path();
    let mut store = load_remote_token_store(&path);
    let trimmed_remote_url = remote_url.trim();
    if !trimmed_remote_url.is_empty() {
        store.last_remote_url = Some(trimmed_remote_url.to_string());
    }
    store
        .tokens_by_remote
        .insert(remote_token_store_key(remote_url), token.trim().to_string());
    save_remote_token_store(&path, &store)?;
    Ok(path)
}

pub(crate) fn resolved_remote_token(
    remote_url: &str,
    custom_token: Option<&str>,
) -> Option<String> {
    if let Some(explicit) = format_authorization_header(custom_token) {
        return Some(explicit);
    }
    let environment_token = optional_env_var(REMOTE_TOKEN_ENV);
    if let Some(environment_header) = format_authorization_header(environment_token.as_deref()) {
        return Some(environment_header);
    }
    format_authorization_header(read_stored_remote_token_for_remote(remote_url).as_deref())
}
