//! Filesystem representation and location of the authentication store.

use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct ApiAuthStoreFile {
    #[serde(default)]
    pub(crate) users: Vec<ApiAuthUser>,
    #[serde(default)]
    pub(crate) api_keys: Vec<ApiAuthKey>,
}

impl ApiAuthStoreFile {
    pub(crate) fn load(path: &Path) -> Self {
        let Ok(raw) = fs::read_to_string(path) else {
            return Self::default();
        };
        match serde_json::from_str::<Self>(&raw) {
            Ok(parsed) => parsed,
            Err(error) => {
                log::warn!(
                    "Failed to parse auth store file {}: {}. Starting with empty store.",
                    path.display(),
                    error
                );
                Self::default()
            }
        }
    }
}

pub(crate) fn resolve_auth_store_path() -> PathBuf {
    if let Some(path) = optional_env_var(AUTH_STORE_PATH_ENV) {
        return PathBuf::from(path);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".kapsl")
            .join(DEFAULT_AUTH_STORE_FILENAME);
    }
    if let Some(profile) = std::env::var_os("USERPROFILE") {
        return PathBuf::from(profile)
            .join(".kapsl")
            .join(DEFAULT_AUTH_STORE_FILENAME);
    }
    PathBuf::from(format!(".{}", DEFAULT_AUTH_STORE_FILENAME))
}
