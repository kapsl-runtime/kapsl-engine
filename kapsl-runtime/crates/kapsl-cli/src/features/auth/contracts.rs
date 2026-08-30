//! Request and response contracts exposed by the authentication API.

use super::*;

#[derive(Debug, Serialize)]
pub(crate) struct ApiAuthStatusResponse {
    pub(crate) auth_enabled: bool,
    pub(crate) role_token_auth_enabled: bool,
    pub(crate) store_path: String,
    pub(crate) user_count: usize,
    pub(crate) key_count: usize,
    pub(crate) active_key_count: usize,
    pub(crate) active_admin_key_count: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct ApiRoleSummary {
    pub(crate) role: ApiRole,
    pub(crate) description: String,
    pub(crate) user_count: usize,
    pub(crate) active_key_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ApiAuthUserView {
    pub(crate) id: String,
    pub(crate) username: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) display_name: Option<String>,
    pub(crate) role: ApiRole,
    pub(crate) status: ApiUserStatus,
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
    pub(crate) key_count: usize,
    pub(crate) active_key_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ApiAuthKeyView {
    pub(crate) id: String,
    pub(crate) user_id: String,
    pub(crate) username: String,
    pub(crate) user_role: ApiRole,
    pub(crate) name: String,
    pub(crate) key_prefix: String,
    pub(crate) scopes: Vec<String>,
    pub(crate) created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) created_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_used_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) expires_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) revoked_at: Option<u64>,
    pub(crate) active: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CreateAuthUserRequest {
    pub(crate) username: String,
    #[serde(default)]
    pub(crate) display_name: Option<String>,
    pub(crate) role: ApiRole,
    #[serde(default)]
    pub(crate) status: Option<ApiUserStatus>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct UpdateAuthUserRequest {
    #[serde(default)]
    pub(crate) display_name: Option<Option<String>>,
    #[serde(default)]
    pub(crate) role: Option<ApiRole>,
    #[serde(default)]
    pub(crate) status: Option<ApiUserStatus>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CreateApiKeyRequest {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) scopes: Option<Vec<String>>,
    #[serde(default)]
    pub(crate) expires_in_days: Option<u32>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CreateApiKeyResponse {
    pub(crate) api_key: ApiAuthKeyView,
    pub(crate) raw_key: String,
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct ApiAuthLoginRequest {
    #[serde(default)]
    pub(crate) token: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ApiAuthLoginAccess {
    pub(crate) read: bool,
    pub(crate) write: bool,
    pub(crate) admin: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct ApiAuthLoginResponse {
    pub(crate) authenticated: bool,
    pub(crate) auth_enabled: bool,
    pub(crate) role_token_auth_enabled: bool,
    pub(crate) role: ApiRole,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) scopes: Vec<String>,
    pub(crate) mode: String,
    pub(crate) access: ApiAuthLoginAccess,
}
