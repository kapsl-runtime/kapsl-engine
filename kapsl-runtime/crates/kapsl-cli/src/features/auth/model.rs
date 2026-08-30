//! Core authentication and authorization types.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ApiRole {
    Reader,
    Writer,
    Admin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApiScope {
    Read,
    Write,
    Admin,
}

impl ApiRole {
    pub(crate) fn allows(self, required: ApiRole) -> bool {
        use ApiRole::{Admin, Reader, Writer};
        matches!(
            (self, required),
            (Admin, _) | (Writer, Reader) | (Writer, Writer) | (Reader, Reader)
        )
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ApiRoleTokenConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reader_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) writer_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) admin_token: Option<String>,
}

impl ApiRoleTokenConfig {
    pub(crate) fn from_env() -> Self {
        Self {
            reader_token: optional_env_var(API_READER_TOKEN_ENV),
            writer_token: optional_env_var(API_WRITER_TOKEN_ENV),
            admin_token: optional_env_var(API_ADMIN_TOKEN_ENV),
        }
    }

    pub(crate) fn has_configured_tokens(&self) -> bool {
        self.reader_token.is_some() || self.writer_token.is_some() || self.admin_token.is_some()
    }

    pub(super) fn role_for_token(&self, presented_token: &str) -> Option<ApiRole> {
        [
            (ApiRole::Admin, self.admin_token.as_deref()),
            (ApiRole::Writer, self.writer_token.as_deref()),
            (ApiRole::Reader, self.reader_token.as_deref()),
        ]
        .into_iter()
        .find_map(|(role, expected_token)| {
            expected_token
                .is_some_and(|expected| constant_time_eq(presented_token, expected))
                .then_some(role)
        })
    }

    pub(crate) fn update_from_payload(
        &mut self,
        payload: ApiRoleTokenConfig,
    ) -> Result<(), String> {
        self.reader_token = normalize_optional_text(payload.reader_token);
        self.writer_token = normalize_optional_text(payload.writer_token);
        self.admin_token = normalize_optional_text(payload.admin_token);
        if self.has_configured_tokens() && self.admin_token.is_none() {
            return Err("admin_token is required when role auth is enabled".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ApiUserStatus {
    #[default]
    Active,
    Suspended,
}

impl ApiUserStatus {
    pub(crate) fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ApiAuthUser {
    pub(crate) id: String,
    pub(crate) username: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) display_name: Option<String>,
    pub(crate) role: ApiRole,
    #[serde(default)]
    pub(crate) status: ApiUserStatus,
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ApiAuthKey {
    pub(crate) id: String,
    pub(crate) user_id: String,
    pub(crate) name: String,
    pub(crate) key_prefix: String,
    pub(crate) key_hash: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
}

#[derive(Debug, Clone)]
pub(crate) struct ApiAuthGrant {
    pub(crate) role: ApiRole,
    pub(crate) scopes: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub(crate) struct ApiAuthGrantMatch {
    pub(crate) grant: ApiAuthGrant,
    pub(crate) matched_key_index: Option<usize>,
}

pub(crate) fn role_description(role: ApiRole) -> &'static str {
    match role {
        ApiRole::Admin => "Full control of runtime and access management",
        ApiRole::Writer => "Can modify runtime state and extensions",
        ApiRole::Reader => "Read-only runtime access",
    }
}
