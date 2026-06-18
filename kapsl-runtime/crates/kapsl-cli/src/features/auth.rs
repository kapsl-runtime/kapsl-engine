use super::*;

#[derive(Debug)]
pub(crate) struct ApiUnauthorized;

impl warp::reject::Reject for ApiUnauthorized {}

#[derive(Debug)]
pub(crate) struct ApiForbidden;

impl warp::reject::Reject for ApiForbidden {}

#[derive(Debug)]
pub(crate) struct ApiLocalOnly;

impl warp::reject::Reject for ApiLocalOnly {}

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
    pub(crate) fn normalize_token(value: Option<String>) -> Option<String> {
        value.and_then(|raw| {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
    }

    pub(crate) fn from_env() -> Self {
        let shared_token = optional_env_var(API_TOKEN_ENV);
        Self {
            reader_token: optional_env_var(API_READER_TOKEN_ENV).or(shared_token.clone()),
            writer_token: optional_env_var(API_WRITER_TOKEN_ENV).or(shared_token.clone()),
            admin_token: optional_env_var(API_ADMIN_TOKEN_ENV).or(shared_token),
        }
    }

    pub(crate) fn auth_enabled(&self) -> bool {
        self.reader_token.is_some() || self.writer_token.is_some() || self.admin_token.is_some()
    }

    pub(crate) fn role_for_token(&self, presented_token: &str) -> Option<ApiRole> {
        if self
            .admin_token
            .as_deref()
            .is_some_and(|token| authorization_matches_token(Some(presented_token), token))
        {
            return Some(ApiRole::Admin);
        }
        if self
            .writer_token
            .as_deref()
            .is_some_and(|token| authorization_matches_token(Some(presented_token), token))
        {
            return Some(ApiRole::Writer);
        }
        if self
            .reader_token
            .as_deref()
            .is_some_and(|token| authorization_matches_token(Some(presented_token), token))
        {
            return Some(ApiRole::Reader);
        }
        None
    }

    pub(crate) fn role_from_authorization_header(
        &self,
        authorization: Option<&str>,
    ) -> Option<ApiRole> {
        let raw_header = authorization?;
        let trimmed = raw_header.trim();
        if trimmed.is_empty() {
            return None;
        }
        if let Some((scheme, token)) = trimmed.split_once(' ') {
            if scheme.eq_ignore_ascii_case("bearer") {
                return self.role_for_token(token.trim());
            }
        }
        self.role_for_token(trimmed)
    }

    pub(crate) fn update_from_payload(
        &mut self,
        payload: ApiRoleTokenConfig,
    ) -> Result<(), String> {
        self.reader_token = Self::normalize_token(payload.reader_token);
        self.writer_token = Self::normalize_token(payload.writer_token);
        self.admin_token = Self::normalize_token(payload.admin_token);
        if self.auth_enabled() && self.admin_token.is_none() {
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

#[derive(Debug, Serialize)]
pub(crate) struct ApiAuthStatusResponse {
    pub(crate) auth_enabled: bool,
    pub(crate) legacy_auth_enabled: bool,
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
    pub(crate) legacy_auth_enabled: bool,
    pub(crate) role: ApiRole,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) scopes: Vec<String>,
    pub(crate) mode: String,
    pub(crate) access: ApiAuthLoginAccess,
}

#[derive(Debug)]
pub(crate) struct ApiAuthState {
    pub(crate) legacy_tokens: ApiRoleTokenConfig,
    pub(crate) store_path: PathBuf,
    pub(crate) store: ApiAuthStoreFile,
    pub(crate) key_hash_index: HashMap<String, usize>,
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

impl ApiAuthState {
    pub(crate) fn from_store_path(store_path: PathBuf) -> Self {
        let legacy_tokens = ApiRoleTokenConfig::from_env();
        let store = ApiAuthStoreFile::load(&store_path);
        let mut state = Self {
            legacy_tokens,
            key_hash_index: Self::build_key_hash_index(&store),
            store,
            store_path,
        };
        if state.store.users.is_empty() {
            state.seed_default_users();
            if let Err(error) = state.save_store() {
                log::warn!("Failed to persist default auth users: {}", error);
            }
        }
        state
    }

    pub(crate) fn build_key_hash_index(store: &ApiAuthStoreFile) -> HashMap<String, usize> {
        let mut index = HashMap::with_capacity(store.api_keys.len());
        for (position, key) in store.api_keys.iter().enumerate() {
            if index.insert(key.key_hash.clone(), position).is_some() {
                log::warn!(
                    "Duplicate key hash detected in auth store; latest entry will be used for lookup"
                );
            }
        }
        index
    }

    pub(crate) fn save_store(&self) -> Result<(), String> {
        if let Some(parent) = self.store_path.parent() {
            fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "failed to create auth store directory {}: {}",
                    parent.display(),
                    error
                )
            })?;
        }
        let serialized = serde_json::to_string_pretty(&self.store)
            .map_err(|error| format!("failed to serialize auth store: {}", error))?;
        let tmp_path = self.store_path.with_extension("tmp");
        fs::write(&tmp_path, serialized).map_err(|error| {
            format!(
                "failed to write auth store temp file {}: {}",
                tmp_path.display(),
                error
            )
        })?;
        if self.store_path.exists() {
            fs::remove_file(&self.store_path).map_err(|error| {
                format!(
                    "failed to replace existing auth store file {}: {}",
                    self.store_path.display(),
                    error
                )
            })?;
        }
        fs::rename(&tmp_path, &self.store_path).map_err(|error| {
            format!(
                "failed to replace auth store file {}: {}",
                self.store_path.display(),
                error
            )
        })?;
        Ok(())
    }

    pub(crate) fn seed_default_users(&mut self) {
        let now = now_unix_seconds();
        self.store.users = vec![
            ApiAuthUser {
                id: generate_random_id("usr"),
                username: "admin".to_string(),
                display_name: Some("Default Admin".to_string()),
                role: ApiRole::Admin,
                status: ApiUserStatus::Active,
                created_at: now,
                updated_at: now,
            },
            ApiAuthUser {
                id: generate_random_id("usr"),
                username: "operator".to_string(),
                display_name: Some("Runtime Operator".to_string()),
                role: ApiRole::Writer,
                status: ApiUserStatus::Active,
                created_at: now,
                updated_at: now,
            },
            ApiAuthUser {
                id: generate_random_id("usr"),
                username: "viewer".to_string(),
                display_name: Some("Read-Only Viewer".to_string()),
                role: ApiRole::Reader,
                status: ApiUserStatus::Active,
                created_at: now,
                updated_at: now,
            },
        ];
    }

    pub(crate) fn auth_enabled(&self) -> bool {
        self.legacy_tokens.auth_enabled() || self.active_key_count() > 0
    }

    pub(crate) fn active_key_count(&self) -> usize {
        let now = now_unix_seconds();
        self.store
            .api_keys
            .iter()
            .filter(|key| {
                self.user_by_id(&key.user_id)
                    .is_some_and(|user| Self::is_key_active_for_user(key, user, now))
            })
            .count()
    }

    pub(crate) fn active_admin_key_count(&self) -> usize {
        let now = now_unix_seconds();
        self.store
            .api_keys
            .iter()
            .filter(|key| {
                self.user_by_id(&key.user_id).is_some_and(|user| {
                    user.role == ApiRole::Admin && Self::is_key_active_for_user(key, user, now)
                })
            })
            .count()
    }

    pub(crate) fn active_key_count_for_user(&self, user_id: &str) -> usize {
        let now = now_unix_seconds();
        self.store
            .api_keys
            .iter()
            .filter(|key| {
                if key.user_id != user_id {
                    return false;
                }
                self.user_by_id(&key.user_id)
                    .is_some_and(|user| Self::is_key_active_for_user(key, user, now))
            })
            .count()
    }

    pub(crate) fn is_key_active_for_user(key: &ApiAuthKey, user: &ApiAuthUser, now: u64) -> bool {
        if !user.status.is_active() || key.revoked_at.is_some() {
            return false;
        }
        key.expires_at.is_none_or(|expiry| expiry > now)
    }

    pub(crate) fn user_by_id(&self, user_id: &str) -> Option<&ApiAuthUser> {
        self.store.users.iter().find(|user| user.id == user_id)
    }

    pub(crate) fn grant_from_authorization_header_read(
        &self,
        authorization: Option<&str>,
    ) -> Option<ApiAuthGrantMatch> {
        let presented = parse_authorization_token(authorization)?;
        if let Some((role, scopes, key_index)) = self.grant_for_api_key_token_read(presented) {
            return Some(ApiAuthGrantMatch {
                grant: ApiAuthGrant {
                    role,
                    scopes: Some(scopes),
                },
                matched_key_index: Some(key_index),
            });
        }
        self.legacy_tokens
            .role_from_authorization_header(authorization)
            .map(|role| ApiAuthGrantMatch {
                grant: ApiAuthGrant { role, scopes: None },
                matched_key_index: None,
            })
    }

    #[cfg(test)]
    pub(crate) fn role_from_authorization_header(
        &mut self,
        authorization: Option<&str>,
    ) -> Option<ApiRole> {
        self.grant_from_authorization_header_read(authorization)
            .map(|matched| matched.grant.role)
    }

    pub(crate) fn grant_for_api_key_token_read(
        &self,
        presented_token: &str,
    ) -> Option<(ApiRole, Vec<String>, usize)> {
        let token_hash = sha256_hex(presented_token);
        let key_index = self.key_hash_index.get(&token_hash).copied()?;
        let key = self.store.api_keys.get(key_index)?;
        if !constant_time_eq(&key.key_hash, &token_hash) {
            return None;
        }

        let now = now_unix_seconds();
        let user = self.user_by_id(&key.user_id)?;
        let role = user.role;
        let scopes = key.scopes.clone();
        let is_active = Self::is_key_active_for_user(key, user, now);
        if !is_active {
            return None;
        }
        Some((role, scopes, key_index))
    }

    pub(crate) fn touch_key_last_used_by_index(&mut self, key_index: usize, now: u64) {
        if let Some(key) = self.store.api_keys.get_mut(key_index) {
            if key.last_used_at != Some(now) {
                key.last_used_at = Some(now);
            }
        }
    }

    pub(crate) fn legacy_token_config(&self) -> ApiRoleTokenConfig {
        self.legacy_tokens.clone()
    }

    pub(crate) fn update_legacy_token_config(
        &mut self,
        payload: ApiRoleTokenConfig,
    ) -> Result<ApiRoleTokenConfig, String> {
        self.legacy_tokens.update_from_payload(payload)?;
        Ok(self.legacy_tokens.clone())
    }

    pub(crate) fn status_response(&self) -> ApiAuthStatusResponse {
        ApiAuthStatusResponse {
            auth_enabled: self.auth_enabled(),
            legacy_auth_enabled: self.legacy_tokens.auth_enabled(),
            store_path: self.store_path.to_string_lossy().to_string(),
            user_count: self.store.users.len(),
            key_count: self.store.api_keys.len(),
            active_key_count: self.active_key_count(),
            active_admin_key_count: self.active_admin_key_count(),
        }
    }

    pub(crate) fn role_summaries(&self) -> Vec<ApiRoleSummary> {
        [ApiRole::Admin, ApiRole::Writer, ApiRole::Reader]
            .iter()
            .copied()
            .map(|role| {
                let user_count = self
                    .store
                    .users
                    .iter()
                    .filter(|user| user.role == role)
                    .count();
                let now = now_unix_seconds();
                let active_key_count = self
                    .store
                    .api_keys
                    .iter()
                    .filter(|key| {
                        self.user_by_id(&key.user_id).is_some_and(|user| {
                            user.role == role && Self::is_key_active_for_user(key, user, now)
                        })
                    })
                    .count();
                ApiRoleSummary {
                    role,
                    description: role_description(role).to_string(),
                    user_count,
                    active_key_count,
                }
            })
            .collect()
    }

    pub(crate) fn list_users(&self) -> Vec<ApiAuthUserView> {
        self.store
            .users
            .iter()
            .map(|user| self.user_view(user))
            .collect()
    }

    pub(crate) fn user_view(&self, user: &ApiAuthUser) -> ApiAuthUserView {
        let key_count = self
            .store
            .api_keys
            .iter()
            .filter(|key| key.user_id == user.id)
            .count();
        let active_key_count = self.active_key_count_for_user(&user.id);
        ApiAuthUserView {
            id: user.id.clone(),
            username: user.username.clone(),
            display_name: user.display_name.clone(),
            role: user.role,
            status: user.status,
            created_at: user.created_at,
            updated_at: user.updated_at,
            key_count,
            active_key_count,
        }
    }

    pub(crate) fn list_keys(&self, user_id: Option<&str>) -> Vec<ApiAuthKeyView> {
        let now = now_unix_seconds();
        let mut keys = self
            .store
            .api_keys
            .iter()
            .filter(|key| user_id.is_none_or(|expected| expected == key.user_id))
            .filter_map(|key| {
                let user = self.user_by_id(&key.user_id)?;
                Some(self.key_view(key, user, now))
            })
            .collect::<Vec<_>>();
        keys.sort_by(|left, right| right.created_at.cmp(&left.created_at));
        keys
    }

    pub(crate) fn key_view(
        &self,
        key: &ApiAuthKey,
        user: &ApiAuthUser,
        now: u64,
    ) -> ApiAuthKeyView {
        ApiAuthKeyView {
            id: key.id.clone(),
            user_id: key.user_id.clone(),
            username: user.username.clone(),
            user_role: user.role,
            name: key.name.clone(),
            key_prefix: key.key_prefix.clone(),
            scopes: key.scopes.clone(),
            created_at: key.created_at,
            created_by: key.created_by.clone(),
            last_used_at: key.last_used_at,
            expires_at: key.expires_at,
            revoked_at: key.revoked_at,
            active: Self::is_key_active_for_user(key, user, now),
        }
    }

    pub(crate) fn create_user(
        &mut self,
        request: CreateAuthUserRequest,
    ) -> Result<ApiAuthUserView, String> {
        let username = normalize_username(&request.username)?;
        if self
            .store
            .users
            .iter()
            .any(|user| user.username.eq_ignore_ascii_case(&username))
        {
            return Err(format!("user `{}` already exists", username));
        }

        let now = now_unix_seconds();
        let user = ApiAuthUser {
            id: generate_random_id("usr"),
            username,
            display_name: normalize_optional_text(request.display_name),
            role: request.role,
            status: request.status.unwrap_or(ApiUserStatus::Active),
            created_at: now,
            updated_at: now,
        };
        let user_id = user.id.clone();
        self.store.users.push(user);
        self.save_store()?;
        let created = self
            .store
            .users
            .iter()
            .find(|user| user.id == user_id)
            .ok_or_else(|| "failed to load created user".to_string())?;
        Ok(self.user_view(created))
    }

    pub(crate) fn update_user(
        &mut self,
        user_id: &str,
        request: UpdateAuthUserRequest,
    ) -> Result<ApiAuthUserView, String> {
        let user_index = self
            .store
            .users
            .iter()
            .position(|user| user.id == user_id)
            .ok_or_else(|| format!("user `{}` not found", user_id))?;

        let mut updated_user = self.store.users[user_index].clone();
        if let Some(display_name) = request.display_name {
            updated_user.display_name = normalize_optional_text(display_name);
        }
        if let Some(new_role) = request.role {
            if updated_user.role == ApiRole::Admin
                && new_role != ApiRole::Admin
                && self.active_key_count_for_user(&updated_user.id) > 0
                && self.active_admin_key_count() <= self.active_key_count_for_user(&updated_user.id)
                && self.active_key_count() > self.active_key_count_for_user(&updated_user.id)
            {
                return Err(
                    "cannot remove admin role from the last admin with active API keys".to_string(),
                );
            }
            updated_user.role = new_role;
        }
        if let Some(new_status) = request.status {
            updated_user.status = new_status;
        }
        updated_user.updated_at = now_unix_seconds();
        self.store.users[user_index] = updated_user.clone();
        self.save_store()?;
        Ok(self.user_view(&updated_user))
    }

    pub(crate) fn create_api_key(
        &mut self,
        user_id: &str,
        request: CreateApiKeyRequest,
    ) -> Result<CreateApiKeyResponse, String> {
        let user = self
            .user_by_id(user_id)
            .cloned()
            .ok_or_else(|| format!("user `{}` not found", user_id))?;
        if !user.status.is_active() {
            return Err("cannot create API key for a suspended user".to_string());
        }
        if self.active_key_count() == 0 && user.role != ApiRole::Admin {
            return Err("first API key must belong to an admin user".to_string());
        }

        let name = normalize_required_text(&request.name, "name")?;
        let expires_at = match request.expires_in_days {
            Some(0) => return Err("expires_in_days must be greater than 0".to_string()),
            Some(days) => Some(now_unix_seconds() + (days as u64 * 86_400)),
            None => None,
        };
        let scopes = normalize_scopes(request.scopes);
        let raw_key = generate_api_key();
        let key_hash = sha256_hex(&raw_key);
        let key_prefix: String = raw_key.chars().take(12).collect();
        if self.key_hash_index.contains_key(&key_hash) {
            return Err("generated API key collided, retry".to_string());
        }

        let now = now_unix_seconds();
        let key_hash_for_index = key_hash.clone();
        let key = ApiAuthKey {
            id: generate_random_id("key"),
            user_id: user.id.clone(),
            name,
            key_prefix,
            key_hash,
            scopes,
            created_at: now,
            created_by: None,
            last_used_at: None,
            expires_at,
            revoked_at: None,
        };
        let key_id = key.id.clone();
        let key_index = self.store.api_keys.len();
        self.store.api_keys.push(key);
        self.key_hash_index.insert(key_hash_for_index, key_index);
        self.save_store()?;
        let created_key = self
            .store
            .api_keys
            .iter()
            .find(|existing| existing.id == key_id)
            .ok_or_else(|| "failed to load created API key".to_string())?;
        let view = self.key_view(created_key, &user, now);
        Ok(CreateApiKeyResponse {
            api_key: view,
            raw_key,
        })
    }

    pub(crate) fn revoke_api_key(&mut self, key_id: &str) -> Result<ApiAuthKeyView, String> {
        let key_index = self
            .store
            .api_keys
            .iter()
            .position(|key| key.id == key_id)
            .ok_or_else(|| format!("api key `{}` not found", key_id))?;
        let now = now_unix_seconds();
        let key = self.store.api_keys[key_index].clone();
        let user = self
            .user_by_id(&key.user_id)
            .cloned()
            .ok_or_else(|| format!("user `{}` not found for key", key.user_id))?;
        let key_active = Self::is_key_active_for_user(&key, &user, now);
        if key_active
            && user.role == ApiRole::Admin
            && self.active_admin_key_count() <= 1
            && self.active_key_count() > 1
        {
            return Err(
                "cannot revoke the last active admin key while other keys remain active"
                    .to_string(),
            );
        }

        if self.store.api_keys[key_index].revoked_at.is_none() {
            self.store.api_keys[key_index].revoked_at = Some(now);
            self.save_store()?;
        }
        let updated = self.store.api_keys[key_index].clone();
        Ok(self.key_view(&updated, &user, now))
    }
}

pub(crate) fn role_description(role: ApiRole) -> &'static str {
    match role {
        ApiRole::Admin => "Full control of runtime and access management",
        ApiRole::Writer => "Can modify runtime state and extensions",
        ApiRole::Reader => "Read-only runtime access",
    }
}

pub(crate) fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
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

pub(crate) fn parse_authorization_token(header_value: Option<&str>) -> Option<&str> {
    let raw_header = header_value?;
    let trimmed = raw_header.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some((scheme, token)) = trimmed.split_once(' ') {
        if scheme.eq_ignore_ascii_case("bearer") {
            let parsed = token.trim();
            if parsed.is_empty() {
                return None;
            }
            return Some(parsed);
        }
    }
    Some(trimmed)
}

pub(crate) fn normalize_required_text(value: &str, field: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(format!("{} is required", field))
    } else {
        Ok(trimmed.to_string())
    }
}

pub(crate) fn normalize_optional_text(value: Option<String>) -> Option<String> {
    value.and_then(|raw| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

pub(crate) fn normalize_username(username: &str) -> Result<String, String> {
    let normalized = normalize_required_text(username, "username")?;
    Ok(normalized.to_ascii_lowercase())
}

pub(crate) fn normalize_scopes(scopes: Option<Vec<String>>) -> Vec<String> {
    let mut seen = HashSet::new();
    scopes
        .unwrap_or_default()
        .into_iter()
        .filter_map(|scope| {
            let trimmed = scope.trim();
            if trimmed.is_empty() {
                None
            } else {
                let normalized = trimmed.to_string();
                if seen.insert(normalized.clone()) {
                    Some(normalized)
                } else {
                    None
                }
            }
        })
        .collect()
}

pub(crate) fn scope_token_allows(scope: &str, required: ApiScope) -> bool {
    let normalized = scope.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return false;
    }
    if normalized == "*" || normalized == "*:*" {
        return true;
    }

    match required {
        ApiScope::Read => matches!(
            normalized.as_str(),
            "api:read" | "read" | "api:write" | "write" | "api:admin" | "admin" | "api:*"
        ),
        ApiScope::Write => {
            matches!(
                normalized.as_str(),
                "api:write" | "write" | "api:admin" | "admin" | "api:*"
            )
        }
        ApiScope::Admin => matches!(normalized.as_str(), "api:admin" | "admin" | "api:*"),
    }
}

pub(crate) fn key_scopes_allow(scopes: &[String], required: ApiScope) -> bool {
    // Backward compatibility: empty scopes behave like unrestricted role-based keys.
    scopes.is_empty()
        || scopes
            .iter()
            .any(|scope| scope_token_allows(scope, required))
}

pub(crate) fn is_loopback_remote(remote: Option<std::net::SocketAddr>) -> bool {
    remote.is_some_and(|addr| addr.ip().is_loopback())
}

pub(crate) fn generate_random_id(prefix: &str) -> String {
    let mut bytes = [0u8; 8];
    OsRng.fill_bytes(&mut bytes);
    let mut suffix = String::with_capacity(16);
    for byte in bytes {
        suffix.push_str(&format!("{:02x}", byte));
    }
    format!("{}_{}", prefix, suffix)
}

pub(crate) fn generate_api_key() -> String {
    let mut bytes = [0u8; 24];
    OsRng.fill_bytes(&mut bytes);
    let secret = BASE64_URL_SAFE_NO_PAD.encode(bytes);
    format!("kpsl_{}", secret)
}

pub(crate) fn sha256_hex(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push_str(&format!("{:02x}", byte));
    }
    output
}

pub(crate) fn constant_time_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut diff = 0u8;
    for (lhs, rhs) in left.as_bytes().iter().zip(right.as_bytes()) {
        diff |= lhs ^ rhs;
    }
    diff == 0
}

pub(crate) fn authorization_matches_token(
    header_value: Option<&str>,
    expected_token: &str,
) -> bool {
    let Some(raw_header) = header_value else {
        return false;
    };
    let trimmed = raw_header.trim();
    if trimmed.is_empty() {
        return false;
    }
    if let Some((scheme, token)) = trimmed.split_once(' ') {
        if scheme.eq_ignore_ascii_case("bearer") {
            return constant_time_eq(token.trim(), expected_token);
        }
    }
    constant_time_eq(trimmed, expected_token)
}

pub(crate) fn api_auth_filter(
    required_role: ApiRole,
    required_scope: ApiScope,
    auth_state: Arc<RwLock<ApiAuthState>>,
) -> impl Filter<Extract = (), Error = warp::Rejection> + Clone {
    warp::header::optional::<String>("authorization")
        .and(warp::addr::remote())
        .and_then(
            move |authorization: Option<String>, remote: Option<std::net::SocketAddr>| {
                let auth_state = auth_state.clone();
                async move {
                    let grant_match = {
                        let state = auth_state.read();
                        if !state.auth_enabled() {
                            if is_loopback_remote(remote) {
                                return Ok::<(), warp::Rejection>(());
                            }
                            return Err(warp::reject::custom(ApiLocalOnly));
                        }

                        state.grant_from_authorization_header_read(authorization.as_deref())
                    };

                    let Some(grant_match) = grant_match else {
                        return Err(warp::reject::custom(ApiUnauthorized));
                    };

                    if !grant_match.grant.role.allows(required_role) {
                        return Err(warp::reject::custom(ApiForbidden));
                    }

                    if let Some(scopes) = grant_match.grant.scopes.as_ref() {
                        if !key_scopes_allow(scopes, required_scope) {
                            return Err(warp::reject::custom(ApiForbidden));
                        }
                    }

                    if let Some(key_index) = grant_match.matched_key_index {
                        if let Some(mut state) = auth_state.try_write() {
                            state.touch_key_last_used_by_index(key_index, now_unix_seconds());
                        }
                    }

                    Ok(())
                }
            },
        )
        .untuple_one()
}

pub(crate) async fn map_api_auth_rejection(
    rejection: warp::Rejection,
) -> Result<(warp::reply::Response,), warp::Rejection> {
    if rejection.find::<ApiForbidden>().is_some() {
        return Ok((warp::reply::with_status(
            warp::reply::json(&json!({
                "error": "Forbidden"
            })),
            warp::http::StatusCode::FORBIDDEN,
        )
        .into_response(),));
    }
    if rejection.find::<ApiUnauthorized>().is_some() {
        return Ok((warp::reply::with_status(
            warp::reply::json(&json!({
                "error": "Unauthorized"
            })),
            warp::http::StatusCode::UNAUTHORIZED,
        )
        .into_response(),));
    }
    if rejection.find::<ApiLocalOnly>().is_some() {
        return Ok((warp::reply::with_status(
            warp::reply::json(&json!({
                "error": "Unauthorized",
                "detail": "Authentication is disabled; this endpoint is restricted to loopback clients only."
            })),
            warp::http::StatusCode::FORBIDDEN,
        )
        .into_response(),));
    }
    Err(rejection)
}
