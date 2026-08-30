//! Authentication state and user/API-key lifecycle operations.

use super::*;

#[derive(Debug)]
pub(crate) struct ApiAuthState {
    pub(crate) role_tokens: ApiRoleTokenConfig,
    pub(crate) store_path: PathBuf,
    pub(crate) store: ApiAuthStoreFile,
    pub(crate) key_hash_index: HashMap<String, usize>,
}

impl ApiAuthState {
    pub(crate) fn from_store_path(store_path: PathBuf) -> Self {
        let role_tokens = ApiRoleTokenConfig::from_env();
        let store = ApiAuthStoreFile::load(&store_path);
        let mut state = Self {
            role_tokens,
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
        self.role_tokens.has_configured_tokens() || self.active_key_count() > 0
    }

    pub(crate) fn active_key_count(&self) -> usize {
        self.active_key_count_matching(|_, _| true)
    }

    pub(crate) fn active_admin_key_count(&self) -> usize {
        self.active_key_count_matching(|_, user| user.role == ApiRole::Admin)
    }

    pub(crate) fn active_key_count_for_user(&self, user_id: &str) -> usize {
        self.active_key_count_matching(|key, _| key.user_id == user_id)
    }

    fn active_key_count_matching(
        &self,
        predicate: impl Fn(&ApiAuthKey, &ApiAuthUser) -> bool,
    ) -> usize {
        self.active_keys_at(now_unix_seconds())
            .filter(|(key, user)| predicate(key, user))
            .count()
    }

    fn active_keys_at(&self, now: u64) -> impl Iterator<Item = (&ApiAuthKey, &ApiAuthUser)> {
        self.store.api_keys.iter().filter_map(move |key| {
            let user = self.user_by_id(&key.user_id)?;
            Self::is_key_active_for_user(key, user, now).then_some((key, user))
        })
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
        self.role_tokens
            .role_for_token(presented)
            .map(|role| ApiAuthGrantMatch {
                grant: ApiAuthGrant { role, scopes: None },
                matched_key_index: None,
            })
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

    pub(crate) fn role_token_config(&self) -> ApiRoleTokenConfig {
        self.role_tokens.clone()
    }

    pub(crate) fn update_role_token_config(
        &mut self,
        payload: ApiRoleTokenConfig,
    ) -> Result<ApiRoleTokenConfig, String> {
        self.role_tokens.update_from_payload(payload)?;
        Ok(self.role_tokens.clone())
    }

    pub(crate) fn status_response(&self) -> ApiAuthStatusResponse {
        ApiAuthStatusResponse {
            auth_enabled: self.auth_enabled(),
            role_token_auth_enabled: self.role_tokens.has_configured_tokens(),
            store_path: self.store_path.to_string_lossy().to_string(),
            user_count: self.store.users.len(),
            key_count: self.store.api_keys.len(),
            active_key_count: self.active_key_count(),
            active_admin_key_count: self.active_admin_key_count(),
        }
    }

    pub(crate) fn role_summaries(&self) -> Vec<ApiRoleSummary> {
        let now = now_unix_seconds();
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
                let active_key_count = self
                    .active_keys_at(now)
                    .filter(|(_, user)| user.role == role)
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
            let active_user_key_count = self.active_key_count_for_user(&updated_user.id);
            if updated_user.role == ApiRole::Admin
                && new_role != ApiRole::Admin
                && active_user_key_count > 0
                && self.active_admin_key_count() <= active_user_key_count
                && self.active_key_count() > active_user_key_count
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

pub(crate) fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
