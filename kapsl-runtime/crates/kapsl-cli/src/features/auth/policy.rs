//! Input normalization and API-key scope policy.

use super::*;

/// Apply the shared API authorization policy for any northbound protocol.
///
/// Protocol adapters extract credentials and the peer address; this function
/// owns the loopback fallback, role hierarchy, API-key scopes, and usage touch.
pub(crate) fn authorize_api_request(
    auth_state: &RwLock<ApiAuthState>,
    required_role: ApiRole,
    required_scope: ApiScope,
    authorization: Option<&str>,
    remote_ip: Option<IpAddr>,
) -> Result<AuthorizedAccess, ApiAuthorizationError> {
    let result = evaluate_api_access(
        auth_state,
        required_role,
        required_scope,
        authorization,
        remote_ip,
    );
    // Warp can evaluate more than one role group before selecting a route.
    // These are policy evaluations, not completed request audit records.
    log::debug!(target: "kapsl::authorization",
    "required_role={:?} required_scope={:?} outcome={}",
    required_role, required_scope,
    match &result {
        Ok(_) => "allowed",
        Err(ApiAuthorizationError::Unauthorized) => "unauthorized",
        Err(ApiAuthorizationError::Forbidden) => "forbidden",
        Err(ApiAuthorizationError::LocalOnly) => "local_only",
    });

    result
}

fn evaluate_api_access(
    auth_state: &RwLock<ApiAuthState>,
    required_role: ApiRole,
    required_scope: ApiScope,
    authorization: Option<&str>,
    remote_ip: Option<IpAddr>,
) -> Result<AuthorizedAccess, ApiAuthorizationError> {
    let grant_match = {
        let state = auth_state.read();
        if !state.auth_enabled() {
            return if remote_ip.is_some_and(|ip| ip.is_loopback()) {
                Ok(AuthorizedAccess {
                    grant: ApiAuthGrant {
                        role: ApiRole::Admin,
                        scopes: None,
                    },
                    mode: "local-loopback",
                })
            } else {
                Err(ApiAuthorizationError::LocalOnly)
            };
        }
        state.grant_from_authorization_header_read(authorization)
    };

    let Some(grant_match) = grant_match else {
        return Err(ApiAuthorizationError::Unauthorized);
    };
    if !grant_match.grant.allows(required_role, required_scope) {
        return Err(ApiAuthorizationError::Forbidden);
    }
    if let Some(key_index) = grant_match.matched_key_index {
        if let Some(mut state) = auth_state.try_write() {
            state.touch_key_last_used_by_index(key_index, now_unix_seconds());
        }
    }
    Ok(AuthorizedAccess {
        mode: if grant_match.matched_key_index.is_some() {
            "api-key"
        } else {
            "role-token"
        },
        grant: grant_match.grant,
    })
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
