//! Input normalization and API-key scope policy.

use super::*;

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
