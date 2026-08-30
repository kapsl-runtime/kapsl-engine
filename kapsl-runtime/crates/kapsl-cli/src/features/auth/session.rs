//! Isolation of client session identifiers by authenticated principal.

use super::*;

/// Converts a client-controlled session id into an internal KV-cache key scoped
/// to the credential that authenticated the request.
///
/// The credential and client session id are hashed so neither secret is copied
/// into backend session maps. Authentication-disabled loopback traffic shares a
/// local trust domain, which matches the API's existing local-only policy.
pub(crate) fn scope_session_id_for_authorization(
    session_id: Option<&str>,
    authorization: Option<&str>,
) -> Option<String> {
    let session_id = session_id?.trim();
    if session_id.is_empty() {
        return None;
    }

    let principal = parse_authorization_token(authorization)
        .map(|token| sha256_hex(&format!("kapsl-principal-v1\0{token}")))
        .unwrap_or_else(|| "local-loopback".to_string());

    Some(format!(
        "ks1_{}",
        sha256_hex(&format!("kapsl-session-v1\0{principal}\0{session_id}"))
    ))
}
