//! Isolation of client session identifiers by authenticated principal.

use super::*;

const LOCAL_LOOPBACK_PRINCIPAL: &str = "local-loopback";

/// Stable, non-secret identity of the credential that authenticated a request.
///
/// Only this module derives it from a credential. Protocol adapters receive it
/// from the authorization evaluator (or, for routes that still pass the raw
/// header, through [`scope_session_id_for_authorization`]) and carry it
/// opaquely, so every northbound protocol maps the same credential to the same
/// session namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ApiPrincipal(String);

impl ApiPrincipal {
    /// Derives the principal for a presented `Authorization` value.
    ///
    /// The token is hashed so it is never copied into backend session maps.
    /// Requests without a token share the local trust domain, which matches
    /// the API's local-only policy when authentication is disabled.
    pub(crate) fn from_authorization(authorization: Option<&str>) -> Self {
        Self(
            parse_authorization_token(authorization)
                .map(|token| sha256_hex(&format!("kapsl-principal-v1\0{token}")))
                .unwrap_or_else(|| LOCAL_LOOPBACK_PRINCIPAL.to_string()),
        )
    }

    /// Converts a client-controlled session id into an internal KV-cache key
    /// scoped to this principal. Blank session ids mean "no session".
    pub(crate) fn scope_session_id(&self, session_id: Option<&str>) -> Option<String> {
        let session_id = session_id?.trim();
        if session_id.is_empty() {
            return None;
        }

        Some(format!(
            "ks1_{}",
            sha256_hex(&format!("kapsl-session-v1\0{}\0{session_id}", self.0))
        ))
    }
}

/// Scopes a client session id for adapters that hold the raw `Authorization`
/// value rather than an evaluated [`AuthorizedAccess`].
pub(crate) fn scope_session_id_for_authorization(
    session_id: Option<&str>,
    authorization: Option<&str>,
) -> Option<String> {
    ApiPrincipal::from_authorization(authorization).scope_session_id(session_id)
}
