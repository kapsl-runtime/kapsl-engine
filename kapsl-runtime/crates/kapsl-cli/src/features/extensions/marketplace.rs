use crate::app::{EXTENSION_MARKETPLACE_URL, EXTENSION_MARKETPLACE_URL_ENV};
use crate::features::http_client::{format_remote_http_error, native_tls_http_agent};
use crate::runtime::optional_env_var;
use sha2::{Digest, Sha256};

/// Set to `1` to install marketplace archives whose catalog entry carries no
/// SHA-256 pin. A pin that is present is always enforced.
pub(super) const EXTENSION_ALLOW_UNPINNED_ENV: &str = "KAPSL_EXTENSION_ALLOW_UNPINNED";

struct ExtensionMarketplaceClient {
    base_url: String,
    agent: ureq::Agent,
}

impl ExtensionMarketplaceClient {
    fn from_config(custom_url: Option<&str>) -> Self {
        let base_url = custom_url
            .map(str::trim)
            .filter(|url| !url.is_empty())
            .map(str::to_owned)
            .or_else(|| optional_env_var(EXTENSION_MARKETPLACE_URL_ENV))
            .unwrap_or_else(|| EXTENSION_MARKETPLACE_URL.to_string())
            .trim_end_matches('/')
            .to_string();

        Self {
            base_url,
            agent: native_tls_http_agent(),
        }
    }

    fn fetch_catalog(&self, query: Option<&str>) -> Result<serde_json::Value, String> {
        let mut request = self.agent.get(&self.base_url);
        if let Some(query) = query.map(str::trim).filter(|query| !query.is_empty()) {
            request = request.query("q", query);
        }

        let mut response = request.call().map_err(|error| {
            format!(
                "Failed to query extension marketplace {}: {}",
                self.base_url,
                format_remote_http_error(error)
            )
        })?;
        let body = response.body_mut().read_to_string().map_err(|error| {
            format!(
                "Failed to read extension marketplace response from {}: {}",
                self.base_url, error
            )
        })?;

        serde_json::from_str(&body).map_err(|error| {
            format!(
                "Failed to parse extension marketplace response as JSON from {}: {}",
                self.base_url, error
            )
        })
    }

    fn download_archive(&self, extension_id: &str) -> Result<Vec<u8>, String> {
        let download_url = format!("{}/{extension_id}/download", self.base_url);
        let mut response = self.agent.get(&download_url).call().map_err(|error| {
            format!(
                "Failed to download extension `{}` from marketplace {}: {}",
                extension_id,
                self.base_url,
                format_remote_http_error(error)
            )
        })?;

        response.body_mut().read_to_vec().map_err(|error| {
            format!(
                "Failed to read downloaded extension `{}` archive from {}: {}",
                extension_id, download_url, error
            )
        })
    }
}

pub(crate) fn fetch_extension_marketplace(
    query: Option<&str>,
    marketplace_url: Option<&str>,
) -> Result<serde_json::Value, String> {
    ExtensionMarketplaceClient::from_config(marketplace_url).fetch_catalog(query)
}

/// Downloads an extension archive and checks it against the SHA-256 the
/// marketplace catalog pins for that extension. Returns the verified bytes.
pub(super) fn download_verified_extension_archive(
    extension_id: &str,
    marketplace_url: Option<&str>,
) -> Result<Vec<u8>, String> {
    download_verified_archive_with(
        &ExtensionMarketplaceClient::from_config(marketplace_url),
        extension_id,
        allow_unpinned_from_env(),
    )
}

fn download_verified_archive_with(
    client: &ExtensionMarketplaceClient,
    extension_id: &str,
    allow_unpinned: bool,
) -> Result<Vec<u8>, String> {
    let catalog = client.fetch_catalog(Some(extension_id))?;
    let pinned = pinned_archive_sha256(&catalog, extension_id)?;
    let archive = client.download_archive(extension_id)?;
    verify_archive_sha256(extension_id, &archive, pinned.as_deref(), allow_unpinned)?;
    Ok(archive)
}

fn allow_unpinned_from_env() -> bool {
    optional_env_var(EXTENSION_ALLOW_UNPINNED_ENV).is_some_and(|value| value.trim() == "1")
}

fn catalog_entries(catalog: &serde_json::Value) -> Vec<&serde_json::Value> {
    if let Some(items) = catalog.as_array() {
        return items.iter().collect();
    }
    for key in ["extensions", "items", "results"] {
        if let Some(items) = catalog.get(key).and_then(serde_json::Value::as_array) {
            return items.iter().collect();
        }
    }
    Vec::new()
}

fn entry_extension_id(entry: &serde_json::Value) -> Option<&str> {
    entry
        .pointer("/manifest/id")
        .or_else(|| entry.get("id"))
        .and_then(serde_json::Value::as_str)
}

/// Normalizes a pin such as `sha256:ABC...` to 64 lowercase hex characters.
fn normalize_sha256(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let hex = trimmed.strip_prefix("sha256:").unwrap_or(trimmed);
    (hex.len() == 64 && hex.chars().all(|ch| ch.is_ascii_hexdigit()))
        .then(|| hex.to_ascii_lowercase())
}

/// Finds the SHA-256 the catalog pins for `extension_id`.
///
/// Returns `Ok(None)` when the extension is listed without a pin, and an error
/// when the extension is not listed or the pin is malformed.
fn pinned_archive_sha256(
    catalog: &serde_json::Value,
    extension_id: &str,
) -> Result<Option<String>, String> {
    let entry = catalog_entries(catalog)
        .into_iter()
        .find(|entry| entry_extension_id(entry) == Some(extension_id))
        .ok_or_else(|| {
            format!(
                "Extension `{extension_id}` was not found in the marketplace catalog, so its archive cannot be verified"
            )
        })?;
    let raw = entry
        .get("archive_sha256")
        .or_else(|| entry.get("sha256"))
        .and_then(serde_json::Value::as_str);
    match raw {
        None => Ok(None),
        Some(raw) => normalize_sha256(raw).map(Some).ok_or_else(|| {
            format!(
                "Marketplace catalog entry for `{extension_id}` has a malformed SHA-256 pin `{raw}`; expected 64 hex characters"
            )
        }),
    }
}

/// Refuses an archive whose SHA-256 differs from the pinned value. A missing
/// pin is refused too unless `allow_unpinned` is set.
fn verify_archive_sha256(
    extension_id: &str,
    archive: &[u8],
    pinned: Option<&str>,
    allow_unpinned: bool,
) -> Result<(), String> {
    let Some(pinned) = pinned else {
        if allow_unpinned {
            eprintln!(
                "warning: installing extension `{extension_id}` without a marketplace SHA-256 pin ({EXTENSION_ALLOW_UNPINNED_ENV}=1)"
            );
            return Ok(());
        }
        return Err(format!(
            "Marketplace does not pin a SHA-256 for extension `{extension_id}`; refusing to install an unverified archive (set {EXTENSION_ALLOW_UNPINNED_ENV}=1 to override)"
        ));
    };
    let actual = format!("{:x}", Sha256::digest(archive));
    if actual != pinned {
        return Err(format!(
            "Extension `{extension_id}` archive failed SHA-256 verification: marketplace pinned {pinned}, downloaded archive is {actual}; refusing to install"
        ));
    }
    Ok(())
}

#[cfg(test)]
/// Serves `catalog` for the marketplace query and `archive` for `/<id>/download`.
/// Returns the base URL; the server thread handles `requests` connections.
pub(super) fn serve_for_tests(catalog: String, archive: Vec<u8>, requests: usize) -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/extensions", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for _ in 0..requests {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buffer = [0u8; 4096];
            let read = stream.read(&mut buffer).unwrap_or(0);
            let head = String::from_utf8_lossy(&buffer[..read]).to_string();
            let (content_type, body) = if head.starts_with("GET /extensions/") {
                ("application/gzip", archive.clone())
            } else {
                ("application/json", catalog.clone().into_bytes())
            };
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&body);
        }
    });
    base
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const GOOD: &[u8] = b"genuine extension archive";

    fn sha(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[test]
    fn pin_is_read_from_catalog_entry_and_normalized() {
        let upper = sha(GOOD).to_ascii_uppercase();
        let catalog = json!([
            {"manifest": {"id": "other"}, "archive_sha256": sha(b"x")},
            {"manifest": {"id": "connector.s3"}, "archive_sha256": format!("sha256:{upper}")},
        ]);
        assert_eq!(
            pinned_archive_sha256(&catalog, "connector.s3").unwrap(),
            Some(sha(GOOD))
        );
        let wrapped = json!({"extensions": [{"id": "a", "sha256": sha(GOOD)}]});
        assert_eq!(
            pinned_archive_sha256(&wrapped, "a").unwrap(),
            Some(sha(GOOD))
        );
    }

    #[test]
    fn missing_entry_or_malformed_pin_is_an_error_and_missing_pin_is_none() {
        let catalog = json!([
            {"manifest": {"id": "nopin"}},
            {"manifest": {"id": "bad"}, "archive_sha256": "abc123"},
        ]);
        assert_eq!(pinned_archive_sha256(&catalog, "nopin").unwrap(), None);
        assert!(pinned_archive_sha256(&catalog, "bad")
            .unwrap_err()
            .contains("malformed"));
        assert!(pinned_archive_sha256(&catalog, "absent")
            .unwrap_err()
            .contains("not found"));
    }

    #[test]
    fn matching_digest_passes_and_mismatch_is_refused() {
        let pin = sha(GOOD);
        assert!(verify_archive_sha256("e", GOOD, Some(&pin), false).is_ok());
        let error = verify_archive_sha256("e", b"tampered", Some(&pin), false).unwrap_err();
        assert!(error.contains("failed SHA-256 verification"), "{error}");
        assert!(error.contains(&pin), "{error}");
        // The override never excuses a wrong digest.
        assert!(verify_archive_sha256("e", b"tampered", Some(&pin), true).is_err());
    }

    #[test]
    fn missing_pin_is_refused_unless_explicitly_allowed() {
        let error = verify_archive_sha256("e", GOOD, None, false).unwrap_err();
        assert!(error.contains(EXTENSION_ALLOW_UNPINNED_ENV), "{error}");
        assert!(verify_archive_sha256("e", GOOD, None, true).is_ok());
    }

    fn client(base: &str) -> ExtensionMarketplaceClient {
        ExtensionMarketplaceClient::from_config(Some(base))
    }

    #[test]
    fn download_refuses_an_archive_that_does_not_match_the_catalog_pin() {
        let catalog = json!([{"manifest": {"id": "connector.s3"}, "archive_sha256": sha(GOOD)}]);
        let base = serve_for_tests(catalog.to_string(), b"tampered module".to_vec(), 2);
        let error =
            download_verified_archive_with(&client(&base), "connector.s3", false).unwrap_err();
        assert!(error.contains("failed SHA-256 verification"), "{error}");
    }

    #[test]
    fn download_returns_the_archive_when_the_pin_matches() {
        let catalog = json!([{"manifest": {"id": "connector.s3"}, "archive_sha256": sha(GOOD)}]);
        let base = serve_for_tests(catalog.to_string(), GOOD.to_vec(), 2);
        let archive =
            download_verified_archive_with(&client(&base), "connector.s3", false).unwrap();
        assert_eq!(archive, GOOD);
    }
}
