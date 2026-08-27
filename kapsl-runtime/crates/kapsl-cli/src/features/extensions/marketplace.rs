use crate::app::{EXTENSION_MARKETPLACE_URL, EXTENSION_MARKETPLACE_URL_ENV};
use crate::features::http_client::{format_remote_http_error, native_tls_http_agent};
use crate::runtime::optional_env_var;

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

pub(super) fn download_extension_archive(
    extension_id: &str,
    marketplace_url: Option<&str>,
) -> Result<Vec<u8>, String> {
    ExtensionMarketplaceClient::from_config(marketplace_url).download_archive(extension_id)
}
