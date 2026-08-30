use serde::Deserialize;

const DEFAULT_DOWNLOAD_BASE_URL: &str = "https://downloads.kapsl.net";
const DOWNLOAD_BASE_URL_ENV: &str = "KAPSL_BASE_URL";
const RELEASE_VERSION_ENV: &str = "KAPSL_VERSION";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProviderPack {
    Cuda12,
    TensorRt10,
}

impl ProviderPack {
    pub(super) fn provider(self) -> &'static str {
        match self {
            Self::Cuda12 => "cuda",
            Self::TensorRt10 => "tensorrt",
        }
    }

    pub(super) fn package_stem(self) -> &'static str {
        match self {
            Self::Cuda12 => "kapsl-provider-cuda12",
            Self::TensorRt10 => "kapsl-provider-tensorrt10",
        }
    }

    pub(super) fn manifest_name(self) -> &'static str {
        match self {
            Self::Cuda12 => "kapsl-provider-cuda12.json",
            Self::TensorRt10 => "kapsl-provider-tensorrt10.json",
        }
    }

    pub(super) fn display_name(self) -> &'static str {
        match self {
            Self::Cuda12 => "CUDA 12",
            Self::TensorRt10 => "TensorRT 10",
        }
    }

    pub(super) fn owns_manifest_name(self, file_name: &str) -> bool {
        let normalized = file_name.to_ascii_lowercase();
        normalized.starts_with(self.package_stem()) && normalized.ends_with(".json")
    }
}

#[derive(Debug, Deserialize)]
pub(super) struct ProviderPackManifest {
    pub(super) provider: String,
    pub(super) runtime_version: String,
    pub(super) platform: String,
    pub(super) files: Vec<String>,
}

pub(super) fn configured_download_base_url() -> String {
    std::env::var(DOWNLOAD_BASE_URL_ENV).unwrap_or_else(|_| DEFAULT_DOWNLOAD_BASE_URL.to_string())
}

pub(super) fn release_version() -> String {
    std::env::var(RELEASE_VERSION_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            option_env!("KAPSL_VERSION")
                .unwrap_or(env!("CARGO_PKG_VERSION"))
                .to_string()
        })
}

pub(super) fn provider_release_url(base_url: &str, version: &str) -> String {
    let channel_path = if version.contains("-beta.") {
        "runtime/beta"
    } else {
        "runtime"
    };
    format!(
        "{}/{channel_path}/v{}",
        base_url.trim_end_matches('/'),
        version
    )
}

pub(super) fn provider_archive_name(pack: ProviderPack, version: &str) -> String {
    format!("{}-{}-windows-x86_64.zip", pack.package_stem(), version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_urls_follow_release_channel() {
        assert_eq!(
            provider_release_url("https://downloads.kapsl.net/", "0.1.18"),
            "https://downloads.kapsl.net/runtime/v0.1.18"
        );
        assert_eq!(
            provider_release_url(
                "https://downloads.kapsl.net",
                "0.1.18-beta.20260727.abcdef12"
            ),
            "https://downloads.kapsl.net/runtime/beta/v0.1.18-beta.20260727.abcdef12"
        );
    }

    #[test]
    fn provider_archive_names_are_versioned() {
        assert_eq!(
            provider_archive_name(ProviderPack::Cuda12, "0.1.18"),
            "kapsl-provider-cuda12-0.1.18-windows-x86_64.zip"
        );
        assert_eq!(
            provider_archive_name(ProviderPack::TensorRt10, "0.1.18"),
            "kapsl-provider-tensorrt10-0.1.18-windows-x86_64.zip"
        );
    }
}
