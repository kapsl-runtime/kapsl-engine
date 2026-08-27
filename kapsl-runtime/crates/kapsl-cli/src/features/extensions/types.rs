use kapsl_rag::extension::InstalledExtension;
use kapsl_rag_sdk::manifest::ConnectorManifest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub(crate) struct SyncExtensionRequest {
    pub(crate) workspace_id: String,
    pub(crate) source_id: Option<String>,
    pub(crate) cursor: Option<String>,
    pub(crate) tenant_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ExtensionInstallRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) extension_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) marketplace_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstalledExtensionPayload {
    manifest: ConnectorManifest,
    path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ExtensionInstallResponse {
    status: String,
    extension: InstalledExtensionPayload,
}

impl ExtensionInstallResponse {
    pub(crate) fn installed(extension: InstalledExtension) -> Self {
        Self {
            status: "ok".to_string(),
            extension: InstalledExtensionPayload {
                manifest: extension.manifest,
                path: extension.path.to_string_lossy().into_owned(),
            },
        }
    }

    pub(crate) fn is_success(&self) -> bool {
        self.status == "ok"
    }

    pub(crate) fn display_name(&self) -> &str {
        &self.extension.manifest.name
    }

    pub(crate) fn version(&self) -> &str {
        &self.extension.manifest.version
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ExtensionErrorResponse {
    pub(crate) error: String,
}
