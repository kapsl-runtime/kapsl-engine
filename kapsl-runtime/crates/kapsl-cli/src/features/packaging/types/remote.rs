use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub(crate) struct PushKapslRequest {
    pub(crate) kapsl_path: String,
    pub(crate) target: String,
    pub(crate) remote_url: Option<String>,
    pub(crate) remote_token: Option<String>,
    #[serde(default)]
    pub(crate) interactive_login: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct PushKapslResponse {
    pub(crate) status: String,
    pub(crate) remote_url: String,
    pub(crate) artifact_url: String,
    pub(crate) bytes_uploaded: u64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PullKapslRequest {
    pub(crate) target: String,
    pub(crate) destination_dir: Option<String>,
    pub(crate) remote_url: Option<String>,
    pub(crate) remote_token: Option<String>,
    #[serde(default)]
    pub(crate) interactive_login: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct PullKapslResponse {
    pub(crate) status: String,
    pub(crate) remote_url: String,
    pub(crate) artifact_url: String,
    pub(crate) kapsl_path: String,
    pub(crate) bytes_downloaded: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RemoteArtifactLabelSummary {
    pub(crate) label: String,
    pub(crate) reference: String,
    pub(crate) size_bytes: u64,
    pub(crate) updated_at: String,
    pub(crate) download_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RemoteArtifactModelSummary {
    pub(crate) name: String,
    pub(crate) latest_label: Option<String>,
    pub(crate) latest_reference: Option<String>,
    pub(crate) artifact_count: usize,
    #[serde(default)]
    pub(crate) labels: Vec<RemoteArtifactLabelSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RemoteArtifactInventoryResponse {
    pub(crate) status: String,
    pub(crate) repo: String,
    #[serde(default)]
    pub(crate) available_repos: Vec<String>,
    #[serde(default)]
    pub(crate) models: Vec<RemoteArtifactModelSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimeRemoteArtifactInventoryResponse {
    pub(crate) status: String,
    pub(crate) remote_url: String,
    pub(crate) repo: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) available_repos: Vec<String>,
    #[serde(default)]
    pub(crate) models: Vec<RemoteArtifactModelSummary>,
}
