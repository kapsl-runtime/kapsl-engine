use super::*;

#[derive(Debug, Deserialize)]
pub(crate) struct PackageKapslRequest {
    pub(crate) model_path: String,
    pub(crate) output_path: Option<String>,
    pub(crate) project_name: Option<String>,
    pub(crate) framework: Option<String>,
    #[serde(default)]
    pub(crate) format: Option<String>,
    #[serde(default)]
    pub(crate) model_type: Option<String>,
    #[serde(default)]
    pub(crate) task: Option<String>,
    pub(crate) version: Option<String>,
    pub(crate) metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PackageKapslResponse {
    pub(crate) status: String,
    pub(crate) kapsl_path: String,
    pub(crate) project_name: String,
    pub(crate) framework: String,
    pub(crate) version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) metadata_path: Option<String>,
}

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
    pub(crate) mirrored_path: String,
    pub(crate) bytes_uploaded: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) manifest_digest: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PullKapslRequest {
    pub(crate) target: String,
    pub(crate) reference: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct RemoteTokenStoreFile {
    #[serde(default)]
    pub(crate) tokens_by_remote: HashMap<String, String>,
    #[serde(default)]
    pub(crate) last_remote_url: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct LoginResponse {
    pub(crate) status: String,
    pub(crate) remote_url: String,
    pub(crate) auth_base_url: String,
    pub(crate) provider: String,
    pub(crate) login_method: String,
    pub(crate) callback_url: String,
    pub(crate) token_store_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) verification_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) user_code: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DeviceCodeStartResponse {
    pub(crate) device_code: String,
    pub(crate) user_code: String,
    pub(crate) verification_uri: String,
    pub(crate) verification_uri_complete: Option<String>,
    pub(crate) expires_in: Option<u64>,
    pub(crate) interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DeviceCodePollResponse {
    pub(crate) status: String,
    pub(crate) token: Option<String>,
    pub(crate) error: Option<String>,
    pub(crate) error_description: Option<String>,
    pub(crate) interval: Option<u64>,
}
