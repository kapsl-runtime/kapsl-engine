use crate::serving_backend::ServingBackendPolicy;
use serde::{Deserialize, Serialize};

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
    #[serde(default)]
    pub(crate) serving_backend: Option<ServingBackendPolicy>,
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
