use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConnectorRuntime {
    Wasm,
    Sidecar,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConnectorCapability {
    Sync,
    OnDemand,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConnectorAuthMethod {
    None,
    ApiKey,
    OAuthDeviceCode,
    OAuthAuthorizationCode,
    AwsAccessKey,
    AwsAssumeRole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorManifest {
    pub id: String,
    pub name: String,
    pub version: String,
    pub runtime: ConnectorRuntime,
    pub capabilities: Vec<ConnectorCapability>,
    pub auth: Vec<ConnectorAuthMethod>,
    pub permissions: Vec<String>,
    pub description: Option<String>,
    pub entrypoint: Option<String>,
    pub config_schema: Option<serde_json::Value>,
}
