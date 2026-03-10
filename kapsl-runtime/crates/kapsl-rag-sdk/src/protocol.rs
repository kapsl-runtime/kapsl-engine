use serde::{Deserialize, Serialize};

use crate::types::{
    ConnectorConfig, DocumentDelta, DocumentPayload, ExternalAcl, SourceDescriptor,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorRequest {
    pub id: String,
    #[serde(flatten)]
    pub kind: ConnectorRequestKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum ConnectorRequestKind {
    ValidateConfig {
        config: ConnectorConfig,
    },
    AuthStart {
        config: ConnectorConfig,
    },
    AuthCallback {
        code: String,
        state: Option<String>,
    },
    ListSources {
        config: ConnectorConfig,
    },
    Sync {
        source_id: String,
        cursor: Option<String>,
    },
    FetchDocument {
        document_id: String,
    },
    ResolveAcl {
        acl: ExternalAcl,
    },
    Health,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorResponse {
    pub id: String,
    #[serde(flatten)]
    pub kind: ConnectorResponseKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", content = "result")]
pub enum ConnectorResponseKind {
    Ok(ConnectorResult),
    Err(ConnectorErrorPayload),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ConnectorResult {
    Unit,
    AuthUrl(String),
    Sources(Vec<SourceDescriptor>),
    Deltas(Vec<DocumentDelta>),
    Document(DocumentPayload),
    Acl(ExternalAcl),
    Health(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorErrorPayload {
    pub message: String,
    pub code: Option<String>,
}
