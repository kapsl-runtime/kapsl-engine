use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub type ConnectorConfig = Value;
pub type Metadata = HashMap<String, String>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDescriptor {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeltaOp {
    Upsert,
    Delete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentDelta {
    pub id: String,
    pub op: DeltaOp,
    pub etag: Option<String>,
    pub modified_at: Option<String>,
    pub metadata: Metadata,
    pub acl: ExternalAcl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentPayload {
    pub id: String,
    pub content_type: String,
    pub bytes_b64: String,
    pub metadata: Metadata,
    pub acl: ExternalAcl,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExternalAcl {
    pub allow_users: Vec<String>,
    pub allow_groups: Vec<String>,
    pub deny_users: Vec<String>,
    pub deny_groups: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncCursor {
    pub value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRequest {
    pub source_id: String,
    pub cursor: Option<SyncCursor>,
}
