use async_trait::async_trait;
use kapsl_rag_sdk::server::{serve_stdio, Connector, ConnectorError};
use kapsl_rag_sdk::types::{
    ConnectorConfig, DocumentDelta, DocumentPayload, ExternalAcl, SourceDescriptor,
};

struct EchoConnector;

#[async_trait]
impl Connector for EchoConnector {
    async fn validate_config(&self, _config: ConnectorConfig) -> Result<(), ConnectorError> {
        Ok(())
    }

    async fn auth_start(&self, _config: ConnectorConfig) -> Result<String, ConnectorError> {
        Err(ConnectorError::Unsupported(
            "auth not implemented".to_string(),
        ))
    }

    async fn auth_callback(
        &self,
        _code: String,
        _state: Option<String>,
    ) -> Result<(), ConnectorError> {
        Err(ConnectorError::Unsupported(
            "auth not implemented".to_string(),
        ))
    }

    async fn list_sources(
        &self,
        _config: ConnectorConfig,
    ) -> Result<Vec<SourceDescriptor>, ConnectorError> {
        Ok(vec![SourceDescriptor {
            id: "echo".to_string(),
            name: "Echo Source".to_string(),
            kind: "echo".to_string(),
            metadata: Default::default(),
        }])
    }

    async fn sync(
        &self,
        _source_id: String,
        _cursor: Option<String>,
    ) -> Result<Vec<DocumentDelta>, ConnectorError> {
        Ok(Vec::new())
    }

    async fn fetch_document(
        &self,
        _document_id: String,
    ) -> Result<DocumentPayload, ConnectorError> {
        Err(ConnectorError::Unsupported(
            "fetch not implemented".to_string(),
        ))
    }

    async fn resolve_acl(&self, acl: ExternalAcl) -> Result<ExternalAcl, ConnectorError> {
        Ok(acl)
    }
}

fn main() -> Result<(), ConnectorError> {
    serve_stdio(EchoConnector)
}
