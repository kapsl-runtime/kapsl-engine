use async_trait::async_trait;
use futures::executor::block_on;
use std::io::{self, BufRead, Write};

use crate::protocol::{
    ConnectorRequest, ConnectorRequestKind, ConnectorResponse, ConnectorResponseKind,
    ConnectorResult,
};
use crate::types::{
    ConnectorConfig, DocumentDelta, DocumentPayload, ExternalAcl, SourceDescriptor,
};

#[derive(thiserror::Error, Debug)]
pub enum ConnectorError {
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<io::Error> for ConnectorError {
    fn from(err: io::Error) -> Self {
        ConnectorError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for ConnectorError {
    fn from(err: serde_json::Error) -> Self {
        ConnectorError::Serialization(err.to_string())
    }
}

#[async_trait]
pub trait Connector: Send + Sync {
    async fn validate_config(&self, config: ConnectorConfig) -> Result<(), ConnectorError>;
    async fn auth_start(&self, config: ConnectorConfig) -> Result<String, ConnectorError>;
    async fn auth_callback(
        &self,
        code: String,
        state: Option<String>,
    ) -> Result<(), ConnectorError>;
    async fn list_sources(
        &self,
        config: ConnectorConfig,
    ) -> Result<Vec<SourceDescriptor>, ConnectorError>;
    async fn sync(
        &self,
        source_id: String,
        cursor: Option<String>,
    ) -> Result<Vec<DocumentDelta>, ConnectorError>;
    async fn fetch_document(&self, document_id: String) -> Result<DocumentPayload, ConnectorError>;
    async fn resolve_acl(&self, acl: ExternalAcl) -> Result<ExternalAcl, ConnectorError>;
    async fn health(&self) -> Result<String, ConnectorError> {
        Ok("ok".to_string())
    }
}

pub fn serve_stdio<C: Connector>(connector: C) -> Result<(), ConnectorError> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: ConnectorRequest = serde_json::from_str(&line)?;
        let response = block_on(dispatch(&connector, request));
        let json = serde_json::to_string(&response)?;
        writeln!(stdout, "{}", json)?;
        stdout.flush()?;
    }
    Ok(())
}

async fn dispatch<C: Connector>(connector: &C, request: ConnectorRequest) -> ConnectorResponse {
    let id = request.id.clone();
    let result = match request.kind {
        ConnectorRequestKind::ValidateConfig { config } => connector
            .validate_config(config)
            .await
            .map(|_| ConnectorResult::Unit),
        ConnectorRequestKind::AuthStart { config } => connector
            .auth_start(config)
            .await
            .map(ConnectorResult::AuthUrl),
        ConnectorRequestKind::AuthCallback { code, state } => connector
            .auth_callback(code, state)
            .await
            .map(|_| ConnectorResult::Unit),
        ConnectorRequestKind::ListSources { config } => connector
            .list_sources(config)
            .await
            .map(ConnectorResult::Sources),
        ConnectorRequestKind::Sync { source_id, cursor } => connector
            .sync(source_id, cursor)
            .await
            .map(ConnectorResult::Deltas),
        ConnectorRequestKind::FetchDocument { document_id } => connector
            .fetch_document(document_id)
            .await
            .map(ConnectorResult::Document),
        ConnectorRequestKind::ResolveAcl { acl } => {
            connector.resolve_acl(acl).await.map(ConnectorResult::Acl)
        }
        ConnectorRequestKind::Health => connector.health().await.map(ConnectorResult::Health),
    };

    let kind = match result {
        Ok(value) => ConnectorResponseKind::Ok(value),
        Err(err) => ConnectorResponseKind::Err(crate::protocol::ConnectorErrorPayload {
            message: err.to_string(),
            code: None,
        }),
    };

    ConnectorResponse { id, kind }
}
