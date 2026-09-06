//! Model Context Protocol ingress for kapsl-engine.
//!
//! This crate owns MCP-specific transport and schema concerns. Runtime policy,
//! authorization decisions, scheduling, and memory governance remain behind
//! the [`EngineFacade`] and [`RequestAuthorizer`] boundaries supplied by the
//! engine composition root.

use async_trait::async_trait;
use axum::{
    extract::{ConnectInfo, Request, State},
    http::{header::AUTHORIZATION, StatusCode},
    middleware::{self, Next},
    response::Response,
    Router,
};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router, Json, ServerHandler,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use std::{
    fmt,
    net::{IpAddr, SocketAddr},
    sync::Arc,
};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Engine operations exposed through MCP.
///
/// Implementations must route inference through the engine's governed ingress
/// rather than invoking a concrete backend directly.
#[async_trait]
pub trait EngineFacade: Send + Sync + 'static {
    /// Return the currently loaded, addressable models.
    async fn list_models(&self) -> Result<Value, String>;

    /// Execute one backend-neutral inference request through Kapsl scheduling.
    async fn infer(&self, model_id: u32, request: Value) -> Result<Value, String>;
}

/// Result of applying the engine's shared access policy to an MCP request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorizationError {
    Unauthorized,
    Forbidden,
    LocalOnly,
}

/// Adapter to the engine-owned authorization policy.
pub trait RequestAuthorizer: Send + Sync + 'static {
    /// Authorize reader access for one HTTP request.
    fn authorize_reader(
        &self,
        authorization: Option<&str>,
        remote_ip: IpAddr,
    ) -> Result<(), AuthorizationError>;
}

/// Configuration for the dedicated MCP Streamable HTTP listener.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    pub bind_addr: IpAddr,
    pub port: u16,
    /// Authorities accepted in the HTTP `Host` header.
    pub allowed_hosts: Vec<String>,
}

impl McpServerConfig {
    pub fn new(bind_addr: IpAddr, port: u16) -> Self {
        let host = bind_addr.to_string();
        Self {
            bind_addr,
            port,
            allowed_hosts: vec![
                "localhost".to_string(),
                "127.0.0.1".to_string(),
                "::1".to_string(),
                host.clone(),
                format!("{host}:{port}"),
            ],
        }
    }

    pub fn with_allowed_hosts(
        mut self,
        allowed_hosts: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allowed_hosts
            .extend(allowed_hosts.into_iter().map(Into::into));
        self.allowed_hosts.sort();
        self.allowed_hosts.dedup();
        self
    }
}

/// A bound MCP listener and its lifecycle controls.
pub struct McpServerHandle {
    bound_addr: SocketAddr,
    cancellation: CancellationToken,
    task: JoinHandle<Result<(), std::io::Error>>,
}

impl McpServerHandle {
    pub fn bound_addr(&self) -> SocketAddr {
        self.bound_addr
    }

    pub async fn wait(&mut self) -> Result<Result<(), std::io::Error>, tokio::task::JoinError> {
        (&mut self.task).await
    }

    pub fn abort(&self) {
        self.cancellation.cancel();
        self.task.abort();
    }
}

impl Drop for McpServerHandle {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.task.abort();
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct InferToolRequest {
    /// Numeric ID returned by kapsl_list_models.
    model_id: u32,
    /// A backend-neutral JSON inference envelope containing tensor or media input.
    request: Value,
}

#[derive(Clone)]
struct KapslMcpHandler {
    engine: Arc<dyn EngineFacade>,
    tool_router: ToolRouter<Self>,
}

impl fmt::Debug for KapslMcpHandler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KapslMcpHandler")
            .finish_non_exhaustive()
    }
}

#[tool_router(router = tool_router)]
impl KapslMcpHandler {
    fn new(engine: Arc<dyn EngineFacade>) -> Self {
        Self {
            engine,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "kapsl_list_models",
        description = "List models currently loaded and addressable by kapsl-engine"
    )]
    async fn list_models(&self) -> Result<Json<Value>, String> {
        self.engine.list_models().await.map(Json)
    }

    #[tool(
        name = "kapsl_infer",
        description = "Run inference through kapsl-engine scheduling and memory governance"
    )]
    async fn infer(
        &self,
        Parameters(request): Parameters<InferToolRequest>,
    ) -> Result<Json<Value>, String> {
        self.engine
            .infer(request.model_id, request.request)
            .await
            .map(Json)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for KapslMcpHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                "kapsl-engine",
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "Use kapsl_list_models to discover live models, then kapsl_infer to submit work through Kapsl governance.",
            )
    }
}

#[derive(Clone)]
struct AuthorizationState(Arc<dyn RequestAuthorizer>);

async fn authorize_request(
    State(authorizer): State<AuthorizationState>,
    ConnectInfo(remote): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let authorization = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok());
    authorizer
        .0
        .authorize_reader(authorization, remote.ip())
        .map_err(|error| match error {
            AuthorizationError::Unauthorized => StatusCode::UNAUTHORIZED,
            AuthorizationError::Forbidden | AuthorizationError::LocalOnly => StatusCode::FORBIDDEN,
        })?;
    Ok(next.run(request).await)
}

/// Bind and start a Streamable HTTP MCP server at `/mcp`.
pub async fn start_mcp_server(
    config: McpServerConfig,
    engine: Arc<dyn EngineFacade>,
    authorizer: Arc<dyn RequestAuthorizer>,
) -> Result<McpServerHandle, std::io::Error> {
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };

    let cancellation = CancellationToken::new();
    let mcp_config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        .with_allowed_hosts(config.allowed_hosts)
        .with_cancellation_token(cancellation.child_token());
    let handler_engine = engine.clone();
    let service: StreamableHttpService<KapslMcpHandler, LocalSessionManager> =
        StreamableHttpService::new(
            move || Ok(KapslMcpHandler::new(handler_engine.clone())),
            Default::default(),
            mcp_config,
        );
    let auth_state = AuthorizationState(authorizer);
    let app = Router::new()
        .nest_service("/mcp", service)
        .layer(middleware::from_fn_with_state(
            auth_state,
            authorize_request,
        ));
    let listener = tokio::net::TcpListener::bind((config.bind_addr, config.port)).await?;
    let bound_addr = listener.local_addr()?;
    let shutdown = cancellation.clone();
    let task = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await
    });

    Ok(McpServerHandle {
        bound_addr,
        cancellation,
        task,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    struct MockEngine;

    #[async_trait]
    impl EngineFacade for MockEngine {
        async fn list_models(&self) -> Result<Value, String> {
            Ok(json!([{"id": 7, "name": "test"}]))
        }

        async fn infer(&self, model_id: u32, request: Value) -> Result<Value, String> {
            Ok(json!({"model_id": model_id, "request": request}))
        }
    }

    struct LoopbackAuthorizer;

    impl RequestAuthorizer for LoopbackAuthorizer {
        fn authorize_reader(
            &self,
            _authorization: Option<&str>,
            remote_ip: IpAddr,
        ) -> Result<(), AuthorizationError> {
            remote_ip
                .is_loopback()
                .then_some(())
                .ok_or(AuthorizationError::LocalOnly)
        }
    }

    struct DenyAuthorizer;

    impl RequestAuthorizer for DenyAuthorizer {
        fn authorize_reader(
            &self,
            _authorization: Option<&str>,
            _remote_ip: IpAddr,
        ) -> Result<(), AuthorizationError> {
            Err(AuthorizationError::Unauthorized)
        }
    }

    async fn initialize_request(address: SocketAddr) -> u16 {
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"}
            }
        })
        .to_string();
        let request = format!(
            "POST /mcp HTTP/1.1\r\nHost: {address}\r\nAccept: application/json, text/event-stream\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let mut stream = tokio::net::TcpStream::connect(address)
            .await
            .expect("connect to MCP server");
        stream
            .write_all(request.as_bytes())
            .await
            .expect("write MCP request");
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .await
            .expect("read MCP response");
        let status_line = String::from_utf8_lossy(&response)
            .lines()
            .next()
            .expect("response has status line")
            .to_string();
        status_line
            .split_whitespace()
            .nth(1)
            .expect("response status code")
            .parse()
            .expect("numeric response status")
    }

    #[tokio::test]
    async fn handler_delegates_inference_to_engine_facade() {
        let handler = KapslMcpHandler::new(Arc::new(MockEngine));
        let tool_names = handler
            .tool_router
            .list_all()
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<Vec<_>>();
        assert_eq!(tool_names.len(), 2);
        assert!(tool_names.iter().any(|name| name == "kapsl_infer"));
        assert!(tool_names.iter().any(|name| name == "kapsl_list_models"));
        let response = handler
            .infer(Parameters(InferToolRequest {
                model_id: 7,
                request: json!({"input": [1, 2, 3]}),
            }))
            .await
            .expect("inference succeeds");
        assert_eq!(response.0["model_id"], 7);
        assert_eq!(response.0["request"]["input"], json!([1, 2, 3]));
    }

    #[tokio::test]
    async fn streamable_http_serves_mcp_initialize() {
        let mut server = start_mcp_server(
            McpServerConfig::new(IpAddr::from([127, 0, 0, 1]), 0),
            Arc::new(MockEngine),
            Arc::new(LoopbackAuthorizer),
        )
        .await
        .expect("server binds");
        assert_eq!(initialize_request(server.bound_addr()).await, 200);
        server.abort();
        let _ = server.wait().await;
    }

    #[tokio::test]
    async fn streamable_http_applies_authorizer_before_mcp_dispatch() {
        let mut server = start_mcp_server(
            McpServerConfig::new(IpAddr::from([127, 0, 0, 1]), 0),
            Arc::new(MockEngine),
            Arc::new(DenyAuthorizer),
        )
        .await
        .expect("server binds");
        assert_eq!(initialize_request(server.bound_addr()).await, 401);
        server.abort();
        let _ = server.wait().await;
    }
}
