//! Model Context Protocol ingress for kapsl-engine.
//!
//! This crate owns MCP-specific transport and schema concerns. Runtime policy,
//! authorization decisions, scheduling, and memory governance remain behind
//! the [`EngineFacade`] and [`RequestAuthorizer`] boundaries supplied by the
//! engine composition root. Caller identity follows the same rule: the
//! authorizer issues a [`Principal`], and this crate only carries it from the
//! HTTP boundary to the engine facade.

use async_trait::async_trait;
use axum::{
    extract::{ConnectInfo, Request, State},
    http::{header::AUTHORIZATION, request::Parts, StatusCode},
    middleware::{self, Next},
    response::Response,
    Router,
};
use rmcp::{
    handler::server::{router::tool::ToolRouter, tool::Extension, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerConfig},
    tool, tool_handler, tool_router, Json, ServerHandler,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use std::{
    any::Any,
    fmt,
    net::{IpAddr, SocketAddr},
    sync::Arc,
};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Engine-issued identity of the credential that authorized an MCP request.
///
/// [`RequestAuthorizer`] creates it and [`EngineFacade`] consumes it. This
/// crate never inspects or derives it, so caller identity stays owned by the
/// engine's authentication module.
#[derive(Clone)]
pub struct Principal(Arc<dyn Any + Send + Sync>);

impl Principal {
    /// Wrap the engine's own identity type.
    pub fn new<T: Any + Send + Sync>(identity: T) -> Self {
        Self(Arc::new(identity))
    }

    /// Recover the engine's identity type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.0.downcast_ref()
    }
}

impl fmt::Debug for Principal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Principal(..)")
    }
}

/// Engine operations exposed through MCP.
///
/// Implementations must route inference through the engine's governed ingress
/// rather than invoking a concrete backend directly.
#[async_trait]
pub trait EngineFacade: Send + Sync + 'static {
    /// Return the currently loaded, addressable models.
    async fn list_models(&self) -> Result<Value, String>;

    /// Execute one backend-neutral inference request through Kapsl scheduling
    /// on behalf of the principal the [`RequestAuthorizer`] issued for it.
    async fn infer(
        &self,
        principal: &Principal,
        model_id: u32,
        request: Value,
    ) -> Result<Value, String>;
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
    /// Authorize reader access for one HTTP request and return the identity
    /// of the credential that was evaluated.
    fn authorize_reader(
        &self,
        authorization: Option<&str>,
        remote_ip: IpAddr,
    ) -> Result<Principal, AuthorizationError>;
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
        Extension(http): Extension<Parts>,
        Parameters(request): Parameters<InferToolRequest>,
    ) -> Result<Json<Value>, String> {
        // The authorization middleware is the only writer of this extension.
        // Its absence means the request bypassed authorization: fail closed.
        let principal = http
            .extensions
            .get::<Principal>()
            .ok_or_else(|| "Request has no authorized principal".to_string())?;
        self.engine
            .infer(principal, request.model_id, request.request)
            .await
            .map(Json)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for KapslMcpHandler {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
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
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let started = std::time::Instant::now();
    let method = request.method().clone();
    let authorization = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok());
    let decision = authorizer
        .0
        .authorize_reader(authorization, remote.ip())
        .map_err(|error| match error {
            AuthorizationError::Unauthorized => StatusCode::UNAUTHORIZED,
            AuthorizationError::Forbidden | AuthorizationError::LocalOnly => StatusCode::FORBIDDEN,
        });
    let result = match decision {
        Ok(principal) => {
            // rmcp forwards these request parts, extensions included, to tools.
            request.extensions_mut().insert(principal);
            Ok(next.run(request).await)
        }
        Err(status) => Err(status),
    };
    let status = match &result {
        Ok(response) => response.status(),
        Err(status) => *status,
    };
    tracing::info!(
        target: "kapsl::access",
        protocol = "mcp",
        method = method.as_str(),
        status = status.as_u16(),
        elapsed_us = started.elapsed().as_micros() as u64,
        "request completed"
    );
    result
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

    /// Echoes the request together with the identity it was executed for.
    struct MockEngine;

    #[async_trait]
    impl EngineFacade for MockEngine {
        async fn list_models(&self) -> Result<Value, String> {
            Ok(json!([{"id": 7, "name": "test"}]))
        }

        async fn infer(
            &self,
            principal: &Principal,
            model_id: u32,
            request: Value,
        ) -> Result<Value, String> {
            Ok(json!({
                "principal": principal.downcast_ref::<String>(),
                "model_id": model_id,
                "request": request,
            }))
        }
    }

    struct LoopbackAuthorizer;

    impl RequestAuthorizer for LoopbackAuthorizer {
        fn authorize_reader(
            &self,
            _authorization: Option<&str>,
            remote_ip: IpAddr,
        ) -> Result<Principal, AuthorizationError> {
            remote_ip
                .is_loopback()
                .then(|| Principal::new("loopback".to_string()))
                .ok_or(AuthorizationError::LocalOnly)
        }
    }

    /// Issues the presented credential as the identity, so a test can tell
    /// which request's principal reached the engine.
    struct CredentialEchoAuthorizer;

    impl RequestAuthorizer for CredentialEchoAuthorizer {
        fn authorize_reader(
            &self,
            authorization: Option<&str>,
            _remote_ip: IpAddr,
        ) -> Result<Principal, AuthorizationError> {
            authorization
                .map(|credential| Principal::new(credential.to_string()))
                .ok_or(AuthorizationError::Unauthorized)
        }
    }

    struct DenyAuthorizer;

    impl RequestAuthorizer for DenyAuthorizer {
        fn authorize_reader(
            &self,
            _authorization: Option<&str>,
            _remote_ip: IpAddr,
        ) -> Result<Principal, AuthorizationError> {
            Err(AuthorizationError::Unauthorized)
        }
    }

    fn request_parts(principal: Option<Principal>) -> Parts {
        let (mut parts, ()) = axum::http::Request::new(()).into_parts();
        if let Some(principal) = principal {
            parts.extensions.insert(principal);
        }
        parts
    }

    fn infer_tool_request() -> InferToolRequest {
        InferToolRequest {
            model_id: 7,
            request: json!({"input": [1, 2, 3]}),
        }
    }

    /// POST one JSON-RPC message to `/mcp`; returns the status and raw body.
    async fn post_mcp(
        address: SocketAddr,
        authorization: Option<&str>,
        message: Value,
    ) -> (u16, String) {
        let body = message.to_string();
        let authorization = authorization
            .map(|credential| format!("Authorization: {credential}\r\n"))
            .unwrap_or_default();
        let request = format!(
            "POST /mcp HTTP/1.1\r\nHost: {address}\r\n{authorization}Accept: application/json, text/event-stream\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
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
        let response = String::from_utf8_lossy(&response).into_owned();
        let status = response
            .lines()
            .next()
            .and_then(|status_line| status_line.split_whitespace().nth(1))
            .expect("response status code")
            .parse()
            .expect("numeric response status");
        let body = response
            .split_once("\r\n\r\n")
            .map(|(_, body)| body.to_string())
            .unwrap_or_default();
        (status, body)
    }

    async fn initialize_request(address: SocketAddr) -> u16 {
        let message = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"}
            }
        });
        post_mcp(address, None, message).await.0
    }

    /// Extract the JSON-RPC object from a JSON, chunked, or SSE response body.
    fn json_rpc_payload(body: &str) -> Value {
        let start = body.find('{').expect("body contains JSON");
        let end = body.rfind('}').expect("body contains JSON");
        serde_json::from_str(&body[start..=end]).expect("valid JSON-RPC payload")
    }

    #[tokio::test]
    async fn handler_delegates_inference_with_authorized_principal() {
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
        let principal = Principal::new("caller-a".to_string());
        let response = handler
            .infer(
                Extension(request_parts(Some(principal))),
                Parameters(infer_tool_request()),
            )
            .await
            .expect("inference succeeds");
        assert_eq!(response.0["principal"], "caller-a");
        assert_eq!(response.0["model_id"], 7);
        assert_eq!(response.0["request"]["input"], json!([1, 2, 3]));
    }

    #[tokio::test]
    async fn handler_fails_closed_without_authorized_principal() {
        let handler = KapslMcpHandler::new(Arc::new(MockEngine));
        let error = handler
            .infer(
                Extension(request_parts(None)),
                Parameters(infer_tool_request()),
            )
            .await
            .err()
            .expect("inference without a principal is rejected");
        assert!(error.contains("principal"), "{error}");
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

    #[tokio::test]
    async fn streamable_http_delivers_each_requests_principal_to_engine() {
        let mut server = start_mcp_server(
            McpServerConfig::new(IpAddr::from([127, 0, 0, 1]), 0),
            Arc::new(MockEngine),
            Arc::new(CredentialEchoAuthorizer),
        )
        .await
        .expect("server binds");
        for credential in ["Bearer caller-a", "Bearer caller-b"] {
            let call = json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "kapsl_infer",
                    "arguments": {"model_id": 7, "request": {"session_id": "shared"}}
                }
            });
            let (status, body) = post_mcp(server.bound_addr(), Some(credential), call).await;
            assert_eq!(status, 200, "{body}");
            let result = &json_rpc_payload(&body)["result"]["structuredContent"];
            assert_eq!(result["principal"], credential, "{body}");
            assert_eq!(result["request"]["session_id"], "shared", "{body}");
        }
        server.abort();
        let _ = server.wait().await;
    }
}
