//! Runtime adapters for the backend-neutral MCP ingress crate.

use super::*;
use kapsl_mcp::{
    AuthorizationError as McpAuthorizationError, EngineFacade, McpServerConfig, McpServerHandle,
    Principal, RequestAuthorizer,
};

/// Scopes an MCP client's session id to the principal issued by
/// [`RuntimeMcpAuthorizer`], using the same derivation as HTTP and gRPC.
pub(crate) fn scope_mcp_session_id(
    principal: &Principal,
    session_id: Option<&str>,
) -> Result<Option<String>, String> {
    let principal = principal
        .downcast_ref::<ApiPrincipal>()
        .ok_or_else(|| "MCP request principal was not issued by the engine".to_string())?;
    Ok(principal.scope_session_id(session_id))
}

struct RuntimeMcpEngine {
    models: Arc<ModelManager>,
    inference: Arc<InferenceService>,
    request_adapters: RequestAdapterRegistry,
}

impl RuntimeMcpEngine {
    fn new(models: Arc<ModelManager>, inference: Arc<InferenceService>) -> Self {
        Self {
            models,
            inference,
            request_adapters: default_request_adapter_registry(),
        }
    }
}

#[async_trait::async_trait]
impl EngineFacade for RuntimeMcpEngine {
    async fn list_models(&self) -> Result<serde_json::Value, String> {
        let models = self
            .models
            .registry()
            .list()
            .into_iter()
            .filter(|model| self.models.can_serve(model.id))
            .collect::<Vec<_>>();
        serde_json::to_value(models).map_err(|error| error.to_string())
    }

    async fn infer(
        &self,
        principal: &Principal,
        model_id: u32,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        // With idle reload on, a model stopped for being idle is loaded
        // again before the request is served.
        match self.inference.ensure_loaded(model_id).await {
            Ok(()) => {}
            Err(IdleReloadError::NotLoaded) => {
                return Err(format!("Model {model_id} is not loaded"));
            }
            Err(error) => return Err(error.to_string()),
        }
        let model = self
            .models
            .registry()
            .get(model_id)
            .ok_or_else(|| format!("Model {model_id} is not registered"))?;
        let mut request = parse_inference_request_with_registry(
            payload,
            &model.framework,
            &self.request_adapters,
        )
        .map_err(|error| error.to_string())?;

        request.session_id = scope_mcp_session_id(principal, request.session_id.as_deref())?;
        let priority = self.inference.priority_for_request(&request);
        let force_cpu = request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.force_cpu)
            .unwrap_or(false);
        let response = self
            .inference
            .infer(model_id, request, priority, force_cpu)
            .await
            .map_err(|error| error.to_string())?;
        serde_json::to_value(response).map_err(|error| error.to_string())
    }
}

pub(crate) struct RuntimeMcpAuthorizer {
    auth_state: Arc<RwLock<ApiAuthState>>,
}

impl RuntimeMcpAuthorizer {
    pub(crate) fn new(auth_state: Arc<RwLock<ApiAuthState>>) -> Self {
        Self { auth_state }
    }
}

impl RequestAuthorizer for RuntimeMcpAuthorizer {
    fn authorize_reader(
        &self,
        authorization: Option<&str>,
        remote_ip: IpAddr,
    ) -> Result<Principal, McpAuthorizationError> {
        authorize_api_request(
            &self.auth_state,
            ApiRole::Reader,
            ApiScope::Read,
            authorization,
            Some(remote_ip),
        )
        .map(|access| Principal::new(access.principal))
        .map_err(|error| match error {
            ApiAuthorizationError::Unauthorized => McpAuthorizationError::Unauthorized,
            ApiAuthorizationError::Forbidden => McpAuthorizationError::Forbidden,
            ApiAuthorizationError::LocalOnly => McpAuthorizationError::LocalOnly,
        })
    }
}

pub(crate) async fn start_runtime_mcp_server(
    bind_addr: IpAddr,
    port: u16,
    allowed_hosts: Vec<String>,
    models: Arc<ModelManager>,
    inference: Arc<InferenceService>,
    auth_state: Arc<RwLock<ApiAuthState>>,
) -> Result<McpServerHandle, std::io::Error> {
    kapsl_mcp::start_mcp_server(
        McpServerConfig::new(bind_addr, port).with_allowed_hosts(allowed_hosts),
        Arc::new(RuntimeMcpEngine::new(models, inference)),
        Arc::new(RuntimeMcpAuthorizer::new(auth_state)),
    )
    .await
}
