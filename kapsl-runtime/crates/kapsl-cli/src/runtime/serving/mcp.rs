//! Runtime adapters for the backend-neutral MCP ingress crate.

use super::*;
use kapsl_mcp::{
    AuthorizationError as McpAuthorizationError, EngineFacade, McpServerConfig, McpServerHandle,
    RequestAuthorizer,
};

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
            .filter(|model| self.models.contains_pool(model.id))
            .collect::<Vec<_>>();
        serde_json::to_value(models).map_err(|error| error.to_string())
    }

    async fn infer(
        &self,
        model_id: u32,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        if !self.models.contains_pool(model_id) {
            return Err(format!("Model {model_id} is not loaded"));
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

        // Authentication happens at the MCP transport boundary, while the
        // current engine session namespace is credential-derived. Until the
        // facade carries a principal identity, do not accept a client-selected
        // session that could cross credential boundaries.
        request.session_id = None;
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

struct RuntimeMcpAuthorizer {
    auth_state: Arc<RwLock<ApiAuthState>>,
}

impl RequestAuthorizer for RuntimeMcpAuthorizer {
    fn authorize_reader(
        &self,
        authorization: Option<&str>,
        remote_ip: IpAddr,
    ) -> Result<(), McpAuthorizationError> {
        authorize_api_request(
            &self.auth_state,
            ApiRole::Reader,
            ApiScope::Read,
            authorization,
            Some(remote_ip),
        )
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
        Arc::new(RuntimeMcpAuthorizer { auth_state }),
    )
    .await
}
