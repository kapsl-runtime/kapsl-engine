//! Compose the gRPC adapter with the runtime's policy and inference services.

use super::*;
use kapsl_grpc::{tonic::Status, EngineFacade, EngineStream, RequestAuthorizer};

pub(crate) struct GrpcEngine {
    pub(crate) models: Arc<ModelManager>,
    pub(crate) inference: Arc<InferenceService>,
}

#[async_trait::async_trait]
impl EngineFacade for GrpcEngine {
    fn models(&self) -> Vec<kapsl_grpc::Model> {
        self.models
            .registry()
            .list()
            .into_iter()
            .filter(|model| model.replica_id == 0)
            .map(|model| {
                let pool = self.models.pool(model.id);
                kapsl_grpc::Model {
                    id: model.id,
                    name: model.name,
                    version: model.version,
                    ready: pool.as_ref().is_some_and(|pool| pool.is_healthy()),
                    info: pool.and_then(|pool| pool.model_info()),
                }
            })
            .collect()
    }

    async fn infer(
        &self,
        model_id: u32,
        request: InferenceRequest,
    ) -> Result<BinaryTensorPacket, EngineError> {
        let priority = self.inference.priority_for_request(&request);
        self.inference
            .infer(model_id, request, priority, false)
            .await
    }

    async fn infer_stream(
        &self,
        model_id: u32,
        request: InferenceRequest,
    ) -> Result<EngineStream, EngineError> {
        let priority = self.inference.priority_for_request(&request);
        self.inference
            .infer_stream(model_id, request, priority, false)
            .await
    }
}

pub(crate) struct GrpcAuthorizer(pub(crate) Arc<RwLock<ApiAuthState>>);

impl RequestAuthorizer for GrpcAuthorizer {
    fn authorize_reader(
        &self,
        authorization: Option<&str>,
        remote_ip: Option<IpAddr>,
    ) -> Result<(), Status> {
        authorize_api_request(
            &self.0,
            ApiRole::Reader,
            ApiScope::Read,
            authorization,
            remote_ip,
        )
        .map(|_| ())
        .map_err(|error| match error {
            ApiAuthorizationError::Unauthorized => {
                Status::unauthenticated("Invalid or missing API token")
            }
            ApiAuthorizationError::Forbidden => {
                Status::permission_denied("Token does not grant reader access")
            }
            ApiAuthorizationError::LocalOnly => {
                Status::permission_denied("Authentication is disabled; loopback clients only")
            }
        })
    }

    fn scope_session_id(
        &self,
        session_id: Option<&str>,
        authorization: Option<&str>,
    ) -> Option<String> {
        scope_session_id_for_authorization(session_id, authorization)
    }
}
