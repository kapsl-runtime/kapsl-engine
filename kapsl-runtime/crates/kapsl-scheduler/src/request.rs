use kapsl_engine_api::{BinaryTensorPacket, EngineError, InferenceRequest};
use tokio::sync::oneshot;

pub struct Request {
    pub input: InferenceRequest,
    pub response_tx: oneshot::Sender<Result<BinaryTensorPacket, EngineError>>,
}
