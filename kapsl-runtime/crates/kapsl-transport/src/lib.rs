use async_trait::async_trait;
use kapsl_engine_api::BinaryTensorPacket;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod connection_pool;
pub mod tcp;

/// Common error type for all transport implementations
#[derive(Error, Debug)]
pub enum TransportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Timeout error")]
    Timeout,

    #[error("Model not found: {0}")]
    ModelNotFound(u32),

    #[error("Queue full")]
    QueueFull,

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Server error: {0}")]
    ServerError(String),
}

/// Request metadata shared between transport implementations
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RequestMetadata {
    pub request_id: u64,
    pub model_id: u32,
    pub priority: u8,
    pub force_cpu: bool,
    pub _padding: [u8; 2],
    pub timestamp_ns: u64,
}

impl RequestMetadata {
    pub fn new(request_id: u64, model_id: u32, priority: u8, force_cpu: bool) -> Self {
        Self {
            request_id,
            model_id,
            priority,
            force_cpu,
            _padding: [0; 2],
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
}

/// Response metadata shared between transport implementations
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub request_id: u64,
    pub status: u8,
    pub _padding: [u8; 7],
    pub latency_ns: u64,
}

impl ResponseMetadata {
    pub fn success(request_id: u64, latency_ns: u64) -> Self {
        Self {
            request_id,
            status: 0,
            _padding: [0; 7],
            latency_ns,
        }
    }

    pub fn error(request_id: u64, latency_ns: u64) -> Self {
        Self {
            request_id,
            status: 1,
            _padding: [0; 7],
            latency_ns,
        }
    }

    pub fn is_success(&self) -> bool {
        self.status == 0
    }
}

/// Server-side transport trait
#[async_trait]
pub trait TransportServer: Send + Sync {
    /// Start the transport server
    /// This should block until the server is shut down
    async fn run(&self) -> Result<(), TransportError>;

    /// Gracefully shutdown the server
    async fn shutdown(&self) -> Result<(), TransportError>;

    /// Get the transport type name (for logging/debugging)
    fn transport_type(&self) -> &'static str;
}

/// Client-side transport trait
#[async_trait]
pub trait TransportClient: Send + Sync {
    /// Send inference request and wait for response
    async fn infer(
        &self,
        model_id: u32,
        input: BinaryTensorPacket,
    ) -> Result<BinaryTensorPacket, TransportError>;

    /// Send streaming inference request
    #[cfg(feature = "streaming")]
    async fn infer_stream(
        &self,
        model_id: u32,
        input: BinaryTensorPacket,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<BinaryTensorPacket, TransportError>> + Send>,
        >,
        TransportError,
    >;

    /// Get the transport type name (for logging/debugging)
    fn transport_type(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_metadata_creation() {
        let req = RequestMetadata::new(123, 0, 1, false);
        assert_eq!(req.request_id, 123);
        assert_eq!(req.model_id, 0);
        assert_eq!(req.priority, 1);
        assert!(!req.force_cpu);
        assert!(req.timestamp_ns > 0);
    }

    #[test]
    fn test_response_metadata() {
        let resp = ResponseMetadata::success(456, 1000);
        assert_eq!(resp.request_id, 456);
        assert!(resp.is_success());

        let err_resp = ResponseMetadata::error(789, 2000);
        assert_eq!(err_resp.request_id, 789);
        assert!(!err_resp.is_success());
    }
}
