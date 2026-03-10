use thiserror::Error;

#[derive(Error, Debug)]
pub enum OptimizerError {
    #[error("Failed to parse ONNX model: {0}")]
    ParseError(String),

    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    #[error("Optimization pass failed: {0}")]
    PassFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

pub type Result<T> = std::result::Result<T, OptimizerError>;
