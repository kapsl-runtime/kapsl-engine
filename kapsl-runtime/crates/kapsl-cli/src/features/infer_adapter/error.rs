use std::fmt;

/// Failure to convert a client inference payload into an engine request.
#[derive(Debug, Clone)]
pub(crate) enum InferRequestError {
    /// The payload is invalid or unsupported and should be reported as a
    /// client error.
    BadRequest(String),
    /// Conversion failed because of a runtime dependency or local I/O error.
    Internal(String),
}

impl InferRequestError {
    pub(super) fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    pub(super) fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    pub(crate) fn is_internal(&self) -> bool {
        matches!(self, Self::Internal(_))
    }
}

impl fmt::Display for InferRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferRequestError::BadRequest(message) => write!(f, "{}", message),
            InferRequestError::Internal(message) => write!(f, "{}", message),
        }
    }
}

pub(crate) type InferResult<T> = Result<T, InferRequestError>;
