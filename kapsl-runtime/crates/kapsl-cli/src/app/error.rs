//! Application-level error construction.

use crate::DynError;

pub(crate) fn dyn_error_from_message(message: impl Into<String>) -> DynError {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.into(),
    ))
}
