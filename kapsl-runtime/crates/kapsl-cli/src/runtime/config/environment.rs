//! Environment access helpers shared across runtime configuration.

/// Read an environment variable and normalize blank values to `None`.
pub(crate) fn optional_env_var(name: &str) -> Option<String> {
    let value = std::env::var(name).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}
