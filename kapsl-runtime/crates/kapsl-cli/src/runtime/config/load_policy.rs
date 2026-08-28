//! Process-level model loading policy.

use super::optional_env_var;
use crate::app::MODEL_LOAD_PARALLELISM_ENV;

/// Resolve how many startup models may load concurrently.
pub(crate) fn resolve_model_load_parallelism(model_count: usize) -> usize {
    if model_count <= 1 {
        return 1;
    }
    optional_env_var(MODEL_LOAD_PARALLELISM_ENV)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| model_count.min(4))
        .clamp(1, model_count)
}
