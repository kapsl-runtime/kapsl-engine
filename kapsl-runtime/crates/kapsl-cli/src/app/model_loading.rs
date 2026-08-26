//! CLI-to-runtime composition for model loading.
//!
//! This adapter is the only place that knows both [`Args`] and the runtime's
//! model-loading abstractions. Runtime code receives an injected planner or a
//! narrow worker configuration and never depends on CLI fields directly.

use super::*;
use crate::runtime::model::{
    BackendTuningProvider, ModelLoadDefaults, ModelLoadPlacement, ModelLoadPlanner,
};

/// Model-loading policy resolved from CLI input before hardware is probed.
#[derive(Clone)]
pub(crate) struct ModelLoadingConfig {
    defaults: ModelLoadDefaults,
    tuning_provider: Arc<dyn BackendTuningProvider>,
}

impl ModelLoadingConfig {
    pub(crate) fn from_args(args: &Args) -> Result<Self, String> {
        Ok(Self {
            defaults: ModelLoadDefaults {
                batch_size: args.batch_size,
                scheduler_queue_size: args.scheduler_queue_size,
                scheduler_max_micro_batch: args.scheduler_max_micro_batch,
                scheduler_queue_delay_ms: args.scheduler_queue_delay_ms,
                placement: ModelLoadPlacement::new(args.topology.clone(), args.tp_degree),
            },
            tuning_provider: Arc::new(build_onnx_tuning_profile(args)?),
        })
    }

    pub(crate) fn planner(&self, device_info: Arc<DeviceInfo>) -> ModelLoadPlanner {
        ModelLoadPlanner::new(
            device_info,
            self.defaults.clone(),
            self.tuning_provider.clone(),
        )
    }
}
