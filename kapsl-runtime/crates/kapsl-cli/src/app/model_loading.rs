//! CLI-to-runtime composition for model loading.
//!
//! This adapter is the only place that knows both [`Args`] and the runtime's
//! model-loading abstractions. Runtime code receives an injected planner or a
//! narrow worker configuration and never depends on CLI fields directly.

use super::*;
use crate::runtime::model::{
    ModelLoadDefaults, ModelLoadPlacement, ModelLoadPlanner, WorkerRunConfig,
};

pub(crate) fn build_model_load_planner(
    args: &Args,
    device_info: Arc<DeviceInfo>,
) -> Result<ModelLoadPlanner, String> {
    let tuning_provider = Arc::new(build_onnx_tuning_profile(args)?);
    let defaults = ModelLoadDefaults {
        batch_size: args.batch_size,
        scheduler_queue_size: args.scheduler_queue_size,
        scheduler_max_micro_batch: args.scheduler_max_micro_batch,
        scheduler_queue_delay_ms: args.scheduler_queue_delay_ms,
        placement: ModelLoadPlacement::new(args.topology.clone(), args.tp_degree),
    };

    Ok(ModelLoadPlanner::new(
        device_info,
        defaults,
        tuning_provider,
    ))
}

pub(crate) fn build_worker_run_config(args: &Args) -> Result<WorkerRunConfig, String> {
    let [model_path] = args.model.as_slice() else {
        return Err("worker mode expects exactly one --model".to_string());
    };
    Ok(WorkerRunConfig::new(
        args.worker_model_id.unwrap_or(0),
        model_path.clone(),
        args.socket.clone(),
    ))
}
