use super::*;
use serde::{Deserialize, Serialize};
use warp::Filter;

mod infer;
mod infer_stream;
mod lifecycle;
mod reader;
mod scaling;
mod swap;

use infer::{build_model_infer_route, ModelInferRouteConfig};
pub(crate) use infer_stream::{build_model_infer_stream_route, ModelInferStreamRouteConfig};
use lifecycle::{build_model_lifecycle_routes, ModelLifecycleRoutesConfig};
use reader::{build_model_reader_routes, ModelReaderRoutesConfig};
use scaling::{build_model_scaling_routes, ModelScalingRoutesConfig};
use swap::{build_model_swap_routes, ModelSwapRoutesConfig};

pub(crate) struct ModelRoutes {
    pub(crate) reader: warp::filters::BoxedFilter<(warp::reply::Response,)>,
    pub(crate) admin: warp::filters::BoxedFilter<(warp::reply::Response,)>,
}

pub(crate) struct ModelRoutesConfig {
    pub(crate) models: Arc<ModelManager>,
    pub(crate) inference: Arc<InferenceService>,
    pub(crate) shared_metrics: kapsl_monitor::metrics::KapslMetrics,
    pub(crate) throughput_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>>,
    pub(crate) generated_token_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>>,
    pub(crate) total_token_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>>,
    pub(crate) latency_samples: Arc<RwLock<HashMap<u32, LatencyWindow>>>,
    pub(crate) device_info: Arc<DeviceInfo>,
    pub(crate) batch_size: usize,
    pub(crate) scheduler_queue_size: usize,
    pub(crate) scheduler_max_micro_batch: usize,
    pub(crate) scheduler_queue_delay_ms: u64,
    pub(crate) onnx_tuning_profile: Arc<OnnxTuningProfile>,
    pub(crate) resources: Arc<RuntimeResources>,
    pub(crate) rag_state: RagRuntimeState,
    pub(crate) inter_model_relay_state: Arc<InterModelRelayState>,
    pub(crate) auto_scaler: Arc<RwLock<AutoScaler>>,
    pub(crate) log_sensitive_ids: bool,
}

pub(crate) fn build_model_routes(config: ModelRoutesConfig) -> ModelRoutes {
    let ModelRoutesConfig {
        models,
        inference,
        shared_metrics: shared_metrics_clone,
        throughput_samples: throughput_samples_clone,
        generated_token_samples: generated_token_samples_clone,
        total_token_samples: total_token_samples_clone,
        latency_samples: latency_samples_clone,
        device_info: device_info_for_api,
        batch_size,
        scheduler_queue_size,
        scheduler_max_micro_batch,
        scheduler_queue_delay_ms,
        onnx_tuning_profile: onnx_tuning_profile_for_api,
        resources,
        rag_state: rag_state_for_api,
        inter_model_relay_state,
        auto_scaler: auto_scaler_api,
        log_sensitive_ids: log_sensitive_ids_for_api,
    } = config;

    let reader_routes = build_model_reader_routes(ModelReaderRoutesConfig {
        models: models.clone(),
        shared_metrics: shared_metrics_clone.clone(),
        throughput_samples: throughput_samples_clone.clone(),
        generated_token_samples: generated_token_samples_clone.clone(),
        total_token_samples: total_token_samples_clone.clone(),
        latency_samples: latency_samples_clone.clone(),
    });

    let lifecycle_routes = build_model_lifecycle_routes(ModelLifecycleRoutesConfig {
        models: models.clone(),
        device_info: device_info_for_api.clone(),
        batch_size,
        scheduler_queue_size,
        scheduler_max_micro_batch,
        scheduler_queue_delay_ms,
        shared_metrics: shared_metrics_clone.clone(),
        onnx_tuning_profile: onnx_tuning_profile_for_api.clone(),
        resources: resources.clone(),
    });

    let swap_routes = build_model_swap_routes(ModelSwapRoutesConfig {
        models: models.clone(),
    });

    let infer_route = build_model_infer_route(ModelInferRouteConfig {
        models: models.clone(),
        inference: inference.clone(),
        log_sensitive_ids: log_sensitive_ids_for_api,
        rag_state: rag_state_for_api.clone(),
        inter_model_relay_state: inter_model_relay_state.clone(),
    });

    let infer_stream_route = build_model_infer_stream_route(ModelInferStreamRouteConfig {
        models: models.clone(),
        inference,
        log_sensitive_ids: log_sensitive_ids_for_api,
        rag_state: rag_state_for_api.clone(),
    });

    let scaling_routes = build_model_scaling_routes(ModelScalingRoutesConfig {
        auto_scaler: auto_scaler_api.clone(),
    });

    let reader = reader_routes
        .or(infer_route)
        .or(infer_stream_route)
        .or(scaling_routes.reader)
        .map(reply_into_response)
        .boxed();
    let admin = lifecycle_routes
        .or(swap_routes)
        .or(scaling_routes.admin)
        .map(reply_into_response)
        .boxed();

    ModelRoutes { reader, admin }
}
