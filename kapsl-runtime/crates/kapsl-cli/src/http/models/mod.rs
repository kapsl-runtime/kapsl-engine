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
    pub(crate) model_runtime: Arc<ModelRuntime>,
    pub(crate) inference: Arc<InferenceService>,
    pub(crate) telemetry: Arc<ModelTelemetry>,
    pub(crate) rag_state: RagRuntimeState,
    pub(crate) auto_scaler: Arc<RwLock<AutoScaler>>,
    pub(crate) log_sensitive_ids: bool,
}

pub(crate) fn build_model_routes(config: ModelRoutesConfig) -> ModelRoutes {
    let ModelRoutesConfig {
        model_runtime,
        inference,
        telemetry,
        rag_state: rag_state_for_api,
        auto_scaler: auto_scaler_api,
        log_sensitive_ids: log_sensitive_ids_for_api,
    } = config;
    let models = model_runtime.models().clone();
    let shared_metrics_clone = model_runtime.shared_metrics().clone();

    let reader_routes = build_model_reader_routes(ModelReaderRoutesConfig {
        models: models.clone(),
        shared_metrics: shared_metrics_clone.clone(),
        telemetry,
    });

    let lifecycle_routes = build_model_lifecycle_routes(ModelLifecycleRoutesConfig {
        model_runtime: model_runtime.clone(),
    });

    let swap_routes = build_model_swap_routes(ModelSwapRoutesConfig {
        models: models.clone(),
    });

    let infer_route = build_model_infer_route(ModelInferRouteConfig {
        models: models.clone(),
        inference: inference.clone(),
        log_sensitive_ids: log_sensitive_ids_for_api,
        rag_state: rag_state_for_api.clone(),
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
