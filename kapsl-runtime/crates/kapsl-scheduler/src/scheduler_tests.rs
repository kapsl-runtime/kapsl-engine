use super::*;
use crate::replica_pool::ReplicaScheduler;
use crate::request::Request;
use crate::scheduler::QueueOverflowPolicy;
use async_trait::async_trait;
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineStream, InferenceRequest,
    TensorDtype,
};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use tokio::sync::oneshot;

struct MockEngine {
    metrics: EngineMetrics,
}

impl MockEngine {
    fn new(metrics: EngineMetrics) -> Self {
        Self { metrics }
    }
}

#[async_trait]
impl Engine for MockEngine {
    async fn load(&mut self, _model_path: &std::path::Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        Ok(request.input.clone())
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let output = Ok(request.input.clone());
        Box::pin(futures::stream::once(async move { output }))
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        self.metrics.clone()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

fn make_inference_request(session_id: Option<&str>) -> InferenceRequest {
    let input = BinaryTensorPacket {
        shape: vec![1],
        dtype: TensorDtype::Float32,
        data: vec![0, 0, 0, 0],
    };
    let request = InferenceRequest::new(input);
    match session_id {
        Some(id) => request.with_session_id(id),
        None => request,
    }
}

fn make_request(session_id: Option<&str>) -> Request {
    let (response_tx, _response_rx) = oneshot::channel();
    Request {
        input: make_inference_request(session_id),
        response_tx,
    }
}

fn build_scheduler_for_queue_tests(
    engines: Vec<EngineHandle>,
    queue_size: usize,
    cpu_active: usize,
) -> Scheduler {
    let cpu_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("Failed to create CPU thread pool");

    let high_queue = crate::gpu_executor::WorkQueue::new(queue_size);
    let low_queue = crate::gpu_executor::WorkQueue::new(queue_size);

    use crate::mesh_routing::MeshRouter;

    let scheduler = Scheduler {
        engines: engines.clone(),
        cpu_pool,
        gpu_high_priority_queues: vec![high_queue],
        gpu_low_priority_queues: vec![low_queue],
        _enable_fallback: false,
        cpu_active_count: Arc::new(AtomicUsize::new(cpu_active)),
        gpu_in_flight_count: Arc::new(AtomicUsize::new(0)),
        device_mesh: None,
        router: MeshRouter::new(None, 1),
        max_micro_batch: 1,
        queue_overflow_policy: QueueOverflowPolicy::Block,
    };

    scheduler
}

#[tokio::test]
async fn test_get_worker_index_round_robin() {
    let engine_handle: EngineHandle = Arc::new(MockEngine::new(EngineMetrics::default()));
    let scheduler = Scheduler::new(vec![engine_handle], 1, 3, 8, false, 1, 0, None);

    let indices = (0..4)
        .map(|_| scheduler.get_worker_index(&None))
        .collect::<Vec<_>>();

    assert_eq!(indices, vec![0, 1, 2, 0]);
}

#[tokio::test]
async fn test_get_worker_index_sticky_session() {
    let engine_handle: EngineHandle = Arc::new(MockEngine::new(EngineMetrics::default()));
    let scheduler = Scheduler::new(vec![engine_handle], 1, 4, 8, false, 1, 0, None);

    let session_id = Some("session-1".to_string());
    let first = scheduler.get_worker_index(&session_id);
    let second = scheduler.get_worker_index(&session_id);

    assert_eq!(first, second);
    assert!(first < scheduler.gpu_high_priority_queues.len());
}

#[test]
fn test_is_healthy_threshold() {
    let engine_handle: EngineHandle = Arc::new(MockEngine::new(EngineMetrics::default()));
    let scheduler = build_scheduler_for_queue_tests(vec![engine_handle], 10, 0);

    for _ in 0..7 {
        assert!(scheduler.gpu_high_priority_queues[0]
            .try_push_drop_newest(make_request(None))
            .is_ok());
    }
    assert!(scheduler.is_healthy());

    assert!(scheduler.gpu_high_priority_queues[0]
        .try_push_drop_newest(make_request(None))
        .is_ok());
    assert!(!scheduler.is_healthy());
}

#[test]
fn test_get_queue_depth_counts_cpu_and_gpu() {
    let engine_handle: EngineHandle = Arc::new(MockEngine::new(EngineMetrics::default()));
    let scheduler = build_scheduler_for_queue_tests(vec![engine_handle], 5, 2);

    assert!(scheduler.gpu_high_priority_queues[0]
        .try_push_drop_newest(make_request(None))
        .is_ok());
    assert!(scheduler.gpu_low_priority_queues[0]
        .try_push_drop_newest(make_request(None))
        .is_ok());
    assert!(scheduler.gpu_low_priority_queues[0]
        .try_push_drop_newest(make_request(None))
        .is_ok());

    let (cpu_depth, gpu_depth) = scheduler.get_queue_depth();
    assert_eq!(cpu_depth, 2);
    assert_eq!(gpu_depth, 3);
}

#[test]
fn test_metrics_aggregation() {
    let engine_a: EngineHandle = Arc::new(MockEngine::new(EngineMetrics {
        memory_usage: 10,
        gpu_utilization: 0.2,
        throughput: 5.0,
        ..EngineMetrics::default()
    }));
    let engine_b: EngineHandle = Arc::new(MockEngine::new(EngineMetrics {
        memory_usage: 20,
        gpu_utilization: 0.6,
        throughput: 7.0,
        ..EngineMetrics::default()
    }));
    let scheduler = build_scheduler_for_queue_tests(vec![engine_a, engine_b], 5, 1);

    assert!(scheduler.gpu_high_priority_queues[0]
        .try_push_drop_newest(make_request(None))
        .is_ok());
    assert!(scheduler.gpu_low_priority_queues[0]
        .try_push_drop_newest(make_request(None))
        .is_ok());

    let metrics = ReplicaScheduler::get_metrics(&scheduler);

    assert_eq!(metrics.memory_usage, 30);
    assert_eq!(metrics.throughput, 12.0);
    assert_eq!(metrics.queue_depth, 3);
    assert!((metrics.gpu_utilization - 0.4).abs() < 1e-6);
}
