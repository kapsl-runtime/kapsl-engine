pub mod gpu_executor;
pub mod mesh_routing;
pub mod priority;
pub mod replica_pool;
pub mod request;
pub mod request_metadata;
pub mod scheduler;

// Re-export main types
pub use priority::Priority;
pub use replica_pool::{PoolStrategy, ReplicaPool, ReplicaScheduler, ReplicaStats};
pub use request_metadata::{determine_priority, RequestMetadata};
pub use scheduler::{QueueOverflowPolicy, Scheduler};

#[cfg(test)]
mod tests {
    use crate::priority::Priority;
    use crate::scheduler::Scheduler;
    use async_trait::async_trait;
    use kapsl_engine_api::{
        BinaryTensorPacket, Engine, EngineError, InferenceRequest, TensorDtype,
    };
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct MockEngine {
        call_count: Arc<Mutex<usize>>,
        delay: Option<Duration>,
    }

    impl MockEngine {
        fn new() -> Self {
            Self {
                call_count: Arc::new(Mutex::new(0)),
                delay: None,
            }
        }

        fn with_delay(ms: u64) -> Self {
            Self {
                call_count: Arc::new(Mutex::new(0)),
                delay: Some(Duration::from_millis(ms)),
            }
        }
    }

    #[async_trait]
    impl Engine for MockEngine {
        async fn load(&mut self, _model_path: &std::path::Path) -> Result<(), EngineError> {
            Ok(())
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            if let Some(delay) = self.delay {
                std::thread::sleep(delay);
            }
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
            Ok(request.input.clone())
        }

        fn infer_stream(
            &self,
            request: &InferenceRequest,
        ) -> std::pin::Pin<
            Box<dyn futures::stream::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
        > {
            let result = Ok(request.input.clone());
            Box::pin(futures::stream::once(async move { result }))
        }

        fn unload(&mut self) {}

        fn metrics(&self) -> kapsl_engine_api::EngineMetrics {
            kapsl_engine_api::EngineMetrics::default()
        }

        fn health_check(&self) -> Result<(), EngineError> {
            Ok(()) // Mock is always healthy
        }
    }

    fn make_request() -> InferenceRequest {
        InferenceRequest {
            input: BinaryTensorPacket {
                shape: vec![1, 1],
                dtype: TensorDtype::Float32,
                data: vec![0, 0, 0, 0],
            },
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        }
    }

    #[tokio::test]
    async fn test_cpu_scheduling() {
        let engine_handle: Arc<dyn Engine> = Arc::new(MockEngine::new());
        let scheduler = Scheduler::new(vec![engine_handle], 2, 1, 1000, true, 1, 0, None);

        let result = scheduler
            .infer(make_request(), Priority::Throughput, true)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_gpu_scheduling() {
        let engine_handle: Arc<dyn Engine> = Arc::new(MockEngine::new());
        let scheduler = Scheduler::new(vec![engine_handle], 2, 1, 1000, true, 1, 0, None);

        let result = scheduler
            .infer(make_request(), Priority::LatencyCritical, false)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_fallback() {
        let engine_handle: Arc<dyn Engine> = Arc::new(MockEngine::with_delay(50));
        let scheduler = Scheduler::new(vec![engine_handle], 2, 1, 1000, true, 1, 0, None);

        let result = scheduler
            .infer(make_request(), Priority::Throughput, false)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cpu_queue_depth_tracking() {
        // Use an engine with delay to observe queue depth
        let engine_handle: Arc<dyn Engine> = Arc::new(MockEngine::with_delay(100));
        let scheduler = Scheduler::new(vec![engine_handle], 2, 1, 1000, true, 1, 0, None);

        // Initially 0
        let (cpu, _gpu) = scheduler.get_queue_depth();
        assert_eq!(cpu, 0);

        // Start an inference in the background
        let scheduler_clone = Arc::new(scheduler);
        let s2 = scheduler_clone.clone();
        let handle =
            tokio::spawn(async move { s2.infer(make_request(), Priority::Throughput, true).await });

        // Give it a moment to start
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Should be 1
        let (cpu, _gpu) = scheduler_clone.get_queue_depth();
        assert_eq!(cpu, 1);

        // Wait for it to finish
        let _ = handle.await.unwrap();

        // Should be 0 again
        let (cpu, _gpu) = scheduler_clone.get_queue_depth();
        assert_eq!(cpu, 0);
    }
}
