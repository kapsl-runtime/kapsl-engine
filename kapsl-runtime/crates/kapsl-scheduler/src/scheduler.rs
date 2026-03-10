use crate::gpu_executor::{GpuExecutor, WorkQueue};
use crate::mesh_routing::{MeshRouter, MeshRouterStats};
use crate::priority::Priority;
use crate::request::Request;
use kapsl_engine_api::{
    BinaryTensorPacket, EngineError, EngineHandle, EngineModelInfo, InferenceRequest,
};
use std::sync::atomic::AtomicUsize;
use tokio::sync::oneshot;

use kapsl_hal::device_mesh::DeviceMesh;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueOverflowPolicy {
    Block,
    DropNewest,
    DropOldest,
}

impl QueueOverflowPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::DropNewest => "drop_newest",
            Self::DropOldest => "drop_oldest",
        }
    }
}

/// Main scheduler that coordinates CPU and GPU execution
pub struct Scheduler {
    engines: Vec<EngineHandle>,
    cpu_pool: rayon::ThreadPool,
    // Vector of channels, one pair per worker
    gpu_high_priority_queues: Vec<WorkQueue>,
    gpu_low_priority_queues: Vec<WorkQueue>,
    _enable_fallback: bool,
    // Track active CPU inferences
    cpu_active_count: Arc<std::sync::atomic::AtomicUsize>,
    // Track in-flight GPU work (requests already dequeued from the channels, but not finished).
    gpu_in_flight_count: Arc<AtomicUsize>,
    // Device mesh for distributed inference
    device_mesh: Option<Arc<DeviceMesh>>,
    // Mesh router for topology-aware routing
    router: MeshRouter,
    max_micro_batch: usize,
    queue_overflow_policy: QueueOverflowPolicy,
}

impl Scheduler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engines: Vec<EngineHandle>,
        cpu_workers: usize,
        workers_per_device: usize,
        queue_size: usize,
        enable_fallback: bool,
        max_micro_batch: usize,
        queue_delay_ms: u64,
        device_mesh: Option<Arc<DeviceMesh>>,
    ) -> Self {
        // Create CPU thread pool
        let cpu_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cpu_workers)
            .build()
            .expect("Failed to create CPU thread pool");

        let num_devices = engines.len();
        let total_workers = num_devices * workers_per_device;

        let mut gpu_high_priority_queues = Vec::with_capacity(total_workers);
        let mut gpu_low_priority_queues = Vec::with_capacity(total_workers);
        let gpu_in_flight_count = Arc::new(AtomicUsize::new(0));

        for engine in &engines {
            for _ in 0..workers_per_device {
                // Create GPU executor channels for this worker
                let high_queue = WorkQueue::new(queue_size);
                let low_queue = WorkQueue::new(queue_size);

                gpu_high_priority_queues.push(high_queue.clone());
                gpu_low_priority_queues.push(low_queue.clone());

                // Spawn GPU executor for this worker
                let gpu_executor = GpuExecutor::new(
                    high_queue,
                    low_queue,
                    engine.clone(),
                    max_micro_batch,
                    queue_delay_ms,
                    gpu_in_flight_count.clone(),
                );
                tokio::spawn(gpu_executor.run());
            }
        }

        // Create mesh router for topology-aware routing
        let router = MeshRouter::new(device_mesh.clone(), total_workers);

        Self {
            engines,
            cpu_pool,
            gpu_high_priority_queues,
            gpu_low_priority_queues,
            _enable_fallback: enable_fallback,
            cpu_active_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            gpu_in_flight_count,
            device_mesh,
            router,
            max_micro_batch,
            queue_overflow_policy: QueueOverflowPolicy::Block,
        }
    }

    pub fn with_queue_overflow_policy(mut self, policy: QueueOverflowPolicy) -> Self {
        self.queue_overflow_policy = policy;
        self
    }

    pub fn queue_overflow_policy(&self) -> QueueOverflowPolicy {
        self.queue_overflow_policy
    }

    /// Get worker index using topology-aware routing
    fn get_worker_index(&self, session_id: &Option<String>) -> usize {
        self.router.route(session_id, None)
    }

    /// Get worker index with TP group hint for tensor parallelism
    #[allow(dead_code)]
    fn get_worker_index_with_hint(
        &self,
        session_id: &Option<String>,
        tp_group_hint: Option<usize>,
    ) -> usize {
        self.router.route(session_id, tp_group_hint)
    }

    /// Get mesh routing statistics
    pub fn mesh_stats(&self) -> Option<MeshRouterStats> {
        self.router.mesh_stats()
    }

    /// Get the device mesh if available
    pub fn device_mesh(&self) -> Option<Arc<DeviceMesh>> {
        self.device_mesh.clone()
    }

    pub async fn infer(
        &self,
        input: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        let (response_tx, response_rx) = oneshot::channel();

        // Determine worker index before moving input
        let worker_idx = if !force_cpu {
            self.get_worker_index(&input.session_id)
        } else {
            0
        };

        let request = Request { input, response_tx };

        if force_cpu {
            // Execute on CPU thread pool
            let engine_idx = self.get_worker_index(&None) % self.engines.len();
            let engine = self.engines[engine_idx].clone();
            let cpu_input = request.input.clone();
            let cpu_response_tx = request.response_tx;

            self.cpu_active_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let cpu_active_count = self.cpu_active_count.clone();

            self.cpu_pool.spawn(move || {
                let result = engine.infer(&cpu_input);
                let _ = cpu_response_tx.send(result);
                cpu_active_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            });

            response_rx
                .await
                .map_err(|_| EngineError::overloaded("Scheduler dropped request".to_string()))?
        } else {
            // Execute on GPU via priority queue
            let queue = match priority {
                Priority::LatencyCritical => &self.gpu_high_priority_queues[worker_idx],
                Priority::Throughput => &self.gpu_low_priority_queues[worker_idx],
            };

            match self.queue_overflow_policy {
                QueueOverflowPolicy::Block => {
                    queue.push_block(request).await;
                }
                QueueOverflowPolicy::DropNewest => {
                    if queue.try_push_drop_newest(request).is_err() {
                        return Err(EngineError::overloaded(format!(
                            "GPU queue full (policy={})",
                            self.queue_overflow_policy.as_str()
                        )));
                    }
                }
                QueueOverflowPolicy::DropOldest => {
                    if let Some(dropped) = queue.push_drop_oldest(request) {
                        let _ = dropped.response_tx.send(Err(EngineError::overloaded(
                            "GPU queue full: dropped oldest request (policy=drop_oldest)"
                                .to_string(),
                        )));
                    }
                }
            }

            response_rx.await.map_err(|_| EngineError::InferenceError {
                reason: "GPU execution failed".to_string(),
                source: None,
            })?
        }
    }

    pub fn get_queue_depth(&self) -> (usize, usize) {
        let cpu_depth = self
            .cpu_active_count
            .load(std::sync::atomic::Ordering::Relaxed);

        let mut gpu_total = 0;
        for (high_queue, low_queue) in self
            .gpu_high_priority_queues
            .iter()
            .zip(self.gpu_low_priority_queues.iter())
        {
            gpu_total += high_queue.len();
            gpu_total += low_queue.len();
        }
        gpu_total += self
            .gpu_in_flight_count
            .load(std::sync::atomic::Ordering::Relaxed);
        (cpu_depth, gpu_total)
    }

    pub fn is_healthy(&self) -> bool {
        for (high_queue, low_queue) in self
            .gpu_high_priority_queues
            .iter()
            .zip(self.gpu_low_priority_queues.iter())
        {
            let high_capacity = high_queue.capacity();
            let low_capacity = low_queue.capacity();
            let high_depth = high_queue.len();
            let low_depth = low_queue.len();

            if high_capacity > 0
                && low_capacity > 0
                && ((high_depth as f64 / high_capacity as f64) >= 0.8
                    || (low_depth as f64 / low_capacity as f64) >= 0.8)
            {
                return false;
            }
        }
        true
    }
}

#[async_trait::async_trait]
impl crate::replica_pool::ReplicaScheduler for Scheduler {
    fn get_queue_depth(&self) -> (usize, usize) {
        self.get_queue_depth()
    }

    fn is_healthy(&self) -> bool {
        self.is_healthy()
    }

    fn get_metrics(&self) -> kapsl_engine_api::EngineMetrics {
        let mut total_memory = 0;
        let mut total_gpu_util = 0.0;
        let mut total_throughput = 0.0;
        let mut total_kv_bytes_used = 0;
        let mut total_kv_bytes_capacity = 0;
        let mut total_kv_blocks_total = 0;
        let mut total_kv_blocks_free = 0;
        let mut total_kv_sequences = 0;
        let mut total_kv_evicted_blocks = 0;
        let mut total_kv_evicted_sequences = 0;
        let mut total_kv_packed_layers = 0;
        let count = self.engines.len();

        for engine in &self.engines {
            let m = engine.metrics();
            total_memory += m.memory_usage;
            total_gpu_util += m.gpu_utilization;
            total_throughput += m.throughput;
            total_kv_bytes_used += m.kv_cache_bytes_used;
            total_kv_bytes_capacity += m.kv_cache_bytes_capacity;
            total_kv_blocks_total += m.kv_cache_blocks_total;
            total_kv_blocks_free += m.kv_cache_blocks_free;
            total_kv_sequences += m.kv_cache_sequences;
            total_kv_evicted_blocks += m.kv_cache_evicted_blocks;
            total_kv_evicted_sequences += m.kv_cache_evicted_sequences;
            total_kv_packed_layers += m.kv_cache_packed_layers;
        }

        let (cpu_q, gpu_q) = self.get_queue_depth();

        kapsl_engine_api::EngineMetrics {
            memory_usage: total_memory,
            gpu_utilization: if count > 0 {
                total_gpu_util / count as f64
            } else {
                0.0
            },
            throughput: total_throughput,
            batch_size: self.max_micro_batch,
            queue_depth: cpu_q + gpu_q,
            kv_cache_bytes_used: total_kv_bytes_used,
            kv_cache_bytes_capacity: total_kv_bytes_capacity,
            kv_cache_blocks_total: total_kv_blocks_total,
            kv_cache_blocks_free: total_kv_blocks_free,
            kv_cache_sequences: total_kv_sequences,
            kv_cache_evicted_blocks: total_kv_evicted_blocks,
            kv_cache_evicted_sequences: total_kv_evicted_sequences,
            kv_cache_packed_layers: total_kv_packed_layers,
            ..kapsl_engine_api::EngineMetrics::default()
        }
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        self.engines.iter().find_map(|engine| engine.model_info())
    }

    async fn infer(
        &self,
        request: &InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        self.infer(request.clone(), priority, force_cpu).await
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
        _priority: Priority,
        force_cpu: bool,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
        >,
        EngineError,
    > {
        if !self.is_healthy() {
            return Err(EngineError::overloaded("Scheduler overloaded".to_string()));
        }

        let worker_idx = if !force_cpu {
            self.get_worker_index(&request.session_id)
        } else {
            0
        };

        let engine_idx = worker_idx % self.engines.len();
        let engine = self.engines[engine_idx].clone();

        let stream = engine.infer_stream(&request);
        Ok(stream)
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod scheduler_tests;
