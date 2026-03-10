use crate::priority::Priority;
use kapsl_engine_api::{BinaryTensorPacket, EngineError, InferenceRequest};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Strategy for selecting which replica to route requests to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolStrategy {
    /// Distribute requests evenly in round-robin fashion
    RoundRobin,
    /// Route to replica with lowest queue depth
    LeastLoaded,
    /// Sticky session routing based on session_id
    Sticky,
}

/// Statistics for a single replica
#[derive(Debug, Clone)]
pub struct ReplicaStats {
    pub replica_id: u32,
    pub requests_total: u64,
    pub queue_depth: (usize, usize),
    pub healthy: bool,
}

/// Pool of replicas for load balancing
pub struct ReplicaPool<T> {
    replicas: Arc<RwLock<Vec<PooledReplica<T>>>>,
    strategy: PoolStrategy,
    round_robin_counter: AtomicUsize,
}

struct PooledReplica<T> {
    replica_id: u32,
    scheduler: Arc<T>,
    requests_total: AtomicUsize,
}

impl<T> ReplicaPool<T>
where
    T: ReplicaScheduler + Send + Sync + 'static,
{
    /// Create a new replica pool with the specified strategy
    pub fn new(strategy: PoolStrategy) -> Self {
        Self {
            replicas: Arc::new(RwLock::new(Vec::new())),
            strategy,
            round_robin_counter: AtomicUsize::new(0),
        }
    }

    /// Add a replica to the pool
    pub async fn add_replica(&self, replica_id: u32, scheduler: Arc<T>) {
        let mut replicas = self.replicas.write().await;
        replicas.push(PooledReplica {
            replica_id,
            scheduler,
            requests_total: AtomicUsize::new(0),
        });
    }

    /// Remove a replica from the pool by replica_id
    pub async fn remove_replica(&self, replica_id: u32) -> bool {
        let mut replicas = self.replicas.write().await;
        if let Some(pos) = replicas.iter().position(|r| r.replica_id == replica_id) {
            replicas.remove(pos);
            true
        } else {
            false
        }
    }

    /// Get the number of replicas in the pool
    pub async fn size(&self) -> usize {
        self.replicas.read().await.len()
    }

    /// Get statistics about a specific replica
    pub async fn get_replica_stats(&self, replica_id: u32) -> Option<ReplicaStats> {
        let replicas = self.replicas.read().await;
        replicas
            .iter()
            .find(|r| r.replica_id == replica_id)
            .map(|r| ReplicaStats {
                replica_id: r.replica_id,
                requests_total: r.requests_total.load(Ordering::Relaxed) as u64,
                queue_depth: r.scheduler.get_queue_depth(),
                healthy: r.scheduler.is_healthy(),
            })
    }

    /// Get the total number of replicas in the pool
    pub fn get_replica_count(&self) -> usize {
        // Use blocking read since this is called from sync context
        // We can safely use tokio::runtime::Handle since we're always in a tokio context
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async { self.replicas.read().await.len() })
        })
    }

    /// Get the number of healthy replicas in the pool
    pub fn get_healthy_replica_count(&self) -> usize {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.replicas
                    .read()
                    .await
                    .iter()
                    .filter(|replica| replica.scheduler.is_healthy())
                    .count()
            })
        })
    }

    /// Get statistics for all replicas
    pub async fn stats(&self) -> Vec<ReplicaStats> {
        let replicas = self.replicas.read().await;
        let mut stats = Vec::new();
        for replica in replicas.iter() {
            stats.push(ReplicaStats {
                replica_id: replica.replica_id,
                requests_total: replica.requests_total.load(Ordering::Relaxed) as u64,
                queue_depth: replica.scheduler.get_queue_depth(),
                healthy: replica.scheduler.is_healthy(),
            });
        }
        stats
    }

    /// Execute a request on an appropriate replica
    pub async fn execute(
        &self,
        request: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        // Important: do not hold the pool lock across awaits (inferences can be long),
        // otherwise scale-up/down cannot add/remove replicas until all in-flight
        // requests finish.
        let (selected_replica_id, selected_scheduler, fallback_schedulers) = {
            let replicas = self.replicas.read().await;

            if replicas.is_empty() {
                return Err(EngineError::overloaded(
                    "No replicas available in pool".to_string(),
                ));
            }

            let selected_idx = match self.strategy {
                PoolStrategy::RoundRobin => self.select_round_robin(&replicas),
                PoolStrategy::LeastLoaded => self.select_least_loaded(&replicas),
                PoolStrategy::Sticky => self.select_sticky(&replicas, &request),
            };

            let selected = &replicas[selected_idx];
            selected.requests_total.fetch_add(1, Ordering::Relaxed);

            let mut fallbacks = Vec::new();
            if replicas.len() > 1 {
                for (idx, other) in replicas.iter().enumerate() {
                    if idx == selected_idx {
                        continue;
                    }
                    if other.scheduler.is_healthy() {
                        fallbacks.push((other.replica_id, other.scheduler.clone()));
                    }
                }
            }

            (selected.replica_id, selected.scheduler.clone(), fallbacks)
        };

        let result = selected_scheduler
            .infer(&request, priority, force_cpu)
            .await;

        if result.is_err() && !fallback_schedulers.is_empty() {
            log::warn!(
                "Request failed on replica {}, attempting failover",
                selected_replica_id
            );

            for (replica_id, scheduler) in fallback_schedulers {
                if !scheduler.is_healthy() {
                    continue;
                }
                log::info!("Failing over to replica {}", replica_id);
                if let Ok(response) = scheduler.infer(&request, priority, force_cpu).await {
                    if let Ok(replicas) = self.replicas.try_read() {
                        if let Some(replica) = replicas.iter().find(|r| r.replica_id == replica_id)
                        {
                            replica.requests_total.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    return Ok(response);
                }
            }
        }

        result
    }

    fn select_round_robin(&self, replicas: &[PooledReplica<T>]) -> usize {
        let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
        counter % replicas.len()
    }

    fn select_least_loaded(&self, replicas: &[PooledReplica<T>]) -> usize {
        let mut min_load = usize::MAX;
        let mut min_idx = 0;

        for (idx, replica) in replicas.iter().enumerate() {
            if !replica.scheduler.is_healthy() {
                continue;
            }

            let (high, low) = replica.scheduler.get_queue_depth();
            let total_depth = high + low;

            if total_depth < min_load {
                min_load = total_depth;
                min_idx = idx;
            }
        }

        min_idx
    }

    fn select_sticky(&self, replicas: &[PooledReplica<T>], request: &InferenceRequest) -> usize {
        // If request has session_id, use hash-based routing
        if let Some(ref session_id) = request.session_id {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            session_id.hash(&mut hasher);
            (hasher.finish() as usize) % replicas.len()
        } else {
            // No session_id, fall back to round-robin
            self.select_round_robin(replicas)
        }
    }
}

#[async_trait::async_trait]
impl<T> ReplicaScheduler for ReplicaPool<T>
where
    T: ReplicaScheduler + Send + Sync + 'static,
{
    fn get_queue_depth(&self) -> (usize, usize) {
        if let Ok(replicas) = self.replicas.try_read() {
            let mut total_high = 0;
            let mut total_low = 0;
            for replica in replicas.iter() {
                let (h, l) = replica.scheduler.get_queue_depth();
                total_high += h;
                total_low += l;
            }
            (total_high, total_low)
        } else {
            (0, 0)
        }
    }

    fn is_healthy(&self) -> bool {
        if let Ok(replicas) = self.replicas.try_read() {
            if replicas.is_empty() {
                return true;
            }
            replicas.iter().any(|r| r.scheduler.is_healthy())
        } else {
            true
        }
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
        let mut cpu_q = 0;
        let mut gpu_q = 0;
        let mut count = 0;

        if let Ok(replicas) = self.replicas.try_read() {
            count = replicas.len();
            for replica in replicas.iter() {
                let m = replica.scheduler.get_metrics();
                let (cq, gq) = replica.scheduler.get_queue_depth();
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
                cpu_q += cq;
                gpu_q += gq;
            }
        }

        kapsl_engine_api::EngineMetrics {
            memory_usage: total_memory,
            gpu_utilization: if count > 0 {
                total_gpu_util / count as f64
            } else {
                0.0
            },
            throughput: total_throughput,
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

    fn model_info(&self) -> Option<kapsl_engine_api::EngineModelInfo> {
        if let Ok(replicas) = self.replicas.try_read() {
            for replica in replicas.iter() {
                if let Some(model_info) = replica.scheduler.model_info() {
                    return Some(model_info);
                }
            }
        }
        None
    }

    async fn infer(
        &self,
        request: &InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        self.execute(request.clone(), priority, force_cpu).await
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
        >,
        EngineError,
    > {
        // Select the schedulers to attempt without holding the pool lock across awaits.
        let (selected_replica_id, selected_scheduler, fallback_schedulers) = {
            let replicas = self.replicas.read().await;

            if replicas.is_empty() {
                return Err(EngineError::overloaded(
                    "No replicas available in pool".to_string(),
                ));
            }

            let selected_idx = match self.strategy {
                PoolStrategy::RoundRobin => self.select_round_robin(&replicas),
                PoolStrategy::LeastLoaded => self.select_least_loaded(&replicas),
                PoolStrategy::Sticky => self.select_sticky(&replicas, &request),
            };

            let selected = &replicas[selected_idx];
            selected.requests_total.fetch_add(1, Ordering::Relaxed);

            let mut fallbacks = Vec::new();
            if replicas.len() > 1 {
                for (idx, other) in replicas.iter().enumerate() {
                    if idx == selected_idx {
                        continue;
                    }
                    if other.scheduler.is_healthy() {
                        fallbacks.push((other.replica_id, other.scheduler.clone()));
                    }
                }
            }

            (selected.replica_id, selected.scheduler.clone(), fallbacks)
        };

        if selected_scheduler.is_healthy() {
            match selected_scheduler
                .infer_stream(request.clone(), priority, force_cpu)
                .await
            {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    log::warn!(
                        "Streaming request failed on replica {}: {}, attempting failover",
                        selected_replica_id,
                        e
                    );
                }
            }
        }

        // Failover to other healthy replicas
        if !fallback_schedulers.is_empty() {
            for (replica_id, scheduler) in fallback_schedulers {
                if !scheduler.is_healthy() {
                    continue;
                }
                log::info!("Failing over streaming request to replica {}", replica_id);
                match scheduler
                    .infer_stream(request.clone(), priority, force_cpu)
                    .await
                {
                    Ok(stream) => {
                        if let Ok(replicas) = self.replicas.try_read() {
                            if let Some(replica) =
                                replicas.iter().find(|r| r.replica_id == replica_id)
                            {
                                replica.requests_total.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        return Ok(stream);
                    }
                    Err(_) => continue,
                }
            }
        }

        // If we got here, all attempts failed
        Err(EngineError::overloaded(
            "All replicas failed or overloaded".to_string(),
        ))
    }
}

/// Trait that replica schedulers must implement
#[async_trait::async_trait]
pub trait ReplicaScheduler: Send + Sync {
    fn get_queue_depth(&self) -> (usize, usize);
    fn is_healthy(&self) -> bool;
    fn get_metrics(&self) -> kapsl_engine_api::EngineMetrics;
    fn model_info(&self) -> Option<kapsl_engine_api::EngineModelInfo> {
        None
    }
    async fn infer(
        &self,
        request: &InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError>;

    async fn infer_stream(
        &self,
        request: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
        >,
        EngineError,
    >;
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use kapsl_engine_api::BinaryTensorPacket;
    use kapsl_engine_api::TensorDtype;

    struct MockScheduler {
        queue_depth: (usize, usize),
        healthy: bool,
    }

    impl MockScheduler {
        fn new(queue_depth: (usize, usize), healthy: bool) -> Self {
            Self {
                queue_depth,
                healthy,
            }
        }
    }

    #[async_trait::async_trait]
    impl ReplicaScheduler for MockScheduler {
        fn get_queue_depth(&self) -> (usize, usize) {
            self.queue_depth
        }

        fn is_healthy(&self) -> bool {
            self.healthy
        }

        async fn infer(
            &self,
            request: &InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<BinaryTensorPacket, EngineError> {
            if !self.healthy {
                return Err(EngineError::InferenceError {
                    reason: "Unhealthy replica".to_string(),
                    source: None,
                });
            }
            Ok(request.input.clone())
        }

        fn get_metrics(&self) -> kapsl_engine_api::EngineMetrics {
            kapsl_engine_api::EngineMetrics {
                queue_depth: self.queue_depth.0 + self.queue_depth.1,
                ..kapsl_engine_api::EngineMetrics::default()
            }
        }

        async fn infer_stream(
            &self,
            request: InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<
            std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
            >,
            EngineError,
        > {
            if !self.healthy {
                return Err(EngineError::InferenceError {
                    reason: "Unhealthy replica".to_string(),
                    source: None,
                });
            }
            let result = Ok(request.input.clone());
            Ok(Box::pin(futures::stream::once(async move { result })))
        }
    }

    #[tokio::test]
    async fn test_round_robin_distribution() {
        let pool = ReplicaPool::new(PoolStrategy::RoundRobin);

        // Add 3 replicas
        for i in 0..3 {
            pool.add_replica(i, Arc::new(MockScheduler::new((0, 0), true)))
                .await;
        }

        let request = InferenceRequest::new(BinaryTensorPacket {
            shape: vec![1, 1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 0, 0],
        });

        // Execute 9 requests
        for _ in 0..9 {
            let _ = pool
                .execute(request.clone(), Priority::Throughput, false)
                .await;
        }

        // Verify distribution is even (each should have 3 requests)
        let stats = pool.stats().await;
        for stat in stats {
            assert_eq!(stat.requests_total, 3);
        }
    }

    #[tokio::test]
    async fn test_least_loaded_selection() {
        let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);

        // Add replicas with different queue depths
        pool.add_replica(0, Arc::new(MockScheduler::new((10, 5), true)))
            .await;
        pool.add_replica(1, Arc::new(MockScheduler::new((2, 1), true)))
            .await;
        pool.add_replica(2, Arc::new(MockScheduler::new((5, 3), true)))
            .await;

        let request = InferenceRequest::new(BinaryTensorPacket {
            shape: vec![1, 1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 0, 0],
        });

        // Execute request - should go to replica 1 (lowest queue depth of 3)
        let _ = pool.execute(request, Priority::Throughput, false).await;

        let stats = pool.stats().await;
        // Replica 1 should have received the request
        assert_eq!(stats[1].requests_total, 1);
        assert_eq!(stats[0].requests_total, 0);
        assert_eq!(stats[2].requests_total, 0);
    }

    #[tokio::test]
    async fn test_sticky_routing() {
        let pool = ReplicaPool::new(PoolStrategy::Sticky);

        // Add 3 replicas
        for i in 0..3 {
            pool.add_replica(i, Arc::new(MockScheduler::new((0, 0), true)))
                .await;
        }

        let request = InferenceRequest::new(BinaryTensorPacket {
            shape: vec![1, 1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 0, 0],
        })
        .with_session_id("session123");

        // Execute same session multiple times
        for _ in 0..5 {
            let _ = pool
                .execute(request.clone(), Priority::Throughput, false)
                .await;
        }

        let stats = pool.stats().await;
        // All requests should go to the same replica (whichever the hash maps to)
        let total_requests: u64 = stats.iter().map(|s| s.requests_total).sum();
        assert_eq!(total_requests, 5);

        // One replica should have all 5 requests
        assert!(stats.iter().any(|s| s.requests_total == 5));
    }

    #[tokio::test]
    async fn test_failover() {
        let pool = ReplicaPool::new(PoolStrategy::RoundRobin);

        // Add unhealthy and healthy replicas
        pool.add_replica(0, Arc::new(MockScheduler::new((0, 0), false)))
            .await;
        pool.add_replica(1, Arc::new(MockScheduler::new((0, 0), true)))
            .await;

        let request = InferenceRequest::new(BinaryTensorPacket {
            shape: vec![1, 1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 0, 0],
        });

        // First request goes to replica 0 (unhealthy), should failover to replica 1
        let result = pool.execute(request, Priority::Throughput, false).await;
        assert!(result.is_ok());

        let stats = pool.stats().await;
        // Replica 1 should have received the failover request
        assert_eq!(stats[1].requests_total, 1);
    }

    #[tokio::test]
    async fn test_streaming_failover() {
        let pool = ReplicaPool::new(PoolStrategy::RoundRobin);

        // Add unhealthy and healthy replicas
        pool.add_replica(0, Arc::new(MockScheduler::new((0, 0), false)))
            .await;
        pool.add_replica(1, Arc::new(MockScheduler::new((0, 0), true)))
            .await;

        let request = InferenceRequest::new(BinaryTensorPacket {
            shape: vec![1, 1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 0, 0],
        });

        // First attempt should pick replica 0 (Round Robin) and failover to replica 1
        let result = pool
            .infer_stream(request.clone(), Priority::LatencyCritical, false)
            .await;

        assert!(
            result.is_ok(),
            "Streaming request should succeed via failover"
        );
        let mut stream = result.unwrap();

        // Consume stream to verify it works
        let item = stream.next().await;
        assert!(item.is_some());
        assert!(item.unwrap().is_ok());

        let stats = pool.stats().await;
        // Replica 1 should have received the request (Replica 0 failed)
        assert!(stats[1].requests_total >= 1);
    }

    #[tokio::test]
    async fn test_queue_depth_aggregation() {
        let pool = ReplicaPool::new(PoolStrategy::RoundRobin);

        // Add replicas with different queue depths
        pool.add_replica(0, Arc::new(MockScheduler::new((10, 5), true)))
            .await;
        pool.add_replica(1, Arc::new(MockScheduler::new((2, 1), true)))
            .await;
        pool.add_replica(2, Arc::new(MockScheduler::new((5, 3), true)))
            .await;

        // Total high: 10 + 2 + 5 = 17
        // Total low: 5 + 1 + 3 = 9
        let (high, low) = pool.get_queue_depth();
        assert_eq!(high, 17);
        assert_eq!(low, 9);
    }
}
