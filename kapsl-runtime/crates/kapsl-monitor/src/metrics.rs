use prometheus::{HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct KapslMetrics {
    pub registry: Arc<Registry>,
    pub inference_latency: HistogramVec,
    pub inference_count: IntCounterVec,
    pub active_inferences: IntGaugeVec,
    pub batch_size_hist: HistogramVec,
    pub queue_wait_hist: HistogramVec,
    // Pool-specific metrics
    pub pool_active_replicas: IntGaugeVec,
    pub pool_queue_depth_high: IntGaugeVec,
    pub pool_queue_depth_low: IntGaugeVec,
    pub pool_healthy_replicas: IntGaugeVec,
    // KV cache metrics
    pub kv_cache_bytes_used: IntGaugeVec,
    pub kv_cache_bytes_capacity: IntGaugeVec,
    pub kv_cache_blocks_total: IntGaugeVec,
    pub kv_cache_blocks_free: IntGaugeVec,
    pub kv_cache_sequences: IntGaugeVec,
    pub kv_cache_evicted_blocks: IntGaugeVec,
    pub kv_cache_evicted_sequences: IntGaugeVec,
    pub kv_cache_packed_layers: IntGaugeVec,
}

impl KapslMetrics {
    pub fn new(registry: &Arc<Registry>) -> Self {
        let inference_latency = HistogramVec::new(
            HistogramOpts::new(
                "kapsl_inference_latency_seconds",
                "Inference latency (seconds)",
            )
            .buckets(vec![
                0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
            ]),
            &["model", "version", "status"],
        )
        .unwrap();

        let inference_count = IntCounterVec::new(
            Opts::new("kapsl_inference_total", "Number of inferences"),
            &["model", "status"],
        )
        .unwrap();

        let active_inferences = IntGaugeVec::new(
            Opts::new("kapsl_active_inferences", "Active running inferences"),
            &["model"],
        )
        .unwrap();

        let batch_size_hist = HistogramVec::new(
            HistogramOpts::new("kapsl_batch_size", "Observed batch sizes"),
            &["model"],
        )
        .unwrap();

        let queue_wait_hist = HistogramVec::new(
            HistogramOpts::new("kapsl_queue_wait_seconds", "Scheduler queue wait time"),
            &["model"],
        )
        .unwrap();

        // Pool-specific metrics
        let pool_active_replicas = IntGaugeVec::new(
            Opts::new(
                "kapsl_pool_active_replicas",
                "Number of active replicas in the pool",
            ),
            &["model"],
        )
        .unwrap();

        let pool_queue_depth_high = IntGaugeVec::new(
            Opts::new(
                "kapsl_pool_queue_depth_high",
                "High priority queue depth across all replicas",
            ),
            &["model"],
        )
        .unwrap();

        let pool_queue_depth_low = IntGaugeVec::new(
            Opts::new(
                "kapsl_pool_queue_depth_low",
                "Low priority queue depth across all replicas",
            ),
            &["model"],
        )
        .unwrap();

        let pool_healthy_replicas = IntGaugeVec::new(
            Opts::new(
                "kapsl_pool_healthy_replicas",
                "Number of healthy replicas in the pool",
            ),
            &["model"],
        )
        .unwrap();

        let kv_cache_bytes_used = IntGaugeVec::new(
            Opts::new("kapsl_kv_cache_bytes_used", "KV cache bytes currently used"),
            &["model"],
        )
        .unwrap();
        let kv_cache_bytes_capacity = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_cache_bytes_capacity",
                "KV cache total byte capacity",
            ),
            &["model"],
        )
        .unwrap();
        let kv_cache_blocks_total = IntGaugeVec::new(
            Opts::new("kapsl_kv_cache_blocks_total", "KV cache total blocks"),
            &["model"],
        )
        .unwrap();
        let kv_cache_blocks_free = IntGaugeVec::new(
            Opts::new("kapsl_kv_cache_blocks_free", "KV cache free blocks"),
            &["model"],
        )
        .unwrap();
        let kv_cache_sequences = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_cache_sequences",
                "Active sequences tracked in KV cache",
            ),
            &["model"],
        )
        .unwrap();
        let kv_cache_evicted_blocks = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_cache_evicted_blocks_total",
                "Total evicted KV blocks",
            ),
            &["model"],
        )
        .unwrap();
        let kv_cache_evicted_sequences = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_cache_evicted_sequences_total",
                "Total evicted KV sequences",
            ),
            &["model"],
        )
        .unwrap();
        let kv_cache_packed_layers = IntGaugeVec::new(
            Opts::new(
                "kapsl_kv_cache_packed_layers",
                "Number of packed KV layers cached",
            ),
            &["model"],
        )
        .unwrap();

        registry
            .register(Box::new(inference_latency.clone()))
            .expect("Failed to register inference_latency");
        registry
            .register(Box::new(inference_count.clone()))
            .expect("Failed to register inference_count");
        registry
            .register(Box::new(active_inferences.clone()))
            .expect("Failed to register active_inferences");
        registry
            .register(Box::new(batch_size_hist.clone()))
            .expect("Failed to register batch_size_hist");
        registry
            .register(Box::new(queue_wait_hist.clone()))
            .expect("Failed to register queue_wait_hist");
        registry
            .register(Box::new(pool_active_replicas.clone()))
            .expect("Failed to register pool_active_replicas");
        registry
            .register(Box::new(pool_queue_depth_high.clone()))
            .expect("Failed to register pool_queue_depth_high");
        registry
            .register(Box::new(pool_queue_depth_low.clone()))
            .expect("Failed to register pool_queue_depth_low");
        registry
            .register(Box::new(pool_healthy_replicas.clone()))
            .expect("Failed to register pool_healthy_replicas");
        registry
            .register(Box::new(kv_cache_bytes_used.clone()))
            .expect("Failed to register kv_cache_bytes_used");
        registry
            .register(Box::new(kv_cache_bytes_capacity.clone()))
            .expect("Failed to register kv_cache_bytes_capacity");
        registry
            .register(Box::new(kv_cache_blocks_total.clone()))
            .expect("Failed to register kv_cache_blocks_total");
        registry
            .register(Box::new(kv_cache_blocks_free.clone()))
            .expect("Failed to register kv_cache_blocks_free");
        registry
            .register(Box::new(kv_cache_sequences.clone()))
            .expect("Failed to register kv_cache_sequences");
        registry
            .register(Box::new(kv_cache_evicted_blocks.clone()))
            .expect("Failed to register kv_cache_evicted_blocks");
        registry
            .register(Box::new(kv_cache_evicted_sequences.clone()))
            .expect("Failed to register kv_cache_evicted_sequences");
        registry
            .register(Box::new(kv_cache_packed_layers.clone()))
            .expect("Failed to register kv_cache_packed_layers");

        Self {
            registry: registry.clone(),
            inference_latency,
            inference_count,
            active_inferences,
            batch_size_hist,
            queue_wait_hist,
            pool_active_replicas,
            pool_queue_depth_high,
            pool_queue_depth_low,
            pool_healthy_replicas,
            kv_cache_bytes_used,
            kv_cache_bytes_capacity,
            kv_cache_blocks_total,
            kv_cache_blocks_free,
            kv_cache_sequences,
            kv_cache_evicted_blocks,
            kv_cache_evicted_sequences,
            kv_cache_packed_layers,
        }
    }
}

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod metrics_tests;
