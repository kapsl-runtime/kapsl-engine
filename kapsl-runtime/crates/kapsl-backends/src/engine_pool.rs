use kapsl_engine_api::{Engine, EngineError};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

type EngineCache = Arc<Mutex<LruCache<(u32, usize), Arc<dyn Engine>>>>;
type EvictionCallback = Arc<dyn Fn(u32, usize, Arc<dyn Engine>) + Send + Sync>;
type EvictionCallbackSlot = Arc<Mutex<Option<EvictionCallback>>>;

#[derive(Debug, Clone, Copy)]
pub struct PoolMetrics {
    pub hit_rate: f64,
    pub hit: u64,
    pub evictions: u64, // Total number of evictions
    pub failure: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnginePoolConfig {
    /// Maximum number of engines to keep in the pool
    #[serde(default = "default_max_size")]
    pub max_size: usize,

    /// Minimum number of engines to keep in the pool
    #[serde(default = "default_min_size")]
    pub min_size: usize,

    /// Time-to-live for engines in the pool
    #[serde(default = "default_ttl")]
    pub ttl: Duration,

    /// Health check interval
    #[serde(default = "default_health_check_interval")]
    pub health_check_interval: Duration,

    // NEW: Warmup configuration
    #[serde(default)]
    pub warmup_enabled: bool,

    #[serde(default)]
    pub warmup_size: usize, // How many engines to pre-create (usually = min_size)
}

fn default_max_size() -> usize {
    5
}

fn default_min_size() -> usize {
    1
}

fn default_ttl() -> Duration {
    Duration::from_secs(60)
}

fn default_health_check_interval() -> Duration {
    Duration::from_secs(10)
}

impl Default for EnginePoolConfig {
    fn default() -> Self {
        Self {
            max_size: default_max_size(),
            min_size: default_min_size(),
            ttl: default_ttl(),
            health_check_interval: default_health_check_interval(),
            warmup_enabled: true,
            warmup_size: default_min_size(),
        }
    }
}

/// A pool for reusing backend engine instances
#[derive(Clone)]
pub struct EnginePool {
    config: EnginePoolConfig,
    metrics: Arc<Mutex<PoolMetrics>>,
    // LRU cache mapping (model_id, device_id) -> Engine
    // We use a tuple key to distinguish instances on different devices
    cache: EngineCache,
    // Optional eviction callback called when an engine is evicted from the pool.
    // Signature: (model_id, device_id, evicted_engine)
    eviction_callback: EvictionCallbackSlot,
}

impl EnginePool {
    pub fn new(config: EnginePoolConfig) -> Self {
        let capacity = NonZeroUsize::new(config.max_size).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            config,
            cache: Arc::new(Mutex::new(LruCache::new(capacity))),
            metrics: Arc::new(Mutex::new(PoolMetrics {
                hit_rate: 0.0,
                hit: 0,
                evictions: 0,
                failure: 0,
            })),
            eviction_callback: Arc::new(Mutex::new(None)),
        }
    }

    /// Set an eviction callback that will be invoked whenever an engine is evicted.
    /// The callback receives (model_id, device_id, evicted_engine).
    pub async fn set_eviction_callback<F>(&self, cb: F)
    where
        F: Fn(u32, usize, Arc<dyn Engine>) + Send + Sync + 'static,
    {
        let mut guard = self.eviction_callback.lock().await;
        *guard = Some(Arc::new(cb));
    }

    /// Clear any previously set eviction callback.
    pub async fn clear_eviction_callback(&self) {
        let mut guard = self.eviction_callback.lock().await;
        *guard = None;
    }

    pub fn start_health_check_task(&self) -> tokio::task::JoinHandle<()> {
        let pool = self.clone(); // Clone the Arc references
        let interval = self.config.health_check_interval;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);

            loop {
                ticker.tick().await;
                log::debug!("Running background health checks...");

                // Get all keys in the cache
                let keys: Vec<(u32, usize)> = {
                    let cache = pool.cache.lock().await;
                    cache.iter().map(|(k, _)| *k).collect()
                };

                // Check each engine's health
                for (model_id, device_id) in keys {
                    if let Some(_engine) = pool.get(model_id, device_id).await {
                        // get() already does health check, so this removes unhealthy ones
                        log::trace!("Engine ({}, {}) is healthy", model_id, device_id);
                    }
                }
            }
        })
    }

    pub fn max_size(&self) -> usize {
        self.config.max_size
    }

    pub fn min_size(&self) -> usize {
        self.config.min_size
    }

    pub fn ttl(&self) -> Duration {
        self.config.ttl
    }

    pub fn health_check_interval(&self) -> Duration {
        self.config.health_check_interval
    }

    /// Get an existing engine from the pool if available
    /// Returns None if engine doesn't exist or fails health check
    pub async fn get(&self, model_id: u32, device_id: usize) -> Option<Arc<dyn Engine>> {
        let mut cache = self.cache.lock().await;

        if let Some(engine) = cache.get(&(model_id, device_id)) {
            // Perform health check before returning
            match engine.health_check() {
                Ok(()) => {
                    // Engine is healthy, return it
                    self.metrics.lock().await.hit += 1;
                    Some(engine.clone())
                }
                Err(e) => {
                    // Engine failed health check, remove from pool
                    log::warn!(
                        "Engine (model_id={}, device_id={}) failed health check: {}. Removing from pool.",
                        model_id,
                        device_id,
                        e
                    );
                    self.metrics.lock().await.failure += 1;
                    cache.pop(&(model_id, device_id));
                    None
                }
            }
        } else {
            None
        }
    }

    /// Add an engine to the pool
    ///
    /// If adding this engine causes an eviction, the eviction callback (if set) will be
    /// invoked asynchronously in a separate task to avoid blocking pool operations.
    pub async fn put(&self, model_id: u32, device_id: usize, engine: Arc<dyn Engine>) {
        // First, handle cache update and get evicted engine (if any)
        let evicted_entry = {
            let mut cache = self.cache.lock().await;
            cache.push((model_id, device_id), engine)
        }; // cache lock released here

        if let Some((evicted_key, evicted_engine)) = evicted_entry {
            let (evicted_model_id, evicted_device_id) = evicted_key;
            // Update metrics without holding cache lock
            {
                let mut metrics = self.metrics.lock().await;
                metrics.evictions += 1;
                log::info!(
                    "Engine evicted from pool for model_id={}, device_id={}. Evictions total={}",
                    evicted_model_id,
                    evicted_device_id,
                    metrics.evictions
                );
            } // metrics lock released here

            // Invoke callback without holding any locks to prevent deadlocks
            let cb_opt = self.eviction_callback.lock().await.clone();
            if let Some(cb) = cb_opt {
                // Spawn callback in separate task to avoid blocking pool operations
                tokio::spawn(async move {
                    (cb)(evicted_model_id, evicted_device_id, evicted_engine);
                });
            }
        }
    }

    /// Remove an engine from the pool (e.g. on error or unload)
    pub async fn remove(&self, model_id: u32, device_id: usize) {
        let mut cache = self.cache.lock().await;
        cache.pop(&(model_id, device_id));
    }

    pub async fn len(&self) -> usize {
        let cache = self.cache.lock().await;
        cache.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.cache.lock().await.is_empty()
    }

    /// Warm up the pool by pre-creating engines
    ///
    /// `engine_factory` is an async function that creates an engine for given (model_id, device_id)
    pub async fn warmup<F, Fut>(
        &self,
        engine_configs: Vec<(u32, usize)>, // Vec of (model_id, device_id) pairs
        engine_factory: F,
    ) -> Result<(), EngineError>
    where
        F: Fn(u32, usize) -> Fut,
        Fut: std::future::Future<Output = Result<Arc<dyn Engine>, EngineError>>,
    {
        log::info!("Starting pool warmup with {} engines", engine_configs.len());

        for (model_id, device_id) in engine_configs {
            match engine_factory(model_id, device_id).await {
                Ok(engine) => {
                    self.put(model_id, device_id, engine).await;
                    log::info!(
                        "Warmed up engine for model_id={}, device_id={}",
                        model_id,
                        device_id
                    );
                }
                Err(e) => {
                    log::warn!(
                        "Failed to warm up engine for model_id={}, device_id={}: {}",
                        model_id,
                        device_id,
                        e
                    );
                    // Continue warming up other engines even if one fails
                }
            }
        }

        log::info!("Pool warmup complete. Pool size: {}", self.len().await);
        Ok(())
    }

    pub async fn pool_metrics(&self) -> PoolMetrics {
        let mut metrics = self.metrics.lock().await;
        metrics.hit_rate = (metrics.hit as f64) / (metrics.hit + metrics.failure) as f64;
        *metrics
    }
}

#[cfg(test)]
#[path = "engine_pool_tests.rs"]
mod engine_pool_tests;
