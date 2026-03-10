use super::*;
use async_trait::async_trait;
use kapsl_engine_api::{BinaryTensorPacket, EngineError, InferenceRequest, TensorDtype};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc as StdArc, Mutex as StdMutex};

struct MockEngine;

#[async_trait]
impl Engine for MockEngine {
    async fn load(&mut self, _path: &std::path::Path) -> Result<(), EngineError> {
        Ok(())
    }
    fn infer(&self, _req: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        Ok(BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: vec![0; 4],
        })
    }
    fn infer_stream(
        &self,
        req: &InferenceRequest,
    ) -> std::pin::Pin<
        Box<dyn futures::stream::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
    > {
        let result = self.infer(req);
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

struct FailingHealthEngine;

#[async_trait]
impl Engine for FailingHealthEngine {
    async fn load(&mut self, _path: &std::path::Path) -> Result<(), EngineError> {
        Ok(())
    }
    fn infer(&self, _req: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        Ok(BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: vec![0; 4],
        })
    }
    fn infer_stream(
        &self,
        req: &InferenceRequest,
    ) -> std::pin::Pin<
        Box<dyn futures::stream::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
    > {
        let result = self.infer(req);
        Box::pin(futures::stream::once(async move { result }))
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> kapsl_engine_api::EngineMetrics {
        kapsl_engine_api::EngineMetrics::default()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Err(EngineError::Backend {
            message: "unhealthy".to_string(),
            source: None,
        })
    }
}

#[tokio::test]
async fn test_engine_pool_lru() {
    let config = EnginePoolConfig {
        max_size: 2,
        min_size: 1,
        ttl: Duration::from_secs(60),
        health_check_interval: Duration::from_secs(10),
        warmup_enabled: true,
        warmup_size: 1,
    };
    let pool = EnginePool::new(config);

    let engine1: Arc<dyn Engine> = Arc::new(MockEngine);
    let engine2: Arc<dyn Engine> = Arc::new(MockEngine);
    let engine3: Arc<dyn Engine> = Arc::new(MockEngine);

    // Add 1 and 2
    pool.put(1, 0, engine1.clone()).await;
    pool.put(2, 0, engine2.clone()).await;

    assert!(pool.get(1, 0).await.is_some());
    assert!(pool.get(2, 0).await.is_some());

    // Add 3, should evict 1 (LRU) because 2 was just accessed.
    pool.put(3, 0, engine3.clone()).await;

    assert!(pool.get(1, 0).await.is_none()); // Evicted
    assert!(pool.get(2, 0).await.is_some());
    assert!(pool.get(3, 0).await.is_some());
}

#[tokio::test]
async fn test_eviction_callback_called() {
    let config = EnginePoolConfig {
        max_size: 2,
        min_size: 1,
        ttl: Duration::from_secs(60),
        health_check_interval: Duration::from_secs(10),
        warmup_enabled: true,
        warmup_size: 1,
    };
    let pool = EnginePool::new(config);

    // Counter to verify callback invocation
    let counter = StdArc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    // Register eviction callback
    pool.set_eviction_callback(move |_model_id, _device_id, _evicted_engine| {
        counter_clone.fetch_add(1, Ordering::SeqCst);
    })
    .await;

    let engine1: Arc<dyn Engine> = Arc::new(MockEngine);
    let engine2: Arc<dyn Engine> = Arc::new(MockEngine);
    let engine3: Arc<dyn Engine> = Arc::new(MockEngine);

    // Add first two engines
    pool.put(1, 0, engine1.clone()).await;
    pool.put(2, 0, engine2.clone()).await;

    // Adding third should evict one engine and trigger callback
    pool.put(3, 0, engine3.clone()).await;

    // Wait briefly for the spawned callback task to run.
    let mut calls = counter.load(Ordering::SeqCst);
    for _ in 0..10 {
        if calls >= 1 {
            break;
        }
        tokio::task::yield_now().await;
        calls = counter.load(Ordering::SeqCst);
    }
    assert_eq!(
        calls, 1,
        "Eviction callback should have been called exactly once"
    );
}

#[tokio::test]
async fn test_get_unhealthy_engine_removes_from_pool() {
    let config = EnginePoolConfig {
        max_size: 2,
        min_size: 1,
        ttl: Duration::from_secs(60),
        health_check_interval: Duration::from_secs(10),
        warmup_enabled: true,
        warmup_size: 1,
    };
    let pool = EnginePool::new(config);

    let engine: Arc<dyn Engine> = Arc::new(FailingHealthEngine);
    pool.put(1, 0, engine).await;

    assert!(pool.get(1, 0).await.is_none());
    assert_eq!(pool.len().await, 0);

    let metrics = pool.pool_metrics().await;
    assert_eq!(metrics.hit, 0);
    assert_eq!(metrics.failure, 1);
}

#[tokio::test]
async fn test_eviction_callback_reports_evicted_key() {
    let config = EnginePoolConfig {
        max_size: 1,
        min_size: 1,
        ttl: Duration::from_secs(60),
        health_check_interval: Duration::from_secs(10),
        warmup_enabled: true,
        warmup_size: 1,
    };
    let pool = EnginePool::new(config);

    let evicted = StdArc::new(StdMutex::new(Vec::new()));
    let evicted_clone = evicted.clone();
    pool.set_eviction_callback(move |model_id, device_id, _evicted_engine| {
        let mut guard = evicted_clone.lock().expect("evicted lock");
        guard.push((model_id, device_id));
    })
    .await;

    let engine1: Arc<dyn Engine> = Arc::new(MockEngine);
    let engine2: Arc<dyn Engine> = Arc::new(MockEngine);

    pool.put(1, 0, engine1).await;
    pool.put(2, 0, engine2).await;

    let mut len = 0;
    for _ in 0..10 {
        len = evicted.lock().expect("evicted lock").len();
        if len >= 1 {
            break;
        }
        tokio::task::yield_now().await;
    }

    let entries = evicted.lock().expect("evicted lock").clone();
    assert_eq!(len, 1);
    assert_eq!(entries, vec![(1, 0)]);
}

#[tokio::test]
async fn test_warmup_skips_failed_engines() {
    let config = EnginePoolConfig {
        max_size: 3,
        min_size: 1,
        ttl: Duration::from_secs(60),
        health_check_interval: Duration::from_secs(10),
        warmup_enabled: true,
        warmup_size: 1,
    };
    let pool = EnginePool::new(config);

    let engine_configs = vec![(1, 0), (2, 0)];
    pool.warmup(engine_configs, |model_id, _device_id| async move {
        if model_id == 1 {
            Err(EngineError::backend("warmup failed"))
        } else {
            Ok(Arc::new(MockEngine) as Arc<dyn Engine>)
        }
    })
    .await
    .expect("warmup should return Ok");

    assert_eq!(pool.len().await, 1);
    assert!(pool.get(2, 0).await.is_some());
}
