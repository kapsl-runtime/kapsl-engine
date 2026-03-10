use async_trait::async_trait;
use futures::Stream;
use kapsl_engine_api::Engine;
use kapsl_engine_api::{BinaryTensorPacket, EngineError, EngineModelInfo, InferenceRequest};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use crate::metrics::KapslMetrics;

pub struct MonitoringMiddleware<E: Engine> {
    inner: E,
    metrics: KapslMetrics,
    model_id: String,
    version: String,
    auto_tune: Arc<ConcurrencyAutoTuneState>,
}

struct MetricStream {
    inner: Pin<Box<dyn Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>>,
    metrics: KapslMetrics,
    model_id: String,
    version: String,
    start: Instant,
    finished: bool,
    saw_error: bool,
    auto_tune: Arc<ConcurrencyAutoTuneState>,
}

const PEAK_CONCURRENCY_WINDOW_ENV: &str = "KAPSL_PEAK_CONCURRENCY_WINDOW";
const PEAK_CONCURRENCY_SAMPLE_STRIDE_ENV: &str = "KAPSL_PEAK_CONCURRENCY_SAMPLE_STRIDE";
const DEFAULT_PEAK_CONCURRENCY_WINDOW: usize = 512;
const DEFAULT_PEAK_CONCURRENCY_SAMPLE_STRIDE: u64 = 1;

#[derive(Debug)]
struct ConcurrencyAutoTuneState {
    in_flight: AtomicUsize,
    sample_counter: AtomicU64,
    samples: Mutex<VecDeque<u32>>,
    window: usize,
    sample_stride: u64,
}

impl ConcurrencyAutoTuneState {
    fn new() -> Self {
        let window = std::env::var(PEAK_CONCURRENCY_WINDOW_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_PEAK_CONCURRENCY_WINDOW)
            .max(1);
        let sample_stride = std::env::var(PEAK_CONCURRENCY_SAMPLE_STRIDE_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_PEAK_CONCURRENCY_SAMPLE_STRIDE)
            .max(1);
        Self {
            in_flight: AtomicUsize::new(0),
            sample_counter: AtomicU64::new(0),
            samples: Mutex::new(VecDeque::with_capacity(window)),
            window,
            sample_stride,
        }
    }

    fn on_request_start(&self) {
        let active = self
            .in_flight
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        let count = self
            .sample_counter
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        if count.is_multiple_of(self.sample_stride) {
            let mut samples = self
                .samples
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if samples.len() >= self.window {
                samples.pop_front();
            }
            samples.push_back(active.min(u32::MAX as usize) as u32);
        }
    }

    fn on_request_end(&self) {
        let _ = self.in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    fn estimated_peak_concurrency(&self) -> Option<u32> {
        let mut values: Vec<u32> = {
            let samples = self
                .samples
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if samples.is_empty() {
                return None;
            }
            samples.iter().copied().collect()
        };
        values.sort_unstable();
        let idx = ((values.len() - 1) * 95) / 100;
        Some(values[idx].max(1))
    }
}

impl MetricStream {
    fn new(
        inner: Pin<Box<dyn Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>>,
        metrics: KapslMetrics,
        model_id: String,
        version: String,
        start: Instant,
        auto_tune: Arc<ConcurrencyAutoTuneState>,
    ) -> Self {
        Self {
            inner,
            metrics,
            model_id,
            version,
            start,
            finished: false,
            saw_error: false,
            auto_tune,
        }
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }

        let status = if self.saw_error { "err" } else { "ok" };
        let elapsed = self.start.elapsed().as_secs_f64();

        self.metrics
            .inference_count
            .with_label_values(&[self.model_id.as_str(), status])
            .inc();
        self.metrics
            .inference_latency
            .with_label_values(&[self.model_id.as_str(), self.version.as_str(), status])
            .observe(elapsed);
        self.metrics
            .active_inferences
            .with_label_values(&[self.model_id.as_str()])
            .dec();
        self.auto_tune.on_request_end();

        self.finished = true;
    }
}

impl Stream for MetricStream {
    type Item = Result<BinaryTensorPacket, EngineError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(item))) => Poll::Ready(Some(Ok(item))),
            Poll::Ready(Some(Err(err))) => {
                this.saw_error = true;
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(None) => {
                this.finish();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for MetricStream {
    fn drop(&mut self) {
        if !self.finished {
            self.saw_error = true;
            self.finish();
        }
    }
}

impl<E: Engine> MonitoringMiddleware<E> {
    pub fn new(
        inner: E,
        model_id: String,
        version: String,
        registry: &std::sync::Arc<prometheus::Registry>,
    ) -> Self {
        Self {
            inner,
            metrics: KapslMetrics::new(registry),
            model_id,
            version,
            auto_tune: Arc::new(ConcurrencyAutoTuneState::new()),
        }
    }

    pub fn new_with_metrics(
        inner: E,
        model_id: String,
        version: String,
        metrics: KapslMetrics,
    ) -> Self {
        Self {
            inner,
            metrics,
            model_id,
            version,
            auto_tune: Arc::new(ConcurrencyAutoTuneState::new()),
        }
    }

    pub fn registry(&self) -> &std::sync::Arc<prometheus::Registry> {
        &self.metrics.registry
    }
}

#[async_trait]
impl<E: Engine> Engine for MonitoringMiddleware<E> {
    async fn load(&mut self, model_path: &std::path::Path) -> Result<(), EngineError> {
        self.inner.load(model_path).await
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let start_active = Instant::now();
        self.metrics
            .active_inferences
            .with_label_values(&[self.model_id.as_str()])
            .inc();
        self.auto_tune.on_request_start();

        self.metrics
            .batch_size_hist
            .with_label_values(&[self.model_id.as_str()])
            .observe(request.input.shape[0] as f64);

        let start = Instant::now();
        let result = self.inner.infer(request);
        let elapsed = start.elapsed().as_secs_f64();

        match &result {
            Ok(_) => {
                self.metrics
                    .inference_count
                    .with_label_values(&[self.model_id.as_str(), "ok"])
                    .inc();
                self.metrics
                    .inference_latency
                    .with_label_values(&[self.model_id.as_str(), self.version.as_str(), "ok"])
                    .observe(elapsed);
            }
            Err(_) => {
                self.metrics
                    .inference_count
                    .with_label_values(&[self.model_id.as_str(), "err"])
                    .inc();
                self.metrics
                    .inference_latency
                    .with_label_values(&[self.model_id.as_str(), self.version.as_str(), "err"])
                    .observe(elapsed);
            }
        }

        let _active_elapsed = start_active.elapsed().as_secs_f64();

        self.metrics
            .active_inferences
            .with_label_values(&[self.model_id.as_str()])
            .dec();
        self.auto_tune.on_request_end();

        result
    }

    fn infer_stream(
        &self,
        request: &InferenceRequest,
    ) -> std::pin::Pin<
        Box<dyn futures::stream::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
    > {
        self.metrics
            .active_inferences
            .with_label_values(&[self.model_id.as_str()])
            .inc();
        self.auto_tune.on_request_start();
        self.metrics
            .batch_size_hist
            .with_label_values(&[self.model_id.as_str()])
            .observe(request.input.shape[0] as f64);

        let start = Instant::now();
        let inner = self.inner.infer_stream(request);
        let wrapped = MetricStream::new(
            inner,
            self.metrics.clone(),
            self.model_id.clone(),
            self.version.clone(),
            start,
            self.auto_tune.clone(),
        );

        Box::pin(wrapped)
    }

    fn unload(&mut self) {
        self.inner.unload()
    }

    fn metrics(&self) -> kapsl_engine_api::EngineMetrics {
        self.inner.metrics()
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        let mut info = self.inner.model_info()?;
        if let Some(observed_p95) = self.auto_tune.estimated_peak_concurrency() {
            info.peak_concurrency = Some(info.peak_concurrency.unwrap_or(1).max(observed_p95));
        }
        Some(info)
    }

    fn health_check(&self) -> Result<(), EngineError> {
        self.inner.health_check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_engine_api::EngineError;
    use kapsl_engine_api::TensorDtype;

    struct MockEngine;

    #[async_trait]
    impl Engine for MockEngine {
        async fn load(&mut self, _model_path: &std::path::Path) -> Result<(), EngineError> {
            Ok(())
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
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
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_monitoring_middleware() {
        let engine = MockEngine;
        let registry = std::sync::Arc::new(prometheus::Registry::new());
        let middleware = MonitoringMiddleware::new(
            engine,
            "test_model".to_string(),
            "v1".to_string(),
            &registry,
        );

        let input = BinaryTensorPacket {
            shape: vec![1, 1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 0, 0],
        };
        let request = InferenceRequest {
            input,
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let result = middleware.infer(&request);

        assert!(result.is_ok());
    }
}

#[cfg(test)]
#[path = "middleware_tests.rs"]
mod middleware_tests;
