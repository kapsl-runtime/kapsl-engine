use super::*;
use futures::StreamExt;
use kapsl_engine_api::{OpenAiWireRequest, OpenAiWireResponse, OpenAiWireStreamResponse};
use kapsl_scheduler::Priority;
use std::pin::Pin;

type InferenceStream =
    Pin<Box<dyn futures::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>>;

/// One runtime inference boundary shared by HTTP, OpenAI, and native
/// transports. Protocol adapters decode requests and responses; this service
/// owns live scheduler lookup and execution policy.
pub(crate) struct InferenceService {
    models: Arc<ModelManager>,
    pressure: Arc<ResourcePressure>,
    latency_samples: Arc<RwLock<HashMap<u32, LatencyWindow>>>,
}

impl InferenceService {
    pub(crate) fn new(
        models: Arc<ModelManager>,
        pressure: Arc<ResourcePressure>,
        latency_samples: Arc<RwLock<HashMap<u32, LatencyWindow>>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            models,
            pressure,
            latency_samples,
        })
    }

    pub(crate) fn priority_for_request(&self, request: &InferenceRequest) -> Priority {
        scheduler_priority_for_request(request)
    }

    pub(crate) fn priority_for_openai_wire_parts(
        &self,
        body_bytes: usize,
        metadata: Option<&kapsl_engine_api::OpenAiWireMetadata>,
    ) -> Priority {
        scheduler_priority_for_openai_wire_parts(body_bytes, metadata)
    }

    /// Resolve the current generation cap before the OpenAI route serializes
    /// its normalized upstream body. This keeps pressure policy in front of
    /// the wire fast path without forcing a second JSON decode/encode cycle.
    pub(crate) fn openai_wire_generation_cap(
        &self,
        priority: Priority,
    ) -> Result<Option<u32>, EngineError> {
        self.enforce_openai_wire_pressure(priority)
    }

    pub(crate) async fn infer(
        &self,
        model_id: u32,
        mut request: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        let pool = self.ready_pool(model_id)?;
        self.apply_pressure_policy(&mut request, priority)?;
        let force_cpu = effective_force_cpu(&request, force_cpu);
        let cancellation = request
            .cancellation
            .get_or_insert_with(kapsl_engine_api::CancellationToken::new)
            .clone();
        let _cancel_on_drop = CancelOnDrop(cancellation.clone());
        let timeout_ms = request_timeout_ms(&request);
        let started = Instant::now();
        let infer = pool.infer(&request, priority, force_cpu);
        let result = if let Some(timeout_ms) = timeout_ms {
            match tokio::time::timeout(Duration::from_millis(timeout_ms), infer).await {
                Ok(result) => result,
                Err(_) => {
                    cancellation.cancel();
                    Err(EngineError::timeout(format!(
                        "Inference timed out after {timeout_ms}ms"
                    )))
                }
            }
        } else {
            infer.await
        };

        if result.is_ok() {
            self.latency_samples
                .write()
                .entry(model_id)
                .or_default()
                .record(started.elapsed().as_secs_f64() * 1000.0);
        }
        result
    }

    pub(crate) async fn infer_stream(
        &self,
        model_id: u32,
        mut request: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<InferenceStream, EngineError> {
        let pool = self.ready_pool(model_id)?;
        self.apply_pressure_policy(&mut request, priority)?;
        let force_cpu = effective_force_cpu(&request, force_cpu);
        let cancellation = request
            .cancellation
            .get_or_insert_with(kapsl_engine_api::CancellationToken::new)
            .clone();
        let timeout_ms = request_timeout_ms(&request);
        let start = pool.infer_stream(request, priority, force_cpu);
        let stream = if let Some(timeout_ms) = timeout_ms {
            match tokio::time::timeout(Duration::from_millis(timeout_ms), start).await {
                Ok(result) => result?,
                Err(_) => {
                    cancellation.cancel();
                    return Err(EngineError::timeout(format!(
                        "Inference stream did not start within {timeout_ms}ms"
                    )));
                }
            }
        } else {
            start.await?
        };

        let guard = CancelOnDrop(cancellation);
        Ok(Box::pin(stream.map(move |item| {
            let _hold = &guard;
            item
        })))
    }

    pub(crate) async fn infer_openai_wire(
        &self,
        model_id: u32,
        mut request: OpenAiWireRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<OpenAiWireResponse, EngineError> {
        let pool = self.ready_pool(model_id)?;
        if let Some(cap) = self.enforce_openai_wire_pressure(priority)? {
            validate_openai_wire_generation_cap(&request, cap)?;
        }
        let cancellation = request
            .cancellation
            .get_or_insert_with(kapsl_engine_api::CancellationToken::new)
            .clone();
        let _cancel_on_drop = CancelOnDrop(cancellation.clone());
        let timeout_ms = openai_wire_timeout_ms(&request);
        let started = Instant::now();
        let infer = pool.infer_openai_wire(request, priority, force_cpu);
        let result = if let Some(timeout_ms) = timeout_ms {
            match tokio::time::timeout(Duration::from_millis(timeout_ms), infer).await {
                Ok(result) => result,
                Err(_) => {
                    cancellation.cancel();
                    Err(EngineError::timeout(format!(
                        "OpenAI inference timed out after {timeout_ms}ms"
                    )))
                }
            }
        } else {
            infer.await
        };

        if result.is_ok() {
            self.latency_samples
                .write()
                .entry(model_id)
                .or_default()
                .record(started.elapsed().as_secs_f64() * 1000.0);
        }
        result
    }

    pub(crate) async fn infer_openai_wire_stream(
        &self,
        model_id: u32,
        mut request: OpenAiWireRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        let pool = self.ready_pool(model_id)?;
        if let Some(cap) = self.enforce_openai_wire_pressure(priority)? {
            validate_openai_wire_generation_cap(&request, cap)?;
        }
        let cancellation = request
            .cancellation
            .get_or_insert_with(kapsl_engine_api::CancellationToken::new)
            .clone();
        let guard = CancelOnDrop(cancellation.clone());
        let timeout_ms = openai_wire_timeout_ms(&request);
        let start = pool.infer_openai_wire_stream(request, priority, force_cpu);
        let response = if let Some(timeout_ms) = timeout_ms {
            match tokio::time::timeout(Duration::from_millis(timeout_ms), start).await {
                Ok(result) => result?,
                Err(_) => {
                    cancellation.cancel();
                    return Err(EngineError::timeout(format!(
                        "OpenAI inference stream did not start within {timeout_ms}ms"
                    )));
                }
            }
        } else {
            start.await?
        };

        Ok(OpenAiWireStreamResponse {
            head: response.head,
            body: Box::pin(response.body.map(move |item| {
                let _hold = &guard;
                item
            })),
        })
    }

    /// Dynamic adapter used by socket/TCP/SHM. The returned scheduler applies
    /// this service's policy and re-resolves the live pool at execution time.
    pub(crate) fn scheduler_for_transport(
        self: &Arc<Self>,
        model_id: u32,
    ) -> Option<Arc<dyn ReplicaScheduler + Send + Sync>> {
        self.models.contains_pool(model_id).then(|| {
            Arc::new(InferenceServiceScheduler {
                service: self.clone(),
                model_id,
            }) as Arc<dyn ReplicaScheduler + Send + Sync>
        })
    }

    pub(crate) fn scheduler_snapshot(
        self: &Arc<Self>,
    ) -> HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>> {
        self.models
            .pools()
            .into_iter()
            .filter_map(|(model_id, _)| {
                self.scheduler_for_transport(model_id)
                    .map(|scheduler| (model_id, scheduler))
            })
            .collect()
    }

    fn ready_pool(&self, model_id: u32) -> Result<Arc<ReplicaPool<Scheduler>>, EngineError> {
        let pool = self
            .models
            .pool(model_id)
            .ok_or(EngineError::ModelNotLoaded)?;
        if !pool.is_healthy() {
            return Err(EngineError::overloaded("Model pool is overloaded"));
        }
        Ok(pool)
    }

    fn apply_pressure_policy(
        &self,
        request: &mut InferenceRequest,
        priority: Priority,
    ) -> Result<(), EngineError> {
        let pressure_state =
            RuntimePressureState::from_u8(self.pressure.state().load(Ordering::Relaxed));
        if pressure_state == RuntimePressureState::Emergency
            && matches!(priority, Priority::Throughput)
        {
            return Err(EngineError::resource_exhausted(format!(
                "runtime pressure {}: throughput requests are temporarily rejected",
                pressure_state.as_str()
            )));
        }
        if let Some(cap) = self.pressure.config().max_new_tokens_cap(pressure_state) {
            let metadata = request
                .metadata
                .get_or_insert_with(kapsl_engine_api::RequestMetadata::default);
            metadata.max_new_tokens = Some(
                metadata
                    .max_new_tokens
                    .map(|existing| existing.min(cap))
                    .unwrap_or(cap),
            );
        }
        Ok(())
    }

    fn enforce_openai_wire_pressure(&self, priority: Priority) -> Result<Option<u32>, EngineError> {
        let pressure_state =
            RuntimePressureState::from_u8(self.pressure.state().load(Ordering::Relaxed));
        if pressure_state == RuntimePressureState::Emergency
            && matches!(priority, Priority::Throughput)
        {
            return Err(EngineError::resource_exhausted(format!(
                "runtime pressure {}: throughput requests are temporarily rejected",
                pressure_state.as_str()
            )));
        }
        Ok(self.pressure.config().max_new_tokens_cap(pressure_state))
    }
}

struct CancelOnDrop(kapsl_engine_api::CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

fn request_timeout_ms(request: &InferenceRequest) -> Option<u64> {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.timeout_ms)
        .filter(|timeout_ms| *timeout_ms > 0)
}

fn openai_wire_timeout_ms(request: &OpenAiWireRequest) -> Option<u64> {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.timeout_ms)
        .filter(|timeout_ms| *timeout_ms > 0)
}

fn validate_openai_wire_generation_cap(
    request: &OpenAiWireRequest,
    cap: u32,
) -> Result<(), EngineError> {
    let body: serde_json::Value = serde_json::from_slice(&request.body).map_err(|error| {
        EngineError::invalid_input(format!(
            "OpenAI wire request body is not valid JSON while applying runtime pressure policy: {error}"
        ))
    })?;
    let object = body.as_object().ok_or_else(|| {
        EngineError::invalid_input("OpenAI wire request body must be a JSON object")
    })?;
    let mut bounded = false;
    for field in ["max_tokens", "max_completion_tokens"] {
        let Some(value) = object.get(field) else {
            continue;
        };
        let value = value.as_u64().ok_or_else(|| {
            EngineError::invalid_input(format!(
                "OpenAI wire field '{field}' must be an unsigned integer under runtime pressure"
            ))
        })?;
        if value > u64::from(cap) {
            return Err(EngineError::resource_exhausted(format!(
                "OpenAI wire field '{field}' requests {value} tokens, above the current runtime pressure cap of {cap}"
            )));
        }
        bounded = true;
    }
    if !bounded {
        return Err(EngineError::resource_exhausted(format!(
            "OpenAI wire request does not carry a generation limit under the current runtime pressure cap of {cap}"
        )));
    }
    Ok(())
}

fn effective_force_cpu(request: &InferenceRequest, adapter_force_cpu: bool) -> bool {
    adapter_force_cpu
        || request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.force_cpu)
            .unwrap_or(false)
}

struct InferenceServiceScheduler {
    service: Arc<InferenceService>,
    model_id: u32,
}

#[async_trait::async_trait]
impl ReplicaScheduler for InferenceServiceScheduler {
    fn get_queue_depth(&self) -> (usize, usize) {
        self.service
            .models
            .pool(self.model_id)
            .map(|pool| pool.get_queue_depth())
            .unwrap_or_default()
    }

    fn is_healthy(&self) -> bool {
        self.service
            .models
            .pool(self.model_id)
            .is_some_and(|pool| pool.is_healthy())
    }

    fn get_metrics(&self) -> EngineMetrics {
        self.service
            .models
            .pool(self.model_id)
            .map(|pool| pool.get_metrics())
            .unwrap_or_default()
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        self.service
            .models
            .pool(self.model_id)
            .and_then(|pool| pool.model_info())
    }

    async fn infer(
        &self,
        request: &InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<BinaryTensorPacket, EngineError> {
        self.service
            .infer(self.model_id, request.clone(), priority, force_cpu)
            .await
    }

    async fn infer_openai_wire(
        &self,
        request: OpenAiWireRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<OpenAiWireResponse, EngineError> {
        self.service
            .infer_openai_wire(self.model_id, request, priority, force_cpu)
            .await
    }

    async fn infer_openai_wire_stream(
        &self,
        request: OpenAiWireRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        self.service
            .infer_openai_wire_stream(self.model_id, request, priority, force_cpu)
            .await
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
        priority: Priority,
        force_cpu: bool,
    ) -> Result<InferenceStream, EngineError> {
        self.service
            .infer_stream(self.model_id, request, priority, force_cpu)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_engine_api::{OpenAiWireEndpoint, OpenAiWireFormat};

    const MODEL_ID: u32 = 7;

    #[test]
    fn wire_pressure_validation_requires_every_authored_limit_to_fit() {
        let request = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::Json,
            br#"{"max_tokens":8,"max_completion_tokens":8}"#.to_vec(),
        );
        validate_openai_wire_generation_cap(&request, 8).unwrap();

        let oversized = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::Json,
            br#"{"max_tokens":8,"max_completion_tokens":9}"#.to_vec(),
        );
        assert!(validate_openai_wire_generation_cap(&oversized, 8).is_err());

        let unbounded = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::Json,
            br#"{"model":"test"}"#.to_vec(),
        );
        assert!(validate_openai_wire_generation_cap(&unbounded, 8).is_err());
    }

    struct RecordingEngine {
        max_new_tokens: Arc<Mutex<Option<u32>>>,
    }

    struct PendingWireEngine {
        entered: Arc<tokio::sync::Notify>,
    }

    #[async_trait::async_trait]
    impl Engine for RecordingEngine {
        async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
            Ok(())
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            *self.max_new_tokens.lock() = request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.max_new_tokens);
            Ok(request.input.clone())
        }

        fn infer_stream(&self, request: &InferenceRequest) -> kapsl_engine_api::EngineStream {
            Box::pin(futures::stream::once({
                let packet = request.input.clone();
                async move { Ok(packet) }
            }))
        }

        fn unload(&mut self) {}

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        fn health_check(&self) -> Result<(), EngineError> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl Engine for PendingWireEngine {
        async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
            Ok(())
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            Ok(request.input.clone())
        }

        fn infer_stream(&self, request: &InferenceRequest) -> kapsl_engine_api::EngineStream {
            let packet = request.input.clone();
            Box::pin(futures::stream::once(async move { Ok(packet) }))
        }

        fn supports_openai_wire(&self) -> bool {
            true
        }

        async fn infer_openai_wire_stream(
            &self,
            _request: &OpenAiWireRequest,
        ) -> Result<OpenAiWireStreamResponse, EngineError> {
            self.entered.notify_one();
            std::future::pending().await
        }

        fn unload(&mut self) {}

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        fn health_check(&self) -> Result<(), EngineError> {
            Ok(())
        }
    }

    fn pressure_config() -> Arc<RuntimePressureConfig> {
        Arc::new(RuntimePressureConfig {
            memory_conserve_ratio: 0.8,
            memory_emergency_ratio: 0.9,
            gpu_util_conserve_ratio: 0.8,
            gpu_util_emergency_ratio: 0.9,
            gpu_mem_conserve_ratio: 0.8,
            gpu_mem_emergency_ratio: 0.9,
            conserve_max_new_tokens: Some(11),
            emergency_max_new_tokens: Some(5),
        })
    }

    fn service(state: Arc<AtomicU8>) -> (Arc<InferenceService>, Arc<Mutex<Option<u32>>>) {
        let models = ModelManager::new(Arc::new(ModelRegistry::new()));
        let seen_cap = Arc::new(Mutex::new(None));
        let engine: EngineHandle = Arc::new(RecordingEngine {
            max_new_tokens: seen_cap.clone(),
        });
        let scheduler = Arc::new(Scheduler::new(vec![engine], 1, 1, 8, true, 1, 0, None));
        let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
        pool.add_replica(0, scheduler);
        models.install_loaded(
            MODEL_ID,
            PathBuf::from("/test/model"),
            Arc::new(pool),
            vec![],
        );
        let pressure = ResourcePressure::new(state, pressure_config());
        let service =
            InferenceService::new(models, pressure, Arc::new(RwLock::new(HashMap::new())));
        (service, seen_cap)
    }

    #[tokio::test]
    async fn dropping_wire_stream_start_cancels_before_upstream_headers() {
        let models = ModelManager::new(Arc::new(ModelRegistry::new()));
        let entered = Arc::new(tokio::sync::Notify::new());
        let engine: EngineHandle = Arc::new(PendingWireEngine {
            entered: entered.clone(),
        });
        let scheduler = Arc::new(Scheduler::new(vec![engine], 1, 1, 8, true, 1, 0, None));
        let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
        pool.add_replica(0, scheduler);
        models.install_loaded(
            MODEL_ID,
            PathBuf::from("/test/wire-model"),
            Arc::new(pool),
            vec![],
        );
        let pressure = ResourcePressure::new(
            Arc::new(AtomicU8::new(RuntimePressureState::Normal as u8)),
            pressure_config(),
        );
        let service =
            InferenceService::new(models, pressure, Arc::new(RwLock::new(HashMap::new())));
        let cancellation = kapsl_engine_api::CancellationToken::new();
        let mut request = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::ServerSentEvents,
            br#"{"max_completion_tokens":8}"#.to_vec(),
        );
        request.cancellation = Some(cancellation.clone());

        let operation = tokio::spawn({
            let service = service.clone();
            async move {
                service
                    .infer_openai_wire_stream(MODEL_ID, request, Priority::LatencyCritical, false)
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .expect("wire engine should start waiting for headers");
        operation.abort();
        let _ = operation.await;
        assert!(
            cancellation.is_cancelled(),
            "dropping stream start must cancel queued/upstream work"
        );
    }

    fn request(max_new_tokens: u32) -> InferenceRequest {
        InferenceRequest::new(BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Uint8,
            data: vec![1],
        })
        .with_metadata(kapsl_engine_api::RequestMetadata {
            max_new_tokens: Some(max_new_tokens),
            ..kapsl_engine_api::RequestMetadata::default()
        })
    }

    #[tokio::test]
    async fn transport_and_direct_calls_share_pressure_policy() {
        let state = Arc::new(AtomicU8::new(RuntimePressureState::Emergency as u8));
        let (service, seen_cap) = service(state.clone());
        let transport = service
            .scheduler_for_transport(MODEL_ID)
            .expect("transport scheduler");

        let error = transport
            .infer(&request(99), Priority::Throughput, false)
            .await
            .expect_err("emergency throughput must be shed");
        assert!(matches!(error, EngineError::ResourceExhausted { .. }));
        assert_eq!(*seen_cap.lock(), None);

        state.store(RuntimePressureState::Conserve as u8, Ordering::Relaxed);
        service
            .infer(MODEL_ID, request(99), Priority::Throughput, false)
            .await
            .expect("conserve mode should clamp and serve");
        assert_eq!(*seen_cap.lock(), Some(11));
    }

    #[tokio::test]
    async fn dropping_a_service_stream_cancels_generation() {
        let state = Arc::new(AtomicU8::new(RuntimePressureState::Normal as u8));
        let (service, _) = service(state);
        let cancellation = kapsl_engine_api::CancellationToken::new();
        let mut request = request(10);
        request.cancellation = Some(cancellation.clone());

        let stream = service
            .infer_stream(MODEL_ID, request, Priority::LatencyCritical, false)
            .await
            .expect("start stream");
        assert!(!cancellation.is_cancelled());
        drop(stream);
        assert!(cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn transport_lookup_tracks_model_lifecycle() {
        let models = ModelManager::new(Arc::new(ModelRegistry::new()));
        let pressure = ResourcePressure::new(
            Arc::new(AtomicU8::new(RuntimePressureState::Normal as u8)),
            pressure_config(),
        );
        let service = InferenceService::new(
            models.clone(),
            pressure,
            Arc::new(RwLock::new(HashMap::new())),
        );
        assert!(service.scheduler_for_transport(MODEL_ID).is_none());

        let engine: EngineHandle = Arc::new(RecordingEngine {
            max_new_tokens: Arc::new(Mutex::new(None)),
        });
        let scheduler = Arc::new(Scheduler::new(vec![engine], 1, 1, 8, true, 1, 0, None));
        let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
        pool.add_replica(0, scheduler);
        models.install_loaded(
            MODEL_ID,
            PathBuf::from("/test/model"),
            Arc::new(pool),
            vec![],
        );
        assert!(service.scheduler_for_transport(MODEL_ID).is_some());
        assert!(service.scheduler_snapshot().contains_key(&MODEL_ID));

        models.stop_runtime(MODEL_ID);
        assert!(service.scheduler_for_transport(MODEL_ID).is_none());
        assert!(!service.scheduler_snapshot().contains_key(&MODEL_ID));
    }
}
