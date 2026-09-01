use super::*;
use kapsl_backends::OnnxRuntimeTuning;
use kapsl_engine_api::{
    EngineStream, ExternalDeviceMemoryReport, MemoryReport, OpenAiWireRequest, OpenAiWireResponse,
    OpenAiWireStreamResponse, RequestMemoryAdmission,
};

struct MemoryTrackedEngine {
    inner: Box<dyn kapsl_engine_api::Engine>,
    lease: std::sync::Mutex<Option<MemoryLease>>,
    resources: Arc<RuntimeResources>,
    owner: MemoryOwner,
    engine_kind: EngineKind,
    priority_lease: std::sync::Mutex<Option<ModelPriorityLease>>,
    reconciliation_error: std::sync::Mutex<Option<String>>,
    #[cfg(feature = "gpu-device-pool")]
    staged_memory_lease: std::sync::Mutex<Option<MemorySwapLease>>,
}

impl MemoryTrackedEngine {
    fn new(
        inner: Box<dyn kapsl_engine_api::Engine>,
        lease: MemoryLease,
        resources: Arc<RuntimeResources>,
        owner: MemoryOwner,
        engine_kind: EngineKind,
        priority_lease: ModelPriorityLease,
    ) -> Self {
        Self {
            inner,
            lease: std::sync::Mutex::new(Some(lease)),
            resources,
            owner,
            engine_kind,
            priority_lease: std::sync::Mutex::new(Some(priority_lease)),
            reconciliation_error: std::sync::Mutex::new(None),
            #[cfg(feature = "gpu-device-pool")]
            staged_memory_lease: std::sync::Mutex::new(None),
        }
    }

    fn unload_and_release(&mut self) {
        self.inner.unload();
        #[cfg(feature = "gpu-device-pool")]
        self.staged_memory_lease.get_mut().unwrap().take();
        self.lease.get_mut().unwrap().take();
        self.priority_lease.get_mut().unwrap().take();
    }

    fn request_plan(&self, requests: &[InferenceRequest]) -> MemoryPlan {
        let mut plan = MemoryPlan::new();
        for request in requests {
            let mut request_plan = MemoryPlan::request_from_backend_report(
                self.owner,
                &self.inner.planned_request_memory(request),
            );
            if !request_plan.contains_class(MemoryAllocationClass::KvCache) {
                let (templates, persistent_kv_bytes) = self
                    .lease
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|lease| {
                        (
                            lease.backend_claim_templates_for_class(MemoryAllocationClass::KvCache),
                            lease.reserved_bytes_for_class(MemoryAllocationClass::KvCache),
                        )
                    })
                    .unwrap_or_default();
                if persistent_kv_bytes == 0 && !templates.is_empty() {
                    request_plan.extend(self.resources.kv().request_memory_plan(
                        self.owner,
                        request,
                        &templates,
                        self.engine_kind == EngineKind::OnnxGenerate,
                    ));
                }
            }
            plan.extend(request_plan);
        }
        plan
    }

    fn acquire_request_plan(
        resources: &RuntimeResources,
        plan: &MemoryPlan,
    ) -> Result<MemoryLease, EngineError> {
        let mut lease = resources
            .memory()
            .admit(&MemoryPlan::new())
            .map_err(EngineError::resource_exhausted)?;
        lease.grow(plan).map_err(EngineError::resource_exhausted)?;
        Ok(lease)
    }

    fn request_lease(&self, requests: &[InferenceRequest]) -> Result<MemoryLease, EngineError> {
        Self::acquire_request_plan(&self.resources, &self.request_plan(requests))
    }

    fn request_admission(&self, request: &InferenceRequest) -> RequestMemoryAdmission {
        let resources = Arc::clone(&self.resources);
        let plan = self.request_plan(std::slice::from_ref(request));
        RequestMemoryAdmission::new(move || Self::acquire_request_plan(&resources, &plan))
    }

    fn openai_wire_request_plan(&self, request: &OpenAiWireRequest) -> MemoryPlan {
        MemoryPlan::request_from_backend_report(
            self.owner,
            &self.inner.planned_openai_wire_request_memory(request),
        )
    }

    fn openai_wire_request_admission(&self, request: &OpenAiWireRequest) -> RequestMemoryAdmission {
        let resources = Arc::clone(&self.resources);
        let plan = self.openai_wire_request_plan(request);
        RequestMemoryAdmission::new(move || Self::acquire_request_plan(&resources, &plan))
    }

    fn reconcile_actual_report(&self, report: &MemoryReport) {
        let result = self
            .lease
            .lock()
            .unwrap()
            .as_mut()
            .map(|lease| lease.reconcile(report));
        let Some(result) = result else {
            return;
        };
        let mut previous = self.reconciliation_error.lock().unwrap();
        match result {
            Ok(()) => {
                if previous.take().is_some() {
                    log::info!(
                        "[memory-authority] continuous reconciliation recovered for {}",
                        self.owner
                    );
                }
            }
            Err(error) => {
                if previous.as_deref() != Some(error.as_str()) {
                    log::warn!(
                        "[memory-authority] continuous reconciliation rejected for {}: {}",
                        self.owner,
                        error
                    );
                    *previous = Some(error);
                }
            }
        }
    }

    fn sample_actual_memory(&self) -> MemoryReport {
        let report = self.inner.actual_memory();
        self.reconcile_actual_report(&report);
        report
    }
}

impl Drop for MemoryTrackedEngine {
    fn drop(&mut self) {
        self.unload_and_release();
    }
}

#[async_trait::async_trait]
impl kapsl_engine_api::Engine for MemoryTrackedEngine {
    fn planned_memory(&self, path: &Path) -> Result<MemoryReport, EngineError> {
        self.inner.planned_memory(path)
    }
    fn planned_external_device_memory(
        &self,
        path: &Path,
    ) -> Result<ExternalDeviceMemoryReport, EngineError> {
        self.inner.planned_external_device_memory(path)
    }
    async fn load(&mut self, path: &Path) -> Result<(), EngineError> {
        self.inner.load(path).await
    }
    fn actual_external_device_memory(&self) -> ExternalDeviceMemoryReport {
        // Older monitoring middleware asks only for the legacy external CUDA
        // report. Sample the full cross-domain report here as well so a live
        // runtime built against that SDK still drives continuous authority
        // reconciliation.
        let external = self.inner.actual_external_device_memory();
        self.reconcile_actual_report(&self.inner.actual_memory());
        external
    }
    fn actual_memory(&self) -> MemoryReport {
        self.sample_actual_memory()
    }
    fn planned_request_memory(&self, request: &InferenceRequest) -> MemoryReport {
        self.inner.planned_request_memory(request)
    }
    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        self.inner
            .infer_with_memory_admission(request, self.request_admission(request))
    }
    fn supports_openai_wire(&self) -> bool {
        self.inner.supports_openai_wire()
    }
    fn planned_openai_wire_request_memory(&self, request: &OpenAiWireRequest) -> MemoryReport {
        self.inner.planned_openai_wire_request_memory(request)
    }
    async fn infer_openai_wire(
        &self,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireResponse, EngineError> {
        self.inner
            .infer_openai_wire_with_memory_admission(
                request,
                self.openai_wire_request_admission(request),
            )
            .await
    }
    async fn infer_openai_wire_stream(
        &self,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        self.inner
            .infer_openai_wire_stream_with_memory_admission(
                request,
                self.openai_wire_request_admission(request),
            )
            .await
    }
    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
        let _request_lease = self.request_lease(requests)?;
        self.inner.infer_batch(requests)
    }
    fn max_batch(&self) -> usize {
        self.inner.max_batch()
    }
    fn self_batches(&self) -> bool {
        self.inner.self_batches()
    }
    fn batching_policy(&self) -> BatchingPolicy {
        self.inner.batching_policy()
    }
    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        self.inner
            .infer_stream_with_memory_admission(request, self.request_admission(request))
    }
    async fn warmup(&self) -> Result<(), EngineError> {
        self.inner.warmup().await
    }
    fn unload(&mut self) {
        self.unload_and_release();
    }
    fn metrics(&self) -> EngineMetrics {
        self.inner.metrics()
    }
    fn model_info(&self) -> Option<EngineModelInfo> {
        self.inner.model_info()
    }
    fn health_check(&self) -> Result<(), EngineError> {
        self.inner.health_check()
    }
    fn supports_swap(&self) -> bool {
        self.inner.supports_swap()
    }
    fn is_staged(&self) -> bool {
        self.inner.is_staged()
    }
    async fn stage(&self, path: &Path) -> Result<(), EngineError> {
        #[cfg(feature = "gpu-device-pool")]
        {
            if self.staged_memory_lease.lock().unwrap().is_some() {
                return Err(EngineError::backend(
                    "target model is already staged with a memory lease",
                ));
            }
            let report = self.inner.planned_memory(path)?;
            let plan = MemoryPlan::from_backend_report(self.owner, &report);
            let swap_lease = self
                .resources
                .memory()
                .begin_swap(&plan)
                .await
                .map_err(EngineError::resource_exhausted)?;
            self.inner.stage(path).await?;
            *self.staged_memory_lease.lock().unwrap() = Some(swap_lease);
            Ok(())
        }
        #[cfg(not(feature = "gpu-device-pool"))]
        {
            self.inner.stage(path).await
        }
    }
    async fn swap(&self) -> Result<(), EngineError> {
        #[cfg(feature = "gpu-device-pool")]
        {
            let swap_lease = self
                .staged_memory_lease
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| EngineError::backend("no staged memory plan; call stage() first"))?;
            let swap_result = self.inner.swap().await;
            swap_result?;

            let report = self.inner.actual_memory();
            if let Some(lease) = self.lease.lock().unwrap().as_mut() {
                swap_lease
                    .finish(lease, &report)
                    .map_err(EngineError::backend)?;
            } else {
                drop(swap_lease);
            }

            // The transfer report can still describe the activation peak if
            // the backend released the old pooled buffers while completing
            // its swap reply. Resample after the temporary lease is gone and
            // contract the persistent rows before returning to the caller.
            let settled_report = self.inner.actual_memory();
            if let Some(lease) = self.lease.lock().unwrap().as_mut() {
                lease
                    .reconcile(&settled_report)
                    .map_err(EngineError::backend)?;
            }
            self.resources.memory().refresh_cuda_pool_metrics();
            Ok(())
        }
        #[cfg(not(feature = "gpu-device-pool"))]
        {
            self.inner.swap().await
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct EstimatedModelMemory {
    session_bytes: usize,
    workspace_bytes: usize,
}

fn estimated_model_memory(model_path: &Path) -> Result<EstimatedModelMemory, String> {
    let serialized = std::fs::metadata(model_path)
        .map_err(|error| format!("stat model {}: {error}", model_path.display()))?
        .len() as usize;
    // Account for decoded/aligned weights and a bounded execution workspace.
    Ok(EstimatedModelMemory {
        session_bytes: serialized.saturating_mul(5).saturating_div(4),
        workspace_bytes: (serialized / 4).max(256 * 1024 * 1024),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn create_runtime_backend_for_device(
    manifest: &Manifest,
    provider: &str,
    device_id: usize,
    device_info: &DeviceInfo,
    tuning: Option<&OnnxRuntimeTuning>,
    resources: &RuntimeResources,
    model_id: u32,
    replica_id: u32,
) -> Result<Box<dyn kapsl_engine_api::Engine>, String> {
    let engine_kind = EngineKind::resolve(manifest);
    #[cfg(not(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    )))]
    let _ = (resources, model_id, replica_id);

    if engine_kind.is_gguf() {
        if let Some(backend) = create_llama_cpp_pack_engine(
            manifest,
            device_info,
            resources,
            device_id,
            model_id,
            replica_id,
        )? {
            return Ok(backend);
        }
    }

    #[cfg(feature = "gguf-native")]
    if engine_kind.is_gguf() {
        let backend = if let Some(pool) = resources.device_pool(device_id) {
            BackendFactory::create_gguf_native_device_pool_for_replica(
                device_id as i32,
                pool,
                model_id,
                replica_id,
            )?
        } else {
            BackendFactory::create_gguf_native(device_id as i32, None)?
        };
        return Ok(Box::new(backend));
    }

    #[cfg(all(feature = "gguf-cuda-shared-kv", not(feature = "gguf-native")))]
    if engine_kind.is_gguf() {
        let backend = if let Some(pool) = resources.device_pool(device_id) {
            BackendFactory::create_gguf_cuda_device_pool_for_replica(
                device_id as i32,
                pool,
                model_id,
                replica_id,
            )?
        } else {
            BackendFactory::create_gguf_cuda_shared_kv(device_id as i32, None)?
        };
        return Ok(Box::new(backend));
    }

    #[cfg(feature = "native")]
    if engine_kind == EngineKind::Native {
        if let Some(pool) = resources.device_pool(device_id) {
            return BackendFactory::create_native_device_pool_for_replica(
                device_id as i32,
                pool,
                model_id,
                replica_id,
            )
            .map(|backend| Box::new(backend) as Box<dyn kapsl_engine_api::Engine>);
        }
    }

    if engine_kind.uses_onnx_session() && generic_native_backend_packs_enabled()? {
        return create_native_backend_pack_engine(
            manifest, provider, resources, device_id, model_id, replica_id,
        );
    }

    if engine_kind.is_onnx_generate() {
        // The SDK's automatic ONNX-generate constructor may fall back to CPU.
        // Bind the exact provider chosen by Kapsl policy so a missing CUDA or
        // TensorRT pack fails closed just like the tensor-pipeline path.
        let backend = LLMBackend::with_device(provider.to_owned(), device_id as i32)
            .with_memory_owner(model_id, replica_id);
        #[cfg(feature = "gpu-device-pool")]
        let backend = backend.with_env_allocators(resources.uses_env_allocators(device_id));
        return Ok(Box::new(backend));
    }

    let default_tuning = OnnxRuntimeTuning::default();
    let tuning = match tuning {
        Some(tuning) => tuning,
        None if engine_kind.uses_onnx_session() => {
            return Err(format!(
                "missing ONNX runtime tuning for {} backend",
                engine_kind.label()
            ));
        }
        // The SDK factory still accepts an ONNX tuning reference at its
        // backend-neutral boundary. Non-ONNX branches ignore this default.
        None => &default_tuning,
    };

    BackendFactory::create_backend_for_device_with_tuning_and_owner(
        manifest,
        provider,
        device_id,
        device_info,
        tuning,
        model_id,
        replica_id,
    )
}

/// Execute the runtime-owned backend load transaction.
///
/// One backend-neutral plan is admitted before the backend can allocate,
/// reconciled against the backend's cross-domain report plus observed CUDA/RSS
/// deltas after a successful load, and committed into one lease owned by the
/// returned engine.
#[allow(clippy::too_many_arguments)]
pub(super) async fn load_runtime_backend(
    mut backend: Box<dyn kapsl_engine_api::Engine>,
    model_file_path: &Path,
    admission_domains: &[MemoryDomain],
    resources: &Arc<RuntimeResources>,
    model_id: u32,
    replica_id: u32,
    engine_kind: EngineKind,
    priority_weight: u32,
    load_context: &str,
) -> Result<Box<dyn kapsl_engine_api::Engine>, DynError> {
    let owner = MemoryOwner::new(model_id, replica_id);
    let estimate = estimated_model_memory(model_file_path)?;
    let planned_report = backend
        .planned_memory(model_file_path)
        .map_err(|error| format!("backend memory plan failed: {error}"))?;
    let plan = resources.memory().model_load_plan_with_report(
        admission_domains,
        owner,
        estimate.session_bytes,
        estimate.workspace_bytes,
        &planned_report,
    )?;
    let priority_lease = resources.priority().register(
        owner,
        priority_weight,
        plan.claims().iter().map(|claim| claim.domain.clone()),
    );
    let mut admission = resources
        .memory()
        .begin_load(&plan, engine_kind)
        .await
        .map_err(|error| MemoryAdmissionFailure::new(owner, priority_weight, &plan, error))?;

    if let Err(error) = backend.load(model_file_path).await {
        backend.unload();
        return Err(format!("{load_context}: {error}").into());
    }

    let actual_report = backend.actual_memory();
    if let Err(error) = admission.reconcile(&actual_report) {
        backend.unload();
        return Err(Box::new(MemoryAdmissionFailure::new(
            owner,
            priority_weight,
            &plan,
            error,
        )));
    }
    let lease = admission.commit();
    if !lease.is_empty() {
        log::info!(
            "[memory-authority] committed {} claims for {}",
            lease.claims().len(),
            owner
        );
    }
    Ok(Box::new(MemoryTrackedEngine::new(
        backend,
        lease,
        resources.clone(),
        owner,
        engine_kind,
        priority_lease,
    )))
}

pub(super) fn monitor_runtime_backend(
    backend: Box<dyn kapsl_engine_api::Engine>,
    model_id: u32,
    model_version: &str,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
) -> EngineHandle {
    let monitored_backend = MonitoringMiddleware::new_with_metrics(
        backend,
        model_id.to_string(),
        model_version.to_owned(),
        shared_metrics.clone(),
    );
    let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
    Arc::from(engine_box)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use kapsl_engine_api::{
        MemoryAllocation, MemoryAllocationClass as EngineMemoryClass, MemoryAllocationSource,
        MemoryDomain as EngineMemoryDomain, OpenAiWireEndpoint, OpenAiWireFormat,
        OpenAiWireResponseHead,
    };

    const MIB: usize = 1024 * 1024;

    struct MutableMemoryEngine {
        bytes: Arc<AtomicUsize>,
    }

    struct DeferredStreamEngine {
        activate: Arc<tokio::sync::Notify>,
        acquired: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    struct WireStreamEngine;

    #[cfg(feature = "gpu-device-pool")]
    struct PeakThenSettledSwapEngine {
        swapped: AtomicBool,
        post_swap_samples: AtomicUsize,
    }

    #[cfg(feature = "gpu-device-pool")]
    impl PeakThenSettledSwapEngine {
        fn report(bytes: usize) -> MemoryReport {
            MemoryReport {
                allocations: vec![MemoryAllocation {
                    allocation_id: "swap:host-weights".to_string(),
                    domain: EngineMemoryDomain::Host,
                    class: EngineMemoryClass::PersistentWeights,
                    source: MemoryAllocationSource::BackendManaged,
                    bytes,
                }],
            }
        }
    }

    impl MutableMemoryEngine {
        fn report(&self) -> MemoryReport {
            MemoryReport {
                allocations: vec![MemoryAllocation {
                    allocation_id: "mutable:host-kv".to_string(),
                    domain: EngineMemoryDomain::Host,
                    class: EngineMemoryClass::KvCache,
                    source: MemoryAllocationSource::BackendManaged,
                    bytes: self.bytes.load(Ordering::Acquire),
                }],
            }
        }
    }

    #[async_trait::async_trait]
    impl kapsl_engine_api::Engine for MutableMemoryEngine {
        async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
            Ok(())
        }

        fn actual_memory(&self) -> MemoryReport {
            self.report()
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            Ok(request.input.clone())
        }

        fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
            let result = Ok(request.input.clone());
            Box::pin(stream::once(async move { result }))
        }

        fn unload(&mut self) {
            self.bytes.store(0, Ordering::Release);
        }

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        fn health_check(&self) -> Result<(), EngineError> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl kapsl_engine_api::Engine for DeferredStreamEngine {
        fn planned_request_memory(&self, _request: &InferenceRequest) -> MemoryReport {
            MemoryReport {
                allocations: vec![MemoryAllocation {
                    allocation_id: "deferred:request".to_string(),
                    domain: EngineMemoryDomain::Host,
                    class: EngineMemoryClass::RequestTransient,
                    source: MemoryAllocationSource::BackendManaged,
                    bytes: MIB,
                }],
            }
        }

        async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
            Ok(())
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            Ok(request.input.clone())
        }

        fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
            let result = Ok(request.input.clone());
            Box::pin(stream::once(async move { result }))
        }

        fn infer_stream_with_memory_admission(
            &self,
            request: &InferenceRequest,
            admission: RequestMemoryAdmission,
        ) -> EngineStream {
            let activate = Arc::clone(&self.activate);
            let acquired = Arc::clone(&self.acquired);
            let release = Arc::clone(&self.release);
            let packet = request.input.clone();
            Box::pin(stream::once(async move {
                activate.notified().await;
                let guard = admission.acquire()?;
                acquired.notify_one();
                release.notified().await;
                drop(guard);
                Ok(packet)
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
    impl kapsl_engine_api::Engine for WireStreamEngine {
        async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
            Ok(())
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            Ok(request.input.clone())
        }

        fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
            let result = Ok(request.input.clone());
            Box::pin(stream::once(async move { result }))
        }

        fn supports_openai_wire(&self) -> bool {
            true
        }

        fn planned_openai_wire_request_memory(&self, _request: &OpenAiWireRequest) -> MemoryReport {
            MemoryReport {
                allocations: vec![MemoryAllocation {
                    allocation_id: "wire:request".to_string(),
                    domain: EngineMemoryDomain::Host,
                    class: EngineMemoryClass::RequestTransient,
                    source: MemoryAllocationSource::BackendManaged,
                    bytes: MIB,
                }],
            }
        }

        async fn infer_openai_wire(
            &self,
            request: &OpenAiWireRequest,
        ) -> Result<OpenAiWireResponse, EngineError> {
            Ok(OpenAiWireResponse {
                head: OpenAiWireResponseHead::new(200, Vec::new())?,
                body: request.body.clone(),
            })
        }

        async fn infer_openai_wire_stream(
            &self,
            _request: &OpenAiWireRequest,
        ) -> Result<OpenAiWireStreamResponse, EngineError> {
            Ok(OpenAiWireStreamResponse {
                head: OpenAiWireResponseHead::new(200, Vec::new())?,
                body: Box::pin(stream::pending()),
            })
        }

        fn unload(&mut self) {}

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        fn health_check(&self) -> Result<(), EngineError> {
            Ok(())
        }
    }

    #[cfg(feature = "gpu-device-pool")]
    #[async_trait::async_trait]
    impl kapsl_engine_api::Engine for PeakThenSettledSwapEngine {
        fn planned_memory(&self, _model_path: &Path) -> Result<MemoryReport, EngineError> {
            Ok(Self::report(MIB))
        }

        async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
            Ok(())
        }

        fn actual_memory(&self) -> MemoryReport {
            if !self.swapped.load(Ordering::Acquire) {
                return Self::report(MIB);
            }
            let sample = self.post_swap_samples.fetch_add(1, Ordering::AcqRel);
            Self::report(if sample == 0 { 2 * MIB } else { MIB })
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            Ok(request.input.clone())
        }

        fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
            let result = Ok(request.input.clone());
            Box::pin(stream::once(async move { result }))
        }

        fn unload(&mut self) {}

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        fn health_check(&self) -> Result<(), EngineError> {
            Ok(())
        }

        fn supports_swap(&self) -> bool {
            true
        }

        async fn stage(&self, _path: &Path) -> Result<(), EngineError> {
            Ok(())
        }

        async fn swap(&self) -> Result<(), EngineError> {
            self.swapped.store(true, Ordering::Release);
            Ok(())
        }
    }

    fn device_info() -> DeviceInfo {
        DeviceInfo {
            cpu_cores: 1,
            total_memory: 10 * 1024 * 1024,
            os_type: "test".to_string(),
            os_release: "test".to_string(),
            has_cuda: false,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: Vec::new(),
        }
    }

    #[tokio::test]
    async fn live_backend_reports_resize_the_persistent_lease() {
        let resources = RuntimeResources::new(&device_info()).unwrap();
        let owner = MemoryOwner::new(41, 0);
        let mut plan = MemoryPlan::new();
        plan.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::KvCache,
            MIB,
        ));
        let initial = MutableMemoryEngine {
            bytes: Arc::new(AtomicUsize::new(MIB)),
        }
        .report();
        let mut admission = resources
            .memory()
            .begin_load(&plan, EngineKind::OnnxGenerate)
            .await
            .unwrap();
        admission.reconcile(&initial).unwrap();
        let lease = admission.commit();
        let bytes = Arc::new(AtomicUsize::new(MIB));
        let priority = resources
            .priority()
            .register(owner, 1, [MemoryDomain::Host]);
        let engine = MemoryTrackedEngine::new(
            Box::new(MutableMemoryEngine {
                bytes: bytes.clone(),
            }),
            lease,
            resources.clone(),
            owner,
            EngineKind::OnnxGenerate,
            priority,
        );

        bytes.store(2 * MIB, Ordering::Release);
        // Exercise the compatibility path used by the currently pinned
        // monitoring middleware as well as the full report path.
        let _ = engine.actual_external_device_memory();
        let row = resources
            .memory()
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner == owner && row.class == MemoryAllocationClass::KvCache)
            .unwrap();
        assert_eq!(row.reserved_bytes, 2 * MIB);
        assert_eq!(row.observed_bytes, 2 * MIB);

        bytes.store(MIB / 2, Ordering::Release);
        let _ = engine.actual_memory();
        let row = resources
            .memory()
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner == owner && row.class == MemoryAllocationClass::KvCache)
            .unwrap();
        assert_eq!(row.reserved_bytes, MIB / 2);
        assert_eq!(row.committed_bytes, MIB / 2);
        assert_eq!(row.observed_bytes, MIB / 2);
    }

    #[tokio::test]
    async fn deferred_request_lease_tracks_active_slot_not_waiting_stream() {
        let resources = RuntimeResources::new(&device_info()).unwrap();
        let owner = MemoryOwner::new(43, 0);
        let activate = Arc::new(tokio::sync::Notify::new());
        let acquired = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let lease = resources
            .memory()
            .admit(&MemoryPlan::new())
            .expect("empty model lease");
        let priority = resources
            .priority()
            .register(owner, 1, [MemoryDomain::Host]);
        let engine = MemoryTrackedEngine::new(
            Box::new(DeferredStreamEngine {
                activate: Arc::clone(&activate),
                acquired: Arc::clone(&acquired),
                release: Arc::clone(&release),
            }),
            lease,
            resources.clone(),
            owner,
            EngineKind::OnnxGenerate,
            priority,
        );
        let request = InferenceRequest::new(
            BinaryTensorPacket::new(vec![1], TensorDtype::Uint8, vec![1]).unwrap(),
        );
        let mut response = engine.infer_stream(&request);
        let response_task = tokio::spawn(async move { response.next().await });
        tokio::task::yield_now().await;

        assert!(resources
            .memory()
            .snapshot()
            .rows
            .iter()
            .all(|row| row.owner != owner || row.class != MemoryAllocationClass::RequestTransient));

        let acquired_wait = acquired.notified();
        activate.notify_one();
        acquired_wait.await;
        let active_row = resources
            .memory()
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner == owner && row.class == MemoryAllocationClass::RequestTransient)
            .unwrap();
        assert_eq!(active_row.reserved_bytes, MIB);

        release.notify_one();
        let result = response_task.await.unwrap().unwrap().unwrap();
        assert_eq!(result.data, vec![1]);
        assert!(resources
            .memory()
            .snapshot()
            .rows
            .iter()
            .all(|row| row.owner != owner || row.class != MemoryAllocationClass::RequestTransient));
    }

    #[tokio::test]
    async fn openai_wire_delegation_holds_request_lease_until_stream_drop() {
        let resources = RuntimeResources::new(&device_info()).unwrap();
        let owner = MemoryOwner::new(44, 0);
        let lease = resources
            .memory()
            .admit(&MemoryPlan::new())
            .expect("empty model lease");
        let priority = resources
            .priority()
            .register(owner, 1, [MemoryDomain::Host]);
        let engine = MemoryTrackedEngine::new(
            Box::new(WireStreamEngine),
            lease,
            resources.clone(),
            owner,
            EngineKind::OnnxGenerate,
            priority,
        );

        assert!(engine.supports_openai_wire());
        let unary = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::Json,
            b"wire".to_vec(),
        );
        assert_eq!(
            engine.infer_openai_wire(&unary).await.unwrap().body,
            b"wire"
        );

        let streaming = OpenAiWireRequest::new(
            OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::ServerSentEvents,
            b"stream".to_vec(),
        );
        assert_eq!(
            engine
                .planned_openai_wire_request_memory(&streaming)
                .bytes_for_domain(&EngineMemoryDomain::Host),
            MIB
        );
        let response = engine.infer_openai_wire_stream(&streaming).await.unwrap();
        let active_row = resources
            .memory()
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner == owner && row.class == MemoryAllocationClass::RequestTransient)
            .expect("wire stream should retain its request admission lease");
        assert_eq!(active_row.reserved_bytes, MIB);

        drop(response);
        assert!(resources
            .memory()
            .snapshot()
            .rows
            .iter()
            .all(|row| row.owner != owner || row.class != MemoryAllocationClass::RequestTransient));
    }

    #[cfg(feature = "gpu-device-pool")]
    #[tokio::test]
    async fn swap_resamples_and_contracts_activation_peak_before_returning() {
        let resources = RuntimeResources::new(&device_info()).unwrap();
        let owner = MemoryOwner::new(42, 0);
        let backend = PeakThenSettledSwapEngine {
            swapped: AtomicBool::new(false),
            post_swap_samples: AtomicUsize::new(0),
        };
        let initial_report = PeakThenSettledSwapEngine::report(MIB);
        let plan = MemoryPlan::from_backend_report(owner, &initial_report);
        let mut admission = resources
            .memory()
            .begin_load(&plan, EngineKind::Native)
            .await
            .unwrap();
        admission.reconcile(&initial_report).unwrap();
        let lease = admission.commit();
        let priority = resources
            .priority()
            .register(owner, 1, [MemoryDomain::Host]);
        let engine = MemoryTrackedEngine::new(
            Box::new(backend),
            lease,
            resources.clone(),
            owner,
            EngineKind::Native,
            priority,
        );

        engine.stage(Path::new("ignored")).await.unwrap();
        engine.swap().await.unwrap();

        let row = resources
            .memory()
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner == owner && row.class == MemoryAllocationClass::PersistentWeights)
            .unwrap();
        assert_eq!(row.reserved_bytes, MIB);
        assert_eq!(row.committed_bytes, MIB);
        assert_eq!(row.observed_bytes, MIB);
    }
}
