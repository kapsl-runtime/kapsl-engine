use super::*;
use kapsl_backends::OnnxRuntimeTuning;
use kapsl_engine_api::{
    EngineStream, ExternalDeviceMemoryReport, MemoryReport, OpenAiWireRequest, OpenAiWireResponse,
    OpenAiWireStreamResponse, RequestMemoryAdmission,
};

/// A backend and the pool registrations it needs from planning through final
/// destruction. Drop the engine before releasing allocator callback storage.
pub(super) struct PreparedBackend {
    inner: Box<dyn kapsl_engine_api::Engine>,
    #[cfg(any(feature = "gpu-device-pool", test))]
    _pool_clients: Vec<PoolClientLease>,
}

impl PreparedBackend {
    #[cfg(any(feature = "gpu-device-pool", test))]
    pub(super) fn with_pool_clients(mut self, clients: Vec<PoolClientLease>) -> Self {
        self._pool_clients.extend(clients);
        self
    }
}

impl From<Box<dyn kapsl_engine_api::Engine>> for PreparedBackend {
    fn from(inner: Box<dyn kapsl_engine_api::Engine>) -> Self {
        Self {
            inner,
            #[cfg(any(feature = "gpu-device-pool", test))]
            _pool_clients: Vec::new(),
        }
    }
}

impl<T: kapsl_engine_api::Engine + 'static> From<Box<T>> for PreparedBackend {
    fn from(inner: Box<T>) -> Self {
        Self::from(inner as Box<dyn kapsl_engine_api::Engine>)
    }
}

impl std::ops::Deref for PreparedBackend {
    type Target = dyn kapsl_engine_api::Engine;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

impl std::ops::DerefMut for PreparedBackend {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut()
    }
}

impl Drop for PreparedBackend {
    fn drop(&mut self) {
        self.inner.unload();
        // Fields drop in declaration order: engine first, then client leases.
    }
}

#[cfg(feature = "gpu-device-pool")]
pub(super) fn acquire_embedded_ort_pool_clients(
    resources: &RuntimeResources,
    provider: &str,
    device_ids: &[usize],
) -> Result<Vec<PoolClientLease>, String> {
    if !provider.eq_ignore_ascii_case("cuda") && !provider.eq_ignore_ascii_case("tensorrt") {
        return Ok(Vec::new());
    }
    let mut clients = Vec::new();
    for &device_id in device_ids {
        let device =
            i32::try_from(device_id).map_err(|_| "ORT device ID exceeds i32".to_string())?;
        if let Some(client) =
            resources
                .memory()
                .acquire_cuda_pool_client(device_id, "embedded-ort", |pool| {
                    kapsl_backends::ort_pool_allocator::register_pool_allocator(device, pool)?;
                    let pool = pool.clone();
                    Ok(Box::new(move || {
                        kapsl_backends::ort_pool_allocator::unregister_pool_allocator(device, &pool)
                    }))
                })?
        {
            clients.push(client);
        }
    }
    Ok(clients)
}

struct MemoryTrackedEngine {
    inner: PreparedBackend,
    lease: std::sync::Mutex<Option<MemoryLease>>,
    resources: Arc<RuntimeResources>,
    owner: MemoryOwner,
    engine_kind: EngineKind,
    request_profile: Arc<crate::backend::request_profile::RequestProfile>,
    priority_lease: std::sync::Mutex<Option<ModelPriorityLease>>,
    reconciliation_error: std::sync::Mutex<Option<String>>,
    #[cfg(feature = "gpu-device-pool")]
    staged_memory_lease: std::sync::Mutex<Option<MemorySwapLease>>,
}

impl MemoryTrackedEngine {
    fn new(
        inner: impl Into<PreparedBackend>,
        lease: MemoryLease,
        resources: Arc<RuntimeResources>,
        owner: MemoryOwner,
        engine_kind: EngineKind,
        priority_lease: ModelPriorityLease,
    ) -> Self {
        Self {
            inner: inner.into(),
            lease: std::sync::Mutex::new(Some(lease)),
            resources,
            owner,
            engine_kind,
            request_profile: Arc::new(crate::backend::request_profile::RequestProfile::new(
                "engine.dispatch",
                owner.model_id,
                owner.replica_id,
            )),
            priority_lease: std::sync::Mutex::new(Some(priority_lease)),
            reconciliation_error: std::sync::Mutex::new(None),
            #[cfg(feature = "gpu-device-pool")]
            staged_memory_lease: std::sync::Mutex::new(None),
        }
    }

    fn unload_and_release(&mut self) {
        self.inner.unload();
        self.request_profile.flush(|line| log::info!("{line}"));
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
        let profile = Arc::clone(&self.request_profile);
        RequestMemoryAdmission::new(move || {
            let mut timing = profile.start("memory_admission", 0);
            let result = Self::acquire_request_plan(&resources, &plan);
            timing.mark("acquire_request_plan");
            result
        })
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
        let mut timing = self.request_profile.start("infer", 0);
        let admission = self.request_admission(request);
        timing.mark("request_planning");
        let result = self.inner.infer_with_memory_admission(request, admission);
        timing.mark("admission_and_backend");
        result
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
    onnx_route: Option<&OnnxBackendRoute>,
    provider: &str,
    device_id: usize,
    device_info: &DeviceInfo,
    tuning: Option<&OnnxRuntimeTuning>,
    resources: &RuntimeResources,
    model_id: u32,
    replica_id: u32,
) -> Result<PreparedBackend, String> {
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
            return Ok(backend.into());
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
        return Ok(Box::new(backend).into());
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
        return Ok(Box::new(backend).into());
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
            .map(|backend| PreparedBackend::from(Box::new(backend)));
        }
    }

    if engine_kind.uses_onnx_session() {
        match onnx_route.ok_or_else(|| {
            format!(
                "ONNX model `{}` reached backend construction without an immutable route decision",
                manifest.project_name
            )
        })? {
            OnnxBackendRoute::SignedPack { identity, reason } => {
                log::info!(
                    "Activating signed backend route {}/{} version {} for model `{}`: {}",
                    identity.backend,
                    identity.profile,
                    identity.pack_version,
                    manifest.project_name,
                    reason
                );
                return create_native_backend_pack_engine(
                    identity,
                    manifest,
                    resources,
                    device_id,
                    model_id,
                    replica_id,
                    &onnx_adapter_options(tuning),
                )
                .map(PreparedBackend::from);
            }
            OnnxBackendRoute::EmbeddedRollback { reason } => {
                log::warn!(
                    "Activating embedded ORT rollback route for model `{}`: {}",
                    manifest.project_name,
                    reason
                );
            }
        }
    } else if onnx_route.is_some() {
        return Err(format!(
            "non-ONNX model `{}` received an ONNX backend route",
            manifest.project_name
        ));
    }

    #[cfg(feature = "gpu-device-pool")]
    let pool_clients = if engine_kind.uses_onnx_session() {
        // Signed packs returned above. Only the explicit embedded route can
        // register the legacy environment allocator, before memory planning.
        acquire_embedded_ort_pool_clients(resources, provider, &[device_id])?
    } else {
        Vec::new()
    };

    if engine_kind.is_onnx_generate() {
        // The SDK's automatic ONNX-generate constructor may fall back to CPU.
        // Bind the exact provider chosen by Kapsl policy so a missing CUDA or
        // TensorRT pack fails closed just like the tensor-pipeline path.
        let backend = LLMBackend::with_device(provider.to_owned(), device_id as i32)
            .with_memory_owner(model_id, replica_id);
        #[cfg(feature = "gpu-device-pool")]
        let backend = backend.with_env_allocators(resources.uses_env_allocators(device_id));
        let backend = PreparedBackend::from(Box::new(backend));
        #[cfg(feature = "gpu-device-pool")]
        let backend = backend.with_pool_clients(pool_clients);
        return Ok(backend);
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

    let backend = BackendFactory::create_backend_for_device_with_tuning_and_owner(
        manifest,
        provider,
        device_id,
        device_info,
        tuning,
        model_id,
        replica_id,
    )?;
    let backend = PreparedBackend::from(backend);
    #[cfg(feature = "gpu-device-pool")]
    let backend = backend.with_pool_clients(pool_clients);
    Ok(backend)
}

/// Execute the runtime-owned backend load transaction.
///
/// One backend-neutral plan is admitted before the backend can allocate,
/// reconciled against the backend's cross-domain report plus observed CUDA/RSS
/// deltas after a successful load, and committed into one lease owned by the
/// returned engine.
#[allow(clippy::too_many_arguments)]
pub(super) async fn load_runtime_backend(
    mut backend: PreparedBackend,
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

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum LoadFailure {
        None,
        Planning,
        Admission,
        Loading,
        Cancellation,
        Reconciliation,
    }

    struct PoolAwareEngine {
        clients: Arc<PoolClients>,
        events: Arc<std::sync::Mutex<Vec<&'static str>>>,
        failure: LoadFailure,
    }

    impl PoolAwareEngine {
        fn report(&self, live: bool) -> MemoryReport {
            MemoryReport {
                allocations: [
                    ("weights", EngineMemoryClass::PersistentWeights),
                    ("scratch", EngineMemoryClass::TransientWorkspace),
                ]
                .into_iter()
                .map(|(id, class)| MemoryAllocation {
                    allocation_id: id.to_string(),
                    domain: EngineMemoryDomain::Host,
                    class,
                    source: MemoryAllocationSource::BackendManaged,
                    bytes: if self.failure == LoadFailure::Admission
                        || (live && self.failure == LoadFailure::Reconciliation)
                    {
                        usize::MAX / 4
                    } else {
                        128
                    },
                })
                .collect(),
            }
        }
    }

    impl Drop for PoolAwareEngine {
        fn drop(&mut self) {
            self.events.lock().unwrap().push("destroy");
        }
    }

    #[async_trait::async_trait]
    impl kapsl_engine_api::Engine for PoolAwareEngine {
        fn planned_memory(&self, _path: &Path) -> Result<MemoryReport, EngineError> {
            assert!(
                !self.clients.retire().unwrap(),
                "registration must precede planning"
            );
            self.events.lock().unwrap().push("plan");
            if self.failure == LoadFailure::Planning {
                return Err(EngineError::backend("planned failure"));
            }
            Ok(self.report(false))
        }

        async fn load(&mut self, _path: &Path) -> Result<(), EngineError> {
            assert!(
                !self.clients.retire().unwrap(),
                "registration must survive loading"
            );
            self.events.lock().unwrap().push("load");
            if self.failure == LoadFailure::Cancellation {
                return std::future::pending().await;
            }
            if self.failure == LoadFailure::Loading {
                return Err(EngineError::backend("load failure"));
            }
            Ok(())
        }

        fn actual_memory(&self) -> MemoryReport {
            self.report(true)
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            Ok(request.input.clone())
        }

        fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
            let output = request.input.clone();
            Box::pin(stream::once(async move { Ok(output) }))
        }

        fn unload(&mut self) {
            self.events.lock().unwrap().push("unload");
        }

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        fn health_check(&self) -> Result<(), EngineError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn pool_registration_survives_planning_and_releases_after_backend_destruction() {
        use kapsl_engine_api::Engine;
        let file = tempfile::NamedTempFile::new().unwrap();
        for failure in [
            LoadFailure::None,
            LoadFailure::Planning,
            LoadFailure::Admission,
            LoadFailure::Loading,
            LoadFailure::Cancellation,
            LoadFailure::Reconciliation,
        ] {
            let resources = RuntimeResources::new(&device_info()).unwrap();
            let clients = Arc::new(PoolClients::default());
            let events = Arc::new(std::sync::Mutex::new(Vec::new()));
            let cleanup_events = events.clone();
            let client = clients
                .acquire("fake-adapter", || {
                    events.lock().unwrap().push("register");
                    Ok(Box::new(move || {
                        let mut events = cleanup_events.lock().unwrap();
                        assert_eq!(
                            events.last(),
                            Some(&"destroy"),
                            "allocator outlives backend destructor"
                        );
                        events.push("unregister");
                        Ok(())
                    }))
                })
                .unwrap();
            let prepared = PreparedBackend::from(Box::new(PoolAwareEngine {
                clients: clients.clone(),
                events: events.clone(),
                failure,
            }))
            .with_pool_clients(vec![client]);
            let mut loading = Box::pin(load_runtime_backend(
                prepared,
                file.path(),
                &[MemoryDomain::Host],
                &resources,
                61,
                0,
                EngineKind::Native,
                1,
                "fake adapter lifecycle",
            ));
            if failure == LoadFailure::Cancellation {
                assert!(futures::poll!(loading.as_mut()).is_pending());
                assert!(events.lock().unwrap().contains(&"load"));
                assert!(!clients.retire().unwrap());
                drop(loading);
            } else {
                let result = loading.await;
                if failure == LoadFailure::None {
                    let mut backend = result.unwrap();
                    assert!(!clients.retire().unwrap());
                    backend.unload();
                    assert!(
                        !clients.retire().unwrap(),
                        "backend can still hold allocator pointers until destruction"
                    );
                    drop(backend);
                } else {
                    assert!(result.is_err(), "{failure:?} should reject loading");
                }
            }
            assert!(clients.retire().unwrap());
            let events = events.lock().unwrap();
            assert_eq!(events.first(), Some(&"register"));
            assert_eq!(&events[events.len() - 2..], &["destroy", "unregister"]);
            if matches!(failure, LoadFailure::Planning | LoadFailure::Admission) {
                assert!(
                    !events.contains(&"load"),
                    "failure before admission must not load the backend"
                );
            }
            assert!(resources
                .memory()
                .snapshot()
                .rows
                .iter()
                .filter(|row| row.owner == MemoryOwner::new(61, 0))
                .all(|row| row.reserved_bytes == 0));
        }
    }

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
