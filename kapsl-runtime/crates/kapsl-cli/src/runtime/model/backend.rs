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
    // ONNX models use ORT's own arena unless the pool is opted into
    // (KAPSL_ONNX_CUDA_POOL=1); see `onnx_cuda_pool_opt_in`.
    if !ort_pool_registration_wanted(provider, onnx_cuda_pool_opt_in()) {
        return Ok(Vec::new());
    }
    let mut clients = Vec::new();
    for &device_id in device_ids {
        // ORT identifies the device by CUDA ordinal; the pool is keyed by the
        // inventory id, so register the allocator under the mapped ordinal.
        let device = kapsl_llm::cuda_ordinal::to_cuda_ordinal(device_id)
            .map_err(|error| error.to_string())
            .and_then(|ordinal| {
                i32::try_from(ordinal).map_err(|_| "ORT device ID exceeds i32".to_string())
            })?;
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
    serialized_bytes: usize,
    session_bytes: usize,
    workspace_bytes: usize,
}

fn estimated_model_memory(model_path: &Path) -> Result<EstimatedModelMemory, String> {
    let serialized = std::fs::metadata(model_path)
        .map_err(|error| format!("stat model {}: {error}", model_path.display()))?
        .len() as usize;
    // Account for decoded/aligned weights and a bounded execution workspace.
    Ok(EstimatedModelMemory {
        serialized_bytes: serialized,
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
    create_runtime_backend_for_device_with_pool(
        manifest,
        onnx_route,
        provider,
        device_id,
        device_info,
        tuning,
        resources,
        model_id,
        replica_id,
        true,
    )
}

/// As [`create_runtime_backend_for_device`]; `allow_pool: false` builds the
/// backend as a process without a runtime device pool would (an isolated worker
/// has none unless the operator gave it an explicit GPU boundary), so a GGUF
/// then plans llama.cpp's native KV instead of a pool-owned one.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_runtime_backend_for_device_with_pool(
    manifest: &Manifest,
    onnx_route: Option<&OnnxBackendRoute>,
    provider: &str,
    device_id: usize,
    device_info: &DeviceInfo,
    tuning: Option<&OnnxRuntimeTuning>,
    resources: &RuntimeResources,
    model_id: u32,
    replica_id: u32,
    allow_pool: bool,
) -> Result<PreparedBackend, String> {
    #[cfg(not(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    )))]
    let _ = allow_pool;
    let engine_kind = EngineKind::resolve(manifest);
    #[cfg(not(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    )))]
    let _ = (resources, model_id, replica_id);

    if engine_kind.is_gguf() {
        // The `create_gguf_*` constructors below take no device list; give the GGUF
        // placement code the inventory so it can map this engine device to a
        // llama.cpp CUDA device by PCI bus id.
        BackendFactory::register_gguf_engine_devices(device_info);
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
        let backend = if let Some(pool) = allow_pool
            .then(|| resources.device_pool(device_id))
            .flatten()
        {
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
        let backend = if let Some(pool) = allow_pool
            .then(|| resources.device_pool(device_id))
            .flatten()
        {
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
        if let Some(pool) = allow_pool
            .then(|| resources.device_pool(device_id))
            .flatten()
        {
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
        let backend = backend.with_env_allocators(
            onnx_cuda_pool_opt_in() && resources.uses_env_allocators(device_id),
        );
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

/// The parent's memory reservation for an isolated worker. Dropping it
/// releases the lease and the priority registration.
pub(super) struct IsolatedWorkerMemory {
    _lease: MemoryLease,
    _priority: ModelPriorityLease,
}

/// Planned memory of an isolated worker in the parent's authority (MEM-08).
///
/// The worker is a separate process with its own ledger, so nothing in the
/// parent charged its weights: N workers could each be admitted against the
/// whole card. This reports the load-time estimate (the same one an in-process
/// load starts from) as one backend-managed allocation per selected domain,
/// charged against the budget outside the pool, before the child is spawned.
///
/// An ONNX model on ORT's own arena (the default) is the exception on CUDA: its
/// worker grows to the same peak an in-process load is admitted at (weights and
/// workspace for every busy session), so the lease is that peak plan, split
/// the way `raise_onnx_arena_claims_to_peak` splits it (for a pipeline mesh the
/// model is shared out over the CUDA devices, as the bootstrap plan does). The estimate would
/// charge a ResNet-50 worker 0.38 GB where its arena reaches 1.8 GB, and
/// the lease is never reconciled upward (the parent gets no report from the
/// child), so the hold has nothing to raise it later.
pub(super) fn isolated_worker_memory_report(
    model_file_path: &Path,
    domains: &[MemoryDomain],
    model_id: u32,
    replica_id: u32,
    engine_kind: EngineKind,
    onnx_sessions: usize,
    pipeline: bool,
) -> Result<MemoryReport, String> {
    use kapsl_engine_api::{
        MemoryAllocation, MemoryAllocationClass as EngineClass,
        MemoryAllocationSource as EngineSource, MemoryDomain as EngineDomain,
    };
    let estimate = estimated_model_memory(model_file_path)?;
    let cuda_devices = domains
        .iter()
        .filter(|domain| matches!(domain, MemoryDomain::Cuda { .. }))
        .count()
        .max(1);
    let arena_peak = onnx_arena_claims_held(engine_kind).then(|| {
        let model_bytes = if pipeline {
            estimate.serialized_bytes.saturating_add(cuda_devices - 1) / cuda_devices
        } else {
            estimate.serialized_bytes
        };
        let sessions = onnx_sessions.max(1);
        (
            model_bytes.saturating_mul(sessions),
            onnx_arena_workspace_per_session_bytes(model_bytes).saturating_mul(sessions),
        )
    });
    let mut allocations = Vec::new();
    for domain in domains {
        let (engine_domain, weights_class) = match domain {
            MemoryDomain::Cuda { device_id } => (
                EngineDomain::Cuda {
                    device_id: *device_id,
                },
                EngineClass::PersistentWeights,
            ),
            MemoryDomain::Host => (EngineDomain::Host, EngineClass::ModelSession),
            // Other providers are planned from the plain estimate below.
            _ => continue,
        };
        let (weights_bytes, workspace_bytes) = match (&domain, arena_peak) {
            (MemoryDomain::Cuda { .. }, Some(peak)) => peak,
            _ => (estimate.session_bytes, estimate.workspace_bytes),
        };
        for (suffix, class, bytes) in [
            ("weights", weights_class, weights_bytes),
            (
                "workspace",
                EngineClass::TransientWorkspace,
                workspace_bytes,
            ),
        ] {
            allocations.push(MemoryAllocation {
                allocation_id: format!("isolated-worker:{model_id}:{replica_id}:{domain}:{suffix}"),
                domain: engine_domain.clone(),
                class,
                source: EngineSource::BackendManaged,
                bytes,
            });
        }
    }
    Ok(MemoryReport { allocations })
}

/// What an in-process load of a GGUF model would plan for the devices an
/// isolated worker uses: the backend's own `planned_memory` (weights, compute
/// buffers and, with llama.cpp-owned KV, the KV), from a backend built the way the
/// in-process load builds it and never loaded. One backend per CUDA device; a
/// worker with no CUDA domain (a CPU host) is planned once on the host. Run 9
/// (check m): the file-size estimate held 882,685,496 B for a model whose
/// in-process plan asked for 2,030,401,376 B.
/// `Err(reason)` for other kinds, or when a backend cannot be built or planned
/// (the caller then keeps the estimate and says why: planning must never block
/// isolation).
#[cfg(feature = "gpu-device-pool")]
#[allow(clippy::too_many_arguments)]
pub(super) fn planned_gguf_report_for_worker(
    manifest: &Manifest,
    onnx_route: Option<&OnnxBackendRoute>,
    domains: &[MemoryDomain],
    device_info: &DeviceInfo,
    tuning: Option<&OnnxRuntimeTuning>,
    resources: &RuntimeResources,
    model_id: u32,
    replica_id: u32,
    model_file_path: &Path,
) -> Result<MemoryReport, String> {
    if !EngineKind::resolve(manifest).is_gguf() {
        return Err("not a GGUF model: the worker keeps the file-size estimate".to_string());
    }
    let mut targets: Vec<(&'static str, usize)> = domains
        .iter()
        .filter_map(|domain| match domain {
            MemoryDomain::Cuda { device_id } => Some(("cuda", *device_id)),
            _ => None,
        })
        .collect();
    if targets.is_empty() {
        targets.push(("cpu", 0));
    }
    let mut merged = MemoryReport {
        allocations: Vec::new(),
    };
    for (provider, device_id) in targets {
        let planned = create_runtime_backend_for_device_with_pool(
            manifest,
            onnx_route,
            provider,
            device_id,
            device_info,
            tuning,
            resources,
            model_id,
            replica_id,
            crate::runtime::serving::isolated_worker_gpu_pool_allowed(),
        )
        .and_then(|backend| {
            backend
                .planned_memory(model_file_path)
                .map_err(|error| error.to_string())
        });
        match planned {
            Ok(report) => merged.allocations.extend(report.allocations),
            Err(error) => {
                let reason = format!("planning on {provider}:{device_id} failed ({error})");
                log::warn!(
                    "[memory-authority] could not plan the isolated worker of model {model_id} replica {replica_id} from its backend: {reason}; using the file-size estimate"
                );
                return Err(reason);
            }
        }
    }
    if merged.allocations.is_empty() {
        let reason = "the backend planned nothing".to_string();
        log::warn!(
            "[memory-authority] the isolated worker of model {model_id} replica {replica_id} got an empty plan from its backend; using the file-size estimate"
        );
        return Err(reason);
    }
    Ok(merged)
}

/// The parent's reservation for an isolated worker from a backend's in-process
/// plan: every backend-managed allocation on the worker's domains, summed per
/// (domain, class) under the isolated-worker ids. Runtime-managed rows (a pool
/// KV the child owns itself) are not the parent's to charge.
///
/// A worker domain the plan leaves without a model row or a workspace row (the
/// backend placed nothing there: another ordinal, `KAPSL_GGUF_N_GPU_LAYERS=0`)
/// takes that row from `estimate`, the file-size report. Left empty, the
/// authority would fill it with a runtime-managed claim, and a runtime-managed
/// CUDA claim is refused without an admitted pool workload. Returns the report and
/// a note per row that fell back.
pub(super) fn isolated_worker_report_from_plan(
    planned: &MemoryReport,
    estimate: &MemoryReport,
    domains: &[MemoryDomain],
    model_id: u32,
    replica_id: u32,
) -> (MemoryReport, Vec<String>) {
    use kapsl_engine_api::{
        MemoryAllocation, MemoryAllocationClass as EngineClass,
        MemoryAllocationSource as EngineSource, MemoryDomain as EngineDomain,
    };
    let wanted = |domain: &EngineDomain| {
        domains.iter().any(|candidate| match (candidate, domain) {
            (MemoryDomain::Cuda { device_id }, EngineDomain::Cuda { device_id: other }) => {
                device_id == other
            }
            (MemoryDomain::Host, EngineDomain::Host) => true,
            _ => false,
        })
    };
    let label_of = |domain: &EngineDomain| match domain {
        EngineDomain::Cuda { device_id } => format!("cuda:{device_id}"),
        EngineDomain::Host => "host".to_string(),
        other => format!("{other:?}").to_ascii_lowercase(),
    };
    let mut rows: Vec<MemoryAllocation> = Vec::new();
    for allocation in &planned.allocations {
        if allocation.source != EngineSource::BackendManaged
            || allocation.bytes == 0
            || !wanted(&allocation.domain)
        {
            continue;
        }
        if let Some(row) = rows
            .iter_mut()
            .find(|row| row.domain == allocation.domain && row.class == allocation.class)
        {
            row.bytes = row.bytes.saturating_add(allocation.bytes);
            continue;
        }
        rows.push(MemoryAllocation {
            allocation_id: format!(
                "isolated-worker:{model_id}:{replica_id}:{}:{}",
                label_of(&allocation.domain),
                format!("{:?}", allocation.class).to_ascii_lowercase()
            ),
            domain: allocation.domain.clone(),
            class: allocation.class,
            source: EngineSource::BackendManaged,
            bytes: allocation.bytes,
        });
    }
    let mut notes = Vec::new();
    let is_model = |class: EngineClass| {
        matches!(
            class,
            EngineClass::PersistentWeights | EngineClass::ModelSession
        )
    };
    for domain in estimate
        .allocations
        .iter()
        .map(|row| row.domain.clone())
        .fold(Vec::<EngineDomain>::new(), |mut seen, domain| {
            if !seen.contains(&domain) {
                seen.push(domain);
            }
            seen
        })
    {
        if !wanted(&domain) {
            continue;
        }
        for (what, covered) in [
            (
                "model",
                rows.iter()
                    .any(|row| row.domain == domain && is_model(row.class)),
            ),
            (
                "workspace",
                rows.iter().any(|row| {
                    row.domain == domain && row.class == EngineClass::TransientWorkspace
                }),
            ),
        ] {
            if covered {
                continue;
            }
            for row in estimate.allocations.iter().filter(|row| {
                row.domain == domain
                    && if what == "model" {
                        is_model(row.class)
                    } else {
                        row.class == EngineClass::TransientWorkspace
                    }
            }) {
                notes.push(format!(
                    "{} {what} from the file-size estimate",
                    label_of(&domain)
                ));
                rows.push(row.clone());
            }
        }
    }
    (MemoryReport { allocations: rows }, notes)
}

/// The one line that says what an isolated worker's parent reservation is made of
/// (the harness reads it: `claude/gpu-verify/lib/gv9.py`, `RE_ISOLATED_RESERVATION`).
pub(super) fn isolated_reservation_line(
    model_id: u32,
    replica_id: u32,
    source: &str,
    report: &MemoryReport,
    notes: &[String],
) -> String {
    use kapsl_engine_api::MemoryDomain as EngineDomain;
    let rows: Vec<String> = report
        .allocations
        .iter()
        .map(|row| {
            let domain = match &row.domain {
                EngineDomain::Cuda { device_id } => format!("cuda:{device_id}"),
                EngineDomain::Host => "host".to_string(),
                other => format!("{other:?}").to_ascii_lowercase(),
            };
            format!(
                "{domain}:{}={}",
                format!("{:?}", row.class).to_ascii_lowercase(),
                row.bytes
            )
        })
        .collect();
    let total: usize = report.allocations.iter().map(|row| row.bytes).sum();
    format!(
        "[memory-authority] isolated worker {model_id}/{replica_id} reservation: source={source} total={total} rows=[{}] notes=[{}]",
        rows.join(", "),
        notes.join("; ")
    )
}

/// Admit and commit the parent-side reservation for an isolated worker. Runs
/// before the child process is spawned; a denial is the same typed
/// [`MemoryAdmissionFailure`] an in-process load would produce.
#[allow(clippy::too_many_arguments)]
pub(super) async fn reserve_isolated_worker_memory(
    model_file_path: &Path,
    domains: &[MemoryDomain],
    resources: &Arc<RuntimeResources>,
    model_id: u32,
    replica_id: u32,
    engine_kind: EngineKind,
    onnx_sessions: usize,
    pipeline: bool,
    priority_weight: u32,
    planned: Result<&MemoryReport, &str>,
) -> Result<IsolatedWorkerMemory, DynError> {
    let owner = MemoryOwner::new(model_id, replica_id);
    let estimate = estimated_model_memory(model_file_path)?;
    let estimate_report = isolated_worker_memory_report(
        model_file_path,
        domains,
        model_id,
        replica_id,
        engine_kind,
        onnx_sessions,
        pipeline,
    )?;
    let (report, source, notes) = match planned {
        Ok(planned) => {
            let (report, notes) = isolated_worker_report_from_plan(
                planned,
                &estimate_report,
                domains,
                model_id,
                replica_id,
            );
            let source = if notes.is_empty() {
                "planned"
            } else {
                "planned+estimate"
            };
            (report, source, notes)
        }
        Err(reason) => (estimate_report, "estimate", vec![reason.to_string()]),
    };
    if !matches!(source, "planned") && planned.is_ok() {
        log::warn!(
            "[memory-authority] the plan of the isolated worker of model {model_id} replica {replica_id} left rows empty; the file-size estimate fills them: {}",
            notes.join("; ")
        );
    }
    log::info!(
        "{}",
        isolated_reservation_line(model_id, replica_id, source, &report, &notes)
    );
    let plan = resources.memory().model_load_plan_with_report(
        domains,
        owner,
        estimate.session_bytes,
        estimate.workspace_bytes,
        &report,
    )?;
    resources
        .memory()
        .check_cuda_load_sampling(&plan)
        .map_err(|error| {
            MemoryAdmissionFailure::new(owner, priority_weight, &plan, error).unreclaimable()
        })?;
    let priority = resources.priority().register(
        owner,
        priority_weight,
        plan.claims().iter().map(|claim| claim.domain.clone()),
    );
    let mut admission = resources
        .memory()
        .begin_load(&plan, engine_kind)
        .await
        .map_err(|error| MemoryAdmissionFailure::new(owner, priority_weight, &plan, error))?;
    admission
        .reconcile(&report)
        .map_err(|error| MemoryAdmissionFailure::new(owner, priority_weight, &plan, error))?;
    Ok(IsolatedWorkerMemory {
        _lease: admission.commit(),
        _priority: priority,
    })
}

/// A backend's own load failure behind the load context. The backend's error
/// stays the source, so callers can still tell why the load failed.
#[derive(Debug)]
pub(crate) struct BackendLoadFailure {
    context: String,
    source: EngineError,
}

impl BackendLoadFailure {
    pub(crate) fn new(context: impl Into<String>, source: EngineError) -> Self {
        Self {
            context: context.into(),
            source,
        }
    }
}

impl std::fmt::Display for BackendLoadFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.context, self.source)
    }
}

impl std::error::Error for BackendLoadFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
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
    let mut planned_report = backend
        .planned_memory(model_file_path)
        .map_err(|error| format!("backend memory plan failed: {error}"))?;
    if engine_kind.uses_onnx_session() {
        // ORT's own arena grows with busy sessions (ResNet-50 loads at 950 MB
        // and reaches 1758 MB), so reserve the peak, not the load-time figure.
        let model_bytes = std::fs::metadata(model_file_path)
            .map(|metadata| metadata.len() as usize)
            .unwrap_or(0);
        raise_onnx_arena_claims_to_peak(&mut planned_report, model_bytes);
    }
    let plan = resources.memory().model_load_plan_with_report(
        admission_domains,
        owner,
        estimate.session_bytes,
        estimate.workspace_bytes,
        &planned_report,
    )?;
    resources
        .memory()
        .check_cuda_load_sampling(&plan)
        .map_err(|error| {
            MemoryAdmissionFailure::new(owner, priority_weight, &plan, error).unreclaimable()
        })?;
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
        return Err(Box::new(BackendLoadFailure::new(load_context, error)));
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

    fn roomy_host_info() -> DeviceInfo {
        let mut info = device_info();
        // `total_memory` is in KiB: 64 GiB.
        info.total_memory = 64 * 1024 * 1024;
        info
    }

    #[test]
    fn isolated_worker_report_charges_weights_and_workspace_per_cuda_device() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(400 * MIB as u64).unwrap();
        let domains = [
            MemoryDomain::Cuda { device_id: 0 },
            MemoryDomain::Cuda { device_id: 1 },
        ];
        let report = isolated_worker_memory_report(
            file.path(),
            &domains,
            9,
            0,
            EngineKind::Native,
            1,
            false,
        )
        .unwrap();
        assert_eq!(report.allocations.len(), 4);
        let estimate = estimated_model_memory(file.path()).unwrap();
        let weights: Vec<_> = report
            .allocations
            .iter()
            .filter(|a| a.class == EngineMemoryClass::PersistentWeights)
            .collect();
        assert_eq!(weights.len(), 2);
        assert!(weights.iter().all(|a| a.bytes == estimate.session_bytes));
        assert!(weights
            .iter()
            .all(|a| a.source == MemoryAllocationSource::BackendManaged));
        let ids: std::collections::HashSet<_> = report
            .allocations
            .iter()
            .map(|a| &a.allocation_id)
            .collect();
        assert_eq!(
            ids.len(),
            4,
            "allocation ids are unique per device and kind"
        );
    }

    #[test]
    fn an_isolated_onnx_worker_on_orts_arena_is_leased_at_the_same_peak_plan_as_in_process() {
        // ResNet-50, 4 sessions: the in-process claim is raised to this peak, and the
        // worker's arena reaches it too, so the parent's lease must not be the
        // 1.25 x file estimate.
        let model = 102_576_593usize;
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(model as u64).unwrap();
        let domains = [MemoryDomain::Cuda { device_id: 0 }, MemoryDomain::Host];
        let total = |report: &MemoryReport, host: bool| {
            report
                .allocations
                .iter()
                .filter(|a| matches!(a.domain, kapsl_engine_api::MemoryDomain::Host) == host)
                .map(|a| a.bytes)
                .sum::<usize>()
        };
        let onnx = isolated_worker_memory_report(
            file.path(),
            &domains,
            9,
            0,
            EngineKind::OnnxClassify,
            4,
            false,
        )
        .unwrap();
        assert_eq!(
            total(&onnx, false),
            onnx_arena_peak_bytes(model, 4),
            "CUDA lease equals the in-process peak plan"
        );
        // Same split as `raise_onnx_arena_claims_to_peak`: weights for every session.
        let weights = onnx
            .allocations
            .iter()
            .find(|a| {
                a.class == EngineMemoryClass::PersistentWeights
                    && !matches!(a.domain, kapsl_engine_api::MemoryDomain::Host)
            })
            .unwrap();
        assert_eq!(weights.bytes, model * 4);
        // The host row of the same worker keeps the plain estimate.
        let estimate = estimated_model_memory(file.path()).unwrap();
        assert_eq!(
            total(&onnx, true),
            estimate.session_bytes + estimate.workspace_bytes
        );
        // A pipeline mesh shares one model copy out over its CUDA devices.
        let two = [
            MemoryDomain::Cuda { device_id: 0 },
            MemoryDomain::Cuda { device_id: 1 },
        ];
        let sharded = isolated_worker_memory_report(
            file.path(),
            &two,
            9,
            0,
            EngineKind::OnnxClassify,
            4,
            true,
        )
        .unwrap();
        assert_eq!(
            total(&sharded, false),
            2 * onnx_arena_peak_bytes(model.div_ceil(2), 4)
        );
        // Kinds that are not held (GGUF, native) keep the estimate on CUDA.
        let gguf = isolated_worker_memory_report(
            file.path(),
            &domains,
            9,
            0,
            EngineKind::GgufGenerate,
            4,
            false,
        )
        .unwrap();
        assert_eq!(
            total(&gguf, false),
            estimate.session_bytes + estimate.workspace_bytes
        );
    }

    // Run 9, check m: a 491,400,032 B Qwen GGUF. The file-size estimate held
    // 882,685,496 B for its worker while an in-process load of the same model
    // planned 2,030,401,376 B (weights, compute buffers and llama.cpp's native KV,
    // which the backend plans when no runtime pool owns the KV).
    const QWEN_FILE: usize = 491_400_032;
    const QWEN_SCRATCH: usize = 703_727_104;
    const QWEN_NATIVE_KV: usize = 835_274_240;

    fn gguf_plan(device_ids: &[usize], with_host: bool) -> MemoryReport {
        use kapsl_engine_api::{
            MemoryAllocation, MemoryAllocationClass as C, MemoryAllocationSource as Source,
            MemoryDomain as D,
        };
        let row = |domain: D, class: C, source: Source, bytes: usize, id: &str| MemoryAllocation {
            allocation_id: id.to_string(),
            domain,
            class,
            source,
            bytes,
        };
        let mut allocations = Vec::new();
        for &device_id in device_ids {
            let cuda = || D::Cuda { device_id };
            allocations.push(row(
                cuda(),
                C::PersistentWeights,
                Source::BackendManaged,
                QWEN_FILE,
                "w",
            ));
            allocations.push(row(
                cuda(),
                C::TransientWorkspace,
                Source::BackendManaged,
                QWEN_SCRATCH,
                "s",
            ));
            allocations.push(row(
                cuda(),
                C::KvCache,
                Source::BackendManaged,
                QWEN_NATIVE_KV,
                "kv",
            ));
        }
        if with_host {
            allocations.push(row(
                D::Host,
                C::ModelSession,
                Source::BackendManaged,
                93_592_576,
                "h",
            ));
        }
        // A pool-owned KV row is the runtime's, never the parent's to charge for a child.
        allocations.push(row(
            D::Cuda {
                device_id: device_ids[0],
            },
            C::KvCache,
            Source::RuntimeManaged,
            0,
            "pool-kv",
        ));
        MemoryReport { allocations }
    }

    fn cuda_total(report: &MemoryReport, device_id: usize) -> usize {
        report
            .allocations
            .iter()
            .filter(|a| a.domain == kapsl_engine_api::MemoryDomain::Cuda { device_id })
            .map(|a| a.bytes)
            .sum()
    }

    #[test]
    fn an_isolated_gguf_worker_is_leased_at_the_in_process_plan() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(QWEN_FILE as u64).unwrap();
        let domains = [MemoryDomain::Cuda { device_id: 0 }];
        // What the file-size estimate held in run 9.
        let estimate = isolated_worker_memory_report(
            file.path(),
            &domains,
            1,
            0,
            EngineKind::GgufGenerate,
            1,
            false,
        )
        .unwrap();
        assert_eq!(cuda_total(&estimate, 0), 882_685_496);

        let planned = gguf_plan(&[0], true);
        let (report, notes) = isolated_worker_report_from_plan(&planned, &estimate, &domains, 1, 0);
        assert!(notes.is_empty(), "nothing fell back: {notes:?}");
        assert_eq!(cuda_total(&report, 0), 2_030_401_376, "the in-process plan");
        assert!(
            cuda_total(&report, 0) > cuda_total(&estimate, 0),
            "KV and compute buffers are no longer left out"
        );
        // Only the worker's own domain, only what a backend manages itself.
        assert!(report.allocations.iter().all(|a| a.domain
            == kapsl_engine_api::MemoryDomain::Cuda { device_id: 0 }
            && a.source == kapsl_engine_api::MemoryAllocationSource::BackendManaged
            && a.allocation_id.starts_with("isolated-worker:1:0:cuda:0:")));
        // One row per class, with distinct ids (the ledger keys on them).
        let ids: std::collections::HashSet<_> = report
            .allocations
            .iter()
            .map(|a| a.allocation_id.clone())
            .collect();
        assert_eq!(ids.len(), report.allocations.len());
    }

    #[test]
    fn an_isolated_gguf_report_follows_the_devices_the_worker_uses() {
        let planned = gguf_plan(&[0, 1], true);
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(QWEN_FILE as u64).unwrap();
        let both = [
            MemoryDomain::Cuda { device_id: 0 },
            MemoryDomain::Cuda { device_id: 1 },
            MemoryDomain::Host,
        ];
        let estimate_for = |domains: &[MemoryDomain]| {
            isolated_worker_memory_report(
                file.path(),
                domains,
                2,
                1,
                EngineKind::GgufGenerate,
                1,
                false,
            )
            .unwrap()
        };
        let (report, notes) =
            isolated_worker_report_from_plan(&planned, &estimate_for(&both), &both, 2, 1);
        // The plan's single host row is a model row; its workspace comes from the estimate.
        assert_eq!(
            notes,
            vec!["host workspace from the file-size estimate".to_string()]
        );
        assert_eq!(cuda_total(&report, 0), 2_030_401_376);
        assert_eq!(cuda_total(&report, 1), 2_030_401_376);
        assert!(report
            .allocations
            .iter()
            .any(|a| a.domain == kapsl_engine_api::MemoryDomain::Host));
        let one = [MemoryDomain::Cuda { device_id: 1 }];
        let (only_one, _) =
            isolated_worker_report_from_plan(&planned, &estimate_for(&one), &one, 2, 1);
        assert_eq!(cuda_total(&only_one, 0), 0, "device 0 is not the worker's");
        assert_eq!(cuda_total(&only_one, 1), 2_030_401_376);
    }

    #[test]
    fn a_worker_domain_the_plan_leaves_empty_takes_the_estimate_rows() {
        // The plan places everything on cuda:0; the worker also uses cuda:1 (another ordinal,
        // or `KAPSL_GGUF_N_GPU_LAYERS=0`). An empty domain would get runtime-managed CUDA
        // claims from the authority, which are refused without an admitted pool workload.
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(QWEN_FILE as u64).unwrap();
        let domains = [
            MemoryDomain::Cuda { device_id: 0 },
            MemoryDomain::Cuda { device_id: 1 },
        ];
        let estimate = isolated_worker_memory_report(
            file.path(),
            &domains,
            3,
            0,
            EngineKind::GgufGenerate,
            1,
            false,
        )
        .unwrap();
        let planned = gguf_plan(&[0], false);
        let (report, notes) = isolated_worker_report_from_plan(&planned, &estimate, &domains, 3, 0);
        assert_eq!(
            cuda_total(&report, 0),
            2_030_401_376,
            "the planned device keeps its plan"
        );
        assert_eq!(
            cuda_total(&report, 1),
            cuda_total(&estimate, 1),
            "the empty device takes the estimate"
        );
        assert_eq!(notes.len(), 2, "a note per missing row: {notes:?}");
        assert!(notes.iter().all(|note| note.starts_with("cuda:1 ")));
        // Every worker domain now has a model row and a workspace row.
        for device_id in [0usize, 1] {
            let domain = kapsl_engine_api::MemoryDomain::Cuda { device_id };
            assert!(report.allocations.iter().any(|a| a.domain == domain
                && a.class == kapsl_engine_api::MemoryAllocationClass::PersistentWeights));
            assert!(report.allocations.iter().any(|a| a.domain == domain
                && a.class == kapsl_engine_api::MemoryAllocationClass::TransientWorkspace));
        }
    }

    #[test]
    fn a_plan_that_only_holds_host_rows_leaves_a_cuda_worker_on_the_estimate() {
        // Every layer on the CPU: the backend plans host rows only, whatever the worker's domain.
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(QWEN_FILE as u64).unwrap();
        let domains = [MemoryDomain::Cuda { device_id: 0 }];
        let estimate = isolated_worker_memory_report(
            file.path(),
            &domains,
            4,
            0,
            EngineKind::GgufGenerate,
            1,
            false,
        )
        .unwrap();
        let host_only = MemoryReport {
            allocations: vec![kapsl_engine_api::MemoryAllocation {
                allocation_id: "h".into(),
                domain: kapsl_engine_api::MemoryDomain::Host,
                class: kapsl_engine_api::MemoryAllocationClass::ModelSession,
                source: kapsl_engine_api::MemoryAllocationSource::BackendManaged,
                bytes: QWEN_FILE,
            }],
        };
        let (report, notes) =
            isolated_worker_report_from_plan(&host_only, &estimate, &domains, 4, 0);
        assert_eq!(cuda_total(&report, 0), cuda_total(&estimate, 0));
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(report
            .allocations
            .iter()
            .all(|a| a.domain == kapsl_engine_api::MemoryDomain::Cuda { device_id: 0 }));
    }

    #[test]
    fn the_reservation_line_says_where_the_bytes_came_from() {
        let planned = gguf_plan(&[0], false);
        let domains = [MemoryDomain::Cuda { device_id: 0 }];
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(QWEN_FILE as u64).unwrap();
        let estimate = isolated_worker_memory_report(
            file.path(),
            &domains,
            5,
            2,
            EngineKind::GgufGenerate,
            1,
            false,
        )
        .unwrap();
        let (report, notes) = isolated_worker_report_from_plan(&planned, &estimate, &domains, 5, 2);
        let line = isolated_reservation_line(5, 2, "planned", &report, &notes);
        assert!(line.starts_with("[memory-authority] isolated worker 5/2 reservation: source=planned total=2030401376 rows=["), "{line}");
        assert!(
            line.contains("cuda:0:persistentweights=491400032"),
            "{line}"
        );
        assert!(
            line.contains("cuda:0:transientworkspace=703727104"),
            "{line}"
        );
        assert!(line.contains("cuda:0:kvcache=835274240"), "{line}");
        assert!(line.ends_with("notes=[]"), "{line}");
        let fallback = isolated_reservation_line(
            5,
            2,
            "estimate",
            &estimate,
            &["planning on cuda:0 failed (stat GGUF model)".to_string()],
        );
        assert!(fallback.contains("source=estimate"), "{fallback}");
        assert!(
            fallback.contains("notes=[planning on cuda:0 failed (stat GGUF model)]"),
            "{fallback}"
        );
    }

    #[tokio::test]
    async fn a_planned_isolated_worker_is_leased_at_the_plan_and_released_with_it() {
        let resources = RuntimeResources::new(&roomy_host_info()).unwrap();
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(64 * MIB as u64).unwrap();
        let owner = MemoryOwner::new(73, 0);
        let used = |resources: &Arc<RuntimeResources>| {
            resources
                .memory()
                .snapshot()
                .rows
                .iter()
                .filter(|row| row.owner == owner)
                .map(|row| row.reserved_bytes.max(row.committed_bytes))
                .sum::<usize>()
        };
        // A host-domain plan: a 300 MiB session row and a 700 MiB workspace row.
        let planned = MemoryReport {
            allocations: vec![
                kapsl_engine_api::MemoryAllocation {
                    allocation_id: "w".into(),
                    domain: kapsl_engine_api::MemoryDomain::Host,
                    class: kapsl_engine_api::MemoryAllocationClass::ModelSession,
                    source: kapsl_engine_api::MemoryAllocationSource::BackendManaged,
                    bytes: 300 * MIB,
                },
                kapsl_engine_api::MemoryAllocation {
                    allocation_id: "s".into(),
                    domain: kapsl_engine_api::MemoryDomain::Host,
                    class: kapsl_engine_api::MemoryAllocationClass::TransientWorkspace,
                    source: kapsl_engine_api::MemoryAllocationSource::BackendManaged,
                    bytes: 700 * MIB,
                },
            ],
        };
        let memory = reserve_isolated_worker_memory(
            file.path(),
            &[MemoryDomain::Host],
            &resources,
            73,
            0,
            EngineKind::GgufGenerate,
            1,
            false,
            1,
            Ok(&planned),
        )
        .await
        .expect("fits");
        assert_eq!(
            used(&resources),
            1000 * MIB,
            "the lease is the plan, not the file estimate"
        );
        drop(memory);
        assert_eq!(used(&resources), 0);
    }

    #[tokio::test]
    async fn an_isolated_worker_holds_a_parent_lease_until_it_exits() {
        let resources = RuntimeResources::new(&roomy_host_info()).unwrap();
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(64 * MIB as u64).unwrap();
        let owner = MemoryOwner::new(71, 0);
        let host_used = |resources: &Arc<RuntimeResources>| {
            resources
                .memory()
                .snapshot()
                .rows
                .iter()
                .filter(|row| row.owner == owner)
                .map(|row| row.reserved_bytes.max(row.committed_bytes))
                .sum::<usize>()
        };
        assert_eq!(host_used(&resources), 0);
        let memory = reserve_isolated_worker_memory(
            file.path(),
            &[MemoryDomain::Host],
            &resources,
            71,
            0,
            EngineKind::Native,
            1,
            false,
            1,
            Err("test: no plan"),
        )
        .await
        .expect("a 64 MiB model fits a 64 GiB host");
        let estimate = estimated_model_memory(file.path()).unwrap();
        assert!(
            host_used(&resources) >= estimate.session_bytes,
            "the parent now charges the worker's weights: {}",
            host_used(&resources)
        );
        drop(memory);
        assert_eq!(host_used(&resources), 0, "released when the worker is gone");
    }

    #[tokio::test]
    async fn an_isolated_worker_that_does_not_fit_is_denied_before_spawning() {
        // A 9 GiB (sparse) model plans 11+ GiB on a 10 GiB host.
        let resources = RuntimeResources::new(&device_info()).unwrap();
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(9 * 1024 * MIB as u64).unwrap();
        let error = reserve_isolated_worker_memory(
            file.path(),
            &[MemoryDomain::Host],
            &resources,
            72,
            0,
            EngineKind::Native,
            1,
            false,
            1,
            Err("test: no plan"),
        )
        .await
        .err()
        .expect("denied");
        assert!(
            error.downcast_ref::<MemoryAdmissionFailure>().is_some(),
            "typed, so the API answers 409 and the arbiter may reclaim: {error}"
        );
    }

    // ---- real-GGUF checks of the planning backend the parent builds for an isolated worker (opt-in)
    //
    //   KAPSL_GGUF_N_GPU_LAYERS=0 KAPSL_TEST_GGUF=/path/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    //     cargo test -p kapsl --features gpu-device-pool real_gguf_planning -- --ignored --test-threads=1 --nocapture
    #[cfg(feature = "gpu-device-pool")]
    fn real_gguf() -> PathBuf {
        PathBuf::from(
            std::env::var("KAPSL_TEST_GGUF").expect("set KAPSL_TEST_GGUF to a .gguf path"),
        )
    }

    #[cfg(feature = "gpu-device-pool")]
    fn gguf_manifest() -> Manifest {
        serde_json::from_value(serde_json::json!({
            "project_name": "qwen-test",
            "framework": "gguf",
            "version": "1.0.0",
            "created_at": "2026-11-22",
            "model_file": "model.gguf",
            "hardware_requirements": { "preferred_provider": "cpu" },
        }))
        .expect("a GGUF manifest")
    }

    #[cfg(feature = "gpu-device-pool")]
    fn backend_managed_total(
        report: &MemoryReport,
        domain: &kapsl_engine_api::MemoryDomain,
    ) -> usize {
        report
            .allocations
            .iter()
            .filter(|a| {
                &a.domain == domain
                    && a.source == kapsl_engine_api::MemoryAllocationSource::BackendManaged
            })
            .map(|a| a.bytes)
            .sum()
    }

    #[cfg(feature = "gpu-device-pool")]
    #[tokio::test]
    #[ignore = "needs KAPSL_TEST_GGUF (real GGUF, CPU)"]
    async fn real_gguf_planning_a_cpu_isolated_worker_is_leased_at_its_backends_planned_memory() {
        let path = real_gguf();
        let info = roomy_host_info();
        let resources = RuntimeResources::new(&info).unwrap();
        let manifest = gguf_manifest();
        // What the backend itself plans for an in-process load (built without a pool, as the worker is).
        let direct = create_runtime_backend_for_device_with_pool(
            &manifest, None, "cpu", 0, &info, None, &resources, 7, 0, false,
        )
        .expect("a GGUF backend is built on the CPU")
        .planned_memory(&path)
        .expect("planned_memory on a real GGUF");
        let host = kapsl_engine_api::MemoryDomain::Host;
        let direct_host = backend_managed_total(&direct, &host);
        eprintln!(
            "planned_memory rows: {:?}",
            direct
                .allocations
                .iter()
                .map(|a| (
                    format!("{:?}", a.domain),
                    format!("{:?}", a.class),
                    a.source,
                    a.bytes
                ))
                .collect::<Vec<_>>()
        );
        assert!(
            direct_host > 0,
            "the backend plans host memory for a CPU GGUF"
        );

        // The helper (no CUDA domain: the host is planned once) returns the same plan.
        let domains = [MemoryDomain::Host];
        let planned = planned_gguf_report_for_worker(
            &manifest, None, &domains, &info, None, &resources, 7, 0, &path,
        )
        .expect("planned_gguf_report_for_worker on a real GGUF");
        assert_eq!(backend_managed_total(&planned, &host), direct_host);

        // The reservation it becomes equals that plan to the byte (nothing fell back) ...
        let estimate = isolated_worker_memory_report(
            &path,
            &domains,
            7,
            0,
            EngineKind::GgufGenerate,
            1,
            false,
        )
        .unwrap();
        let (report, notes) = isolated_worker_report_from_plan(&planned, &estimate, &domains, 7, 0);
        eprintln!(
            "{}",
            isolated_reservation_line(7, 0, "planned", &report, &notes)
        );
        let reserved: usize = report.allocations.iter().map(|a| a.bytes).sum();
        let from_plan = estimate_notes_free(&notes, reserved, direct_host, &estimate);
        assert!(
            from_plan,
            "reserved {reserved} B, backend planned {direct_host} B, notes {notes:?}"
        );

        // ... and so does the lease the authority holds, released with the worker.
        let owner = MemoryOwner::new(7, 0);
        let used = |resources: &Arc<RuntimeResources>| {
            resources
                .memory()
                .snapshot()
                .rows
                .iter()
                .filter(|row| row.owner == owner)
                .map(|row| row.reserved_bytes.max(row.committed_bytes))
                .sum::<usize>()
        };
        let lease = reserve_isolated_worker_memory(
            &path,
            &domains,
            &resources,
            7,
            0,
            EngineKind::GgufGenerate,
            1,
            false,
            1,
            Ok(&planned),
        )
        .await
        .expect("a 0.5B model fits a 64 GiB host");
        assert_eq!(used(&resources), reserved, "the lease is the reservation");
        drop(lease);
        assert_eq!(used(&resources), 0);
    }

    /// The reservation equals the backend's plan when no row fell back; when the plan lacked a workspace
    /// row the estimate filled exactly that row.
    #[cfg(feature = "gpu-device-pool")]
    fn estimate_notes_free(
        notes: &[String],
        reserved: usize,
        planned_host: usize,
        estimate: &MemoryReport,
    ) -> bool {
        if notes.is_empty() {
            return reserved == planned_host;
        }
        let ws: usize = estimate
            .allocations
            .iter()
            .filter(|a| a.class == kapsl_engine_api::MemoryAllocationClass::TransientWorkspace)
            .map(|a| a.bytes)
            .sum();
        notes.len() == 1 && reserved == planned_host + ws
    }

    #[cfg(feature = "gpu-device-pool")]
    #[test]
    #[ignore = "needs KAPSL_TEST_GGUF (real GGUF, CPU)"]
    fn real_gguf_planning_a_cuda_worker_on_a_cpu_build_falls_back_to_the_estimate_per_domain() {
        let path = real_gguf();
        let info = roomy_host_info();
        let resources = RuntimeResources::new(&info).unwrap();
        let manifest = gguf_manifest();
        let domains = [MemoryDomain::Cuda { device_id: 0 }];
        let estimate = isolated_worker_memory_report(
            &path,
            &domains,
            8,
            0,
            EngineKind::GgufGenerate,
            1,
            false,
        )
        .unwrap();
        let cuda = kapsl_engine_api::MemoryDomain::Cuda { device_id: 0 };
        match planned_gguf_report_for_worker(
            &manifest, None, &domains, &info, None, &resources, 8, 0, &path,
        ) {
            Err(reason) => {
                eprintln!("planning refused (warning logged): {reason}");
                assert!(!reason.is_empty());
            }
            Ok(planned) => {
                // A CPU build places nothing on a CUDA ordinal: the domain must not be left empty.
                let (report, notes) =
                    isolated_worker_report_from_plan(&planned, &estimate, &domains, 8, 0);
                eprintln!(
                    "{}",
                    isolated_reservation_line(8, 0, "planned+estimate", &report, &notes)
                );
                assert_eq!(
                    backend_managed_total(&report, &cuda),
                    backend_managed_total(&estimate, &cuda),
                    "a CUDA worker domain with no planned rows reserves the estimate (notes {notes:?})"
                );
                assert!(!notes.is_empty());
            }
        }
    }

    #[cfg(feature = "gpu-device-pool")]
    #[test]
    #[ignore = "needs KAPSL_TEST_GGUF (real GGUF, CPU)"]
    fn real_gguf_planning_a_model_that_cannot_be_planned_keeps_the_estimate() {
        let real = real_gguf();
        let info = roomy_host_info();
        let resources = RuntimeResources::new(&info).unwrap();
        let manifest = gguf_manifest();
        // `planned_memory` stats the file: a path that is gone cannot be planned.
        let gone = real.with_file_name("no-such-model.gguf");
        let err = planned_gguf_report_for_worker(
            &manifest,
            None,
            &[MemoryDomain::Host],
            &info,
            None,
            &resources,
            9,
            0,
            &gone,
        )
        .expect_err("a missing model cannot be planned");
        assert!(err.contains("planning on cpu:0 failed"), "{err}");
        // A model that is not GGUF is not planned from a backend at all.
        let mut onnx = gguf_manifest();
        onnx.framework = "onnx".to_string();
        let err = planned_gguf_report_for_worker(
            &onnx,
            None,
            &[MemoryDomain::Host],
            &info,
            None,
            &resources,
            9,
            0,
            &real,
        )
        .expect_err("not GGUF");
        assert!(err.contains("not a GGUF model"), "{err}");
        // The reservation then reports the estimate, and says why.
        let estimate = isolated_worker_memory_report(
            &real,
            &[MemoryDomain::Host],
            9,
            0,
            EngineKind::GgufGenerate,
            1,
            false,
        )
        .unwrap();
        let line = isolated_reservation_line(9, 0, "estimate", &estimate, &[err]);
        assert!(
            line.contains("source=estimate") && line.contains("notes=[not a GGUF model"),
            "{line}"
        );
    }
}
