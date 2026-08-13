use super::*;
use kapsl_engine_api::{EngineStream, ExternalDeviceMemoryReport, MemoryReport};

struct MemoryTrackedEngine {
    inner: Box<dyn kapsl_engine_api::Engine>,
    lease: std::sync::Mutex<Option<MemoryLease>>,
    resources: Arc<RuntimeResources>,
    owner: MemoryOwner,
    #[cfg(feature = "gpu-device-pool")]
    staged_memory_plan: std::sync::Mutex<Option<MemoryPlan>>,
}

impl MemoryTrackedEngine {
    fn new(
        inner: Box<dyn kapsl_engine_api::Engine>,
        lease: MemoryLease,
        resources: Arc<RuntimeResources>,
        owner: MemoryOwner,
    ) -> Self {
        Self {
            inner,
            lease: std::sync::Mutex::new(Some(lease)),
            resources,
            owner,
            #[cfg(feature = "gpu-device-pool")]
            staged_memory_plan: std::sync::Mutex::new(None),
        }
    }

    fn unload_and_release(&mut self) {
        self.inner.unload();
        self.lease.get_mut().unwrap().take();
    }

    fn request_lease(&self, requests: &[InferenceRequest]) -> Result<MemoryLease, EngineError> {
        let mut plan = MemoryPlan::new();
        for request in requests {
            plan.extend(MemoryPlan::request_from_backend_report(
                self.owner,
                &self.inner.planned_request_memory(request),
            ));
        }
        self.resources
            .memory()
            .admit(&plan)
            .map_err(EngineError::resource_exhausted)
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
        self.inner.actual_external_device_memory()
    }
    fn actual_memory(&self) -> MemoryReport {
        self.inner.actual_memory()
    }
    fn planned_request_memory(&self, request: &InferenceRequest) -> MemoryReport {
        self.inner.planned_request_memory(request)
    }
    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let _request_lease = self.request_lease(std::slice::from_ref(request))?;
        self.inner.infer(request)
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
        let request_lease = match self.request_lease(std::slice::from_ref(request)) {
            Ok(lease) => lease,
            Err(error) => return Box::pin(futures::stream::once(async move { Err(error) })),
        };
        Box::pin(self.inner.infer_stream(request).map(move |item| {
            let _hold = &request_lease;
            item
        }))
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
            let report = self.inner.planned_memory(path)?;
            self.inner.stage(path).await?;
            *self.staged_memory_plan.lock().unwrap() =
                Some(MemoryPlan::from_backend_report(self.owner, &report));
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
            let plan = self
                .staged_memory_plan
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| EngineError::backend("no staged memory plan; call stage() first"))?;
            let swap_lease = self
                .resources
                .memory()
                .begin_swap(&plan)
                .await
                .map_err(EngineError::backend)?;
            let swap_result = self.inner.swap().await;
            self.staged_memory_plan.lock().unwrap().take();
            swap_result?;

            // Activation has released the old weights. Return the temporary
            // peak before reconciling the persistent lease to the target.
            drop(swap_lease);
            let report = self.inner.actual_memory();
            if let Some(lease) = self.lease.lock().unwrap().as_mut() {
                lease
                    .reconcile_report(&report)
                    .map_err(EngineError::backend)?;
            }
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

pub(super) fn create_runtime_backend_for_device(
    manifest: &Manifest,
    provider: &str,
    device_id: usize,
    device_info: &DeviceInfo,
    tuning: &OnnxRuntimeTuning,
    resources: &RuntimeResources,
    model_id: u32,
    replica_id: u32,
) -> Result<Box<dyn kapsl_engine_api::Engine>, String> {
    #[cfg(not(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    )))]
    let _ = (resources, model_id, replica_id);
    #[cfg(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    ))]
    let kind = EngineKind::resolve(manifest);

    #[cfg(feature = "gguf-native")]
    if kind.is_gguf() {
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
    if kind.is_gguf() {
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
    if kind == EngineKind::Native {
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

pub(super) fn create_runtime_best_backend(
    manifest: &Manifest,
    device_info: &DeviceInfo,
    tuning: &OnnxRuntimeTuning,
    resources: &RuntimeResources,
    model_id: u32,
    replica_id: u32,
) -> Result<Box<dyn kapsl_engine_api::Engine>, String> {
    #[cfg(not(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    )))]
    let _ = (resources, model_id, replica_id);
    #[cfg(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    ))]
    {
        let kind = EngineKind::resolve(manifest);
        let device_id = manifest.hardware_requirements.device_id.unwrap_or(0) as usize;

        #[cfg(feature = "gguf-native")]
        if kind.is_gguf() {
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
        if kind.is_gguf() {
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
        if kind == EngineKind::Native {
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
    }

    BackendFactory::create_best_backend_with_tuning_and_owner(
        manifest,
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
    let mut admission = resources.memory().begin_load(&plan, engine_kind).await?;

    if let Err(error) = backend.load(model_file_path).await {
        backend.unload();
        return Err(format!("{load_context}: {error}").into());
    }

    let actual_report = backend.actual_memory();
    if let Err(error) = admission.reconcile(&actual_report) {
        backend.unload();
        return Err(error.into());
    }
    let lease = admission.commit();
    if lease.is_empty() {
        return Ok(backend);
    }
    log::info!(
        "[memory-authority] committed {} claims for {}",
        lease.claims().len(),
        owner
    );
    Ok(Box::new(MemoryTrackedEngine::new(
        backend,
        lease,
        resources.clone(),
        owner,
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
