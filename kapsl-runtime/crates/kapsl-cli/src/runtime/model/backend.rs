use super::*;
use kapsl_engine_api::{EngineStream, ExternalDeviceMemoryReport};

struct HostMemoryTrackedEngine {
    inner: Box<dyn kapsl_engine_api::Engine>,
    lease: Option<super::super::host_memory::HostMemoryLease>,
}

impl Drop for HostMemoryTrackedEngine {
    fn drop(&mut self) {
        self.inner.unload();
        self.lease.take();
    }
}

#[async_trait::async_trait]
impl kapsl_engine_api::Engine for HostMemoryTrackedEngine {
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
    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        self.inner.infer(request)
    }
    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
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
        self.inner.infer_stream(request)
    }
    async fn warmup(&self) -> Result<(), EngineError> {
        self.inner.warmup().await
    }
    fn unload(&mut self) {
        self.inner.unload();
        self.lease.take();
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
        self.inner.stage(path).await
    }
    async fn swap(&self) -> Result<(), EngineError> {
        self.inner.swap().await
    }
}

fn estimated_host_model_bytes(model_path: &Path) -> Result<usize, String> {
    let serialized = std::fs::metadata(model_path)
        .map_err(|error| format!("stat model {}: {error}", model_path.display()))?
        .len() as usize;
    // Account for decoded/aligned weights and a bounded execution workspace.
    Ok(serialized
        .saturating_mul(5)
        .saturating_div(4)
        .saturating_add((serialized / 4).max(256 * 1024 * 1024)))
}

#[cfg(feature = "gpu-device-pool")]
struct DeviceMemoryTrackedEngine {
    // Keep the backend before its leases so backend resources are torn down
    // before a lease can return admission budget.
    inner: Box<dyn kapsl_engine_api::Engine>,
    leases: std::sync::Mutex<Vec<DeviceMemoryLease>>,
    resources: Arc<RuntimeResources>,
    model_id: u32,
    staged_memory_plan: std::sync::Mutex<Option<ExternalDeviceMemoryReport>>,
}

#[cfg(feature = "gpu-device-pool")]
impl DeviceMemoryTrackedEngine {
    fn new(
        inner: Box<dyn kapsl_engine_api::Engine>,
        leases: Vec<DeviceMemoryLease>,
        resources: Arc<RuntimeResources>,
        model_id: u32,
    ) -> Self {
        Self {
            inner,
            leases: std::sync::Mutex::new(leases),
            resources,
            model_id,
            staged_memory_plan: std::sync::Mutex::new(None),
        }
    }

    fn unload_and_release(&mut self) {
        self.inner.unload();
        self.leases.get_mut().unwrap().clear();
    }
}

#[cfg(feature = "gpu-device-pool")]
impl Drop for DeviceMemoryTrackedEngine {
    fn drop(&mut self) {
        self.unload_and_release();
    }
}

#[cfg(feature = "gpu-device-pool")]
#[async_trait::async_trait]
impl kapsl_engine_api::Engine for DeviceMemoryTrackedEngine {
    fn planned_external_device_memory(
        &self,
        model_path: &Path,
    ) -> Result<ExternalDeviceMemoryReport, EngineError> {
        self.inner.planned_external_device_memory(model_path)
    }

    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        self.inner.load(model_path).await
    }

    fn actual_external_device_memory(&self) -> ExternalDeviceMemoryReport {
        self.inner.actual_external_device_memory()
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        self.inner.infer(request)
    }

    fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<BinaryTensorPacket>, EngineError> {
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
        self.inner.infer_stream(request)
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
        let planned_report = self.inner.planned_external_device_memory(path)?;
        self.inner.stage(path).await?;
        *self.staged_memory_plan.lock().unwrap() = Some(planned_report);
        Ok(())
    }

    async fn swap(&self) -> Result<(), EngineError> {
        let planned_report = self
            .staged_memory_plan
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| EngineError::backend("no staged memory plan; call stage() first"))?;
        let mut device_ids: Vec<_> = planned_report
            .allocations
            .iter()
            .map(|allocation| allocation.device_id)
            .collect();
        device_ids.sort_unstable();
        device_ids.dedup();

        let mut swap_admissions = Vec::new();
        for device_id in device_ids {
            if let Some(admission) = self
                .resources
                .begin_device_memory_swap_admission(device_id, self.model_id, &planned_report)
                .await
                .map_err(EngineError::backend)?
            {
                swap_admissions.push(admission);
            }
        }

        let swap_result = self.inner.swap().await;
        self.staged_memory_plan.lock().unwrap().take();
        swap_result?;

        // Activation has released the old weights. Return the temporary peak
        // reservation before replacing the persistent lease with the target's
        // measured footprint. There is no await between these operations, so a
        // waiting load cannot observe the intermediate ledger state.
        drop(swap_admissions);
        let report = self.inner.actual_external_device_memory();
        for lease in self.leases.lock().unwrap().iter_mut() {
            lease
                .reconcile_report(&report)
                .map_err(EngineError::backend)?;
        }
        Ok(())
    }
}

#[cfg(feature = "gpu-device-pool")]
pub(super) fn track_device_memory(
    engine: Box<dyn kapsl_engine_api::Engine>,
    leases: Vec<DeviceMemoryLease>,
    resources: Arc<RuntimeResources>,
    model_id: u32,
) -> Box<dyn kapsl_engine_api::Engine> {
    Box::new(DeviceMemoryTrackedEngine::new(
        engine, leases, resources, model_id,
    ))
}

pub(super) fn create_runtime_backend_for_device(
    manifest: &Manifest,
    provider: &str,
    device_id: usize,
    device_info: &DeviceInfo,
    tuning: &OnnxRuntimeTuning,
    resources: &RuntimeResources,
    model_id: u32,
) -> Result<Box<dyn kapsl_engine_api::Engine>, String> {
    #[cfg(not(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    )))]
    let _ = (resources, model_id);
    #[cfg(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    ))]
    let kind = EngineKind::resolve(manifest);

    #[cfg(feature = "gguf-native")]
    if kind.is_gguf() {
        let backend = if let Some(pool) = resources.device_pool(device_id) {
            BackendFactory::create_gguf_native_device_pool(device_id as i32, pool, model_id)?
        } else {
            BackendFactory::create_gguf_native(device_id as i32, None)?
        };
        return Ok(Box::new(backend));
    }

    #[cfg(all(feature = "gguf-cuda-shared-kv", not(feature = "gguf-native")))]
    if kind.is_gguf() {
        let backend = if let Some(pool) = resources.device_pool(device_id) {
            BackendFactory::create_gguf_cuda_device_pool(device_id as i32, pool, model_id)?
        } else {
            BackendFactory::create_gguf_cuda_shared_kv(device_id as i32, None)?
        };
        return Ok(Box::new(backend));
    }

    #[cfg(feature = "native")]
    if kind == EngineKind::Native {
        if let Some(pool) = resources.device_pool(device_id) {
            return BackendFactory::create_native_device_pool(device_id as i32, pool, model_id)
                .map(|backend| Box::new(backend) as Box<dyn kapsl_engine_api::Engine>);
        }
    }

    BackendFactory::create_backend_for_device_with_tuning(
        manifest,
        provider,
        device_id,
        device_info,
        tuning,
    )
}

pub(super) fn create_runtime_best_backend(
    manifest: &Manifest,
    device_info: &DeviceInfo,
    tuning: &OnnxRuntimeTuning,
    resources: &RuntimeResources,
    model_id: u32,
) -> Result<Box<dyn kapsl_engine_api::Engine>, String> {
    #[cfg(not(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    )))]
    let _ = (resources, model_id);
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
                BackendFactory::create_gguf_native_device_pool(device_id as i32, pool, model_id)?
            } else {
                BackendFactory::create_gguf_native(device_id as i32, None)?
            };
            return Ok(Box::new(backend));
        }

        #[cfg(all(feature = "gguf-cuda-shared-kv", not(feature = "gguf-native")))]
        if kind.is_gguf() {
            let backend = if let Some(pool) = resources.device_pool(device_id) {
                BackendFactory::create_gguf_cuda_device_pool(device_id as i32, pool, model_id)?
            } else {
                BackendFactory::create_gguf_cuda_shared_kv(device_id as i32, None)?
            };
            return Ok(Box::new(backend));
        }

        #[cfg(feature = "native")]
        if kind == EngineKind::Native {
            if let Some(pool) = resources.device_pool(device_id) {
                return BackendFactory::create_native_device_pool(device_id as i32, pool, model_id)
                    .map(|backend| Box::new(backend) as Box<dyn kapsl_engine_api::Engine>);
            }
        }
    }

    BackendFactory::create_best_backend_with_tuning(manifest, device_info, tuning)
}

/// Execute the runtime-owned backend load transaction.
///
/// Device admission is acquired from the planned report before the backend can
/// allocate, reconciled against the measured report after a successful load,
/// and committed into leases owned by the returned engine. Keeping this in one
/// place prevents primary loads and autoscaled replicas from drifting apart.
#[allow(clippy::too_many_arguments)]
pub(super) async fn load_runtime_backend(
    mut backend: Box<dyn kapsl_engine_api::Engine>,
    model_file_path: &Path,
    admission_device_ids: &[usize],
    resources: &Arc<RuntimeResources>,
    model_id: u32,
    engine_kind: EngineKind,
    load_context: &str,
) -> Result<Box<dyn kapsl_engine_api::Engine>, DynError> {
    let host_memory_lease = resources.begin_host_memory_admission(
        admission_device_ids,
        model_id,
        estimated_host_model_bytes(model_file_path)?,
    )?;
    #[cfg(not(feature = "gpu-device-pool"))]
    let _ = (admission_device_ids, resources, model_id, engine_kind);

    #[cfg(feature = "gpu-device-pool")]
    let mut device_memory_admissions = {
        let planned_report = backend
            .planned_external_device_memory(model_file_path)
            .map_err(|error| format!("backend memory plan failed: {error}"))?;
        let mut device_ids = admission_device_ids.to_vec();
        device_ids.sort_unstable();
        device_ids.dedup();

        let mut admissions = Vec::new();
        for device_id in device_ids {
            if let Some(admission) = resources
                .begin_device_memory_admission(device_id, model_id, engine_kind, &planned_report)
                .await?
            {
                admissions.push(admission);
            }
        }
        admissions
    };

    if let Err(error) = backend.load(model_file_path).await {
        backend.unload();
        return Err(format!("{load_context}: {error}").into());
    }

    #[cfg(feature = "gpu-device-pool")]
    {
        let actual_report = backend.actual_external_device_memory();
        for admission in &mut device_memory_admissions {
            if let Err(error) = admission.reconcile(&actual_report) {
                backend.unload();
                return Err(error.into());
            }
        }
        let leases = device_memory_admissions
            .into_iter()
            .map(DeviceMemoryAdmission::commit)
            .collect();
        let backend = track_device_memory(backend, leases, resources.clone(), model_id);
        Ok(if host_memory_lease.is_some() {
            Box::new(HostMemoryTrackedEngine {
                inner: backend,
                lease: host_memory_lease,
            })
        } else {
            backend
        })
    }

    #[cfg(not(feature = "gpu-device-pool"))]
    Ok(if host_memory_lease.is_some() {
        Box::new(HostMemoryTrackedEngine {
            inner: backend,
            lease: host_memory_lease,
        })
    } else {
        backend
    })
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
