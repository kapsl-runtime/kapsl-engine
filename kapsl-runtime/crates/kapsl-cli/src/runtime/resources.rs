use super::*;

/// Shared pressure snapshot consumed by ingress and resource policy.
pub(crate) struct ResourcePressure {
    state: Arc<AtomicU8>,
    config: Arc<RuntimePressureConfig>,
}

impl ResourcePressure {
    fn from_env() -> Arc<Self> {
        Self::new(
            Arc::new(AtomicU8::new(RuntimePressureState::Normal as u8)),
            Arc::new(RuntimePressureConfig::from_env()),
        )
    }

    pub(crate) fn new(state: Arc<AtomicU8>, config: Arc<RuntimePressureConfig>) -> Arc<Self> {
        Arc::new(Self { state, config })
    }

    pub(crate) fn state(&self) -> Arc<AtomicU8> {
        self.state.clone()
    }

    pub(crate) fn config(&self) -> Arc<RuntimePressureConfig> {
        self.config.clone()
    }
}

/// Coordinating facade for runtime-owned resources.
///
/// Logical KV allocation, physical CUDA pooling/admission, and pressure
/// monitoring remain independent components; this type only gives lifecycle
/// code one stable owner and a consistent per-process authority boundary.
pub(crate) struct RuntimeResources {
    kv: KvCoordinator,
    #[cfg(feature = "gpu-device-pool")]
    device_memory: Option<Arc<DeviceMemoryManager>>,
    pressure: Arc<ResourcePressure>,
}

impl RuntimeResources {
    pub(crate) fn new(device_info: &DeviceInfo) -> Result<Arc<Self>, String> {
        Ok(Arc::new(Self {
            kv: KvCoordinatorInner::new(device_info),
            #[cfg(feature = "gpu-device-pool")]
            device_memory: DeviceMemoryManager::from_env(device_info)?,
            pressure: ResourcePressure::from_env(),
        }))
    }

    pub(crate) fn kv(&self) -> &KvCoordinator {
        &self.kv
    }

    pub(crate) fn pressure(&self) -> &Arc<ResourcePressure> {
        &self.pressure
    }

    #[cfg(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    ))]
    pub(crate) fn device_pool(
        &self,
        device_id: usize,
    ) -> Option<Arc<kapsl_hal::gpu_arena::GpuDevicePool>> {
        self.device_memory
            .as_ref()
            .and_then(|manager| manager.pool(device_id))
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn uses_env_allocators(&self, device_id: usize) -> bool {
        self.device_memory
            .as_ref()
            .is_some_and(|manager| manager.has_pool(device_id))
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) async fn begin_device_memory_admission(
        &self,
        device_id: usize,
        model_id: u32,
        kind: EngineKind,
        planned_report: &ExternalDeviceMemoryReport,
    ) -> Result<Option<DeviceMemoryAdmission>, String> {
        let Some(manager) = self.device_memory.as_ref() else {
            return Ok(None);
        };
        manager
            .begin_admission(device_id, model_id, kind, planned_report)
            .await
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) async fn begin_device_memory_swap_admission(
        &self,
        device_id: usize,
        model_id: u32,
        planned_report: &ExternalDeviceMemoryReport,
    ) -> Result<Option<DeviceMemorySwapAdmission>, String> {
        let Some(manager) = self.device_memory.as_ref() else {
            return Ok(None);
        };
        manager
            .begin_swap_admission(device_id, model_id, planned_report)
            .await
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn attach_device_memory_metrics(
        &self,
        metrics: kapsl_monitor::metrics::KapslMetrics,
    ) {
        if let Some(manager) = self.device_memory.as_ref() {
            manager.attach_metrics(metrics);
        }
    }
}
