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
/// Logical KV allocation and pressure monitoring remain independent policy
/// components. All physical/accounting memory domains live below one
/// `MemoryAuthority` owned here.
pub(crate) struct RuntimeResources {
    kv: KvCoordinator,
    memory: Arc<MemoryAuthority>,
    priority: Arc<PriorityArbiter>,
    pressure: Arc<ResourcePressure>,
    managed_vllm: RwLock<Option<Arc<ManagedVllmDeployment>>>,
}

impl RuntimeResources {
    #[cfg_attr(feature = "gpu-device-pool", allow(dead_code))]
    pub(crate) fn new(device_info: &DeviceInfo) -> Result<Arc<Self>, String> {
        #[cfg(feature = "gpu-device-pool")]
        {
            Self::new_with_device_memory_plan(device_info, &DeviceMemoryBootstrapPlan::default())
        }
        #[cfg(not(feature = "gpu-device-pool"))]
        {
            let memory = MemoryAuthority::new(device_info)?;
            Ok(Arc::new(Self {
                kv: KvCoordinatorInner::new(memory.clone()),
                memory,
                priority: PriorityArbiter::new(),
                pressure: ResourcePressure::from_env(),
                managed_vllm: RwLock::new(None),
            }))
        }
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn new_with_device_memory_plan(
        device_info: &DeviceInfo,
        bootstrap: &DeviceMemoryBootstrapPlan,
    ) -> Result<Arc<Self>, String> {
        let memory = MemoryAuthority::new_with_cuda_plan(device_info, bootstrap)?;
        Ok(Arc::new(Self {
            kv: KvCoordinatorInner::new(memory.clone()),
            memory,
            priority: PriorityArbiter::new(),
            pressure: ResourcePressure::from_env(),
            managed_vllm: RwLock::new(None),
        }))
    }

    pub(crate) fn kv(&self) -> &KvCoordinator {
        &self.kv
    }

    pub(crate) fn pressure(&self) -> &Arc<ResourcePressure> {
        &self.pressure
    }

    pub(crate) fn memory(&self) -> &Arc<MemoryAuthority> {
        &self.memory
    }

    pub(crate) fn priority(&self) -> &Arc<PriorityArbiter> {
        &self.priority
    }

    pub(crate) fn install_managed_vllm(
        &self,
        deployment: Arc<ManagedVllmDeployment>,
    ) -> Result<(), String> {
        let mut current = self.managed_vllm.write();
        if current.is_some() {
            return Err("managed vLLM deployment is already configured".to_string());
        }
        *current = Some(deployment);
        Ok(())
    }

    pub(crate) fn managed_vllm(&self) -> Option<Arc<ManagedVllmDeployment>> {
        self.managed_vllm.read().clone()
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn device_pool(
        &self,
        device_id: usize,
    ) -> Option<Arc<kapsl_hal::gpu_arena::GpuDevicePool>> {
        self.memory.cuda_pool(device_id)
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn uses_env_allocators(&self, device_id: usize) -> bool {
        self.memory.uses_cuda_environment_allocator(device_id)
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn ensure_device_pools(
        &self,
        bootstrap: &DeviceMemoryBootstrapPlan,
    ) -> Result<(), String> {
        self.memory.ensure_cuda_pools(bootstrap)
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn attach_device_memory_metrics(
        &self,
        metrics: kapsl_monitor::metrics::KapslMetrics,
    ) {
        self.memory.attach_cuda_metrics(metrics);
    }

    pub(crate) fn refresh_device_pool_metrics(&self) {
        self.memory.refresh_cuda_pool_metrics();
    }
}
