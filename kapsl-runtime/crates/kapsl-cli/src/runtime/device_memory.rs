use super::*;
use cudarc::driver::CudaDevice;
use kapsl_hal::gpu_arena::{GpuDevicePool, PoolOwner};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Runtime owner of the one stable backing allocation created for each CUDA
/// device. Backends receive cloned pool handles; none of them may replace or
/// register the allocation.
pub(crate) struct DeviceMemoryManager {
    pools: HashMap<usize, Arc<GpuDevicePool>>,
    admission_refs: Mutex<HashMap<(usize, PoolOwner), usize>>,
    model_admissions: Mutex<HashMap<u32, Vec<(usize, PoolOwner)>>>,
}

impl DeviceMemoryManager {
    /// Build all explicitly configured device pools and register them with ORT
    /// before any model session is constructed. With no pool-size setting the
    /// feature remains disabled and startup performs no CUDA allocation.
    pub(crate) fn from_env(device_info: &DeviceInfo) -> Result<Option<Arc<Self>>, String> {
        let mut pools = HashMap::new();
        for device in &device_info.devices {
            if !device.backend.to_string().eq_ignore_ascii_case("cuda") {
                continue;
            }
            let Some(raw_capacity) = configured_bytes(GPU_DEVICE_POOL_BYTES_ENV, device.id)? else {
                continue;
            };
            let physical_bytes = (device.memory_mb as usize).saturating_mul(1024 * 1024);
            let safe_max = effective_ceiling_bytes(device.id, physical_bytes, 0);
            if raw_capacity > safe_max {
                return Err(format!(
                    "{GPU_DEVICE_POOL_BYTES_ENV} for CUDA device {} is {} bytes, but the safe device budget is {} bytes after the required reserve",
                    device.id, raw_capacity, safe_max
                ));
            }

            let cuda = CudaDevice::new(device.id)
                .map_err(|error| format!("failed to open CUDA device {}: {error}", device.id))?;
            let pool = Arc::new(GpuDevicePool::new(cuda, raw_capacity).map_err(|error| {
                format!(
                    "failed to create {}-byte device pool on CUDA device {}: {error}",
                    raw_capacity, device.id
                )
            })?);
            kapsl_backends::ort_pool_allocator::register_pool_allocator(device.id as i32, &pool)?;
            log::info!(
                "[device-memory] CUDA device {}: runtime pool enabled, capacity={} bytes, safe_budget={} bytes",
                device.id,
                raw_capacity,
                safe_max
            );
            pools.insert(device.id, pool);
        }

        if pools.is_empty() {
            log::info!(
                "[device-memory] runtime GPU pool disabled; set {} to enable it",
                GPU_DEVICE_POOL_BYTES_ENV
            );
            return Ok(None);
        }

        Ok(Some(Arc::new(Self {
            pools,
            admission_refs: Mutex::new(HashMap::new()),
            model_admissions: Mutex::new(HashMap::new()),
        })))
    }

    #[cfg(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    ))]
    pub(crate) fn pool(&self, device_id: usize) -> Option<Arc<GpuDevicePool>> {
        self.pools.get(&device_id).cloned()
    }

    pub(crate) fn has_pool(&self, device_id: usize) -> bool {
        self.pools.contains_key(&device_id)
    }

    /// Protect this workload's configured guarantee during model load. The
    /// returned guard rolls admission back automatically if load fails before
    /// `commit` is called.
    pub(crate) fn begin_admission(
        self: &Arc<Self>,
        device_id: usize,
        model_id: u32,
        kind: EngineKind,
    ) -> Result<Option<DeviceMemoryAdmission>, String> {
        let Some(pool) = self.pools.get(&device_id) else {
            return Ok(None);
        };
        let owner = owner_for(kind, model_id);
        let (guaranteed, max) = configured_quota(pool, owner, device_id)?;
        let key = (device_id, owner);
        let mut refs = self.admission_refs.lock().unwrap();
        let count = refs.entry(key).or_default();
        if *count == 0 {
            pool.set_owner_quota(owner, guaranteed, max)
                .map_err(|error| format!("device-pool quota for {owner:?}: {error}"))?;
            pool.set_owner_admitted(owner, true)
                .map_err(|error| format!("device-pool admission for {owner:?}: {error}"))?;
        }
        *count += 1;
        drop(refs);
        log::info!(
            "[device-memory] admitted {:?} on CUDA device {}: guaranteed={} max={} currently_allocatable={} bytes",
            owner,
            device_id,
            guaranteed,
            max,
            pool.max_allocatable(owner, 1, 1)
        );
        Ok(Some(DeviceMemoryAdmission {
            manager: Arc::clone(self),
            model_id,
            device_id,
            owner,
            committed: false,
        }))
    }

    pub(crate) fn release_model(&self, model_id: u32) {
        let admissions = self
            .model_admissions
            .lock()
            .unwrap()
            .remove(&model_id)
            .unwrap_or_default();
        for (device_id, owner) in admissions {
            self.release_one(device_id, owner);
        }
    }

    fn commit_admission(&self, model_id: u32, device_id: usize, owner: PoolOwner) {
        self.model_admissions
            .lock()
            .unwrap()
            .entry(model_id)
            .or_default()
            .push((device_id, owner));
    }

    fn release_one(&self, device_id: usize, owner: PoolOwner) {
        let key = (device_id, owner);
        let should_unadmit = {
            let mut refs = self.admission_refs.lock().unwrap();
            let Some(count) = refs.get_mut(&key) else {
                return;
            };
            *count = count.saturating_sub(1);
            if *count == 0 {
                refs.remove(&key);
                true
            } else {
                false
            }
        };
        if !should_unadmit {
            return;
        }
        let Some(pool) = self.pools.get(&device_id) else {
            return;
        };
        match pool.set_owner_admitted(owner, false) {
            Ok(()) => log::info!(
                "[device-memory] released {:?} admission on CUDA device {}",
                owner,
                device_id
            ),
            Err(error) => log::warn!(
                "[device-memory] retaining {:?} reservation on CUDA device {} while allocations remain: {}",
                owner,
                device_id,
                error
            ),
        }
    }
}

#[must_use = "commit the admission after model load succeeds"]
pub(crate) struct DeviceMemoryAdmission {
    manager: Arc<DeviceMemoryManager>,
    model_id: u32,
    device_id: usize,
    owner: PoolOwner,
    committed: bool,
}

impl DeviceMemoryAdmission {
    pub(crate) fn commit(mut self) {
        self.manager
            .commit_admission(self.model_id, self.device_id, self.owner);
        self.committed = true;
    }
}

impl Drop for DeviceMemoryAdmission {
    fn drop(&mut self) {
        if !self.committed {
            self.manager.release_one(self.device_id, self.owner);
        }
    }
}

fn owner_for(kind: EngineKind, model_id: u32) -> PoolOwner {
    if kind.is_gguf() {
        PoolOwner::GgufKv { model_id }
    } else if kind == EngineKind::Native {
        PoolOwner::NativeKv { model_id }
    } else {
        PoolOwner::Onnx
    }
}

fn configured_quota(
    pool: &GpuDevicePool,
    owner: PoolOwner,
    device_id: usize,
) -> Result<(usize, usize), String> {
    let (guaranteed_name, max_name) = match owner {
        PoolOwner::Onnx => (GPU_ONNX_GUARANTEED_BYTES_ENV, GPU_ONNX_MAX_BYTES_ENV),
        PoolOwner::GgufKv { .. } => (GPU_GGUF_GUARANTEED_BYTES_ENV, GPU_GGUF_MAX_BYTES_ENV),
        PoolOwner::NativeKv { .. } => (GPU_NATIVE_GUARANTEED_BYTES_ENV, GPU_NATIVE_MAX_BYTES_ENV),
    };
    let guaranteed = configured_bytes(guaranteed_name, device_id)?.unwrap_or(0);
    let max = configured_bytes(max_name, device_id)?.unwrap_or(pool.capacity_bytes());
    if guaranteed > max || max > pool.capacity_bytes() {
        return Err(format!(
            "invalid quota for {owner:?} on device {device_id}: guaranteed={guaranteed}, max={max}, pool_capacity={}",
            pool.capacity_bytes()
        ));
    }
    Ok((guaranteed, max))
}

fn configured_bytes(name: &str, device_id: usize) -> Result<Option<usize>, String> {
    let per_device_name = format!("{name}_{device_id}");
    let Some(raw) = optional_env_var(&per_device_name).or_else(|| optional_env_var(name)) else {
        return Ok(None);
    };
    parse_cuda_memory_limit(&raw).map(Some).ok_or_else(|| {
        format!("invalid byte size `{raw}` for {name}; use a positive byte count or a k/m/g suffix")
    })
}
