use super::device_budget::DeviceBudgetLedger;
use super::*;
use cudarc::driver::{result as cuda_result, CudaDevice};
use kapsl_hal::gpu_arena::{GpuDevicePool, PoolOwner};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

const RELEASE_RETRY_INTERVAL: Duration = Duration::from_millis(25);
const RELEASE_RETRY_ATTEMPTS: usize = 400;

struct DeviceAuthority {
    cuda: Arc<CudaDevice>,
    load_lock: Arc<AsyncMutex<()>>,
}

#[derive(Debug)]
struct ExternalReservation {
    allocation_id: String,
    owns_charge: bool,
}

/// Runtime memory authority for each CUDA device.
///
/// Backends receive cloned elastic-pool handles where available, while this
/// manager retains the global device budget and accounts allocations that
/// cannot yet live in that pool (notably GGUF/native weights).
pub(crate) struct DeviceMemoryManager {
    devices: HashMap<usize, DeviceAuthority>,
    pools: HashMap<usize, Arc<GpuDevicePool>>,
    budget: Mutex<DeviceBudgetLedger>,
    admission_refs: Mutex<HashMap<(usize, PoolOwner), usize>>,
    next_fallback_allocation: AtomicU64,
    metrics: Mutex<Option<kapsl_monitor::metrics::KapslMetrics>>,
}

impl DeviceMemoryManager {
    /// Create one authority for every CUDA device. Explicitly configured
    /// elastic pools are built and registered with ORT before any model session
    /// is constructed; external-memory accounting remains active without a
    /// pool.
    pub(crate) fn from_env(device_info: &DeviceInfo) -> Result<Option<Arc<Self>>, String> {
        let mut devices = HashMap::new();
        let mut pools = HashMap::new();
        let mut budget = DeviceBudgetLedger::default();
        for device in &device_info.devices {
            if !device.backend.to_string().eq_ignore_ascii_case("cuda") {
                continue;
            }
            let physical_bytes = (device.memory_mb as usize).saturating_mul(1024 * 1024);
            let safe_budget = effective_ceiling_bytes(device.id, physical_bytes, 0);

            let cuda = CudaDevice::new(device.id)
                .map_err(|error| format!("failed to open CUDA device {}: {error}", device.id))?;
            let configured_pool = configured_bytes(GPU_DEVICE_POOL_BYTES_ENV, device.id)?;
            if let Some(raw_capacity) = configured_pool {
                if raw_capacity > safe_budget {
                    return Err(format!(
                        "{GPU_DEVICE_POOL_BYTES_ENV} for CUDA device {} is {} bytes, but the safe device budget is {} bytes after the required reserve",
                        device.id, raw_capacity, safe_budget
                    ));
                }
                let pool = Arc::new(GpuDevicePool::new(Arc::clone(&cuda), raw_capacity).map_err(
                    |error| {
                        format!(
                            "failed to create {}-byte device pool on CUDA device {}: {error}",
                            raw_capacity, device.id
                        )
                    },
                )?);
                kapsl_backends::ort_pool_allocator::register_pool_allocator(
                    device.id as i32,
                    &pool,
                )?;
                pools.insert(device.id, pool);
            }
            budget.insert_device(device.id, safe_budget, configured_pool.unwrap_or_default())?;
            log::info!(
                "[device-memory] CUDA device {}: runtime authority enabled, safe_budget={} pooled={} external_available={} bytes",
                device.id,
                safe_budget,
                configured_pool.unwrap_or_default(),
                safe_budget.saturating_sub(configured_pool.unwrap_or_default())
            );
            devices.insert(
                device.id,
                DeviceAuthority {
                    cuda,
                    load_lock: Arc::new(AsyncMutex::new(())),
                },
            );
        }

        if devices.is_empty() {
            return Ok(None);
        }
        if pools.is_empty() {
            log::info!(
                "[device-memory] elastic GPU pool disabled; external-memory accounting remains enabled (set {} to enable pooling)",
                GPU_DEVICE_POOL_BYTES_ENV
            );
        }

        Ok(Some(Arc::new(Self {
            devices,
            pools,
            budget: Mutex::new(budget),
            admission_refs: Mutex::new(HashMap::new()),
            next_fallback_allocation: AtomicU64::new(1),
            metrics: Mutex::new(None),
        })))
    }

    pub(crate) fn attach_metrics(&self, metrics: kapsl_monitor::metrics::KapslMetrics) {
        *self.metrics.lock().unwrap() = Some(metrics);
        let snapshots: Vec<_> = {
            let budget = self.budget.lock().unwrap();
            self.devices
                .keys()
                .filter_map(|&device_id| {
                    budget
                        .snapshot(device_id)
                        .map(|snapshot| (device_id, snapshot))
                })
                .collect()
        };
        for (device_id, snapshot) in snapshots {
            self.publish_metrics(device_id, snapshot);
        }
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

    /// Reserve this workload's planned external weight bytes and protect its
    /// configured elastic-pool guarantee during model load. Loads are
    /// serialized per device so the before/after CUDA samples can be attributed
    /// to one backend. The returned guard rolls both reservations back if load
    /// fails before `commit` is called.
    pub(crate) async fn begin_admission(
        self: &Arc<Self>,
        device_id: usize,
        model_id: u32,
        kind: EngineKind,
        planned_report: &ExternalDeviceMemoryReport,
    ) -> Result<Option<DeviceMemoryAdmission>, String> {
        let Some(authority) = self.devices.get(&device_id) else {
            return Ok(None);
        };
        let load_guard = Arc::clone(&authority.load_lock).lock_owned().await;
        let owner = owner_for(kind, model_id);
        let mut planned_allocations: Vec<_> = planned_report
            .allocations
            .iter()
            .filter(|allocation| allocation.device_id == device_id)
            .cloned()
            .collect();
        if planned_allocations.is_empty() {
            planned_allocations.push(ExternalDeviceMemory {
                allocation_id: format!(
                    "runtime-fallback:{device_id}:{}",
                    self.next_fallback_allocation
                        .fetch_add(1, Ordering::Relaxed)
                ),
                device_id,
                bytes: 0,
            });
        }
        let mut reservations = Vec::with_capacity(planned_allocations.len());
        let snapshot = {
            let mut budget = self.budget.lock().unwrap();
            let mut current = budget.snapshot(device_id).ok_or_else(|| {
                format!("CUDA device {device_id} has no runtime memory authority")
            })?;
            for allocation in &planned_allocations {
                match budget.reserve_external(
                    device_id,
                    &allocation.allocation_id,
                    allocation.bytes,
                ) {
                    Ok((snapshot, owns_charge)) => {
                        current = snapshot;
                        reservations.push(ExternalReservation {
                            allocation_id: allocation.allocation_id.clone(),
                            owns_charge,
                        });
                    }
                    Err(error) => {
                        for reservation in &reservations {
                            budget.release_external(device_id, &reservation.allocation_id);
                        }
                        return Err(error);
                    }
                }
            }
            current
        };
        self.publish_metrics(device_id, snapshot);
        if let Err(error) = self.admit_pool(device_id, owner) {
            self.release_external_reservations(device_id, &reservations);
            return Err(error);
        }
        let free_before_load = match self.free_device_bytes(device_id) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.release_external_reservations(device_id, &reservations);
                self.release_pool_one(device_id, owner);
                return Err(error);
            }
        };
        let planned_external_bytes = planned_allocations
            .iter()
            .map(|allocation| allocation.bytes)
            .sum::<usize>();
        log::info!(
            "[device-memory] admitted {:?} on CUDA device {}: planned_external_weights={} global_used={} global_available={} bytes",
            owner,
            device_id,
            planned_external_bytes,
            snapshot.used_bytes(),
            snapshot.available_bytes()
        );
        Ok(Some(DeviceMemoryAdmission {
            manager: Arc::clone(self),
            device_id,
            owner,
            reservations,
            free_before_load,
            _load_guard: Some(load_guard),
            reconciled: false,
            committed: false,
        }))
    }

    fn admit_pool(&self, device_id: usize, owner: PoolOwner) -> Result<(), String> {
        let Some(pool) = self.pools.get(&device_id) else {
            return Ok(());
        };
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
        log::info!(
            "[device-memory] elastic quota for {:?} on CUDA device {}: guaranteed={} max={} currently_allocatable={} bytes",
            owner,
            device_id,
            guaranteed,
            max,
            pool.max_allocatable(owner, 1, 1)
        );
        Ok(())
    }

    fn free_device_bytes(&self, device_id: usize) -> Result<usize, String> {
        let authority = self
            .devices
            .get(&device_id)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;
        authority
            .cuda
            .bind_to_thread()
            .map_err(|error| format!("failed to bind CUDA device {device_id}: {error}"))?;
        cuda_result::mem_get_info()
            .map(|(free, _total)| free)
            .map_err(|error| format!("failed to query CUDA device {device_id} memory: {error}"))
    }

    fn reconcile_external(
        &self,
        device_id: usize,
        allocation_id: &str,
        actual_bytes: usize,
    ) -> Result<(), String> {
        let result =
            self.budget
                .lock()
                .unwrap()
                .reconcile_external(device_id, allocation_id, actual_bytes);
        let snapshot = self
            .budget
            .lock()
            .unwrap()
            .snapshot(device_id)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;
        self.publish_metrics(device_id, snapshot);
        log::info!(
            "[device-memory] CUDA device {} backend allocation `{}`: actual_external={} global_used={} global_available={} bytes",
            device_id,
            allocation_id,
            actual_bytes,
            snapshot.used_bytes(),
            snapshot.available_bytes()
        );
        result.map(|_| ())
    }

    fn release_external_reservations(
        &self,
        device_id: usize,
        reservations: &[ExternalReservation],
    ) {
        let snapshot = {
            let mut budget = self.budget.lock().unwrap();
            for reservation in reservations {
                budget.release_external(device_id, &reservation.allocation_id);
            }
            budget.snapshot(device_id)
        };
        if let Some(snapshot) = snapshot {
            self.publish_metrics(device_id, snapshot);
        }
    }

    fn release_one(
        self: &Arc<Self>,
        device_id: usize,
        owner: PoolOwner,
        reservations: &[ExternalReservation],
    ) {
        self.release_external_reservations(device_id, reservations);
        self.release_pool_one(device_id, owner);
    }

    fn publish_metrics(
        &self,
        device_id: usize,
        snapshot: super::device_budget::DeviceBudgetSnapshot,
    ) {
        let metrics = self.metrics.lock().unwrap().clone();
        let Some(metrics) = metrics else {
            return;
        };
        let device = device_id.to_string();
        let labels = &[device.as_str()];
        let to_i64 = |bytes: usize| i64::try_from(bytes).unwrap_or(i64::MAX);
        metrics
            .device_memory_budget_bytes
            .with_label_values(labels)
            .set(to_i64(snapshot.budget_bytes));
        metrics
            .device_memory_pooled_bytes
            .with_label_values(labels)
            .set(to_i64(snapshot.pooled_bytes));
        metrics
            .device_memory_planned_external_bytes
            .with_label_values(labels)
            .set(to_i64(snapshot.planned_external_bytes));
        metrics
            .device_memory_external_bytes
            .with_label_values(labels)
            .set(to_i64(snapshot.external_bytes));
        metrics
            .device_memory_available_bytes
            .with_label_values(labels)
            .set(to_i64(snapshot.available_bytes()));
    }

    fn release_pool_one(self: &Arc<Self>, device_id: usize, owner: PoolOwner) {
        let key = (device_id, owner);
        let should_unadmit = {
            let mut refs = self.admission_refs.lock().unwrap();
            let Some(count) = refs.get_mut(&key) else {
                return;
            };
            *count = count.saturating_sub(1);
            *count == 0
        };
        if !should_unadmit {
            return;
        }

        if self.try_finish_release(device_id, owner) {
            return;
        }

        let usage = self
            .pools
            .get(&device_id)
            .map(|pool| pool.owner_usage_bytes(owner))
            .unwrap_or(0);
        log::info!(
            "[device-memory] deferring {:?} admission release on CUDA device {} until {} live bytes are freed",
            owner,
            device_id,
            usage
        );
        let manager = Arc::clone(self);
        std::thread::spawn(move || {
            for _ in 0..RELEASE_RETRY_ATTEMPTS {
                std::thread::sleep(RELEASE_RETRY_INTERVAL);
                if manager.try_finish_release(device_id, owner) {
                    return;
                }
            }
            let usage = manager
                .pools
                .get(&device_id)
                .map(|pool| pool.owner_usage_bytes(owner))
                .unwrap_or(0);
            log::warn!(
                "[device-memory] timed out releasing {:?} admission on CUDA device {}; {} bytes remain live",
                owner,
                device_id,
                usage
            );
        });
    }

    /// Complete a pending release once backend teardown has returned every
    /// allocation. The admission-ref lock serializes this with re-admission of
    /// the same model ID, preventing a delayed cleanup from unprotecting a
    /// newly started workload.
    fn try_finish_release(&self, device_id: usize, owner: PoolOwner) -> bool {
        let key = (device_id, owner);
        let mut refs = self.admission_refs.lock().unwrap();
        if refs.get(&key).copied().unwrap_or(0) != 0 {
            return true;
        }
        let Some(pool) = self.pools.get(&device_id) else {
            refs.remove(&key);
            return true;
        };
        if pool.owner_usage_bytes(owner) != 0 {
            return false;
        }
        match pool.set_owner_admitted(owner, false) {
            Ok(()) => {
                refs.remove(&key);
                log::info!(
                    "[device-memory] released {:?} admission on CUDA device {}",
                    owner,
                    device_id
                );
                true
            }
            Err(error) => {
                log::warn!(
                    "[device-memory] failed to release {:?} admission on CUDA device {}: {}",
                    owner,
                    device_id,
                    error
                );
                false
            }
        }
    }
}

#[must_use = "commit the admission after model load succeeds"]
pub(crate) struct DeviceMemoryAdmission {
    manager: Arc<DeviceMemoryManager>,
    device_id: usize,
    owner: PoolOwner,
    reservations: Vec<ExternalReservation>,
    free_before_load: usize,
    _load_guard: Option<OwnedMutexGuard<()>>,
    reconciled: bool,
    committed: bool,
}

impl DeviceMemoryAdmission {
    /// Sample the backend's actual external footprint, reconcile the planned
    /// reservation, and retain the resulting charge in this guard. Multi-device
    /// loads reconcile every guard before committing any of them, so one device
    /// rejecting the load rolls the whole set back.
    pub(crate) fn reconcile(
        &mut self,
        actual_report: &ExternalDeviceMemoryReport,
    ) -> Result<(), String> {
        let free_after_load = self.manager.free_device_bytes(self.device_id)?;
        let observed_external_bytes = self.free_before_load.saturating_sub(free_after_load);
        let mut actual_by_id = HashMap::<String, usize>::new();
        for allocation in actual_report
            .allocations
            .iter()
            .filter(|allocation| allocation.device_id == self.device_id)
        {
            let bytes = actual_by_id
                .entry(allocation.allocation_id.clone())
                .or_default();
            *bytes = bytes.saturating_add(allocation.bytes);
        }
        let mut newly_charged_reported_bytes = 0usize;

        for reservation in &self.reservations {
            let actual_bytes = actual_by_id.remove(&reservation.allocation_id).unwrap_or(0);
            if reservation.owns_charge {
                newly_charged_reported_bytes =
                    newly_charged_reported_bytes.saturating_add(actual_bytes);
                self.manager.reconcile_external(
                    self.device_id,
                    &reservation.allocation_id,
                    actual_bytes,
                )?;
            }
        }

        for (allocation_id, actual_bytes) in actual_by_id {
            let (_, owns_charge) = self.manager.budget.lock().unwrap().reserve_external(
                self.device_id,
                &allocation_id,
                0,
            )?;
            self.reservations.push(ExternalReservation {
                allocation_id: allocation_id.clone(),
                owns_charge,
            });
            if owns_charge {
                newly_charged_reported_bytes =
                    newly_charged_reported_bytes.saturating_add(actual_bytes);
                self.manager
                    .reconcile_external(self.device_id, &allocation_id, actual_bytes)?;
            }
        }

        let residual_bytes = observed_external_bytes.saturating_sub(newly_charged_reported_bytes);
        if residual_bytes > 0 {
            let allocation_id = format!(
                "runtime-observed:{}:{}",
                self.device_id,
                self.manager
                    .next_fallback_allocation
                    .fetch_add(1, Ordering::Relaxed)
            );
            let (_, owns_charge) = self.manager.budget.lock().unwrap().reserve_external(
                self.device_id,
                &allocation_id,
                0,
            )?;
            debug_assert!(owns_charge);
            self.reservations.push(ExternalReservation {
                allocation_id: allocation_id.clone(),
                owns_charge,
            });
            self.manager
                .reconcile_external(self.device_id, &allocation_id, residual_bytes)?;
        }
        self.reconciled = true;
        Ok(())
    }

    pub(crate) fn commit(mut self) -> DeviceMemoryLease {
        assert!(
            self.reconciled,
            "device memory admission must be reconciled before commit"
        );
        self.committed = true;
        self._load_guard.take();
        DeviceMemoryLease {
            manager: Arc::clone(&self.manager),
            device_id: self.device_id,
            owner: self.owner,
            reservations: std::mem::take(&mut self.reservations),
        }
    }
}

impl Drop for DeviceMemoryAdmission {
    fn drop(&mut self) {
        if !self.committed {
            self.manager
                .release_one(self.device_id, self.owner, &self.reservations);
        }
    }
}

pub(crate) struct DeviceMemoryLease {
    manager: Arc<DeviceMemoryManager>,
    device_id: usize,
    owner: PoolOwner,
    reservations: Vec<ExternalReservation>,
}

impl DeviceMemoryLease {
    pub(crate) fn reconcile_report(
        &mut self,
        report: &ExternalDeviceMemoryReport,
    ) -> Result<(), String> {
        let actual_by_id: HashMap<_, _> = report
            .allocations
            .iter()
            .filter(|allocation| allocation.device_id == self.device_id)
            .map(|allocation| (allocation.allocation_id.as_str(), allocation.bytes))
            .collect();
        for reservation in &self.reservations {
            if !reservation.owns_charge {
                continue;
            }
            if let Some(&actual_bytes) = actual_by_id.get(reservation.allocation_id.as_str()) {
                self.manager.reconcile_external(
                    self.device_id,
                    &reservation.allocation_id,
                    actual_bytes,
                )?;
            }
        }
        Ok(())
    }
}

impl Drop for DeviceMemoryLease {
    fn drop(&mut self) {
        self.manager
            .release_one(self.device_id, self.owner, &self.reservations);
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
