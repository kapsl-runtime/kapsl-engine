use super::device_budget::{
    choose_auto_pool_capacity, AutoPoolSizingDecision, AutoPoolSizingInput, DeviceBudgetLedger,
};
use super::*;
use cudarc::driver::{result as cuda_result, CudaDevice};
use kapsl_hal::gpu_arena::{GpuDevicePool, PoolAllocationClass, PoolBackend, PoolOwner};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

const RELEASE_RETRY_INTERVAL: Duration = Duration::from_millis(25);
const RELEASE_RETRY_ATTEMPTS: usize = 400;

struct DeviceAuthority {
    cuda: Arc<CudaDevice>,
    load_lock: Arc<AsyncMutex<()>>,
    pool_init_lock: Mutex<()>,
    pool_mode: DevicePoolMode,
    unpooled_reserve_bytes: Option<usize>,
    auto_driver_reserve_bytes: usize,
    implicit_auto_attempted: AtomicBool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DevicePoolMode {
    Off,
    Fixed(usize),
    Auto { explicit: bool },
}

#[derive(Debug, Clone, Default)]
struct DeviceBootstrapDemand {
    wants_pool: bool,
    has_isolated_worker: bool,
    external_allocations: HashMap<String, usize>,
    pooled_allocations: HashMap<String, usize>,
}

impl DeviceBootstrapDemand {
    fn planned_external_bytes(&self) -> usize {
        self.external_allocations
            .values()
            .copied()
            .fold(0usize, usize::saturating_add)
    }

    fn minimum_pool_bytes(&self) -> usize {
        self.pooled_allocations
            .values()
            .copied()
            .fold(0usize, usize::saturating_add)
    }
}

/// Host-only demand summary built before a physical CUDA backing allocation is
/// chosen. Allocation IDs let duplicate immutable GGUF weights be counted once.
#[derive(Debug, Clone, Default)]
pub(crate) struct DeviceMemoryBootstrapPlan {
    devices: HashMap<usize, DeviceBootstrapDemand>,
}

impl DeviceMemoryBootstrapPlan {
    pub(crate) fn mark_pool_consumer(&mut self, device_id: usize) {
        self.devices.entry(device_id).or_default().wants_pool = true;
    }

    pub(crate) fn mark_isolated_worker(&mut self, device_id: usize) {
        self.devices
            .entry(device_id)
            .or_default()
            .has_isolated_worker = true;
    }

    pub(crate) fn add_external_allocation(
        &mut self,
        device_id: usize,
        allocation_id: impl Into<String>,
        bytes: usize,
    ) {
        let allocations = &mut self
            .devices
            .entry(device_id)
            .or_default()
            .external_allocations;
        allocations
            .entry(allocation_id.into())
            .and_modify(|current| *current = (*current).max(bytes))
            .or_insert(bytes);
    }

    pub(crate) fn add_pooled_allocation(
        &mut self,
        device_id: usize,
        allocation_id: impl Into<String>,
        bytes: usize,
    ) {
        let allocations = &mut self
            .devices
            .entry(device_id)
            .or_default()
            .pooled_allocations;
        allocations
            .entry(allocation_id.into())
            .and_modify(|current| *current = (*current).max(bytes))
            .or_insert(bytes);
    }

    fn demand(&self, device_id: usize) -> DeviceBootstrapDemand {
        self.devices.get(&device_id).cloned().unwrap_or_default()
    }
}

#[derive(Debug)]
struct ExternalReservation {
    allocation_id: String,
    memory_owner: MemoryOwner,
    owns_charge: bool,
}

/// Runtime memory authority for each CUDA device.
///
/// Backends receive cloned elastic-pool handles where available, while this
/// manager retains the global device budget and accounts allocations that
/// cannot yet live in that pool (notably llama.cpp weights and compute scratch).
pub(crate) struct DeviceMemoryManager {
    devices: HashMap<usize, DeviceAuthority>,
    pools: Mutex<HashMap<usize, Arc<GpuDevicePool>>>,
    budget: Mutex<DeviceBudgetLedger>,
    admission_refs: Mutex<HashMap<(usize, PoolOwner), usize>>,
    next_fallback_allocation: AtomicU64,
    metrics: Mutex<Option<kapsl_monitor::metrics::KapslMetrics>>,
}

impl DeviceMemoryManager {
    /// Create one authority for every CUDA device. Fixed and model-aware
    /// automatic pools are built and registered with ORT before any model
    /// session is constructed; external-memory accounting remains active when
    /// no physical pool is materialized.
    pub(crate) fn from_env_with_plan(
        device_info: &DeviceInfo,
        bootstrap: &DeviceMemoryBootstrapPlan,
    ) -> Result<Option<Arc<Self>>, String> {
        let mut devices = HashMap::new();
        let mut budget = DeviceBudgetLedger::default();
        let pooling_disabled = env_flag(GPU_DEVICE_POOL_DISABLED_ENV);
        if pooling_disabled {
            log::info!(
                "[device-memory] physical CUDA pooling disabled for this process by {} (admission accounting remains enabled)",
                GPU_DEVICE_POOL_DISABLED_ENV
            );
        }
        for device in &device_info.devices {
            if !device.backend.to_string().eq_ignore_ascii_case("cuda") {
                continue;
            }
            let physical_bytes = (device.memory_mb as usize).saturating_mul(1024 * 1024);
            let cuda = CudaDevice::new(device.id)
                .map_err(|error| format!("failed to open CUDA device {}: {error}", device.id))?;
            let pool_mode = resolve_device_pool_mode(device.id, pooling_disabled)?;
            let unpooled_reserve_bytes = if matches!(pool_mode, DevicePoolMode::Auto { .. }) {
                configured_nonnegative_bytes(GPU_DEVICE_POOL_UNPOOLED_RESERVE_BYTES_ENV, device.id)?
            } else {
                None
            };
            let (safe_budget, auto_driver_reserve_bytes) =
                if matches!(pool_mode, DevicePoolMode::Auto { .. }) {
                    auto_safe_budget(device.id, physical_bytes, &cuda)?
                } else {
                    (effective_ceiling_bytes(device.id, physical_bytes, 0), 0)
                };
            budget.insert_device(device.id, safe_budget, 0)?;
            log::info!(
                "[device-memory] CUDA device {}: runtime authority enabled, safe_budget={} bytes pool_mode={:?}",
                device.id,
                safe_budget,
                pool_mode,
            );
            devices.insert(
                device.id,
                DeviceAuthority {
                    cuda,
                    load_lock: Arc::new(AsyncMutex::new(())),
                    pool_init_lock: Mutex::new(()),
                    pool_mode,
                    unpooled_reserve_bytes,
                    auto_driver_reserve_bytes,
                    implicit_auto_attempted: AtomicBool::new(false),
                },
            );
        }

        if devices.is_empty() {
            return Ok(None);
        }
        let manager = Arc::new(Self {
            devices,
            pools: Mutex::new(HashMap::new()),
            budget: Mutex::new(budget),
            admission_refs: Mutex::new(HashMap::new()),
            next_fallback_allocation: AtomicU64::new(1),
            metrics: Mutex::new(None),
        });
        manager.materialize_configured_pools(bootstrap)?;
        if manager.pools.lock().unwrap().is_empty() {
            log::info!(
                "[device-memory] no physical GPU pool materialized; external-memory accounting remains enabled"
            );
        }
        Ok(Some(manager))
    }

    fn materialize_configured_pools(
        &self,
        bootstrap: &DeviceMemoryBootstrapPlan,
    ) -> Result<(), String> {
        for (&device_id, authority) in &self.devices {
            let demand = bootstrap.demand(device_id);
            if matches!(
                authority.pool_mode,
                DevicePoolMode::Auto { explicit: false }
            ) && demand.has_isolated_worker
            {
                // A parent-owned backing allocation would compete with the
                // child before it can load weights. Keep the existing worker
                // isolation contract: only an explicit operator pool policy
                // may override this conservative suppression.
                authority
                    .implicit_auto_attempted
                    .store(true, Ordering::Release);
                log::info!(
                    "[device-memory] implicit parent CUDA pool suppressed on device {} because an isolated model worker targets that device",
                    device_id
                );
            }
        }

        // Validate every deterministic sizing decision before registering the
        // first process-global ORT allocator. Physical allocation failures can
        // still happen later if live free VRAM changes, but a bad configuration
        // cannot leave startup half-registered across devices.
        for (&device_id, authority) in &self.devices {
            let demand = bootstrap.demand(device_id);
            match authority.pool_mode {
                DevicePoolMode::Fixed(capacity) => {
                    self.validate_fixed_pool(device_id, capacity, &demand)?;
                }
                DevicePoolMode::Auto { explicit }
                    if explicit || (demand.wants_pool && !demand.has_isolated_worker) =>
                {
                    let (_, decision) = self.auto_pool_sizing_decision(device_id, &demand)?;
                    if explicit && decision.capacity_bytes.is_none() {
                        return Err(format!(
                            "automatic CUDA pool sizing failed on device {device_id}: {}",
                            decision
                                .reason
                                .unwrap_or_else(|| "no viable capacity".to_string())
                        ));
                    }
                }
                DevicePoolMode::Off | DevicePoolMode::Auto { .. } => {}
            }
        }

        for (&device_id, authority) in &self.devices {
            let demand = bootstrap.demand(device_id);
            match authority.pool_mode {
                DevicePoolMode::Off => {}
                DevicePoolMode::Fixed(capacity) => {
                    self.materialize_pool(device_id, capacity, false)?;
                }
                DevicePoolMode::Auto { explicit }
                    if explicit || (demand.wants_pool && !demand.has_isolated_worker) =>
                {
                    self.materialize_auto_pool(device_id, &demand, explicit)?;
                }
                DevicePoolMode::Auto { .. } => {}
            }
        }
        Ok(())
    }

    fn validate_fixed_pool(
        &self,
        device_id: usize,
        capacity: usize,
        demand: &DeviceBootstrapDemand,
    ) -> Result<(), String> {
        let planned = demand.planned_external_bytes();
        let minimum_pool = demand.minimum_pool_bytes();
        let safe_budget = self
            .budget
            .lock()
            .unwrap()
            .snapshot(device_id)
            .map(|snapshot| snapshot.budget_bytes)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;
        if capacity.saturating_add(planned) > safe_budget {
            return Err(format!(
                "fixed CUDA pool on device {device_id} ({capacity} bytes) plus planned external weights ({planned} bytes) exceeds the {safe_budget}-byte safe budget"
            ));
        }
        if capacity < minimum_pool {
            return Err(format!(
                "fixed CUDA pool on device {device_id} ({capacity} bytes) is smaller than the {minimum_pool}-byte planned pooled model footprint"
            ));
        }
        Ok(())
    }

    /// Materialize a deferred implicit-auto pool before constructing the first
    /// backend/session targeting that device. Existing pools remain immutable.
    pub(crate) fn ensure_pools_for_plan(
        &self,
        bootstrap: &DeviceMemoryBootstrapPlan,
    ) -> Result<(), String> {
        for (&device_id, demand) in &bootstrap.devices {
            if !demand.wants_pool || demand.has_isolated_worker || self.has_pool(device_id) {
                continue;
            }
            let Some(authority) = self.devices.get(&device_id) else {
                continue;
            };
            match authority.pool_mode {
                DevicePoolMode::Fixed(capacity) => {
                    self.validate_fixed_pool(device_id, capacity, demand)?;
                    self.materialize_pool(device_id, capacity, false)?;
                }
                DevicePoolMode::Auto { explicit } => {
                    self.materialize_auto_pool(device_id, demand, explicit)?;
                }
                DevicePoolMode::Off => {}
            }
        }
        Ok(())
    }

    fn materialize_auto_pool(
        &self,
        device_id: usize,
        demand: &DeviceBootstrapDemand,
        explicit: bool,
    ) -> Result<(), String> {
        let authority = self
            .devices
            .get(&device_id)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;
        // Serialize the sizing decision as well as allocation/registration.
        // Concurrent first model starts must not size competing immutable
        // pools from different demand snapshots.
        let _init_guard = authority.pool_init_lock.lock().unwrap();
        if self.has_pool(device_id) {
            return Ok(());
        }
        if !explicit
            && authority
                .implicit_auto_attempted
                .swap(true, Ordering::AcqRel)
        {
            // An implicit-auto decision is process-lifetime. In particular,
            // do not retry a declined aggregate startup decision later with a
            // smaller single-model view that omits its peers' weights.
            return Ok(());
        }
        let (safe_budget, decision) = self.auto_pool_sizing_decision(device_id, demand)?;
        let Some(capacity) = decision.capacity_bytes else {
            let reason = decision
                .reason
                .unwrap_or_else(|| "no viable capacity".to_string());
            if explicit {
                return Err(format!(
                    "automatic CUDA pool sizing failed on device {device_id}: {reason}"
                ));
            }
            log::warn!(
                "[device-memory] automatic CUDA pool disabled on device {}: {} (safe_budget={} planned_external={} unpooled_reserve={})",
                device_id,
                reason,
                safe_budget,
                decision.planned_external_bytes,
                decision.unpooled_reserve_bytes,
            );
            return Ok(());
        };
        log::info!(
            "[device-memory] automatic CUDA pool sizing on device {}: safe_budget={} planned_external={} minimum_pooled={} unpooled_reserve={} pooled={} bytes",
            device_id,
            safe_budget,
            decision.planned_external_bytes,
            demand.minimum_pool_bytes(),
            decision.unpooled_reserve_bytes,
            capacity,
        );
        self.materialize_pool_locked(device_id, capacity, !explicit)
    }

    fn auto_pool_sizing_decision(
        &self,
        device_id: usize,
        demand: &DeviceBootstrapDemand,
    ) -> Result<(usize, AutoPoolSizingDecision), String> {
        let snapshot = self
            .budget
            .lock()
            .unwrap()
            .snapshot(device_id)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;
        let already_charged = snapshot
            .planned_external_bytes
            .saturating_add(snapshot.external_bytes);
        let planned_external = already_charged.saturating_add(demand.planned_external_bytes());
        let authority = self
            .devices
            .get(&device_id)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;
        // A no-model process may defer allocation for hours. Refresh physical
        // availability at the one-time materialization point, while adding
        // already-ledgered bytes back before applying the same accounting so
        // they are not subtracted twice.
        let current_free = self.free_device_bytes(device_id)?;
        let live_adjusted_budget = current_free
            .saturating_sub(authority.auto_driver_reserve_bytes)
            .saturating_add(already_charged);
        let sizing_budget = snapshot.budget_bytes.min(live_adjusted_budget);
        let decision = choose_auto_pool_capacity(AutoPoolSizingInput {
            safe_budget_bytes: sizing_budget,
            planned_external_bytes: planned_external,
            minimum_pool_bytes: demand.minimum_pool_bytes(),
            unpooled_reserve_bytes: authority.unpooled_reserve_bytes,
        });
        Ok((sizing_budget, decision))
    }

    fn materialize_pool(
        &self,
        device_id: usize,
        capacity: usize,
        allow_allocation_fallback: bool,
    ) -> Result<(), String> {
        let authority = self
            .devices
            .get(&device_id)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;
        let _init_guard = authority.pool_init_lock.lock().unwrap();
        self.materialize_pool_locked(device_id, capacity, allow_allocation_fallback)
    }

    /// Materialize while the caller holds this device's `pool_init_lock`.
    fn materialize_pool_locked(
        &self,
        device_id: usize,
        capacity: usize,
        allow_allocation_fallback: bool,
    ) -> Result<(), String> {
        if self.has_pool(device_id) {
            return Ok(());
        }
        let authority = self
            .devices
            .get(&device_id)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;

        let previous = self
            .budget
            .lock()
            .unwrap()
            .snapshot(device_id)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;
        let charged = self
            .budget
            .lock()
            .unwrap()
            .set_pooled_bytes(device_id, capacity)?;
        let pool = match GpuDevicePool::new(Arc::clone(&authority.cuda), capacity) {
            Ok(pool) => Arc::new(pool),
            Err(error) => {
                let _ = self
                    .budget
                    .lock()
                    .unwrap()
                    .set_pooled_bytes(device_id, previous.pooled_bytes);
                if allow_allocation_fallback {
                    log::warn!(
                        "[device-memory] automatic {}-byte CUDA pool allocation failed on device {}; continuing without a runtime-owned pool: {}",
                        capacity,
                        device_id,
                        error
                    );
                    return Ok(());
                }
                return Err(format!(
                    "failed to create {capacity}-byte device pool on CUDA device {device_id}: {error}"
                ));
            }
        };
        if let Err(error) =
            kapsl_backends::ort_pool_allocator::register_pool_allocator(device_id as i32, &pool)
        {
            let _ = self
                .budget
                .lock()
                .unwrap()
                .set_pooled_bytes(device_id, previous.pooled_bytes);
            return Err(error);
        }
        self.pools.lock().unwrap().insert(device_id, pool);
        self.publish_metrics(device_id, charged);
        Ok(())
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
        self.refresh_pool_metrics();
    }

    /// Refresh inference-time allocator metrics immediately before a
    /// Prometheus scrape. Budget metrics are event-driven, but ORT and KV
    /// suballocations can change on every inference, so they must be sampled
    /// from the live pool instead of waiting for another model lifecycle event.
    pub(crate) fn refresh_pool_metrics(&self) {
        let metrics = self.metrics.lock().unwrap().clone();
        let Some(metrics) = metrics else {
            return;
        };
        let pools: Vec<_> = self
            .pools
            .lock()
            .unwrap()
            .iter()
            .map(|(&device_id, pool)| (device_id, Arc::clone(pool)))
            .collect();
        for (device_id, pool) in pools {
            let snapshot = pool.snapshot();
            metrics.set_gpu_device_pool_metrics(
                &device_id.to_string(),
                &pool_snapshot_metrics(snapshot),
            );
        }
    }

    #[cfg(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    ))]
    pub(crate) fn pool(&self, device_id: usize) -> Option<Arc<GpuDevicePool>> {
        self.pools.lock().unwrap().get(&device_id).cloned()
    }

    pub(crate) fn has_pool(&self, device_id: usize) -> bool {
        self.pools.lock().unwrap().contains_key(&device_id)
    }

    /// Admit one request-lifetime CUDA claim without taking the model-load
    /// sampling lock. Pool-managed rows are already bounded by the admitted
    /// workload quota. Backend-managed rows receive a unique ledger ID so
    /// concurrent requests are charged additively rather than mistaken for a
    /// shared persistent allocation.
    pub(crate) fn admit_transient(
        self: &Arc<Self>,
        claim: &MemoryClaim,
    ) -> Result<Option<DeviceMemoryTransientLease>, String> {
        let MemoryDomain::Cuda { device_id } = claim.domain else {
            return Err(format!(
                "device memory manager cannot admit a {} claim",
                claim.domain
            ));
        };
        if claim.bytes == 0 {
            return Ok(None);
        }
        if matches!(claim.source, MemoryClaimSource::Runtime { .. }) {
            if !self.has_pool(device_id) {
                return Err(format!(
                    "runtime-managed CUDA claim for {} targets device {} without a physical pool",
                    claim.owner, device_id
                ));
            }
            return Ok(None);
        }

        let reported_id = match &claim.source {
            MemoryClaimSource::External { allocation_id } => allocation_id.as_str(),
            MemoryClaimSource::Runtime { .. } => unreachable!("handled above"),
        };
        let allocation_id = format!(
            "request:{}:{}:{}:{}:{}",
            claim.owner.model_id,
            claim.owner.replica_id,
            device_id,
            self.next_fallback_allocation
                .fetch_add(1, Ordering::Relaxed),
            reported_id
        );
        let snapshot = {
            let mut budget = self.budget.lock().unwrap();
            let (snapshot, owns_charge) = budget.reserve_external(
                device_id,
                &allocation_id,
                claim.bytes,
                claim.owner,
                claim.class,
            )?;
            debug_assert!(owns_charge, "request allocation IDs are unique");
            snapshot
        };
        self.publish_metrics(device_id, snapshot);
        Ok(Some(DeviceMemoryTransientLease {
            manager: Arc::clone(self),
            device_id,
            reservation: ExternalReservation {
                allocation_id,
                memory_owner: claim.owner,
                owns_charge: true,
            },
        }))
    }

    /// Reserve this workload's planned external weight bytes and protect its
    /// configured elastic-pool guarantee during model load. Loads are
    /// serialized per device so the before/after CUDA samples can be attributed
    /// to one backend. The returned guard rolls both reservations back if load
    /// fails before `commit` is called.
    pub(crate) async fn begin_admission(
        self: &Arc<Self>,
        device_id: usize,
        memory_owner: MemoryOwner,
        kind: EngineKind,
        planned_report: &ExternalDeviceMemoryReport,
    ) -> Result<Option<DeviceMemoryAdmission>, String> {
        let Some(authority) = self.devices.get(&device_id) else {
            return Ok(None);
        };
        let load_guard = Arc::clone(&authority.load_lock).lock_owned().await;
        let pool_owner = pool_owner_for(kind, memory_owner);
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
                    memory_owner,
                    classify_external_allocation(&allocation.allocation_id),
                ) {
                    Ok((snapshot, owns_charge)) => {
                        current = snapshot;
                        reservations.push(ExternalReservation {
                            allocation_id: allocation.allocation_id.clone(),
                            memory_owner,
                            owns_charge,
                        });
                    }
                    Err(error) => {
                        for reservation in &reservations {
                            budget.release_external(
                                device_id,
                                &reservation.allocation_id,
                                reservation.memory_owner,
                            );
                        }
                        return Err(error);
                    }
                }
            }
            current
        };
        self.publish_metrics(device_id, snapshot);
        if let Err(error) = self.admit_pool(device_id, pool_owner) {
            self.release_external_reservations(device_id, &reservations);
            return Err(error);
        }
        let free_before_load = match self.free_device_bytes(device_id) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.release_external_reservations(device_id, &reservations);
                self.release_pool_one(device_id, pool_owner);
                return Err(error);
            }
        };
        let planned_external_bytes = planned_allocations
            .iter()
            .map(|allocation| allocation.bytes)
            .sum::<usize>();
        log::info!(
            "[device-memory] admitted {} ({:?}) on CUDA device {}: planned_external={} global_used={} global_available={} bytes",
            memory_owner,
            pool_owner,
            device_id,
            planned_external_bytes,
            snapshot.used_bytes(),
            snapshot.available_bytes()
        );
        Ok(Some(DeviceMemoryAdmission {
            manager: Arc::clone(self),
            device_id,
            memory_owner,
            pool_owner,
            reservations,
            free_before_load,
            _load_guard: Some(load_guard),
            reconciled: false,
            committed: false,
        }))
    }

    /// Reserve the target model's full external footprint while a hot-swap
    /// uploads it alongside the active model. The temporary allocation IDs
    /// intentionally differ from the backend's stable weight IDs: during
    /// activation both copies are live, so shared-allocation de-duplication
    /// must not hide the peak.
    pub(crate) async fn begin_swap_admission(
        self: &Arc<Self>,
        device_id: usize,
        memory_owner: MemoryOwner,
        planned_report: &ExternalDeviceMemoryReport,
    ) -> Result<Option<DeviceMemorySwapAdmission>, String> {
        let Some(authority) = self.devices.get(&device_id) else {
            return Ok(None);
        };
        let planned_allocations: Vec<_> = planned_report
            .allocations
            .iter()
            .filter(|allocation| allocation.device_id == device_id)
            .collect();
        if planned_allocations.is_empty() {
            return Ok(None);
        }

        let load_guard = Arc::clone(&authority.load_lock).lock_owned().await;
        let swap_id = self
            .next_fallback_allocation
            .fetch_add(1, Ordering::Relaxed);
        let mut reservations = Vec::with_capacity(planned_allocations.len());
        let snapshot = {
            let mut budget = self.budget.lock().unwrap();
            let mut current = budget.snapshot(device_id).ok_or_else(|| {
                format!("CUDA device {device_id} has no runtime memory authority")
            })?;
            for (index, allocation) in planned_allocations.iter().enumerate() {
                let allocation_id = format!(
                    "runtime-swap-peak:{}:{}:{device_id}:{swap_id}:{index}",
                    memory_owner.model_id, memory_owner.replica_id
                );
                match budget.reserve_external(
                    device_id,
                    &allocation_id,
                    allocation.bytes,
                    memory_owner,
                    classify_external_allocation(&allocation.allocation_id),
                ) {
                    Ok((snapshot, owns_charge)) => {
                        debug_assert!(owns_charge);
                        current = snapshot;
                        reservations.push(ExternalReservation {
                            allocation_id,
                            memory_owner,
                            owns_charge,
                        });
                    }
                    Err(error) => {
                        for reservation in &reservations {
                            budget.release_external(
                                device_id,
                                &reservation.allocation_id,
                                reservation.memory_owner,
                            );
                        }
                        return Err(format!(
                            "hot-swap memory admission for {memory_owner} failed: {error}"
                        ));
                    }
                }
            }
            current
        };
        self.publish_metrics(device_id, snapshot);
        log::info!(
            "[device-memory] admitted hot-swap peak for {} on CUDA device {}: target_external={} global_used={} global_available={} bytes",
            memory_owner,
            device_id,
            planned_allocations
                .iter()
                .map(|allocation| allocation.bytes)
                .sum::<usize>(),
            snapshot.used_bytes(),
            snapshot.available_bytes()
        );
        Ok(Some(DeviceMemorySwapAdmission {
            manager: Arc::clone(self),
            device_id,
            reservations,
            _load_guard: load_guard,
        }))
    }

    fn admit_pool(&self, device_id: usize, owner: PoolOwner) -> Result<(), String> {
        let Some(pool) = self.pools.lock().unwrap().get(&device_id).cloned() else {
            return Ok(());
        };
        let (guaranteed, max) = configured_quota(&pool, owner, device_id)?;
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
                budget.release_external(
                    device_id,
                    &reservation.allocation_id,
                    reservation.memory_owner,
                );
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
            .lock()
            .unwrap()
            .get(&device_id)
            .map(|pool| pool.workload_usage_bytes(owner))
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
                .lock()
                .unwrap()
                .get(&device_id)
                .map(|pool| pool.workload_usage_bytes(owner))
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
        let Some(pool) = self.pools.lock().unwrap().get(&device_id).cloned() else {
            refs.remove(&key);
            return true;
        };
        if pool.workload_usage_bytes(owner) != 0 {
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
                drop(refs);
                self.try_reclaim_pool(device_id);
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

    /// Drop an idle backing allocation after the final admitted consumer has
    /// torn down. ORT must be unregistered before its stable allocator Box and
    /// pool Arc can be released.
    fn try_reclaim_pool(&self, device_id: usize) {
        if self
            .admission_refs
            .lock()
            .unwrap()
            .keys()
            .any(|(candidate, _)| *candidate == device_id)
        {
            return;
        }
        let Some(authority) = self.devices.get(&device_id) else {
            return;
        };
        let _init_guard = authority.pool_init_lock.lock().unwrap();
        let Some(pool) = self.pools.lock().unwrap().get(&device_id).cloned() else {
            return;
        };
        if pool.free_bytes() != pool.capacity_bytes() {
            return;
        }
        if let Err(error) =
            kapsl_backends::ort_pool_allocator::unregister_pool_allocator(device_id as i32, &pool)
        {
            log::warn!(
                "[device-memory] cannot reclaim idle CUDA pool on device {}: {}",
                device_id,
                error
            );
            return;
        }
        self.pools.lock().unwrap().remove(&device_id);
        authority
            .implicit_auto_attempted
            .store(false, Ordering::Release);
        let snapshot = self.budget.lock().unwrap().set_pooled_bytes(device_id, 0);
        if let Ok(snapshot) = snapshot {
            self.publish_metrics(device_id, snapshot);
        }
        log::info!(
            "[device-memory] reclaimed idle CUDA pool backing on device {}",
            device_id
        );
    }
}

#[must_use = "commit the admission after model load succeeds"]
pub(crate) struct DeviceMemoryAdmission {
    manager: Arc<DeviceMemoryManager>,
    device_id: usize,
    memory_owner: MemoryOwner,
    pool_owner: PoolOwner,
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
                self.memory_owner,
                classify_external_allocation(&allocation_id),
            )?;
            self.reservations.push(ExternalReservation {
                allocation_id: allocation_id.clone(),
                memory_owner: self.memory_owner,
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
                self.memory_owner,
                MemoryAllocationClass::ExternallyOwned,
            )?;
            debug_assert!(owns_charge);
            self.reservations.push(ExternalReservation {
                allocation_id: allocation_id.clone(),
                memory_owner: self.memory_owner,
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
            memory_owner: self.memory_owner,
            pool_owner: self.pool_owner,
            reservations: std::mem::take(&mut self.reservations),
        }
    }
}

impl Drop for DeviceMemoryAdmission {
    fn drop(&mut self) {
        if !self.committed {
            self.manager
                .release_one(self.device_id, self.pool_owner, &self.reservations);
        }
    }
}

/// Short-lived reservation for the second set of weights that exists during
/// hot-swap activation. Dropping it releases both the peak charge and the
/// per-device load/swap serialization guard.
#[must_use = "hold the swap admission until backend activation finishes"]
pub(crate) struct DeviceMemorySwapAdmission {
    manager: Arc<DeviceMemoryManager>,
    device_id: usize,
    reservations: Vec<ExternalReservation>,
    _load_guard: OwnedMutexGuard<()>,
}

impl Drop for DeviceMemorySwapAdmission {
    fn drop(&mut self) {
        self.manager
            .release_external_reservations(self.device_id, &self.reservations);
    }
}

pub(crate) struct DeviceMemoryLease {
    manager: Arc<DeviceMemoryManager>,
    device_id: usize,
    memory_owner: MemoryOwner,
    pool_owner: PoolOwner,
    reservations: Vec<ExternalReservation>,
}

/// Request-lifetime reservation for CUDA memory allocated outside the pool.
pub(crate) struct DeviceMemoryTransientLease {
    manager: Arc<DeviceMemoryManager>,
    device_id: usize,
    reservation: ExternalReservation,
}

impl Drop for DeviceMemoryTransientLease {
    fn drop(&mut self) {
        self.manager
            .release_external_reservations(self.device_id, std::slice::from_ref(&self.reservation));
    }
}

impl DeviceMemoryLease {
    pub(crate) fn owner(&self) -> MemoryOwner {
        self.memory_owner
    }

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
            .release_one(self.device_id, self.pool_owner, &self.reservations);
    }
}

fn pool_owner_for(kind: EngineKind, owner: MemoryOwner) -> PoolOwner {
    if kind.is_gguf() {
        PoolOwner::gguf(
            owner.model_id,
            owner.replica_id,
            PoolAllocationClass::PersistentWeights,
        )
    } else if kind == EngineKind::Native {
        PoolOwner::native(
            owner.model_id,
            owner.replica_id,
            PoolAllocationClass::PersistentWeights,
        )
    } else {
        PoolOwner::onnx(
            owner.model_id,
            owner.replica_id,
            PoolAllocationClass::PersistentWeights,
        )
    }
}

fn pool_owner_metric_label(owner: PoolOwner) -> String {
    let backend = match owner.backend() {
        PoolBackend::Onnx => "onnx",
        PoolBackend::Gguf => "gguf",
        PoolBackend::Native => "native",
    };
    let class = match owner.class() {
        PoolAllocationClass::PersistentWeights => "persistent-weights",
        PoolAllocationClass::KvCache => "kv-cache",
        PoolAllocationClass::TransientWorkspace => "transient-workspace",
        PoolAllocationClass::BlockTable => "block-table",
        PoolAllocationClass::RequestTransient => "request-transient",
        PoolAllocationClass::ExternallyOwned => "externally-owned",
    };
    match (owner.model_id(), owner.replica_id()) {
        (Some(model_id), Some(replica_id)) => {
            format!("{backend}:{model_id}:{replica_id}:{class}")
        }
        _ => format!("{backend}:unattributed:{class}"),
    }
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn pool_snapshot_metrics(
    snapshot: kapsl_hal::gpu_arena::GpuDevicePoolSnapshot,
) -> kapsl_monitor::metrics::GpuDevicePoolMetrics {
    let owners = snapshot
        .owners
        .into_iter()
        .map(|owner| kapsl_monitor::metrics::GpuDevicePoolOwnerMetrics {
            owner: pool_owner_metric_label(owner.owner),
            usage_bytes: usize_to_u64(owner.usage_bytes),
            guaranteed_bytes: usize_to_u64(owner.guaranteed_bytes),
            max_bytes: usize_to_u64(owner.max_bytes),
            admitted: owner.admitted,
            allocatable_bytes: usize_to_u64(owner.allocatable_bytes),
        })
        .collect();
    kapsl_monitor::metrics::GpuDevicePoolMetrics {
        allocated_bytes: usize_to_u64(snapshot.allocated_bytes),
        live_allocations: usize_to_u64(snapshot.live_allocation_count),
        free_bytes: usize_to_u64(snapshot.free_bytes),
        free_ranges: usize_to_u64(snapshot.free_range_count),
        largest_free_range_bytes: usize_to_u64(snapshot.largest_free_range_bytes),
        fragmentation_ratio: snapshot.fragmentation_ratio,
        owners,
    }
}

fn configured_quota(
    pool: &GpuDevicePool,
    owner: PoolOwner,
    device_id: usize,
) -> Result<(usize, usize), String> {
    let (guaranteed_name, max_name) = match owner.backend() {
        PoolBackend::Onnx => (GPU_ONNX_GUARANTEED_BYTES_ENV, GPU_ONNX_MAX_BYTES_ENV),
        PoolBackend::Gguf => (GPU_GGUF_GUARANTEED_BYTES_ENV, GPU_GGUF_MAX_BYTES_ENV),
        PoolBackend::Native => (GPU_NATIVE_GUARANTEED_BYTES_ENV, GPU_NATIVE_MAX_BYTES_ENV),
    };
    let max = configured_bytes(max_name, device_id)?.unwrap_or(pool.capacity_bytes());
    // Protect a useful share for each admitted backend by default. Four-way
    // sharing covers the common ORT + multiple KV-owner deployment while
    // explicit per-owner settings retain full control (including zero).
    let guaranteed = configured_bytes(guaranteed_name, device_id)?
        .unwrap_or_else(|| (pool.capacity_bytes() / 4).min(max));
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

fn configured_value(name: &str, device_id: usize) -> Option<(String, String)> {
    let per_device_name = format!("{name}_{device_id}");
    if let Some(value) = optional_env_var(&per_device_name) {
        return Some((per_device_name, value));
    }
    optional_env_var(name).map(|value| (name.to_string(), value))
}

fn parse_nonnegative_cuda_bytes(raw: &str) -> Option<usize> {
    let lowered = raw.trim().to_ascii_lowercase();
    if lowered.is_empty() {
        return None;
    }
    let digits = lowered.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    let multiplier = match &lowered[digits.len()..] {
        "" | "b" => 1usize,
        "k" | "kb" => 1024,
        "m" | "mb" => 1024 * 1024,
        "g" | "gb" => 1024 * 1024 * 1024,
        _ => return None,
    };
    digits
        .parse::<usize>()
        .ok()
        .and_then(|amount| amount.checked_mul(multiplier))
}

fn configured_nonnegative_bytes(name: &str, device_id: usize) -> Result<Option<usize>, String> {
    let Some((resolved_name, raw)) = configured_value(name, device_id) else {
        return Ok(None);
    };
    parse_nonnegative_cuda_bytes(&raw).map(Some).ok_or_else(|| {
        format!(
            "invalid byte size `{raw}` for {resolved_name}; use a non-negative byte count or a k/m/g suffix"
        )
    })
}

fn resolve_device_pool_mode(
    device_id: usize,
    pooling_disabled: bool,
) -> Result<DevicePoolMode, String> {
    if pooling_disabled {
        return Ok(DevicePoolMode::Off);
    }
    let bytes = configured_bytes(GPU_DEVICE_POOL_BYTES_ENV, device_id)?;
    let configured_mode = configured_value(GPU_DEVICE_POOL_MODE_ENV, device_id);
    resolve_device_pool_mode_values(
        pooling_disabled,
        bytes,
        configured_mode
            .as_ref()
            .map(|(name, value)| (name.as_str(), value.as_str())),
        cfg!(feature = "gguf-cuda-shared-kv"),
    )
}

fn resolve_device_pool_mode_values(
    pooling_disabled: bool,
    bytes: Option<usize>,
    configured_mode: Option<(&str, &str)>,
    implicit_auto_enabled: bool,
) -> Result<DevicePoolMode, String> {
    if pooling_disabled {
        return Ok(DevicePoolMode::Off);
    }
    let Some((resolved_name, raw_mode)) = configured_mode else {
        return Ok(match bytes {
            Some(capacity) => DevicePoolMode::Fixed(capacity),
            None if implicit_auto_enabled => DevicePoolMode::Auto { explicit: false },
            None => DevicePoolMode::Off,
        });
    };

    match raw_mode.trim().to_ascii_lowercase().as_str() {
        "off" | "disabled" => {
            if bytes.is_some() {
                return Err(format!(
                    "{resolved_name}=off conflicts with {GPU_DEVICE_POOL_BYTES_ENV}; remove the byte override or select fixed mode"
                ));
            }
            Ok(DevicePoolMode::Off)
        }
        "fixed" => bytes.map(DevicePoolMode::Fixed).ok_or_else(|| {
            format!("{resolved_name}=fixed requires {GPU_DEVICE_POOL_BYTES_ENV}[_<device>]")
        }),
        "auto" => {
            if bytes.is_some() {
                return Err(format!(
                    "{resolved_name}=auto conflicts with {GPU_DEVICE_POOL_BYTES_ENV}; automatic sizing does not silently resize an explicit override"
                ));
            }
            Ok(DevicePoolMode::Auto { explicit: true })
        }
        _ => Err(format!(
            "invalid `{raw_mode}` for {resolved_name}; expected auto, fixed, or off"
        )),
    }
}

fn strict_device_vram_cap_bytes(device_id: usize) -> Result<Option<usize>, String> {
    if let Some((name, raw)) = configured_value(CUDA_DEVICE_MEMORY_LIMIT_ENV, device_id) {
        return parse_cuda_memory_limit(&raw).map(Some).ok_or_else(|| {
            format!(
                "invalid CUDA memory cap `{raw}` in {name}; automatic pool sizing requires a positive byte count or k/m/g suffix"
            )
        });
    }
    let Some(raw_mb) = optional_env_var(KAPSL_GPU_MEMORY_LIMIT_MB_ENV) else {
        return Ok(None);
    };
    raw_mb
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .and_then(|value| value.checked_mul(1024 * 1024))
        .map(Some)
        .ok_or_else(|| {
            format!(
                "invalid CUDA memory cap `{raw_mb}` in {KAPSL_GPU_MEMORY_LIMIT_MB_ENV}; expected a positive MiB count"
            )
        })
}

fn auto_safe_budget(
    device_id: usize,
    physical_bytes: usize,
    cuda: &Arc<CudaDevice>,
) -> Result<(usize, usize), String> {
    let declared_bytes = strict_device_vram_cap_bytes(device_id)?
        .map_or(physical_bytes, |cap| physical_bytes.min(cap));
    cuda.bind_to_thread()
        .map_err(|error| format!("failed to bind CUDA device {device_id}: {error}"))?;
    let (free_bytes, _total_bytes) = cuda_result::mem_get_info()
        .map_err(|error| format!("failed to query CUDA device {device_id} memory: {error}"))?;
    let (safe_budget, driver_reserve) =
        auto_safe_budget_from_inputs(physical_bytes, Some(declared_bytes), free_bytes);
    log::info!(
        "[device-memory] CUDA device {} automatic sizing inputs: physical={} declared={} free={} driver_reserve={} safe_budget={} bytes",
        device_id,
        physical_bytes,
        declared_bytes,
        free_bytes,
        driver_reserve,
        safe_budget,
    );
    Ok((safe_budget, driver_reserve))
}

fn auto_safe_budget_from_inputs(
    physical_bytes: usize,
    configured_cap_bytes: Option<usize>,
    free_bytes: usize,
) -> (usize, usize) {
    const DRIVER_RESERVE_FLOOR_BYTES: usize = 512 * 1024 * 1024;
    let declared_bytes = configured_cap_bytes.map_or(physical_bytes, |cap| physical_bytes.min(cap));
    let live_ceiling = declared_bytes.min(free_bytes);
    let driver_reserve = (declared_bytes / 10).max(DRIVER_RESERVE_FLOOR_BYTES);
    (live_ceiling.saturating_sub(driver_reserve), driver_reserve)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_disable_wins_over_every_public_pool_setting() {
        assert_eq!(
            resolve_device_pool_mode_values(true, Some(8), Some(("MODE", "auto")), true).unwrap(),
            DevicePoolMode::Off
        );
    }

    #[test]
    fn implicit_auto_is_application_profile_specific() {
        assert_eq!(
            resolve_device_pool_mode_values(false, None, None, true).unwrap(),
            DevicePoolMode::Auto { explicit: false }
        );
        assert_eq!(
            resolve_device_pool_mode_values(false, None, None, false).unwrap(),
            DevicePoolMode::Off
        );
    }

    #[test]
    fn exact_bytes_remain_a_fixed_override() {
        assert_eq!(
            resolve_device_pool_mode_values(false, Some(4096), None, true).unwrap(),
            DevicePoolMode::Fixed(4096)
        );
        assert_eq!(
            resolve_device_pool_mode_values(
                false,
                Some(4096),
                Some(("KAPSL_GPU_DEVICE_POOL_MODE", "fixed")),
                true,
            )
            .unwrap(),
            DevicePoolMode::Fixed(4096)
        );
    }

    #[test]
    fn mode_conflicts_are_rejected_instead_of_silently_resizing() {
        let error = resolve_device_pool_mode_values(
            false,
            Some(4096),
            Some(("KAPSL_GPU_DEVICE_POOL_MODE", "auto")),
            true,
        )
        .unwrap_err();
        assert!(error.contains("conflicts"), "{error}");

        let error = resolve_device_pool_mode_values(
            false,
            None,
            Some(("KAPSL_GPU_DEVICE_POOL_MODE", "fixed")),
            true,
        )
        .unwrap_err();
        assert!(error.contains("requires"), "{error}");

        let error = resolve_device_pool_mode_values(
            false,
            Some(4096),
            Some(("KAPSL_GPU_DEVICE_POOL_MODE_0", "off")),
            true,
        )
        .unwrap_err();
        assert!(error.contains("conflicts"), "{error}");
    }

    #[test]
    fn explicit_mode_is_case_insensitive_and_invalid_values_fail() {
        assert_eq!(
            resolve_device_pool_mode_values(
                false,
                None,
                Some(("KAPSL_GPU_DEVICE_POOL_MODE", " AuTo ")),
                false,
            )
            .unwrap(),
            DevicePoolMode::Auto { explicit: true }
        );
        assert_eq!(
            resolve_device_pool_mode_values(
                false,
                None,
                Some(("KAPSL_GPU_DEVICE_POOL_MODE", "off")),
                true,
            )
            .unwrap(),
            DevicePoolMode::Off
        );
        assert!(resolve_device_pool_mode_values(
            false,
            None,
            Some(("KAPSL_GPU_DEVICE_POOL_MODE", "elastic")),
            true,
        )
        .unwrap_err()
        .contains("expected auto, fixed, or off"));
    }

    #[test]
    fn unpooled_reserve_parser_accepts_zero_and_binary_suffixes() {
        assert_eq!(parse_nonnegative_cuda_bytes("0"), Some(0));
        assert_eq!(
            parse_nonnegative_cuda_bytes("2g"),
            Some(2 * 1024 * 1024 * 1024)
        );
        assert_eq!(parse_nonnegative_cuda_bytes("8Gi"), None);
    }

    #[test]
    fn bootstrap_deduplicates_stable_external_allocations() {
        let mut bootstrap = DeviceMemoryBootstrapPlan::default();
        bootstrap.mark_pool_consumer(2);
        bootstrap.add_external_allocation(2, "weights:model", 100);
        bootstrap.add_external_allocation(2, "weights:model", 80);
        bootstrap.add_external_allocation(2, "weights:other", 50);
        bootstrap.add_pooled_allocation(2, "onnx:model", 400);
        bootstrap.add_pooled_allocation(2, "onnx:model", 300);

        let demand = bootstrap.demand(2);
        assert!(demand.wants_pool);
        assert_eq!(demand.planned_external_bytes(), 150);
        assert_eq!(demand.minimum_pool_bytes(), 400);
    }

    #[test]
    fn automatic_safe_budget_clamps_to_cap_and_live_free_memory() {
        const GIB: usize = 1024 * 1024 * 1024;
        let (budget, reserve) = auto_safe_budget_from_inputs(24 * GIB, Some(16 * GIB), 12 * GIB);
        assert_eq!(reserve, 16 * GIB / 10);
        assert_eq!(budget, 12 * GIB - reserve);

        let (budget, reserve) = auto_safe_budget_from_inputs(4 * GIB, None, 256 * 1024 * 1024);
        assert_eq!(reserve, 512 * 1024 * 1024);
        assert_eq!(budget, 0);
    }

    #[test]
    fn pool_owner_metric_labels_are_stable_and_include_model_identity() {
        assert_eq!(
            pool_owner_metric_label(PoolOwner::onnx(
                11,
                2,
                PoolAllocationClass::TransientWorkspace
            )),
            "onnx:11:2:transient-workspace"
        );
        assert_eq!(
            pool_owner_metric_label(PoolOwner::gguf(42, 3, PoolAllocationClass::KvCache)),
            "gguf:42:3:kv-cache"
        );
        assert_eq!(
            pool_owner_metric_label(PoolOwner::unattributed(
                PoolBackend::Native,
                PoolAllocationClass::ExternallyOwned
            )),
            "native:unattributed:externally-owned"
        );
    }

    #[test]
    fn pool_snapshot_maps_every_live_and_owner_metric() {
        let metrics = pool_snapshot_metrics(kapsl_hal::gpu_arena::GpuDevicePoolSnapshot {
            capacity_bytes: 1_000,
            allocated_bytes: 400,
            live_allocation_count: 3,
            free_bytes: 600,
            free_range_count: 2,
            largest_free_range_bytes: 500,
            fragmentation_ratio: 1.0 / 6.0,
            owners: vec![kapsl_hal::gpu_arena::PoolOwnerSnapshot {
                owner: PoolOwner::gguf(9, 4, PoolAllocationClass::KvCache),
                usage_bytes: 400,
                guaranteed_bytes: 300,
                max_bytes: 800,
                admitted: true,
                allocatable_bytes: 400,
            }],
        });

        assert_eq!(metrics.allocated_bytes, 400);
        assert_eq!(metrics.live_allocations, 3);
        assert_eq!(metrics.free_bytes, 600);
        assert_eq!(metrics.free_ranges, 2);
        assert_eq!(metrics.largest_free_range_bytes, 500);
        assert_eq!(metrics.fragmentation_ratio, 1.0 / 6.0);
        assert_eq!(metrics.owners.len(), 1);
        assert_eq!(metrics.owners[0].owner, "gguf:9:4:kv-cache");
        assert_eq!(metrics.owners[0].usage_bytes, 400);
        assert_eq!(metrics.owners[0].guaranteed_bytes, 300);
        assert_eq!(metrics.owners[0].max_bytes, 800);
        assert!(metrics.owners[0].admitted);
        assert_eq!(metrics.owners[0].allocatable_bytes, 400);
    }
}
