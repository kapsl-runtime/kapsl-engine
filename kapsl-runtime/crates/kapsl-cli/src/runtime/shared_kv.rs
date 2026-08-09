use super::*;

/// Shared KV cache pool registry and cross-model token-budget coordinator.
///
/// Always held behind `Arc` (`type SharedKvState = Arc<SharedKvStateInner>`)
/// so cloning is a single atomic reference-count increment with no heap
/// allocation.  All `LLMBackend` instances on the same physical GPU share the
/// same `SharedBlockAllocator`, enforcing a single unified block budget.
pub(crate) struct SharedKvStateInner {
    /// Total VRAM bytes per device ID — used to size pools on first access.
    device_bytes: HashMap<usize, usize>,
    /// Per-device shared KV block allocators (lazily created).
    pools: Mutex<HashMap<usize, SharedBlockAllocator>>,
    /// Cross-model token-budget coordinator (hard admission gate in Phase 2).
    /// Uses parking_lot::Mutex so a panic in one engine's thread cannot poison
    /// the lock and propagate to all other engines.
    scheduler: Arc<parking_lot::Mutex<GlobalKvScheduler>>,
    /// Monotonically increasing counter for stable engine IDs.
    next_engine_id: AtomicU32,
    /// model_id → engine_ids assigned by attach_engine (supports multiple replicas).
    model_engine_ids: Mutex<HashMap<u32, Vec<u32>>>,
    /// Live per-engine KV block caps shared with LLMBackend instances.
    /// Updated by rebalance_kv_caps() on every engine attach / detach so that
    /// backends read the current fair-share cap without needing a restart.
    live_kv_caps: Mutex<HashMap<u32, Arc<AtomicUsize>>>,
    /// Last `GlobalKvScheduler::health_epoch` we rebalanced KV caps for. The
    /// periodic loop compares against the live epoch and rebalances when an
    /// engine's health changes, reclaiming a degraded/dead engine's block quota
    /// for healthy engines without waiting for a full detach.
    last_health_epoch: AtomicU64,
    /// Live per-device *soft* KV ceiling in bytes, foreign-aware. Refreshed by
    /// `refresh_ceilings` from the monitor loop: the declared budget minus VRAM
    /// held by co-tenant processes (e.g. a training job on the same card) minus
    /// a safety reserve. Drives `rebalance_kv_caps` so concurrency backs off
    /// under a noisy neighbor instead of OOMing it. Empty until the first
    /// refresh, and never populated when the co-tenancy path is disabled, so the
    /// soft budget falls back to `device_bytes` and default behavior is
    /// unchanged. Distinct from the *hard* arena in `get_or_create_pool`, which
    /// stays sized off `device_bytes` because handed-out blocks can't be
    /// reclaimed.
    live_ceiling: Mutex<HashMap<usize, Arc<AtomicUsize>>>,
    /// Stable runtime-owned byte pools, initialized before model loading.
    #[cfg(feature = "gpu-device-pool")]
    device_memory: std::sync::OnceLock<Arc<DeviceMemoryManager>>,
}

pub(crate) type SharedKvState = Arc<SharedKvStateInner>;

/// One device's ceiling arithmetic from a `refresh_ceilings` tick, surfaced so
/// the monitor loop can export it (Prometheus gauges + transition logs) —
/// otherwise the shrink-fast/recover-slow behavior is invisible from outside
/// the process. `target_bytes` is the instantaneous foreign-aware ceiling;
/// `smoothed_bytes` is the damped value that actually drives the KV caps, and
/// during grow-slow recovery it lags `target_bytes` — that gap is the
/// hysteresis, and it is exactly what a dashboard should show.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CeilingSample {
    pub(crate) device_id: usize,
    /// Co-tenant VRAM bytes observed on this device this tick.
    pub(crate) foreign_bytes: usize,
    /// Instantaneous ceiling: declared budget minus foreign minus reserve.
    pub(crate) target_bytes: usize,
    /// Damped live ceiling after shrink-fast/grow-slow smoothing.
    pub(crate) smoothed_bytes: usize,
    /// Live ceiling before this tick (equals `smoothed_bytes` when unchanged).
    pub(crate) previous_bytes: usize,
    /// Whether this device currently reads as squeezed by a co-tenant — the
    /// same per-device predicate `foreign_pressure_active` ORs together, so a
    /// dashboard can show exactly why the autoscaler is suppressed.
    pub(crate) squeezed: bool,
}

/// The per-device squeeze predicate behind `foreign_pressure_active`: the live
/// soft ceiling sits below 90% of the no-foreign target (the reserve alone
/// keeps the idle ceiling under the declared bytes, so the baseline is the
/// idle target, not the raw budget). Shared with `refresh_ceilings` so the
/// exported `CeilingSample::squeezed` can never drift from the autoscaler's
/// suppression signal.
fn ceiling_is_squeezed(device_id: usize, declared: usize, live_bytes: usize) -> bool {
    let idle_target = effective_ceiling_bytes(device_id, declared, 0);
    live_bytes < idle_target.saturating_mul(9) / 10
}

impl SharedKvStateInner {
    pub(crate) fn new(device_info: &DeviceInfo) -> SharedKvState {
        const KV_BYTES_PER_BLOCK: usize = 2 * 1024 * 1024;
        const KV_BLOCK_SIZE: usize = 16;
        let mut device_bytes = HashMap::new();
        let mut estimated_kv_tokens: usize = 0;
        for device in &device_info.devices {
            if device.backend.to_string().eq_ignore_ascii_case("cpu") {
                continue;
            }
            let total = (device.memory_mb as usize).saturating_mul(1024 * 1024);
            // Cooperative software-vGPU clamp: when a per-device VRAM cap is
            // configured (HAMi env or the kapsl alias), size the KV budget to
            // the slice rather than the whole card. A no-op when no cap is set
            // or the cap exceeds the card, so default behavior is unchanged and
            // a MIG slice (which already reports its true size) is never inflated.
            let total = device_vram_cap_bytes(device.id).map_or(total, |cap| total.min(cap));
            device_bytes.insert(device.id, total);
            let kv_blocks = (total / 2) / KV_BYTES_PER_BLOCK;
            estimated_kv_tokens = estimated_kv_tokens.saturating_add(kv_blocks * KV_BLOCK_SIZE);
        }
        Arc::new(Self {
            device_bytes,
            pools: Mutex::new(HashMap::new()),
            scheduler: Arc::new(parking_lot::Mutex::new(GlobalKvScheduler::new(
                estimated_kv_tokens.max(16_384),
            ))),
            next_engine_id: AtomicU32::new(1),
            model_engine_ids: Mutex::new(HashMap::new()),
            live_kv_caps: Mutex::new(HashMap::new()),
            last_health_epoch: AtomicU64::new(0),
            live_ceiling: Mutex::new(HashMap::new()),
            #[cfg(feature = "gpu-device-pool")]
            device_memory: std::sync::OnceLock::new(),
        })
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn new_runtime(device_info: &DeviceInfo) -> Result<SharedKvState, String> {
        let state = Self::new(device_info);
        if let Some(manager) = DeviceMemoryManager::from_env(device_info)? {
            state
                .device_memory
                .set(manager)
                .map_err(|_| "device memory manager initialized twice".to_string())?;
        }
        Ok(state)
    }

    #[cfg(not(feature = "gpu-device-pool"))]
    pub(crate) fn new_runtime(device_info: &DeviceInfo) -> Result<SharedKvState, String> {
        Ok(Self::new(device_info))
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
            .get()
            .and_then(|manager| manager.pool(device_id))
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn uses_env_allocators(&self, device_id: usize) -> bool {
        self.device_memory
            .get()
            .is_some_and(|manager| manager.has_pool(device_id))
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn begin_device_memory_admission(
        &self,
        device_id: usize,
        model_id: u32,
        kind: EngineKind,
    ) -> Result<Option<DeviceMemoryAdmission>, String> {
        self.device_memory
            .get()
            .map(|manager| manager.begin_admission(device_id, model_id, kind))
            .transpose()
            .map(Option::flatten)
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn release_device_memory(&self, model_id: u32) {
        if let Some(manager) = self.device_memory.get() {
            manager.release_model(model_id);
        }
    }

    /// Return (or lazily create) the shared block allocator for `device_id`.
    pub(crate) fn get_or_create_pool(&self, device_id: usize) -> SharedBlockAllocator {
        let mut pools = self.pools.lock();
        if let Some(existing) = pools.get(&device_id) {
            return existing.clone();
        }
        const KV_BYTES_PER_BLOCK: usize = 2 * 1024 * 1024;
        const KV_BLOCK_SIZE: usize = 16;
        let total_bytes = self.device_bytes.get(&device_id).copied().unwrap_or(0);
        let total_blocks = ((total_bytes / 2) / KV_BYTES_PER_BLOCK).max(256);
        let allocator = new_shared_allocator(total_blocks, KV_BLOCK_SIZE, device_id);
        pools.insert(device_id, allocator.clone());
        allocator
    }

    /// Attach a new engine to the shared pool and register it with the global
    /// scheduler.  Returns:
    /// - the shared block allocator
    /// - the recommended per-engine `total_blocks` cap
    /// - an `Arc` to the global scheduler (for `LLMBackend::with_global_scheduler`)
    /// - the stable engine ID assigned to this engine
    pub(crate) fn attach_engine(
        &self,
        device_id: usize,
        model_id: u32,
        weight: u32,
    ) -> (
        SharedBlockAllocator,
        usize,
        Arc<parking_lot::Mutex<GlobalKvScheduler>>,
        u32,
        Arc<AtomicUsize>,
    ) {
        let allocator = self.get_or_create_pool(device_id);
        let engine_id = self.next_engine_id.fetch_add(1, Ordering::Relaxed);
        self.scheduler.lock().register(KvEngineHandle {
            engine_id,
            share_weight: weight.max(1),
            guaranteed_min_tokens: 0,
            max_tokens: None,
        });
        self.model_engine_ids
            .lock()
            .entry(model_id)
            .or_default()
            .push(engine_id);
        const KV_BYTES_PER_BLOCK: usize = 2 * 1024 * 1024;
        const MIN_BLOCKS_PER_ENGINE: usize = 256;
        let total_bytes = self.device_bytes.get(&device_id).copied().unwrap_or(0);
        let total_blocks = ((total_bytes / 2) / KV_BYTES_PER_BLOCK).max(MIN_BLOCKS_PER_ENGINE);
        let engine_count = (engine_id + 1) as usize;
        let initial_cap = (total_blocks / engine_count).max(MIN_BLOCKS_PER_ENGINE);
        // Register live-cap atomic and trigger rebalancing across all engines.
        let live_cap = Arc::new(AtomicUsize::new(initial_cap));
        self.live_kv_caps.lock().insert(engine_id, live_cap.clone());
        self.rebalance_kv_caps();
        (
            allocator,
            initial_cap,
            self.scheduler.clone(),
            engine_id,
            live_cap,
        )
    }

    /// Detach a single engine (e.g. after its `run_loop` task dies). Removes it
    /// from the scheduler registry, drops its live KV cap, purges it from the
    /// model→engine map, and rebalances remaining engines. Idempotent: a second
    /// call for an already-detached engine is a no-op.
    pub(crate) fn detach_engine(&self, engine_id: u32) {
        self.scheduler.lock().deregister(engine_id);
        self.live_kv_caps.lock().remove(&engine_id);
        {
            let mut map = self.model_engine_ids.lock();
            map.retain(|_, ids| {
                ids.retain(|&id| id != engine_id);
                !ids.is_empty()
            });
        }
        self.rebalance_kv_caps();
    }

    /// Deregister all engines for a model (call on full model stop/remove).
    pub(crate) fn detach_engine_for_model(&self, model_id: u32) {
        if let Some(ids) = self.model_engine_ids.lock().remove(&model_id) {
            let mut sched = self.scheduler.lock();
            let mut caps = self.live_kv_caps.lock();
            for id in ids {
                sched.deregister(id);
                caps.remove(&id);
            }
        }
        self.rebalance_kv_caps();
    }

    /// Recompute per-engine KV block caps from the global scheduler's budget
    /// allocation and push the new values into each engine's `Arc<AtomicUsize>`.
    ///
    /// Called on every engine attach/detach so live caps track the current set
    /// of loaded models without requiring engine restarts.
    /// Rebalance KV block caps if any engine's health changed since the last
    /// rebalance. Cheap to call frequently: it only takes the scheduler lock to
    /// read an integer and does real work (recomputing caps from the now
    /// health-aware budgets) only on an actual health transition.
    pub(crate) fn maybe_rebalance_for_health(&self) {
        let epoch = self.scheduler.lock().health_epoch();
        if self.last_health_epoch.swap(epoch, Ordering::Relaxed) != epoch {
            self.rebalance_kv_caps();
        }
    }

    /// Recompute the live, foreign-aware soft ceiling for every device from a
    /// fresh co-tenancy sample and, if any ceiling actually moved, rebalance the
    /// per-engine KV caps so admission backs off (or recovers). Called from the
    /// monitor loop next to `maybe_rebalance_for_health`.
    ///
    /// The per-device value is smoothed asymmetrically (shrink fast, grow slow)
    /// so a trainer's sawtooth footprint doesn't make the KV batch width flap,
    /// and the rebalance is gated on an actual change so an idle GPU costs only a
    /// map read. A ceiling that floors to zero is stored as-is; the
    /// `MIN_BLOCKS_PER_ENGINE` clamp in `rebalance_kv_caps` keeps each engine at
    /// a minimal batch rather than fully stalling.
    ///
    /// Returns one `CeilingSample` per device so the caller can export the
    /// arithmetic (metrics + logs) without re-deriving it.
    pub(crate) fn refresh_ceilings(&self, foreign: &HashMap<usize, usize>) -> Vec<CeilingSample> {
        let mut changed = false;
        let mut samples = Vec::with_capacity(self.device_bytes.len());
        {
            let mut live = self.live_ceiling.lock();
            for (&device_id, &declared) in &self.device_bytes {
                let foreign_bytes = foreign.get(&device_id).copied().unwrap_or(0);
                let target = effective_ceiling_bytes(device_id, declared, foreign_bytes);
                let atom = live
                    .entry(device_id)
                    .or_insert_with(|| Arc::new(AtomicUsize::new(declared)));
                let previous = atom.load(Ordering::Relaxed);
                let smoothed = smooth_ceiling_bytes(previous, target);
                if smoothed != previous {
                    atom.store(smoothed, Ordering::Relaxed);
                    changed = true;
                }
                samples.push(CeilingSample {
                    device_id,
                    foreign_bytes,
                    target_bytes: target,
                    smoothed_bytes: smoothed,
                    previous_bytes: previous,
                    squeezed: ceiling_is_squeezed(device_id, declared, smoothed),
                });
            }
        }
        if changed {
            self.rebalance_kv_caps();
        }
        samples
    }

    /// True while a co-tenant process is squeezing any device's KV ceiling:
    /// the live soft ceiling sits below 90% of what it would be with no foreign
    /// footprint (the reserve alone keeps the idle ceiling under the declared
    /// bytes, so the comparison baseline is the no-foreign target, not the raw
    /// budget). The autoscaler uses this to tell queue depth caused by a noisy
    /// neighbor from real load growth — adding a replica on a starved GPU only
    /// thrashes. The grow-slow smoothing keeps this true for a few ticks after
    /// the neighbor exits, which doubles as scale-up hysteresis. Always false
    /// when the co-tenancy guard is off (`live_ceiling` never populated).
    pub(crate) fn foreign_pressure_active(&self) -> bool {
        let live = self.live_ceiling.lock();
        live.iter().any(|(device_id, atom)| {
            let declared = self.device_bytes.get(device_id).copied().unwrap_or(0);
            ceiling_is_squeezed(*device_id, declared, atom.load(Ordering::Relaxed))
        })
    }

    /// The soft KV budget in bytes for a device: the live foreign-aware ceiling
    /// when one has been refreshed, otherwise the static declared `device_bytes`.
    /// A floored (zero) live ceiling is honored rather than falling back, so a
    /// GPU fully claimed by a trainer shrinks engines to their block minimum
    /// instead of resetting to the full budget.
    fn device_soft_ceiling_bytes(&self, device_id: usize) -> usize {
        self.live_ceiling
            .lock()
            .get(&device_id)
            .map(|atom| atom.load(Ordering::Relaxed))
            .or_else(|| self.device_bytes.get(&device_id).copied())
            .unwrap_or(0)
    }

    pub(crate) fn rebalance_kv_caps(&self) {
        const KV_BYTES_PER_BLOCK: usize = 2 * 1024 * 1024;
        const MIN_BLOCKS_PER_ENGINE: usize = 256;

        let budgets = self.scheduler.lock().allocate_budgets();
        if budgets.is_empty() {
            return;
        }

        let total_tokens: usize = budgets.iter().map(|b| b.max_tokens).sum::<usize>().max(1);
        let caps = self.live_kv_caps.lock();

        for budget in &budgets {
            let Some(cap_atom) = caps.get(&budget.engine_id) else {
                continue;
            };
            // Translate token fraction → block fraction using the device's
            // total block pool for this engine's device.
            let device_id = self
                .model_engine_ids
                .lock()
                .values()
                .find(|ids| ids.contains(&budget.engine_id))
                .and_then(|_| {
                    // We don't track engine_id→device_id directly; use device_bytes
                    // to get the first device that has memory configured.
                    self.device_bytes.keys().next().copied()
                })
                .unwrap_or(0);
            // Soft budget: the live foreign-aware ceiling when refreshed, else the
            // static declared bytes. The hard arena in get_or_create_pool keeps
            // reading device_bytes so already-handed-out blocks are never revoked.
            let total_bytes = self.device_soft_ceiling_bytes(device_id);
            let total_blocks = ((total_bytes / 2) / KV_BYTES_PER_BLOCK).max(MIN_BLOCKS_PER_ENGINE);
            let new_cap =
                (total_blocks * budget.max_tokens / total_tokens).max(MIN_BLOCKS_PER_ENGINE);
            cap_atom.store(new_cap, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod vram_clamp_tests {
    use super::SharedKvStateInner;
    use crate::app::constants::CUDA_DEVICE_MEMORY_LIMIT_ENV;
    use kapsl_hal::device::{Device, DeviceBackend, DeviceInfo};

    const GIB: usize = 1024 * 1024 * 1024;

    fn cuda_device(id: usize, memory_mb: u64) -> Device {
        Device {
            id,
            name: format!("test-gpu-{id}"),
            backend: DeviceBackend::Cuda,
            memory_mb,
            compute_units: 1,
            pci_bus_id: None,
            partition_id: None,
            driver_version: None,
            cuda_version: None,
            compute_capability: None,
            utilization_gpu_pct: None,
            temperature_c: None,
            supports_fp16: true,
            supports_int8: true,
        }
    }

    fn device_info(devices: Vec<Device>) -> DeviceInfo {
        DeviceInfo {
            cpu_cores: 1,
            total_memory: 0,
            os_type: "test".to_string(),
            os_release: "test".to_string(),
            has_cuda: true,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices,
        }
    }

    #[test]
    fn device_bytes_unchanged_without_a_cap() {
        // device id 4242 has no per-device cap env, and the bare
        // CUDA_DEVICE_MEMORY_LIMIT / KAPSL_GPU_MEMORY_LIMIT_MB globals are never
        // set by any test, so the cooperative clamp is a no-op here.
        let info = device_info(vec![cuda_device(4242, 24576)]);
        let state = SharedKvStateInner::new(&info);
        assert_eq!(state.device_bytes.get(&4242).copied(), Some(24 * GIB));
    }

    #[test]
    fn refresh_ceilings_shrinks_fast_then_recovers_slowly() {
        use std::collections::HashMap;
        // 40 GiB card, no cap env → declared == physical.
        let device_id = 4245;
        let info = device_info(vec![cuda_device(device_id, 40 * 1024)]);
        let state = SharedKvStateInner::new(&info);

        // Before any refresh the soft ceiling is the full declared budget.
        assert_eq!(state.device_soft_ceiling_bytes(device_id), 40 * GIB);

        // A 6 GiB trainer appears: 40 - 6 - 4 (10% reserve) = 30 GiB, and a
        // shrink is applied immediately (no damping on the safety side).
        let mut foreign = HashMap::new();
        foreign.insert(device_id, 6 * GIB);
        state.refresh_ceilings(&foreign);
        assert_eq!(state.device_soft_ceiling_bytes(device_id), 30 * GIB);

        // Trainer exits: target recovers to 40 - 4 = 36 GiB but growth is damped
        // to a quarter of the gap → 30 + (36 - 30)/4 = 31.5 GiB.
        state.refresh_ceilings(&HashMap::new());
        assert_eq!(
            state.device_soft_ceiling_bytes(device_id),
            30 * GIB + (6 * GIB) / 4
        );
    }

    #[test]
    fn refresh_ceilings_samples_expose_the_smoothing_gap_and_squeeze() {
        use std::collections::HashMap;
        // Same 40 GiB card as the shrink/recover test; the returned samples must
        // mirror the stored arithmetic so metrics can't drift from behavior.
        let device_id = 4247;
        let info = device_info(vec![cuda_device(device_id, 40 * 1024)]);
        let state = SharedKvStateInner::new(&info);

        // A 12 GiB trainer: target = 40 - 12 - 4 (reserve) = 24 GiB, shrink is
        // immediate so smoothed == target, and the device reads as squeezed —
        // the same predicate the autoscaler suppression uses.
        let mut foreign = HashMap::new();
        foreign.insert(device_id, 12 * GIB);
        let samples = state.refresh_ceilings(&foreign);
        assert_eq!(samples.len(), 1);
        let sample = samples[0];
        assert_eq!(sample.device_id, device_id);
        assert_eq!(sample.foreign_bytes, 12 * GIB);
        assert_eq!(sample.target_bytes, 24 * GIB);
        assert_eq!(sample.smoothed_bytes, 24 * GIB);
        assert_eq!(sample.previous_bytes, 40 * GIB);
        assert!(sample.squeezed);
        assert_eq!(sample.squeezed, state.foreign_pressure_active());

        // Trainer exits: target snaps back to the 36 GiB idle ceiling but the
        // smoothed value only closes a quarter of the gap — the sample exposes
        // exactly that lag (24 + 12/4 = 27 GiB) while still reading squeezed.
        let samples = state.refresh_ceilings(&HashMap::new());
        let sample = samples[0];
        assert_eq!(sample.foreign_bytes, 0);
        assert_eq!(sample.target_bytes, 36 * GIB);
        assert_eq!(sample.previous_bytes, 24 * GIB);
        assert_eq!(sample.smoothed_bytes, 24 * GIB + 3 * GIB);
        assert!(sample.squeezed);

        // Once recovery converges the squeeze clears in the sample and in the
        // autoscaler signal together.
        let mut last = sample;
        for _ in 0..16 {
            last = state.refresh_ceilings(&HashMap::new())[0];
        }
        assert!(!last.squeezed);
        assert!(!state.foreign_pressure_active());
    }

    #[test]
    fn foreign_pressure_tracks_the_squeeze_and_releases_with_hysteresis() {
        use std::collections::HashMap;
        let device_id = 4246;
        let info = device_info(vec![cuda_device(device_id, 40 * 1024)]);
        let state = SharedKvStateInner::new(&info);

        // Guard off / never refreshed → never "under pressure".
        assert!(!state.foreign_pressure_active());

        // Idle refresh (no co-tenant): the reserve alone must not read as
        // pressure, or the autoscaler would be permanently suppressed.
        state.refresh_ceilings(&HashMap::new());
        assert!(!state.foreign_pressure_active());

        // A 12 GiB trainer squeezes the ceiling well below 90% of idle.
        let mut foreign = HashMap::new();
        foreign.insert(device_id, 12 * GIB);
        state.refresh_ceilings(&foreign);
        assert!(state.foreign_pressure_active());

        // Trainer exits: grow-slow recovery keeps pressure asserted for a few
        // ticks (scale-up hysteresis), then releases.
        state.refresh_ceilings(&HashMap::new());
        assert!(state.foreign_pressure_active());
        for _ in 0..16 {
            state.refresh_ceilings(&HashMap::new());
        }
        assert!(!state.foreign_pressure_active());
    }

    #[test]
    fn device_bytes_clamped_to_the_configured_cap() {
        // Unique device index so the per-device env never collides with other
        // tests running in parallel.
        let device_id = 4243;
        let var = format!("{CUDA_DEVICE_MEMORY_LIMIT_ENV}_{device_id}");
        std::env::set_var(&var, "8g");
        let info = device_info(vec![cuda_device(device_id, 24576)]);
        let state = SharedKvStateInner::new(&info);
        std::env::remove_var(&var);
        // The KV budget sizes to the 8 GiB slice, not the 24 GiB physical card,
        // so the whole downstream KV chain (pools, per-engine caps, rebalancing)
        // self-limits.
        assert_eq!(state.device_bytes.get(&device_id).copied(), Some(8 * GIB));
    }
}
