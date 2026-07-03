use super::*;

#[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
#[derive(Clone)]
pub(crate) struct GpuPoolMember {
    model_id: u32,
    weight: u32,
    cap: Arc<AtomicUsize>,
}

#[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
#[derive(Clone)]
pub(crate) struct DeviceGpuPoolState {
    handle: GpuPoolHandle,
    members: Vec<GpuPoolMember>,
}

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
    /// Per-device GPU pool registry for gguf-native/native backends.
    /// Stores compatible pools and their members so quota caps can be
    /// rebalanced by model priority.
    #[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
    gpu_pools: Mutex<HashMap<usize, Vec<DeviceGpuPoolState>>>,
    /// GPU-wide session-level block admission and cross-device migration.
    #[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
    cross_device_sched: Mutex<CrossDevicePoolScheduler>,
}

pub(crate) type SharedKvState = Arc<SharedKvStateInner>;

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
            #[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
            gpu_pools: Mutex::new(HashMap::new()),
            #[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
            cross_device_sched: Mutex::new(CrossDevicePoolScheduler::new(0.85, 2048)),
        })
    }

    /// Return the existing pool handle for `device_id` so a new backend can
    /// attach to it before calling load().
    #[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
    pub(crate) fn get_gpu_pool(&self, device_id: usize) -> Option<GpuPoolHandle> {
        self.gpu_pools
            .lock()
            .get(&device_id)
            .and_then(|pools| pools.first())
            .map(|state| state.handle.for_engine(state.handle.cap()))
    }

    /// Register (or re-register) a pool handle after a backend finishes load().
    /// Rebalances per-engine caps by model priority weight. If load created a
    /// private pool due to incompatible geometry, it becomes a separate pool
    /// group for future compatible models.
    #[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
    pub(crate) fn attach_gpu_pool(
        &self,
        device_id: usize,
        model_id: u32,
        weight: u32,
        handle: GpuPoolHandle,
    ) {
        const MIN_BLOCKS_PER_ENGINE: usize = 64;
        let mut pools = self.gpu_pools.lock();
        let device_pools = pools.entry(device_id).or_default();
        let pool_index = device_pools
            .iter()
            .position(|state| Arc::ptr_eq(&state.handle.pool, &handle.pool));

        let state = if let Some(index) = pool_index {
            &mut device_pools[index]
        } else {
            device_pools.push(DeviceGpuPoolState {
                handle: GpuPoolHandle::with_cap(handle.pool.clone(), handle.pool.total_blocks()),
                members: Vec::new(),
            });
            device_pools.last_mut().expect("inserted pool state")
        };

        state.members.push(GpuPoolMember {
            model_id,
            weight: weight.max(1),
            cap: handle.blocks_per_engine.clone(),
        });

        let total_weight = state
            .members
            .iter()
            .map(|member| member.weight as usize)
            .sum::<usize>()
            .max(1);
        let total_blocks = state.handle.pool.total_blocks();
        for member in &state.members {
            let weighted_cap = total_blocks.saturating_mul(member.weight as usize) / total_weight;
            member
                .cap
                .store(weighted_cap.max(MIN_BLOCKS_PER_ENGINE), Ordering::Relaxed);
        }
        log::info!(
            "[gpu-pool] device {}: {} model(s) sharing {} blocks with weighted caps: {}",
            device_id,
            state.members.len(),
            total_blocks,
            state
                .members
                .iter()
                .map(|member| format!(
                    "model={} weight={} cap={}",
                    member.model_id,
                    member.weight,
                    member.cap.load(Ordering::Relaxed)
                ))
                .collect::<Vec<_>>()
                .join(", "),
        );

        // Also register with the cross-device scheduler so session-level
        // admission and migration can see this pool.
        self.cross_device_sched
            .lock()
            .register_pool(device_id, handle.pool.clone());
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
    pub(crate) fn refresh_ceilings(&self, foreign: &HashMap<usize, usize>) {
        let mut changed = false;
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
            }
        }
        if changed {
            self.rebalance_kv_caps();
        }
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
            let idle_target = effective_ceiling_bytes(*device_id, declared, 0);
            atom.load(Ordering::Relaxed) < idle_target.saturating_mul(9) / 10
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

    /// Remove a model from the GPU block pool registry and rebalance remaining
    /// members' quota caps.  No-op if the model is not registered.
    #[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
    pub(crate) fn detach_gpu_pool(&self, model_id: u32) {
        const MIN_BLOCKS_PER_ENGINE: usize = 64;
        let emptied_devices: Vec<usize> = {
            let mut pools = self.gpu_pools.lock();
            for device_pools in pools.values_mut() {
                for state in device_pools.iter_mut() {
                    let before = state.members.len();
                    state.members.retain(|m| m.model_id != model_id);
                    if state.members.len() == before {
                        continue;
                    }
                    if state.members.is_empty() {
                        log::info!("[gpu-pool] model {} detached; pool now empty", model_id);
                        continue;
                    }
                    let total_weight = state
                        .members
                        .iter()
                        .map(|m| m.weight as usize)
                        .sum::<usize>()
                        .max(1);
                    let total_blocks = state.handle.pool.total_blocks();
                    for member in &state.members {
                        let weighted_cap =
                            total_blocks.saturating_mul(member.weight as usize) / total_weight;
                        member
                            .cap
                            .store(weighted_cap.max(MIN_BLOCKS_PER_ENGINE), Ordering::Relaxed);
                    }
                    log::info!(
                        "[gpu-pool] model {} detached; {} member(s) remaining, caps rebalanced: {}",
                        model_id,
                        state.members.len(),
                        state
                            .members
                            .iter()
                            .map(|m| format!(
                                "model={} cap={}",
                                m.model_id,
                                m.cap.load(Ordering::Relaxed)
                            ))
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                }
                device_pools.retain(|state| !state.members.is_empty());
            }
            // Collect device IDs whose last pool was just removed before we release
            // the lock, so we can clean up the cross-device scheduler outside it.
            pools
                .iter()
                .filter(|(_, v)| v.is_empty())
                .map(|(&k, _)| k)
                .collect()
        };

        // Unregister fully-empty devices from the cross-device scheduler now
        // that gpu_pools lock is released.
        if !emptied_devices.is_empty() {
            let mut sched = self.cross_device_sched.lock();
            for dev_id in emptied_devices {
                sched.unregister_device(dev_id);
            }
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
