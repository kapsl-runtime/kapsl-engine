use super::*;

struct KvEngineRecord {
    model_id: u32,
    replica_id: u32,
    device_id: usize,
    live_cap: Arc<AtomicUsize>,
}

/// Logical KV cache registry and cross-model token-budget coordinator.
///
/// Always held behind `Arc` (`type KvCoordinator = Arc<KvCoordinatorInner>`)
/// so cloning is a single atomic reference-count increment with no heap
/// allocation.  All `LLMBackend` instances on the same physical GPU share the
/// same `SharedBlockAllocator`, enforcing a single unified block budget.
pub(crate) struct KvCoordinatorInner {
    /// The sole source of physical-domain capacity and pressure state.
    memory: Arc<MemoryAuthority>,
    /// Per-device shared KV block allocators (lazily created).
    logical_block_allocators: Mutex<HashMap<usize, SharedBlockAllocator>>,
    /// Cross-model token-budget coordinator (hard admission gate in Phase 2).
    /// Uses parking_lot::Mutex so a panic in one engine's thread cannot poison
    /// the lock and propagate to all other engines.
    scheduler: Arc<parking_lot::Mutex<GlobalKvScheduler>>,
    /// Monotonically increasing counter for stable engine IDs.
    next_engine_id: AtomicU32,
    /// Explicit engine identity and placement. Keeping device identity here is
    /// what lets cap rebalancing operate independently on each device.
    engine_records: Mutex<HashMap<u32, KvEngineRecord>>,
    /// Last `GlobalKvScheduler::health_epoch` we rebalanced KV caps for. The
    /// periodic loop compares against the live epoch and rebalances when an
    /// engine's health changes, reclaiming a degraded/dead engine's block quota
    /// for healthy engines without waiting for a full detach.
    last_health_epoch: AtomicU64,
}

pub(crate) type KvCoordinator = Arc<KvCoordinatorInner>;

impl KvCoordinatorInner {
    pub(crate) fn new(memory: Arc<MemoryAuthority>) -> KvCoordinator {
        const KV_BYTES_PER_BLOCK: usize = 2 * 1024 * 1024;
        const KV_BLOCK_SIZE: usize = 16;
        let mut estimated_kv_tokens: usize = 0;
        for (_, kv_bytes) in memory.kv_device_budgets() {
            let kv_blocks = kv_bytes / KV_BYTES_PER_BLOCK;
            estimated_kv_tokens = estimated_kv_tokens.saturating_add(kv_blocks * KV_BLOCK_SIZE);
        }
        Arc::new(Self {
            memory,
            logical_block_allocators: Mutex::new(HashMap::new()),
            scheduler: Arc::new(parking_lot::Mutex::new(GlobalKvScheduler::new(
                estimated_kv_tokens.max(16_384),
            ))),
            next_engine_id: AtomicU32::new(1),
            engine_records: Mutex::new(HashMap::new()),
            last_health_epoch: AtomicU64::new(0),
        })
    }

    /// Return (or lazily create) the shared block allocator for `device_id`.
    pub(crate) fn get_or_create_pool(&self, device_id: usize) -> SharedBlockAllocator {
        let mut allocators = self.logical_block_allocators.lock();
        if let Some(existing) = allocators.get(&device_id) {
            return existing.clone();
        }
        const KV_BYTES_PER_BLOCK: usize = 2 * 1024 * 1024;
        const KV_BLOCK_SIZE: usize = 16;
        let total_bytes = self.memory.kv_budget_bytes(device_id);
        let total_blocks = (total_bytes / KV_BYTES_PER_BLOCK).max(1);
        let allocator = new_shared_allocator(total_blocks, KV_BLOCK_SIZE, device_id);
        allocators.insert(device_id, allocator.clone());
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
        replica_id: u32,
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
        const KV_BYTES_PER_BLOCK: usize = 2 * 1024 * 1024;
        const MIN_BLOCKS_PER_ENGINE: usize = 1;
        let total_bytes = self.memory.kv_budget_bytes(device_id);
        let total_blocks = (total_bytes / KV_BYTES_PER_BLOCK).max(MIN_BLOCKS_PER_ENGINE);
        let (initial_cap, live_cap) = {
            let mut records = self.engine_records.lock();
            let engine_count = records
                .values()
                .filter(|record| record.device_id == device_id)
                .count()
                + 1;
            let initial_cap = (total_blocks / engine_count).max(MIN_BLOCKS_PER_ENGINE);
            let live_cap = Arc::new(AtomicUsize::new(initial_cap));
            records.insert(
                engine_id,
                KvEngineRecord {
                    model_id,
                    replica_id,
                    device_id,
                    live_cap: live_cap.clone(),
                },
            );
            (initial_cap, live_cap)
        };
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
        self.engine_records.lock().remove(&engine_id);
        self.rebalance_kv_caps();
    }

    /// Deregister all engines for a model (call on full model stop/remove).
    pub(crate) fn detach_engine_for_model(&self, model_id: u32) {
        let engine_ids = {
            let mut records = self.engine_records.lock();
            let ids: Vec<_> = records
                .iter()
                .filter_map(|(&engine_id, record)| {
                    (record.model_id == model_id).then_some(engine_id)
                })
                .collect();
            for engine_id in &ids {
                records.remove(engine_id);
            }
            ids
        };
        if !engine_ids.is_empty() {
            let mut sched = self.scheduler.lock();
            for engine_id in engine_ids {
                sched.deregister(engine_id);
            }
        }
        self.rebalance_kv_caps();
    }

    /// Conservative request-time KV reservation expressed in authority bytes.
    /// Backends with model-specific geometry publish their own KV row; this is
    /// the compatibility floor for older built-ins that only expose logical
    /// block admission.
    pub(crate) fn request_memory_plan(
        &self,
        owner: MemoryOwner,
        request: &InferenceRequest,
        fallback_claims: &[MemoryClaim],
        reserve_host_fallback: bool,
    ) -> MemoryPlan {
        const KV_BYTES_PER_BLOCK: usize = 2 * 1024 * 1024;
        const KV_BLOCK_SIZE: usize = 16;
        let max_new_tokens = request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.max_new_tokens)
            .map(|tokens| tokens as usize)
            .unwrap_or(512);
        // Integer-token packets contain one token per element; UTF-8/byte
        // prompts can tokenize to at most one token per input byte. Auxiliary
        // tensors (attention masks, media, etc.) do not create KV positions.
        let element_bytes = request.input.dtype.size_bytes().max(1);
        let input_tokens = request
            .input
            .data
            .len()
            .saturating_add(element_bytes - 1)
            .saturating_div(element_bytes);
        let tokens = input_tokens.saturating_add(max_new_tokens).max(1);
        let requested_blocks = tokens
            .saturating_add(KV_BLOCK_SIZE - 1)
            .saturating_div(KV_BLOCK_SIZE);

        let record = self.engine_records.lock().values().find_map(|record| {
            (record.model_id == owner.model_id && record.replica_id == owner.replica_id)
                .then_some((record.device_id, record.live_cap.load(Ordering::Relaxed)))
        });
        let blocks = record
            .map(|(_, live_cap)| requested_blocks.min(live_cap))
            .unwrap_or(requested_blocks);
        let mut claims = fallback_claims.to_vec();
        if claims.is_empty() {
            let domain = record
                .map(|(device_id, _)| self.memory.domain_for_device(device_id))
                .unwrap_or(MemoryDomain::Host);
            claims.push(MemoryClaim::runtime(
                domain,
                owner,
                MemoryAllocationClass::KvCache,
                0,
            ));
        }
        if reserve_host_fallback
            && claims
                .iter()
                .any(|claim| matches!(claim.domain, MemoryDomain::Cuda { .. }))
            && !claims.iter().any(|claim| {
                matches!(
                    claim.domain,
                    MemoryDomain::Host
                        | MemoryDomain::HostPinned { .. }
                        | MemoryDomain::HostMapped { .. }
                )
            })
        {
            // Older SDK backends expose only the CUDA KV template. Reserve the
            // host destination in the same transaction so provider fallback
            // can never create an unleased second copy.
            claims.push(MemoryClaim::runtime(
                MemoryDomain::Host,
                owner,
                MemoryAllocationClass::KvCache,
                0,
            ));
        }

        let mut plan = MemoryPlan::new();
        for mut claim in claims {
            claim.bytes = blocks.saturating_mul(KV_BYTES_PER_BLOCK);
            plan.push(claim);
        }
        plan
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
    #[cfg(test)]
    pub(crate) fn refresh_ceilings(&self, foreign: &HashMap<usize, usize>) -> Vec<CeilingSample> {
        let samples = self.memory.reconcile_external_device_memory(foreign);
        if samples
            .iter()
            .any(|sample| sample.smoothed_bytes != sample.previous_bytes)
        {
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
    /// when the co-tenancy guard has not observed a squeeze.
    #[cfg(test)]
    pub(crate) fn foreign_pressure_active(&self) -> bool {
        self.memory.snapshot().foreign_pressure_active
    }

    /// The soft KV budget in bytes for a device: the live foreign-aware ceiling
    /// when one has been refreshed, otherwise the authority's static budget.
    /// A floored (zero) live ceiling is honored rather than falling back, so a
    /// GPU fully claimed by a trainer shrinks engines to their block minimum
    /// instead of resetting to the full budget.
    fn device_soft_ceiling_bytes(&self, device_id: usize) -> usize {
        self.memory.kv_budget_bytes(device_id)
    }

    pub(crate) fn rebalance_kv_caps(&self) {
        const KV_BYTES_PER_BLOCK: usize = 2 * 1024 * 1024;
        const MIN_BLOCKS_PER_ENGINE: usize = 1;

        let budgets = self.scheduler.lock().allocate_budgets();
        if budgets.is_empty() {
            return;
        }

        let records = self.engine_records.lock();
        let mut device_token_totals = HashMap::<usize, usize>::new();
        for budget in &budgets {
            if let Some(record) = records.get(&budget.engine_id) {
                *device_token_totals.entry(record.device_id).or_default() += budget.max_tokens;
            }
        }

        for budget in &budgets {
            let Some(record) = records.get(&budget.engine_id) else {
                continue;
            };
            let device_id = record.device_id;
            let device_tokens = device_token_totals
                .get(&device_id)
                .copied()
                .unwrap_or(1)
                .max(1);
            // Soft budget: the live foreign-aware ceiling when refreshed, else
            // the authority's static budget. Already-handed-out logical blocks
            // are never revoked here.
            let total_bytes = self.device_soft_ceiling_bytes(device_id);
            let total_blocks = (total_bytes / KV_BYTES_PER_BLOCK).max(MIN_BLOCKS_PER_ENGINE);
            let new_cap =
                (total_blocks * budget.max_tokens / device_tokens).max(MIN_BLOCKS_PER_ENGINE);
            record.live_cap.store(new_cap, Ordering::Relaxed);
            log::trace!(
                "[memory-authority] KV cap for model {} replica {} on device {}: {} blocks",
                record.model_id,
                record.replica_id,
                device_id,
                new_cap
            );
        }
    }
}

#[cfg(test)]
mod vram_clamp_tests {
    use super::KvCoordinatorInner;
    use crate::app::constants::CUDA_DEVICE_MEMORY_LIMIT_ENV;
    use crate::runtime::device_limits::effective_ceiling_bytes;
    use crate::runtime::memory::{
        MemoryAllocationClass, MemoryAuthority, MemoryClaim, MemoryDomain,
    };
    use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, TensorDtype};
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

    fn cpu_device(id: usize) -> Device {
        Device {
            id,
            name: "test-cpu".to_string(),
            backend: DeviceBackend::Cpu,
            memory_mb: 0,
            compute_units: 1,
            pci_bus_id: None,
            partition_id: None,
            driver_version: None,
            cuda_version: None,
            compute_capability: None,
            utilization_gpu_pct: None,
            temperature_c: None,
            supports_fp16: false,
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

    fn coordinator(info: &DeviceInfo) -> super::KvCoordinator {
        #[cfg(feature = "gpu-device-pool")]
        let authority = MemoryAuthority::new_accounting_only_for_test(info).unwrap();
        #[cfg(not(feature = "gpu-device-pool"))]
        let authority = MemoryAuthority::new(info).unwrap();
        KvCoordinatorInner::new(authority)
    }

    #[test]
    fn cpu_kv_budget_is_derived_from_safe_host_memory() {
        let cpu_id = 4260;
        let mut info = device_info(vec![cpu_device(cpu_id)]);
        // DeviceInfo reports host memory in KiB. The host budget retains 20%,
        // then the KV coordinator dedicates half of that safe budget to KV.
        info.total_memory = 20 * 1024 * 1024;
        let state = coordinator(&info);

        assert_eq!(state.memory.kv_budget_bytes(cpu_id), 8 * GIB);
        let (_, cap, _, _, _) = state.attach_engine(cpu_id, 10, 0, 1);
        assert_eq!(cap, 4_096);
    }

    #[test]
    fn device_bytes_unchanged_without_a_cap() {
        // device id 4242 has no per-device cap env, and the bare
        // CUDA_DEVICE_MEMORY_LIMIT / KAPSL_GPU_MEMORY_LIMIT_MB globals are never
        // set by any test, so the cooperative clamp is a no-op here.
        let info = device_info(vec![cuda_device(4242, 24576)]);
        let state = coordinator(&info);
        assert_eq!(
            state.memory.kv_budget_bytes(4242),
            effective_ceiling_bytes(4242, 24 * GIB, 0) / 2
        );
    }

    #[test]
    fn legacy_cuda_request_plan_reserves_the_host_fallback_too() {
        let device_id = 4252;
        let owner = crate::runtime::memory::MemoryOwner::new(77, 1);
        let state = coordinator(&device_info(vec![cuda_device(device_id, 8 * 1024)]));
        let request = InferenceRequest {
            input: BinaryTensorPacket {
                shape: vec![1],
                dtype: TensorDtype::Uint8,
                data: vec![1, 2, 3, 4],
            },
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };
        let template = MemoryClaim::runtime_allocation(
            MemoryDomain::Cuda { device_id },
            owner,
            MemoryAllocationClass::KvCache,
            "legacy:cuda-kv",
            0,
        );
        let templates = vec![template];
        let plan = state.request_memory_plan(owner, &request, &templates, true);

        assert_eq!(plan.claims().len(), 2);
        assert!(plan
            .claims()
            .iter()
            .any(|claim| claim.domain == MemoryDomain::Cuda { device_id }));
        assert!(plan
            .claims()
            .iter()
            .any(|claim| claim.domain == MemoryDomain::Host));
        assert!(plan.claims().iter().all(|claim| claim.bytes > 0));

        let native_plan = state.request_memory_plan(owner, &request, &templates, false);
        assert!(native_plan
            .claims()
            .iter()
            .any(|claim| claim.domain == MemoryDomain::Cuda { device_id }));
        assert!(!native_plan
            .claims()
            .iter()
            .any(|claim| claim.domain == MemoryDomain::Host));
    }

    #[test]
    fn kv_caps_are_rebalanced_independently_per_device() {
        let first_device = 4250;
        let second_device = 4251;
        let info = device_info(vec![
            cuda_device(first_device, 8 * 1024),
            cuda_device(second_device, 16 * 1024),
        ]);
        let state = coordinator(&info);

        let (_, _, _, _, first_cap) = state.attach_engine(first_device, 10, 0, 1);
        let (_, _, _, _, second_cap) = state.attach_engine(second_device, 20, 3, 1);

        // Each device has one engine, so each receives its own full logical KV
        // budget: half of VRAM divided into 2 MiB blocks.
        assert_eq!(
            first_cap.load(std::sync::atomic::Ordering::Relaxed),
            state.memory.kv_budget_bytes(first_device) / (2 * 1024 * 1024)
        );
        assert_eq!(
            second_cap.load(std::sync::atomic::Ordering::Relaxed),
            state.memory.kv_budget_bytes(second_device) / (2 * 1024 * 1024)
        );
    }

    #[test]
    fn refresh_ceilings_shrinks_fast_then_recovers_slowly() {
        use std::collections::HashMap;
        // 40 GiB card, no cap env → declared == physical.
        let device_id = 4245;
        let info = device_info(vec![cuda_device(device_id, 40 * 1024)]);
        let state = coordinator(&info);

        // The authority retains its device reserve, then exposes half of the
        // safe domain budget to logical KV policy.
        assert_eq!(state.device_soft_ceiling_bytes(device_id), 18 * GIB);

        // A 6 GiB trainer appears: 40 - 6 - 4 (10% reserve) = 30 GiB, and a
        // shrink is applied immediately (no damping on the safety side).
        let mut foreign = HashMap::new();
        foreign.insert(device_id, 6 * GIB);
        state.refresh_ceilings(&foreign);
        assert_eq!(state.device_soft_ceiling_bytes(device_id), 15 * GIB);

        // Trainer exits: target recovers to 40 - 4 = 36 GiB but growth is damped
        // to a quarter of the gap → 30 + (36 - 30)/4 = 31.5 GiB.
        state.refresh_ceilings(&HashMap::new());
        assert_eq!(
            state.device_soft_ceiling_bytes(device_id),
            (30 * GIB + (6 * GIB) / 4) / 2
        );
    }

    #[test]
    fn refresh_ceilings_samples_expose_the_smoothing_gap_and_squeeze() {
        use std::collections::HashMap;
        // Same 40 GiB card as the shrink/recover test; the returned samples must
        // mirror the stored arithmetic so metrics can't drift from behavior.
        let device_id = 4247;
        let info = device_info(vec![cuda_device(device_id, 40 * 1024)]);
        let state = coordinator(&info);

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
        assert_eq!(sample.previous_bytes, 36 * GIB);
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
        let state = coordinator(&info);

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
        let state = coordinator(&info);
        std::env::remove_var(&var);
        // The KV budget sizes to the 8 GiB slice, not the 24 GiB physical card,
        // so the whole downstream KV chain (pools, per-engine caps, rebalancing)
        // self-limits.
        assert_eq!(
            state.memory.kv_budget_bytes(device_id),
            effective_ceiling_bytes(device_id, 8 * GIB, 0) / 2
        );
    }
}
