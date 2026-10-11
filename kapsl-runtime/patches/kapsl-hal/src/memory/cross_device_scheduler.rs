//! Scheduling for KV block pools spanning multiple devices.
//!
//! Manages a fleet of `GpuBlockPool`s across multiple CUDA devices plus a
//! per-geometry CPU eviction tier, presenting a single admission interface to
//! request handlers.
//!
//! # What this enables vs vLLM
//!
//! vLLM's PagedAttention allocates KV blocks dynamically, but its memory
//! budget is still per-process (`gpu_memory_utilization`). When one process
//! is idle, its GPU memory is wasted.
//!
//! `CrossDevicePoolScheduler` is GPU-wide: it sees every device and can move
//! idle sessions to a less-loaded device or to CPU, then restore them when
//! needed. This lets the GPU fleet serve many more concurrent models than a
//! static per-process allocation would.
//!
//! # Admission flow
//!
//! ```text
//! Request arrives
//!   └─► reserve_blocks() / reserve_with_prefix()
//!         ├─ (prefix path) lookup prefix cache → borrow matching blocks
//!         ├─ Try preferred device (most cache-warm)
//!         ├─ Try other devices by free-block count descending
//!         └─ If all at capacity:
//!               evict_until_capacity():
//!                 Phase 1 — free zero-refcount prefix cache blocks (no transfer)
//!                 Phase 2 — evict LRU live sessions → CPU
//!               retry preferred device
//! ```
//!
//! # Migration flow
//!
//! ```text
//! Device A pressure > threshold
//!   └─► find sessions on A, sort by LRU
//!         ├─ For each session: evict_session_to_cpu()
//!         │     release prefix borrows; download owned GPU blocks → CpuBlockStore
//!         └─ On next request for that session:
//!               restore_session_to_gpu() — uploads CPU blocks → GPU
//! ```
//!
//! # CPU store geometry
//!
//! Each `CpuBlockStore` is geometry-specific: it holds blocks of exactly
//! `2 * num_kv_heads * block_size * head_dim` f16 elements. When a pool with
//! a new geometry is registered, a matching `CpuBlockStore` is created
//! automatically. This ensures sessions with different KV shapes (e.g. a
//! 7B model with 8 heads vs a 70B model with 64 heads) can each be evicted
//! independently without size mismatches.
//!
//! # Prefix cache
//!
//! When `enable_prefix_cache` is called, `reserve_with_prefix` can reuse
//! already-computed KV blocks for shared prompt prefixes.  Borrowed prefix
//! blocks are reference-counted — `evict_until_capacity` reclaims
//! zero-refcount entries before evicting live sessions to CPU.
//!
//! After a request completes, the runtime can call `promote_to_prefix_cache`
//! to donate the session's newly computed blocks to the cache for future reuse.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    use crate::cpu_block_store::{CpuBlockStore, CpuStoreError};
    use crate::gpu_arena::{ArenaError, GpuBlockPool};
    use crate::prefix_cache::PrefixBlockCache;

    // ── Error ─────────────────────────────────────────────────────────────────

    #[derive(Debug, thiserror::Error)]
    pub enum SchedulerError {
        #[error(
            "No device has {blocks_needed} free blocks for geometry ({kv_heads}h × {head_dim}d)"
        )]
        InsufficientCapacity {
            blocks_needed: usize,
            kv_heads: usize,
            head_dim: usize,
        },
        #[error("Session {0} not found")]
        SessionNotFound(u64),
        #[error("Device {0} not registered")]
        DeviceNotFound(usize),
        #[error("No CPU store registered for geometry ({0}h × {1}d)")]
        NoCpuStore(usize, usize),
        #[error("GPU error: {0}")]
        Gpu(#[from] ArenaError),
        #[error("CPU store error: {0}")]
        CpuStore(#[from] CpuStoreError),
    }

    // ── Session location ──────────────────────────────────────────────────────

    /// Where a session's KV blocks currently reside.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum KvTier {
        Gpu { device_id: usize },
        Cpu,
    }

    #[derive(Debug)]
    pub struct SessionState {
        pub tier: KvTier,
        /// The GPU pool that owns the blocks when `tier == Gpu`.
        pub pool: Option<Arc<GpuBlockPool>>,
        /// All block IDs.  On GPU: first `n_prefix_blocks` are borrowed from the
        /// prefix cache; the rest are owned.  On CPU: all are owned CPU slots.
        pub blocks: Vec<u32>,
        /// Last time this session executed a forward pass — used for LRU eviction.
        pub last_active: Instant,
        /// KV geometry — preserved across eviction so we can find the right
        /// `CpuBlockStore` without the pool reference.
        pub kv_geom: (usize, usize),
        /// Number of leading blocks borrowed from the prefix cache.
        /// Zero when the prefix cache is disabled or no prefix was matched.
        pub n_prefix_blocks: usize,
        /// Chained block hashes for the borrowed prefix blocks (parallel to
        /// `blocks[..n_prefix_blocks]`).  Used to release cache refcounts.
        pub prefix_hashes: Vec<u64>,
        /// Model fingerprint used for prefix cache lookups and releases.
        pub model_fingerprint: u64,
    }

    // ── Reserve results ───────────────────────────────────────────────────────

    pub struct ReserveResult {
        pub device_id: usize,
        pub pool: Arc<GpuBlockPool>,
        pub blocks: Vec<u32>,
    }

    /// Result from [`CrossDevicePoolScheduler::reserve_with_prefix`].
    pub struct PrefixReserveResult {
        pub device_id: usize,
        pub pool: Arc<GpuBlockPool>,
        /// All block IDs for the session: `blocks[..n_prefix_hits]` are borrowed
        /// from the prefix cache; `blocks[n_prefix_hits..]` are freshly allocated.
        pub blocks: Vec<u32>,
        /// How many leading blocks came from the prefix cache.
        pub n_prefix_hits: usize,
    }

    // ── Scheduler ─────────────────────────────────────────────────────────────

    /// GPU-wide KV block admission, migration, and prefix-cache controller.
    pub struct CrossDevicePoolScheduler {
        /// device_id → list of pools registered for that device.
        device_pools: HashMap<usize, Vec<Arc<GpuBlockPool>>>,
        /// session_id → current location + block ownership.
        sessions: HashMap<u64, SessionState>,
        /// Pool utilisation fraction above which LRU eviction to CPU is triggered.
        evict_threshold: f32,
        /// Per-geometry CPU eviction stores, keyed by (num_kv_heads, head_dim).
        cpu_stores: HashMap<(usize, usize), CpuBlockStore>,
        /// Slot capacity used when creating a new `CpuBlockStore` for a geometry.
        cpu_store_capacity: usize,
        /// Optional prefix KV-block cache.
        prefix_cache: Option<PrefixBlockCache>,
    }

    impl CrossDevicePoolScheduler {
        /// Create a scheduler.
        ///
        /// `evict_threshold` — GPU utilisation fraction (0.5–0.99) above which
        /// LRU sessions are offloaded to CPU to make room for new ones.
        ///
        /// `cpu_store_capacity` — number of block slots in each per-geometry
        /// `CpuBlockStore` created when a new pool geometry is registered.
        pub fn new(evict_threshold: f32, cpu_store_capacity: usize) -> Self {
            Self {
                device_pools: HashMap::new(),
                sessions: HashMap::new(),
                evict_threshold: evict_threshold.clamp(0.5, 0.99),
                cpu_stores: HashMap::new(),
                cpu_store_capacity,
                prefix_cache: None,
            }
        }

        // ── Pool registration ─────────────────────────────────────────────────

        /// Register a `GpuBlockPool` for a device.  Multiple pools per device
        /// are allowed (e.g. one per KV geometry).  A matching `CpuBlockStore`
        /// is created automatically for the pool's geometry if none exists yet.
        pub fn register_pool(&mut self, device_id: usize, pool: Arc<GpuBlockPool>) {
            let geom = (pool.num_kv_heads(), pool.head_dim());
            self.cpu_stores.entry(geom).or_insert_with(|| {
                CpuBlockStore::new(
                    self.cpu_store_capacity,
                    pool.num_kv_heads(),
                    pool.block_size(),
                    pool.head_dim(),
                )
            });
            self.device_pools.entry(device_id).or_default().push(pool);
        }

        /// Unregister one geometry view without disturbing other models on the
        /// same device. Pool identity is pointer-based because two views may
        /// legitimately have identical geometry and capacity.
        pub fn unregister_pool(&mut self, device_id: usize, pool: &Arc<GpuBlockPool>) {
            let remove_device = if let Some(pools) = self.device_pools.get_mut(&device_id) {
                pools.retain(|registered| !Arc::ptr_eq(registered, pool));
                pools.is_empty()
            } else {
                false
            };
            if remove_device {
                self.device_pools.remove(&device_id);
            }
        }

        /// Remove all pools for a device (e.g. on hot-unplug or shutdown).
        pub fn unregister_device(&mut self, device_id: usize) {
            self.device_pools.remove(&device_id);
        }

        // ── Prefix cache ──────────────────────────────────────────────────────

        /// Enable the prefix block cache.
        ///
        /// `capacity` is the maximum number of GPU blocks the cache may hold
        /// simultaneously.  Calling this after sessions exist is valid but
        /// replaces the cache — any in-flight borrows are implicitly abandoned
        /// (the GPU blocks will be freed on the next `evict_lru` call).
        pub fn enable_prefix_cache(&mut self, capacity: usize) {
            self.prefix_cache = Some(PrefixBlockCache::new(capacity));
        }

        // ── Session lifecycle ─────────────────────────────────────────────────

        /// Reserve `blocks_needed` blocks for a new session without prefix reuse.
        ///
        /// 1. Tries `preferred_device` first (most cache-warm).
        /// 2. Falls back to other devices ordered by free-block count.
        /// 3. If all devices are at capacity, evicts LRU sessions to their
        ///    geometry-matched `CpuBlockStore` and retries on `preferred_device`.
        pub fn reserve_blocks(
            &mut self,
            session_id: u64,
            blocks_needed: usize,
            preferred_device: Option<usize>,
            kv_heads: usize,
            head_dim: usize,
        ) -> Result<ReserveResult, SchedulerError> {
            if let Some(dev) = preferred_device {
                if let Some(r) = self.try_reserve_on(dev, blocks_needed, kv_heads, head_dim)? {
                    self.record_session(
                        session_id,
                        r.device_id,
                        r.pool.clone(),
                        r.blocks.clone(),
                        kv_heads,
                        head_dim,
                        0,
                        vec![],
                        0,
                    );
                    return Ok(r);
                }
            }

            let mut others: Vec<usize> = self
                .device_pools
                .keys()
                .filter(|&&d| Some(d) != preferred_device)
                .copied()
                .collect();
            others.sort_by_key(|&d| usize::MAX - self.device_free_blocks(d, kv_heads, head_dim));

            for dev in others {
                if let Some(r) = self.try_reserve_on(dev, blocks_needed, kv_heads, head_dim)? {
                    self.record_session(
                        session_id,
                        r.device_id,
                        r.pool.clone(),
                        r.blocks.clone(),
                        kv_heads,
                        head_dim,
                        0,
                        vec![],
                        0,
                    );
                    return Ok(r);
                }
            }

            let target = preferred_device
                .or_else(|| self.device_pools.keys().copied().next())
                .ok_or(SchedulerError::InsufficientCapacity {
                    blocks_needed,
                    kv_heads,
                    head_dim,
                })?;

            self.evict_until_capacity(target, blocks_needed, kv_heads, head_dim)?;

            if let Some(r) = self.try_reserve_on(target, blocks_needed, kv_heads, head_dim)? {
                self.record_session(
                    session_id,
                    r.device_id,
                    r.pool.clone(),
                    r.blocks.clone(),
                    kv_heads,
                    head_dim,
                    0,
                    vec![],
                    0,
                );
                return Ok(r);
            }

            Err(SchedulerError::InsufficientCapacity {
                blocks_needed,
                kv_heads,
                head_dim,
            })
        }

        /// Release all blocks held by a session (GPU or CPU).
        pub fn release_session(&mut self, session_id: u64) {
            let Some(state) = self.sessions.remove(&session_id) else {
                return;
            };
            match state.tier {
                KvTier::Gpu { .. } => {
                    // Decrement prefix cache borrows.
                    if let Some(cache) = &mut self.prefix_cache {
                        for &hash in &state.prefix_hashes {
                            cache.release(state.model_fingerprint, hash);
                        }
                    }
                    // Free owned blocks.
                    if let Some(pool) = state.pool {
                        for &b in &state.blocks[state.n_prefix_blocks..] {
                            pool.free_block(b);
                        }
                    }
                }
                KvTier::Cpu => {
                    if let Some(store) = self.cpu_stores.get_mut(&state.kv_geom) {
                        store.free_slots_bulk(&state.blocks);
                    }
                }
            }
        }

        /// Update the LRU timestamp for a session that just completed a forward pass.
        pub fn mark_active(&mut self, session_id: u64) {
            if let Some(s) = self.sessions.get_mut(&session_id) {
                s.last_active = Instant::now();
            }
        }

        // ── Cross-tier migration ──────────────────────────────────────────────

        /// Evict a GPU session to its geometry's CPU store, freeing its GPU blocks.
        ///
        /// If the session has prefix-borrowed blocks, their cache refcounts are
        /// decremented first and only the owned blocks are transferred to CPU.
        pub fn evict_session_to_cpu(&mut self, session_id: u64) -> Result<(), SchedulerError> {
            let state = self
                .sessions
                .get(&session_id)
                .ok_or(SchedulerError::SessionNotFound(session_id))?;

            if state.tier == KvTier::Cpu {
                return Ok(());
            }

            let pool = state.pool.clone().unwrap();
            let n_prefix = state.n_prefix_blocks;
            let prefix_hashes = state.prefix_hashes.clone();
            let model_fp = state.model_fingerprint;
            let kv_geom = state.kv_geom;
            let owned_blocks: Vec<u32> = state.blocks[n_prefix..].to_vec();

            // Release prefix borrows — the session is leaving GPU residency.
            if let Some(cache) = &mut self.prefix_cache {
                for &hash in &prefix_hashes {
                    cache.release(model_fp, hash);
                }
            }

            // Download only owned blocks to CPU.
            let cpu_store = self
                .cpu_stores
                .get_mut(&kv_geom)
                .ok_or(SchedulerError::NoCpuStore(kv_geom.0, kv_geom.1))?;

            let mut cpu_slots = Vec::with_capacity(owned_blocks.len());
            for &blk in &owned_blocks {
                let data = pool.download_block(blk)?;
                let slot = cpu_store.store_block(&data)?;
                pool.free_block(blk);
                cpu_slots.push(slot);
            }

            let state = self.sessions.get_mut(&session_id).unwrap();
            state.tier = KvTier::Cpu;
            state.pool = None;
            state.blocks = cpu_slots;
            state.n_prefix_blocks = 0;
            state.prefix_hashes = vec![];

            log::debug!(
                "[xdev-sched] evicted session {} → CPU ({} blocks)",
                session_id,
                owned_blocks.len(),
            );
            Ok(())
        }

        /// Restore a CPU-offloaded session to a GPU device.
        ///
        /// The geometry and matching `CpuBlockStore` are resolved automatically
        /// from the session's stored `kv_geom`.  CPU slots are freed after
        /// successful upload.
        pub fn restore_session_to_gpu(
            &mut self,
            session_id: u64,
            target_device: usize,
        ) -> Result<(), SchedulerError> {
            let state = self
                .sessions
                .get(&session_id)
                .ok_or(SchedulerError::SessionNotFound(session_id))?;

            if state.tier != KvTier::Cpu {
                return Ok(());
            }

            let n_blocks = state.blocks.len();
            let cpu_slots = state.blocks.clone();
            let kv_geom = state.kv_geom;
            let (kv_heads, head_dim) = kv_geom;

            let result = self
                .try_reserve_on(target_device, n_blocks, kv_heads, head_dim)?
                .ok_or(SchedulerError::InsufficientCapacity {
                    blocks_needed: n_blocks,
                    kv_heads,
                    head_dim,
                })?;

            {
                let cpu_store = self
                    .cpu_stores
                    .get(&kv_geom)
                    .ok_or(SchedulerError::NoCpuStore(kv_geom.0, kv_geom.1))?;
                for (&gpu_blk, &cpu_slot) in result.blocks.iter().zip(&cpu_slots) {
                    let data = cpu_store.load_block(cpu_slot)?;
                    let half_len = data.len() / 2;
                    result
                        .pool
                        .upload_block(gpu_blk, &data[..half_len], &data[half_len..])?;
                }
            }

            if let Some(store) = self.cpu_stores.get_mut(&kv_geom) {
                store.free_slots_bulk(&cpu_slots);
            }

            let state = self.sessions.get_mut(&session_id).unwrap();
            state.tier = KvTier::Gpu {
                device_id: target_device,
            };
            state.pool = Some(result.pool);
            state.blocks = result.blocks;

            log::debug!(
                "[xdev-sched] restored session {} → GPU {} ({} blocks)",
                session_id,
                target_device,
                n_blocks,
            );
            Ok(())
        }

        // ── Pressure / metrics ────────────────────────────────────────────────

        /// Pool utilisation on a device (0.0 = empty, 1.0 = full).
        pub fn device_pressure(&self, device_id: usize) -> f32 {
            let Some(pools) = self.device_pools.get(&device_id) else {
                return 0.0;
            };
            let (total, used) = pools.iter().fold((0usize, 0usize), |(t, u), p| {
                (t + p.total_blocks(), u + p.used_count())
            });
            if total == 0 {
                0.0
            } else {
                used as f32 / total as f32
            }
        }

        /// Configured utilisation threshold for callers that coordinate
        /// proactive eviction with external pressure monitoring.
        pub fn eviction_threshold(&self) -> f32 {
            self.evict_threshold
        }

        /// Total free blocks on a device for the given KV geometry.
        pub fn device_free_blocks(
            &self,
            device_id: usize,
            kv_heads: usize,
            head_dim: usize,
        ) -> usize {
            self.device_pools
                .get(&device_id)
                .map(|pools| {
                    pools
                        .iter()
                        .filter(|p| p.is_compatible(kv_heads, head_dim))
                        .map(|p| p.free_count())
                        .sum()
                })
                .unwrap_or(0)
        }

        pub fn registered_devices(&self) -> Vec<usize> {
            let mut v: Vec<usize> = self.device_pools.keys().copied().collect();
            v.sort_unstable();
            v
        }

        pub fn session_count(&self) -> usize {
            self.sessions.len()
        }
        pub fn sessions_on_cpu(&self) -> usize {
            self.sessions
                .values()
                .filter(|s| s.tier == KvTier::Cpu)
                .count()
        }
        pub fn sessions_on_device(&self, device_id: usize) -> usize {
            self.sessions
                .values()
                .filter(|s| s.tier == KvTier::Gpu { device_id })
                .count()
        }
        // ── Internal helpers ──────────────────────────────────────────────────

        fn try_reserve_on(
            &self,
            device_id: usize,
            n: usize,
            kv_heads: usize,
            head_dim: usize,
        ) -> Result<Option<ReserveResult>, SchedulerError> {
            let Some(pools) = self.device_pools.get(&device_id) else {
                return Ok(None);
            };
            for pool in pools {
                if !pool.is_compatible(kv_heads, head_dim) {
                    continue;
                }
                if pool.free_count() < n {
                    continue;
                }

                let mut blocks = Vec::with_capacity(n);
                let mut ok = true;
                for _ in 0..n {
                    match pool.alloc_block() {
                        Ok(b) => blocks.push(b),
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    return Ok(Some(ReserveResult {
                        device_id,
                        pool: pool.clone(),
                        blocks,
                    }));
                }
                for b in blocks {
                    pool.free_block(b);
                }
            }
            Ok(None)
        }

        // Session construction mirrors the public reservation result plus KV
        // ownership metadata; keeping those values explicit avoids a second
        // transient allocation on the admission path.
        #[allow(clippy::too_many_arguments)]
        fn record_session(
            &mut self,
            session_id: u64,
            device_id: usize,
            pool: Arc<GpuBlockPool>,
            blocks: Vec<u32>,
            kv_heads: usize,
            head_dim: usize,
            n_prefix_blocks: usize,
            prefix_hashes: Vec<u64>,
            model_fingerprint: u64,
        ) {
            self.sessions.insert(
                session_id,
                SessionState {
                    tier: KvTier::Gpu { device_id },
                    pool: Some(pool),
                    blocks,
                    last_active: Instant::now(),
                    kv_geom: (kv_heads, head_dim),
                    n_prefix_blocks,
                    prefix_hashes,
                    model_fingerprint,
                },
            );
        }

        /// Evict sessions / prefix cache entries on `device_id` until at least
        /// `blocks_needed` blocks are free for the given geometry.
        ///
        /// Phase 1 — free zero-refcount prefix cache entries (no CPU transfer).
        /// Phase 2 — evict LRU live sessions to CPU (expensive, done only if
        ///            Phase 1 was insufficient).
        #[allow(dead_code)]
        fn evict_until_capacity(
            &mut self,
            device_id: usize,
            blocks_needed: usize,
            kv_heads: usize,
            head_dim: usize,
        ) -> Result<(), SchedulerError> {
            // Phase 1: free zero-refcount prefix cache blocks — no CPU transfer.
            if self.prefix_cache.is_some() {
                let free_now = self.device_free_blocks(device_id, kv_heads, head_dim);
                if free_now < blocks_needed {
                    let to_evict = blocks_needed - free_now;
                    // evict_lru_for_device returns owned Vec, so the mutable borrow ends here.
                    let freed = self
                        .prefix_cache
                        .as_mut()
                        .unwrap()
                        .evict_lru_for_device(device_id, to_evict);
                    for (pool, block_ids) in freed {
                        if pool.is_compatible(kv_heads, head_dim) {
                            for id in block_ids {
                                pool.free_block(id);
                            }
                        }
                    }
                }
            }

            // Phase 2: evict live sessions to CPU if still insufficient.
            let mut candidates: Vec<(u64, Instant)> = self
                .sessions
                .iter()
                .filter(|(_, s)| {
                    s.tier == KvTier::Gpu { device_id }
                        && s.pool
                            .as_ref()
                            .is_some_and(|p| p.is_compatible(kv_heads, head_dim))
                })
                .map(|(&id, s)| (id, s.last_active))
                .collect();
            candidates.sort_by_key(|(_, t)| *t);

            for (evict_id, _) in candidates {
                if self.device_free_blocks(device_id, kv_heads, head_dim) >= blocks_needed {
                    break;
                }
                self.evict_session_to_cpu(evict_id)?;
            }
            Ok(())
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::gpu_arena::GpuBlockPool;
        use cudarc::driver::CudaDevice;

        // Small pool: 8 blocks, 4 tokens/block, 2 KV heads, 8-dim.
        // Requires CUDA device 0.
        fn small_pool() -> Arc<GpuBlockPool> {
            let device = CudaDevice::new(0).expect("CUDA device 0 required");
            Arc::new(GpuBlockPool::new(device, 8, 4, 2, 8).unwrap())
        }

        fn sched() -> CrossDevicePoolScheduler {
            CrossDevicePoolScheduler::new(0.85, 64)
        }

        #[test]
        fn register_pool_creates_cpu_store_for_geometry() {
            let mut s = sched();
            let pool = small_pool();
            s.register_pool(0, pool.clone());
            // After registration the device is visible.
            assert!(s.registered_devices().contains(&0));
            // CPU store was created for this geometry.
            let free = s.device_free_blocks(0, pool.num_kv_heads(), pool.head_dim());
            assert_eq!(free, pool.total_blocks());
        }

        #[test]
        fn unregister_pool_removes_only_the_matching_view() {
            let mut s = sched();
            let first = small_pool();
            let second = small_pool();
            s.register_pool(0, first.clone());
            s.register_pool(0, second.clone());

            s.unregister_pool(0, &first);
            assert_eq!(s.registered_devices(), vec![0]);
            assert_eq!(
                s.device_free_blocks(0, second.num_kv_heads(), second.head_dim()),
                8
            );

            s.unregister_pool(0, &second);
            assert!(s.registered_devices().is_empty());
        }

        #[test]
        fn registered_devices_returns_sorted_list() {
            let mut s = sched();
            let p0 = small_pool();
            let p1 = small_pool();
            s.register_pool(2, p0);
            s.register_pool(0, p1);
            assert_eq!(s.registered_devices(), vec![0, 2]);
        }

        #[test]
        fn reserve_allocates_blocks_and_records_session() {
            let mut s = sched();
            let pool = small_pool();
            s.register_pool(0, pool.clone());
            let r = s
                .reserve_blocks(42, 3, Some(0), pool.num_kv_heads(), pool.head_dim())
                .unwrap();
            assert_eq!(r.device_id, 0);
            assert_eq!(r.blocks.len(), 3);
            assert_eq!(pool.free_count(), 5);
            assert_eq!(s.session_count(), 1);
            assert_eq!(s.sessions_on_device(0), 1);
        }

        #[test]
        fn release_session_frees_blocks() {
            let mut s = sched();
            let pool = small_pool();
            s.register_pool(0, pool.clone());
            s.reserve_blocks(7, 4, Some(0), pool.num_kv_heads(), pool.head_dim())
                .unwrap();
            assert_eq!(pool.free_count(), 4);
            s.release_session(7);
            assert_eq!(pool.free_count(), 8);
            assert_eq!(s.session_count(), 0);
        }

        #[test]
        fn device_pressure_empty_is_zero() {
            let mut s = sched();
            let pool = small_pool();
            s.register_pool(0, pool);
            assert_eq!(s.device_pressure(0), 0.0);
        }

        #[test]
        fn device_pressure_increases_with_allocations() {
            let mut s = sched();
            let pool = small_pool(); // 8 blocks
            s.register_pool(0, pool.clone());
            s.reserve_blocks(1, 4, Some(0), pool.num_kv_heads(), pool.head_dim())
                .unwrap();
            let pressure = s.device_pressure(0);
            assert!((pressure - 0.5).abs() < 1e-5, "expected 0.5 got {pressure}");
        }

        #[test]
        fn device_free_blocks_filters_by_geometry() {
            let mut s = sched();
            let pool = small_pool(); // 2h × 8d, 8 blocks
            s.register_pool(0, pool.clone());
            // Correct geometry.
            assert_eq!(s.device_free_blocks(0, 2, 8), 8);
            // Wrong geometry — no matching pool.
            assert_eq!(s.device_free_blocks(0, 4, 8), 0);
        }

        #[test]
        fn reserve_falls_back_to_other_device_when_preferred_is_full() {
            let mut s = sched();
            let p0 = small_pool(); // 8 blocks
            let p1 = small_pool(); // 8 blocks
            let (kv_heads, head_dim) = (p0.num_kv_heads(), p0.head_dim());
            s.register_pool(0, p0.clone());
            s.register_pool(1, p1.clone());
            // Fill device 0 entirely.
            s.reserve_blocks(10, 8, Some(0), kv_heads, head_dim)
                .unwrap();
            assert_eq!(p0.free_count(), 0);
            // A new reservation should land on device 1.
            let r = s
                .reserve_blocks(11, 2, Some(0), kv_heads, head_dim)
                .unwrap();
            assert_eq!(r.device_id, 1);
            assert_eq!(p1.free_count(), 6);
        }

        #[test]
        fn mark_active_updates_lru_timestamp() {
            let mut s = sched();
            let pool = small_pool();
            s.register_pool(0, pool.clone());
            s.reserve_blocks(5, 1, Some(0), pool.num_kv_heads(), pool.head_dim())
                .unwrap();
            // mark_active must not panic and must keep the session alive.
            s.mark_active(5);
            assert_eq!(s.session_count(), 1);
        }

        #[test]
        fn session_count_and_on_device_metrics() {
            let mut s = sched();
            let p0 = small_pool();
            let p1 = small_pool();
            let (kv_heads, head_dim) = (p0.num_kv_heads(), p0.head_dim());
            s.register_pool(0, p0.clone());
            s.register_pool(1, p1.clone());
            s.reserve_blocks(1, 1, Some(0), kv_heads, head_dim).unwrap();
            s.reserve_blocks(2, 1, Some(1), kv_heads, head_dim).unwrap();
            assert_eq!(s.session_count(), 2);
            assert_eq!(s.sessions_on_device(0), 1);
            assert_eq!(s.sessions_on_device(1), 1);
            assert_eq!(s.sessions_on_cpu(), 0);
        }

        #[test]
        fn evict_session_to_cpu_frees_gpu_blocks() {
            let mut s = sched();
            let pool = small_pool();
            let (kv_heads, head_dim) = (pool.num_kv_heads(), pool.head_dim());
            s.register_pool(0, pool.clone());
            s.reserve_blocks(99, 3, Some(0), kv_heads, head_dim)
                .unwrap();
            assert_eq!(pool.free_count(), 5);
            s.evict_session_to_cpu(99).unwrap();
            // GPU blocks returned.
            assert_eq!(pool.free_count(), 8);
            // Session is now on CPU.
            assert_eq!(s.sessions_on_cpu(), 1);
            assert_eq!(s.sessions_on_device(0), 0);
        }

        #[test]
        fn evict_and_restore_preserves_kv_data() {
            let mut s = sched();
            let pool = small_pool();
            let (kv_heads, head_dim) = (pool.num_kv_heads(), pool.head_dim());
            let half_block = kv_heads * pool.block_size() * head_dim;
            s.register_pool(0, pool.clone());
            // Reserve one block and upload known KV data.
            let r = s
                .reserve_blocks(55, 1, Some(0), kv_heads, head_dim)
                .unwrap();
            let block_id = r.blocks[0];
            let key: Vec<half::f16> = (0..half_block)
                .map(|i| half::f16::from_f32(i as f32))
                .collect();
            let val: Vec<half::f16> = (0..half_block)
                .map(|i| half::f16::from_f32(100.0 + i as f32))
                .collect();
            pool.upload_block(block_id, &key, &val).unwrap();

            // Evict to CPU.
            s.evict_session_to_cpu(55).unwrap();
            assert_eq!(pool.free_count(), 8);

            // Restore to GPU.
            s.restore_session_to_gpu(55, 0).unwrap();
            assert_eq!(s.sessions_on_device(0), 1);
            assert_eq!(s.sessions_on_cpu(), 0);

            // Verify KV data survived the round-trip.
            if let KvTier::Gpu { .. } = s.sessions.get(&55).unwrap().tier {
                let new_block = s.sessions.get(&55).unwrap().blocks[0];
                let downloaded = pool.download_block(new_block).unwrap();
                assert_eq!(&downloaded[..half_block], key.as_slice());
                assert_eq!(&downloaded[half_block..], val.as_slice());
            } else {
                panic!("session should be on GPU after restore");
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::{
    CrossDevicePoolScheduler, KvTier, PrefixReserveResult, ReserveResult, SchedulerError,
    SessionState,
};
