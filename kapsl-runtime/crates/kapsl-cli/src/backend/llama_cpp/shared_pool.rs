//! Runtime-owned CUDA KV-pool callbacks for lazy llama.cpp backend packs.
//!
//! Nothing ownership-bearing crosses the pack boundary. The pack receives a
//! device pointer, block-table geometry, and C callbacks. All allocations and
//! reservation bookkeeping remain in Kapsl core and are attributed to the
//! model/replica in the process-wide `GpuDevicePool`.

use kapsl_backend_abi::*;
#[cfg(any(feature = "gpu-device-pool", test))]
use std::collections::HashMap;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Debug, Default)]
#[cfg(any(feature = "gpu-device-pool", test))]
struct SequenceReservation {
    layers: Vec<Vec<u32>>,
    logical_blocks: usize,
}

/// CPU-side ownership and block-table mirror. Kept independent of CUDA so the
/// grow/release invariants can be certified on every CI runner.
#[derive(Debug)]
#[cfg(any(feature = "gpu-device-pool", test))]
struct ReservationBook {
    num_layers: usize,
    max_blocks_per_sequence: usize,
    sequence_slots: usize,
    sequences: HashMap<u64, SequenceReservation>,
    slot_owners: Vec<Option<u64>>,
    combined_host: Vec<u32>,
    dirty: bool,
}

#[cfg(any(feature = "gpu-device-pool", test))]
impl ReservationBook {
    fn new(num_layers: usize, max_blocks_per_sequence: usize, sequence_slots: usize) -> Self {
        let sequence_slots = sequence_slots.max(1);
        let sequence_stride = num_layers.saturating_mul(max_blocks_per_sequence);
        Self {
            num_layers,
            max_blocks_per_sequence,
            sequence_slots,
            sequences: HashMap::new(),
            slot_owners: vec![None; sequence_slots],
            combined_host: vec![0; sequence_slots.saturating_mul(sequence_stride)],
            dirty: true,
        }
    }

    fn sequence_stride(&self) -> usize {
        self.num_layers.saturating_mul(self.max_blocks_per_sequence)
    }

    fn growth_blocks(&self, sequence_id: u64, logical_blocks: usize) -> Result<usize, String> {
        if logical_blocks == 0 || logical_blocks > self.max_blocks_per_sequence {
            return Err(format!(
                "logical block request {logical_blocks} exceeds per-sequence limit {}",
                self.max_blocks_per_sequence
            ));
        }
        let current = self
            .sequences
            .get(&sequence_id)
            .map(|reservation| reservation.logical_blocks)
            .unwrap_or(0);
        if logical_blocks < current {
            return Err(format!(
                "shared KV reservations are grow-only: sequence {sequence_id} owns {current} blocks, requested {logical_blocks}"
            ));
        }
        logical_blocks
            .saturating_sub(current)
            .checked_mul(self.num_layers)
            .ok_or_else(|| "shared KV growth size overflow".to_string())
    }

    fn logical_blocks(&self, sequence_id: u64) -> usize {
        self.sequences
            .get(&sequence_id)
            .map(|reservation| reservation.logical_blocks)
            .unwrap_or(0)
    }

    fn commit_growth(
        &mut self,
        sequence_id: u64,
        slot: usize,
        logical_blocks: usize,
        fresh_blocks: Vec<u32>,
    ) -> Result<(), String> {
        if slot >= self.sequence_slots {
            return Err(format!(
                "sequence slot {slot} exceeds slot count {}",
                self.sequence_slots
            ));
        }
        if let Some(owner) = self.slot_owners[slot] {
            if owner != sequence_id {
                return Err(format!(
                    "sequence slot {slot} is still owned by sequence {owner}"
                ));
            }
        }
        if let Some(existing) = self
            .slot_owners
            .iter()
            .position(|owner| *owner == Some(sequence_id))
        {
            if existing != slot {
                return Err(format!(
                    "sequence {sequence_id} already owns slot {existing}, cannot move it to slot {slot}"
                ));
            }
        }
        let required = self.growth_blocks(sequence_id, logical_blocks)?;
        if fresh_blocks.len() != required {
            return Err(format!(
                "shared KV growth supplied {} blocks, expected {required}",
                fresh_blocks.len()
            ));
        }

        let sequence_stride = self.sequence_stride();
        let reservation =
            self.sequences
                .entry(sequence_id)
                .or_insert_with(|| SequenceReservation {
                    layers: vec![Vec::new(); self.num_layers],
                    logical_blocks: 0,
                });
        let previous = reservation.logical_blocks;
        let delta = logical_blocks.saturating_sub(previous);
        for layer in 0..self.num_layers {
            let start = layer.saturating_mul(delta);
            reservation.layers[layer]
                .extend_from_slice(&fresh_blocks[start..start.saturating_add(delta)]);
        }
        reservation.logical_blocks = logical_blocks;
        self.slot_owners[slot] = Some(sequence_id);

        let sequence_base = slot.saturating_mul(sequence_stride);
        for layer in 0..self.num_layers {
            let layer_base = sequence_base + layer.saturating_mul(self.max_blocks_per_sequence);
            self.combined_host[layer_base..layer_base + logical_blocks]
                .copy_from_slice(&reservation.layers[layer][..logical_blocks]);
        }
        self.dirty = true;
        Ok(())
    }

    fn rollback_growth(&mut self, sequence_id: u64, slot: usize, previous: usize) -> Vec<u32> {
        let Some(reservation) = self.sequences.get_mut(&sequence_id) else {
            return Vec::new();
        };
        let current = reservation.logical_blocks;
        if previous >= current {
            return Vec::new();
        }
        let mut released = Vec::new();
        for layer in &mut reservation.layers {
            released.extend(layer.drain(previous..));
        }
        reservation.logical_blocks = previous;

        let sequence_base =
            slot.saturating_mul(self.num_layers.saturating_mul(self.max_blocks_per_sequence));
        for layer in 0..self.num_layers {
            let layer_base = sequence_base + layer.saturating_mul(self.max_blocks_per_sequence);
            self.combined_host[layer_base + previous..layer_base + current].fill(0);
        }
        if previous == 0 {
            self.sequences.remove(&sequence_id);
            if self.slot_owners.get(slot) == Some(&Some(sequence_id)) {
                self.slot_owners[slot] = None;
            }
        }
        self.dirty = true;
        released
    }

    fn release(&mut self, sequence_id: u64) -> Vec<u32> {
        let Some(reservation) = self.sequences.remove(&sequence_id) else {
            return Vec::new();
        };
        if let Some(slot) = self
            .slot_owners
            .iter()
            .position(|owner| *owner == Some(sequence_id))
        {
            self.slot_owners[slot] = None;
            let start = slot.saturating_mul(self.sequence_stride());
            let end = start.saturating_add(self.sequence_stride());
            self.combined_host[start..end].fill(0);
            self.dirty = true;
        }
        reservation.layers.into_iter().flatten().collect()
    }

    #[cfg(feature = "gpu-device-pool")]
    fn contains(&self, sequence_id: u64) -> bool {
        self.sequences.contains_key(&sequence_id)
    }
}

#[cfg(feature = "gpu-device-pool")]
mod cuda {
    use super::*;
    use cudarc::driver::{result, sys};
    use kapsl_hal::gpu_arena::{
        GpuAllocation, GpuDevicePool, GpuKvPoolView, PoolAllocationClass, PoolOwner,
    };
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};

    pub(crate) struct LlamaCppSharedPoolHost {
        device_pool: Arc<GpuDevicePool>,
        device_id: usize,
        model_id: u32,
        replica_id: u32,
        live_pools: Mutex<HashSet<usize>>,
    }

    impl LlamaCppSharedPoolHost {
        pub(crate) fn new(
            device_pool: Arc<GpuDevicePool>,
            device_id: usize,
            model_id: u32,
            replica_id: u32,
        ) -> Self {
            Self {
                device_pool,
                device_id,
                model_id,
                replica_id,
                live_pools: Mutex::new(HashSet::new()),
            }
        }

        pub(crate) fn callbacks(&mut self) -> KapslLlamaHostCallbacksV1 {
            KapslLlamaHostCallbacksV1 {
                struct_size: std::mem::size_of::<KapslLlamaHostCallbacksV1>() as u32,
                user_data: (self as *mut Self).cast(),
                log: Some(super::host_log_bridge),
                create_shared_pool: Some(create_shared_pool),
                destroy_shared_pool: Some(destroy_shared_pool),
                shared_pool_bytes: Some(shared_pool_bytes),
            }
        }

        fn register(&self, pool: *mut RuntimeSharedPool) {
            self.live_pools
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(pool as usize);
        }

        fn unregister(&self, pool: *mut c_void) -> bool {
            self.live_pools
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&(pool as usize))
        }

        fn contains(&self, pool: *mut c_void) -> bool {
            self.live_pools
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains(&(pool as usize))
        }
    }

    impl Drop for LlamaCppSharedPoolHost {
        fn drop(&mut self) {
            let live: Vec<usize> = self
                .live_pools
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .drain()
                .collect();
            for pointer in live {
                log::error!(
                    "llama.cpp pack did not destroy runtime shared pool before host shutdown"
                );
                // SAFETY: only pointers allocated and registered below enter this set.
                unsafe { drop(Box::from_raw(pointer as *mut RuntimeSharedPool)) };
            }
        }
    }

    struct RuntimeSharedPool {
        view: GpuKvPoolView,
        device_pool: Arc<GpuDevicePool>,
        block_table: Option<GpuAllocation>,
        block_table_bytes: usize,
        block_size_tokens: usize,
        book: Mutex<ReservationBook>,
    }

    impl RuntimeSharedPool {
        fn new(
            host: &LlamaCppSharedPoolHost,
            geometry: &KapslSharedPoolGeometryV1,
        ) -> Result<Self, String> {
            validate_geometry(host, geometry)?;
            let requested_blocks = usize::try_from(geometry.requested_blocks)
                .map_err(|_| "requested shared KV blocks exceed this platform".to_string())?;
            let block_size_tokens = geometry.block_size_tokens as usize;
            let num_layers = geometry.num_layers as usize;
            let num_kv_heads = geometry.num_kv_heads as usize;
            let head_dim = geometry.key_head_dim as usize;
            let max_blocks_per_sequence = geometry.max_blocks_per_sequence as usize;
            let sequence_slots = (geometry.max_sequences as usize).max(1);
            let owner =
                PoolOwner::gguf(host.model_id, host.replica_id, PoolAllocationClass::KvCache);
            let view = GpuKvPoolView::from_device_pool(
                Arc::clone(&host.device_pool),
                owner,
                requested_blocks,
                block_size_tokens,
                num_kv_heads,
                head_dim,
            )
            .map_err(|error| format!("create runtime shared KV view: {error}"))?;
            if view.total_blocks() < num_layers {
                return Err(format!(
                    "runtime shared KV view has {} blocks, but {num_layers} are required for one logical block",
                    view.total_blocks()
                ));
            }

            let table_entries = sequence_slots
                .checked_mul(num_layers)
                .and_then(|value| value.checked_mul(max_blocks_per_sequence))
                .ok_or_else(|| "shared KV block-table size overflow".to_string())?;
            let block_table_bytes = table_entries
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| "shared KV block-table byte size overflow".to_string())?;
            let table_owner = owner.with_class(PoolAllocationClass::BlockTable);
            let block_table = host
                .device_pool
                .alloc(table_owner, block_table_bytes, std::mem::align_of::<u32>())
                .map_err(|error| format!("allocate runtime shared KV block table: {error}"))?;
            let pool = Self {
                view,
                device_pool: Arc::clone(&host.device_pool),
                block_table: Some(block_table),
                block_table_bytes,
                block_size_tokens,
                book: Mutex::new(ReservationBook::new(
                    num_layers,
                    max_blocks_per_sequence,
                    sequence_slots,
                )),
            };
            pool.upload_table()?;
            Ok(pool)
        }

        fn table_ptr(&self) -> *mut u32 {
            self.block_table
                .as_ref()
                .map(|allocation| self.device_pool.allocation_ptr(allocation).cast())
                .unwrap_or(std::ptr::null_mut())
        }

        fn upload_table(&self) -> Result<(), String> {
            let book = self
                .book
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.device_pool
                .device()
                .bind_to_thread()
                .map_err(|error| format!("bind CUDA device for shared KV table upload: {error}"))?;
            let device_ptr = self.table_ptr() as usize as sys::CUdeviceptr;
            // SAFETY: block_table owns exactly combined_host.len() u32s in the
            // runtime pool and the synchronous copy completes before return.
            unsafe { result::memcpy_htod_sync(device_ptr, &book.combined_host) }
                .map_err(|error| format!("upload runtime shared KV block table: {error}"))?;
            Ok(())
        }

        fn reserve(
            &self,
            sequence_id: u64,
            slot: usize,
            tokens_needed: u32,
            upload: bool,
        ) -> Result<u32, String> {
            let logical_blocks = (tokens_needed as usize)
                .div_ceil(self.block_size_tokens)
                .max(1);
            let (previous, required) = {
                let book = self
                    .book
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                (
                    book.logical_blocks(sequence_id),
                    book.growth_blocks(sequence_id, logical_blocks)?,
                )
            };
            let mut fresh = Vec::with_capacity(required);
            for _ in 0..required {
                match self.view.alloc_block() {
                    Ok(block) => fresh.push(block),
                    Err(error) => {
                        for block in fresh {
                            self.view.free_block(block);
                        }
                        return Err(format!("reserve runtime shared KV block: {error}"));
                    }
                }
            }
            let commit = self
                .book
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .commit_growth(sequence_id, slot, logical_blocks, fresh.clone());
            if let Err(error) = commit {
                for block in fresh {
                    self.view.free_block(block);
                }
                return Err(error);
            }
            if upload {
                if let Err(error) = self.upload_table() {
                    let released = self
                        .book
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .rollback_growth(sequence_id, slot, previous);
                    for block in released {
                        self.view.free_block(block);
                    }
                    return Err(error);
                }
                self.book
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .dirty = false;
            }
            u32::try_from(logical_blocks)
                .map_err(|_| "logical block count exceeds ABI v1".to_string())
        }

        fn commit_sequences(&self) -> Result<(), String> {
            let dirty = self
                .book
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .dirty;
            if dirty {
                self.upload_table()?;
                self.book
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .dirty = false;
            }
            Ok(())
        }

        fn release(&self, sequence_id: u64) {
            let blocks = self
                .book
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .release(sequence_id);
            for block in blocks {
                self.view.free_block(block);
            }
        }

        fn bytes(&self) -> u64 {
            u64::try_from(
                self.view
                    .used_bytes()
                    .saturating_add(self.block_table_bytes),
            )
            .unwrap_or(u64::MAX)
        }
    }

    impl Drop for RuntimeSharedPool {
        fn drop(&mut self) {
            if let Some(allocation) = self.block_table.take() {
                if let Err(error) = self.device_pool.free(allocation) {
                    log::error!("failed to release llama.cpp shared KV block table: {error}");
                }
            }
        }
    }

    fn validate_geometry(
        host: &LlamaCppSharedPoolHost,
        geometry: &KapslSharedPoolGeometryV1,
    ) -> Result<(), String> {
        if geometry.struct_size < std::mem::size_of::<KapslSharedPoolGeometryV1>() as u32 {
            return Err("shared KV geometry is smaller than ABI v1".to_string());
        }
        if geometry.device_id as usize != host.device_id {
            return Err(format!(
                "shared KV geometry requested device {}, host owns device {}",
                geometry.device_id, host.device_id
            ));
        }
        if geometry.requested_blocks == 0
            || geometry.block_size_tokens == 0
            || geometry.num_layers == 0
            || geometry.num_kv_heads == 0
            || geometry.key_head_dim == 0
            || geometry.max_blocks_per_sequence == 0
            || geometry.max_sequences == 0
        {
            return Err("shared KV geometry contains a zero dimension".to_string());
        }
        if geometry.key_head_dim != geometry.value_head_dim {
            return Err(format!(
                "shared KV requires equal K/V head dimensions, got {}/{}",
                geometry.key_head_dim, geometry.value_head_dim
            ));
        }
        if geometry.element_bytes != 2 {
            return Err(format!(
                "shared KV ABI v1 requires f16 elements, got {} bytes",
                geometry.element_bytes
            ));
        }
        Ok(())
    }

    unsafe fn host<'a>(user_data: *mut c_void) -> Option<&'a LlamaCppSharedPoolHost> {
        if user_data.is_null() {
            None
        } else {
            // SAFETY: PackInstance retains this host until pack shutdown returns.
            Some(unsafe { &*(user_data as *const LlamaCppSharedPoolHost) })
        }
    }

    unsafe fn pool<'a>(pool_context: *mut c_void) -> Option<&'a RuntimeSharedPool> {
        if pool_context.is_null() {
            None
        } else {
            // SAFETY: the pointer remains registered until destroy_shared_pool.
            Some(unsafe { &*(pool_context as *const RuntimeSharedPool) })
        }
    }

    unsafe extern "C" fn create_shared_pool(
        user_data: *mut c_void,
        geometry: *const KapslSharedPoolGeometryV1,
        descriptor_out: *mut KapslSharedPoolDescriptorV1,
        error_out: *mut KapslOwnedBuffer,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            if !error_out.is_null() {
                // Host callbacks cannot transfer a core allocation to the pack's
                // `free_buffer`; errors are logged and returned as status only.
                unsafe { *error_out = KapslOwnedBuffer::empty() };
            }
            let Some(host) = (unsafe { host(user_data) }) else {
                return KAPSL_STATUS_INVALID_ARGUMENT;
            };
            if geometry.is_null() || descriptor_out.is_null() {
                return KAPSL_STATUS_INVALID_ARGUMENT;
            }
            let geometry = unsafe { &*geometry };
            let shared = match RuntimeSharedPool::new(host, geometry) {
                Ok(shared) => Box::new(shared),
                Err(error) => {
                    log::error!("create llama.cpp runtime shared pool: {error}");
                    return KAPSL_STATUS_BACKEND_ERROR;
                }
            };
            let sequence_stride = match (geometry.num_layers as u64)
                .checked_mul(geometry.max_blocks_per_sequence as u64)
                .and_then(|value| u32::try_from(value).ok())
            {
                Some(value) => value,
                None => return KAPSL_STATUS_INVALID_ARGUMENT,
            };
            let pointer = Box::into_raw(shared);
            let shared = unsafe { &*pointer };
            let descriptor = KapslSharedPoolDescriptorV1 {
                struct_size: std::mem::size_of::<KapslSharedPoolDescriptorV1>() as u32,
                pool_context: pointer.cast(),
                device_base: shared.device_pool.base_ptr(),
                addressable_blocks: u64::try_from(shared.view.addressable_blocks())
                    .unwrap_or(u64::MAX),
                block_table_device: shared.table_ptr(),
                block_table_layer_stride: geometry.max_blocks_per_sequence,
                block_table_sequence_stride: sequence_stride,
                sequence_slots: geometry.max_sequences,
                reserve: Some(reserve),
                reserve_sequence: Some(reserve_sequence),
                commit_sequences: Some(commit_sequences),
                release: Some(release),
                touch: Some(touch),
            };
            host.register(pointer);
            unsafe { *descriptor_out = descriptor };
            log::info!(
                "llama.cpp pack attached runtime-owned shared KV pool: device={} model={} replica={} cap_blocks={} addressable_blocks={}",
                host.device_id,
                host.model_id,
                host.replica_id,
                shared.view.total_blocks(),
                shared.view.addressable_blocks(),
            );
            KAPSL_STATUS_OK
        }))
        .unwrap_or(KAPSL_STATUS_PANIC)
    }

    unsafe extern "C" fn destroy_shared_pool(user_data: *mut c_void, pool_context: *mut c_void) {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let Some(host) = (unsafe { host(user_data) }) else {
                return;
            };
            if !pool_context.is_null() && host.unregister(pool_context) {
                // SAFETY: unregister proves this is a live Box allocated above.
                unsafe { drop(Box::from_raw(pool_context as *mut RuntimeSharedPool)) };
            }
        }));
    }

    unsafe extern "C" fn shared_pool_bytes(
        user_data: *mut c_void,
        pool_context: *mut c_void,
    ) -> u64 {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(host) = (unsafe { host(user_data) }) else {
                return 0;
            };
            if !host.contains(pool_context) {
                return 0;
            }
            unsafe { pool(pool_context) }
                .map(RuntimeSharedPool::bytes)
                .unwrap_or(0)
        }))
        .unwrap_or(0)
    }

    unsafe fn write_reservation_outputs(
        shared: &RuntimeSharedPool,
        logical_blocks: u32,
        block_table_device_out: *mut *mut u32,
        blocks_out: *mut u32,
    ) -> u32 {
        if block_table_device_out.is_null() || blocks_out.is_null() {
            return 0;
        }
        unsafe {
            *block_table_device_out = shared.table_ptr();
            *blocks_out = logical_blocks;
        }
        1
    }

    unsafe extern "C" fn reserve(
        pool_context: *mut c_void,
        session_id: u64,
        tokens_needed: u32,
        block_table_device_out: *mut *mut u32,
        blocks_out: *mut u32,
    ) -> u32 {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(shared) = (unsafe { pool(pool_context) }) else {
                return 0;
            };
            match shared.reserve(session_id, 0, tokens_needed, true) {
                Ok(logical) => unsafe {
                    write_reservation_outputs(shared, logical, block_table_device_out, blocks_out)
                },
                Err(error) => {
                    log::warn!("llama.cpp shared KV reservation rejected: {error}");
                    0
                }
            }
        }))
        .unwrap_or(0)
    }

    unsafe extern "C" fn reserve_sequence(
        pool_context: *mut c_void,
        sequence_id: u64,
        tokens_needed: u32,
        block_table_device_out: *mut *mut u32,
        blocks_out: *mut u32,
    ) -> u32 {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(shared) = (unsafe { pool(pool_context) }) else {
                return 0;
            };
            let slot = match usize::try_from(sequence_id) {
                Ok(slot) => slot,
                Err(_) => return 0,
            };
            match shared.reserve(sequence_id, slot, tokens_needed, false) {
                Ok(logical) => unsafe {
                    write_reservation_outputs(shared, logical, block_table_device_out, blocks_out)
                },
                Err(error) => {
                    log::warn!("llama.cpp shared KV sequence reservation rejected: {error}");
                    0
                }
            }
        }))
        .unwrap_or(0)
    }

    unsafe extern "C" fn commit_sequences(
        pool_context: *mut c_void,
        block_table_device_out: *mut *mut u32,
    ) -> u32 {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(shared) = (unsafe { pool(pool_context) }) else {
                return 0;
            };
            if block_table_device_out.is_null() {
                return 0;
            }
            if let Err(error) = shared.commit_sequences() {
                log::error!("commit llama.cpp shared KV sequences: {error}");
                return 0;
            }
            unsafe { *block_table_device_out = shared.table_ptr() };
            1
        }))
        .unwrap_or(0)
    }

    unsafe extern "C" fn release(pool_context: *mut c_void, sequence_id: u64) {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            if let Some(shared) = unsafe { pool(pool_context) } {
                shared.release(sequence_id);
            }
        }));
    }

    unsafe extern "C" fn touch(pool_context: *mut c_void, sequence_id: u64) -> u32 {
        catch_unwind(AssertUnwindSafe(|| {
            unsafe { pool(pool_context) }
                .map(|shared| {
                    u32::from(
                        shared
                            .book
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .contains(sequence_id),
                    )
                })
                .unwrap_or(0)
        }))
        .unwrap_or(0)
    }
}

#[cfg(feature = "gpu-device-pool")]
pub(crate) use cuda::LlamaCppSharedPoolHost;

// Kept in this module so both the basic and shared callback tables use exactly
// the same log routing without making the CUDA host type visible elsewhere.
pub(crate) unsafe extern "C" fn host_log_bridge(
    _user_data: *mut c_void,
    level: u32,
    message: KapslSlice,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let Some(bytes) = (unsafe { message.as_bytes() }) else {
            return;
        };
        let message = String::from_utf8_lossy(bytes);
        match level {
            KAPSL_LOG_ERROR => log::error!("[llama-pack] {message}"),
            KAPSL_LOG_WARN => log::warn!("[llama-pack] {message}"),
            KAPSL_LOG_DEBUG => log::debug!("[llama-pack] {message}"),
            KAPSL_LOG_TRACE => log::trace!("[llama-pack] {message}"),
            _ => log::info!("[llama-pack] {message}"),
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_grow_without_replacing_live_blocks() {
        let mut book = ReservationBook::new(2, 4, 2);
        assert_eq!(book.growth_blocks(0, 1).unwrap(), 2);
        book.commit_growth(0, 0, 1, vec![10, 20]).unwrap();
        assert_eq!(book.growth_blocks(0, 3).unwrap(), 4);
        book.commit_growth(0, 0, 3, vec![11, 12, 21, 22]).unwrap();

        assert_eq!(&book.combined_host[0..3], &[10, 11, 12]);
        assert_eq!(&book.combined_host[4..7], &[20, 21, 22]);
        assert!(book.growth_blocks(0, 2).is_err());
    }

    #[test]
    fn sequence_slots_are_disjoint_and_release_clears_one_slot() {
        let mut book = ReservationBook::new(2, 3, 2);
        book.commit_growth(0, 0, 1, vec![1, 2]).unwrap();
        book.commit_growth(1, 1, 2, vec![3, 4, 5, 6]).unwrap();
        assert_eq!(&book.combined_host[6..8], &[3, 4]);
        assert_eq!(&book.combined_host[9..11], &[5, 6]);

        let released = book.release(0);
        assert_eq!(released, vec![1, 2]);
        assert_eq!(&book.combined_host[0..6], &[0; 6]);
        assert_eq!(&book.combined_host[6..8], &[3, 4]);
    }

    #[test]
    fn failed_or_cross_owned_growth_cannot_mutate_the_table() {
        let mut book = ReservationBook::new(1, 2, 2);
        book.commit_growth(7, 0, 1, vec![42]).unwrap();
        let before = book.combined_host.clone();
        assert!(book.commit_growth(8, 0, 1, vec![43]).is_err());
        assert!(book.commit_growth(7, 0, 2, Vec::new()).is_err());
        assert!(book.commit_growth(7, 1, 1, Vec::new()).is_err());
        assert_eq!(book.combined_host, before);
    }

    #[test]
    fn failed_upload_growth_can_be_rolled_back_without_losing_live_blocks() {
        let mut book = ReservationBook::new(2, 4, 1);
        book.commit_growth(5, 0, 1, vec![10, 20]).unwrap();
        book.commit_growth(5, 0, 3, vec![11, 12, 21, 22]).unwrap();

        assert_eq!(book.rollback_growth(5, 0, 1), vec![11, 12, 21, 22]);
        assert_eq!(book.logical_blocks(5), 1);
        assert_eq!(&book.combined_host[0..4], &[10, 0, 0, 0]);
        assert_eq!(&book.combined_host[4..8], &[20, 0, 0, 0]);
    }
}
