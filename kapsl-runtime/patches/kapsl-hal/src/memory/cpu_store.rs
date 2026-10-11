//! CPU-side KV block storage used as the eviction tier below GPU pools.
//!
//! Blocks evicted from a `GpuBlockPool` under pressure land here. When the
//! owning session is scheduled again, blocks are uploaded back to GPU.
//!
//! The backing allocation is plain heap memory today. In production this should
//! become CUDA pinned memory (`cudaMallocHost`) so async GPU↔CPU DMA can
//! overlap with compute. That change is transparent to callers — only the
//! constructor needs to change.

use half::f16;

#[derive(Debug, thiserror::Error)]
pub enum CpuStoreError {
    #[error("CPU block store is full ({capacity} blocks allocated)")]
    Full { capacity: usize },
    #[error("Slot {0} is out of range")]
    OutOfRange(u32),
    #[error("Data length {got} does not match expected {expected} elements per block")]
    SizeMismatch { got: usize, expected: usize },
}

/// CPU-side flat block storage.
///
/// Each slot holds one KV block: `2 * num_kv_heads * block_size * head_dim` f16
/// elements (K half then V half, matching `GpuBlockPool`'s layout).
pub struct CpuBlockStore {
    data: Vec<f16>,
    elems_per_block: usize,
    capacity: usize,
    free_slots: Vec<u32>,
}

impl CpuBlockStore {
    /// Allocate storage for `capacity` blocks with the given KV geometry.
    pub fn new(capacity: usize, num_kv_heads: usize, block_size: usize, head_dim: usize) -> Self {
        let elems_per_block = 2 * num_kv_heads * block_size * head_dim;
        Self {
            data: vec![f16::ZERO; capacity * elems_per_block],
            elems_per_block,
            capacity,
            free_slots: (0..capacity as u32).rev().collect(),
        }
    }

    /// Number of f16 elements per block.
    pub fn elems_per_block(&self) -> usize {
        self.elems_per_block
    }

    /// Total block capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of free (unoccupied) slots.
    pub fn free_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Number of occupied slots.
    pub fn used_count(&self) -> usize {
        self.capacity - self.free_slots.len()
    }

    /// Write `data` into a free slot and return its slot index.
    ///
    /// `data` must have exactly `elems_per_block()` elements.
    pub fn store_block(&mut self, data: &[f16]) -> Result<u32, CpuStoreError> {
        if data.len() != self.elems_per_block {
            return Err(CpuStoreError::SizeMismatch {
                got: data.len(),
                expected: self.elems_per_block,
            });
        }
        let slot = self.free_slots.pop().ok_or(CpuStoreError::Full {
            capacity: self.capacity,
        })?;
        let off = slot as usize * self.elems_per_block;
        self.data[off..off + self.elems_per_block].copy_from_slice(data);
        Ok(slot)
    }

    /// Read the block at `slot`.
    pub fn load_block(&self, slot: u32) -> Result<&[f16], CpuStoreError> {
        if slot as usize >= self.capacity {
            return Err(CpuStoreError::OutOfRange(slot));
        }
        let off = slot as usize * self.elems_per_block;
        Ok(&self.data[off..off + self.elems_per_block])
    }

    /// Return `slot` to the free list.
    pub fn free_slot(&mut self, slot: u32) {
        debug_assert!((slot as usize) < self.capacity, "slot out of range");
        self.free_slots.push(slot);
    }

    /// Release all slots belonging to a session (bulk free).
    pub fn free_slots_bulk(&mut self, slots: &[u32]) {
        self.free_slots.extend_from_slice(slots);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn make_store(
        capacity: usize,
        kv_heads: usize,
        block_size: usize,
        head_dim: usize,
    ) -> CpuBlockStore {
        CpuBlockStore::new(capacity, kv_heads, block_size, head_dim)
    }

    fn block_data(val: f32, elems: usize) -> Vec<f16> {
        vec![f16::from_f32(val); elems]
    }

    #[test]
    fn new_store_has_correct_capacity() {
        let store = make_store(7, 2, 16, 64);
        assert_eq!(store.capacity(), 7);
    }

    #[test]
    fn new_store_all_slots_free() {
        let store = make_store(4, 2, 16, 64);
        assert_eq!(store.free_count(), 4);
        assert_eq!(store.used_count(), 0);
    }

    #[test]
    fn store_and_load_roundtrip() {
        let mut store = make_store(4, 2, 8, 32);
        let elems = store.elems_per_block();
        let data: Vec<f16> = (0..elems).map(|i| f16::from_f32(i as f32)).collect();
        let slot = store.store_block(&data).unwrap();
        assert_eq!(store.load_block(slot).unwrap(), data.as_slice());
    }

    #[test]
    fn store_decrements_free_count() {
        let mut store = make_store(3, 1, 4, 8);
        let elems = store.elems_per_block();
        let data = block_data(1.0, elems);
        assert_eq!(store.free_count(), 3);
        store.store_block(&data).unwrap();
        assert_eq!(store.free_count(), 2);
        store.store_block(&data).unwrap();
        assert_eq!(store.free_count(), 1);
        store.store_block(&data).unwrap();
        assert_eq!(store.free_count(), 0);
        assert_eq!(store.used_count(), 3);
    }

    #[test]
    fn full_store_returns_error() {
        let mut store = make_store(1, 1, 4, 8);
        let elems = store.elems_per_block();
        let data = block_data(0.0, elems);
        store.store_block(&data).unwrap();
        let err = store.store_block(&data).unwrap_err();
        assert!(matches!(err, CpuStoreError::Full { capacity: 1 }));
    }

    #[test]
    fn size_mismatch_returns_error() {
        let mut store = make_store(4, 2, 8, 32);
        let wrong = vec![f16::ZERO; 5]; // not elems_per_block
        let err = store.store_block(&wrong).unwrap_err();
        assert!(matches!(err, CpuStoreError::SizeMismatch { .. }));
    }

    #[test]
    fn load_out_of_range_returns_error() {
        let store = make_store(4, 2, 8, 32);
        let err = store.load_block(99).unwrap_err();
        assert!(matches!(err, CpuStoreError::OutOfRange(99)));
    }

    #[test]
    fn free_slot_increments_free_count() {
        let mut store = make_store(3, 1, 4, 8);
        let elems = store.elems_per_block();
        let data = block_data(1.0, elems);
        let slot = store.store_block(&data).unwrap();
        assert_eq!(store.free_count(), 2);
        store.free_slot(slot);
        assert_eq!(store.free_count(), 3);
    }

    #[test]
    fn free_slots_bulk_restores_multiple() {
        let mut store = make_store(4, 1, 4, 8);
        let elems = store.elems_per_block();
        let data = block_data(0.0, elems);
        let s0 = store.store_block(&data).unwrap();
        let s1 = store.store_block(&data).unwrap();
        let s2 = store.store_block(&data).unwrap();
        assert_eq!(store.free_count(), 1);
        store.free_slots_bulk(&[s0, s1, s2]);
        assert_eq!(store.free_count(), 4);
    }

    #[test]
    fn multiple_slots_store_independently() {
        let mut store = make_store(3, 1, 4, 8);
        let elems = store.elems_per_block();
        let data_a: Vec<f16> = (0..elems).map(|i| f16::from_f32(i as f32)).collect();
        let data_b: Vec<f16> = (0..elems).map(|i| f16::from_f32(i as f32 * 2.0)).collect();
        let slot_a = store.store_block(&data_a).unwrap();
        let slot_b = store.store_block(&data_b).unwrap();
        assert_ne!(slot_a, slot_b);
        assert_eq!(store.load_block(slot_a).unwrap(), data_a.as_slice());
        assert_eq!(store.load_block(slot_b).unwrap(), data_b.as_slice());
    }

    #[test]
    fn reused_slot_reflects_new_data() {
        let mut store = make_store(1, 1, 4, 8);
        let elems = store.elems_per_block();
        let data_a = block_data(1.0, elems);
        let data_b = block_data(2.0, elems);
        let slot = store.store_block(&data_a).unwrap();
        store.free_slot(slot);
        let slot2 = store.store_block(&data_b).unwrap();
        assert_eq!(store.load_block(slot2).unwrap(), data_b.as_slice());
    }

    #[test]
    fn elems_per_block_matches_formula() {
        // 2 * kv_heads * block_size * head_dim
        let store = CpuBlockStore::new(1, 3, 16, 128);
        assert_eq!(store.elems_per_block(), 2 * 3 * 16 * 128);
    }
}
