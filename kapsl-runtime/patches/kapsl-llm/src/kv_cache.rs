use crate::radix_tree::RadixTree;
use half::f16;
use ndarray::Array4;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvCacheMode {
    #[default]
    Dense,
    Paged,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum KvEvictionPolicy {
    #[default]
    None,
    LruInactive,
    Fifo,
}

#[derive(Clone, Debug)]
pub struct KvCacheConfig {
    pub mode: KvCacheMode,
    pub block_size: usize,
    pub total_blocks: usize,
    pub eviction_policy: KvEvictionPolicy,
    /// Max number of completed SequenceKvCache objects to hold in the dense
    /// cache free-list for reuse. Capping this prevents unbounded RSS growth
    /// when many sequences finish without new ones arriving.
    pub dense_free_list_cap: usize,
    /// Initial per-sequence KvTensor allocation in tokens. Sequences grow
    /// dynamically up to max_seq_len. Tuned per-model via metadata.json
    /// `kv_cache.initial_seq_len`. Default: 256.
    pub initial_seq_len: usize,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            mode: KvCacheMode::Dense,
            block_size: 16,
            total_blocks: 2048,
            eviction_policy: KvEvictionPolicy::LruInactive,
            dense_free_list_cap: 32,
            initial_seq_len: 256,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KvCacheStats {
    pub mode: KvCacheMode,
    pub bytes_used: usize,
    pub bytes_capacity: usize,
    pub blocks_total: usize,
    pub blocks_free: usize,
    pub sequences: usize,
    pub evicted_blocks: u64,
    pub evicted_sequences: u64,
    pub packed_layers: usize,
    /// Blocks currently held in the CPU offload store (paged mode only).
    pub cpu_offloaded_blocks: u64,
}

#[derive(Debug)]
pub enum KvCacheError {
    OutOfBlocks,
}

impl std::fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCacheError::OutOfBlocks => write!(f, "KV cache out of blocks"),
        }
    }
}

impl std::error::Error for KvCacheError {}

pub struct KvView {
    pub key: Arc<[f16]>,
    pub value: Arc<[f16]>,
    pub length: usize,
}

pub struct PackedKvView {
    pub key: Arc<[f16]>,
    pub value: Arc<[f16]>,
    pub length: usize,
}

#[derive(Debug)]
pub struct KvTensor {
    data: Vec<f16>,
    num_heads: usize,
    /// The currently allocated sequence length (may be less than max_seq_len).
    allocated_seq_len: usize,
    /// The absolute maximum sequence length for this tensor.
    max_seq_len: usize,
    head_dim: usize,
}

impl KvTensor {
    /// Create a new KvTensor.
    ///
    /// `initial_seq_len` controls how many positions are allocated up front.
    /// Pass `KvCacheConfig::initial_seq_len` (or 256 as a safe default).
    /// The tensor grows on demand up to `max_seq_len`.
    pub fn new(
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        initial_seq_len: usize,
    ) -> Self {
        let initial_cap = initial_seq_len.clamp(1, max_seq_len.max(1));
        let size = num_heads * initial_cap * head_dim;
        Self {
            data: vec![f16::ZERO; size],
            num_heads,
            allocated_seq_len: initial_cap,
            max_seq_len,
            head_dim,
        }
    }

    /// Grow the backing buffer so that `allocated_seq_len` is at least `required`.
    /// Existing data is preserved via `Vec::resize`. The layout is
    /// `[num_heads, allocated_seq_len, head_dim]` so growing the seq dimension
    /// means we must re-layout the existing heads.
    fn grow_to(&mut self, required: usize) {
        if required <= self.allocated_seq_len {
            return;
        }
        let new_cap = required.next_power_of_two().min(self.max_seq_len);
        let old_seq = self.allocated_seq_len;
        let new_size = self.num_heads * new_cap * self.head_dim;
        let mut new_data = vec![f16::ZERO; new_size];
        // Copy each head's existing rows into the new (wider) layout.
        let hd = self.head_dim;
        for h in 0..self.num_heads {
            let old_base = h * old_seq * hd;
            let new_base = h * new_cap * hd;
            let copy_bytes = old_seq * hd;
            new_data[new_base..new_base + copy_bytes]
                .copy_from_slice(&self.data[old_base..old_base + copy_bytes]);
        }
        self.data = new_data;
        self.allocated_seq_len = new_cap;
    }

    #[inline(always)]
    fn index(&self, head: usize, pos: usize, dim: usize) -> usize {
        debug_assert!(head < self.num_heads);
        debug_assert!(pos < self.allocated_seq_len);
        debug_assert!(dim < self.head_dim);

        (head * self.allocated_seq_len * self.head_dim) + (pos * self.head_dim) + dim
    }

    #[inline(always)]
    pub fn write_head(&mut self, head: usize, pos: usize, values: &[f16]) {
        self.grow_to(pos + 1);
        let base = self.index(head, pos, 0);
        self.data[base..base + self.head_dim].copy_from_slice(values);
    }

    #[inline(always)]
    pub fn write_head_range(&mut self, head: usize, pos_start: usize, values: &[f16]) {
        debug_assert!(values.len().is_multiple_of(self.head_dim));
        let num_positions = values.len() / self.head_dim;
        self.grow_to(pos_start + num_positions);
        debug_assert!(pos_start + num_positions <= self.allocated_seq_len);
        let base = self.index(head, pos_start, 0);
        let end = base + values.len();
        self.data[base..end].copy_from_slice(values);
    }

    pub fn as_slice(&self) -> &[f16] {
        &self.data
    }

    /// Return the currently allocated sequence length.
    pub fn allocated_seq_len(&self) -> usize {
        self.allocated_seq_len
    }
}

pub struct LayerKv {
    pub key: KvTensor,
    pub value: KvTensor,
}

#[derive(Clone)]
struct PackedLayerKv {
    key: Arc<[f16]>,
    value: Arc<[f16]>,
    length: usize,
}

impl LayerKv {
    pub fn new(
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        initial_seq_len: usize,
    ) -> Self {
        Self {
            key: KvTensor::new(num_heads, max_seq_len, head_dim, initial_seq_len),
            value: KvTensor::new(num_heads, max_seq_len, head_dim, initial_seq_len),
        }
    }
}

pub struct SequenceKvCache {
    pub layers: Vec<LayerKv>,
    pub current_len: usize,
    packed_layers: Vec<Option<PackedLayerKv>>,
}

impl SequenceKvCache {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        initial_seq_len: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKv::new(num_heads, max_seq_len, head_dim, initial_seq_len))
            .collect();

        let mut packed_layers = Vec::with_capacity(num_layers);
        packed_layers.resize_with(num_layers, || None);

        Self {
            layers,
            current_len: 0,
            packed_layers,
        }
    }

    /// The currently allocated sequence length across all layer tensors.
    /// All layer tensors share the same allocated length.
    pub fn allocated_seq_len(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.key.allocated_seq_len())
            .unwrap_or(0)
    }

    /// Ensure every layer's key/value tensor can hold at least `required` positions.
    pub fn ensure_capacity(&mut self, required: usize) {
        for layer in &mut self.layers {
            layer.key.grow_to(required);
            layer.value.grow_to(required);
        }
    }
}

pub struct DenseKvCache {
    sequences: HashMap<u64, SequenceKvCache>,
    free_list: Vec<SequenceKvCache>,
    free_list_cap: usize,
    scratch_key: Vec<f16>,
    scratch_value: Vec<f16>,
    scratch_key_compact: Vec<f16>,
    scratch_val_compact: Vec<f16>,
    max_seq_len: usize,
    initial_seq_len: usize,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
}

impl DenseKvCache {
    pub fn new(num_layers: usize, num_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        Self::new_with_config(num_layers, num_heads, max_seq_len, head_dim, 32, 256)
    }

    pub fn new_with_free_list_cap(
        num_layers: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        free_list_cap: usize,
    ) -> Self {
        Self::new_with_config(
            num_layers,
            num_heads,
            max_seq_len,
            head_dim,
            free_list_cap,
            256,
        )
    }

    pub fn new_with_config(
        num_layers: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        free_list_cap: usize,
        initial_seq_len: usize,
    ) -> Self {
        let expected = num_heads * head_dim;
        Self {
            sequences: HashMap::new(),
            free_list: Vec::new(),
            free_list_cap,
            scratch_key: vec![f16::ZERO; expected],
            scratch_value: vec![f16::ZERO; expected],
            scratch_key_compact: Vec::new(),
            scratch_val_compact: Vec::new(),
            num_layers,
            num_heads,
            max_seq_len,
            initial_seq_len,
            head_dim,
        }
    }

    pub fn allocate_sequence(
        &mut self,
        sequence_id: u64,
        _tokens: &[u32],
    ) -> Result<usize, KvCacheError> {
        if let Some(seq) = self.sequences.get(&sequence_id) {
            return Ok(seq.current_len.min(_tokens.len().saturating_sub(1)));
        }

        let mut seq = if let Some(seq) = self.free_list.pop() {
            seq
        } else {
            SequenceKvCache::new(
                self.num_layers,
                self.num_heads,
                self.max_seq_len,
                self.head_dim,
                self.initial_seq_len,
            )
        };
        seq.current_len = 0;
        if seq.packed_layers.len() != self.num_layers {
            seq.packed_layers.clear();
            seq.packed_layers.resize_with(self.num_layers, || None);
        } else {
            for slot in seq.packed_layers.iter_mut() {
                *slot = None;
            }
        }
        self.sequences.insert(sequence_id, seq);
        Ok(0)
    }

    pub fn append_token(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
        pos: usize,
        key: &[f16],
        value: &[f16],
        _token_id: Option<u64>,
    ) -> Result<(), KvCacheError> {
        let seq = self
            .sequences
            .get_mut(&sequence_id)
            .expect("sequence not allocated");

        assert!(pos < self.max_seq_len, "KV cache overflow");

        seq.ensure_capacity(pos + 1);
        let layer = &mut seq.layers[layer_index];

        let expected = self.num_heads * self.head_dim;
        if self.scratch_key.len() != expected {
            self.scratch_key.resize(expected, f16::ZERO);
        }
        if self.scratch_value.len() != expected {
            self.scratch_value.resize(expected, f16::ZERO);
        }

        let key_slice = if key.len() >= expected {
            &key[..expected]
        } else {
            self.scratch_key[..key.len()].copy_from_slice(key);
            self.scratch_key[key.len()..].fill(f16::ZERO);
            &self.scratch_key[..]
        };

        let value_slice = if value.len() >= expected {
            &value[..expected]
        } else {
            self.scratch_value[..value.len()].copy_from_slice(value);
            self.scratch_value[value.len()..].fill(f16::ZERO);
            &self.scratch_value[..]
        };

        for h in 0..self.num_heads {
            let offset = h * self.head_dim;
            let end = offset + self.head_dim;
            layer.key.write_head(h, pos, &key_slice[offset..end]);
            layer.value.write_head(h, pos, &value_slice[offset..end]);
        }
        Ok(())
    }

    pub fn set_packed_layer(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
        length: usize,
        key: &[f16],
        value: &[f16],
    ) {
        let seq = self
            .sequences
            .get_mut(&sequence_id)
            .expect("sequence not allocated");
        let expected = self.num_heads * self.head_dim * length;
        if key.len() != expected || value.len() != expected {
            log::warn!(
                "Packed KV length mismatch for seq {} layer {}: key={}, value={}, expected={}",
                sequence_id,
                layer_index,
                key.len(),
                value.len(),
                expected
            );
            return;
        }
        if layer_index >= seq.packed_layers.len() {
            return;
        }
        seq.packed_layers[layer_index] = Some(PackedLayerKv {
            key: Arc::from(key),
            value: Arc::from(value),
            length,
        });
    }

    pub fn clear_packed_layer(&mut self, sequence_id: u64, layer_index: usize) {
        if let Some(seq) = self.sequences.get_mut(&sequence_id) {
            if layer_index < seq.packed_layers.len() {
                seq.packed_layers[layer_index] = None;
            }
        }
    }

    pub fn get_packed_layer(&self, sequence_id: u64, layer_index: usize) -> Option<PackedKvView> {
        let seq = self.sequences.get(&sequence_id)?;
        let packed = seq.packed_layers.get(layer_index)?.as_ref()?;
        Some(PackedKvView {
            key: packed.key.clone(),
            value: packed.value.clone(),
            length: packed.length,
        })
    }

    pub fn append_head_range_seq_first(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
        head: usize,
        pos_start: usize,
        key_values: &[f16],
        value_values: &[f16],
    ) -> Result<(), KvCacheError> {
        let seq = self
            .sequences
            .get_mut(&sequence_id)
            .expect("sequence not allocated");

        let layer = &mut seq.layers[layer_index];
        layer.key.write_head_range(head, pos_start, key_values);
        layer.value.write_head_range(head, pos_start, value_values);
        Ok(())
    }

    pub fn advance_sequence(&mut self, sequence_id: u64) {
        let seq = self.sequences.get_mut(&sequence_id).unwrap();
        seq.current_len += 1;
    }

    pub fn advance_sequence_by(&mut self, sequence_id: u64, delta: usize) {
        let seq = self.sequences.get_mut(&sequence_id).unwrap();
        seq.current_len += delta;
    }

    pub fn rollback_sequence(&mut self, sequence_id: u64, length: usize) {
        if let Some(seq) = self.sequences.get_mut(&sequence_id) {
            if length < seq.current_len {
                seq.current_len = length;
                // Dense cache reuses memory, no need to free blocks

                // Invalidate packed layers
                for slot in seq.packed_layers.iter_mut() {
                    *slot = None;
                }
            }
        }
    }

    pub fn overwrite_layer(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
        key: &KvTensor,
        value: &KvTensor,
        length: usize,
    ) {
        let seq = self.sequences.get_mut(&sequence_id).unwrap();
        let layer = &mut seq.layers[layer_index];

        layer.key.data.copy_from_slice(&key.data);
        layer.value.data.copy_from_slice(&value.data);
        seq.current_len = length;
    }

    pub fn remove_sequence(&mut self, sequence_id: u64) {
        if let Some(seq) = self.sequences.remove(&sequence_id) {
            if self.free_list.len() < self.free_list_cap {
                self.free_list.push(seq);
            }
            // else: drop the allocation immediately
        }
    }

    pub fn has_sequence(&self, sequence_id: u64) -> bool {
        self.sequences.contains_key(&sequence_id)
    }

    pub fn sequence_length(&self, sequence_id: u64) -> Option<usize> {
        self.sequences.get(&sequence_id).map(|s| s.current_len)
    }

    pub fn get_layer_view(&mut self, sequence_id: u64, layer_index: usize) -> Option<KvView> {
        let seq = self.sequences.get(&sequence_id)?;
        let length = seq.current_len;
        let layer = &seq.layers[layer_index];

        if length == 0 {
            return Some(KvView {
                key: Arc::from([] as [f16; 0]),
                value: Arc::from([] as [f16; 0]),
                length: 0,
            });
        }

        // Build compact [num_heads, length, head_dim] instead of copying the full
        // [num_heads, allocated_seq_len, head_dim] buffer. Each head's used positions are
        // contiguous (layout: head * allocated_seq_len * head_dim + pos * head_dim + dim).
        // NOTE: must use allocated_seq_len (not max_seq_len) because KvTensor grows lazily.
        let alloc_seq_len = layer.key.allocated_seq_len();
        let stride = alloc_seq_len * self.head_dim;
        let compact_per_head = length * self.head_dim;
        let compact_len = self.num_heads * compact_per_head;

        // Reuse workspace buffers instead of allocating new Vecs each call.
        self.scratch_key_compact.clear();
        self.scratch_key_compact.reserve(compact_len);
        self.scratch_val_compact.clear();
        self.scratch_val_compact.reserve(compact_len);

        let key_slice = layer.key.as_slice();
        let val_slice = layer.value.as_slice();
        for h in 0..self.num_heads {
            let src = h * stride;
            self.scratch_key_compact
                .extend_from_slice(&key_slice[src..src + compact_per_head]);
            self.scratch_val_compact
                .extend_from_slice(&val_slice[src..src + compact_per_head]);
        }

        Some(KvView {
            key: Arc::from(self.scratch_key_compact.as_slice()),
            value: Arc::from(self.scratch_val_compact.as_slice()),
            length,
        })
    }

    pub fn get_layer_as_onnx(&mut self, sequence_id: u64, layer: usize) -> Option<Array4<f16>> {
        let view = self.get_layer_view(sequence_id, layer)?;
        let seq_len = view.length;

        let stride = self.num_heads * self.head_dim;
        if stride == 0 || view.key.len() % stride != 0 {
            return None;
        }
        let max_seq_len = view.key.len() / stride;
        if seq_len > max_seq_len {
            return None;
        }

        let total = self.num_heads * seq_len * self.head_dim;
        let mut packed = vec![f16::ZERO; total];
        for h in 0..self.num_heads {
            let head_offset = h * max_seq_len * self.head_dim;
            let packed_offset = h * seq_len * self.head_dim;
            for pos in 0..seq_len {
                let src = head_offset + pos * self.head_dim;
                let dst = packed_offset + pos * self.head_dim;
                let src_end = src + self.head_dim;
                let dst_end = dst + self.head_dim;
                if src_end <= view.key.len() {
                    packed[dst..dst_end].copy_from_slice(&view.key[src..src_end]);
                }
            }
        }

        Array4::from_shape_vec((1, self.num_heads, seq_len, self.head_dim), packed).ok()
    }

    pub fn stats(&self) -> KvCacheStats {
        let per_seq_bytes =
            self.num_layers * self.num_heads * self.max_seq_len * self.head_dim * 2 * 2;
        let allocated_sequences = self.sequences.len() + self.free_list.len();
        let bytes_capacity = allocated_sequences * per_seq_bytes;
        let bytes_used = self.sequences.len() * per_seq_bytes;
        let packed_layers = self
            .sequences
            .values()
            .map(|seq| seq.packed_layers.iter().filter(|v| v.is_some()).count())
            .sum();
        KvCacheStats {
            mode: KvCacheMode::Dense,
            bytes_used,
            bytes_capacity,
            blocks_total: 0,
            blocks_free: 0,
            sequences: self.sequences.len(),
            evicted_blocks: 0,
            evicted_sequences: 0,
            packed_layers,
            cpu_offloaded_blocks: 0,
        }
    }
}

struct PagedBlockAllocator {
    free_blocks: VecDeque<usize>,
    ref_counts: Vec<u32>,
    total_blocks: usize,
}

impl PagedBlockAllocator {
    fn new(total_blocks: usize) -> Self {
        let free_blocks: VecDeque<usize> = (0..total_blocks).collect();
        let ref_counts = vec![0; total_blocks];
        Self {
            free_blocks,
            ref_counts,
            total_blocks,
        }
    }

    fn allocate(&mut self) -> Option<usize> {
        if let Some(block) = self.free_blocks.pop_front() {
            self.ref_counts[block] = 1;
            Some(block)
        } else {
            None
        }
    }

    fn free(&mut self, block: usize) {
        if block >= self.total_blocks {
            return;
        }
        if self.ref_counts[block] == 0 {
            return;
        }
        self.ref_counts[block] -= 1;
        if self.ref_counts[block] == 0 {
            self.free_blocks.push_front(block);
        }
    }

    fn add_ref(&mut self, block: usize) {
        assert!(block < self.total_blocks);
        assert!(self.ref_counts[block] > 0, "Cannot add ref to free block");
        self.ref_counts[block] += 1;
    }

    fn get_ref_count(&self, block: usize) -> u32 {
        if block < self.total_blocks {
            self.ref_counts[block]
        } else {
            0
        }
    }

    fn free_count(&self) -> usize {
        self.free_blocks.len()
    }
}

struct PagedSequence {
    blocks: VecDeque<usize>,
    current_len: usize,
    packed_layers: Vec<Option<PackedLayerKv>>,
    last_access: u64,
    tokens: Vec<u64>,
    /// Eviction priority mirror from the owning `SequenceGroup`.
    /// Lower value → evicted first when blocks are scarce.
    priority: u8,
}

impl PagedSequence {
    fn new(num_layers: usize) -> Self {
        let mut packed_layers = Vec::with_capacity(num_layers);
        packed_layers.resize_with(num_layers, || None);
        Self {
            blocks: VecDeque::new(),
            current_len: 0,
            packed_layers,
            last_access: 0,
            tokens: Vec::new(),
            priority: 0,
        }
    }
}

/// Snapshot of one offloaded sequence kept in CPU memory.
struct OffloadedSequence {
    /// KV block data in block-index order.
    /// Each entry is `(key_data, value_data)` with `block_stride` f16 elements.
    blocks: Vec<(Vec<f16>, Vec<f16>)>,
    /// Number of valid tokens when the sequence was offloaded.
    current_len: usize,
    /// Priority inherited from the originating `SequenceGroup`.
    priority: u8,
}

/// CPU-side store for sequences whose KV blocks have been offloaded from GPU.
///
/// When the GPU block pool is exhausted and inactive-sequence eviction cannot
/// free space, the lowest-priority inactive sequence's KV data is copied here
/// instead of being permanently discarded.  `restore_offloaded_sequence_inner`
/// allocates fresh GPU blocks and copies the data back, allowing the sequence
/// to continue without recomputing the prefix.
pub struct CpuKvBlockStore {
    /// Maps `sequence_id` → stored block data + metadata.
    sequences: HashMap<u64, OffloadedSequence>,
    /// Elements per block: `num_layers * num_heads * block_size * head_dim`.
    block_stride: usize,
    /// Running total of blocks across all stored sequences.
    total_blocks: usize,
}

impl CpuKvBlockStore {
    fn new(block_stride: usize) -> Self {
        Self {
            sequences: HashMap::new(),
            block_stride,
            total_blocks: 0,
        }
    }

    /// Copy all blocks for a sequence from the GPU storage slices into CPU
    /// memory, preserving their order so they can be restored later.
    fn offload_sequence(
        &mut self,
        sequence_id: u64,
        block_ids: &[usize],
        current_len: usize,
        priority: u8,
        key_storage: &[f16],
        value_storage: &[f16],
    ) {
        let mut block_data = Vec::with_capacity(block_ids.len());
        for &block_id in block_ids {
            let start = block_id * self.block_stride;
            let end = start + self.block_stride;
            if end <= key_storage.len() {
                block_data.push((
                    key_storage[start..end].to_vec(),
                    value_storage[start..end].to_vec(),
                ));
            }
        }
        self.total_blocks += block_data.len();
        self.sequences.insert(
            sequence_id,
            OffloadedSequence {
                blocks: block_data,
                current_len,
                priority,
            },
        );
    }

    fn has_sequence(&self, sequence_id: u64) -> bool {
        self.sequences.contains_key(&sequence_id)
    }

    /// Remove and return the stored data for `sequence_id`, if present.
    fn take_sequence(&mut self, sequence_id: u64) -> Option<OffloadedSequence> {
        if let Some(entry) = self.sequences.remove(&sequence_id) {
            self.total_blocks = self.total_blocks.saturating_sub(entry.blocks.len());
            Some(entry)
        } else {
            None
        }
    }

    /// Put back a sequence that could not be restored (not enough GPU blocks).
    fn return_sequence(&mut self, sequence_id: u64, entry: OffloadedSequence) {
        self.total_blocks += entry.blocks.len();
        self.sequences.insert(sequence_id, entry);
    }

    /// Total number of GPU blocks whose data is currently held in CPU memory.
    pub fn offloaded_block_count(&self) -> usize {
        self.total_blocks
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

pub struct PagedKvCache {
    sequences: HashMap<u64, PagedSequence>,
    allocator: PagedBlockAllocator,
    key_storage: Vec<f16>,
    value_storage: Vec<f16>,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    eviction_policy: KvEvictionPolicy,
    access_counter: u64,
    active_sequences: HashSet<u64>,
    evicted_sequences: Vec<u64>,
    evicted_blocks: u64,
    evicted_sequences_count: u64,
    scratch_key: Vec<f16>,
    scratch_value: Vec<f16>,
    radix_tree: RadixTree,
    /// CPU offload store: blocks moved here instead of being permanently freed
    /// when the GPU pool is exhausted after LRU eviction.
    cpu_store: CpuKvBlockStore,
    /// Running count of blocks currently sitting in `cpu_store`.
    cpu_offloaded_blocks: u64,
}

impl PagedKvCache {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        block_size: usize,
        total_blocks: usize,
        head_dim: usize,
        eviction_policy: KvEvictionPolicy,
    ) -> Self {
        let block_stride = num_layers * num_heads * block_size * head_dim;
        let expected = num_heads * head_dim;
        Self {
            sequences: HashMap::new(),
            allocator: PagedBlockAllocator::new(total_blocks),
            key_storage: Vec::new(),
            value_storage: Vec::new(),
            num_layers,
            num_heads,
            head_dim,
            block_size,
            eviction_policy,
            access_counter: 0,
            active_sequences: HashSet::new(),
            evicted_sequences: Vec::new(),
            evicted_blocks: 0,
            evicted_sequences_count: 0,
            scratch_key: vec![f16::ZERO; expected],
            scratch_value: vec![f16::ZERO; expected],
            radix_tree: RadixTree::new(),
            cpu_store: CpuKvBlockStore::new(block_stride),
            cpu_offloaded_blocks: 0,
        }
    }

    fn block_stride(&self) -> usize {
        self.num_layers * self.num_heads * self.block_size * self.head_dim
    }

    fn layer_stride(&self) -> usize {
        self.num_heads * self.block_size * self.head_dim
    }

    fn head_stride(&self) -> usize {
        self.block_size * self.head_dim
    }

    fn bump_access(&mut self) -> u64 {
        self.access_counter = self.access_counter.wrapping_add(1);
        self.access_counter
    }

    fn allocate_block(&mut self, protected_seq_id: Option<u64>) -> Result<usize, KvCacheError> {
        if let Some(block) = self.allocator.allocate() {
            return Ok(block);
        }

        // First attempt: evict an inactive sequence from GPU memory.
        if self.eviction_policy != KvEvictionPolicy::None
            && self.evict_one_inactive(protected_seq_id)
        {
            if let Some(block) = self.allocator.allocate() {
                return Ok(block);
            }
        }

        // Second attempt: offload an inactive sequence's blocks to CPU instead
        // of discarding them, preserving the ability to restore later.
        if self.try_offload_to_cpu(protected_seq_id) {
            if let Some(block) = self.allocator.allocate() {
                return Ok(block);
            }
        }

        Err(KvCacheError::OutOfBlocks)
    }

    /// Set the eviction priority for an already-allocated sequence.
    ///
    /// Call this after `allocate_sequence` when a request priority is known
    /// (e.g. propagated from `SequenceGroup::priority`). Lower values are
    /// evicted first under memory pressure.
    pub fn set_sequence_priority(&mut self, sequence_id: u64, priority: u8) {
        if let Some(seq) = self.sequences.get_mut(&sequence_id) {
            seq.priority = priority;
        }
    }

    /// Attempt to offload one inactive sequence's blocks to the CPU store,
    /// freeing GPU blocks without permanently losing the KV data.
    ///
    /// Chooses the victim using the same priority-composite key as
    /// `evict_one_inactive`. Returns `true` if at least one GPU block was freed.
    fn try_offload_to_cpu(&mut self, protected_seq_id: Option<u64>) -> bool {
        if self.eviction_policy == KvEvictionPolicy::None {
            return false;
        }

        // Find a victim using the same priority-composite key as evict_one_inactive.
        let mut victim: Option<u64> = None;
        let mut best_key: (u8, u64) = (0, u64::MAX);

        for (seq_id, seq) in &self.sequences {
            if self.active_sequences.contains(seq_id) {
                continue;
            }
            if Some(*seq_id) == protected_seq_id {
                continue;
            }
            let time_val = match self.eviction_policy {
                KvEvictionPolicy::LruInactive => seq.last_access,
                KvEvictionPolicy::Fifo => *seq_id,
                KvEvictionPolicy::None => continue,
            };
            let priority_bucket = u8::MAX - seq.priority;
            let key = (priority_bucket, time_val);
            let is_better = match victim {
                None => true,
                Some(_) => key.0 > best_key.0 || (key.0 == best_key.0 && key.1 < best_key.1),
            };
            if is_better {
                best_key = key;
                victim = Some(*seq_id);
            }
        }

        let victim_id = match victim {
            Some(id) => id,
            None => return false,
        };

        // Capture block IDs and sequence metadata before mutably borrowing storage.
        let (block_ids, current_len, priority): (Vec<usize>, usize, u8) =
            match self.sequences.get(&victim_id) {
                Some(seq) => (
                    seq.blocks.iter().copied().collect(),
                    seq.current_len,
                    seq.priority,
                ),
                None => return false,
            };

        if block_ids.is_empty() {
            return false;
        }

        // Copy KV data to CPU store, then free GPU blocks.
        self.cpu_store.offload_sequence(
            victim_id,
            &block_ids,
            current_len,
            priority,
            &self.key_storage,
            &self.value_storage,
        );
        for &block_id in &block_ids {
            self.allocator.free(block_id);
        }
        self.cpu_offloaded_blocks += block_ids.len() as u64;

        // Remove the sequence so the scheduler sees it as gone.
        self.sequences.remove(&victim_id);
        self.evicted_sequences.push(victim_id);
        self.evicted_sequences_count += 1;

        true
    }

    /// Attempt to restore an offloaded sequence from CPU memory back onto GPU.
    ///
    /// Allocates fresh GPU blocks (which may have different indices than the
    /// originals), copies the stored KV data into them, and re-inserts the
    /// sequence into the live map.  Returns the restored `current_len` on
    /// success, or `None` if the sequence was not found in the CPU store or
    /// there were not enough free GPU blocks to accommodate it.
    fn restore_offloaded_sequence_inner(&mut self, sequence_id: u64) -> Option<usize> {
        // Move the stored entry out so we can freely borrow the rest of self.
        let entry = self.cpu_store.take_sequence(sequence_id)?;
        let num_blocks = entry.blocks.len();

        if num_blocks == 0 {
            // Empty sequence — recreate with no blocks.
            let mut seq = PagedSequence::new(self.num_layers);
            seq.priority = entry.priority;
            seq.last_access = self.access_counter;
            self.sequences.insert(sequence_id, seq);
            self.cpu_offloaded_blocks = self.cpu_offloaded_blocks.saturating_sub(0);
            return Some(0);
        }

        // Try to allocate enough fresh GPU blocks.
        let mut new_block_ids = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            match self.allocator.allocate() {
                Some(id) => new_block_ids.push(id),
                None => {
                    // Not enough free blocks — free what we got and put data back.
                    for &b in &new_block_ids {
                        self.allocator.free(b);
                    }
                    self.cpu_store.return_sequence(sequence_id, entry);
                    return None;
                }
            }
        }

        // Write each stored block into its new GPU slot.
        let block_stride = self.block_stride();
        // Lazily grow storage to cover the highest new block id.
        if let Some(&max_block_id) = new_block_ids.iter().max() {
            let needed_storage = (max_block_id + 1) * block_stride;
            if self.key_storage.len() < needed_storage {
                self.key_storage.resize(needed_storage, f16::ZERO);
                self.value_storage.resize(needed_storage, f16::ZERO);
            }
        }
        for (i, &new_block_id) in new_block_ids.iter().enumerate() {
            let (ref key_data, ref val_data) = entry.blocks[i];
            let start = new_block_id * block_stride;
            let end = start + block_stride;
            self.key_storage[start..end].copy_from_slice(key_data);
            self.value_storage[start..end].copy_from_slice(val_data);
        }

        // Recreate the sequence with the restored state.
        let access = self.bump_access();
        let mut seq = PagedSequence::new(self.num_layers);
        seq.blocks.extend(new_block_ids);
        seq.current_len = entry.current_len;
        seq.priority = entry.priority;
        seq.last_access = access;
        self.sequences.insert(sequence_id, seq);

        let freed = num_blocks as u64;
        self.cpu_offloaded_blocks = self.cpu_offloaded_blocks.saturating_sub(freed);

        Some(entry.current_len)
    }

    fn allocate_blocks(
        &mut self,
        count: usize,
        protected_seq_id: Option<u64>,
    ) -> Result<Vec<usize>, KvCacheError> {
        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            match self.allocate_block(protected_seq_id) {
                Ok(block) => blocks.push(block),
                Err(e) => {
                    // Cleanup allocated blocks
                    for block in blocks {
                        self.allocator.free(block);
                    }
                    return Err(e);
                }
            }
        }
        Ok(blocks)
    }

    fn evict_one_inactive(&mut self, protected_seq_id: Option<u64>) -> bool {
        // Composite eviction key: (priority_bucket, time_val).
        // priority_bucket = 255 - priority, so lower-priority sequences get a
        // higher bucket and are chosen before higher-priority ones.
        // Within the same bucket, the oldest sequence (lowest time_val) wins.
        let mut victim: Option<u64> = None;
        let mut best_key: (u8, u64) = (0, u64::MAX); // sentinel: nothing selected yet

        for (seq_id, seq) in &self.sequences {
            if self.active_sequences.contains(seq_id) {
                continue;
            }
            if Some(*seq_id) == protected_seq_id {
                continue;
            }

            let time_val = match self.eviction_policy {
                KvEvictionPolicy::LruInactive => seq.last_access,
                KvEvictionPolicy::Fifo => *seq_id, // Approximation using ID as FIFO order
                KvEvictionPolicy::None => continue,
            };

            let priority_bucket = u8::MAX - seq.priority;
            let key = (priority_bucket, time_val);

            let is_better = match victim {
                None => true,
                Some(_) => {
                    // Higher priority_bucket (lower priority) → evict first.
                    // Tie-break: lower time_val (older) → evict first.
                    key.0 > best_key.0 || (key.0 == best_key.0 && key.1 < best_key.1)
                }
            };

            if is_better {
                best_key = key;
                victim = Some(*seq_id);
            }
        }

        if let Some(seq_id) = victim {
            self.evict_sequence(seq_id);
            self.evicted_sequences.push(seq_id);
            return true;
        }

        false
    }

    pub fn rollback_sequence(&mut self, sequence_id: u64, length: usize) {
        let mut seq = match self.sequences.remove(&sequence_id) {
            Some(seq) => seq,
            None => return,
        };

        if length >= seq.current_len {
            self.sequences.insert(sequence_id, seq);
            return;
        }

        // Remove stale prefix entries that become invalid after rollback.
        let old_full_blocks = seq.tokens.len() / self.block_size;
        let new_full_blocks = length / self.block_size;
        if old_full_blocks > new_full_blocks {
            for block_idx in new_full_blocks..old_full_blocks {
                let Some(&block_id) = seq.blocks.get(block_idx) else {
                    break;
                };
                if self.allocator.get_ref_count(block_id) <= 1 {
                    let prefix_len = (block_idx + 1) * self.block_size;
                    if prefix_len <= seq.tokens.len() {
                        self.radix_tree.remove(&seq.tokens[..prefix_len]);
                    }
                }
            }
        }

        seq.current_len = length;
        if length < seq.tokens.len() {
            seq.tokens.truncate(length);
        }

        // Free blocks that are completely outside the new length.
        let needed_blocks = length.div_ceil(self.block_size);
        while seq.blocks.len() > needed_blocks {
            if let Some(block) = seq.blocks.pop_back() {
                self.allocator.free(block);
            }
        }

        // Invalidate packed layers
        for slot in seq.packed_layers.iter_mut() {
            *slot = None;
        }

        self.sequences.insert(sequence_id, seq);
    }

    fn evict_sequence(&mut self, sequence_id: u64) {
        if let Some(seq) = self.sequences.remove(&sequence_id) {
            let full_blocks = seq.tokens.len() / self.block_size;
            for block_idx in 0..full_blocks {
                let Some(&block_id) = seq.blocks.get(block_idx) else {
                    break;
                };
                if self.allocator.get_ref_count(block_id) <= 1 {
                    let prefix_len = (block_idx + 1) * self.block_size;
                    if prefix_len <= seq.tokens.len() {
                        self.radix_tree.remove(&seq.tokens[..prefix_len]);
                    }
                }
            }

            let freed_blocks = seq.blocks.len() as u64;
            for block in seq.blocks {
                self.allocator.free(block);
            }
            self.evicted_blocks += freed_blocks;
            self.evicted_sequences_count += 1;
        }
    }

    fn blocks_needed(&self, logical_block: usize, current_blocks: usize) -> usize {
        logical_block
            .saturating_add(1)
            .saturating_sub(current_blocks)
    }

    pub fn set_active_sequences(&mut self, seq_ids: &[u64]) {
        self.active_sequences.clear();
        self.active_sequences.extend(seq_ids.iter().copied());
    }

    pub fn clear_active_sequences(&mut self) {
        self.active_sequences.clear();
    }

    pub fn drain_evicted_sequences(&mut self) -> Vec<u64> {
        let drained = self.evicted_sequences.clone();
        self.evicted_sequences.clear();
        drained
    }

    pub fn allocate_sequence(
        &mut self,
        sequence_id: u64,
        tokens: &[u32],
    ) -> Result<usize, KvCacheError> {
        if let Some(seq) = self.sequences.get(&sequence_id) {
            return Ok(seq.current_len.min(tokens.len().saturating_sub(1)));
        }

        // Before creating a fresh sequence, check whether this ID was previously
        // offloaded to CPU.  If restoration succeeds the sequence resumes from
        // where it left off, avoiding a full prefix recompute.
        if self.cpu_store.has_sequence(sequence_id) {
            if let Some(restored_len) = self.restore_offloaded_sequence_inner(sequence_id) {
                return Ok(restored_len.min(tokens.len().saturating_sub(1)));
            }
            // If restoration failed (not enough GPU blocks), fall through to
            // fresh allocation below — the data remains in cpu_store for a
            // later retry.
        }

        // Try to match prefix
        let tokens_u64: Vec<u64> = tokens.iter().map(|&t| t as u64).collect();
        let (matched_blocks, matched_tokens_len) = self.radix_tree.match_prefix(&tokens_u64);

        let mut seq = PagedSequence::new(self.num_layers);

        // Verify prefix blocks are still valid before reusing them.
        let mut reused_blocks = Vec::new();
        for &block_id in &matched_blocks {
            if self.allocator.get_ref_count(block_id) > 0 {
                self.allocator.add_ref(block_id);
                reused_blocks.push(block_id);
            } else {
                break;
            }
        }

        // Cap reused_len to ensure at least one token is processed (kickstart generation).
        let max_reused_tokens = reused_blocks.len().saturating_mul(self.block_size);
        let matched_len = matched_tokens_len
            .min(max_reused_tokens)
            .min(tokens.len().saturating_sub(1));
        seq.current_len = matched_len;
        if matched_len > 0 {
            let num_reused_blocks = matched_len.div_ceil(self.block_size);
            if reused_blocks.len() > num_reused_blocks {
                for &block_id in &reused_blocks[num_reused_blocks..] {
                    self.allocator.free(block_id);
                }
                reused_blocks.truncate(num_reused_blocks);
            }
            seq.blocks.extend(reused_blocks);
        }
        // Sync tokens tracking
        seq.tokens = tokens[..matched_len].iter().map(|&t| t as u64).collect();
        self.sequences.insert(sequence_id, seq);
        Ok(matched_len)
    }

    pub fn append_token(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
        pos: usize,
        key: &[f16],
        value: &[f16],
        token_id: Option<u64>,
    ) -> Result<(), KvCacheError> {
        let expected = self.num_heads * self.head_dim;
        let logical_block = pos / self.block_size;
        let token_in_block = pos % self.block_size;

        let current_blocks = self
            .sequences
            .get(&sequence_id)
            .map(|s| s.blocks.len())
            .unwrap_or(0);
        let needed = self.blocks_needed(logical_block, current_blocks);
        let new_blocks = if needed > 0 {
            self.allocate_blocks(needed, Some(sequence_id))?
        } else {
            Vec::new()
        };

        let access = self.bump_access();
        let block_stride = self.block_stride();
        let layer_stride = self.layer_stride();
        let head_stride = self.head_stride();
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;

        if self.scratch_key.len() != expected {
            self.scratch_key.resize(expected, f16::ZERO);
        }
        if self.scratch_value.len() != expected {
            self.scratch_value.resize(expected, f16::ZERO);
        }

        let key_slice = if key.len() >= expected {
            &key[..expected]
        } else {
            self.scratch_key[..key.len()].copy_from_slice(key);
            self.scratch_key[key.len()..].fill(f16::ZERO);
            &self.scratch_key[..]
        };

        let value_slice = if value.len() >= expected {
            &value[..expected]
        } else {
            self.scratch_value[..value.len()].copy_from_slice(value);
            self.scratch_value[value.len()..].fill(f16::ZERO);
            &self.scratch_value[..]
        };

        let seq = self
            .sequences
            .get_mut(&sequence_id)
            .expect("sequence not allocated");
        seq.last_access = access;
        seq.blocks.extend(new_blocks);
        seq.packed_layers[layer_index] = None;

        // Track tokens and populate RadixTree
        // Only track for layer 0 to avoid duplicates
        if layer_index == 0 {
            if let Some(tid) = token_id {
                // Ensure we are appending at the end
                if pos == seq.tokens.len() {
                    seq.tokens.push(tid);

                    // If block is full, add to RadixTree
                    if (pos + 1).is_multiple_of(self.block_size) {
                        let block_idx = *seq.blocks.back().expect("block missing");
                        // Get tokens for this block
                        let start = 0;
                        let block_tokens = &seq.tokens[start..=pos];
                        self.radix_tree.insert(block_tokens, block_idx);
                        // Increment ref count for storage in tree?
                        // The allocator counts refs. When we put it in the tree, we should probably add a ref.
                        // But wait, the tree doesn't "own" the block in the sense of allocation.
                        // If we evict the block, we should remove from tree.
                        // For now, let's just use the tree as an index.
                        // We should increment ref count if we *share* it.
                    }
                } else if pos < seq.tokens.len() {
                    // Overwriting? Verify token matches?
                    // For now assume consistent
                }
            }
        }

        let block = *seq.blocks.get(logical_block).expect("block missing");
        // Lazily grow flat storage so this physical block is addressable.
        let needed_storage = (block + 1) * block_stride;
        if self.key_storage.len() < needed_storage {
            self.key_storage.resize(needed_storage, f16::ZERO);
            self.value_storage.resize(needed_storage, f16::ZERO);
        }
        for h in 0..num_heads {
            let offset = h * head_dim;
            let end = offset + head_dim;
            let dst = block * block_stride
                + layer_index * layer_stride
                + h * head_stride
                + token_in_block * head_dim;
            let dst_end = dst + head_dim;
            self.key_storage[dst..dst_end].copy_from_slice(&key_slice[offset..end]);
            self.value_storage[dst..dst_end].copy_from_slice(&value_slice[offset..end]);
        }
        Ok(())
    }

    pub fn append_head_range_seq_first(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
        head: usize,
        pos_start: usize,
        key_values: &[f16],
        value_values: &[f16],
    ) -> Result<(), KvCacheError> {
        let block_size = self.block_size;
        let mut remaining = key_values.len() / self.head_dim;
        let mut src_offset = 0usize;
        let mut pos = pos_start;
        if remaining == 0 {
            return Ok(());
        }

        let last_pos = pos_start + remaining - 1;
        let last_block = last_pos / block_size;
        let current_blocks = self
            .sequences
            .get(&sequence_id)
            .map(|s| s.blocks.len())
            .unwrap_or(0);
        let needed = self.blocks_needed(last_block, current_blocks);
        let new_blocks = if needed > 0 {
            self.allocate_blocks(needed, Some(sequence_id))?
        } else {
            Vec::new()
        };

        let access = self.bump_access();
        let block_stride = self.block_stride();
        let layer_stride = self.layer_stride();
        let head_stride = self.head_stride();
        let head_dim = self.head_dim;

        let seq = self
            .sequences
            .get_mut(&sequence_id)
            .expect("sequence not allocated");
        seq.last_access = access;
        seq.blocks.extend(new_blocks);
        seq.packed_layers[layer_index] = None;

        while remaining > 0 {
            let logical_block = pos / block_size;
            let token_in_block = pos % block_size;

            let block = *seq.blocks.get(logical_block).expect("block missing");
            // Lazily grow flat storage so this physical block is addressable.
            let needed_storage = (block + 1) * block_stride;
            if self.key_storage.len() < needed_storage {
                self.key_storage.resize(needed_storage, f16::ZERO);
                self.value_storage.resize(needed_storage, f16::ZERO);
            }
            let capacity = block_size - token_in_block;
            let to_copy = remaining.min(capacity);
            let copy_len = to_copy * head_dim;

            let dst = block * block_stride
                + layer_index * layer_stride
                + head * head_stride
                + token_in_block * head_dim;
            let dst_end = dst + copy_len;
            let src_end = src_offset + copy_len;

            self.key_storage[dst..dst_end].copy_from_slice(&key_values[src_offset..src_end]);
            self.value_storage[dst..dst_end].copy_from_slice(&value_values[src_offset..src_end]);

            remaining -= to_copy;
            pos += to_copy;
            src_offset = src_end;
        }

        Ok(())
    }

    pub fn set_packed_layer(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
        length: usize,
        key: &[f16],
        value: &[f16],
    ) {
        let seq = self
            .sequences
            .get_mut(&sequence_id)
            .expect("sequence not allocated");
        let expected = self.num_heads * self.head_dim * length;
        if key.len() != expected || value.len() != expected {
            log::warn!(
                "Packed KV length mismatch for seq {} layer {}: key={}, value={}, expected={}",
                sequence_id,
                layer_index,
                key.len(),
                value.len(),
                expected
            );
            return;
        }
        if layer_index >= seq.packed_layers.len() {
            return;
        }
        seq.packed_layers[layer_index] = Some(PackedLayerKv {
            key: Arc::from(key),
            value: Arc::from(value),
            length,
        });
    }

    pub fn clear_packed_layer(&mut self, sequence_id: u64, layer_index: usize) {
        if let Some(seq) = self.sequences.get_mut(&sequence_id) {
            if layer_index < seq.packed_layers.len() {
                seq.packed_layers[layer_index] = None;
            }
        }
    }

    pub fn get_packed_layer(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
    ) -> Option<PackedKvView> {
        let seq_len = self.sequence_length(sequence_id)?;
        if seq_len == 0 {
            return None;
        }
        if let Some(packed) = self
            .sequences
            .get(&sequence_id)
            .and_then(|seq| seq.packed_layers.get(layer_index).and_then(|v| v.as_ref()))
        {
            return Some(PackedKvView {
                key: packed.key.clone(),
                value: packed.value.clone(),
                length: packed.length,
            });
        }

        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let block_stride = self.block_stride();
        let layer_stride = self.layer_stride();
        let head_stride = self.head_stride();
        let blocks: Vec<usize> = self
            .sequences
            .get(&sequence_id)?
            .blocks
            .iter()
            .copied()
            .collect();

        let total = num_heads * seq_len * head_dim;
        let mut key = vec![f16::ZERO; total];
        let mut value = vec![f16::ZERO; total];

        for h in 0..num_heads {
            for pos in 0..seq_len {
                let logical_block = pos / self.block_size;
                let token_in_block = pos % self.block_size;
                let block = *blocks.get(logical_block)?;
                let src = block * block_stride
                    + layer_index * layer_stride
                    + h * head_stride
                    + token_in_block * head_dim;
                let dst = h * seq_len * head_dim + pos * head_dim;
                let src_end = src + head_dim;
                let dst_end = dst + head_dim;
                if src_end <= self.key_storage.len() {
                    key[dst..dst_end].copy_from_slice(&self.key_storage[src..src_end]);
                    value[dst..dst_end].copy_from_slice(&self.value_storage[src..src_end]);
                }
            }
        }

        let packed = PackedLayerKv {
            key: Arc::from(key),
            value: Arc::from(value),
            length: seq_len,
        };

        if let Some(seq) = self.sequences.get_mut(&sequence_id) {
            if layer_index < seq.packed_layers.len() {
                seq.packed_layers[layer_index] = Some(packed.clone());
            }
        }

        Some(PackedKvView {
            key: packed.key.clone(),
            value: packed.value.clone(),
            length: packed.length,
        })
    }

    pub fn advance_sequence(&mut self, sequence_id: u64) {
        let seq = self.sequences.get_mut(&sequence_id).unwrap();
        seq.current_len += 1;
    }

    pub fn advance_sequence_by(&mut self, sequence_id: u64, delta: usize) {
        let seq = self.sequences.get_mut(&sequence_id).unwrap();
        seq.current_len += delta;
    }

    pub fn overwrite_layer(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
        key: &KvTensor,
        value: &KvTensor,
        length: usize,
    ) {
        let expected_len = self.num_heads * length * self.head_dim;
        if key.data.len() < expected_len || value.data.len() < expected_len {
            return;
        }
        let old_blocks = {
            let seq = self.sequences.get_mut(&sequence_id).unwrap();
            let old = std::mem::take(&mut seq.blocks);
            seq.current_len = 0;
            old
        };
        for block in old_blocks {
            self.allocator.free(block);
        }

        let block_size = self.block_size;
        let blocks_needed = length.div_ceil(block_size);
        let new_blocks = match self.allocate_blocks(blocks_needed, Some(sequence_id)) {
            Ok(blocks) => blocks,
            Err(_) => return,
        };

        let block_stride = self.block_stride();
        let layer_stride = self.layer_stride();
        let head_stride = self.head_stride();
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;

        // Lazily grow storage to cover the highest new block id before mutably
        // borrowing seq (borrow checker constraint).
        if let Some(&max_block_id) = new_blocks.iter().max() {
            let needed_storage = (max_block_id + 1) * block_stride;
            if self.key_storage.len() < needed_storage {
                self.key_storage.resize(needed_storage, f16::ZERO);
                self.value_storage.resize(needed_storage, f16::ZERO);
            }
        }

        let seq = self.sequences.get_mut(&sequence_id).unwrap();
        seq.blocks = new_blocks.into();
        seq.packed_layers[layer_index] = None;

        for h in 0..num_heads {
            for pos in 0..length {
                let logical_block = pos / block_size;
                let token_in_block = pos % block_size;
                let block = *seq.blocks.get(logical_block).unwrap();
                let src = h * key.max_seq_len * head_dim + pos * head_dim;
                let dst = block * block_stride
                    + layer_index * layer_stride
                    + h * head_stride
                    + token_in_block * head_dim;
                let src_end = src + head_dim;
                let dst_end = dst + head_dim;
                self.key_storage[dst..dst_end].copy_from_slice(&key.data[src..src_end]);
                self.value_storage[dst..dst_end].copy_from_slice(&value.data[src..src_end]);
            }
        }
        seq.current_len = length;
    }

    pub fn remove_sequence(&mut self, sequence_id: u64) {
        if let Some(seq) = self.sequences.remove(&sequence_id) {
            let full_blocks = seq.tokens.len() / self.block_size;
            for block_idx in 0..full_blocks {
                let Some(&block_id) = seq.blocks.get(block_idx) else {
                    break;
                };
                if self.allocator.get_ref_count(block_id) <= 1 {
                    let prefix_len = (block_idx + 1) * self.block_size;
                    if prefix_len <= seq.tokens.len() {
                        self.radix_tree.remove(&seq.tokens[..prefix_len]);
                    }
                }
            }

            for block in seq.blocks {
                self.allocator.free(block);
            }
        }
    }

    pub fn has_sequence(&self, sequence_id: u64) -> bool {
        self.sequences.contains_key(&sequence_id)
    }

    pub fn sequence_length(&self, sequence_id: u64) -> Option<usize> {
        self.sequences.get(&sequence_id).map(|s| s.current_len)
    }

    pub fn get_layer_view(&mut self, sequence_id: u64, layer_index: usize) -> Option<KvView> {
        let packed = self.get_packed_layer(sequence_id, layer_index)?;
        Some(KvView {
            key: packed.key,
            value: packed.value,
            length: packed.length,
        })
    }

    pub fn get_layer_as_onnx(&mut self, sequence_id: u64, layer: usize) -> Option<Array4<f16>> {
        let view = self.get_layer_view(sequence_id, layer)?;
        let seq_len = view.length;

        let stride = self.num_heads * self.head_dim;
        if stride == 0 || view.key.len() % stride != 0 {
            return None;
        }
        let max_seq_len = view.key.len() / stride;
        if seq_len > max_seq_len {
            return None;
        }

        let total = self.num_heads * seq_len * self.head_dim;
        let mut packed = vec![f16::ZERO; total];
        for h in 0..self.num_heads {
            let head_offset = h * max_seq_len * self.head_dim;
            let packed_offset = h * seq_len * self.head_dim;
            for pos in 0..seq_len {
                let src = head_offset + pos * self.head_dim;
                let dst = packed_offset + pos * self.head_dim;
                let src_end = src + self.head_dim;
                let dst_end = dst + self.head_dim;
                if src_end <= view.key.len() {
                    packed[dst..dst_end].copy_from_slice(&view.key[src..src_end]);
                }
            }
        }

        Array4::from_shape_vec((1, self.num_heads, seq_len, self.head_dim), packed).ok()
    }

    pub fn stats(&self) -> KvCacheStats {
        let blocks_total = self.allocator.total_blocks;
        let blocks_free = self.allocator.free_count();
        let blocks_used = blocks_total - blocks_free;
        let block_stride = self.block_stride();
        let bytes_used = blocks_used * block_stride * 2 * 2;
        let bytes_capacity = self.key_storage.len() * std::mem::size_of::<half::f16>()
            + self.value_storage.len() * std::mem::size_of::<half::f16>();
        let packed_layers = self
            .sequences
            .values()
            .map(|seq| seq.packed_layers.iter().filter(|v| v.is_some()).count())
            .sum();

        KvCacheStats {
            mode: KvCacheMode::Paged,
            bytes_used,
            bytes_capacity,
            blocks_total,
            blocks_free,
            sequences: self.sequences.len(),
            evicted_blocks: self.evicted_blocks,
            evicted_sequences: self.evicted_sequences_count,
            packed_layers,
            // Use the store's authoritative count so stats stay consistent
            // even when blocks are put back during failed restore attempts.
            cpu_offloaded_blocks: self.cpu_store.offloaded_block_count() as u64,
        }
    }
}

enum KvCacheInner {
    Dense(DenseKvCache),
    Paged(Box<PagedKvCache>),
}

pub struct KvCache {
    inner: KvCacheInner,
    mode: KvCacheMode,
}

impl KvCache {
    pub fn new(num_layers: usize, num_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        Self {
            inner: KvCacheInner::Dense(DenseKvCache::new(
                num_layers,
                num_heads,
                max_seq_len,
                head_dim,
            )),
            mode: KvCacheMode::Dense,
        }
    }

    pub fn new_with_config(
        num_layers: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        config: KvCacheConfig,
    ) -> Self {
        match config.mode {
            KvCacheMode::Dense => Self {
                inner: KvCacheInner::Dense(DenseKvCache::new_with_config(
                    num_layers,
                    num_heads,
                    max_seq_len,
                    head_dim,
                    config.dense_free_list_cap,
                    config.initial_seq_len,
                )),
                mode: KvCacheMode::Dense,
            },
            KvCacheMode::Paged => Self {
                inner: KvCacheInner::Paged(Box::new(PagedKvCache::new(
                    num_layers,
                    num_heads,
                    config.block_size,
                    config.total_blocks,
                    head_dim,
                    config.eviction_policy,
                ))),
                mode: KvCacheMode::Paged,
            },
        }
    }

    pub fn mode(&self) -> KvCacheMode {
        self.mode
    }

    pub fn set_active_sequences(&mut self, seq_ids: &[u64]) {
        if let KvCacheInner::Paged(cache) = &mut self.inner {
            cache.set_active_sequences(seq_ids);
        }
    }

    pub fn clear_active_sequences(&mut self) {
        if let KvCacheInner::Paged(cache) = &mut self.inner {
            cache.clear_active_sequences();
        }
    }

    pub fn drain_evicted_sequences(&mut self) -> Vec<u64> {
        if let KvCacheInner::Paged(cache) = &mut self.inner {
            return cache.drain_evicted_sequences();
        }
        Vec::new()
    }

    /// Update the eviction priority of an already-allocated sequence.
    ///
    /// Propagate `SequenceGroup::priority` here after `allocate_sequence` so
    /// that the KV cache uses the same priority ordering as the scheduler when
    /// choosing blocks to offload under memory pressure. No-op in dense mode.
    pub fn set_sequence_priority(&mut self, sequence_id: u64, priority: u8) {
        if let KvCacheInner::Paged(cache) = &mut self.inner {
            cache.set_sequence_priority(sequence_id, priority);
        }
    }

    /// Number of blocks currently held in the CPU offload store (paged mode only).
    pub fn cpu_offloaded_blocks(&self) -> u64 {
        if let KvCacheInner::Paged(cache) = &self.inner {
            return cache.cpu_store.offloaded_block_count() as u64;
        }
        0
    }

    /// Attempt to restore an offloaded sequence from CPU memory back to GPU.
    ///
    /// Returns `true` if the sequence was successfully restored. The next call
    /// to `allocate_sequence` for the same ID will see it as already present.
    /// No-op in dense mode.
    pub fn restore_offloaded_sequence(&mut self, sequence_id: u64) -> bool {
        if let KvCacheInner::Paged(cache) = &mut self.inner {
            return cache
                .restore_offloaded_sequence_inner(sequence_id)
                .is_some();
        }
        false
    }

    pub fn rollback_sequence(&mut self, sequence_id: u64, length: usize) {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => cache.rollback_sequence(sequence_id, length),
            KvCacheInner::Paged(cache) => cache.rollback_sequence(sequence_id, length),
        }
    }

    pub fn allocate_sequence(
        &mut self,
        sequence_id: u64,
        tokens: &[u32],
    ) -> Result<usize, KvCacheError> {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => cache.allocate_sequence(sequence_id, tokens),
            KvCacheInner::Paged(cache) => cache.allocate_sequence(sequence_id, tokens),
        }
    }

    pub fn append_token(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
        pos: usize,
        key: &[f16],
        value: &[f16],
        token_id: Option<u64>,
    ) -> Result<(), KvCacheError> {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => {
                cache.append_token(sequence_id, layer_index, pos, key, value, token_id)
            }
            KvCacheInner::Paged(cache) => {
                cache.append_token(sequence_id, layer_index, pos, key, value, token_id)
            }
        }
    }

    pub fn set_packed_layer(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
        length: usize,
        key: &[f16],
        value: &[f16],
    ) {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => {
                cache.set_packed_layer(sequence_id, layer_index, length, key, value)
            }
            KvCacheInner::Paged(cache) => {
                cache.set_packed_layer(sequence_id, layer_index, length, key, value)
            }
        }
    }

    pub fn clear_packed_layer(&mut self, sequence_id: u64, layer_index: usize) {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => cache.clear_packed_layer(sequence_id, layer_index),
            KvCacheInner::Paged(cache) => cache.clear_packed_layer(sequence_id, layer_index),
        }
    }

    pub fn get_packed_layer(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
    ) -> Option<PackedKvView> {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => cache.get_packed_layer(sequence_id, layer_index),
            KvCacheInner::Paged(cache) => cache.get_packed_layer(sequence_id, layer_index),
        }
    }

    pub fn append_head_range_seq_first(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
        head: usize,
        pos_start: usize,
        key_values: &[f16],
        value_values: &[f16],
    ) -> Result<(), KvCacheError> {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => cache.append_head_range_seq_first(
                sequence_id,
                layer_index,
                head,
                pos_start,
                key_values,
                value_values,
            ),
            KvCacheInner::Paged(cache) => cache.append_head_range_seq_first(
                sequence_id,
                layer_index,
                head,
                pos_start,
                key_values,
                value_values,
            ),
        }
    }

    pub fn advance_sequence(&mut self, sequence_id: u64) {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => cache.advance_sequence(sequence_id),
            KvCacheInner::Paged(cache) => cache.advance_sequence(sequence_id),
        }
    }

    pub fn advance_sequence_by(&mut self, sequence_id: u64, delta: usize) {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => cache.advance_sequence_by(sequence_id, delta),
            KvCacheInner::Paged(cache) => cache.advance_sequence_by(sequence_id, delta),
        }
    }

    pub fn overwrite_layer(
        &mut self,
        sequence_id: u64,
        layer_index: usize,
        key: &KvTensor,
        value: &KvTensor,
        length: usize,
    ) {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => {
                cache.overwrite_layer(sequence_id, layer_index, key, value, length)
            }
            KvCacheInner::Paged(cache) => {
                cache.overwrite_layer(sequence_id, layer_index, key, value, length)
            }
        }
    }

    pub fn remove_sequence(&mut self, sequence_id: u64) {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => cache.remove_sequence(sequence_id),
            KvCacheInner::Paged(cache) => cache.remove_sequence(sequence_id),
        }
    }

    pub fn has_sequence(&self, sequence_id: u64) -> bool {
        match &self.inner {
            KvCacheInner::Dense(cache) => cache.has_sequence(sequence_id),
            KvCacheInner::Paged(cache) => cache.has_sequence(sequence_id),
        }
    }

    pub fn sequence_length(&self, sequence_id: u64) -> Option<usize> {
        match &self.inner {
            KvCacheInner::Dense(cache) => cache.sequence_length(sequence_id),
            KvCacheInner::Paged(cache) => cache.sequence_length(sequence_id),
        }
    }

    pub fn get_layer_view(&mut self, sequence_id: u64, layer_index: usize) -> Option<KvView> {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => cache.get_layer_view(sequence_id, layer_index),
            KvCacheInner::Paged(cache) => cache.get_layer_view(sequence_id, layer_index),
        }
    }

    pub fn get_layer_as_onnx(&mut self, sequence_id: u64, layer: usize) -> Option<Array4<f16>> {
        match &mut self.inner {
            KvCacheInner::Dense(cache) => cache.get_layer_as_onnx(sequence_id, layer),
            KvCacheInner::Paged(cache) => cache.get_layer_as_onnx(sequence_id, layer),
        }
    }

    pub fn stats(&self) -> KvCacheStats {
        match &self.inner {
            KvCacheInner::Dense(cache) => cache.stats(),
            KvCacheInner::Paged(cache) => cache.stats(),
        }
    }
}

#[path = "kv_cache_tests.rs"]
mod kv_cache_tests;
