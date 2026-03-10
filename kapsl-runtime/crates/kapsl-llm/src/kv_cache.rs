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
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            mode: KvCacheMode::Dense,
            block_size: 16,
            total_blocks: 2048,
            eviction_policy: KvEvictionPolicy::LruInactive,
            dense_free_list_cap: 8,
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
    max_seq_len: usize,
    head_dim: usize,
}

impl KvTensor {
    pub fn new(num_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let size = num_heads * max_seq_len * head_dim;
        Self {
            data: vec![f16::ZERO; size],
            num_heads,
            max_seq_len,
            head_dim,
        }
    }

    #[inline(always)]
    fn index(&self, head: usize, pos: usize, dim: usize) -> usize {
        debug_assert!(head < self.num_heads);
        debug_assert!(pos < self.max_seq_len);
        debug_assert!(dim < self.head_dim);

        (head * self.max_seq_len * self.head_dim) + (pos * self.head_dim) + dim
    }

    #[inline(always)]
    pub fn write_head(&mut self, head: usize, pos: usize, values: &[f16]) {
        let base = self.index(head, pos, 0);
        self.data[base..base + self.head_dim].copy_from_slice(values);
    }

    #[inline(always)]
    pub fn write_head_range(&mut self, head: usize, pos_start: usize, values: &[f16]) {
        debug_assert!(pos_start < self.max_seq_len);
        debug_assert!(values.len().is_multiple_of(self.head_dim));
        let num_positions = values.len() / self.head_dim;
        debug_assert!(pos_start + num_positions <= self.max_seq_len);
        let base = self.index(head, pos_start, 0);
        let end = base + values.len();
        self.data[base..end].copy_from_slice(values);
    }

    pub fn as_slice(&self) -> &[f16] {
        &self.data
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
    pub fn new(num_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        Self {
            key: KvTensor::new(num_heads, max_seq_len, head_dim),
            value: KvTensor::new(num_heads, max_seq_len, head_dim),
        }
    }
}

pub struct SequenceKvCache {
    pub layers: Vec<LayerKv>,
    pub current_len: usize,
    packed_layers: Vec<Option<PackedLayerKv>>,
}

impl SequenceKvCache {
    pub fn new(num_layers: usize, num_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKv::new(num_heads, max_seq_len, head_dim))
            .collect();

        let mut packed_layers = Vec::with_capacity(num_layers);
        packed_layers.resize_with(num_layers, || None);

        Self {
            layers,
            current_len: 0,
            packed_layers,
        }
    }
}

pub struct DenseKvCache {
    sequences: HashMap<u64, SequenceKvCache>,
    free_list: Vec<SequenceKvCache>,
    free_list_cap: usize,
    scratch_key: Vec<f16>,
    scratch_value: Vec<f16>,
    max_seq_len: usize,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
}

impl DenseKvCache {
    pub fn new(num_layers: usize, num_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        Self::new_with_free_list_cap(num_layers, num_heads, max_seq_len, head_dim, 8)
    }

    pub fn new_with_free_list_cap(
        num_layers: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        free_list_cap: usize,
    ) -> Self {
        let expected = num_heads * head_dim;
        Self {
            sequences: HashMap::new(),
            free_list: Vec::new(),
            free_list_cap,
            scratch_key: vec![f16::ZERO; expected],
            scratch_value: vec![f16::ZERO; expected],
            num_layers,
            num_heads,
            max_seq_len,
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

    pub fn get_layer_view(&self, sequence_id: u64, layer_index: usize) -> Option<KvView> {
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
        // [num_heads, max_seq_len, head_dim] buffer. Each head's used positions are
        // contiguous (layout: head * max_seq_len * head_dim + pos * head_dim + dim).
        let stride = self.max_seq_len * self.head_dim;
        let compact_per_head = length * self.head_dim;
        let compact_len = self.num_heads * compact_per_head;
        let mut key_compact = Vec::with_capacity(compact_len);
        let mut val_compact = Vec::with_capacity(compact_len);
        let key_slice = layer.key.as_slice();
        let val_slice = layer.value.as_slice();
        for h in 0..self.num_heads {
            let src = h * stride;
            key_compact.extend_from_slice(&key_slice[src..src + compact_per_head]);
            val_compact.extend_from_slice(&val_slice[src..src + compact_per_head]);
        }

        Some(KvView {
            key: Arc::from(key_compact),
            value: Arc::from(val_compact),
            length,
        })
    }

    pub fn get_layer_as_onnx(&self, sequence_id: u64, layer: usize) -> Option<Array4<f16>> {
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
        }
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
        let total_elems = total_blocks * block_stride;
        let expected = num_heads * head_dim;
        Self {
            sequences: HashMap::new(),
            allocator: PagedBlockAllocator::new(total_blocks),
            key_storage: vec![f16::ZERO; total_elems],
            value_storage: vec![f16::ZERO; total_elems],
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

        if self.eviction_policy != KvEvictionPolicy::None
            && self.evict_one_inactive(protected_seq_id)
        {
            if let Some(block) = self.allocator.allocate() {
                return Ok(block);
            }
        }

        Err(KvCacheError::OutOfBlocks)
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
        let mut victim: Option<u64> = None;
        let mut min_val: u64 = u64::MAX;

        for (seq_id, seq) in &self.sequences {
            if self.active_sequences.contains(seq_id) {
                continue;
            }
            if Some(*seq_id) == protected_seq_id {
                continue;
            }

            let val = match self.eviction_policy {
                KvEvictionPolicy::LruInactive => seq.last_access,
                KvEvictionPolicy::Fifo => *seq_id, // Approximation using ID
                KvEvictionPolicy::None => continue,
            };

            if val < min_val {
                min_val = val;
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
                key[dst..dst_end].copy_from_slice(&self.key_storage[src..src_end]);
                value[dst..dst_end].copy_from_slice(&self.value_storage[src..src_end]);
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
        let bytes_capacity = blocks_total * block_stride * 2 * 2;
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
                inner: KvCacheInner::Dense(DenseKvCache::new_with_free_list_cap(
                    num_layers,
                    num_heads,
                    max_seq_len,
                    head_dim,
                    config.dense_free_list_cap,
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
