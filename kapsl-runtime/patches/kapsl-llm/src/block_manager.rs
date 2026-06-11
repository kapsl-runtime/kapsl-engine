use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Represents a physical block of memory in the KV cache
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalTokenBlock {
    pub block_number: usize,
    pub block_size: usize,
    pub device_id: usize,
}

/// Manages the allocation of physical blocks
#[derive(Debug)]
pub struct BlockAllocator {
    free_blocks: VecDeque<usize>,
    total_blocks: usize,
    block_size: usize,
    device_id: usize,
}

impl BlockAllocator {
    pub fn new(total_blocks: usize, block_size: usize, device_id: usize) -> Self {
        let free_blocks: VecDeque<usize> = (0..total_blocks).collect();
        Self {
            free_blocks,
            total_blocks,
            block_size,
            device_id,
        }
    }

    pub fn allocate(&mut self) -> Option<PhysicalTokenBlock> {
        self.free_blocks
            .pop_front()
            .map(|block_number| PhysicalTokenBlock {
                block_number,
                block_size: self.block_size,
                device_id: self.device_id,
            })
    }

    pub fn free(&mut self, block: PhysicalTokenBlock) {
        if block.device_id == self.device_id {
            self.free_blocks.push_front(block.block_number);
        }
    }

    pub fn get_num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn get_num_total_blocks(&self) -> usize {
        self.total_blocks
    }
}

/// A reference-counted, mutex-wrapped [`BlockAllocator`] that can be shared
/// across multiple [`BlockManager`] instances.
///
/// All loaded models that draw from the same `SharedBlockAllocator` form a
/// unified KV block pool: blocks freed by one engine are immediately available
/// to any other engine sharing the same handle. Create one with
/// [`new_shared_allocator`] and hand clones of the `Arc` to each engine's
/// [`BlockManager::new_shared`].
pub type SharedBlockAllocator = Arc<Mutex<BlockAllocator>>;

/// Construct a new shared allocator owning `total_blocks` blocks.
pub fn new_shared_allocator(
    total_blocks: usize,
    block_size: usize,
    device_id: usize,
) -> SharedBlockAllocator {
    Arc::new(Mutex::new(BlockAllocator::new(
        total_blocks,
        block_size,
        device_id,
    )))
}

/// Return the current free-block count of a shared allocator without holding
/// the lock beyond this call.
pub fn shared_allocator_free_blocks(allocator: &SharedBlockAllocator) -> usize {
    allocator
        .lock()
        .expect("SharedBlockAllocator poisoned")
        .get_num_free_blocks()
}

/// Return total-block count of a shared allocator without holding the lock
/// beyond this call.
pub fn shared_allocator_total_blocks(allocator: &SharedBlockAllocator) -> usize {
    allocator
        .lock()
        .expect("SharedBlockAllocator poisoned")
        .get_num_total_blocks()
}

/// Internal dispatch: each `BlockManager` holds either a private allocator or
/// a reference to a pool shared with other managers.
enum BlockManagerAllocator {
    Owned(BlockAllocator),
    Shared(SharedBlockAllocator),
}

impl BlockManagerAllocator {
    fn allocate(&mut self) -> Option<PhysicalTokenBlock> {
        match self {
            Self::Owned(a) => a.allocate(),
            Self::Shared(a) => a.lock().expect("SharedBlockAllocator poisoned").allocate(),
        }
    }

    fn free(&mut self, block: PhysicalTokenBlock) {
        match self {
            Self::Owned(a) => a.free(block),
            Self::Shared(a) => a.lock().expect("SharedBlockAllocator poisoned").free(block),
        }
    }

    fn get_num_free_blocks(&self) -> usize {
        match self {
            Self::Owned(a) => a.get_num_free_blocks(),
            Self::Shared(a) => a
                .lock()
                .expect("SharedBlockAllocator poisoned")
                .get_num_free_blocks(),
        }
    }

    fn get_num_total_blocks(&self) -> usize {
        match self {
            Self::Owned(a) => a.get_num_total_blocks(),
            Self::Shared(a) => a
                .lock()
                .expect("SharedBlockAllocator poisoned")
                .get_num_total_blocks(),
        }
    }
}

/// Maps logical blocks to physical blocks for a sequence
#[derive(Debug, Clone)]
pub struct BlockTable {
    logical_to_physical: Vec<PhysicalTokenBlock>,
    #[allow(dead_code)]
    block_size: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Self {
        Self {
            logical_to_physical: Vec::new(),
            block_size,
        }
    }

    pub fn append(&mut self, block: PhysicalTokenBlock) {
        self.logical_to_physical.push(block);
    }

    pub fn get_physical_blocks(&self) -> &[PhysicalTokenBlock] {
        &self.logical_to_physical
    }

    pub fn len(&self) -> usize {
        self.logical_to_physical.len()
    }

    pub fn is_empty(&self) -> bool {
        self.logical_to_physical.is_empty()
    }
}

/// Manages blocks for all sequences.
///
/// Can operate in two modes:
/// - **Owned**: private `BlockAllocator` (original behaviour, created via [`BlockManager::new`]).
/// - **Shared**: draws from a [`SharedBlockAllocator`] that is also given to other managers,
///   enabling a unified KV block pool across multiple loaded models (created via
///   [`BlockManager::new_shared`]).
pub struct BlockManager {
    allocator: BlockManagerAllocator,
    block_tables: HashMap<u64, BlockTable>, // sequence_id -> BlockTable
    block_size: usize,
}

impl BlockManager {
    /// Create a `BlockManager` with a private block allocator.
    pub fn new(total_blocks: usize, block_size: usize, device_id: usize) -> Self {
        Self {
            allocator: BlockManagerAllocator::Owned(BlockAllocator::new(
                total_blocks,
                block_size,
                device_id,
            )),
            block_tables: HashMap::new(),
            block_size,
        }
    }

    /// Create a `BlockManager` that draws from a shared pool.
    ///
    /// Multiple engines sharing the same `SharedBlockAllocator` form a unified
    /// KV block pool: blocks freed by one engine become immediately available
    /// to any other engine holding a clone of the same allocator.
    pub fn new_shared(allocator: SharedBlockAllocator, block_size: usize) -> Self {
        Self {
            allocator: BlockManagerAllocator::Shared(allocator),
            block_tables: HashMap::new(),
            block_size,
        }
    }

    pub fn allocate(&mut self, sequence_id: u64) -> Option<PhysicalTokenBlock> {
        if let Some(block) = self.allocator.allocate() {
            self.block_tables
                .entry(sequence_id)
                .or_insert_with(|| BlockTable::new(self.block_size))
                .append(block);
            Some(block)
        } else {
            None
        }
    }

    pub fn free(&mut self, sequence_id: u64) {
        if let Some(table) = self.block_tables.remove(&sequence_id) {
            for block in table.get_physical_blocks() {
                self.allocator.free(*block);
            }
        }
    }

    /// Free blocks for `sequence_id` and return the number of blocks returned
    /// to the pool. Returns 0 if the sequence had no tracked blocks.
    pub fn free_returning_count(&mut self, sequence_id: u64) -> usize {
        if let Some(table) = self.block_tables.remove(&sequence_id) {
            let count = table.len();
            for block in table.get_physical_blocks() {
                self.allocator.free(*block);
            }
            count
        } else {
            0
        }
    }

    pub fn get_block_table(&self, sequence_id: u64) -> Option<&BlockTable> {
        self.block_tables.get(&sequence_id)
    }

    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        self.allocator.get_num_free_blocks() >= num_blocks
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of blocks currently held by `sequence_id`.
    pub fn blocks_for_sequence(&self, sequence_id: u64) -> usize {
        self.block_tables
            .get(&sequence_id)
            .map(|t| t.len())
            .unwrap_or(0)
    }

    /// Current free-block count in the pool (owned or shared).
    pub fn free_blocks(&self) -> usize {
        self.allocator.get_num_free_blocks()
    }

    /// Total block count in the pool (owned or shared).
    pub fn total_blocks(&self) -> usize {
        self.allocator.get_num_total_blocks()
    }
}

#[path = "block_manager_tests.rs"]
mod block_manager_tests;
