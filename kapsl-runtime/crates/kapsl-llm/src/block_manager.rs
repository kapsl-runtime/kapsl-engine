use std::collections::{HashMap, VecDeque};

/// Represents a physical block of memory in the KV cache
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalTokenBlock {
    pub block_number: usize,
    pub block_size: usize,
    pub device_id: usize,
}

/// Manages the allocation of physical blocks
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

/// Manages blocks for all sequences
pub struct BlockManager {
    allocator: BlockAllocator,
    block_tables: HashMap<u64, BlockTable>, // sequence_id -> BlockTable
    block_size: usize,
}

impl BlockManager {
    pub fn new(total_blocks: usize, block_size: usize, device_id: usize) -> Self {
        Self {
            allocator: BlockAllocator::new(total_blocks, block_size, device_id),
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

    pub fn get_block_table(&self, sequence_id: u64) -> Option<&BlockTable> {
        self.block_tables.get(&sequence_id)
    }

    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        self.allocator.get_num_free_blocks() >= num_blocks
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

#[path = "block_manager_tests.rs"]
mod block_manager_tests;
