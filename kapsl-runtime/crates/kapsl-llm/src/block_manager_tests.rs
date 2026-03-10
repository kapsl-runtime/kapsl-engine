#[cfg(test)]
mod tests {
    use super::super::{BlockAllocator, BlockManager, PhysicalTokenBlock};

    #[test]
    fn allocator_allocates_and_frees_blocks() {
        let mut allocator = BlockAllocator::new(2, 16, 7);
        assert_eq!(allocator.get_num_total_blocks(), 2);
        assert_eq!(allocator.get_num_free_blocks(), 2);

        let first = allocator.allocate().expect("expected first block");
        assert_eq!(
            first,
            PhysicalTokenBlock {
                block_number: 0,
                block_size: 16,
                device_id: 7,
            }
        );
        let second = allocator.allocate().expect("expected second block");
        assert_eq!(second.block_number, 1);
        assert_eq!(allocator.get_num_free_blocks(), 0);
        assert!(allocator.allocate().is_none());

        allocator.free(first);
        assert_eq!(allocator.get_num_free_blocks(), 1);
        let recycled = allocator.allocate().expect("expected recycled block");
        assert_eq!(recycled.block_number, 0);
    }

    #[test]
    fn block_manager_tracks_block_tables_and_capacity() {
        let mut manager = BlockManager::new(3, 16, 0);
        assert!(manager.can_allocate(3));

        manager.allocate(10).expect("block for seq 10");
        manager.allocate(10).expect("second block for seq 10");
        manager.allocate(11).expect("block for seq 11");

        let table_10 = manager.get_block_table(10).expect("table for seq 10");
        assert_eq!(table_10.len(), 2);
        let table_11 = manager.get_block_table(11).expect("table for seq 11");
        assert_eq!(table_11.len(), 1);
        assert!(!manager.can_allocate(1));

        manager.free(10);
        assert!(manager.get_block_table(10).is_none());
        assert!(manager.can_allocate(2));
    }
}
