#[cfg(test)]
mod tests {
    use super::super::{ShmClassBudget, ShmPoolAllocator, SimpleShmAllocator, TieredShmAllocator};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_try_allocate_rejects_oversized_payload() {
        let allocator = SimpleShmAllocator::new(100, 64, 3);
        assert!(allocator.try_allocate(65).is_none());
    }

    #[test]
    fn test_try_allocate_returns_none_when_pool_is_full() {
        let allocator = SimpleShmAllocator::new(100, 64, 2);

        assert!(allocator.try_allocate(16).is_some());
        assert!(allocator.try_allocate(16).is_some());
        assert!(allocator.try_allocate(16).is_none());
    }

    #[test]
    fn test_release_makes_slot_available_immediately() {
        let allocator = SimpleShmAllocator::new(100, 64, 1);
        let first = allocator.try_allocate(32).expect("first slot");
        assert!(allocator.try_allocate(32).is_none());

        assert!(allocator.release(first));
        let second = allocator
            .try_allocate(32)
            .expect("released slot should be reusable");
        assert_eq!(first, second);
    }

    #[test]
    fn test_expired_lease_is_reused() {
        let allocator = SimpleShmAllocator::new_with_ttl(256, 32, 1, Duration::from_millis(20));
        let first = allocator.try_allocate(8).expect("slot");
        assert!(allocator.try_allocate(8).is_none());
        thread::sleep(Duration::from_millis(30));
        let second = allocator
            .try_allocate(8)
            .expect("expired lease should be reusable");
        assert_eq!(first, second);
    }

    #[test]
    fn test_release_rejects_invalid_offsets() {
        let allocator = SimpleShmAllocator::new(100, 64, 2);
        assert!(!allocator.release(99));
        assert!(!allocator.release(101));
        assert!(!allocator.release(100 + 2 * 64));
    }

    #[test]
    fn test_snapshot_reports_in_use_and_oldest_lease() {
        let allocator = SimpleShmAllocator::new_with_ttl(100, 64, 2, Duration::from_millis(100));
        let a = allocator.try_allocate(16).expect("slot a");
        thread::sleep(Duration::from_millis(10));
        let _b = allocator.try_allocate(16).expect("slot b");

        let snap = allocator.snapshot();
        assert_eq!(snap.in_use_slots, 2);
        assert!(snap.oldest_lease_ms >= 5);

        assert!(allocator.release(a));
        let snap2 = allocator.snapshot();
        assert_eq!(snap2.in_use_slots, 1);
    }

    #[test]
    fn test_tiered_allocator_chooses_smallest_fitting_class() {
        let allocator = TieredShmAllocator::new_with_class_sizes(
            1024,
            16 * 1024 * 1024,
            &[64, 256, 1024],
            Duration::from_secs(1),
        );
        let classes = allocator.classes();
        assert!(classes.len() >= 2);

        let small = allocator
            .try_allocate(32)
            .expect("small payload should fit smallest class");
        let medium = allocator
            .try_allocate(200)
            .expect("medium payload should fit medium class");

        let small_class = classes
            .iter()
            .find(|c| small >= c.base_offset && small < c.base_offset + c.bytes)
            .expect("small offset must belong to one class");
        let medium_class = classes
            .iter()
            .find(|c| medium >= c.base_offset && medium < c.base_offset + c.bytes)
            .expect("medium offset must belong to one class");

        assert!(small_class.slot_size <= medium_class.slot_size);
    }

    #[test]
    fn test_tiered_allocator_release_roundtrip() {
        let allocator = TieredShmAllocator::new_with_class_sizes(
            4096,
            8 * 1024 * 1024,
            &[128, 512],
            Duration::from_secs(1),
        );
        let classes = allocator.classes();
        let off = allocator
            .try_allocate(100)
            .expect("expected allocation in first class");
        assert!(allocator.release(off));
        let off2 = allocator
            .try_allocate(100)
            .expect("released slot should be reusable");
        let class_slot_size = |offset: usize| {
            classes
                .iter()
                .find(|c| offset >= c.base_offset && offset < c.base_offset + c.bytes)
                .map(|c| c.slot_size)
        };
        assert_eq!(class_slot_size(off), class_slot_size(off2));
    }

    #[test]
    fn test_tiered_allocator_snapshot_aggregates_classes() {
        let allocator = TieredShmAllocator::new_with_class_sizes(
            4096,
            8 * 1024 * 1024,
            &[64, 256],
            Duration::from_secs(5),
        );
        let _a = allocator.try_allocate(32).expect("slot a");
        let _b = allocator.try_allocate(200).expect("slot b");
        let snap = allocator.snapshot();
        assert_eq!(snap.in_use_slots, 2);
        assert!(snap.oldest_lease_ms < 5_000);
    }

    #[test]
    fn test_tiered_allocator_weighted_budgets_bias_capacity() {
        let budgets = [
            ShmClassBudget {
                slot_size: 256,
                weight: 1,
            },
            ShmClassBudget {
                slot_size: 4096,
                weight: 8,
            },
        ];
        let allocator = TieredShmAllocator::new_with_class_budgets(
            0,
            2 * 1024 * 1024,
            &budgets,
            Duration::from_secs(1),
        );
        let classes = allocator.classes();
        let small = classes
            .iter()
            .find(|c| c.slot_size == 256)
            .expect("small class must exist");
        let large = classes
            .iter()
            .find(|c| c.slot_size == 4096)
            .expect("large class must exist");
        assert!(large.bytes > small.bytes);
    }
}
