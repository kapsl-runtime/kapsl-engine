//! Contiguous free-list allocation for a pool of fixed-size blocks.
//!
//! `GpuBlockPool` historically tracked free blocks with a LIFO stack, which is
//! enough for the paged KV cache (single-block grain). Sharing the pool with
//! consumers that need arbitrary-sized buffers — e.g. an ONNX Runtime device
//! allocator — additionally requires carving out *contiguous* runs of blocks,
//! since those consumers hand out flat device pointers.
//!
//! This allocator keeps the free set sorted so both grains coexist:
//! single-block allocs pop the lowest free index, run allocs first-fit scan
//! for `n` consecutive indices. It is deliberately CUDA-free so the logic is
//! unit-testable on any host.

use std::collections::BTreeSet;

#[derive(Debug)]
pub struct BlockRangeAllocator {
    total: usize,
    free: BTreeSet<u32>,
}

impl BlockRangeAllocator {
    /// Create an allocator over `total` blocks, all initially free.
    pub fn new(total: usize) -> Self {
        Self {
            total,
            free: (0..total as u32).collect(),
        }
    }

    /// Allocate a single block (lowest free index).
    pub fn alloc(&mut self) -> Option<u32> {
        self.free.pop_first()
    }

    /// Allocate `n` contiguous blocks, returning the first index (first-fit).
    pub fn alloc_run(&mut self, n: usize) -> Option<u32> {
        if n == 0 || n > self.total {
            return None;
        }
        if n == 1 {
            return self.alloc();
        }
        let n = n as u32;
        let mut run_start = u32::MAX;
        let mut run_len = 0u32;
        let mut found = None;
        for &b in &self.free {
            if run_len > 0 && b == run_start + run_len {
                run_len += 1;
            } else {
                run_start = b;
                run_len = 1;
            }
            if run_len == n {
                found = Some(run_start);
                break;
            }
        }
        let first = found?;
        for b in first..first + n {
            self.free.remove(&b);
        }
        Some(first)
    }

    /// Return a single block to the free set.
    pub fn free(&mut self, block: u32) {
        debug_assert!((block as usize) < self.total);
        let inserted = self.free.insert(block);
        debug_assert!(inserted, "double free of block {block}");
    }

    /// Return a run of `n` contiguous blocks starting at `first`.
    pub fn free_run(&mut self, first: u32, n: usize) {
        for b in first..first + n as u32 {
            self.free(b);
        }
    }

    /// Number of free blocks.
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Total number of blocks managed.
    pub fn total(&self) -> usize {
        self.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_all_free() {
        let a = BlockRangeAllocator::new(8);
        assert_eq!(a.free_count(), 8);
        assert_eq!(a.total(), 8);
    }

    #[test]
    fn alloc_returns_lowest() {
        let mut a = BlockRangeAllocator::new(8);
        assert_eq!(a.alloc(), Some(0));
        assert_eq!(a.alloc(), Some(1));
        assert_eq!(a.free_count(), 6);
    }

    #[test]
    fn alloc_exhausted_returns_none() {
        let mut a = BlockRangeAllocator::new(2);
        assert!(a.alloc().is_some());
        assert!(a.alloc().is_some());
        assert_eq!(a.alloc(), None);
    }

    #[test]
    fn free_makes_block_reusable() {
        let mut a = BlockRangeAllocator::new(2);
        let b = a.alloc().unwrap();
        a.free(b);
        assert_eq!(a.free_count(), 2);
        assert_eq!(a.alloc(), Some(b));
    }

    #[test]
    fn alloc_run_contiguous() {
        let mut a = BlockRangeAllocator::new(8);
        let first = a.alloc_run(3).unwrap();
        assert_eq!(first, 0);
        assert_eq!(a.free_count(), 5);
        // Next run starts after the previous one.
        assert_eq!(a.alloc_run(2), Some(3));
    }

    #[test]
    fn alloc_run_skips_fragmented_gaps() {
        let mut a = BlockRangeAllocator::new(8);
        // Occupy blocks 0..4, then free only 1 and 3: free = {1, 3, 4, 5, 6, 7}.
        let first = a.alloc_run(4).unwrap();
        assert_eq!(first, 0);
        a.free(1);
        a.free(3);
        // A 3-run can't use the lone block 1; first fit is 3..6.
        assert_eq!(a.alloc_run(3), Some(3));
        assert_eq!(a.free_count(), 3);
    }

    #[test]
    fn alloc_run_zero_and_oversize() {
        let mut a = BlockRangeAllocator::new(4);
        assert_eq!(a.alloc_run(0), None);
        assert_eq!(a.alloc_run(5), None);
        assert_eq!(a.free_count(), 4);
    }

    #[test]
    fn alloc_run_no_contiguous_space() {
        let mut a = BlockRangeAllocator::new(4);
        // Allocate everything, free alternating blocks: free = {0, 2}.
        assert_eq!(a.alloc_run(4), Some(0));
        a.free(0);
        a.free(2);
        assert_eq!(a.alloc_run(2), None);
        assert_eq!(a.free_count(), 2);
    }

    #[test]
    fn free_run_roundtrip() {
        let mut a = BlockRangeAllocator::new(8);
        let first = a.alloc_run(5).unwrap();
        a.free_run(first, 5);
        assert_eq!(a.free_count(), 8);
        assert_eq!(a.alloc_run(8), Some(0));
    }

    #[test]
    fn singles_and_runs_share_free_set() {
        let mut a = BlockRangeAllocator::new(8);
        let s = a.alloc().unwrap(); // takes block 0
        let run = a.alloc_run(4).unwrap(); // 1..5
        assert_eq!(run, 1);
        a.free(s);
        // 0 is free again but 0..4 isn't contiguous with the run held; a
        // 4-run must come from 5..8 plus... only {0,5,6,7} free → no 4-run.
        assert_eq!(a.alloc_run(4), None);
        a.free_run(run, 4);
        assert_eq!(a.alloc_run(4), Some(0));
    }

    #[test]
    fn whole_pool_run() {
        let mut a = BlockRangeAllocator::new(16);
        assert_eq!(a.alloc_run(16), Some(0));
        assert_eq!(a.free_count(), 0);
        a.free_run(0, 16);
        assert_eq!(a.free_count(), 16);
    }
}
