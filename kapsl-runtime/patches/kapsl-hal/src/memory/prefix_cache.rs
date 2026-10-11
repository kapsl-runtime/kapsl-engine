//! KV-block cache for prompt-prefix reuse.
//!
//! When multiple requests share the same prompt prefix (system prompt, few-shot
//! examples, document context), their KV blocks for that prefix are identical.
//! Instead of recomputing and re-allocating them per request, this cache keeps
//! those GPU blocks alive and loans them out by reference.
//!
//! # Block hashing
//!
//! Each block's identity is a chained hash:
//!
//! ```text
//! hash[0] = H(model_fingerprint, 0,         tokens[0..block_size])
//! hash[i] = H(model_fingerprint, hash[i-1], tokens[i*B..(i+1)*B])
//! ```
//!
//! Chaining means `hash[i]` implicitly encodes the entire prefix up to and
//! including block `i`, so two blocks at position `i` only match if the whole
//! preceding context is also identical. Trailing partial blocks are excluded
//! because their KV content is not yet final.
//!
//! # Multi-block entries (per-layer models)
//!
//! In llama.cpp's paged KV layout a single logical token-block position occupies
//! **one physical block per transformer layer** — all sharing the same token
//! content hash. The cache therefore stores `Vec<u32>` block IDs per entry:
//!
//! - length 1  — pure-Rust inference path (all layers packed in one block)
//! - length N  — llama.cpp path (`N = n_layers` physical blocks per position)
//!
//! Callers must be consistent: `insert` and `lookup` for the same model must
//! always use the same block count per position.
//!
//! # Ownership and refcounts
//!
//! - `refcount == 0` — no active session is using this entry; eligible for LRU
//!   eviction. The GPU blocks are retained until evicted.
//! - `refcount > 0` — one or more sessions hold this entry; it cannot be evicted.
//!
//! `insert` takes an initial `refcount`:
//! - Pass `1` when the session that just computed the KV still holds it.
//! - Pass `0` for a block group that is immediately shareable.
//!
//! `lookup` increments refcounts for every hit.
//! `release` decrements one borrow (called when a session ends or is evicted).

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use crate::gpu_arena::GpuBlockPool;

// ── Cache key ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockCacheKey {
    pub model_fingerprint: u64,
    pub block_hash: u64,
}

// ── Entry (internal) ──────────────────────────────────────────────────────────

struct PrefixCacheEntry {
    device_id: usize,
    /// Pool used to free the blocks when this entry is evicted.
    pool: Arc<GpuBlockPool>,
    /// Physical block IDs: length 1 for all-layers-in-one layout, or N for
    /// per-layer layout (one block per transformer layer).
    block_ids: Vec<u32>,
    refcount: usize,
    last_used: Instant,
}

// ── Public borrow handle ──────────────────────────────────────────────────────

/// A borrowed reference to a cached KV block group, returned by
/// [`PrefixBlockCache::lookup`].
pub struct CachedBlockRef {
    pub device_id: usize,
    pub pool: Arc<GpuBlockPool>,
    /// Physical block IDs for this logical position.
    /// `[0]` for single-block layout; `[0..n_layers]` for per-layer layout.
    pub block_ids: Vec<u32>,
    /// The chained hash that identifies this block group in the cache.
    pub block_hash: u64,
}

/// Outcome of [`PrefixBlockCache::insert`]. Ownership of the offered blocks
/// transfers to the cache only on `Inserted`; on the other variants the caller
/// retains ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixInsert {
    /// Entry stored; the cache now owns the blocks.
    Inserted,
    /// Same (fingerprint, hash) already cached — likely by another session with
    /// its own physical blocks. The caller's blocks were NOT taken.
    AlreadyPresent,
    /// Cache full and every entry has a live borrow. The caller's blocks were
    /// NOT taken.
    Rejected,
}

// ── Cache ─────────────────────────────────────────────────────────────────────

/// In-process GPU KV-block prefix cache.
///
/// Thread-safety: not `Sync` — wrap in `Mutex` when sharing across threads.
pub struct PrefixBlockCache {
    entries: HashMap<BlockCacheKey, PrefixCacheEntry>,
    capacity: usize,
}

impl PrefixBlockCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity,
        }
    }

    // ── Hashing helpers ───────────────────────────────────────────────────────

    /// Compute the chained hash for a single KV block.
    ///
    /// `prev_hash` must be `0` for the first block and equal to the hash of the
    /// immediately preceding block otherwise.
    pub fn compute_block_hash(model_fingerprint: u64, prev_hash: u64, tokens: &[u32]) -> u64 {
        let mut h = DefaultHasher::new();
        model_fingerprint.hash(&mut h);
        prev_hash.hash(&mut h);
        tokens.hash(&mut h);
        h.finish()
    }

    /// Compute per-block chained hashes for `tokens`.
    ///
    /// Only complete blocks of `block_size` tokens are hashed — the trailing
    /// partial block is excluded because its KV state is not yet final.
    pub fn compute_prefix_hashes(
        model_fingerprint: u64,
        tokens: &[u32],
        block_size: usize,
    ) -> Vec<u64> {
        let mut hashes = Vec::new();
        let mut prev = 0u64;
        for chunk in tokens.chunks(block_size) {
            if chunk.len() < block_size {
                break;
            }
            let h = Self::compute_block_hash(model_fingerprint, prev, chunk);
            hashes.push(h);
            prev = h;
        }
        hashes
    }

    /// Variant of [`compute_prefix_hashes`] for token sequences stored as
    /// `i32` (as in llama.cpp's `llama_token` / `int32_t`).
    pub fn compute_prefix_hashes_i32(
        model_fingerprint: u64,
        tokens: &[i32],
        block_size: usize,
    ) -> Vec<u64> {
        // Reinterpret i32 as u32 for hashing — only the bit pattern matters.
        let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        Self::compute_prefix_hashes(model_fingerprint, &tokens_u32, block_size)
    }

    // ── Core operations ───────────────────────────────────────────────────────

    /// Look up a contiguous prefix of cached blocks, incrementing refcounts.
    ///
    /// Stops at the first miss — downstream blocks are invalid without their
    /// full preceding context.
    pub fn lookup(&mut self, model_fingerprint: u64, hashes: &[u64]) -> Vec<CachedBlockRef> {
        let mut result = Vec::new();
        for &hash in hashes {
            let key = BlockCacheKey {
                model_fingerprint,
                block_hash: hash,
            };
            match self.entries.get_mut(&key) {
                Some(entry) => {
                    entry.refcount += 1;
                    entry.last_used = Instant::now();
                    result.push(CachedBlockRef {
                        device_id: entry.device_id,
                        pool: entry.pool.clone(),
                        block_ids: entry.block_ids.clone(),
                        block_hash: hash,
                    });
                }
                None => break,
            }
        }
        result
    }

    /// Insert a block group into the cache.
    ///
    /// `block_ids` holds all physical block IDs for this logical position (one
    /// per layer in the per-layer layout, or a single element in the packed
    /// layout).
    ///
    /// `refcount` should be `1` if the caller still actively uses the blocks,
    /// or `0` if they are immediately available for sharing.
    ///
    /// Ownership of `block_ids` transfers to the cache ONLY on
    /// [`PrefixInsert::Inserted`]. On `AlreadyPresent` and `Rejected` the caller
    /// keeps ownership and must free the blocks itself when done — the cache
    /// must never free them: callers pass blocks that are still referenced by a
    /// live sequence's block table, so freeing here puts in-use blocks back on
    /// the allocator free list (use-after-free, and a double-free when the
    /// session later releases the same blocks).
    pub fn insert(
        &mut self,
        model_fingerprint: u64,
        block_hash: u64,
        device_id: usize,
        pool: Arc<GpuBlockPool>,
        block_ids: Vec<u32>,
        refcount: usize,
    ) -> PrefixInsert {
        let key = BlockCacheKey {
            model_fingerprint,
            block_hash,
        };
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = Instant::now();
            return PrefixInsert::AlreadyPresent;
        }
        if self.entries.len() >= self.capacity && !self.evict_one_lru() {
            return PrefixInsert::Rejected;
        }
        self.entries.insert(
            key,
            PrefixCacheEntry {
                device_id,
                pool,
                block_ids,
                refcount,
                last_used: Instant::now(),
            },
        );
        PrefixInsert::Inserted
    }

    /// Release one borrow on a cached block group (decrement refcount).
    ///
    /// When refcount reaches 0 the entry is eligible for LRU eviction — the GPU
    /// blocks are retained for future requests that share the same prefix.
    pub fn release(&mut self, model_fingerprint: u64, block_hash: u64) {
        let key = BlockCacheKey {
            model_fingerprint,
            block_hash,
        };
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.refcount = entry.refcount.saturating_sub(1);
        }
    }

    // ── Eviction ──────────────────────────────────────────────────────────────

    /// Evict up to `n` zero-refcount block groups from `device_id`, oldest first.
    ///
    /// Returns `(pool, block_ids)` pairs. The caller must call
    /// `pool.free_block(id)` for each `id` in `block_ids` to return the GPU
    /// allocations.
    pub fn evict_lru_for_device(
        &mut self,
        device_id: usize,
        n: usize,
    ) -> Vec<(Arc<GpuBlockPool>, Vec<u32>)> {
        let mut candidates: Vec<(BlockCacheKey, Instant)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.device_id == device_id && e.refcount == 0)
            .map(|(k, e)| (k.clone(), e.last_used))
            .collect();
        candidates.sort_by_key(|(_, t)| *t);

        let mut freed = Vec::new();
        for (key, _) in candidates.into_iter().take(n) {
            if let Some(entry) = self.entries.remove(&key) {
                freed.push((entry.pool, entry.block_ids));
            }
        }
        freed
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    /// Evict the single least-recently-used zero-refcount entry, freeing all its
    /// GPU blocks.  Returns `false` when every entry has a live borrow.
    fn evict_one_lru(&mut self) -> bool {
        let key = self
            .entries
            .iter()
            .filter(|(_, e)| e.refcount == 0)
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, _)| k.clone());
        match key {
            Some(k) => {
                if let Some(entry) = self.entries.remove(&k) {
                    for id in entry.block_ids {
                        entry.pool.free_block(id);
                    }
                }
                true
            }
            None => false,
        }
    }
}
