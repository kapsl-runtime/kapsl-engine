#[cfg(test)]
mod tests {
    use super::super::{KvCache, KvCacheConfig, KvCacheError, KvCacheMode, KvEvictionPolicy};
    use half::f16;

    fn idx(head: usize, pos: usize, dim: usize, max_seq_len: usize, head_dim: usize) -> usize {
        (head * max_seq_len * head_dim) + (pos * head_dim) + dim
    }

    #[test]
    fn allocate_sequence_is_idempotent() {
        let mut cache = KvCache::new(2, 2, 4, 3);
        cache.allocate_sequence(7, &[]).unwrap();
        cache.allocate_sequence(7, &[]).unwrap();
        assert!(cache.has_sequence(7));
        assert_eq!(cache.sequence_length(7), Some(0));
    }

    #[test]
    fn append_token_pads_or_truncates_inputs() {
        let mut cache = KvCache::new(1, 2, 4, 3);
        cache.allocate_sequence(1, &[]).unwrap();

        let key = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let value = vec![
            f16::from_f32(10.0),
            f16::from_f32(11.0),
            f16::from_f32(12.0),
            f16::from_f32(13.0),
            f16::from_f32(14.0),
            f16::from_f32(15.0),
            f16::from_f32(99.0),
        ];

        cache.append_token(1, 0, 0, &key, &value, None).unwrap();
        cache.advance_sequence(1);

        let view = cache.get_layer_view(1, 0).expect("view");
        // get_layer_view returns a compact [num_heads, length, head_dim] buffer;
        // use view.length (= 1) as the stride, not max_seq_len.
        let seq_len = view.length;
        let head_dim = 3;

        assert_eq!(view.key[idx(0, 0, 0, seq_len, head_dim)].to_f32(), 1.0);
        assert_eq!(view.key[idx(0, 0, 2, seq_len, head_dim)].to_f32(), 3.0);
        assert_eq!(view.key[idx(1, 0, 0, seq_len, head_dim)].to_f32(), 4.0);
        assert_eq!(view.key[idx(1, 0, 1, seq_len, head_dim)].to_f32(), 0.0);
        assert_eq!(view.key[idx(1, 0, 2, seq_len, head_dim)].to_f32(), 0.0);

        assert_eq!(view.value[idx(1, 0, 2, seq_len, head_dim)].to_f32(), 15.0);
    }

    #[test]
    fn append_head_range_and_get_layer_as_onnx() {
        let mut cache = KvCache::new(1, 1, 4, 2);
        cache.allocate_sequence(1, &[]).unwrap();

        let key_values = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        cache
            .append_head_range_seq_first(1, 0, 0, 0, &key_values, &key_values)
            .unwrap();
        cache.advance_sequence_by(1, 2);

        let array = cache.get_layer_as_onnx(1, 0).expect("onnx view");
        assert_eq!(array.shape(), &[1, 1, 2, 2]);
        let got: Vec<f32> = array.iter().map(|v| v.to_f32()).collect();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn paged_cache_packs_across_blocks() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Paged,
            block_size: 2,
            total_blocks: 4,
            eviction_policy: KvEvictionPolicy::None,
            dense_free_list_cap: 8,
            initial_seq_len: 256,
        };
        let mut cache = KvCache::new_with_config(1, 1, 8, 2, config);
        cache.allocate_sequence(1, &[]).unwrap();

        let mut expected = Vec::new();
        for pos in 0..4 {
            let key = vec![
                f16::from_f32((pos * 10 + 1) as f32),
                f16::from_f32((pos * 10 + 2) as f32),
            ];
            let value = vec![
                f16::from_f32((pos * 20 + 1) as f32),
                f16::from_f32((pos * 20 + 2) as f32),
            ];
            cache.append_token(1, 0, pos, &key, &value, None).unwrap();
            expected.extend_from_slice(&key);
        }
        cache.advance_sequence_by(1, 4);

        let view = cache.get_layer_view(1, 0).expect("view");
        assert_eq!(view.length, 4);
        let got: Vec<f32> = view.key.iter().map(|v| v.to_f32()).collect();
        let expected_f32: Vec<f32> = expected.iter().map(|v| v.to_f32()).collect();
        assert_eq!(got, expected_f32);
    }

    #[test]
    fn paged_cache_eviction_lru_inactive() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Paged,
            block_size: 2,
            total_blocks: 1,
            eviction_policy: KvEvictionPolicy::LruInactive,
            dense_free_list_cap: 8,
            initial_seq_len: 256,
        };
        let mut cache = KvCache::new_with_config(1, 1, 4, 1, config);
        cache.allocate_sequence(1, &[]).unwrap();
        cache
            .append_token(1, 0, 0, &[f16::from_f32(1.0)], &[f16::from_f32(2.0)], None)
            .unwrap();
        cache.advance_sequence(1);

        cache.allocate_sequence(2, &[]).unwrap();
        cache.set_active_sequences(&[1]);
        let err = cache
            .append_token(2, 0, 0, &[f16::from_f32(3.0)], &[f16::from_f32(4.0)], None)
            .expect_err("expected out of blocks");
        assert!(matches!(err, KvCacheError::OutOfBlocks));

        cache.clear_active_sequences();
        cache.set_active_sequences(&[2]);
        cache
            .append_token(2, 0, 0, &[f16::from_f32(5.0)], &[f16::from_f32(6.0)], None)
            .unwrap();
        let evicted = cache.drain_evicted_sequences();
        assert_eq!(evicted, vec![1]);
        assert!(!cache.has_sequence(1));
    }

    #[test]
    fn paged_cache_eviction_fifo() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Paged,
            block_size: 2,
            total_blocks: 1,
            eviction_policy: KvEvictionPolicy::Fifo,
            dense_free_list_cap: 8,
            initial_seq_len: 256,
        };
        let mut cache = KvCache::new_with_config(1, 1, 4, 1, config);

        // Sequence 1: Allocated first
        cache.allocate_sequence(1, &[]).unwrap();
        cache
            .append_token(1, 0, 0, &[f16::from_f32(1.0)], &[f16::from_f32(2.0)], None)
            .unwrap();
        cache.advance_sequence(1);

        // Sequence 2: Allocated second
        cache.allocate_sequence(2, &[]).unwrap();

        // Sequence 1 is inactive, Sequence 2 is active
        cache.set_active_sequences(&[2]);

        // Try appending to global new seq (3)? No, just try appending to 2.
        // But we need more blocks. Total blocks = 1.
        // Seq 1 owns the block.
        // We want to force eviction of Seq 1.

        // Append to Seq 2 requires allocation
        cache
            .append_token(2, 0, 0, &[f16::from_f32(3.0)], &[f16::from_f32(4.0)], None)
            .unwrap();

        // Expected: Seq 1 evicted (oldest).
        let evicted = cache.drain_evicted_sequences();
        assert_eq!(evicted, vec![1]);
        assert!(!cache.has_sequence(1));
    }

    #[test]
    fn paged_cache_rollback() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Paged,
            block_size: 2,
            total_blocks: 4,
            eviction_policy: KvEvictionPolicy::None,
            dense_free_list_cap: 8,
            initial_seq_len: 256,
        };
        let mut cache = KvCache::new_with_config(1, 1, 8, 2, config);

        cache.allocate_sequence(1, &[]).unwrap();

        // Append 3 tokens: Block 0 (tokens 0, 1), Block 1 (token 2)
        // Tokens: 0, 1, 2
        for i in 0..3 {
            let key = vec![f16::from_f32(i as f32); 2];
            let val = vec![f16::from_f32(i as f32); 2];
            cache.append_token(1, 0, i, &key, &val, None).unwrap();
        }
        cache.advance_sequence_by(1, 3);

        // Verify current length
        assert_eq!(cache.sequence_length(1), Some(3));

        // Check blocks usage: should be 2 blocks
        // We can check stats logic indirectly? Or internal state if accessible?
        // Let's assume correct if appending more works.

        // Rollback to 1. Should free Block 1.
        cache.rollback_sequence(1, 1);
        assert_eq!(cache.sequence_length(1), Some(1));

        // Append loop should overwrite
        let key = vec![f16::from_f32(10.0); 2];
        let val = vec![f16::from_f32(10.0); 2];
        cache.append_token(1, 0, 1, &key, &val, None).unwrap(); // Pos 1

        cache.advance_sequence(1);

        // Read back
        let view = cache.get_layer_view(1, 0).expect("view");
        assert_eq!(view.length, 2);
        // Token 0: 0.0, Token 1: 10.0
        assert_eq!(view.key[0].to_f32(), 0.0);
        assert_eq!(view.key[2].to_f32(), 10.0);
    }

    #[test]
    fn paged_cache_prefix_reuse() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Paged,
            block_size: 2,
            total_blocks: 4,
            eviction_policy: KvEvictionPolicy::None,
            dense_free_list_cap: 8,
            initial_seq_len: 256,
        };
        let mut cache = KvCache::new_with_config(1, 1, 8, 2, config);

        let prefix = [100, 101, 102, 103]; // 2 blocks worth (size 2)

        // Sequence 1: Allocate and fill
        cache.allocate_sequence(1, &[]).unwrap();
        for (i, &token) in prefix.iter().enumerate() {
            let key = vec![f16::from_f32(token as f32); 2];
            let val = vec![f16::from_f32(token as f32); 2];
            cache
                .append_token(1, 0, i, &key, &val, Some(token as u64))
                .unwrap();
        }
        cache.advance_sequence_by(1, 4);

        // Verify stats: 2 blocks used
        assert_eq!(cache.stats().blocks_free, 2);

        // Sequence 2: Allocate with SAME prefix
        // Should reuse the 2 blocks
        let cached_len = cache.allocate_sequence(2, &prefix).unwrap();
        assert_eq!(cached_len, 3); // Capped to tokens.len() - 1 to force processing of last token

        // Verify stats: Still 2 blocks free! (Reused)
        assert_eq!(cache.stats().blocks_free, 2);

        // Append one more token to Seq 2 (new block)
        let key = vec![f16::from_f32(200.0); 2];
        let val = vec![f16::from_f32(200.0); 2];
        cache.append_token(2, 0, 4, &key, &val, Some(200)).unwrap();

        // Verify stats: 1 block free (New block allocated)
        assert_eq!(cache.stats().blocks_free, 1);
    }

    #[test]
    fn paged_cache_prefix_reuse_does_not_leak_refs() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Paged,
            block_size: 2,
            total_blocks: 2,
            eviction_policy: KvEvictionPolicy::None,
            dense_free_list_cap: 8,
            initial_seq_len: 256,
        };
        let mut cache = KvCache::new_with_config(1, 1, 8, 2, config);
        let prefix = [100, 101, 102, 103];

        cache.allocate_sequence(1, &[]).unwrap();
        for (i, &token) in prefix.iter().enumerate() {
            let key = vec![f16::from_f32(token as f32); 2];
            let val = vec![f16::from_f32(token as f32); 2];
            cache
                .append_token(1, 0, i, &key, &val, Some(token as u64))
                .unwrap();
        }
        cache.advance_sequence_by(1, 4);

        let cached_len = cache.allocate_sequence(2, &prefix).unwrap();
        assert_eq!(cached_len, 3);
        cache.remove_sequence(1);
        cache.remove_sequence(2);

        let stats = cache.stats();
        assert_eq!(stats.blocks_total, 2);
        assert_eq!(stats.blocks_free, 2);
    }

    #[test]
    fn paged_cache_stale_radix_entry_is_ignored_after_remove() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Paged,
            block_size: 2,
            total_blocks: 1,
            eviction_policy: KvEvictionPolicy::None,
            dense_free_list_cap: 8,
            initial_seq_len: 256,
        };
        let mut cache = KvCache::new_with_config(1, 1, 8, 2, config);
        let prefix = [7, 8];

        cache.allocate_sequence(1, &[]).unwrap();
        for (i, &token) in prefix.iter().enumerate() {
            let key = vec![f16::from_f32(token as f32); 2];
            let val = vec![f16::from_f32(token as f32); 2];
            cache
                .append_token(1, 0, i, &key, &val, Some(token as u64))
                .unwrap();
        }
        cache.advance_sequence_by(1, 2);
        cache.remove_sequence(1);

        let cached_len = cache.allocate_sequence(2, &prefix).unwrap();
        assert_eq!(cached_len, 0);

        cache
            .append_token(
                2,
                0,
                0,
                &[f16::from_f32(1.0), f16::from_f32(2.0)],
                &[f16::from_f32(1.0), f16::from_f32(2.0)],
                Some(1),
            )
            .unwrap();
    }

    #[test]
    fn paged_cache_rollback_clears_full_block_prefix() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Paged,
            block_size: 2,
            total_blocks: 2,
            eviction_policy: KvEvictionPolicy::None,
            dense_free_list_cap: 8,
            initial_seq_len: 256,
        };
        let mut cache = KvCache::new_with_config(1, 1, 8, 2, config);
        let prefix = [41, 42];

        cache.allocate_sequence(1, &[]).unwrap();
        for (i, &token) in prefix.iter().enumerate() {
            let key = vec![f16::from_f32(token as f32); 2];
            let val = vec![f16::from_f32(token as f32); 2];
            cache
                .append_token(1, 0, i, &key, &val, Some(token as u64))
                .unwrap();
        }
        cache.advance_sequence_by(1, 2);
        cache.rollback_sequence(1, 1);

        let cached_len = cache.allocate_sequence(2, &prefix).unwrap();
        assert_eq!(cached_len, 0);
    }
}
