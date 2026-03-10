#[derive(Default, Clone)]
pub struct LLMMetrics {
    pub total_inference_time: f64,
    pub kv_cache_bytes_used: usize,
    pub kv_cache_bytes_capacity: usize,
    pub kv_cache_blocks_total: usize,
    pub kv_cache_blocks_free: usize,
    pub kv_cache_sequences: usize,
    pub kv_cache_evicted_blocks: u64,
    pub kv_cache_evicted_sequences: u64,
    pub kv_cache_packed_layers: usize,
}
