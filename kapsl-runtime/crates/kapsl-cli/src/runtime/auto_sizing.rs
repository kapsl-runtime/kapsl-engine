//! Automatic scheduler and GGUF prefill sizing policies.

use super::{available_ram_mb, largest_model_size_mb, logical_cpu_cores};
use std::path::PathBuf;

pub(crate) struct AutoTunedPolicy {
    pub(crate) batch_size: usize,
    pub(crate) scheduler_max_micro_batch: usize,
    pub(crate) scheduler_queue_delay_ms: u64,
    pub(crate) scheduler_queue_size: usize,
    pub(crate) rationale: String,
}

pub(crate) fn round_down_power_of_two(value: u64) -> u64 {
    if value <= 1 {
        return 1;
    }
    1_u64 << (u64::BITS - 1 - value.leading_zeros())
}

pub(crate) fn auto_tuned_gguf_prefill_chunk_size(
    model_size_mb: u64,
    available_ram_mb: u64,
    batch_size: usize,
) -> Option<usize> {
    if model_size_mb == 0 {
        return None;
    }

    const MIN_CHUNK: usize = 32;
    const MAX_CHUNK: usize = 512;

    let estimated_loaded_model_mb = model_size_mb.saturating_mul(5) / 4;
    let runtime_guard_mb = 1024;
    let model_headroom_mb =
        available_ram_mb.saturating_sub(estimated_loaded_model_mb + runtime_guard_mb);

    let scratch_budget_mb = if model_headroom_mb > 0 {
        (model_headroom_mb / 2).min(available_ram_mb / 4).max(128)
    } else if available_ram_mb > 0 && available_ram_mb < estimated_loaded_model_mb {
        128
    } else {
        available_ram_mb / 8
    };

    let estimated_scratch_per_token_mb = (model_size_mb / 256).clamp(4, 64);
    let concurrency_divisor = batch_size.max(1).div_ceil(2).clamp(1, 4) as u64;
    let raw_chunk = (scratch_budget_mb / concurrency_divisor) / estimated_scratch_per_token_mb;
    let clamped_raw = raw_chunk.max(MIN_CHUNK as u64).min(MAX_CHUNK as u64);
    let chunk = round_down_power_of_two(clamped_raw).max(MIN_CHUNK as u64) as usize;

    Some(chunk)
}

/// Derive scheduler parameters from model file size and available system resources.
///
/// When multiple models are loaded, sizing uses the largest model. The policy
/// falls back to conservative defaults when none of the paths can be read.
pub(crate) fn auto_tune_policy(model_paths: &[PathBuf]) -> AutoTunedPolicy {
    let model_size_mb = largest_model_size_mb(model_paths);
    let available_ram_mb = available_ram_mb();
    let available_ram_gb = available_ram_mb / 1024;
    let cpu_cores = logical_cpu_cores();

    let (mut batch_size, mut micro_batch, delay_ms, mut queue_size, size_tier) =
        if model_size_mb == 0 {
            return AutoTunedPolicy {
                batch_size: 4,
                scheduler_max_micro_batch: 4,
                scheduler_queue_delay_ms: 2,
                scheduler_queue_size: 256,
                rationale: format!(
                    "model={}MB (unknown), ram_avail={}GB, cpu_cores={}, conservative-defaults",
                    model_size_mb, available_ram_gb, cpu_cores
                ),
            };
        } else if model_size_mb < 500 {
            (16, 16, 6, 2048, "tiny (<500 MB)")
        } else if model_size_mb < 2_000 {
            (8, 8, 3, 512, "small (500 MB-2 GB)")
        } else if model_size_mb < 8_000 {
            (4, 4, 2, 256, "medium (2-8 GB)")
        } else {
            (2, 2, 1, 128, "large (>=8 GB)")
        };

    let mut notes = String::new();

    if available_ram_gb > 0 && available_ram_gb < 4 {
        batch_size = (batch_size / 2).max(1);
        queue_size = (queue_size / 2).max(64);
        notes.push_str(&format!(", low-ram (avail={}GB)", available_ram_gb));
    }

    if cpu_cores <= 2 {
        micro_batch = (micro_batch / 2).max(1);
        notes.push_str(&format!(", low-cpu (cores={})", cpu_cores));
    }

    AutoTunedPolicy {
        batch_size,
        scheduler_max_micro_batch: micro_batch,
        scheduler_queue_delay_ms: delay_ms,
        scheduler_queue_size: queue_size,
        rationale: format!(
            "model={}MB ({}), ram_avail={}GB, cpu_cores={}{}",
            model_size_mb, size_tier, available_ram_gb, cpu_cores, notes
        ),
    }
}
