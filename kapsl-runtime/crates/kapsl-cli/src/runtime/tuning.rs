use super::*;

pub(crate) fn optional_env_var(name: &str) -> Option<String> {
    let value = std::env::var(name).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

pub(crate) fn optional_env_var_alias(primary: &str, legacy: &str) -> Option<String> {
    optional_env_var(primary).or_else(|| optional_env_var(legacy))
}

/// Resolve the per-device VRAM cap (bytes) for cooperative software-vGPU
/// self-limiting, reading the environment so a HAMi-capped pod self-configures
/// with zero model metadata. Returns `None` when nothing is configured (every
/// clamp built on it is then a no-op, so default behavior is unchanged).
///
/// Priority order: HAMi's per-device `CUDA_DEVICE_MEMORY_LIMIT_<id>`, then its
/// process-wide `CUDA_DEVICE_MEMORY_LIMIT`, then the kapsl alias
/// `KAPSL_GPU_MEMORY_LIMIT_MB`. The model-metadata path
/// (`HardwareRequirements::gpu_memory_limit_mb`) is intentionally routed through
/// this same helper later; today it is env-only, which covers the HAMi case.
pub(crate) fn device_vram_cap_bytes(device_id: usize) -> Option<usize> {
    resolve_vram_cap_bytes(
        optional_env_var(&format!("{CUDA_DEVICE_MEMORY_LIMIT_ENV}_{device_id}")),
        optional_env_var(CUDA_DEVICE_MEMORY_LIMIT_ENV),
        optional_env_var(KAPSL_GPU_MEMORY_LIMIT_MB_ENV),
    )
}

/// Pure resolution of the cap-source priority, split out from environment access
/// so it is unit-testable without touching process-global state. A malformed
/// higher-priority value falls through to the next source rather than failing.
fn resolve_vram_cap_bytes(
    per_device: Option<String>,
    process_wide: Option<String>,
    kapsl_mb: Option<String>,
) -> Option<usize> {
    per_device
        .as_deref()
        .and_then(parse_cuda_memory_limit)
        .or_else(|| process_wide.as_deref().and_then(parse_cuda_memory_limit))
        .or_else(|| {
            kapsl_mb
                .as_deref()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|mb| *mb > 0)
                .map(|mb| mb.saturating_mul(1024 * 1024))
        })
}

/// Parse a HAMi `CUDA_DEVICE_MEMORY_LIMIT` value into bytes. Accepts a plain
/// byte count or a value with a binary unit suffix (`k`/`m`/`g`, optionally
/// followed by `b`, case-insensitive — e.g. `2560m`, `4g`, `8gb`). Returns
/// `None` for empty, zero, or otherwise malformed input.
fn parse_cuda_memory_limit(value: &str) -> Option<usize> {
    let lowered = value.trim().to_ascii_lowercase();
    if lowered.is_empty() {
        return None;
    }
    let digits = lowered.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    let multiplier: usize = match &lowered[digits.len()..] {
        "" | "b" => 1,
        "k" | "kb" => 1024,
        "m" | "mb" => 1024 * 1024,
        "g" | "gb" => 1024 * 1024 * 1024,
        _ => return None,
    };
    digits
        .parse::<usize>()
        .ok()
        .filter(|amount| *amount > 0)
        .map(|amount| amount.saturating_mul(multiplier))
}

pub(crate) fn resolve_model_load_parallelism(model_count: usize) -> usize {
    if model_count <= 1 {
        return 1;
    }
    optional_env_var(MODEL_LOAD_PARALLELISM_ENV)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| model_count.min(4))
        .clamp(1, model_count)
}

pub(crate) fn arg_user_supplied(matches: &ArgMatches, arg: &str) -> bool {
    !matches!(
        matches.value_source(arg),
        None | Some(ValueSource::DefaultValue)
    )
}

pub(crate) struct AutoTunedPolicy {
    pub(crate) batch_size: usize,
    pub(crate) scheduler_max_micro_batch: usize,
    pub(crate) scheduler_queue_delay_ms: u64,
    pub(crate) scheduler_queue_size: usize,
    pub(crate) rationale: String,
}

pub(crate) fn largest_model_size_mb(model_paths: &[PathBuf]) -> u64 {
    model_paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .max()
        .map(|b| b / (1024 * 1024))
        .unwrap_or(0)
}

pub(crate) fn available_ram_mb() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory() / (1024 * 1024)
}

pub(crate) fn logical_cpu_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
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
    let concurrency_divisor = ((batch_size.max(1) + 1) / 2).clamp(1, 4) as u64;
    let raw_chunk = (scratch_budget_mb / concurrency_divisor) / estimated_scratch_per_token_mb;
    let clamped_raw = raw_chunk.max(MIN_CHUNK as u64).min(MAX_CHUNK as u64);
    let chunk = round_down_power_of_two(clamped_raw).max(MIN_CHUNK as u64) as usize;

    Some(chunk)
}

/// Derive scheduler parameters from model file size and available system resources.
///
/// When multiple models are loaded, sizes based on the largest (conservative).
/// Falls back to safe defaults if model paths can't be stat'd.
pub(crate) fn auto_tune_policy(model_paths: &[PathBuf]) -> AutoTunedPolicy {
    // Largest model file size in MB — proxy for parameter count / memory footprint
    let model_size_mb = largest_model_size_mb(model_paths);

    // Available (not total) system RAM in GB
    let available_ram_mb = available_ram_mb();
    let available_ram_gb = available_ram_mb / 1024;

    // Logical CPU cores
    let cpu_cores = logical_cpu_cores();

    // Base policy from model file size
    let (mut batch_size, mut micro_batch, delay_ms, mut queue_size, size_tier) =
        if model_size_mb == 0 {
            // Unknown model size should remain deterministic across environments.
            // Keep the conservative Standard-equivalent defaults unchanged rather
            // than layering in host-dependent RAM/CPU reductions.
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

    // Memory pressure: halve batch and queue if available RAM is tight
    if available_ram_gb > 0 && available_ram_gb < 4 {
        batch_size = (batch_size / 2).max(1);
        queue_size = (queue_size / 2).max(64);
        notes.push_str(&format!(", low-ram (avail={}GB)", available_ram_gb));
    }

    // CPU constraint: halve micro_batch on very low-core systems
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

pub(crate) fn apply_performance_profile(
    args: &mut Args,
    matches: &ArgMatches,
) -> AppliedPerformanceTuning {
    let mut tuning = AppliedPerformanceTuning::default();
    if args.worker || matches!(args.performance_profile, PerformanceProfile::Standard) {
        return tuning;
    }

    let batch_explicit = arg_user_supplied(matches, "batch_size");
    let transport_explicit = arg_user_supplied(matches, "transport");
    let scheduler_queue_size_explicit = arg_user_supplied(matches, "scheduler_queue_size");
    let scheduler_max_micro_batch_explicit =
        arg_user_supplied(matches, "scheduler_max_micro_batch");
    let scheduler_queue_delay_ms_explicit = arg_user_supplied(matches, "scheduler_queue_delay_ms");

    match args.performance_profile {
        PerformanceProfile::Standard => {}
        PerformanceProfile::Auto => {
            let policy = auto_tune_policy(&args.model);
            if !batch_explicit {
                args.batch_size = policy.batch_size;
                tuning.batch_size = Some(args.batch_size);
            }
            if !scheduler_queue_size_explicit {
                args.scheduler_queue_size = policy.scheduler_queue_size;
                tuning.scheduler_queue_size = Some(args.scheduler_queue_size);
            }
            if !scheduler_max_micro_batch_explicit {
                args.scheduler_max_micro_batch = policy.scheduler_max_micro_batch;
                tuning.scheduler_max_micro_batch = Some(args.scheduler_max_micro_batch);
            }
            if !scheduler_queue_delay_ms_explicit {
                args.scheduler_queue_delay_ms = policy.scheduler_queue_delay_ms;
                tuning.scheduler_queue_delay_ms = Some(args.scheduler_queue_delay_ms);
            }
            // Defer the log: env_logger is not yet initialized at this call site.
            tuning.auto_tune_rationale = Some(format!(
                "batch={}, micro_batch={}, delay={}ms, queue_size={} | {}",
                args.batch_size,
                args.scheduler_max_micro_batch,
                args.scheduler_queue_delay_ms,
                args.scheduler_queue_size,
                policy.rationale,
            ));
        }
        PerformanceProfile::Balanced => {
            if !batch_explicit {
                args.batch_size = 8;
                tuning.batch_size = Some(args.batch_size);
            }
            if !transport_explicit {
                args.transport = "hybrid".to_string();
                tuning.transport = Some(args.transport.clone());
            }
            if !scheduler_queue_size_explicit {
                args.scheduler_queue_size = 512;
                tuning.scheduler_queue_size = Some(args.scheduler_queue_size);
            }
            if !scheduler_max_micro_batch_explicit {
                args.scheduler_max_micro_batch = args.batch_size.max(1);
                tuning.scheduler_max_micro_batch = Some(args.scheduler_max_micro_batch);
            }
            if !scheduler_queue_delay_ms_explicit {
                args.scheduler_queue_delay_ms = 3;
                tuning.scheduler_queue_delay_ms = Some(args.scheduler_queue_delay_ms);
            }
        }
        PerformanceProfile::Throughput => {
            if !batch_explicit {
                args.batch_size = 16;
                tuning.batch_size = Some(args.batch_size);
            }
            if !transport_explicit {
                args.transport = "hybrid".to_string();
                tuning.transport = Some(args.transport.clone());
            }
            if !scheduler_queue_size_explicit {
                args.scheduler_queue_size = 2048;
                tuning.scheduler_queue_size = Some(args.scheduler_queue_size);
            }
            if !scheduler_max_micro_batch_explicit {
                args.scheduler_max_micro_batch = args.batch_size.max(1);
                tuning.scheduler_max_micro_batch = Some(args.scheduler_max_micro_batch);
            }
            if !scheduler_queue_delay_ms_explicit {
                args.scheduler_queue_delay_ms = 6;
                tuning.scheduler_queue_delay_ms = Some(args.scheduler_queue_delay_ms);
            }
            if std::env::var_os("RUST_LOG").is_none() {
                std::env::set_var("RUST_LOG", "warn");
                tuning.rust_log = Some("warn".to_string());
            }
        }
        PerformanceProfile::Latency => {
            if !batch_explicit {
                args.batch_size = 1;
                tuning.batch_size = Some(args.batch_size);
            }
            if !transport_explicit {
                args.transport = "socket".to_string();
                tuning.transport = Some(args.transport.clone());
            }
            if !scheduler_queue_size_explicit {
                args.scheduler_queue_size = 128;
                tuning.scheduler_queue_size = Some(args.scheduler_queue_size);
            }
            if !scheduler_max_micro_batch_explicit {
                args.scheduler_max_micro_batch = 1;
                tuning.scheduler_max_micro_batch = Some(args.scheduler_max_micro_batch);
            }
            if !scheduler_queue_delay_ms_explicit {
                args.scheduler_queue_delay_ms = 0;
                tuning.scheduler_queue_delay_ms = Some(args.scheduler_queue_delay_ms);
            }
        }
    }

    tuning
}

#[cfg(test)]
mod vram_cap_tests {
    use super::{device_vram_cap_bytes, parse_cuda_memory_limit, resolve_vram_cap_bytes};
    use crate::app::constants::CUDA_DEVICE_MEMORY_LIMIT_ENV;

    const MIB: usize = 1024 * 1024;
    const GIB: usize = 1024 * 1024 * 1024;

    #[test]
    fn parses_plain_bytes_and_binary_suffixes() {
        assert_eq!(parse_cuda_memory_limit("1048576"), Some(MIB));
        assert_eq!(parse_cuda_memory_limit("8g"), Some(8 * GIB));
        assert_eq!(parse_cuda_memory_limit("2560m"), Some(2560 * MIB));
        assert_eq!(parse_cuda_memory_limit("4gb"), Some(4 * GIB));
        assert_eq!(parse_cuda_memory_limit("512k"), Some(512 * 1024));
        // Case-insensitive and whitespace-tolerant.
        assert_eq!(parse_cuda_memory_limit("  8G  "), Some(8 * GIB));
    }

    #[test]
    fn rejects_empty_zero_and_malformed_values() {
        assert_eq!(parse_cuda_memory_limit(""), None);
        assert_eq!(parse_cuda_memory_limit("   "), None);
        assert_eq!(parse_cuda_memory_limit("0"), None);
        assert_eq!(parse_cuda_memory_limit("0g"), None);
        assert_eq!(parse_cuda_memory_limit("abc"), None);
        assert_eq!(parse_cuda_memory_limit("8t"), None);
        assert_eq!(parse_cuda_memory_limit("8 g"), None);
    }

    #[test]
    fn per_device_cap_wins_over_process_wide_and_kapsl_alias() {
        let cap = resolve_vram_cap_bytes(
            Some("4g".to_string()),
            Some("8g".to_string()),
            Some("16384".to_string()),
        );
        assert_eq!(cap, Some(4 * GIB));
    }

    #[test]
    fn malformed_higher_priority_source_falls_through() {
        // A malformed per-device value defers to the process-wide cap.
        let cap = resolve_vram_cap_bytes(
            Some("garbage".to_string()),
            Some("8g".to_string()),
            None,
        );
        assert_eq!(cap, Some(8 * GIB));
    }

    #[test]
    fn kapsl_alias_is_plain_mib_converted_to_bytes() {
        let cap = resolve_vram_cap_bytes(None, None, Some("2048".to_string()));
        assert_eq!(cap, Some(2048 * MIB));
        // Non-positive / malformed MiB is ignored.
        assert_eq!(resolve_vram_cap_bytes(None, None, Some("0".to_string())), None);
        assert_eq!(resolve_vram_cap_bytes(None, None, Some("x".to_string())), None);
    }

    #[test]
    fn no_sources_configured_yields_no_cap() {
        assert_eq!(resolve_vram_cap_bytes(None, None, None), None);
    }

    #[test]
    fn device_vram_cap_bytes_reads_per_device_env() {
        // Uses a unique device index so the process-global env never collides
        // with other tests running in parallel; the suffixed var is checked
        // first, so this is independent of any bare-name env state.
        let device_id = 9090;
        let var = format!("{CUDA_DEVICE_MEMORY_LIMIT_ENV}_{device_id}");
        std::env::set_var(&var, "8g");
        assert_eq!(device_vram_cap_bytes(device_id), Some(8 * GIB));
        std::env::remove_var(&var);
    }
}
