//! Resolution of ONNX Runtime tuning settings.
//!
//! Settings come from four layers, each overriding the previous: auto-derived
//! defaults, environment variables, command-line flags, and per-model
//! `--onnx-model-tuning` specs. This module owns that precedence chain; the CLI
//! surface itself lives in [`super::cli`].

use super::*;

#[derive(Debug, Clone, Default)]
pub(crate) struct OnnxTuningProfile {
    pub(crate) global: OnnxRuntimeTuning,
    pub(crate) per_model: HashMap<u32, OnnxRuntimeTuning>,
}

pub(crate) fn merge_onnx_runtime_tuning(
    base: &OnnxRuntimeTuning,
    overrides: &OnnxRuntimeTuning,
) -> OnnxRuntimeTuning {
    OnnxRuntimeTuning {
        memory_pattern: overrides.memory_pattern.or(base.memory_pattern),
        disable_cpu_mem_arena: overrides
            .disable_cpu_mem_arena
            .or(base.disable_cpu_mem_arena),
        session_buckets: overrides.session_buckets.or(base.session_buckets),
        bucket_dim_granularity: overrides
            .bucket_dim_granularity
            .or(base.bucket_dim_granularity),
        bucket_max_dims: overrides.bucket_max_dims.or(base.bucket_max_dims),
        peak_concurrency_hint: overrides
            .peak_concurrency_hint
            .or(base.peak_concurrency_hint),
    }
}

impl OnnxTuningProfile {
    pub(crate) fn resolve(&self, model_id: u32) -> OnnxRuntimeTuning {
        if let Some(model_overrides) = self.per_model.get(&model_id) {
            merge_onnx_runtime_tuning(&self.global, model_overrides)
        } else {
            self.global.clone()
        }
    }
}

/// Parse a positive integer tuning value, clamping to at least 1.
///
/// `label` names the setting in the error message -- the canonical key for a
/// `--onnx-model-tuning` pair, or the variable name for an env override.
fn parse_positive_setting<T>(label: &str, value: &str) -> Result<T, String>
where
    T: std::str::FromStr + Ord + From<u8>,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    value
        .trim()
        .parse::<T>()
        .map(|parsed| parsed.max(T::from(1u8)))
        .map_err(|e| format!("invalid {} '{}': {}", label, value, e))
}

pub(crate) fn parse_bool_literal(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid boolean '{}'", value)),
    }
}

pub(crate) fn apply_onnx_tuning_pair(
    target: &mut OnnxRuntimeTuning,
    key: &str,
    value: &str,
) -> Result<(), String> {
    let normalized = key.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "memory_pattern" | "mem_pattern" => {
            target.memory_pattern = Some(parse_bool_literal(value)?);
        }
        "disable_cpu_mem_arena" | "cpu_mem_arena_disabled" => {
            target.disable_cpu_mem_arena = Some(parse_bool_literal(value)?);
        }
        "session_buckets" => {
            target.session_buckets = Some(parse_positive_setting("session_buckets", value)?);
        }
        "bucket_dim_granularity" => {
            target.bucket_dim_granularity =
                Some(parse_positive_setting("bucket_dim_granularity", value)?);
        }
        "bucket_max_dims" => {
            target.bucket_max_dims = Some(parse_positive_setting("bucket_max_dims", value)?);
        }
        "peak_concurrency" | "peak_concurrency_hint" => {
            target.peak_concurrency_hint = Some(parse_positive_setting("peak_concurrency", value)?);
        }
        other => {
            return Err(format!(
                "unknown ONNX tuning key '{}'; expected one of memory_pattern, disable_cpu_mem_arena, session_buckets, bucket_dim_granularity, bucket_max_dims, peak_concurrency",
                other
            ));
        }
    }
    Ok(())
}

pub(crate) fn parse_env_bool_override(name: &str) -> Result<Option<bool>, String> {
    optional_env_var(name)
        .map(|value| parse_bool_literal(&value))
        .transpose()
}

pub(crate) fn parse_env_usize_override(name: &str) -> Result<Option<usize>, String> {
    optional_env_var(name)
        .map(|value| parse_positive_setting(name, &value))
        .transpose()
}

pub(crate) fn parse_env_u32_override(name: &str) -> Result<Option<u32>, String> {
    optional_env_var(name)
        .map(|value| parse_positive_setting(name, &value))
        .transpose()
}

pub(crate) fn auto_onnx_runtime_tuning(args: &Args) -> OnnxRuntimeTuning {
    let batch_size = args.batch_size.max(1);
    let session_pool = batch_size.min(logical_cpu_cores().max(1)).clamp(1, 4);
    let session_buckets = batch_size.max(4).min(8);
    OnnxRuntimeTuning {
        memory_pattern: Some(true),
        disable_cpu_mem_arena: Some(false),
        session_buckets: Some(session_buckets),
        bucket_dim_granularity: Some(64),
        bucket_max_dims: Some(4),
        peak_concurrency_hint: Some(session_pool as u32),
    }
}

pub(crate) fn env_onnx_runtime_tuning() -> Result<OnnxRuntimeTuning, String> {
    Ok(OnnxRuntimeTuning {
        memory_pattern: parse_env_bool_override(ORT_MEMORY_PATTERN_ENV)?,
        disable_cpu_mem_arena: parse_env_bool_override(ORT_DISABLE_CPU_MEM_ARENA_ENV)?,
        session_buckets: parse_env_usize_override(ORT_SESSION_BUCKETS_ENV)?,
        bucket_dim_granularity: parse_env_usize_override(ORT_BUCKET_DIM_GRANULARITY_ENV)?,
        bucket_max_dims: parse_env_usize_override(ORT_BUCKET_MAX_DIMS_ENV)?,
        peak_concurrency_hint: parse_env_u32_override(MODEL_PEAK_CONCURRENCY_ENV)?,
    })
}

pub(crate) fn parse_onnx_model_tuning_spec(
    spec: &str,
) -> Result<(Option<u32>, OnnxRuntimeTuning), String> {
    let (selector_raw, config_raw) = spec.split_once(':').ok_or_else(|| {
        format!(
            "invalid --onnx-model-tuning '{}': expected '<model_id|*>:k=v[,k=v...]'",
            spec
        )
    })?;
    let selector = selector_raw.trim();
    let model_id = if selector == "*" {
        None
    } else {
        Some(
            selector
                .parse::<u32>()
                .map_err(|e| format!("invalid model selector '{}': {}", selector, e))?,
        )
    };

    let mut tuning = OnnxRuntimeTuning::default();
    for pair in config_raw.split(',') {
        let trimmed = pair.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (key, value) = trimmed
            .split_once('=')
            .ok_or_else(|| format!("invalid tuning pair '{}': expected k=v", trimmed))?;
        apply_onnx_tuning_pair(&mut tuning, key, value)?;
    }

    Ok((model_id, tuning))
}

pub(crate) fn build_onnx_tuning_profile(args: &Args) -> Result<OnnxTuningProfile, String> {
    let mut profile = OnnxTuningProfile {
        global: auto_onnx_runtime_tuning(args),
        per_model: HashMap::new(),
    };

    let env_tuning = env_onnx_runtime_tuning()?;
    profile.global = merge_onnx_runtime_tuning(&profile.global, &env_tuning);
    let cli_tuning = OnnxRuntimeTuning {
        memory_pattern: args.onnx_memory_pattern,
        disable_cpu_mem_arena: args.onnx_disable_cpu_mem_arena,
        session_buckets: args.onnx_session_buckets,
        bucket_dim_granularity: args.onnx_bucket_dim_granularity,
        bucket_max_dims: args.onnx_bucket_max_dims,
        peak_concurrency_hint: args.onnx_peak_concurrency_hint,
    };
    profile.global = merge_onnx_runtime_tuning(&profile.global, &cli_tuning);

    for spec in &args.onnx_model_tuning {
        let (model_id, tuning) = parse_onnx_model_tuning_spec(spec)?;
        if let Some(model_id) = model_id {
            let merged = profile
                .per_model
                .get(&model_id)
                .map(|existing| merge_onnx_runtime_tuning(existing, &tuning))
                .unwrap_or(tuning);
            profile.per_model.insert(model_id, merged);
        } else {
            profile.global = merge_onnx_runtime_tuning(&profile.global, &tuning);
        }
    }

    Ok(profile)
}
