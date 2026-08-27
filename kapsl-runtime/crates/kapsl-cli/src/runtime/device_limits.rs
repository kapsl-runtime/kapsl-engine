//! Device-memory limits and co-tenancy ceiling policy.

use super::optional_env_var;
use crate::app::{CUDA_DEVICE_MEMORY_LIMIT_ENV, KAPSL_GPU_MEMORY_LIMIT_MB_ENV};

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
pub(crate) fn parse_cuda_memory_limit(value: &str) -> Option<usize> {
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

/// Instantaneous per-device KV ceiling in bytes under co-tenancy: the declared
/// budget (the physical card, or the HAMi / kapsl software-vGPU cap when that is
/// smaller), minus VRAM held by foreign processes, minus a safety reserve so a
/// trainer's next allocation spike lands in slack instead of racing our
/// allocator into an OOM. The reserve is 10% of the declared budget with a
/// 512 MiB floor. Saturating throughout: a foreign footprint larger than the
/// budget floors the ceiling at zero, and callers clamp back up to their
/// per-engine block minimum. With no cap configured and no foreign process this
/// returns `physical - reserve`, so single-tenant behavior only loses the small
/// reserve band (and the whole path is gated off by default anyway).
pub(crate) fn effective_ceiling_bytes(device_id: usize, physical: usize, foreign: usize) -> usize {
    const RESERVE_FLOOR_BYTES: usize = 512 * 1024 * 1024;
    let declared = device_vram_cap_bytes(device_id).map_or(physical, |cap| physical.min(cap));
    let reserve = (declared / 10).max(RESERVE_FLOOR_BYTES);
    declared.saturating_sub(foreign).saturating_sub(reserve)
}

/// Blend a freshly measured ceiling `target` toward the `previous` live value,
/// asymmetrically. Shrink immediately (drop straight to the lower value) so a
/// trainer's allocation is honored before it can OOM us; grow slowly (a quarter
/// of the gap per tick) so a transient dip in the trainer's footprint between
/// samples does not make the KV batch width flap. An unseeded `previous` of 0
/// adopts the target directly. Downward "flap" is harmless — it just keeps us
/// conservatively low — so only the grow side is damped.
pub(crate) fn smooth_ceiling_bytes(previous: usize, target: usize) -> usize {
    if previous == 0 || target <= previous {
        return target;
    }
    previous + (target - previous) / 4
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
        let cap = resolve_vram_cap_bytes(Some("garbage".to_string()), Some("8g".to_string()), None);
        assert_eq!(cap, Some(8 * GIB));
    }

    #[test]
    fn kapsl_alias_is_plain_mib_converted_to_bytes() {
        let cap = resolve_vram_cap_bytes(None, None, Some("2048".to_string()));
        assert_eq!(cap, Some(2048 * MIB));
        // Non-positive / malformed MiB is ignored.
        assert_eq!(
            resolve_vram_cap_bytes(None, None, Some("0".to_string())),
            None
        );
        assert_eq!(
            resolve_vram_cap_bytes(None, None, Some("x".to_string())),
            None
        );
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

#[cfg(test)]
mod ceiling_tests {
    use super::{effective_ceiling_bytes, smooth_ceiling_bytes};
    use crate::app::constants::CUDA_DEVICE_MEMORY_LIMIT_ENV;

    const GIB: usize = 1024 * 1024 * 1024;
    const RESERVE_FLOOR: usize = 512 * 1024 * 1024;

    #[test]
    fn subtracts_foreign_and_percentage_reserve_when_uncapped() {
        // Unique device id with no cap env → declared == physical (40 GiB).
        // Reserve is 10% (4 GiB, above the 512 MiB floor); a 6 GiB trainer
        // leaves 40 - 6 - 4 = 30 GiB.
        let ceiling = effective_ceiling_bytes(9101, 40 * GIB, 6 * GIB);
        assert_eq!(ceiling, 30 * GIB);
    }

    #[test]
    fn reserve_never_below_the_512mib_floor() {
        // Tiny 2 GiB budget: 10% would be ~205 MiB, so the 512 MiB floor wins.
        let ceiling = effective_ceiling_bytes(9102, 2 * GIB, 0);
        assert_eq!(ceiling, 2 * GIB - RESERVE_FLOOR);
    }

    #[test]
    fn ceiling_is_clamped_to_the_configured_cap_not_the_card() {
        let device_id = 9103;
        let var = format!("{CUDA_DEVICE_MEMORY_LIMIT_ENV}_{device_id}");
        std::env::set_var(&var, "8g");
        // Physical 40 GiB but capped to an 8 GiB slice; reserve is 10% of 8 GiB.
        let ceiling = effective_ceiling_bytes(device_id, 40 * GIB, 0);
        std::env::remove_var(&var);
        assert_eq!(ceiling, 8 * GIB - (8 * GIB / 10));
    }

    #[test]
    fn foreign_larger_than_budget_floors_at_zero() {
        // A trainer holding more than the whole budget must not underflow.
        let ceiling = effective_ceiling_bytes(9104, 8 * GIB, 16 * GIB);
        assert_eq!(ceiling, 0);
    }

    #[test]
    fn smoothing_shrinks_immediately_but_grows_a_quarter_at_a_time() {
        // Unseeded adopts the target.
        assert_eq!(smooth_ceiling_bytes(0, 30 * GIB), 30 * GIB);
        // Shrink: drop straight to the lower value (safety).
        assert_eq!(smooth_ceiling_bytes(30 * GIB, 10 * GIB), 10 * GIB);
        // Grow: move a quarter of the 20 GiB gap → 10 + 5 = 15 GiB.
        assert_eq!(smooth_ceiling_bytes(10 * GIB, 30 * GIB), 15 * GIB);
        // Equal target is a no-op.
        assert_eq!(smooth_ceiling_bytes(15 * GIB, 15 * GIB), 15 * GIB);
    }
}
