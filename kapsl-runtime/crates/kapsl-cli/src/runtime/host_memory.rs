use crate::app::constants::KAPSL_CPU_MEMORY_LIMIT_MB_ENV;

const MIB: usize = 1024 * 1024;
const DEFAULT_HEADROOM_PERCENT: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostMemoryBudget {
    /// The tightest detected physical, container, or operator-provided limit.
    pub(crate) limit_bytes: usize,
    /// Memory available to runtime-owned model state after safety headroom.
    pub(crate) safe_bytes: usize,
}

impl HostMemoryBudget {
    pub(crate) fn detect(system_total_kib: u64) -> Self {
        let physical = usize::try_from(system_total_kib)
            .unwrap_or(usize::MAX / 1024)
            .saturating_mul(1024);
        let configured = std::env::var(KAPSL_CPU_MEMORY_LIMIT_MB_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .map(|mib| mib.saturating_mul(MIB));
        let container = cgroup_memory_limit_bytes();
        Self::from_limits(physical, configured, container)
    }

    fn from_limits(
        physical_bytes: usize,
        configured_bytes: Option<usize>,
        container_bytes: Option<usize>,
    ) -> Self {
        let limit_bytes = [
            Some(physical_bytes).filter(|value| *value > 0),
            configured_bytes,
            container_bytes,
        ]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(0);
        let safe_bytes = limit_bytes.saturating_mul(100 - DEFAULT_HEADROOM_PERCENT) / 100;
        Self {
            limit_bytes,
            safe_bytes,
        }
    }
}

fn cgroup_memory_limit_bytes() -> Option<usize> {
    // v2 first, then the common v1 path. `max` and implausibly large v1
    // sentinel values mean unlimited and are deliberately ignored.
    [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    .into_iter()
    .filter_map(|path| std::fs::read_to_string(path).ok())
    .filter_map(|raw| raw.trim().parse::<u64>().ok())
    .filter(|limit| *limit > 0 && *limit < (1_u64 << 60))
    .filter_map(|limit| usize::try_from(limit).ok())
    .min()
}

#[cfg(test)]
mod tests {
    use super::HostMemoryBudget;

    const GIB: usize = 1024 * 1024 * 1024;

    #[test]
    fn keeps_twenty_percent_headroom() {
        let budget = HostMemoryBudget::from_limits(16 * GIB, None, None);
        assert_eq!(budget.limit_bytes, 16 * GIB);
        assert_eq!(budget.safe_bytes, 16 * GIB * 4 / 5);
    }

    #[test]
    fn tightest_limit_wins() {
        let budget = HostMemoryBudget::from_limits(64 * GIB, Some(24 * GIB), Some(16 * GIB));
        assert_eq!(budget.limit_bytes, 16 * GIB);
        assert_eq!(budget.safe_bytes, 16 * GIB * 4 / 5);
    }

    #[test]
    fn zero_physical_memory_can_fall_back_to_a_container_limit() {
        let budget = HostMemoryBudget::from_limits(0, None, Some(8 * GIB));
        assert_eq!(budget.limit_bytes, 8 * GIB);
    }
}
