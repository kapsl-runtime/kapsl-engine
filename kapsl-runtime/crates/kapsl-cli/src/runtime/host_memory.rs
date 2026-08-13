use crate::app::constants::KAPSL_CPU_MEMORY_LIMIT_MB_ENV;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::Arc;

const MIB: usize = 1024 * 1024;
const DEFAULT_HEADROOM_PERCENT: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostMemoryBudget {
    /// The tightest detected physical, container, or operator-provided limit.
    pub(crate) limit_bytes: usize,
    /// Memory available to runtime-owned model state after safety headroom.
    pub(crate) safe_bytes: usize,
}

pub(crate) struct HostMemoryManager {
    budget: HostMemoryBudget,
    cpu_device_ids: HashSet<usize>,
    reserved_bytes: Mutex<usize>,
}

impl HostMemoryManager {
    pub(crate) fn new(device_info: &kapsl_hal::device::DeviceInfo) -> Arc<Self> {
        Arc::new(Self {
            budget: HostMemoryBudget::detect(device_info.total_memory),
            cpu_device_ids: device_info
                .devices
                .iter()
                .filter(|device| device.backend.to_string().eq_ignore_ascii_case("cpu"))
                .map(|device| device.id)
                .collect(),
            reserved_bytes: Mutex::new(0),
        })
    }

    pub(crate) fn admit(
        self: &Arc<Self>,
        device_ids: &[usize],
        model_id: u32,
        requested_bytes: usize,
    ) -> Result<Option<HostMemoryLease>, String> {
        if requested_bytes == 0
            || !device_ids
                .iter()
                .any(|device_id| self.cpu_device_ids.contains(device_id))
        {
            return Ok(None);
        }
        // The KV coordinator may consume half of the same safe host budget.
        // Keep model/session reservations inside the complementary half so the
        // two independently allocated classes cannot jointly exceed it.
        let admission_budget = self.budget.safe_bytes / 2;
        let mut reserved = self.reserved_bytes.lock();
        let projected = reserved.saturating_add(requested_bytes);
        if projected > admission_budget {
            return Err(format!(
                "CPU memory admission rejected for model {model_id}: requested={requested_bytes} reserved={} projected={projected} model_budget={admission_budget} host_safe_budget={} bytes",
                *reserved, self.budget.safe_bytes,
            ));
        }
        *reserved = projected;
        log::info!(
            "[host-memory] admitted model {}: reserved={} global_reserved={} model_budget={} host_safe_budget={} bytes",
            model_id,
            requested_bytes,
            projected,
            admission_budget,
            self.budget.safe_bytes
        );
        Ok(Some(HostMemoryLease {
            manager: Arc::clone(self),
            bytes: requested_bytes,
        }))
    }

    #[cfg(test)]
    fn reserved_bytes(&self) -> usize {
        *self.reserved_bytes.lock()
    }
}

pub(crate) struct HostMemoryLease {
    manager: Arc<HostMemoryManager>,
    bytes: usize,
}

impl Drop for HostMemoryLease {
    fn drop(&mut self) {
        let mut reserved = self.manager.reserved_bytes.lock();
        *reserved = reserved.saturating_sub(self.bytes);
    }
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
    use super::{HostMemoryBudget, HostMemoryManager};
    use parking_lot::Mutex;
    use std::collections::HashSet;
    use std::sync::Arc;

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

    #[test]
    fn admission_is_released_with_its_lease() {
        let manager = Arc::new(HostMemoryManager {
            budget: HostMemoryBudget {
                limit_bytes: 10 * GIB,
                safe_bytes: 8 * GIB,
            },
            cpu_device_ids: HashSet::from([0]),
            reserved_bytes: Mutex::new(0),
        });
        let lease = manager.admit(&[0], 7, 3 * GIB).unwrap().unwrap();
        assert_eq!(manager.reserved_bytes(), 3 * GIB);
        assert!(manager.admit(&[0], 8, 6 * GIB).is_err());
        drop(lease);
        assert_eq!(manager.reserved_bytes(), 0);
    }

    #[test]
    fn non_cpu_devices_do_not_consume_host_admission_budget() {
        let manager = Arc::new(HostMemoryManager {
            budget: HostMemoryBudget {
                limit_bytes: 10 * GIB,
                safe_bytes: 8 * GIB,
            },
            cpu_device_ids: HashSet::from([0]),
            reserved_bytes: Mutex::new(0),
        });
        assert!(manager.admit(&[1], 7, GIB).unwrap().is_none());
        assert_eq!(manager.reserved_bytes(), 0);
    }
}
