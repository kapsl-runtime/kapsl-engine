//! Host-memory admission and reservation accounting.

use crate::app::config::constants::KAPSL_CPU_MEMORY_LIMIT_MB_ENV;
use crate::runtime::memory::{MemoryAllocationClass, MemoryClaim, MemoryDomain, MemoryOwner};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use sysinfo::{Pid, System};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

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
    reservations: Mutex<HostReservations>,
    load_lock: Arc<AsyncMutex<()>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct HostReservationOwner {
    domain: MemoryDomain,
    owner: MemoryOwner,
    class: MemoryAllocationClass,
}

#[derive(Default)]
struct HostReservations {
    by_owner: HashMap<HostReservationOwner, usize>,
}

impl HostReservations {
    fn total_bytes(&self) -> usize {
        self.by_owner
            .values()
            .copied()
            .fold(0usize, usize::saturating_add)
    }

    fn class_group_bytes(&self, class: MemoryAllocationClass) -> usize {
        self.by_owner
            .iter()
            .filter(|(key, _)| budget_group(key.class) == budget_group(class))
            .map(|(_, bytes)| *bytes)
            .fold(0usize, usize::saturating_add)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostBudgetGroup {
    Model,
    Kv,
}

fn budget_group(class: MemoryAllocationClass) -> HostBudgetGroup {
    match class {
        MemoryAllocationClass::KvCache => HostBudgetGroup::Kv,
        MemoryAllocationClass::PersistentWeights
        | MemoryAllocationClass::ModelSession
        | MemoryAllocationClass::TransientWorkspace
        | MemoryAllocationClass::BlockTable
        | MemoryAllocationClass::RequestTransient
        | MemoryAllocationClass::ExternallyOwned => HostBudgetGroup::Model,
    }
}

impl HostMemoryManager {
    pub(crate) fn new(device_info: &kapsl_hal::device::DeviceInfo) -> Arc<Self> {
        Arc::new(Self {
            budget: HostMemoryBudget::detect(device_info.total_memory),
            reservations: Mutex::new(HostReservations::default()),
            load_lock: Arc::new(AsyncMutex::new(())),
        })
    }

    pub(crate) fn budget(&self) -> HostMemoryBudget {
        self.budget
    }

    pub(crate) fn admit(
        self: &Arc<Self>,
        claim: &MemoryClaim,
    ) -> Result<Option<HostMemoryLease>, String> {
        if !matches!(
            claim.domain,
            MemoryDomain::Host | MemoryDomain::HostPinned { .. } | MemoryDomain::HostMapped { .. }
        ) {
            return Err(format!(
                "host memory manager cannot admit a {} claim",
                claim.domain
            ));
        }
        if claim.bytes == 0 {
            return Ok(None);
        }
        // KV retains its historical half-budget. Model/session and active
        // request-transient memory share the complementary half, so a full KV
        // cache can never push aggregate admitted memory above the safe budget.
        let class_budget = self.budget.class_limit(claim.class);
        let mut reservations = self.reservations.lock();
        let class_reserved = reservations.class_group_bytes(claim.class);
        let class_projected = class_reserved.saturating_add(claim.bytes);
        let global_projected = reservations.total_bytes().saturating_add(claim.bytes);
        if class_projected > class_budget || global_projected > self.budget.safe_bytes {
            return Err(format!(
                "host memory admission rejected for {} class={}: requested={} class_reserved={} class_projected={} class_budget={} global_projected={} host_safe_budget={} bytes",
                claim.owner,
                claim.class,
                claim.bytes,
                class_reserved,
                class_projected,
                class_budget,
                global_projected,
                self.budget.safe_bytes,
            ));
        }
        let key = HostReservationOwner {
            domain: claim.domain.clone(),
            owner: claim.owner,
            class: claim.class,
        };
        let owned = reservations.by_owner.entry(key.clone()).or_default();
        *owned = owned.saturating_add(claim.bytes);
        log::info!(
            "[host-memory] admitted {} class={}: reserved={} class_reserved={} global_reserved={} class_budget={} host_safe_budget={} bytes",
            claim.owner,
            claim.class,
            claim.bytes,
            class_projected,
            global_projected,
            class_budget,
            self.budget.safe_bytes
        );
        Ok(Some(HostMemoryLease {
            manager: Arc::clone(self),
            key,
            bytes: claim.bytes,
        }))
    }

    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.reservations.lock().total_bytes()
    }

    /// Serialize host model loads so a before/after process-RSS delta can be
    /// attributed to one load transaction. Existing inference may still move
    /// RSS while the sample is open, so reconciliation only raises the planned
    /// reservation; it never weakens a conservative estimate.
    pub(crate) async fn begin_load_reconciliation(self: &Arc<Self>) -> HostMemoryLoadAdmission {
        let guard = Arc::clone(&self.load_lock).lock_owned().await;
        HostMemoryLoadAdmission {
            manager: Arc::clone(self),
            rss_before: process_rss_bytes(),
            _guard: Some(guard),
        }
    }

    pub(crate) fn resize_lease(
        &self,
        lease: &mut HostMemoryLease,
        target_bytes: usize,
    ) -> Result<(), String> {
        if lease.bytes == target_bytes {
            return Ok(());
        }
        let mut reservations = self.reservations.lock();
        let class_reserved = reservations.class_group_bytes(lease.key.class);
        let global_reserved = reservations.total_bytes();
        let class_projected = class_reserved
            .saturating_sub(lease.bytes)
            .saturating_add(target_bytes);
        let global_projected = global_reserved
            .saturating_sub(lease.bytes)
            .saturating_add(target_bytes);
        let class_budget = self.budget.class_limit(lease.key.class);
        if class_projected > class_budget || global_projected > self.budget.safe_bytes {
            return Err(format!(
                "host memory reconciliation rejected for {} class={}: planned={} observed_target={} class_projected={} class_budget={} global_projected={} host_safe_budget={} bytes",
                lease.key.owner,
                lease.key.class,
                lease.bytes,
                target_bytes,
                class_projected,
                class_budget,
                global_projected,
                self.budget.safe_bytes,
            ));
        }
        let owned = reservations.by_owner.entry(lease.key.clone()).or_default();
        *owned = owned
            .saturating_sub(lease.bytes)
            .saturating_add(target_bytes);
        if *owned == 0 {
            reservations.by_owner.remove(&lease.key);
        }
        lease.bytes = target_bytes;
        Ok(())
    }
}

pub(crate) struct HostMemoryLease {
    manager: Arc<HostMemoryManager>,
    key: HostReservationOwner,
    bytes: usize,
}

impl HostMemoryLease {
    pub(crate) fn matches(&self, claim: &MemoryClaim) -> bool {
        self.key.domain == claim.domain
            && self.key.owner == claim.owner
            && self.key.class == claim.class
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn claim(&self) -> MemoryClaim {
        MemoryClaim::runtime(
            self.key.domain.clone(),
            self.key.owner,
            self.key.class,
            self.bytes,
        )
    }

    pub(crate) fn resize(&mut self, target_bytes: usize) -> Result<(), String> {
        let manager = Arc::clone(&self.manager);
        manager.resize_lease(self, target_bytes)
    }
}

/// Before/after RSS attribution guard for one host model load.
pub(crate) struct HostMemoryLoadAdmission {
    manager: Arc<HostMemoryManager>,
    rss_before: Option<usize>,
    _guard: Option<OwnedMutexGuard<()>>,
}

impl HostMemoryLoadAdmission {
    pub(crate) fn reconcile(&mut self, leases: &mut [HostMemoryLease]) -> Result<(), String> {
        let planned_total = leases
            .iter()
            .filter(|lease| budget_group(lease.key.class) == HostBudgetGroup::Model)
            .map(|lease| lease.bytes)
            .fold(0usize, usize::saturating_add);
        let observed_delta = self
            .rss_before
            .zip(process_rss_bytes())
            .map(|(before, after)| after.saturating_sub(before));
        let target_total = observed_delta.map_or(planned_total, |bytes| bytes.max(planned_total));

        if target_total > planned_total {
            let Some(index) = leases
                .iter()
                .position(|lease| budget_group(lease.key.class) == HostBudgetGroup::Model)
            else {
                self._guard.take();
                return Err(format!(
                    "observed {} bytes of host RSS growth without a model/session reservation",
                    target_total - planned_total
                ));
            };
            let target = leases[index]
                .bytes
                .saturating_add(target_total - planned_total);
            self.manager.resize_lease(&mut leases[index], target)?;
        }
        log::info!(
            "[host-memory] reconciled model load: planned={} observed_rss_delta={:?} retained={} bytes",
            planned_total,
            observed_delta,
            target_total
        );
        self._guard.take();
        Ok(())
    }
}

pub(crate) fn process_rss_bytes() -> Option<usize> {
    let mut system = System::new();
    let pid = Pid::from_u32(std::process::id());
    system.refresh_process(pid);
    system.process(pid).map(|process| process.memory() as usize)
}

impl Drop for HostMemoryLease {
    fn drop(&mut self) {
        let mut reservations = self.manager.reservations.lock();
        let remove = reservations
            .by_owner
            .get_mut(&self.key)
            .is_some_and(|bytes| {
                *bytes = bytes.saturating_sub(self.bytes);
                *bytes == 0
            });
        if remove {
            reservations.by_owner.remove(&self.key);
        }
    }
}

impl HostMemoryBudget {
    pub(crate) fn class_limit(self, _class: MemoryAllocationClass) -> usize {
        // Each group may use up to half of safe memory, but all groups remain
        // subject to the single global safe-memory cap.
        self.safe_bytes / 2
    }

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
    use super::{HostMemoryBudget, HostMemoryManager, HostReservations};
    use crate::runtime::memory::{MemoryAllocationClass, MemoryClaim, MemoryDomain, MemoryOwner};
    use parking_lot::Mutex;
    use std::sync::Arc;
    use tokio::sync::Mutex as AsyncMutex;

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
            reservations: Mutex::new(HostReservations::default()),
            load_lock: Arc::new(AsyncMutex::new(())),
        });
        let first = MemoryClaim::runtime(
            MemoryDomain::Host,
            MemoryOwner::new(7, 2),
            MemoryAllocationClass::ModelSession,
            3 * GIB,
        );
        let lease = manager.admit(&first).unwrap().unwrap();
        assert_eq!(manager.reserved_bytes(), 3 * GIB);
        let too_large = MemoryClaim::runtime(
            MemoryDomain::Host,
            MemoryOwner::new(8, 0),
            MemoryAllocationClass::PersistentWeights,
            2 * GIB,
        );
        assert!(manager.admit(&too_large).is_err());
        drop(lease);
        assert_eq!(manager.reserved_bytes(), 0);
    }

    #[test]
    fn rss_reconciliation_can_raise_but_not_overcommit_a_lease() {
        let manager = Arc::new(HostMemoryManager {
            budget: HostMemoryBudget {
                limit_bytes: 10 * GIB,
                safe_bytes: 8 * GIB,
            },
            reservations: Mutex::new(HostReservations::default()),
            load_lock: Arc::new(AsyncMutex::new(())),
        });
        let claim = MemoryClaim::runtime(
            MemoryDomain::Host,
            MemoryOwner::new(7, 2),
            MemoryAllocationClass::ModelSession,
            2 * GIB,
        );
        let mut lease = manager.admit(&claim).unwrap().unwrap();
        manager.resize_lease(&mut lease, 3 * GIB).unwrap();
        assert_eq!(manager.reserved_bytes(), 3 * GIB);
        assert!(manager.resize_lease(&mut lease, 5 * GIB).is_err());
        assert_eq!(manager.reserved_bytes(), 3 * GIB);
        drop(lease);
        assert_eq!(manager.reserved_bytes(), 0);
    }

    #[test]
    fn kv_and_model_classes_have_independent_half_budgets() {
        let manager = Arc::new(HostMemoryManager {
            budget: HostMemoryBudget {
                limit_bytes: 10 * GIB,
                safe_bytes: 8 * GIB,
            },
            reservations: Mutex::new(HostReservations::default()),
            load_lock: Arc::new(AsyncMutex::new(())),
        });
        let model = MemoryClaim::runtime(
            MemoryDomain::Host,
            MemoryOwner::new(7, 0),
            MemoryAllocationClass::ModelSession,
            4 * GIB,
        );
        let kv = MemoryClaim::runtime(
            MemoryDomain::Host,
            MemoryOwner::new(7, 0),
            MemoryAllocationClass::KvCache,
            4 * GIB,
        );
        let model_lease = manager.admit(&model).unwrap().unwrap();
        let kv_lease = manager.admit(&kv).unwrap().unwrap();
        assert_eq!(manager.reserved_bytes(), 8 * GIB);
        drop((model_lease, kv_lease));
        assert_eq!(manager.reserved_bytes(), 0);
    }

    #[test]
    fn request_transient_shares_the_non_kv_half_budget() {
        let manager = Arc::new(HostMemoryManager {
            budget: HostMemoryBudget {
                limit_bytes: 10 * GIB,
                safe_bytes: 8 * GIB,
            },
            reservations: Mutex::new(HostReservations::default()),
            load_lock: Arc::new(AsyncMutex::new(())),
        });
        let model = MemoryClaim::runtime(
            MemoryDomain::Host,
            MemoryOwner::new(7, 0),
            MemoryAllocationClass::ModelSession,
            3 * GIB,
        );
        let request = MemoryClaim::runtime(
            MemoryDomain::Host,
            MemoryOwner::new(7, 0),
            MemoryAllocationClass::RequestTransient,
            GIB,
        );
        let model_lease = manager.admit(&model).unwrap().unwrap();
        let request_lease = manager.admit(&request).unwrap().unwrap();
        assert!(manager.admit(&request).is_err());
        drop((model_lease, request_lease));
        assert_eq!(manager.reserved_bytes(), 0);
    }

    #[test]
    fn non_host_domains_are_rejected() {
        let manager = Arc::new(HostMemoryManager {
            budget: HostMemoryBudget {
                limit_bytes: 10 * GIB,
                safe_bytes: 8 * GIB,
            },
            reservations: Mutex::new(HostReservations::default()),
            load_lock: Arc::new(AsyncMutex::new(())),
        });
        let claim = MemoryClaim::runtime(
            MemoryDomain::Cuda { device_id: 1 },
            MemoryOwner::new(7, 0),
            MemoryAllocationClass::ModelSession,
            GIB,
        );
        let error = match manager.admit(&claim) {
            Ok(_) => panic!("CUDA claim must not be admitted by the host manager"),
            Err(error) => error,
        };
        assert!(error.contains("cannot admit"));
    }
}
