use std::collections::HashMap;

/// Accounting-only view of a CUDA device's runtime budget.
///
/// The elastic pool is one physical allocation, so its full backing capacity is
/// charged here even while individual pool pages are free. External bytes are
/// reservations for allocations owned by a backend (weights and unavoidable
/// backend state) and may move from a load-time estimate to an observed value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeviceBudgetSnapshot {
    pub(crate) budget_bytes: usize,
    pub(crate) pooled_bytes: usize,
    pub(crate) planned_external_bytes: usize,
    pub(crate) external_bytes: usize,
}

impl DeviceBudgetSnapshot {
    pub(crate) fn used_bytes(self) -> usize {
        self.pooled_bytes
            .saturating_add(self.planned_external_bytes)
            .saturating_add(self.external_bytes)
    }

    pub(crate) fn available_bytes(self) -> usize {
        self.budget_bytes.saturating_sub(self.used_bytes())
    }
}

#[derive(Debug)]
struct DeviceBudget {
    budget_bytes: usize,
    pooled_bytes: usize,
    allocations: HashMap<String, ExternalAllocation>,
}

#[derive(Debug)]
struct ExternalAllocation {
    bytes: usize,
    refs: usize,
    planned: bool,
}

#[derive(Debug, Default)]
pub(crate) struct DeviceBudgetLedger {
    devices: HashMap<usize, DeviceBudget>,
}

impl DeviceBudgetLedger {
    pub(crate) fn insert_device(
        &mut self,
        device_id: usize,
        budget_bytes: usize,
        pooled_bytes: usize,
    ) -> Result<(), String> {
        if pooled_bytes > budget_bytes {
            return Err(format!(
                "CUDA device {device_id} pool reserves {pooled_bytes} bytes, exceeding its {budget_bytes}-byte safe budget"
            ));
        }
        self.devices.insert(
            device_id,
            DeviceBudget {
                budget_bytes,
                pooled_bytes,
                allocations: HashMap::new(),
            },
        );
        Ok(())
    }

    pub(crate) fn reserve_external(
        &mut self,
        device_id: usize,
        allocation_id: &str,
        bytes: usize,
    ) -> Result<(DeviceBudgetSnapshot, bool), String> {
        let device = self
            .devices
            .get_mut(&device_id)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;
        if let Some(allocation) = device.allocations.get_mut(allocation_id) {
            allocation.refs = allocation.refs.saturating_add(1);
            return Ok((snapshot(device), false));
        }
        let current = snapshot(device);
        let projected = current.used_bytes().saturating_add(bytes);
        if projected > device.budget_bytes {
            return Err(format!(
                "CUDA device {device_id} memory admission rejected: pool={} external_current={} external_inflight={} external_planned={} projected={} safe_budget={} bytes",
                device.pooled_bytes,
                current.external_bytes,
                current.planned_external_bytes,
                bytes,
                projected,
                device.budget_bytes
            ));
        }
        device.allocations.insert(
            allocation_id.to_string(),
            ExternalAllocation {
                bytes,
                refs: 1,
                planned: true,
            },
        );
        Ok((snapshot(device), true))
    }

    /// Replace one load's planned reservation with its observed external use.
    ///
    /// The observed value is retained even when it crosses the budget. The
    /// caller will reject and tear down that load, while the retained charge
    /// prevents another load from entering during cleanup.
    pub(crate) fn reconcile_external(
        &mut self,
        device_id: usize,
        allocation_id: &str,
        actual_bytes: usize,
    ) -> Result<DeviceBudgetSnapshot, String> {
        let device = self
            .devices
            .get_mut(&device_id)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;
        let allocation = device.allocations.get_mut(allocation_id).ok_or_else(|| {
            format!(
                "CUDA device {device_id} has no external allocation reservation `{allocation_id}`"
            )
        })?;
        allocation.bytes = actual_bytes;
        allocation.planned = false;
        let current = snapshot(device);
        if current.used_bytes() > current.budget_bytes {
            return Err(format!(
                "CUDA device {device_id} exceeded its safe memory budget after backend load: pool={} external_actual_total={} used={} safe_budget={} bytes",
                current.pooled_bytes,
                current.external_bytes,
                current.used_bytes(),
                current.budget_bytes
            ));
        }
        Ok(current)
    }

    pub(crate) fn release_external(&mut self, device_id: usize, allocation_id: &str) {
        if let Some(device) = self.devices.get_mut(&device_id) {
            let remove = if let Some(allocation) = device.allocations.get_mut(allocation_id) {
                allocation.refs = allocation.refs.saturating_sub(1);
                allocation.refs == 0
            } else {
                false
            };
            if remove {
                device.allocations.remove(allocation_id);
            }
        }
    }

    pub(crate) fn snapshot(&self, device_id: usize) -> Option<DeviceBudgetSnapshot> {
        self.devices.get(&device_id).map(snapshot)
    }
}

fn snapshot(device: &DeviceBudget) -> DeviceBudgetSnapshot {
    let (planned_external_bytes, external_bytes) =
        device
            .allocations
            .values()
            .fold((0usize, 0usize), |(planned, actual), allocation| {
                if allocation.planned {
                    (planned.saturating_add(allocation.bytes), actual)
                } else {
                    (planned, actual.saturating_add(allocation.bytes))
                }
            });
    DeviceBudgetSnapshot {
        budget_bytes: device.budget_bytes,
        pooled_bytes: device.pooled_bytes,
        planned_external_bytes,
        external_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEVICE: usize = 3;
    const GIB: usize = 1024 * 1024 * 1024;

    fn ledger() -> DeviceBudgetLedger {
        let mut ledger = DeviceBudgetLedger::default();
        ledger
            .insert_device(DEVICE, 20 * GIB, 8 * GIB)
            .expect("insert test device");
        ledger
    }

    #[test]
    fn pool_and_external_reservations_share_one_budget() {
        let mut ledger = ledger();
        let (snapshot, owns_charge) = ledger
            .reserve_external(DEVICE, "weights-a", 12 * GIB)
            .expect("exact fit should be admitted");
        assert!(owns_charge);
        assert_eq!(snapshot.used_bytes(), 20 * GIB);
        assert_eq!(snapshot.available_bytes(), 0);
        assert!(ledger.reserve_external(DEVICE, "weights-b", 1).is_err());
    }

    #[test]
    fn reconciliation_releases_overestimated_headroom() {
        let mut ledger = ledger();
        ledger
            .reserve_external(DEVICE, "weights", 8 * GIB)
            .expect("planned load");
        let snapshot = ledger
            .reconcile_external(DEVICE, "weights", 5 * GIB)
            .expect("actual load");
        assert_eq!(snapshot.planned_external_bytes, 0);
        assert_eq!(snapshot.external_bytes, 5 * GIB);
        assert_eq!(snapshot.available_bytes(), 7 * GIB);
    }

    #[test]
    fn over_budget_actual_is_retained_until_failed_load_cleans_up() {
        let mut ledger = ledger();
        ledger
            .reserve_external(DEVICE, "weights", 8 * GIB)
            .expect("planned load");
        assert!(ledger
            .reconcile_external(DEVICE, "weights", 13 * GIB)
            .is_err());
        assert_eq!(
            ledger.snapshot(DEVICE).expect("device").external_bytes,
            13 * GIB
        );
        assert!(ledger.reserve_external(DEVICE, "other", 1).is_err());

        ledger.release_external(DEVICE, "weights");
        assert_eq!(ledger.snapshot(DEVICE).expect("device").external_bytes, 0);
    }

    #[test]
    fn shared_allocation_is_charged_once_and_reference_counted() {
        let mut ledger = ledger();
        let (_, first_owns_charge) = ledger
            .reserve_external(DEVICE, "shared-weights", 7 * GIB)
            .expect("first replica");
        ledger
            .reconcile_external(DEVICE, "shared-weights", 6 * GIB)
            .expect("actual weights");
        let (snapshot, second_owns_charge) = ledger
            .reserve_external(DEVICE, "shared-weights", 7 * GIB)
            .expect("second replica");
        assert!(first_owns_charge);
        assert!(!second_owns_charge);
        assert_eq!(snapshot.external_bytes, 6 * GIB);

        ledger.release_external(DEVICE, "shared-weights");
        assert_eq!(
            ledger.snapshot(DEVICE).expect("device").external_bytes,
            6 * GIB
        );
        ledger.release_external(DEVICE, "shared-weights");
        assert_eq!(ledger.snapshot(DEVICE).expect("device").external_bytes, 0);
    }

    #[test]
    fn rejects_a_pool_larger_than_the_device_budget() {
        let mut ledger = DeviceBudgetLedger::default();
        assert!(ledger.insert_device(DEVICE, 4 * GIB, 5 * GIB).is_err());
    }
}
