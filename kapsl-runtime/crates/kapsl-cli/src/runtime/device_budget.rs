use super::memory::{MemoryAllocationClass, MemoryOwner};
use std::collections::HashMap;

pub(crate) const AUTO_POOL_ALIGNMENT_BYTES: usize = 2 * 1024 * 1024;
pub(crate) const AUTO_POOL_MIN_BYTES: usize = 256 * 1024 * 1024;
pub(crate) const AUTO_POOL_GROWTH_FLOOR_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AutoPoolSizingInput {
    pub(crate) safe_budget_bytes: usize,
    pub(crate) planned_external_bytes: usize,
    pub(crate) minimum_pool_bytes: usize,
    pub(crate) unpooled_reserve_bytes: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AutoPoolSizingDecision {
    pub(crate) capacity_bytes: Option<usize>,
    pub(crate) planned_external_bytes: usize,
    pub(crate) unpooled_reserve_bytes: usize,
    pub(crate) reason: Option<String>,
}

/// Choose one immutable backing allocation after retaining room for known
/// external weights and for unpooled scratch/native-KV/future model loads.
/// The caller has already removed the driver safety band from `safe_budget`.
pub(crate) fn choose_auto_pool_capacity(input: AutoPoolSizingInput) -> AutoPoolSizingDecision {
    let available_after_external = input
        .safe_budget_bytes
        .saturating_sub(input.planned_external_bytes);
    if input.planned_external_bytes >= input.safe_budget_bytes {
        return AutoPoolSizingDecision {
            capacity_bytes: None,
            planned_external_bytes: input.planned_external_bytes,
            unpooled_reserve_bytes: 0,
            reason: Some(format!(
                "planned external memory ({}) leaves no safe pool capacity in the {}-byte budget",
                input.planned_external_bytes, input.safe_budget_bytes
            )),
        };
    }

    let required_pool_bytes = AUTO_POOL_MIN_BYTES.max(input.minimum_pool_bytes);
    let default_reserve = (input.safe_budget_bytes / 5)
        .max(1024 * 1024 * 1024)
        .min(input.safe_budget_bytes / 3);
    let requested_reserve = input.unpooled_reserve_bytes.unwrap_or(default_reserve);
    let max_reserve = available_after_external.saturating_sub(required_pool_bytes);
    if input.unpooled_reserve_bytes.is_some() && requested_reserve > max_reserve {
        return AutoPoolSizingDecision {
            capacity_bytes: None,
            planned_external_bytes: input.planned_external_bytes,
            unpooled_reserve_bytes: requested_reserve,
            reason: Some(format!(
                "the explicit {requested_reserve}-byte unpooled reserve leaves less than the {}-byte automatic-pool minimum",
                required_pool_bytes
            )),
        };
    }
    let retained_reserve = requested_reserve.min(max_reserve);
    let allocatable_capacity = available_after_external.saturating_sub(retained_reserve);
    // Automatic mode is demand-sized. Retain bounded growth room instead of
    // eagerly converting every otherwise-free byte into an immutable backing.
    let growth_headroom = (required_pool_bytes / 4).max(AUTO_POOL_GROWTH_FLOOR_BYTES);
    let raw_capacity = required_pool_bytes
        .saturating_add(growth_headroom)
        .min(allocatable_capacity);
    let capacity = raw_capacity / AUTO_POOL_ALIGNMENT_BYTES * AUTO_POOL_ALIGNMENT_BYTES;

    if capacity < required_pool_bytes {
        return AutoPoolSizingDecision {
            capacity_bytes: None,
            planned_external_bytes: input.planned_external_bytes,
            unpooled_reserve_bytes: retained_reserve,
            reason: Some(format!(
                "only {capacity} aligned bytes remain, below the {}-byte automatic-pool minimum",
                required_pool_bytes
            )),
        };
    }

    AutoPoolSizingDecision {
        capacity_bytes: Some(capacity),
        planned_external_bytes: input.planned_external_bytes,
        unpooled_reserve_bytes: retained_reserve,
        reason: None,
    }
}

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
    class: MemoryAllocationClass,
    owners: HashMap<MemoryOwner, usize>,
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
        owner: MemoryOwner,
        class: MemoryAllocationClass,
    ) -> Result<(DeviceBudgetSnapshot, bool), String> {
        let device = self
            .devices
            .get_mut(&device_id)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;
        if let Some(allocation) = device.allocations.get_mut(allocation_id) {
            if allocation.class != class {
                return Err(format!(
                    "CUDA device {device_id} external allocation `{allocation_id}` is already classified as {}, not {} for {}",
                    allocation.class, class, owner
                ));
            }
            let refs = allocation.owners.entry(owner).or_default();
            *refs = refs.saturating_add(1);
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
        let mut owners = HashMap::new();
        owners.insert(owner, 1);
        device.allocations.insert(
            allocation_id.to_string(),
            ExternalAllocation {
                bytes,
                class,
                owners,
                planned: true,
            },
        );
        Ok((snapshot(device), true))
    }

    /// Update the one process-lifetime backing allocation after deferred/auto
    /// sizing. This is intentionally not a general resize API: callers set it
    /// once before publishing the pool to any backend.
    pub(crate) fn set_pooled_bytes(
        &mut self,
        device_id: usize,
        pooled_bytes: usize,
    ) -> Result<DeviceBudgetSnapshot, String> {
        let device = self
            .devices
            .get_mut(&device_id)
            .ok_or_else(|| format!("CUDA device {device_id} has no runtime memory authority"))?;
        let external = snapshot(device)
            .planned_external_bytes
            .saturating_add(snapshot(device).external_bytes);
        let projected = pooled_bytes.saturating_add(external);
        if projected > device.budget_bytes {
            return Err(format!(
                "CUDA device {device_id} pool reserves {pooled_bytes} bytes with {external} external bytes already charged, exceeding its {}-byte safe budget",
                device.budget_bytes
            ));
        }
        device.pooled_bytes = pooled_bytes;
        Ok(snapshot(device))
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

    pub(crate) fn release_external(
        &mut self,
        device_id: usize,
        allocation_id: &str,
        owner: MemoryOwner,
    ) {
        if let Some(device) = self.devices.get_mut(&device_id) {
            let remove = if let Some(allocation) = device.allocations.get_mut(allocation_id) {
                let remove_owner = allocation.owners.get_mut(&owner).is_some_and(|refs| {
                    *refs = refs.saturating_sub(1);
                    *refs == 0
                });
                if remove_owner {
                    allocation.owners.remove(&owner);
                }
                allocation.owners.is_empty()
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
    const OWNER: MemoryOwner = MemoryOwner::new(7, 0);

    fn reserve(
        ledger: &mut DeviceBudgetLedger,
        allocation_id: &str,
        bytes: usize,
    ) -> Result<(DeviceBudgetSnapshot, bool), String> {
        reserve_for(ledger, allocation_id, bytes, OWNER)
    }

    fn reserve_for(
        ledger: &mut DeviceBudgetLedger,
        allocation_id: &str,
        bytes: usize,
        owner: MemoryOwner,
    ) -> Result<(DeviceBudgetSnapshot, bool), String> {
        ledger.reserve_external(
            DEVICE,
            allocation_id,
            bytes,
            owner,
            MemoryAllocationClass::PersistentWeights,
        )
    }

    fn ledger() -> DeviceBudgetLedger {
        let mut ledger = DeviceBudgetLedger::default();
        ledger
            .insert_device(DEVICE, 20 * GIB, 8 * GIB)
            .expect("insert test device");
        ledger
    }

    #[test]
    fn auto_pool_subtracts_weights_and_retains_unpooled_headroom() {
        let decision = choose_auto_pool_capacity(AutoPoolSizingInput {
            safe_budget_bytes: 20 * GIB,
            planned_external_bytes: 6 * GIB,
            minimum_pool_bytes: 0,
            unpooled_reserve_bytes: Some(2 * GIB),
        });
        assert_eq!(decision.capacity_bytes, Some(512 * 1024 * 1024));
        assert_eq!(decision.unpooled_reserve_bytes, 2 * GIB);
    }

    #[test]
    fn auto_pool_default_reserve_is_bounded_and_aligned() {
        let safe_budget = 24 * GIB;
        let planned_external = 5 * GIB;
        let decision = choose_auto_pool_capacity(AutoPoolSizingInput {
            safe_budget_bytes: safe_budget,
            planned_external_bytes: planned_external,
            minimum_pool_bytes: 0,
            unpooled_reserve_bytes: None,
        });
        let capacity = decision.capacity_bytes.expect("automatic pool");
        let expected_reserve = (safe_budget / 5).max(GIB).min(safe_budget / 3);
        let expected_capacity = 512 * 1024 * 1024;
        assert_eq!(capacity % AUTO_POOL_ALIGNMENT_BYTES, 0);
        assert_eq!(decision.unpooled_reserve_bytes, expected_reserve);
        assert_eq!(capacity, expected_capacity);
    }

    #[test]
    fn auto_pool_disables_when_external_memory_consumes_budget() {
        let decision = choose_auto_pool_capacity(AutoPoolSizingInput {
            safe_budget_bytes: 8 * GIB,
            planned_external_bytes: 8 * GIB,
            minimum_pool_bytes: 0,
            unpooled_reserve_bytes: None,
        });
        assert_eq!(decision.capacity_bytes, None);
        assert!(decision.reason.unwrap().contains("leaves no safe pool"));
    }

    #[test]
    fn automatic_pool_minimum_boundary_is_inclusive() {
        let exact = choose_auto_pool_capacity(AutoPoolSizingInput {
            safe_budget_bytes: AUTO_POOL_MIN_BYTES,
            planned_external_bytes: 0,
            minimum_pool_bytes: 0,
            unpooled_reserve_bytes: Some(0),
        });
        assert_eq!(exact.capacity_bytes, Some(AUTO_POOL_MIN_BYTES));

        let below = choose_auto_pool_capacity(AutoPoolSizingInput {
            safe_budget_bytes: AUTO_POOL_MIN_BYTES - 1,
            planned_external_bytes: 0,
            minimum_pool_bytes: 0,
            unpooled_reserve_bytes: Some(0),
        });
        assert_eq!(below.capacity_bytes, None);
    }

    #[test]
    fn explicit_zero_unpooled_reserve_is_honored() {
        let decision = choose_auto_pool_capacity(AutoPoolSizingInput {
            safe_budget_bytes: 4 * GIB,
            planned_external_bytes: GIB,
            minimum_pool_bytes: 0,
            unpooled_reserve_bytes: Some(0),
        });
        assert_eq!(decision.capacity_bytes, Some(512 * 1024 * 1024));
        assert_eq!(decision.unpooled_reserve_bytes, 0);
    }

    #[test]
    fn explicit_unpooled_reserve_is_not_silently_reduced() {
        let decision = choose_auto_pool_capacity(AutoPoolSizingInput {
            safe_budget_bytes: 4 * GIB,
            planned_external_bytes: 3 * GIB,
            minimum_pool_bytes: 0,
            unpooled_reserve_bytes: Some(GIB),
        });
        assert_eq!(decision.capacity_bytes, None);
        assert_eq!(decision.unpooled_reserve_bytes, GIB);
        assert!(decision.reason.unwrap().contains("explicit"));
    }

    #[test]
    fn pooled_model_footprint_is_a_hard_minimum() {
        let decision = choose_auto_pool_capacity(AutoPoolSizingInput {
            safe_budget_bytes: 8 * GIB,
            planned_external_bytes: 2 * GIB,
            minimum_pool_bytes: 5 * GIB,
            unpooled_reserve_bytes: Some(2 * GIB),
        });
        assert_eq!(decision.capacity_bytes, None);
        assert!(decision.reason.unwrap().contains("5"));

        let decision = choose_auto_pool_capacity(AutoPoolSizingInput {
            safe_budget_bytes: 8 * GIB,
            planned_external_bytes: 2 * GIB,
            minimum_pool_bytes: 5 * GIB,
            unpooled_reserve_bytes: None,
        });
        assert!(decision.capacity_bytes.unwrap() >= 5 * GIB);
    }

    #[test]
    fn pool_and_external_reservations_share_one_budget() {
        let mut ledger = ledger();
        let (snapshot, owns_charge) =
            reserve(&mut ledger, "weights-a", 12 * GIB).expect("exact fit should be admitted");
        assert!(owns_charge);
        assert_eq!(snapshot.used_bytes(), 20 * GIB);
        assert_eq!(snapshot.available_bytes(), 0);
        assert!(reserve(&mut ledger, "weights-b", 1).is_err());
    }

    #[test]
    fn deferred_pool_charge_accounts_for_existing_external_bytes() {
        let mut ledger = DeviceBudgetLedger::default();
        ledger.insert_device(DEVICE, 20 * GIB, 0).unwrap();
        reserve(&mut ledger, "weights", 6 * GIB).unwrap();

        let snapshot = ledger.set_pooled_bytes(DEVICE, 12 * GIB).unwrap();
        assert_eq!(snapshot.pooled_bytes, 12 * GIB);
        assert_eq!(snapshot.available_bytes(), 2 * GIB);

        let error = ledger
            .set_pooled_bytes(DEVICE, 15 * GIB)
            .expect_err("pool plus weights must remain within budget");
        assert!(error.contains("exceeding"), "{error}");
        assert_eq!(ledger.snapshot(DEVICE).unwrap().pooled_bytes, 12 * GIB);
    }

    #[test]
    fn reconciliation_releases_overestimated_headroom() {
        let mut ledger = ledger();
        reserve(&mut ledger, "weights", 8 * GIB).expect("planned load");
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
        reserve(&mut ledger, "weights", 8 * GIB).expect("planned load");
        assert!(ledger
            .reconcile_external(DEVICE, "weights", 13 * GIB)
            .is_err());
        assert_eq!(
            ledger.snapshot(DEVICE).expect("device").external_bytes,
            13 * GIB
        );
        assert!(reserve(&mut ledger, "other", 1).is_err());

        ledger.release_external(DEVICE, "weights", OWNER);
        assert_eq!(ledger.snapshot(DEVICE).expect("device").external_bytes, 0);
    }

    #[test]
    fn shared_allocation_is_charged_once_and_reference_counted() {
        let mut ledger = ledger();
        let first_owner = MemoryOwner::new(7, 0);
        let second_owner = MemoryOwner::new(7, 1);
        let (_, first_owns_charge) =
            reserve_for(&mut ledger, "shared-weights", 7 * GIB, first_owner)
                .expect("first replica");
        ledger
            .reconcile_external(DEVICE, "shared-weights", 6 * GIB)
            .expect("actual weights");
        let (snapshot, second_owns_charge) =
            reserve_for(&mut ledger, "shared-weights", 7 * GIB, second_owner)
                .expect("second replica");
        assert!(first_owns_charge);
        assert!(!second_owns_charge);
        assert_eq!(snapshot.external_bytes, 6 * GIB);

        let allocation = ledger.devices[&DEVICE]
            .allocations
            .get("shared-weights")
            .unwrap();
        assert_eq!(allocation.class, MemoryAllocationClass::PersistentWeights);
        assert!(allocation.owners.contains_key(&first_owner));
        assert!(allocation.owners.contains_key(&second_owner));

        ledger.release_external(DEVICE, "shared-weights", first_owner);
        assert_eq!(
            ledger.snapshot(DEVICE).expect("device").external_bytes,
            6 * GIB
        );
        ledger.release_external(DEVICE, "shared-weights", second_owner);
        assert_eq!(ledger.snapshot(DEVICE).expect("device").external_bytes, 0);
    }

    #[test]
    fn swap_peak_requires_room_for_active_and_target_weights() {
        let mut ledger = ledger();
        reserve(&mut ledger, "active-weights", 7 * GIB).expect("active model plan");
        ledger
            .reconcile_external(DEVICE, "active-weights", 7 * GIB)
            .expect("active model actual");

        assert!(reserve(&mut ledger, "swap-peak-1", 6 * GIB).is_err());
        let snapshot = ledger.snapshot(DEVICE).expect("device");
        assert_eq!(snapshot.external_bytes, 7 * GIB);
        assert_eq!(snapshot.planned_external_bytes, 0);

        let (snapshot, owns_charge) = reserve(&mut ledger, "swap-peak-2", 5 * GIB)
            .expect("active plus target is an exact fit");
        assert!(owns_charge);
        assert_eq!(snapshot.external_bytes, 7 * GIB);
        assert_eq!(snapshot.planned_external_bytes, 5 * GIB);
        assert_eq!(snapshot.available_bytes(), 0);

        ledger.release_external(DEVICE, "swap-peak-2", OWNER);
        assert_eq!(
            ledger.snapshot(DEVICE).expect("device").available_bytes(),
            5 * GIB
        );
    }

    #[test]
    fn rejects_a_pool_larger_than_the_device_budget() {
        let mut ledger = DeviceBudgetLedger::default();
        assert!(ledger.insert_device(DEVICE, 4 * GIB, 5 * GIB).is_err());
    }
}
