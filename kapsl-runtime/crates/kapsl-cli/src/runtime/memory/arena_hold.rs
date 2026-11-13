//! Keeps an ONNX model's CUDA arena claim at its peak.
//!
//! ORT's own arena reports the bytes it holds *now*. Right after load that is
//! small (ResNet-50: 815 MB idle) and under load it grows (1,623 MB), but the
//! device budget is what lets a second model in. If the claim follows the idle
//! figure down, the second model is admitted against memory the first one
//! reclaims as soon as it is busy. So the claim on each device never drops
//! below the larger of the peak plan it was admitted at and the highest total
//! ever reported; `observed` and `committed` keep reporting the real bytes,
//! only the reservation is held.
//!
//! The floor is on the device total, not on each row. The backend and the plan
//! split the same bytes differently between weights and workspace (ResNet-50,
//! run 8: the arena put 1.09 GB in the weights row against a 0.41 GB planned
//! share, while the workspace row measured 0.67 GB of its 1.61 GB plan), so
//! flooring rows separately summed to 2.71 GB against a 2.02 GB plan. Each row
//! keeps its reported bytes; any shortfall to the floor goes on the
//! transient-workspace row.

use super::{
    BackendMemoryAllocationClass, BackendMemoryAllocationSource, BackendMemoryDomain,
    BackendMemoryReport, MemoryClaim, MemoryClaimSource, MemoryDomain,
};
use std::collections::HashMap;

/// The held allocations of one admitted plan: which CUDA device each belongs
/// to, the peak plan per device, and the highest total reported per device.
#[derive(Debug, Clone, Default)]
pub(crate) struct ArenaHold {
    device_of: HashMap<String, usize>,
    plan_total: HashMap<usize, usize>,
    high_water_total: HashMap<usize, usize>,
}

impl ArenaHold {
    /// Hold the backend-managed CUDA claims of an admitted plan: their planned
    /// (peak) bytes per device become the floor. `None` when the plan has no
    /// such claim.
    pub(crate) fn from_plan(claims: &[MemoryClaim]) -> Option<Self> {
        let mut hold = Self::default();
        for claim in claims {
            let MemoryDomain::Cuda { device_id } = claim.domain else {
                continue;
            };
            if let MemoryClaimSource::External { allocation_id } = &claim.source {
                hold.device_of.insert(allocation_id.clone(), device_id);
                let total = hold.plan_total.entry(device_id).or_insert(0usize);
                *total = total.saturating_add(claim.bytes);
            }
        }
        (!hold.device_of.is_empty()).then_some(hold)
    }

    /// The bytes a device's held claims may not total less than: the larger of
    /// the peak plan and the highest total reported so far.
    #[cfg(test)]
    pub(crate) fn floor_total(&self, device_id: usize) -> usize {
        let plan = self.plan_total.get(&device_id).copied().unwrap_or(0);
        let seen = self.high_water_total.get(&device_id).copied().unwrap_or(0);
        plan.max(seen)
    }

    /// Record the report's real totals, then return the report to reserve
    /// against: per device, the held rows keep their reported bytes and any
    /// shortfall to the floor is added to the transient-workspace row (the
    /// first held row if the report has none). Allocations the plan did not
    /// hold, and every non-CUDA or runtime-managed one, pass through unchanged.
    pub(crate) fn hold(&mut self, report: &BackendMemoryReport) -> BackendMemoryReport {
        let mut held = report.clone();
        let mut reported: HashMap<usize, usize> = HashMap::new();
        let mut target: HashMap<usize, usize> = HashMap::new();
        for (index, allocation) in held.allocations.iter().enumerate() {
            let Some(device_id) = self.held_device(allocation) else {
                continue;
            };
            let total = reported.entry(device_id).or_insert(0usize);
            *total = total.saturating_add(allocation.bytes);
            let slot = target.entry(device_id).or_insert(index);
            let workspace = |i: usize| {
                report.allocations[i].class == BackendMemoryAllocationClass::TransientWorkspace
            };
            if !workspace(*slot) && workspace(index) {
                *slot = index;
            }
        }
        for (device_id, total) in reported {
            let seen = self.high_water_total.entry(device_id).or_insert(0);
            *seen = (*seen).max(total);
            let floor = self
                .plan_total
                .get(&device_id)
                .copied()
                .unwrap_or(0)
                .max(*seen);
            if let Some(&index) = target.get(&device_id) {
                held.allocations[index].bytes += floor.saturating_sub(total);
            }
        }
        held
    }

    fn held_device(&self, allocation: &kapsl_engine_api::MemoryAllocation) -> Option<usize> {
        let BackendMemoryDomain::Cuda { device_id } = &allocation.domain else {
            return None;
        };
        if allocation.source != BackendMemoryAllocationSource::BackendManaged {
            return None;
        }
        self.device_of
            .get(&allocation.allocation_id)
            .filter(|device| *device == device_id)
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::memory::{MemoryAllocationClass, MemoryOwner};
    use kapsl_engine_api::{MemoryAllocation, MemoryAllocationClass as BackendClass};

    const MIB: usize = 1024 * 1024;

    // ResNet-50 on ORT's own arena, 4 sessions (run 8, cuda:0).
    const WEIGHTS: &str = "onnx-session:0:1";
    const WORKSPACE: &str = "onnx-session:0:1:workspace";
    const PLAN_WEIGHTS: usize = 410_306_372;
    const PLAN_WORKSPACE: usize = 1_610_612_736;
    const PLAN_TOTAL: usize = 2_020_919_108;
    // Measured at idle and with 4 busy sessions.
    const IDLE_WEIGHTS: usize = 559_939_584;
    const IDLE_WORKSPACE: usize = 67_108_864;
    const BUSY_WEIGHTS: usize = 1_094_713_344;
    const BUSY_WORKSPACE: usize = 671_088_640;
    const _: () = assert!(BUSY_WEIGHTS + BUSY_WORKSPACE < PLAN_TOTAL);
    // What per-row flooring reserved in run 8.
    const RUN8_PER_ROW_SUM: usize = 2_705_326_080;

    fn claim(
        id: &str,
        device_id: usize,
        class: MemoryAllocationClass,
        bytes: usize,
    ) -> MemoryClaim {
        MemoryClaim {
            domain: MemoryDomain::Cuda { device_id },
            owner: MemoryOwner::new(1, 0),
            class,
            bytes,
            source: MemoryClaimSource::External {
                allocation_id: id.to_string(),
            },
        }
    }

    fn resnet_plan() -> Vec<MemoryClaim> {
        vec![
            claim(
                WEIGHTS,
                0,
                MemoryAllocationClass::PersistentWeights,
                PLAN_WEIGHTS,
            ),
            claim(
                WORKSPACE,
                0,
                MemoryAllocationClass::TransientWorkspace,
                PLAN_WORKSPACE,
            ),
        ]
    }

    fn row(
        id: &str,
        device_id: usize,
        class: BackendClass,
        source: BackendMemoryAllocationSource,
        bytes: usize,
    ) -> MemoryAllocation {
        MemoryAllocation {
            allocation_id: id.to_string(),
            domain: BackendMemoryDomain::Cuda { device_id },
            class,
            source,
            bytes,
        }
    }

    fn weights(bytes: usize) -> MemoryAllocation {
        row(
            WEIGHTS,
            0,
            BackendClass::PersistentWeights,
            BackendMemoryAllocationSource::BackendManaged,
            bytes,
        )
    }

    fn workspace(bytes: usize) -> MemoryAllocation {
        row(
            WORKSPACE,
            0,
            BackendClass::TransientWorkspace,
            BackendMemoryAllocationSource::BackendManaged,
            bytes,
        )
    }

    fn report(allocations: Vec<MemoryAllocation>) -> BackendMemoryReport {
        BackendMemoryReport { allocations }
    }

    fn bytes_of(report: &BackendMemoryReport, id: &str) -> usize {
        report
            .allocations
            .iter()
            .find(|allocation| allocation.allocation_id == id)
            .map(|allocation| allocation.bytes)
            .expect("allocation in report")
    }

    fn total(report: &BackendMemoryReport) -> usize {
        report.allocations.iter().map(|a| a.bytes).sum()
    }

    #[test]
    fn a_plan_without_backend_managed_cuda_claims_is_not_held() {
        let runtime_claim = MemoryClaim::runtime(
            MemoryDomain::Cuda { device_id: 0 },
            MemoryOwner::new(1, 0),
            MemoryAllocationClass::KvCache,
            64 * MIB,
        );
        let mut host_external = claim(
            "host",
            0,
            MemoryAllocationClass::TransientWorkspace,
            64 * MIB,
        );
        host_external.domain = MemoryDomain::Host;
        assert!(ArenaHold::from_plan(&[runtime_claim, host_external]).is_none());
    }

    #[test]
    fn resnet50s_real_split_reserves_the_plan_total_not_the_sum_of_row_maxima() {
        let mut hold = ArenaHold::from_plan(&resnet_plan()).unwrap();
        assert_eq!(PLAN_WEIGHTS + PLAN_WORKSPACE, PLAN_TOTAL);

        // Idle: the arena reports 560 MB of weights and 67 MB of workspace.
        let held = hold.hold(&report(vec![
            weights(IDLE_WEIGHTS),
            workspace(IDLE_WORKSPACE),
        ]));
        assert_eq!(total(&held), PLAN_TOTAL);
        assert_eq!(
            bytes_of(&held, WEIGHTS),
            IDLE_WEIGHTS,
            "each row keeps its report"
        );
        assert_eq!(
            bytes_of(&held, WORKSPACE),
            PLAN_TOTAL - IDLE_WEIGHTS,
            "the shortfall is on the workspace row"
        );

        // Busy: the weights row is 1.09 GB, far over its 0.41 GB planned share.
        let held = hold.hold(&report(vec![
            weights(BUSY_WEIGHTS),
            workspace(BUSY_WORKSPACE),
        ]));
        assert_eq!(bytes_of(&held, WEIGHTS), BUSY_WEIGHTS);
        assert_eq!(bytes_of(&held, WORKSPACE), PLAN_TOTAL - BUSY_WEIGHTS);
        assert_eq!(total(&held), PLAN_TOTAL, "not the run 8 per-row sum");
        assert_ne!(total(&held), RUN8_PER_ROW_SUM);

        // Back to idle: the same total, and the real report is untouched.
        let input = report(vec![weights(IDLE_WEIGHTS), workspace(IDLE_WORKSPACE)]);
        let held = hold.hold(&input);
        assert_eq!(total(&held), PLAN_TOTAL);
        assert_eq!(
            total(&input),
            IDLE_WEIGHTS + IDLE_WORKSPACE,
            "observed stays real"
        );
        assert_eq!(hold.floor_total(0), PLAN_TOTAL);
    }

    #[test]
    fn the_high_water_total_governs_once_it_passes_the_plan() {
        let mut hold = ArenaHold::from_plan(&resnet_plan()).unwrap();
        let reported = |w: usize, ws: usize| report(vec![weights(w), workspace(ws)]);
        // The arena outgrows the plan: 1.5 GB of weights, 1.0 GB of workspace.
        let held = hold.hold(&reported(1500 * MIB, 1000 * MIB));
        assert_eq!(total(&held), 2500 * MIB, "no shortfall: rows as reported");
        assert_eq!(bytes_of(&held, WORKSPACE), 1000 * MIB);
        // It later falls back: the total stays at the highest sum seen.
        let held = hold.hold(&reported(560 * MIB, 67 * MIB));
        assert_eq!(total(&held), 2500 * MIB);
        assert_eq!(hold.floor_total(0), 2500 * MIB);
    }

    #[test]
    fn a_report_without_a_workspace_row_puts_the_shortfall_on_the_first_held_row() {
        let mut hold = ArenaHold::from_plan(&resnet_plan()).unwrap();
        let held = hold.hold(&report(vec![weights(IDLE_WEIGHTS)]));
        assert_eq!(bytes_of(&held, WEIGHTS), PLAN_TOTAL);
    }

    #[test]
    fn only_held_rows_count_and_only_the_workspace_row_grows() {
        let mut hold = ArenaHold::from_plan(&resnet_plan()).unwrap();
        let input = report(vec![
            weights(IDLE_WEIGHTS),
            // Not in the plan: follows the backend and is not part of the total.
            row(
                "other",
                0,
                BackendClass::TransientWorkspace,
                BackendMemoryAllocationSource::BackendManaged,
                100 * MIB,
            ),
            // Runtime-managed (pool) rows are never held.
            row(
                WORKSPACE,
                0,
                BackendClass::TransientWorkspace,
                BackendMemoryAllocationSource::RuntimeManaged,
                100 * MIB,
            ),
            // The same id on another device is not this claim.
            row(
                WORKSPACE,
                1,
                BackendClass::TransientWorkspace,
                BackendMemoryAllocationSource::BackendManaged,
                100 * MIB,
            ),
            workspace(IDLE_WORKSPACE),
        ]);
        let held = hold.hold(&input);
        assert_eq!(held.allocations[1].bytes, 100 * MIB);
        assert_eq!(held.allocations[2].bytes, 100 * MIB);
        assert_eq!(held.allocations[3].bytes, 100 * MIB);
        assert_eq!(held.allocations[0].bytes, IDLE_WEIGHTS);
        assert_eq!(held.allocations[4].bytes, PLAN_TOTAL - IDLE_WEIGHTS);
    }

    #[test]
    fn each_device_has_its_own_floor() {
        // A model split over two cards: 1,000 MB planned on each.
        let plan = [
            claim(
                "w:0",
                0,
                MemoryAllocationClass::PersistentWeights,
                200 * MIB,
            ),
            claim(
                "ws:0",
                0,
                MemoryAllocationClass::TransientWorkspace,
                800 * MIB,
            ),
            claim(
                "w:1",
                1,
                MemoryAllocationClass::PersistentWeights,
                200 * MIB,
            ),
            claim(
                "ws:1",
                1,
                MemoryAllocationClass::TransientWorkspace,
                800 * MIB,
            ),
        ];
        let mut hold = ArenaHold::from_plan(&plan).unwrap();
        let on = |id: &str, device: usize, class: BackendClass, bytes: usize| {
            row(
                id,
                device,
                class,
                BackendMemoryAllocationSource::BackendManaged,
                bytes,
            )
        };
        let held = hold.hold(&report(vec![
            on("w:0", 0, BackendClass::PersistentWeights, 600 * MIB),
            on("ws:0", 0, BackendClass::TransientWorkspace, 100 * MIB),
            on("w:1", 1, BackendClass::PersistentWeights, 300 * MIB),
            on("ws:1", 1, BackendClass::TransientWorkspace, 100 * MIB),
        ]));
        assert_eq!(
            bytes_of(&held, "ws:0"),
            400 * MIB,
            "device 0 total 1,000 MB"
        );
        assert_eq!(
            bytes_of(&held, "ws:1"),
            700 * MIB,
            "device 1 total 1,000 MB"
        );
        assert_eq!(hold.floor_total(0), 1000 * MIB);
        assert_eq!(hold.floor_total(1), 1000 * MIB);
    }

    #[test]
    fn an_allocation_missing_from_a_report_keeps_its_floor() {
        let mut hold = ArenaHold::from_plan(&resnet_plan()).unwrap();
        let _ = hold.hold(&report(vec![weights(1500 * MIB), workspace(1000 * MIB)]));
        let held = hold.hold(&report(vec![]));
        assert!(held.allocations.is_empty());
        assert_eq!(hold.floor_total(0), 2500 * MIB);
    }
}
