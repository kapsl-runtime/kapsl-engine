//! Runtime region identity and physical observations, independent of CUDA.
//!
//! This is the first migration layer, not a second allocator or budget ledger.
//! Existing owners retain backing and completion contracts. Weak registrations
//! let us observe them without extending a backing's lifetime past its charge.

use super::{MemoryAllocationClass, MemoryOwner};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

/// Opaque, process-local identity. Neither pool nor region IDs are reused.
/// This is an observation identity, not permission to free an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct GpuRegionId {
    pool: u64,
    region: u64,
}

impl fmt::Display for GpuRegionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.pool, self.region)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuRegionKind {
    Arena,
    Ipc,
    Vmm,
}

impl fmt::Display for GpuRegionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Arena => "arena",
            Self::Ipc => "ipc",
            Self::Vmm => "vmm",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GpuRegionIsolation {
    /// In-process consumers may share this backing. It is never an isolated
    /// participant export, regardless of how many of its extents are free.
    Local,
    Participant {
        owner: MemoryOwner,
        participant_id: String,
        binding_id: String,
        generation: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuRegionOwnerUsage {
    /// Unscoped provider callbacks remain explicitly unattributed.
    pub(crate) owner: Option<MemoryOwner>,
    pub(crate) backend: &'static str,
    pub(crate) class: MemoryAllocationClass,
    pub(crate) logical_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuRegionUsage {
    /// All physical backing still held, including unmapped VMM handles.
    pub(crate) committed_bytes: usize,
    /// Parent-context mapped bytes; this does not imply worker/scheduler readiness.
    pub(crate) mapped_bytes: usize,
    /// Explicit VMM reservation; zero for fixed allocations.
    pub(crate) virtual_reserved_bytes: usize,
    /// None when the participant owns the block allocator. Unknown is not zero.
    pub(crate) logical_allocated_bytes: Option<usize>,
    pub(crate) reusable_bytes: Option<usize>,
    pub(crate) largest_free_range_bytes: Option<usize>,
    pub(crate) owners: Vec<GpuRegionOwnerUsage>,
}

impl GpuRegionUsage {
    pub(crate) fn exported(
        committed_bytes: usize,
        mapped_bytes: usize,
        virtual_reserved_bytes: usize,
    ) -> Self {
        Self {
            committed_bytes,
            mapped_bytes,
            virtual_reserved_bytes,
            logical_allocated_bytes: None,
            reusable_bytes: None,
            largest_free_range_bytes: None,
            owners: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuRegionSnapshot {
    pub(crate) id: GpuRegionId,
    pub(crate) device_id: usize,
    pub(crate) kind: GpuRegionKind,
    pub(crate) isolation: GpuRegionIsolation,
    pub(crate) usage: GpuRegionUsage,
}

/// Read physical state under the backing's own lock, without CUDA calls.
/// Return None only after successful physical release. A failed unmap/free
/// must continue reporting every retained handle, even if mapped bytes is zero.
pub(crate) trait GpuRegionSource: Send + Sync {
    fn region_usage(&self) -> Option<GpuRegionUsage>;
}

struct RegionEntry {
    kind: GpuRegionKind,
    isolation: GpuRegionIsolation,
    backing: Weak<dyn GpuRegionSource>,
}

/// One registry per managed device, including devices without a local arena.
/// Registration never reserves bytes: arenas use the existing pooled charge,
/// and exported regions retain their existing authority leases.
pub(crate) struct GpuRegionRegistry {
    device_id: usize,
    pool_id: u64,
    next_region_id: AtomicU64,
    entries: Mutex<BTreeMap<GpuRegionId, RegionEntry>>,
}

fn next_id(counter: &AtomicU64) -> u64 {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("GPU region identity exhausted")
}

impl GpuRegionRegistry {
    pub(crate) fn new(device_id: usize) -> Self {
        Self {
            device_id,
            pool_id: next_id(&NEXT_POOL_ID),
            next_region_id: AtomicU64::new(1),
            entries: Mutex::new(BTreeMap::new()),
        }
    }

    pub(crate) fn register(
        &self,
        kind: GpuRegionKind,
        isolation: GpuRegionIsolation,
        backing: &Arc<dyn GpuRegionSource>,
    ) -> GpuRegionId {
        let id = GpuRegionId {
            pool: self.pool_id,
            region: next_id(&self.next_region_id),
        };
        let mut entries = self.entries.lock();
        entries.retain(|_, entry| entry.backing.strong_count() != 0);
        entries.insert(
            id,
            RegionEntry {
                kind,
                isolation,
                backing: Arc::downgrade(backing),
            },
        );
        id
    }

    pub(crate) fn snapshots(&self) -> Vec<GpuRegionSnapshot> {
        // Never hold the registry lock while acquiring a backing's lock.
        let entries: Vec<_> = self
            .entries
            .lock()
            .iter()
            .map(|(&id, entry)| {
                (
                    id,
                    entry.kind,
                    entry.isolation.clone(),
                    entry.backing.clone(),
                )
            })
            .collect();
        let mut snapshots = Vec::with_capacity(entries.len());
        let mut retired = Vec::new();
        for (id, kind, isolation, backing) in entries {
            match backing.upgrade().and_then(|backing| backing.region_usage()) {
                Some(usage) => snapshots.push(GpuRegionSnapshot {
                    id,
                    device_id: self.device_id,
                    kind,
                    isolation,
                    usage,
                }),
                None => retired.push(id),
            }
        }
        let mut entries = self.entries.lock();
        for id in retired {
            entries.remove(&id);
        }
        snapshots
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::memory::device_budget::DeviceBudgetLedger;

    const GIB: usize = 1024 * 1024 * 1024;

    struct TestBacking(Mutex<Option<GpuRegionUsage>>);

    impl GpuRegionSource for TestBacking {
        fn region_usage(&self) -> Option<GpuRegionUsage> {
            self.0.lock().clone()
        }
    }

    fn register(
        registry: &GpuRegionRegistry,
        kind: GpuRegionKind,
        isolation: GpuRegionIsolation,
        backing: &Arc<TestBacking>,
    ) -> GpuRegionId {
        registry.register(
            kind,
            isolation,
            &(backing.clone() as Arc<dyn GpuRegionSource>),
        )
    }

    fn participant(generation: u64) -> GpuRegionIsolation {
        GpuRegionIsolation::Participant {
            owner: MemoryOwner::external_kv(3).unwrap(),
            participant_id: "vllm-worker".into(),
            binding_id: format!("kv:{generation}:0"),
            generation,
        }
    }

    fn arena_usage(logical_bytes: usize) -> GpuRegionUsage {
        GpuRegionUsage {
            committed_bytes: 4 * GIB,
            mapped_bytes: 4 * GIB,
            virtual_reserved_bytes: 0,
            logical_allocated_bytes: Some(logical_bytes),
            reusable_bytes: Some(4 * GIB - logical_bytes),
            largest_free_range_bytes: Some(4 * GIB - logical_bytes),
            owners: vec![GpuRegionOwnerUsage {
                owner: Some(MemoryOwner::new(1, 0)),
                backend: "onnx",
                class: MemoryAllocationClass::TransientWorkspace,
                logical_bytes,
            }],
        }
    }

    #[test]
    fn logical_free_and_virtual_capacity_do_not_create_physical_headroom() {
        let registry = GpuRegionRegistry::new(0);
        let arena = Arc::new(TestBacking(Mutex::new(Some(arena_usage(2 * GIB)))));
        let vmm = Arc::new(TestBacking(Mutex::new(Some(GpuRegionUsage::exported(
            2 * GIB,
            2 * GIB,
            8 * GIB,
        )))));
        register(
            &registry,
            GpuRegionKind::Arena,
            GpuRegionIsolation::Local,
            &arena,
        );
        register(&registry, GpuRegionKind::Vmm, participant(1), &vmm);

        let mut budget = DeviceBudgetLedger::default();
        budget.insert_device(0, 20 * GIB, 4 * GIB).unwrap();
        budget
            .reserve_external(
                0,
                "weights",
                10 * GIB,
                MemoryOwner::new(1, 0),
                MemoryAllocationClass::PersistentWeights,
            )
            .unwrap();
        budget
            .reserve_external(
                0,
                "kv",
                2 * GIB,
                MemoryOwner::external_kv(3).unwrap(),
                MemoryAllocationClass::KvCache,
            )
            .unwrap();
        assert_eq!(budget.snapshot(0).unwrap().available_bytes(), 4 * GIB);
        let before = registry.snapshots();
        assert_eq!(
            before
                .iter()
                .map(|r| r.usage.committed_bytes)
                .sum::<usize>(),
            6 * GIB
        );
        assert_eq!(before[1].usage.virtual_reserved_bytes, 8 * GIB);
        assert_eq!(before[1].usage.logical_allocated_bytes, None);
        assert_eq!(before[1].usage.reusable_bytes, None);

        budget.reconcile_external(0, "kv", 5 * GIB).unwrap();
        *vmm.0.lock() = Some(GpuRegionUsage::exported(5 * GIB, 5 * GIB, 8 * GIB));
        *arena.0.lock() = Some(arena_usage(0));
        let after = registry.snapshots();
        assert_eq!(after[0].id, before[0].id);
        assert_eq!(after[0].usage.reusable_bytes, Some(4 * GIB));
        assert_eq!(after[0].usage.committed_bytes, 4 * GIB);
        assert_eq!(after[1].isolation, participant(1));
        assert_eq!(budget.snapshot(0).unwrap().available_bytes(), GIB);
        assert!(budget
            .reserve_external(
                0,
                "incompatible-ipc",
                2 * GIB,
                MemoryOwner::external_kv(4).unwrap(),
                MemoryAllocationClass::KvCache
            )
            .is_err());
    }

    #[test]
    fn unmapped_handles_remain_committed_until_physical_release() {
        let registry = GpuRegionRegistry::new(7);
        let backing = Arc::new(TestBacking(Mutex::new(Some(GpuRegionUsage::exported(
            2 * GIB,
            0,
            8 * GIB,
        )))));
        register(&registry, GpuRegionKind::Vmm, participant(4), &backing);
        let snapshot = registry.snapshots().pop().unwrap();
        assert_eq!(snapshot.device_id, 7);
        assert_eq!(snapshot.usage.committed_bytes, 2 * GIB);
        assert_eq!(snapshot.usage.mapped_bytes, 0);
        // A failed release leaves the source and charge live; success retires
        // the registration even if the coordinator still holds its Arc.
        assert_eq!(registry.snapshots(), vec![snapshot]);
        *backing.0.lock() = None;
        assert!(registry.snapshots().is_empty());
    }

    #[test]
    fn identities_survive_observation_and_are_never_reused() {
        let registry = GpuRegionRegistry::new(0);
        let other = GpuRegionRegistry::new(1);
        let backing = Arc::new(TestBacking(Mutex::new(Some(GpuRegionUsage::exported(
            64, 64, 0,
        )))));
        let first = register(&registry, GpuRegionKind::Ipc, participant(1), &backing);
        assert_eq!(registry.snapshots()[0].id, first);
        assert_eq!(Arc::strong_count(&backing), 1);
        let second = register(&other, GpuRegionKind::Ipc, participant(1), &backing);
        assert_ne!(first, second);
        drop(backing);
        assert!(registry.snapshots().is_empty());
        assert!(other.snapshots().is_empty());
        let replacement = Arc::new(TestBacking(Mutex::new(Some(GpuRegionUsage::exported(
            64, 64, 0,
        )))));
        let third = register(&registry, GpuRegionKind::Ipc, participant(2), &replacement);
        assert_ne!(first, third);
        assert_eq!(registry.snapshots()[0].isolation, participant(2));
    }

    #[test]
    fn backing_is_sampled_without_holding_the_registry_lock() {
        struct InspectLock(Arc<GpuRegionRegistry>);
        impl GpuRegionSource for InspectLock {
            fn region_usage(&self) -> Option<GpuRegionUsage> {
                assert!(self.0.entries.try_lock().is_some());
                Some(GpuRegionUsage::exported(64, 64, 0))
            }
        }
        let registry = Arc::new(GpuRegionRegistry::new(0));
        let backing: Arc<dyn GpuRegionSource> = Arc::new(InspectLock(registry.clone()));
        registry.register(GpuRegionKind::Ipc, participant(1), &backing);
        assert_eq!(registry.snapshots().len(), 1);
    }
}
