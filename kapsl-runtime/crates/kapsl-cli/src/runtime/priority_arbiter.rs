use super::*;
use std::cmp::Reverse;
use std::sync::Weak;

#[derive(Debug, Clone)]
struct PriorityRegistrationRecord {
    owner: MemoryOwner,
    weight: u32,
    domains: Vec<MemoryDomain>,
}

/// One lower-priority model whose currently leased footprint overlaps a failed
/// admission. Selection is policy-only; lifecycle code performs and verifies
/// the actual unload through `ModelManager` and `MemoryAuthority`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MemoryReclaimCandidate {
    pub(crate) model_id: u32,
    pub(crate) priority_weight: u32,
    pub(crate) reclaimable_bytes: usize,
}

/// Backend-neutral priority policy built over the authority snapshot.
///
/// Every loaded runtime engine registers its model/replica and target domains,
/// regardless of backend. Candidate ranking therefore applies the same model
/// weight semantics to ONNX host/CUDA, GGUF/native, and provider adapters.
pub(crate) struct PriorityArbiter {
    registrations: Mutex<HashMap<u64, PriorityRegistrationRecord>>,
    next_registration_id: AtomicU64,
    reclamation: AsyncMutex<()>,
}

impl PriorityArbiter {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            registrations: Mutex::new(HashMap::new()),
            next_registration_id: AtomicU64::new(1),
            reclamation: AsyncMutex::new(()),
        })
    }

    pub(crate) fn register(
        self: &Arc<Self>,
        owner: MemoryOwner,
        weight: u32,
        domains: impl IntoIterator<Item = MemoryDomain>,
    ) -> ModelPriorityLease {
        let registration_id = self.next_registration_id.fetch_add(1, Ordering::Relaxed);
        let mut domains = domains.into_iter().collect::<Vec<_>>();
        domains.sort_by_key(ToString::to_string);
        domains.dedup();
        self.registrations.lock().insert(
            registration_id,
            PriorityRegistrationRecord {
                owner,
                weight: weight.max(1),
                domains,
            },
        );
        ModelPriorityLease {
            arbiter: Arc::downgrade(self),
            registration_id,
        }
    }

    pub(crate) async fn lock_reclamation(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.reclamation.lock().await
    }

    pub(crate) fn reclaim_candidates(
        &self,
        snapshot: &MemorySnapshot,
        requesting_model_id: u32,
        requesting_weight: u32,
        required_domains: &[MemoryDomain],
    ) -> Vec<MemoryReclaimCandidate> {
        let registrations = self.registrations.lock();
        let mut model_weights = HashMap::<u32, u32>::new();
        let mut model_domains = HashMap::<u32, Vec<MemoryDomain>>::new();
        for registration in registrations.values() {
            model_weights
                .entry(registration.owner.model_id)
                .and_modify(|weight| *weight = (*weight).max(registration.weight))
                .or_insert(registration.weight);
            let domains = model_domains
                .entry(registration.owner.model_id)
                .or_default();
            for domain in &registration.domains {
                if !domains.contains(domain) {
                    domains.push(domain.clone());
                }
            }
        }
        drop(registrations);

        let required_overlaps = |domain: &MemoryDomain| {
            required_domains.is_empty()
                || required_domains
                    .iter()
                    .any(|required| domains_share_capacity(domain, required))
        };
        let mut reclaimable = HashMap::<u32, usize>::new();
        for row in &snapshot.rows {
            if row.owner.model_id == requesting_model_id || !required_overlaps(&row.domain) {
                continue;
            }
            let Some(weight) = model_weights.get(&row.owner.model_id).copied() else {
                continue;
            };
            if weight >= requesting_weight {
                continue;
            }
            let registered_for_domain =
                model_domains
                    .get(&row.owner.model_id)
                    .is_some_and(|domains| {
                        domains
                            .iter()
                            .any(|domain| domains_share_capacity(domain, &row.domain))
                    });
            if !registered_for_domain {
                continue;
            }
            let bytes = row
                .reserved_bytes
                .max(row.committed_bytes)
                .max(row.observed_bytes);
            let total = reclaimable.entry(row.owner.model_id).or_default();
            *total = total.saturating_add(bytes);
        }

        let mut candidates = reclaimable
            .into_iter()
            .filter(|(_, reclaimable_bytes)| *reclaimable_bytes > 0)
            .map(|(model_id, reclaimable_bytes)| MemoryReclaimCandidate {
                model_id,
                priority_weight: model_weights[&model_id],
                reclaimable_bytes,
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|candidate| {
            (
                candidate.priority_weight,
                Reverse(candidate.reclaimable_bytes),
                candidate.model_id,
            )
        });
        candidates
    }

    #[cfg(test)]
    fn registration_count(&self) -> usize {
        self.registrations.lock().len()
    }
}

fn domains_share_capacity(left: &MemoryDomain, right: &MemoryDomain) -> bool {
    let host = |domain: &MemoryDomain| {
        matches!(
            domain,
            MemoryDomain::Host | MemoryDomain::HostPinned { .. } | MemoryDomain::HostMapped { .. }
        )
    };
    (host(left) && host(right)) || left == right
}

/// RAII membership in the priority registry. Dropping a failed load, unloaded
/// engine, removed replica, or hot-swap owner immediately removes its claim on
/// priority arbitration.
pub(crate) struct ModelPriorityLease {
    arbiter: Weak<PriorityArbiter>,
    registration_id: u64,
}

impl Drop for ModelPriorityLease {
    fn drop(&mut self) {
        if let Some(arbiter) = self.arbiter.upgrade() {
            arbiter.registrations.lock().remove(&self.registration_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(model_id: u32, domain: MemoryDomain, bytes: usize) -> MemorySnapshotRow {
        MemorySnapshotRow {
            domain,
            owner: MemoryOwner::new(model_id, 0),
            class: MemoryAllocationClass::KvCache,
            planned_bytes: bytes,
            reserved_bytes: bytes,
            committed_bytes: bytes,
            observed_bytes: bytes,
        }
    }

    #[test]
    fn candidates_are_lower_weight_domain_overlapping_and_largest_first() {
        let arbiter = PriorityArbiter::new();
        let _low_small = arbiter.register(
            MemoryOwner::new(1, 0),
            1,
            [MemoryDomain::Cuda { device_id: 0 }],
        );
        let _low_large = arbiter.register(
            MemoryOwner::new(2, 0),
            1,
            [MemoryDomain::Cuda { device_id: 0 }],
        );
        let _equal = arbiter.register(
            MemoryOwner::new(3, 0),
            5,
            [MemoryDomain::Cuda { device_id: 0 }],
        );
        let _other_device = arbiter.register(
            MemoryOwner::new(4, 0),
            1,
            [MemoryDomain::Cuda { device_id: 1 }],
        );
        let snapshot = MemorySnapshot {
            rows: vec![
                row(1, MemoryDomain::Cuda { device_id: 0 }, 10),
                row(2, MemoryDomain::Cuda { device_id: 0 }, 20),
                row(3, MemoryDomain::Cuda { device_id: 0 }, 30),
                row(4, MemoryDomain::Cuda { device_id: 1 }, 40),
            ],
            ..MemorySnapshot::default()
        };

        let candidates =
            arbiter.reclaim_candidates(&snapshot, 9, 5, &[MemoryDomain::Cuda { device_id: 0 }]);
        assert_eq!(
            candidates,
            vec![
                MemoryReclaimCandidate {
                    model_id: 2,
                    priority_weight: 1,
                    reclaimable_bytes: 20,
                },
                MemoryReclaimCandidate {
                    model_id: 1,
                    priority_weight: 1,
                    reclaimable_bytes: 10,
                },
            ]
        );
    }

    #[test]
    fn host_pinned_and_host_rows_share_one_reclaim_domain() {
        let arbiter = PriorityArbiter::new();
        let _lease = arbiter.register(
            MemoryOwner::new(1, 0),
            1,
            [MemoryDomain::HostPinned {
                provider: "cuda".to_string(),
                device_id: Some(0),
            }],
        );
        let snapshot = MemorySnapshot {
            rows: vec![row(1, MemoryDomain::Host, 12)],
            ..MemorySnapshot::default()
        };
        assert_eq!(
            arbiter
                .reclaim_candidates(&snapshot, 2, 2, &[MemoryDomain::Host])
                .len(),
            1,
        );
    }

    #[test]
    fn registration_lifetime_is_raii() {
        let arbiter = PriorityArbiter::new();
        let lease = arbiter.register(MemoryOwner::new(1, 0), 1, [MemoryDomain::Host]);
        assert_eq!(arbiter.registration_count(), 1);
        drop(lease);
        assert_eq!(arbiter.registration_count(), 0);
    }
}
