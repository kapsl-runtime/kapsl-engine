//! Readiness and lease view of an opaque participant launched by the runtime.
//!
//! A managed shared-pool participant becomes usable when its bindings are
//! activated. An opaque participant has no bindings: it is usable as soon as
//! it has registered, and each registration is identified by its epoch.

use super::*;

/// Registration and live leases of the one opaque participant that a managed
/// backend registered under its supervised base ID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ManagedOpaqueKvSnapshot {
    /// Changes whenever the participant registers again.
    pub(crate) participant_epoch: u64,
    /// Bytes the registered capacity model can lease at most, charged once per
    /// physical domain. `None` when a pool declares no ceiling.
    pub(crate) capacity_bytes: Option<u64>,
    pub(crate) leased_bytes: u64,
    pub(crate) active_leases: u64,
}

impl ExternalKvCoordinator {
    /// Return the opaque participant derived from a supervised base ID.
    ///
    /// `None` is ordinary while the child starts. More than one live
    /// participant, or one that registered a shared pool, fails closed: a
    /// backend launched as opaque must not become routable as anything else.
    pub(crate) fn managed_opaque_kv_snapshot(
        &self,
        participant_base: &str,
    ) -> Result<Option<ManagedOpaqueKvSnapshot>, String> {
        let state = self.state.lock();
        let mut matches = state.participants.iter().filter(|(participant_id, _)| {
            participant_matches_managed_base(participant_id, participant_base)
        });
        let Some((participant_id, participant)) = matches.next() else {
            return Ok(None);
        };
        if matches.next().is_some() {
            return Err(format!(
                "multiple live KV participants match managed base '{participant_base}'"
            ));
        }
        if participant.shared_pools.is_some() || participant.shared_activation.is_some() {
            return Err(format!(
                "managed KV participant base '{participant_base}' registered a shared pool where an opaque participant was expected"
            ));
        }

        let mut active_leases = 0u64;
        let mut leased_bytes = 0u64;
        for lease in state
            .leases
            .values()
            .filter(|lease| lease.participant_id == *participant_id)
        {
            active_leases += 1;
            let bytes = lease.memory.as_ref().map_or(0usize, |memory| {
                memory
                    .claims()
                    .iter()
                    .fold(0usize, |total, claim| total.saturating_add(claim.bytes))
            });
            leased_bytes = leased_bytes.saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX));
        }
        Ok(Some(ManagedOpaqueKvSnapshot {
            participant_epoch: participant.receipt.participant_epoch,
            capacity_bytes: registered_capacity_bytes(&participant.registration),
            leased_bytes,
            active_leases,
        }))
    }
}

/// Upper bound of what one registration can lease. Groups that alias a pool
/// are charged once, at the largest shape, in each of the pool's domains.
fn registered_capacity_bytes(registration: &KvParticipantRegistration) -> Option<u64> {
    let mut pools = BTreeMap::<(&KvMemoryDomain, &str), u64>::new();
    for group in &registration.capacity_model.groups {
        let bytes = group
            .bytes_per_allocation
            .checked_mul(group.max_allocations?)?;
        for domain in &group.memory_domains {
            let pool = pools.entry((domain, group.pool_id.as_str())).or_default();
            *pool = (*pool).max(bytes);
        }
    }
    pools
        .values()
        .try_fold(0u64, |total, bytes| total.checked_add(*bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_hal::device::{Device, DeviceBackend, DeviceInfo};
    use kapsl_kv_abi::{
        KvBackendCapabilities, KvCapacityGroup, KvCapacityModel, KvCoordinator as _,
    };

    fn coordinator() -> Arc<ExternalKvCoordinator> {
        let info = DeviceInfo {
            cpu_cores: 1,
            total_memory: 1024 * 1024,
            os_type: "test".to_string(),
            os_release: "test".to_string(),
            has_cuda: false,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: vec![Device {
                id: 0,
                name: "test-cpu".to_string(),
                backend: DeviceBackend::Cpu,
                memory_mb: 0,
                compute_units: 1,
                pci_bus_id: None,
                partition_id: None,
                driver_version: None,
                cuda_version: None,
                compute_capability: None,
                utilization_gpu_pct: None,
                temperature_c: None,
                supports_fp16: false,
                supports_int8: true,
            }],
        };
        let memory = MemoryAuthority::new(&info).expect("test memory authority");
        ExternalKvCoordinator::new(memory, Duration::from_secs(30)).expect("coordinator")
    }

    fn registration(participant_id: &str) -> KvParticipantRegistration {
        KvParticipantRegistration {
            participant_id: participant_id.to_string(),
            backend: "sglang".to_string(),
            model_fingerprint: "sha256:test".to_string(),
            capabilities: KvBackendCapabilities::opaque_connected(),
            capacity_model: KvCapacityModel {
                groups: vec![KvCapacityGroup {
                    group_id: "sglang.kv.0".to_string(),
                    pool_id: "sglang.pool.0".to_string(),
                    allocation_granularity_tokens: 16,
                    bytes_per_allocation: 4096,
                    memory_domains: vec![KvMemoryDomain::Host],
                    max_allocations: Some(64),
                }],
            },
            adapter_profile: None,
            topology: None,
            provisioning_grant: None,
        }
    }

    fn reservation(request_id: &str) -> KvReserveRequest {
        KvReserveRequest {
            sequence: KvSequenceKey {
                request_id: request_id.to_string(),
                sequence_id: request_id.to_string(),
            },
            groups: vec![KvGroupReservation {
                group_id: "sglang.kv.0".to_string(),
                token_capacity: 32,
                minimum_blocks: None,
            }],
            prefix: None,
            priority: 0,
            ttl_ms: None,
        }
    }

    #[test]
    fn snapshot_follows_registration_leases_and_retirement() {
        let coordinator = coordinator();
        let base = "kapsl-model-1-0";
        let participant = "kapsl-model-1-0:scheduler";
        assert_eq!(coordinator.managed_opaque_kv_snapshot(base).unwrap(), None);

        let receipt = coordinator.register(&registration(participant)).unwrap();
        let idle = coordinator
            .managed_opaque_kv_snapshot(base)
            .unwrap()
            .expect("registered participant");
        assert_eq!(idle.participant_epoch, receipt.participant_epoch);
        assert_eq!(idle.capacity_bytes, Some(4096 * 64));
        assert_eq!((idle.active_leases, idle.leased_bytes), (0, 0));

        let first = coordinator
            .reserve(participant, &reservation("request-1"))
            .unwrap();
        coordinator
            .reserve(participant, &reservation("request-2"))
            .unwrap();
        let busy = coordinator
            .managed_opaque_kv_snapshot(base)
            .unwrap()
            .unwrap();
        // 32 tokens at 16 tokens per 4096-byte allocation, twice.
        assert_eq!((busy.active_leases, busy.leased_bytes), (2, 2 * 2 * 4096));

        coordinator
            .release(participant, &first.lease_id, None)
            .unwrap();
        let after_release = coordinator
            .managed_opaque_kv_snapshot(base)
            .unwrap()
            .unwrap();
        assert_eq!(
            (after_release.active_leases, after_release.leased_bytes),
            (1, 2 * 4096)
        );

        assert_eq!(
            coordinator
                .retire_participants_after_backend_exit(base)
                .unwrap(),
            1
        );
        assert_eq!(coordinator.managed_opaque_kv_snapshot(base).unwrap(), None);

        let restarted = coordinator.register(&registration(participant)).unwrap();
        assert!(restarted.participant_epoch > receipt.participant_epoch);
        let fresh = coordinator
            .managed_opaque_kv_snapshot(base)
            .unwrap()
            .unwrap();
        assert_eq!(fresh.participant_epoch, restarted.participant_epoch);
        assert_eq!((fresh.active_leases, fresh.leased_bytes), (0, 0));
    }

    #[test]
    fn snapshot_matches_the_base_namespace_exactly_and_refuses_ambiguity() {
        let coordinator = coordinator();
        coordinator
            .register(&registration("kapsl-model-10-0:scheduler"))
            .unwrap();
        // `kapsl-model-1` is a prefix of the registered ID, not its namespace.
        assert_eq!(
            coordinator
                .managed_opaque_kv_snapshot("kapsl-model-1")
                .unwrap(),
            None
        );
        assert!(coordinator
            .managed_opaque_kv_snapshot("kapsl-model-10-0")
            .unwrap()
            .is_some());

        coordinator
            .register(&registration("kapsl-model-10-0:second"))
            .unwrap();
        let error = coordinator
            .managed_opaque_kv_snapshot("kapsl-model-10-0")
            .unwrap_err();
        assert!(error.contains("multiple live KV participants"), "{error}");
    }

    #[test]
    fn capacity_charges_aliased_groups_once_and_every_domain() {
        let mut registration = registration("kapsl-model-1-0:scheduler");
        let mut alias = registration.capacity_model.groups[0].clone();
        alias.group_id = "sglang.kv.1".to_string();
        alias.bytes_per_allocation = 1024;
        registration.capacity_model.groups.push(alias);
        assert_eq!(registered_capacity_bytes(&registration), Some(4096 * 64));

        registration.capacity_model.groups[0]
            .memory_domains
            .push(KvMemoryDomain::Cuda { device_id: 0 });
        assert_eq!(
            registered_capacity_bytes(&registration),
            Some(2 * 4096 * 64)
        );

        registration.capacity_model.groups[1].max_allocations = None;
        assert_eq!(registered_capacity_bytes(&registration), None);
    }
}
