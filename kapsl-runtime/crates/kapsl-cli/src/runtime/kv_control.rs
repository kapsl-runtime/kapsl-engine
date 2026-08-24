//! Out-of-process KV participant control plane.
//!
//! The transport is deliberately small: one newline-delimited JSON request per
//! Unix connection. Policy and byte accounting live in `ExternalKvCoordinator`;
//! framing only decodes the versioned `kapsl-kv-abi` envelopes.

use super::memory::{
    MemoryAllocationClass, MemoryAuthority, MemoryClaim, MemoryDomain, MemoryLease, MemoryOwner,
    MemoryPlan,
};
use kapsl_kv_abi::{
    dispatch_control_request, KvBlockHandle, KvCacheOwnership, KvCommitRequest, KvContractError,
    KvControlRequestEnvelope, KvControlResponse, KvControlResponseEnvelope, KvGroupLease,
    KvGroupReservation, KvIntegrationTier, KvLease, KvMemoryDomain, KvMetadataMode,
    KvParticipantRegistration, KvRegistrationReceipt, KvReleaseCompletion, KvReserveRequest,
    KvSequenceKey, KvSharedPoolAllocationMode, KvSharedPoolDescriptor, KAPSL_KV_ABI_VERSION,
};
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_FRAME_BYTES: usize = 1024 * 1024;
const MAX_CONNECTIONS: usize = 64;

/// Physical data-plane storage retained for the lifetime of a provisioned
/// shared pool. Implementations clear every newly assigned block before its
/// handle is published to a participant.
pub(crate) trait SharedPoolBacking: Send + Sync {
    fn zero_blocks(
        &self,
        binding: &KvSharedPoolDescriptor,
        block_indices: &[u64],
    ) -> Result<(), KvContractError>;
}

pub(crate) struct ProvisionedSharedPools {
    pub(crate) descriptors: Vec<KvSharedPoolDescriptor>,
    pub(crate) backing: Arc<dyn SharedPoolBacking>,
}

/// Transport-specific provider boundary. A CUDA IPC implementation must
/// allocate an isolated exportable allocation per participant; exporting the
/// runtime's process-wide CUDA pool would violate model/session isolation. The
/// returned backing must retain the corresponding `MemoryAuthority` lease for
/// its full lifetime so physical capacity is charged once at pool creation,
/// rather than once per logical request block.
pub(crate) trait SharedPoolProvisioner: Send + Sync {
    fn provision(
        &self,
        registration: &KvParticipantRegistration,
        owner: MemoryOwner,
        participant_epoch: u64,
    ) -> Result<ProvisionedSharedPools, KvContractError>;
}

#[derive(Clone)]
struct SharedGroupDefinition {
    capacity_pool_id: String,
    allocation_granularity_tokens: u32,
}

struct SharedPoolAllocatorState {
    free_by_pool: HashMap<String, Vec<u64>>,
    quarantined_by_pool: HashMap<String, BTreeSet<u64>>,
}

struct SharedPoolLeaseAllocation {
    blocks_by_pool: BTreeMap<String, Vec<u64>>,
    requires_release_fence: bool,
}

struct SharedPoolSet {
    groups: HashMap<String, SharedGroupDefinition>,
    bindings_by_pool: HashMap<String, Vec<KvSharedPoolDescriptor>>,
    allocation_modes: HashMap<String, KvSharedPoolAllocationMode>,
    state: Mutex<SharedPoolAllocatorState>,
    backing: Arc<dyn SharedPoolBacking>,
}

impl SharedPoolSet {
    fn new(
        registration: &KvParticipantRegistration,
        receipt: &KvRegistrationReceipt,
        provisioned: ProvisionedSharedPools,
    ) -> Result<Arc<Self>, KvContractError> {
        receipt.validate_for(registration)?;
        if provisioned.descriptors != receipt.shared_pools {
            return Err(KvContractError::Internal {
                message: "shared-pool provisioner receipt changed during registration".to_string(),
            });
        }

        let groups = registration
            .capacity_model
            .groups
            .iter()
            .map(|group| {
                (
                    group.group_id.clone(),
                    SharedGroupDefinition {
                        capacity_pool_id: group.pool_id.clone(),
                        allocation_granularity_tokens: group.allocation_granularity_tokens,
                    },
                )
            })
            .collect::<HashMap<_, _>>();

        let mut bindings_by_pool = HashMap::<String, Vec<KvSharedPoolDescriptor>>::new();
        for descriptor in provisioned.descriptors {
            bindings_by_pool
                .entry(descriptor.capacity_pool_id.clone())
                .or_default()
                .push(descriptor);
        }

        let mut allocation_modes = HashMap::new();
        let mut free_by_pool = HashMap::new();
        let mut quarantined_by_pool = HashMap::new();
        for (capacity_pool_id, bindings) in &bindings_by_pool {
            let block_count = bindings
                .first()
                .expect("receipt validation requires a non-empty binding set")
                .block_count;
            if bindings
                .iter()
                .any(|binding| binding.block_count != block_count)
            {
                return Err(KvContractError::invalid_capabilities(format!(
                    "replicas of shared pool '{capacity_pool_id}' must expose the same block count"
                )));
            }
            let allocation_mode = bindings[0].allocation_mode;
            if bindings
                .iter()
                .any(|binding| binding.allocation_mode != allocation_mode)
            {
                return Err(KvContractError::invalid_capabilities(format!(
                    "replicas of shared pool '{capacity_pool_id}' must use the same allocation mode"
                )));
            }
            allocation_modes.insert(capacity_pool_id.clone(), allocation_mode);
            let block_count = usize::try_from(block_count).map_err(|_| {
                KvContractError::invalid_capabilities(format!(
                    "shared pool '{capacity_pool_id}' block count does not fit this runtime"
                ))
            })?;
            free_by_pool.insert(
                capacity_pool_id.clone(),
                (0..block_count as u64).rev().collect(),
            );
            quarantined_by_pool.insert(capacity_pool_id.clone(), BTreeSet::new());
        }

        Ok(Arc::new(Self {
            groups,
            bindings_by_pool,
            allocation_modes,
            state: Mutex::new(SharedPoolAllocatorState {
                free_by_pool,
                quarantined_by_pool,
            }),
            backing: provisioned.backing,
        }))
    }

    fn reserve(
        &self,
        reservations: &[KvGroupReservation],
    ) -> Result<(Vec<KvGroupLease>, SharedPoolLeaseAllocation), KvContractError> {
        let mut group_blocks = Vec::with_capacity(reservations.len());
        let mut needed_by_pool = BTreeMap::<String, u64>::new();
        for reservation in reservations {
            let group = self.groups.get(&reservation.group_id).ok_or_else(|| {
                KvContractError::invalid_request(format!(
                    "reservation references unprovisioned shared group '{}'",
                    reservation.group_id
                ))
            })?;
            let granularity = u64::from(group.allocation_granularity_tokens);
            let blocks = u64::from(reservation.token_capacity)
                .div_ceil(granularity)
                .max(u64::from(reservation.minimum_blocks.unwrap_or(0)));
            group_blocks.push((reservation, group, blocks));
            needed_by_pool
                .entry(group.capacity_pool_id.clone())
                .and_modify(|current| *current = (*current).max(blocks))
                .or_insert(blocks);
        }

        let blocks_by_pool = {
            let mut state = self.state.lock();
            for (pool_id, needed) in &needed_by_pool {
                let available = state
                    .free_by_pool
                    .get(pool_id)
                    .map(Vec::len)
                    .unwrap_or_default();
                if usize::try_from(*needed).map_or(true, |needed| needed > available) {
                    return Err(KvContractError::CapacityExhausted {
                        message: format!(
                            "shared pool '{pool_id}' has {available} free blocks but needs {needed}"
                        ),
                    });
                }
            }
            let mut allocated = BTreeMap::new();
            for (pool_id, needed) in needed_by_pool {
                let free = state
                    .free_by_pool
                    .get_mut(&pool_id)
                    .expect("validated shared pool must have a free list");
                let needed = usize::try_from(needed).expect("capacity checked above");
                let blocks = (0..needed)
                    .map(|_| free.pop().expect("capacity checked above"))
                    .collect::<Vec<_>>();
                allocated.insert(pool_id, blocks);
            }
            allocated
        };

        for (pool_id, blocks) in &blocks_by_pool {
            if self.allocation_modes.get(pool_id)
                != Some(&KvSharedPoolAllocationMode::RuntimeLeased)
            {
                continue;
            }
            let bindings = self
                .bindings_by_pool
                .get(pool_id)
                .expect("validated shared pool must have physical bindings");
            for binding in bindings {
                if let Err(error) = self.backing.zero_blocks(binding, blocks) {
                    self.return_blocks(&blocks_by_pool);
                    return Err(error);
                }
            }
        }

        let public_groups = group_blocks
            .into_iter()
            .map(|(reservation, group, needed)| {
                let blocks = blocks_by_pool
                    .get(&group.capacity_pool_id)
                    .expect("shared pool was allocated");
                let bindings = self
                    .bindings_by_pool
                    .get(&group.capacity_pool_id)
                    .expect("validated shared pool must have physical bindings");
                let mut handles = Vec::new();
                if self.allocation_modes.get(&group.capacity_pool_id)
                    == Some(&KvSharedPoolAllocationMode::RuntimeLeased)
                {
                    for block_index in blocks.iter().take(needed as usize) {
                        for binding in bindings {
                            handles.push(KvBlockHandle::RuntimePool {
                                pool_id: binding.binding_id.clone(),
                                block_index: *block_index,
                                generation: binding.generation,
                            });
                        }
                    }
                }
                KvGroupLease {
                    group_id: reservation.group_id.clone(),
                    token_capacity: reservation.token_capacity,
                    blocks: handles,
                }
            })
            .collect();

        let requires_release_fence = blocks_by_pool.keys().any(|pool_id| {
            self.allocation_modes.get(pool_id) == Some(&KvSharedPoolAllocationMode::RuntimeLeased)
        });
        Ok((
            public_groups,
            SharedPoolLeaseAllocation {
                blocks_by_pool,
                requires_release_fence,
            },
        ))
    }

    fn release(&self, allocation: SharedPoolLeaseAllocation) {
        self.return_blocks(&allocation.blocks_by_pool);
    }

    fn quarantine(&self, allocation: SharedPoolLeaseAllocation) {
        let mut state = self.state.lock();
        for (pool_id, blocks) in allocation.blocks_by_pool {
            state
                .quarantined_by_pool
                .get_mut(&pool_id)
                .expect("validated shared pool must have a quarantine set")
                .extend(blocks);
        }
    }

    fn return_blocks(&self, blocks_by_pool: &BTreeMap<String, Vec<u64>>) {
        let mut state = self.state.lock();
        for (pool_id, blocks) in blocks_by_pool {
            state
                .free_by_pool
                .get_mut(pool_id)
                .expect("validated shared pool must have a free list")
                .extend(blocks.iter().copied());
        }
    }

    #[cfg(test)]
    fn available_blocks(&self, pool_id: &str) -> usize {
        self.state
            .lock()
            .free_by_pool
            .get(pool_id)
            .map(Vec::len)
            .unwrap_or_default()
    }

    #[cfg(test)]
    fn quarantined_blocks(&self, pool_id: &str) -> usize {
        self.state
            .lock()
            .quarantined_by_pool
            .get(pool_id)
            .map(BTreeSet::len)
            .unwrap_or_default()
    }
}

struct ParticipantRecord {
    registration: KvParticipantRegistration,
    owner: MemoryOwner,
    receipt: KvRegistrationReceipt,
    shared_pools: Option<Arc<SharedPoolSet>>,
}

struct ExternalLeaseRecord {
    public: KvLease,
    request: KvReserveRequest,
    participant_id: String,
    ttl: Duration,
    expires_at: Instant,
    memory: Option<MemoryLease>,
    shared_pools: Option<Arc<SharedPoolSet>>,
    shared_allocation: Option<SharedPoolLeaseAllocation>,
}

#[derive(Default)]
struct CoordinatorState {
    participants: HashMap<String, ParticipantRecord>,
    leases: HashMap<String, ExternalLeaseRecord>,
    sequences: HashMap<(String, KvSequenceKey), String>,
}

/// Runtime implementation of the backend-neutral KV coordinator contract.
///
/// Opaque participants use byte-accounted memory leases. Shared-pool
/// participants are accepted only when a transport-specific provisioner is
/// installed; the ordinary listener therefore remains fail-closed until its
/// physical data plane is configured.
pub(crate) struct ExternalKvCoordinator {
    memory: Arc<MemoryAuthority>,
    state: Mutex<CoordinatorState>,
    next_participant_slot: AtomicU32,
    next_participant_epoch: AtomicU64,
    next_lease_id: AtomicU64,
    maximum_lease_ttl: Duration,
    shared_pool_provisioner: Option<Arc<dyn SharedPoolProvisioner>>,
}

impl ExternalKvCoordinator {
    pub(crate) fn new(
        memory: Arc<MemoryAuthority>,
        maximum_lease_ttl: Duration,
    ) -> Result<Arc<Self>, String> {
        Self::new_with_shared_pool_provisioner(memory, maximum_lease_ttl, None)
    }

    pub(crate) fn new_with_shared_pool_provisioner(
        memory: Arc<MemoryAuthority>,
        maximum_lease_ttl: Duration,
        shared_pool_provisioner: Option<Arc<dyn SharedPoolProvisioner>>,
    ) -> Result<Arc<Self>, String> {
        if maximum_lease_ttl.is_zero() {
            return Err("KV control lease TTL must be non-zero".to_string());
        }
        Ok(Arc::new(Self {
            memory,
            state: Mutex::new(CoordinatorState::default()),
            next_participant_slot: AtomicU32::new(0),
            next_participant_epoch: AtomicU64::new(1),
            next_lease_id: AtomicU64::new(1),
            maximum_lease_ttl,
            shared_pool_provisioner,
        }))
    }

    pub(crate) fn expire_stale(&self) -> usize {
        let now = Instant::now();
        let mut expired = {
            let mut state = self.state.lock();
            let lease_ids = state
                .leases
                .iter()
                .filter_map(|(lease_id, lease)| {
                    (lease.expires_at <= now).then_some(lease_id.clone())
                })
                .collect::<Vec<_>>();
            let mut expired = Vec::with_capacity(lease_ids.len());
            for lease_id in lease_ids {
                if let Some(lease) = state.leases.remove(&lease_id) {
                    state
                        .sequences
                        .remove(&(lease.participant_id.clone(), lease.request.sequence.clone()));
                    expired.push(lease);
                }
            }
            expired
        };
        let count = expired.len();
        let mut quarantined = 0usize;
        for lease in &mut expired {
            if let (Some(pools), Some(allocation)) =
                (lease.shared_pools.as_ref(), lease.shared_allocation.take())
            {
                pools.quarantine(allocation);
                quarantined += 1;
            }
        }
        drop(expired);
        if count > 0 {
            log::warn!("[kv-control] expired {count} stale capacity lease(s)");
        }
        if quarantined > 0 {
            log::error!(
                "[kv-control] quarantined blocks from {quarantined} expired shared-pool lease(s); unfenced blocks are never recycled"
            );
        }
        count
    }

    #[cfg(test)]
    fn participant_count(&self) -> usize {
        self.state.lock().participants.len()
    }

    #[cfg(test)]
    fn lease_count(&self) -> usize {
        self.state.lock().leases.len()
    }

    fn participant_registration(
        &self,
        participant_id: &str,
    ) -> Result<
        (
            KvParticipantRegistration,
            MemoryOwner,
            Option<Arc<SharedPoolSet>>,
        ),
        KvContractError,
    > {
        self.state
            .lock()
            .participants
            .get(participant_id)
            .map(|record| {
                (
                    record.registration.clone(),
                    record.owner,
                    record.shared_pools.clone(),
                )
            })
            .ok_or_else(|| KvContractError::NotFound {
                message: format!("KV participant '{participant_id}' is not registered"),
            })
    }

    fn effective_ttl(&self, requested_ms: Option<u64>) -> Result<Duration, KvContractError> {
        let requested = requested_ms
            .map(Duration::from_millis)
            .unwrap_or(self.maximum_lease_ttl);
        if requested > self.maximum_lease_ttl {
            return Err(KvContractError::invalid_request(format!(
                "requested lease TTL {}ms exceeds the runtime maximum {}ms",
                requested.as_millis(),
                self.maximum_lease_ttl.as_millis()
            )));
        }
        Ok(requested)
    }
}

impl kapsl_kv_abi::KvCoordinator for ExternalKvCoordinator {
    fn register(
        &self,
        registration: &KvParticipantRegistration,
    ) -> Result<KvRegistrationReceipt, KvContractError> {
        registration.validate()?;
        let is_opaque = registration.capabilities.tier == KvIntegrationTier::KvConnected
            && registration.capabilities.metadata_mode == KvMetadataMode::Opaque
            && registration.capabilities.ownership == KvCacheOwnership::Backend;
        let is_shared = registration.capabilities.tier == KvIntegrationTier::SharedPool
            && registration.capabilities.metadata_mode == KvMetadataMode::Structured
            && registration.capabilities.ownership == KvCacheOwnership::KapslRuntime;
        if !is_opaque && !is_shared {
            return Err(KvContractError::invalid_capabilities(
                "external KV participants must be opaque/backend-owned or use a provisioned runtime-owned shared pool",
            ));
        }
        if is_opaque {
            for domain in registration
                .capacity_model
                .groups
                .iter()
                .flat_map(|group| &group.memory_domains)
            {
                let runtime_domain = runtime_memory_domain(domain)?;
                if self.memory.supports_external_leases(&runtime_domain) {
                    continue;
                }
                return Err(KvContractError::invalid_capabilities(format!(
                    "runtime has no bounded external-lease authority for KV domain {runtime_domain}"
                )));
            }
        }

        self.expire_stale();
        let mut state = self.state.lock();
        if let Some(existing) = state.participants.get(&registration.participant_id) {
            if existing.registration == *registration {
                return Ok(existing.receipt.clone());
            }
            if state
                .leases
                .values()
                .any(|lease| lease.participant_id == registration.participant_id)
            {
                return Err(KvContractError::invalid_request(format!(
                    "participant '{}' cannot change registration while leases are active",
                    registration.participant_id
                )));
            }
            if existing.shared_pools.is_some() || is_shared {
                return Err(KvContractError::invalid_request(format!(
                    "shared-pool participant '{}' cannot replace its registration while imported pool mappings may still exist",
                    registration.participant_id
                )));
            }
            let owner = existing.owner;
            let participant_epoch = self.next_participant_epoch.fetch_add(1, Ordering::Relaxed);
            let receipt = KvRegistrationReceipt::opaque(
                registration.participant_id.clone(),
                participant_epoch,
            );
            state.participants.insert(
                registration.participant_id.clone(),
                ParticipantRecord {
                    registration: registration.clone(),
                    owner,
                    receipt: receipt.clone(),
                    shared_pools: None,
                },
            );
            return Ok(receipt);
        }

        let slot = self.next_participant_slot.fetch_add(1, Ordering::Relaxed);
        let owner = MemoryOwner::external_kv(slot).ok_or_else(|| KvContractError::Internal {
            message: "external KV participant owner space exhausted".to_string(),
        })?;
        let participant_epoch = self.next_participant_epoch.fetch_add(1, Ordering::Relaxed);
        let (receipt, shared_pools) = if is_shared {
            let provisioner = self.shared_pool_provisioner.as_ref().ok_or_else(|| {
                KvContractError::invalid_capabilities(
                    "shared_pool was requested but no isolated data-plane provisioner is configured",
                )
            })?;
            let provisioned = provisioner.provision(registration, owner, participant_epoch)?;
            let receipt = KvRegistrationReceipt {
                participant_id: registration.participant_id.clone(),
                participant_epoch,
                shared_pools: provisioned.descriptors.clone(),
            };
            let pools = SharedPoolSet::new(registration, &receipt, provisioned)?;
            (receipt, Some(pools))
        } else {
            (
                KvRegistrationReceipt::opaque(
                    registration.participant_id.clone(),
                    participant_epoch,
                ),
                None,
            )
        };
        state.participants.insert(
            registration.participant_id.clone(),
            ParticipantRecord {
                registration: registration.clone(),
                owner,
                receipt: receipt.clone(),
                shared_pools,
            },
        );
        log::info!(
            "[kv-control] registered participant '{}' backend={} model={} tier={:?} metadata={:?} epoch={}",
            registration.participant_id,
            registration.backend,
            registration.model_fingerprint,
            registration.capabilities.tier,
            registration.capabilities.metadata_mode,
            participant_epoch,
        );
        Ok(receipt)
    }

    fn reserve(
        &self,
        participant_id: &str,
        request: &KvReserveRequest,
    ) -> Result<KvLease, KvContractError> {
        request.validate()?;
        self.expire_stale();

        let (registration, owner, shared_pools) = self.participant_registration(participant_id)?;
        if let Some(prefix) = &request.prefix {
            if prefix.model_fingerprint != registration.model_fingerprint {
                return Err(KvContractError::invalid_request(
                    "prefix model fingerprint does not match the registered participant",
                ));
            }
            return Err(KvContractError::unsupported("reserve_prefix"));
        }

        let sequence_key = (participant_id.to_string(), request.sequence.clone());
        {
            let state = self.state.lock();
            if let Some(existing_id) = state.sequences.get(&sequence_key) {
                let existing = state
                    .leases
                    .get(existing_id)
                    .expect("sequence index must reference a live lease");
                if existing.request == *request {
                    return Ok(existing.public.clone());
                }
                return Err(KvContractError::invalid_request(
                    "sequence already has a lease with a different reservation",
                ));
            }
        }

        let known_groups = registration
            .capacity_model
            .groups
            .iter()
            .map(|group| group.group_id.as_str())
            .collect::<std::collections::HashSet<_>>();
        if request
            .groups
            .iter()
            .any(|group| !known_groups.contains(group.group_id.as_str()))
        {
            return Err(KvContractError::invalid_request(
                "reservation references an unregistered cache group",
            ));
        }
        let ttl = self.effective_ttl(request.ttl_ms)?;
        let numeric_id = self.next_lease_id.fetch_add(1, Ordering::Relaxed);
        let lease_id = format!("kapsl-kv-{numeric_id:016x}");
        let (memory, public_groups, mut shared_allocation) = if let Some(pools) =
            shared_pools.as_ref()
        {
            let (groups, allocation) = pools.reserve(&request.groups)?;
            (None, groups, Some(allocation))
        } else {
            let bytes_by_domain = registration
                    .capacity_model
                    .bytes_by_domain_for_reservations(&request.groups)
                    .ok_or_else(|| KvContractError::CapacityExhausted {
                        message: "reservation exceeds the participant capacity model or byte accounting overflowed"
                            .to_string(),
                    })?;
            let mut plan = MemoryPlan::new();
            for (domain, bytes) in bytes_by_domain {
                let bytes =
                    usize::try_from(bytes).map_err(|_| KvContractError::CapacityExhausted {
                        message: "KV reservation does not fit the runtime address space"
                            .to_string(),
                    })?;
                plan.push(MemoryClaim::external(
                    runtime_memory_domain(&domain)?,
                    owner,
                    MemoryAllocationClass::KvCache,
                    format!("kv-participant:{participant_id}:{lease_id}"),
                    bytes,
                ));
            }
            let memory = self
                .memory
                .admit(&plan)
                .map_err(|message| KvContractError::CapacityExhausted { message })?;
            let groups = request
                .groups
                .iter()
                .map(|group| KvGroupLease {
                    group_id: group.group_id.clone(),
                    token_capacity: group.token_capacity,
                    blocks: Vec::new(),
                })
                .collect();
            (Some(memory), groups, None)
        };

        let expires_at = Instant::now() + ttl;
        let public = KvLease {
            lease_id: lease_id.clone(),
            sequence: request.sequence.clone(),
            groups: public_groups,
            expires_at_unix_ms: Some(unix_ms_after(ttl)),
        };
        if let Err(error) = public.validate() {
            if let (Some(pools), Some(allocation)) =
                (shared_pools.as_ref(), shared_allocation.take())
            {
                pools.release(allocation);
            }
            return Err(error);
        }

        let mut state = self.state.lock();
        if state
            .participants
            .get(participant_id)
            .is_none_or(|current| current.registration != registration)
        {
            drop(state);
            if let (Some(pools), Some(allocation)) =
                (shared_pools.as_ref(), shared_allocation.take())
            {
                pools.release(allocation);
            }
            return Err(KvContractError::invalid_request(
                "participant registration changed while the reservation was being admitted; retry",
            ));
        }
        if let Some(existing_id) = state.sequences.get(&sequence_key) {
            let existing = state
                .leases
                .get(existing_id)
                .expect("sequence index must reference a live lease");
            if existing.request == *request {
                let existing = existing.public.clone();
                drop(state);
                if let (Some(pools), Some(allocation)) =
                    (shared_pools.as_ref(), shared_allocation.take())
                {
                    pools.release(allocation);
                }
                return Ok(existing);
            }
            drop(state);
            if let (Some(pools), Some(allocation)) =
                (shared_pools.as_ref(), shared_allocation.take())
            {
                pools.release(allocation);
            }
            return Err(KvContractError::invalid_request(
                "sequence acquired a conflicting lease concurrently",
            ));
        }
        let record = ExternalLeaseRecord {
            public: public.clone(),
            request: request.clone(),
            participant_id: participant_id.to_string(),
            ttl,
            expires_at,
            memory,
            shared_pools,
            shared_allocation,
        };
        state.sequences.insert(sequence_key, lease_id.clone());
        state.leases.insert(lease_id, record);
        Ok(public)
    }

    fn commit(
        &self,
        participant_id: &str,
        request: &KvCommitRequest,
    ) -> Result<(), KvContractError> {
        request.validate()?;
        self.expire_stale();
        if request.prefix.is_some() {
            return Err(KvContractError::unsupported("commit_prefix"));
        }
        let mut state = self.state.lock();
        let lease =
            state
                .leases
                .get_mut(&request.lease_id)
                .ok_or_else(|| KvContractError::NotFound {
                    message: format!("KV lease '{}' does not exist", request.lease_id),
                })?;
        ensure_lease_owner(lease, participant_id)?;
        if lease
            .public
            .groups
            .iter()
            .any(|group| request.computed_tokens > group.token_capacity)
        {
            return Err(KvContractError::invalid_request(
                "computed_tokens exceeds the reserved token capacity",
            ));
        }
        if let Some(memory) = lease.memory.as_mut() {
            memory.commit_capacity();
        }
        Ok(())
    }

    fn touch(&self, participant_id: &str, lease_id: &str) -> Result<(), KvContractError> {
        self.expire_stale();
        let mut state = self.state.lock();
        let lease = state
            .leases
            .get_mut(lease_id)
            .ok_or_else(|| KvContractError::NotFound {
                message: format!("KV lease '{lease_id}' does not exist"),
            })?;
        ensure_lease_owner(lease, participant_id)?;
        lease.expires_at = Instant::now() + lease.ttl;
        lease.public.expires_at_unix_ms = Some(unix_ms_after(lease.ttl));
        Ok(())
    }

    fn heartbeat(&self, participant_id: &str) -> Result<(), KvContractError> {
        self.expire_stale();
        let mut state = self.state.lock();
        if !state.participants.contains_key(participant_id) {
            return Err(KvContractError::NotFound {
                message: format!("KV participant '{participant_id}' is not registered"),
            });
        }
        let now = Instant::now();
        for lease in state
            .leases
            .values_mut()
            .filter(|lease| lease.participant_id == participant_id)
        {
            lease.expires_at = now + lease.ttl;
            lease.public.expires_at_unix_ms = Some(unix_ms_after(lease.ttl));
        }
        Ok(())
    }

    fn release(
        &self,
        participant_id: &str,
        lease_id: &str,
        completion: Option<&KvReleaseCompletion>,
    ) -> Result<(), KvContractError> {
        self.expire_stale();
        if let Some(completion) = completion {
            completion.validate()?;
        }
        let mut released = {
            let mut state = self.state.lock();
            let lease = state
                .leases
                .get(lease_id)
                .ok_or_else(|| KvContractError::NotFound {
                    message: format!("KV lease '{lease_id}' does not exist"),
                })?;
            ensure_lease_owner(lease, participant_id)?;
            if lease
                .shared_allocation
                .as_ref()
                .is_some_and(|allocation| allocation.requires_release_fence)
            {
                match completion {
                    Some(KvReleaseCompletion::BackendSynchronized) => {}
                    Some(KvReleaseCompletion::TransportFence { .. }) => {
                        return Err(KvContractError::unsupported(
                            "shared_pool_release_transport_fence",
                        ));
                    }
                    None => {
                        return Err(KvContractError::invalid_request(
                            "shared-pool release requires backend_synchronized completion or a supported transport fence",
                        ));
                    }
                }
            }
            let released = state.leases.remove(lease_id).expect("lease checked above");
            state.sequences.remove(&(
                released.participant_id.clone(),
                released.request.sequence.clone(),
            ));
            released
        };
        if let (Some(pools), Some(allocation)) = (
            released.shared_pools.as_ref(),
            released.shared_allocation.take(),
        ) {
            pools.release(allocation);
        }
        drop(released);
        Ok(())
    }
}

fn ensure_lease_owner(
    lease: &ExternalLeaseRecord,
    participant_id: &str,
) -> Result<(), KvContractError> {
    if lease.participant_id == participant_id {
        Ok(())
    } else {
        Err(KvContractError::NotFound {
            message: "KV lease does not belong to this participant".to_string(),
        })
    }
}

fn runtime_memory_domain(domain: &KvMemoryDomain) -> Result<MemoryDomain, KvContractError> {
    let device_id = |value: u32| {
        usize::try_from(value).map_err(|_| {
            KvContractError::invalid_capabilities(
                "KV memory-domain device ID does not fit the runtime address space",
            )
        })
    };
    Ok(match domain {
        KvMemoryDomain::Host => MemoryDomain::Host,
        KvMemoryDomain::HostPinned {
            provider,
            device_id: id,
        } => MemoryDomain::HostPinned {
            provider: provider.clone(),
            device_id: id.map(device_id).transpose()?,
        },
        KvMemoryDomain::HostMapped {
            provider,
            device_id: id,
        } => MemoryDomain::HostMapped {
            provider: provider.clone(),
            device_id: id.map(device_id).transpose()?,
        },
        KvMemoryDomain::Cuda { device_id: id } => MemoryDomain::Cuda {
            device_id: device_id(*id)?,
        },
        KvMemoryDomain::Provider {
            provider,
            device_id: id,
        } => MemoryDomain::Provider {
            provider: provider.clone(),
            device_id: id.map(device_id).transpose()?,
        },
    })
}

fn unix_ms_after(duration: Duration) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let later = now.saturating_add(duration.as_millis());
    u64::try_from(later).unwrap_or(u64::MAX)
}

#[cfg(unix)]
mod unix {
    use super::*;
    use std::io;
    use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
    use std::path::{Path, PathBuf};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{UnixListener, UnixStream};
    use tokio::sync::Semaphore;

    #[derive(Clone, Copy)]
    struct SocketIdentity {
        device: u64,
        inode: u64,
    }

    pub(crate) struct KvControlServer {
        listener: UnixListener,
        path: PathBuf,
        identity: SocketIdentity,
        coordinator: Arc<ExternalKvCoordinator>,
        permits: Arc<Semaphore>,
        max_frame_bytes: usize,
    }

    impl KvControlServer {
        pub(crate) async fn bind(
            path: impl AsRef<Path>,
            coordinator: Arc<ExternalKvCoordinator>,
        ) -> Result<Self, String> {
            let path = path.as_ref().to_path_buf();
            if !path.is_absolute() {
                return Err("KV control socket path must be absolute".to_string());
            }
            let parent = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .ok_or_else(|| "KV control socket path has no parent directory".to_string())?;
            if !parent.is_dir() {
                return Err(format!(
                    "KV control socket parent directory {} does not exist",
                    parent.display()
                ));
            }

            remove_stale_socket(&path).await?;
            let listener = UnixListener::bind(&path).map_err(|error| {
                format!(
                    "failed to bind KV control socket {}: {error}",
                    path.display()
                )
            })?;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).map_err(
                |error| {
                    format!(
                        "failed to secure KV control socket {}: {error}",
                        path.display()
                    )
                },
            )?;
            let metadata = std::fs::metadata(&path).map_err(|error| {
                format!(
                    "failed to inspect KV control socket {}: {error}",
                    path.display()
                )
            })?;
            let identity = SocketIdentity {
                device: metadata.dev(),
                inode: metadata.ino(),
            };
            Ok(Self {
                listener,
                path,
                identity,
                coordinator,
                permits: Arc::new(Semaphore::new(MAX_CONNECTIONS)),
                max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            })
        }

        pub(crate) async fn run(self) -> io::Result<()> {
            let sweep_period = (self.coordinator.maximum_lease_ttl / 2)
                .max(Duration::from_millis(250))
                .min(Duration::from_secs(5));
            let mut sweep = tokio::time::interval(sweep_period);
            loop {
                tokio::select! {
                    accepted = self.listener.accept() => {
                        let (stream, _) = accepted?;
                        let permit = Arc::clone(&self.permits)
                            .acquire_owned()
                            .await
                            .map_err(|_| io::Error::other("KV control connection limiter closed"))?;
                        let coordinator = Arc::clone(&self.coordinator);
                        let max_frame_bytes = self.max_frame_bytes;
                        tokio::spawn(async move {
                            let _permit = permit;
                            if let Err(error) = handle_connection(stream, coordinator, max_frame_bytes).await {
                                log::warn!("[kv-control] connection failed: {error}");
                            }
                        });
                    }
                    _ = sweep.tick() => {
                        self.coordinator.expire_stale();
                    }
                }
            }
        }
    }

    impl Drop for KvControlServer {
        fn drop(&mut self) {
            let same_socket = std::fs::metadata(&self.path).is_ok_and(|metadata| {
                metadata.file_type().is_socket()
                    && metadata.dev() == self.identity.device
                    && metadata.ino() == self.identity.inode
            });
            if same_socket {
                let _ = std::fs::remove_file(&self.path);
            }
        }
    }

    async fn remove_stale_socket(path: &Path) -> Result<(), String> {
        let metadata = match std::fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(format!(
                    "failed to inspect KV control socket {}: {error}",
                    path.display()
                ));
            }
        };
        if !metadata.file_type().is_socket() {
            return Err(format!(
                "refusing to replace non-socket path {}",
                path.display()
            ));
        }
        if UnixStream::connect(path).await.is_ok() {
            return Err(format!(
                "KV control socket {} is already served by another process",
                path.display()
            ));
        }
        std::fs::remove_file(path).map_err(|error| {
            format!(
                "failed to remove stale KV control socket {}: {error}",
                path.display()
            )
        })
    }

    async fn handle_connection(
        mut stream: UnixStream,
        coordinator: Arc<ExternalKvCoordinator>,
        max_frame_bytes: usize,
    ) -> io::Result<()> {
        let response = match tokio::time::timeout(
            Duration::from_secs(5),
            read_frame(&mut stream, max_frame_bytes),
        )
        .await
        {
            Ok(Ok(frame)) => match serde_json::from_slice::<KvControlRequestEnvelope>(&frame) {
                Ok(envelope) => dispatch_control_request(coordinator.as_ref(), envelope),
                Err(error) => error_response(
                    request_id_from_invalid_frame(&frame),
                    KvContractError::invalid_request(format!(
                        "invalid KV control envelope: {error}"
                    )),
                ),
            },
            Ok(Err(error)) => error_response(
                String::new(),
                KvContractError::Transport {
                    message: error.to_string(),
                },
            ),
            Err(_) => error_response(
                String::new(),
                KvContractError::Transport {
                    message: "timed out reading KV control frame".to_string(),
                },
            ),
        };
        let mut encoded = serde_json::to_vec(&response).map_err(io::Error::other)?;
        encoded.push(b'\n');
        stream.write_all(&encoded).await?;
        match stream.shutdown().await {
            Err(error) if error.kind() == io::ErrorKind::NotConnected => Ok(()),
            result => result,
        }
    }

    async fn read_frame(stream: &mut UnixStream, max_frame_bytes: usize) -> io::Result<Vec<u8>> {
        let mut frame = Vec::new();
        let mut buffer = [0u8; 8192];
        loop {
            let read = stream.read(&mut buffer).await?;
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "peer closed before a newline-delimited KV control frame",
                ));
            }
            if let Some(newline) = buffer[..read].iter().position(|byte| *byte == b'\n') {
                frame.extend_from_slice(&buffer[..newline]);
                if frame.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "KV control frame is empty",
                    ));
                }
                if frame.len() > max_frame_bytes {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "KV control frame exceeds maximum size",
                    ));
                }
                return Ok(frame);
            }
            frame.extend_from_slice(&buffer[..read]);
            if frame.len() > max_frame_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "KV control frame exceeds maximum size",
                ));
            }
        }
    }

    fn request_id_from_invalid_frame(frame: &[u8]) -> String {
        serde_json::from_slice::<serde_json::Value>(frame)
            .ok()
            .and_then(|value| value.get("request_id")?.as_str().map(str::to_string))
            .unwrap_or_default()
    }

    fn error_response(request_id: String, error: KvContractError) -> KvControlResponseEnvelope {
        KvControlResponseEnvelope {
            abi_version: KAPSL_KV_ABI_VERSION,
            request_id,
            response: KvControlResponse::Error { error },
        }
    }

    pub(crate) use KvControlServer as Server;
}

#[cfg(unix)]
pub(crate) use unix::Server as KvControlServer;

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_hal::device::{Device, DeviceBackend, DeviceInfo};
    use kapsl_kv_abi::{
        KvBackendCapabilities, KvCacheGeometry, KvCacheGroup, KvCachePolicy, KvCapacityGroup,
        KvCapacityModel, KvControlRequest, KvElementType, KvGroupReservation, KvLayerId,
        KvMemoryDomain, KvSequenceKey, KvShard, KvTensorLayout, KvTopology, KvTransport,
    };

    #[derive(Default)]
    struct TestSharedBacking {
        zeroed_blocks: AtomicU64,
    }

    impl SharedPoolBacking for TestSharedBacking {
        fn zero_blocks(
            &self,
            _binding: &KvSharedPoolDescriptor,
            block_indices: &[u64],
        ) -> Result<(), KvContractError> {
            self.zeroed_blocks
                .fetch_add(block_indices.len() as u64, Ordering::Relaxed);
            Ok(())
        }
    }

    struct TestSharedProvisioner {
        backing: Arc<TestSharedBacking>,
    }

    impl SharedPoolProvisioner for TestSharedProvisioner {
        fn provision(
            &self,
            registration: &KvParticipantRegistration,
            _owner: MemoryOwner,
            participant_epoch: u64,
        ) -> Result<ProvisionedSharedPools, KvContractError> {
            let group = registration
                .capacity_model
                .groups
                .first()
                .expect("test shared registration group");
            Ok(ProvisionedSharedPools {
                descriptors: vec![KvSharedPoolDescriptor {
                    binding_id: format!("test-binding-{participant_epoch}"),
                    capacity_pool_id: group.pool_id.clone(),
                    generation: participant_epoch,
                    group_ids: vec![group.group_id.clone()],
                    memory_domain: KvMemoryDomain::Host,
                    block_count: group.max_allocations.expect("test block cap"),
                    bytes_per_block: group.bytes_per_allocation,
                    allocation_mode: if registration
                        .capabilities
                        .features
                        .contains(&kapsl_kv_abi::KvFeature::ParticipantBlockSelection)
                    {
                        KvSharedPoolAllocationMode::ParticipantManaged
                    } else {
                        KvSharedPoolAllocationMode::RuntimeLeased
                    },
                    transport: KvTransport::Custom {
                        name: "test_direct".to_string(),
                    },
                    descriptor: "test-shared-backing".to_string(),
                }],
                backing: self.backing.clone(),
            })
        }
    }

    fn test_memory() -> Arc<MemoryAuthority> {
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
        MemoryAuthority::new(&info).expect("test memory authority")
    }

    fn registration() -> KvParticipantRegistration {
        KvParticipantRegistration {
            participant_id: "vllm:test:scheduler".to_string(),
            backend: "vllm".to_string(),
            model_fingerprint: "sha256:test".to_string(),
            capabilities: KvBackendCapabilities::opaque_connected(),
            capacity_model: KvCapacityModel {
                groups: vec![KvCapacityGroup {
                    group_id: "vllm.group.0".to_string(),
                    pool_id: "vllm.pool.0".to_string(),
                    allocation_granularity_tokens: 16,
                    bytes_per_allocation: 4096,
                    memory_domains: vec![KvMemoryDomain::Host],
                    max_allocations: Some(1024),
                }],
            },
            topology: None,
        }
    }

    fn shared_registration() -> KvParticipantRegistration {
        let mut capabilities = KvBackendCapabilities::in_process_shared_pool();
        capabilities.transports.clear();
        capabilities.transports.insert(KvTransport::Custom {
            name: "test_direct".to_string(),
        });
        KvParticipantRegistration {
            participant_id: "vllm:test:scheduler".to_string(),
            backend: "vllm".to_string(),
            model_fingerprint: "sha256:test".to_string(),
            capabilities,
            capacity_model: KvCapacityModel {
                groups: vec![KvCapacityGroup {
                    group_id: "vllm.group.0".to_string(),
                    pool_id: "vllm.pool.0".to_string(),
                    allocation_granularity_tokens: 16,
                    bytes_per_allocation: 4096,
                    memory_domains: vec![KvMemoryDomain::Host],
                    max_allocations: Some(4),
                }],
            },
            topology: Some(KvTopology {
                abi_version: KAPSL_KV_ABI_VERSION,
                model_fingerprint: "sha256:test".to_string(),
                shard: KvShard::default(),
                cache_groups: vec![KvCacheGroup {
                    group_id: "vllm.group.0".to_string(),
                    layers: vec![KvLayerId::indexed(0)],
                    geometry: KvCacheGeometry::PagedAttention {
                        block_size_tokens: 16,
                        kv_heads: 1,
                        key_head_dim: 128,
                        value_head_dim: 128,
                        element_type: KvElementType::F16,
                        layout: KvTensorLayout::BlockKvHeadTokenDim,
                    },
                    policy: KvCachePolicy::FullAttention,
                }],
            }),
        }
    }

    fn reservation(ttl_ms: Option<u64>) -> KvReserveRequest {
        KvReserveRequest {
            sequence: KvSequenceKey {
                request_id: "request-1".to_string(),
                sequence_id: "request-1".to_string(),
            },
            groups: vec![KvGroupReservation {
                group_id: "vllm.group.0".to_string(),
                token_capacity: 32,
                minimum_blocks: None,
            }],
            prefix: None,
            priority: 0,
            ttl_ms,
        }
    }

    #[test]
    fn coordinator_lifecycle_is_reflected_in_memory_authority() {
        use kapsl_kv_abi::KvCoordinator as _;

        let memory = test_memory();
        let coordinator =
            ExternalKvCoordinator::new(memory.clone(), Duration::from_secs(30)).unwrap();
        coordinator.register(&registration()).unwrap();
        assert_eq!(coordinator.participant_count(), 1);

        let lease = coordinator
            .reserve("vllm:test:scheduler", &reservation(None))
            .unwrap();
        let duplicate = coordinator
            .reserve("vllm:test:scheduler", &reservation(None))
            .unwrap();
        assert_eq!(lease.lease_id, duplicate.lease_id);
        assert_eq!(coordinator.lease_count(), 1);
        let row = memory
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner.is_external_kv())
            .expect("external KV memory row");
        assert_eq!(row.reserved_bytes, 8192);

        coordinator
            .commit(
                "vllm:test:scheduler",
                &KvCommitRequest {
                    lease_id: lease.lease_id.clone(),
                    computed_tokens: 17,
                    prefix: None,
                },
            )
            .unwrap();
        let committed = memory
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner.is_external_kv())
            .unwrap();
        assert_eq!(committed.committed_bytes, 8192);

        coordinator
            .touch("vllm:test:scheduler", &lease.lease_id)
            .unwrap();
        coordinator.heartbeat("vllm:test:scheduler").unwrap();
        coordinator
            .release("vllm:test:scheduler", &lease.lease_id, None)
            .unwrap();
        assert_eq!(coordinator.lease_count(), 0);
        assert!(memory
            .snapshot()
            .rows
            .iter()
            .all(|row| !row.owner.is_external_kv()));
    }

    #[test]
    fn shared_pool_issues_runtime_handles_and_requires_synchronized_release() {
        use kapsl_kv_abi::KvCoordinator as _;

        let backing = Arc::new(TestSharedBacking::default());
        let provisioner = Arc::new(TestSharedProvisioner {
            backing: backing.clone(),
        });
        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_secs(30),
            Some(provisioner),
        )
        .unwrap();
        let receipt = coordinator.register(&shared_registration()).unwrap();
        assert_eq!(receipt.shared_pools.len(), 1);
        let pools = coordinator
            .state
            .lock()
            .participants
            .get("vllm:test:scheduler")
            .unwrap()
            .shared_pools
            .clone()
            .unwrap();

        let lease = coordinator
            .reserve("vllm:test:scheduler", &reservation(None))
            .unwrap();
        assert_eq!(lease.groups[0].blocks.len(), 2);
        assert!(lease.groups[0].blocks.iter().all(|handle| matches!(
            handle,
            KvBlockHandle::RuntimePool {
                pool_id,
                generation: 1,
                ..
            } if pool_id == "test-binding-1"
        )));
        assert_eq!(backing.zeroed_blocks.load(Ordering::Relaxed), 2);
        assert_eq!(pools.available_blocks("vllm.pool.0"), 2);

        assert!(matches!(
            coordinator.release("vllm:test:scheduler", &lease.lease_id, None),
            Err(KvContractError::InvalidRequest { .. })
        ));
        assert_eq!(coordinator.lease_count(), 1);
        coordinator
            .release(
                "vllm:test:scheduler",
                &lease.lease_id,
                Some(&KvReleaseCompletion::BackendSynchronized),
            )
            .unwrap();
        assert_eq!(coordinator.lease_count(), 0);
        assert_eq!(pools.available_blocks("vllm.pool.0"), 4);
    }

    #[test]
    fn participant_managed_pool_leases_capacity_without_rewriting_backend_indices() {
        use kapsl_kv_abi::KvCoordinator as _;

        let backing = Arc::new(TestSharedBacking::default());
        let provisioner = Arc::new(TestSharedProvisioner {
            backing: backing.clone(),
        });
        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_secs(30),
            Some(provisioner),
        )
        .unwrap();
        let mut registration = shared_registration();
        registration
            .capabilities
            .features
            .insert(kapsl_kv_abi::KvFeature::ParticipantBlockSelection);
        let receipt = coordinator.register(&registration).unwrap();
        assert_eq!(
            receipt.shared_pools[0].allocation_mode,
            KvSharedPoolAllocationMode::ParticipantManaged
        );
        let pools = coordinator
            .state
            .lock()
            .participants
            .get("vllm:test:scheduler")
            .unwrap()
            .shared_pools
            .clone()
            .unwrap();

        let lease = coordinator
            .reserve("vllm:test:scheduler", &reservation(None))
            .unwrap();
        assert!(lease.groups[0].blocks.is_empty());
        assert_eq!(backing.zeroed_blocks.load(Ordering::Relaxed), 0);
        assert_eq!(pools.available_blocks("vllm.pool.0"), 2);
        coordinator
            .release("vllm:test:scheduler", &lease.lease_id, None)
            .unwrap();
        assert_eq!(pools.available_blocks("vllm.pool.0"), 4);
    }

    #[test]
    fn expired_shared_pool_blocks_are_quarantined_instead_of_reused() {
        use kapsl_kv_abi::KvCoordinator as _;

        let provisioner = Arc::new(TestSharedProvisioner {
            backing: Arc::new(TestSharedBacking::default()),
        });
        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_millis(5),
            Some(provisioner),
        )
        .unwrap();
        coordinator.register(&shared_registration()).unwrap();
        let pools = coordinator
            .state
            .lock()
            .participants
            .get("vllm:test:scheduler")
            .unwrap()
            .shared_pools
            .clone()
            .unwrap();
        coordinator
            .reserve("vllm:test:scheduler", &reservation(Some(5)))
            .unwrap();
        std::thread::sleep(Duration::from_millis(15));
        assert_eq!(coordinator.expire_stale(), 1);
        assert_eq!(pools.available_blocks("vllm.pool.0"), 2);
        assert_eq!(pools.quarantined_blocks("vllm.pool.0"), 2);

        let mut too_large = reservation(Some(5));
        too_large.sequence.request_id = "request-2".to_string();
        too_large.sequence.sequence_id = "request-2".to_string();
        too_large.groups[0].token_capacity = 48;
        assert!(matches!(
            coordinator.reserve("vllm:test:scheduler", &too_large),
            Err(KvContractError::CapacityExhausted { .. })
        ));
    }

    #[test]
    fn stale_lease_returns_capacity_to_the_authority() {
        use kapsl_kv_abi::KvCoordinator as _;

        let memory = test_memory();
        let coordinator =
            ExternalKvCoordinator::new(memory.clone(), Duration::from_millis(5)).unwrap();
        coordinator.register(&registration()).unwrap();
        coordinator
            .reserve("vllm:test:scheduler", &reservation(Some(5)))
            .unwrap();
        std::thread::sleep(Duration::from_millis(15));
        assert_eq!(coordinator.expire_stale(), 1);
        assert_eq!(coordinator.lease_count(), 0);
        assert!(memory
            .snapshot()
            .rows
            .iter()
            .all(|row| !row.owner.is_external_kv()));
    }

    #[test]
    fn registration_and_ttl_fail_closed_without_bounded_authority() {
        use kapsl_kv_abi::KvCoordinator as _;

        let memory = test_memory();
        let coordinator =
            ExternalKvCoordinator::new(memory.clone(), Duration::from_secs(30)).unwrap();
        let mut cuda_registration = registration();
        cuda_registration.capacity_model.groups[0].memory_domains =
            vec![KvMemoryDomain::Cuda { device_id: 0 }];
        assert!(matches!(
            coordinator.register(&cuda_registration),
            Err(KvContractError::InvalidCapabilities { .. })
        ));
        assert!(matches!(
            coordinator.register(&shared_registration()),
            Err(KvContractError::InvalidCapabilities { .. })
        ));

        coordinator.register(&registration()).unwrap();
        assert!(matches!(
            coordinator.reserve("vllm:test:scheduler", &reservation(Some(30_001))),
            Err(KvContractError::InvalidRequest { .. })
        ));
        assert_eq!(coordinator.lease_count(), 0);
        assert!(memory
            .snapshot()
            .rows
            .iter()
            .all(|row| !row.owner.is_external_kv()));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_server_dispatches_the_versioned_wire_envelope() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::net::UnixStream;

        let coordinator =
            ExternalKvCoordinator::new(test_memory(), Duration::from_secs(30)).unwrap();
        let path = std::env::temp_dir().join(format!(
            "kapsl-kv-control-test-{}-{}.sock",
            std::process::id(),
            coordinator.next_lease_id.load(Ordering::Relaxed)
        ));
        let server = KvControlServer::bind(&path, coordinator.clone())
            .await
            .unwrap();
        let task = tokio::spawn(server.run());

        let envelope = KvControlRequestEnvelope {
            abi_version: KAPSL_KV_ABI_VERSION,
            request_id: "rpc-register".to_string(),
            request: KvControlRequest::Register {
                registration: registration(),
            },
        };
        let mut stream = UnixStream::connect(&path).await.unwrap();
        let mut frame = serde_json::to_vec(&envelope).unwrap();
        frame.push(b'\n');
        stream.write_all(&frame).await.unwrap();
        let mut reader = BufReader::new(stream);
        let mut response = Vec::new();
        reader.read_until(b'\n', &mut response).await.unwrap();
        let response: KvControlResponseEnvelope = serde_json::from_slice(&response).unwrap();
        assert_eq!(response.request_id, "rpc-register");
        assert!(matches!(
            response.response,
            KvControlResponse::Registered { .. }
        ));
        assert_eq!(coordinator.participant_count(), 1);

        task.abort();
        let _ = task.await;
    }
}
