//! Out-of-process KV participant control plane.
//!
//! The transport is deliberately small: one newline-delimited JSON request per
//! Unix connection. Policy and byte accounting live in `ExternalKvCoordinator`;
//! framing only decodes the versioned `kapsl-kv-abi` envelopes.

use crate::runtime::managed::ManagedVllmKvReadinessFence;
use crate::runtime::memory::{
    MemoryAllocationClass, MemoryAuthority, MemoryClaim, MemoryDomain, MemoryLease, MemoryOwner,
    MemoryPlan, MemorySnapshot,
};
use base64::engine::general_purpose::URL_SAFE_NO_PAD as BASE64_URL_SAFE_NO_PAD;
use base64::Engine as _;
use kapsl_kv_abi::{
    dispatch_control_request, KvAdapterProfile, KvBlockHandle, KvCacheOwnership, KvCommitRequest,
    KvContractError, KvControlRequest, KvControlRequestEnvelope, KvControlResponse,
    KvControlResponseEnvelope, KvFeature, KvGroupLease, KvGroupReservation, KvIntegrationTier,
    KvLease, KvMemoryDomain, KvMetadataMode, KvParticipantRegistration, KvPoolResizeOperation,
    KvPoolResizeStage, KvProvisioningGrant, KvRegistrationReceipt, KvReleaseCompletion,
    KvReserveRequest, KvResizeAckRequest, KvResizeActor, KvResizePollRequest, KvResizePollResult,
    KvSequenceKey, KvSharedPoolAllocationMode, KvSharedPoolAttachment, KvSharedPoolDescriptor,
    KvSharedPoolDetachRequest, KvVmmSegmentDescriptor, KAPSL_KV_ABI_VERSION,
};
use parking_lot::Mutex;
use rand::rngs::OsRng;
use rand::RngCore;
use std::collections::{BTreeMap, BTreeSet, HashMap};
#[cfg(unix)]
use std::os::fd::OwnedFd;
#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_FRAME_BYTES: usize = 1024 * 1024;
const MAX_CONNECTIONS: usize = 64;
const RESIZE_ACK_TIMEOUT: Duration = Duration::from_secs(30);

/// Physical data-plane storage retained for the lifetime of a provisioned
/// shared pool. Implementations clear every newly assigned block before its
/// handle is published to a participant.
pub(crate) trait SharedPoolBacking: Send + Sync {
    fn zero_blocks(
        &self,
        binding: &KvSharedPoolDescriptor,
        block_indices: &[u64],
    ) -> Result<(), KvContractError>;

    /// Release every physical transport allocation after all importers have
    /// been fenced. Failure must leave enough internal state for a later
    /// supervised retry; the owning authority lease remains charged meanwhile.
    fn release_after_fence(&self) -> Result<(), KvContractError> {
        Ok(())
    }

    fn grow_binding(
        &self,
        _binding: &KvSharedPoolDescriptor,
        _target_block_count: u64,
        _resize_generation: u64,
    ) -> Result<Vec<KvVmmSegmentDescriptor>, KvContractError> {
        Err(KvContractError::unsupported("grow_shared_pool_backing"))
    }

    fn shrink_segments(
        &self,
        _binding: &KvSharedPoolDescriptor,
        _target_block_count: u64,
    ) -> Result<Vec<KvVmmSegmentDescriptor>, KvContractError> {
        Err(KvContractError::unsupported("inspect_shared_pool_shrink"))
    }

    /// Resolve a requested shrink to a physical segment boundary. CUDA VMM
    /// allocation handles cannot be partially released, so the result may be
    /// smaller than the requested block count while remaining above the
    /// certified minimum.
    fn shrink_target_boundary(
        &self,
        _binding: &KvSharedPoolDescriptor,
        requested_block_count: u64,
    ) -> Result<u64, KvContractError> {
        Ok(requested_block_count)
    }

    fn release_binding_tail(
        &self,
        _binding: &KvSharedPoolDescriptor,
        _target_block_count: u64,
    ) -> Result<(), KvContractError> {
        Err(KvContractError::unsupported("release_shared_pool_tail"))
    }

    #[cfg(unix)]
    fn export_vmm_segments(
        &self,
        _segments: &[KvVmmSegmentDescriptor],
    ) -> Result<Vec<OwnedFd>, KvContractError> {
        Err(KvContractError::unsupported("export_cuda_vmm_segments"))
    }
}

pub(crate) struct ProvisionedSharedPools {
    pub(crate) descriptors: Vec<KvSharedPoolDescriptor>,
    pub(crate) backing: Arc<dyn SharedPoolBacking>,
    /// Exact authority charge retained alongside the physical backing. A
    /// provisional grant moves its existing lease here without release or
    /// reacquisition; legacy registrations create it inside the provisioner.
    pub(crate) memory_lease: Option<MemoryLease>,
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
        precharged: Option<MemoryLease>,
        minimum_block_count: Option<u64>,
    ) -> Result<ProvisionedSharedPools, KvContractError>;

    fn live_resize_alignment_blocks(
        &self,
        _memory_domains: &BTreeSet<KvMemoryDomain>,
        _bytes_per_block: u64,
    ) -> Result<u64, KvContractError> {
        Err(KvContractError::unsupported(
            "query_live_shared_pool_alignment",
        ))
    }
}

#[derive(Clone)]
struct SharedGroupDefinition {
    capacity_pool_id: String,
    allocation_granularity_tokens: u32,
}

struct SharedPoolAllocatorState {
    free_by_pool: HashMap<String, Vec<u64>>,
    quarantined_by_pool: HashMap<String, BTreeSet<u64>>,
    mapped_blocks_by_pool: HashMap<String, u64>,
}

type ElasticPoolShape<'a> = (&'a str, &'a [KvSharedPoolDescriptor], u64, u64, u64, u64);

struct SharedPoolLeaseAllocation {
    blocks_by_pool: BTreeMap<String, Vec<u64>>,
    requires_release_fence: bool,
}

struct SharedPoolSet {
    owner: MemoryOwner,
    groups: HashMap<String, SharedGroupDefinition>,
    bindings_by_pool: HashMap<String, Vec<KvSharedPoolDescriptor>>,
    allocation_modes: HashMap<String, KvSharedPoolAllocationMode>,
    state: Mutex<SharedPoolAllocatorState>,
    backing: Arc<dyn SharedPoolBacking>,
    memory_lease: Option<Mutex<MemoryLease>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ManagedVllmKvDeviceSnapshot {
    pub(crate) device_id: u32,
    pub(crate) total_blocks: u64,
    pub(crate) allocated_blocks: u64,
    pub(crate) active_blocks: u64,
    pub(crate) idle_blocks: u64,
    pub(crate) quarantined_blocks: u64,
    pub(crate) backing_bytes: u64,
    pub(crate) logical_leased_bytes: u64,
    pub(crate) quarantine_bytes: u64,
    pub(crate) active_sequences: u64,
    pub(crate) participant_active: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ManagedVllmKvResizeSnapshot {
    pub(crate) minimum_block_count: u64,
    pub(crate) current_block_count: u64,
    pub(crate) maximum_block_count: u64,
    pub(crate) resize_alignment_blocks: u64,
    pub(crate) pending_generation: Option<u64>,
    pub(crate) pending_target_block_count: Option<u64>,
    pub(crate) failure: Option<String>,
}

impl SharedPoolSet {
    fn new(
        registration: &KvParticipantRegistration,
        receipt: &KvRegistrationReceipt,
        owner: MemoryOwner,
        mut provisioned: ProvisionedSharedPools,
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
        let mut mapped_blocks_by_pool = HashMap::new();
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
            let mapped_block_count = match bindings[0].elastic.as_ref() {
                Some(elastic) => {
                    if bindings.iter().any(|binding| {
                        binding.elastic.as_ref().is_none_or(|candidate| {
                            candidate.minimum_block_count != elastic.minimum_block_count
                                || candidate.mapped_block_count != elastic.mapped_block_count
                                || candidate.maximum_block_count != elastic.maximum_block_count
                                || candidate.resize_alignment_blocks
                                    != elastic.resize_alignment_blocks
                        })
                    }) {
                        return Err(KvContractError::invalid_capabilities(format!(
                            "replicas of elastic shared pool '{capacity_pool_id}' must expose identical mapped and maximum block geometry"
                        )));
                    }
                    elastic.mapped_block_count
                }
                None => {
                    if bindings.iter().any(|binding| binding.elastic.is_some()) {
                        return Err(KvContractError::invalid_capabilities(format!(
                            "shared pool '{capacity_pool_id}' cannot mix fixed and elastic bindings"
                        )));
                    }
                    block_count
                }
            };
            let mapped_block_count = usize::try_from(mapped_block_count).map_err(|_| {
                KvContractError::invalid_capabilities(format!(
                    "shared pool '{capacity_pool_id}' mapped block count does not fit this runtime"
                ))
            })?;
            free_by_pool.insert(
                capacity_pool_id.clone(),
                (0..mapped_block_count as u64).rev().collect(),
            );
            quarantined_by_pool.insert(capacity_pool_id.clone(), BTreeSet::new());
            mapped_blocks_by_pool.insert(capacity_pool_id.clone(), mapped_block_count as u64);
        }

        let memory_lease = provisioned.memory_lease.take().map(|mut lease| {
            lease.commit_capacity();
            Mutex::new(lease)
        });
        Ok(Arc::new(Self {
            owner,
            groups,
            bindings_by_pool,
            allocation_modes,
            state: Mutex::new(SharedPoolAllocatorState {
                free_by_pool,
                quarantined_by_pool,
                mapped_blocks_by_pool,
            }),
            backing: provisioned.backing,
            memory_lease,
        }))
    }

    fn release_backing_after_fence(&self) -> Result<(), KvContractError> {
        self.backing.release_after_fence()
    }

    fn elastic_shape(&self) -> Result<ElasticPoolShape<'_>, KvContractError> {
        if self.bindings_by_pool.len() != 1 {
            return Err(KvContractError::invalid_capabilities(
                "live resize currently requires exactly one aliased capacity pool",
            ));
        }
        let (pool_id, bindings) = self
            .bindings_by_pool
            .iter()
            .next()
            .expect("one elastic pool was required");
        let elastic = bindings
            .first()
            .and_then(|binding| binding.elastic.as_ref())
            .ok_or_else(|| KvContractError::unsupported("resize_fixed_shared_pool"))?;
        let current = self
            .state
            .lock()
            .mapped_blocks_by_pool
            .get(pool_id)
            .copied()
            .ok_or_else(|| KvContractError::Internal {
                message: "elastic shared pool has no mapped-capacity state".to_string(),
            })?;
        Ok((
            pool_id,
            bindings,
            elastic.minimum_block_count,
            current,
            elastic.maximum_block_count,
            elastic.resize_alignment_blocks,
        ))
    }

    fn resize_memory_plan(
        &self,
        bindings: &[KvSharedPoolDescriptor],
        from_block_count: u64,
        target_block_count: u64,
    ) -> Result<MemoryPlan, KvContractError> {
        let changed_blocks = from_block_count.abs_diff(target_block_count);
        let mut plan = MemoryPlan::new();
        for binding in bindings {
            let bytes = changed_blocks
                .checked_mul(binding.bytes_per_block)
                .and_then(|bytes| usize::try_from(bytes).ok())
                .ok_or_else(|| {
                    KvContractError::invalid_request("elastic KV resize byte size overflows")
                })?;
            plan.push(MemoryClaim::external(
                runtime_memory_domain(&binding.memory_domain)?,
                self.owner,
                MemoryAllocationClass::KvCache,
                format!("resize:{}", binding.binding_id),
                bytes,
            ));
        }
        Ok(plan)
    }

    fn prepare_grow(
        &self,
        target_block_count: u64,
        resize_generation: u64,
    ) -> Result<Vec<KvPoolResizeOperation>, KvContractError> {
        let (_pool_id, bindings, _minimum, current, maximum, alignment) = self.elastic_shape()?;
        if target_block_count <= current
            || target_block_count > maximum
            || !target_block_count.is_multiple_of(alignment)
        {
            return Err(KvContractError::invalid_request(
                "elastic KV growth target is outside or misaligned with certified capacity",
            ));
        }
        let plan = self.resize_memory_plan(bindings, current, target_block_count)?;
        let lease = self
            .memory_lease
            .as_ref()
            .ok_or_else(|| KvContractError::Internal {
                message: "elastic shared pool has no retained MemoryAuthority lease".to_string(),
            })?;
        lease
            .lock()
            .grow(&plan)
            .map_err(|message| KvContractError::CapacityExhausted { message })?;

        let mut next_handle_index = 0u32;
        let mut operations = Vec::with_capacity(bindings.len());
        let mut grown_bindings = Vec::with_capacity(bindings.len());
        let rollback = |error: KvContractError,
                        grown_bindings: &[&KvSharedPoolDescriptor]|
         -> KvContractError {
            let mut rollback_failure = None;
            for grown in grown_bindings.iter().rev() {
                if let Err(release_error) = self.backing.release_binding_tail(grown, current) {
                    rollback_failure = Some(release_error);
                    break;
                }
            }
            if rollback_failure.is_none() {
                if let Err(memory_error) = lease.lock().shrink_after_external_release(&plan) {
                    rollback_failure = Some(KvContractError::Internal {
                        message: format!(
                            "could not roll back failed elastic KV authority growth: {memory_error}"
                        ),
                    });
                }
            }
            match rollback_failure {
                Some(rollback_error) => KvContractError::Internal {
                    message: format!(
                        "elastic KV growth failed ({error}) and rollback retained the charge/backing ({rollback_error})"
                    ),
                },
                None => error,
            }
        };
        for binding in bindings {
            let mut segments =
                match self
                    .backing
                    .grow_binding(binding, target_block_count, resize_generation)
                {
                    Ok(segments) => segments,
                    Err(error) => {
                        // A CUDA failure can occur after a physical handle was
                        // created or mapped. Ask the backing to return this
                        // binding to the old boundary as well; only then may
                        // the authority charge be shrunk. Backings retain any
                        // ambiguous tail so this rollback fails closed.
                        grown_bindings.push(binding);
                        return Err(rollback(error, &grown_bindings));
                    }
                };
            grown_bindings.push(binding);
            for segment in &mut segments {
                segment.handle_index = match next_handle_index.checked_add(1) {
                    Some(next) => {
                        let current_handle = next_handle_index;
                        next_handle_index = next;
                        current_handle
                    }
                    None => {
                        return Err(rollback(
                            KvContractError::Internal {
                                message: "elastic KV resize handle index overflowed".to_string(),
                            },
                            &grown_bindings,
                        ));
                    }
                };
            }
            let elastic = binding.elastic.as_ref().expect("elastic shape checked");
            let operation = KvPoolResizeOperation {
                participant_epoch: binding.generation,
                resize_generation,
                binding_id: binding.binding_id.clone(),
                stage: KvPoolResizeStage::MapWorkers,
                from_block_count: current,
                target_block_count,
                bytes_per_block: binding.bytes_per_block,
                allocation_granularity_bytes: elastic.allocation_granularity_bytes,
                segments,
            };
            if let Err(error) = operation.validate() {
                return Err(rollback(error, &grown_bindings));
            }
            operations.push(operation);
        }
        Ok(operations)
    }

    fn prepare_shrink(
        &self,
        target_block_count: u64,
        resize_generation: u64,
    ) -> Result<Vec<KvPoolResizeOperation>, KvContractError> {
        let (pool_id, bindings, minimum, current, _maximum, alignment) = self.elastic_shape()?;
        if target_block_count < minimum
            || target_block_count >= current
            || !target_block_count.is_multiple_of(alignment)
        {
            return Err(KvContractError::invalid_request(
                "elastic KV shrink target is outside or misaligned with mapped capacity",
            ));
        }
        if self.allocation_modes.get(pool_id) == Some(&KvSharedPoolAllocationMode::RuntimeLeased) {
            let state = self.state.lock();
            let free = state
                .free_by_pool
                .get(pool_id)
                .expect("validated elastic pool has a free list")
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            let quarantined = state
                .quarantined_by_pool
                .get(pool_id)
                .expect("validated elastic pool has a quarantine set");
            if (target_block_count..current)
                .any(|block| !free.contains(&block) || quarantined.contains(&block))
            {
                return Err(KvContractError::invalid_request(
                    "elastic KV shrink can retire only an entirely free, non-quarantined tail",
                ));
            }
        }

        let mut next_handle_index = 0u32;
        let mut operations = Vec::with_capacity(bindings.len());
        for binding in bindings {
            let mut segments = self.backing.shrink_segments(binding, target_block_count)?;
            for segment in &mut segments {
                segment.handle_index = next_handle_index;
                next_handle_index =
                    next_handle_index
                        .checked_add(1)
                        .ok_or_else(|| KvContractError::Internal {
                            message: "elastic KV resize handle index overflowed".to_string(),
                        })?;
            }
            let elastic = binding.elastic.as_ref().expect("elastic shape checked");
            let operation = KvPoolResizeOperation {
                participant_epoch: binding.generation,
                resize_generation,
                binding_id: binding.binding_id.clone(),
                stage: KvPoolResizeStage::UnmapWorkers,
                from_block_count: current,
                target_block_count,
                bytes_per_block: binding.bytes_per_block,
                allocation_granularity_bytes: elastic.allocation_granularity_bytes,
                segments,
            };
            operation.validate()?;
            operations.push(operation);
        }
        Ok(operations)
    }

    fn normalize_shrink_target(&self, requested_block_count: u64) -> Result<u64, KvContractError> {
        let (_pool_id, bindings, minimum, current, _maximum, alignment) = self.elastic_shape()?;
        if requested_block_count < minimum
            || requested_block_count >= current
            || !requested_block_count.is_multiple_of(alignment)
        {
            return Err(KvContractError::invalid_request(
                "elastic KV shrink target is outside or misaligned with mapped capacity",
            ));
        }
        let mut resolved = None;
        for binding in bindings {
            let candidate = self
                .backing
                .shrink_target_boundary(binding, requested_block_count)?;
            if candidate < minimum
                || candidate > requested_block_count
                || candidate >= current
                || !candidate.is_multiple_of(alignment)
            {
                return Err(KvContractError::Internal {
                    message: "shared-pool backing returned an invalid shrink boundary".to_string(),
                });
            }
            match resolved {
                None => resolved = Some(candidate),
                Some(previous) if previous == candidate => {}
                Some(_) => {
                    return Err(KvContractError::invalid_capabilities(
                        "elastic replicas do not share the same physical segment boundaries",
                    ));
                }
            }
        }
        resolved.ok_or_else(|| KvContractError::Internal {
            message: "elastic shared pool has no bindings".to_string(),
        })
    }

    fn worker_operations(
        operations: &[KvPoolResizeOperation],
        stage: KvPoolResizeStage,
    ) -> Vec<KvPoolResizeOperation> {
        operations
            .iter()
            .cloned()
            .map(|mut operation| {
                operation.stage = stage;
                operation
            })
            .collect()
    }

    fn scheduler_operations(
        operations: &[KvPoolResizeOperation],
        stage: KvPoolResizeStage,
    ) -> Vec<KvPoolResizeOperation> {
        operations
            .iter()
            .cloned()
            .map(|mut operation| {
                operation.stage = stage;
                operation.segments.clear();
                operation
            })
            .collect()
    }

    fn commit_grow(
        &self,
        from_block_count: u64,
        target_block_count: u64,
    ) -> Result<(), KvContractError> {
        let (pool_id, _bindings, _minimum, current, _maximum, _alignment) = self.elastic_shape()?;
        if current != from_block_count {
            return Err(KvContractError::Internal {
                message: "elastic KV mapped capacity changed during growth".to_string(),
            });
        }
        let mut state = self.state.lock();
        state
            .free_by_pool
            .get_mut(pool_id)
            .expect("validated elastic pool has a free list")
            .extend((from_block_count..target_block_count).rev());
        *state
            .mapped_blocks_by_pool
            .get_mut(pool_id)
            .expect("validated elastic pool has mapped state") = target_block_count;
        drop(state);
        self.memory_lease
            .as_ref()
            .expect("elastic shared pool retains its authority lease")
            .lock()
            .commit_capacity();
        Ok(())
    }

    fn commit_shrink(
        &self,
        from_block_count: u64,
        target_block_count: u64,
    ) -> Result<(), KvContractError> {
        let (pool_id, bindings, _minimum, current, _maximum, _alignment) = self.elastic_shape()?;
        if current != from_block_count {
            return Err(KvContractError::Internal {
                message: "elastic KV mapped capacity changed during shrink".to_string(),
            });
        }
        for binding in bindings {
            self.backing
                .release_binding_tail(binding, target_block_count)?;
        }
        {
            let mut state = self.state.lock();
            state
                .free_by_pool
                .get_mut(pool_id)
                .expect("validated elastic pool has a free list")
                .retain(|block| *block < target_block_count);
            *state
                .mapped_blocks_by_pool
                .get_mut(pool_id)
                .expect("validated elastic pool has mapped state") = target_block_count;
        }
        let plan = self.resize_memory_plan(bindings, from_block_count, target_block_count)?;
        self.memory_lease
            .as_ref()
            .expect("elastic shared pool retains its authority lease")
            .lock()
            .shrink_after_external_release(&plan)
            .map_err(|message| KvContractError::Internal {
                message: format!(
                    "released elastic KV backing but could not return MemoryAuthority bytes: {message}"
                ),
            })?;
        Ok(())
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

    fn managed_device_snapshots(
        &self,
        active_sequences: u64,
        participant_active: bool,
    ) -> Vec<ManagedVllmKvDeviceSnapshot> {
        let allocator = self.state.lock();
        let mut devices = BTreeMap::<u32, ManagedVllmKvDeviceSnapshot>::new();
        for (pool_id, bindings) in &self.bindings_by_pool {
            let mapped_blocks = allocator
                .mapped_blocks_by_pool
                .get(pool_id)
                .copied()
                .unwrap_or_default();
            let idle_blocks = allocator
                .free_by_pool
                .get(pool_id)
                .map_or(0_u64, |blocks| blocks.len() as u64);
            let quarantined_blocks = allocator
                .quarantined_by_pool
                .get(pool_id)
                .map_or(0_u64, |blocks| blocks.len() as u64);
            for binding in bindings {
                let KvMemoryDomain::Cuda { device_id } = &binding.memory_domain else {
                    continue;
                };
                let active_blocks =
                    mapped_blocks.saturating_sub(idle_blocks.saturating_add(quarantined_blocks));
                let snapshot =
                    devices
                        .entry(*device_id)
                        .or_insert_with(|| ManagedVllmKvDeviceSnapshot {
                            device_id: *device_id,
                            active_sequences,
                            participant_active,
                            ..Default::default()
                        });
                snapshot.total_blocks = snapshot.total_blocks.saturating_add(binding.block_count);
                snapshot.allocated_blocks = snapshot.allocated_blocks.saturating_add(mapped_blocks);
                snapshot.active_blocks = snapshot.active_blocks.saturating_add(active_blocks);
                snapshot.idle_blocks = snapshot.idle_blocks.saturating_add(idle_blocks);
                snapshot.quarantined_blocks = snapshot
                    .quarantined_blocks
                    .saturating_add(quarantined_blocks);
                snapshot.backing_bytes = snapshot
                    .backing_bytes
                    .saturating_add(mapped_blocks.saturating_mul(binding.bytes_per_block));
                snapshot.logical_leased_bytes = snapshot
                    .logical_leased_bytes
                    .saturating_add(active_blocks.saturating_mul(binding.bytes_per_block));
                snapshot.quarantine_bytes = snapshot
                    .quarantine_bytes
                    .saturating_add(quarantined_blocks.saturating_mul(binding.bytes_per_block));
            }
        }
        devices.into_values().collect()
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
    shared_activation: Option<SharedPoolActivation>,
    resize: Option<SharedPoolResizeRecord>,
}

struct ParticipantSnapshot {
    registration: KvParticipantRegistration,
    owner: MemoryOwner,
    shared_pools: Option<Arc<SharedPoolSet>>,
    active: bool,
}

struct SharedPoolActivation {
    attachments: BTreeMap<String, KvSharedPoolAttachment>,
    active: bool,
}

struct SharedPoolResizeRecord {
    generation: u64,
    from_block_count: u64,
    target_block_count: u64,
    stage: KvPoolResizeStage,
    physical_operations: Vec<KvPoolResizeOperation>,
    worker_acks: BTreeSet<String>,
    scheduler_acks: BTreeSet<String>,
    deadline: Instant,
    failure: Option<String>,
}

fn densify_worker_resize_handle_indices(
    operations: &mut [KvPoolResizeOperation],
) -> Result<(), KvContractError> {
    let mut next_handle_index = 0u32;
    for operation in operations {
        operation
            .segments
            .sort_by_key(|segment| segment.offset_bytes);
        for segment in &mut operation.segments {
            segment.handle_index = next_handle_index;
            next_handle_index =
                next_handle_index
                    .checked_add(1)
                    .ok_or_else(|| KvContractError::Internal {
                        message: "resize response handle index overflowed".to_string(),
                    })?;
        }
    }
    Ok(())
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

/// One whole-block exact KV candidate, ordered from preferred to hard minimum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProvisionalKvCandidate {
    pub(crate) block_count: u64,
    pub(crate) bytes_per_block: u64,
    pub(crate) effective_target_concurrency: u64,
}

/// Certified scope used to precharge one managed external KV backing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProvisionalKvGrantRequest {
    pub(crate) participant_base: String,
    pub(crate) model_fingerprint: String,
    pub(crate) geometry_digest: String,
    pub(crate) adapter_profile: KvAdapterProfile,
    pub(crate) capacity_pool_id: String,
    pub(crate) group_ids: BTreeSet<String>,
    pub(crate) memory_domains: BTreeSet<KvMemoryDomain>,
    pub(crate) candidates: Vec<ProvisionalKvCandidate>,
    /// Maximum virtual block count for an opt-in live CUDA VMM pool. The
    /// selected candidate remains the initial physical MemoryAuthority charge.
    pub(crate) maximum_block_count: Option<u64>,
    pub(crate) ttl: Duration,
}

/// Result of one atomic exact-KV selection and MemoryAuthority reservation.
pub(crate) struct ProvisionalKvGrant {
    pub(crate) proof: KvProvisioningGrant,
    pub(crate) selected_candidate: ProvisionalKvCandidate,
    pub(crate) selected_candidate_index: usize,
    pub(crate) authority_snapshot: MemorySnapshot,
}

struct ProvisionalKvGrantRecord {
    proof: KvProvisioningGrant,
    participant_base: String,
    model_fingerprint: String,
    adapter_profile: KvAdapterProfile,
    capacity_pool_id: String,
    group_ids: BTreeSet<String>,
    memory_domains: BTreeSet<KvMemoryDomain>,
    candidate: ProvisionalKvCandidate,
    minimum_block_count: u64,
    maximum_block_count: Option<u64>,
    owner: MemoryOwner,
    lease: Option<MemoryLease>,
    expires_at: Instant,
}

impl ProvisionalKvGrantRecord {
    fn validate_registration(
        &self,
        registration: &KvParticipantRegistration,
        proof: &KvProvisioningGrant,
        now: Instant,
    ) -> Result<(), KvContractError> {
        if now >= self.expires_at {
            return Err(KvContractError::invalid_request(
                "provisioning grant expired before participant registration",
            ));
        }
        if proof != &self.proof {
            return Err(KvContractError::invalid_request(
                "provisioning grant proof does not match the authority record",
            ));
        }
        let suffix = registration
            .participant_id
            .strip_prefix(&self.participant_base)
            .filter(|suffix| suffix.starts_with(':') && suffix.len() > 1);
        if suffix.is_none() {
            return Err(KvContractError::invalid_request(
                "participant identity is outside the provisioning grant namespace",
            ));
        }
        if registration.backend != "vllm"
            || registration.model_fingerprint != self.model_fingerprint
            || registration.adapter_profile.as_ref() != Some(&self.adapter_profile)
        {
            return Err(KvContractError::invalid_request(
                "participant model/backend/profile does not match the provisioning grant",
            ));
        }

        let capacity_group_ids = registration
            .capacity_model
            .groups
            .iter()
            .map(|group| group.group_id.clone())
            .collect::<BTreeSet<_>>();
        let live_resize = registration
            .capabilities
            .features
            .contains(&KvFeature::LivePoolResize);
        let expected_block_count = if live_resize {
            self.maximum_block_count.ok_or_else(|| {
                KvContractError::invalid_capabilities(
                    "live-resize registration used a fixed provisioning grant",
                )
            })?
        } else {
            if self.maximum_block_count.is_some() {
                return Err(KvContractError::invalid_capabilities(
                    "elastic provisioning grant requires the live_pool_resize feature",
                ));
            }
            self.candidate.block_count
        };
        if capacity_group_ids != self.group_ids
            || registration.capacity_model.groups.iter().any(|group| {
                group.pool_id != self.capacity_pool_id
                    || group.bytes_per_allocation != self.candidate.bytes_per_block
                    || group.max_allocations != Some(expected_block_count)
                    || group
                        .memory_domains
                        .iter()
                        .cloned()
                        .collect::<BTreeSet<_>>()
                        != self.memory_domains
            })
        {
            return Err(KvContractError::invalid_capabilities(
                "participant capacity geometry does not exactly match the provisioning grant",
            ));
        }
        Ok(())
    }
}

fn validate_provisional_grant_request(request: &ProvisionalKvGrantRequest) -> Result<(), String> {
    if request.participant_base.trim().is_empty() || request.participant_base.contains(':') {
        return Err(
            "provisional KV participant base must be non-empty and cannot contain ':'".to_string(),
        );
    }
    if request.model_fingerprint.trim().is_empty()
        || request.capacity_pool_id.trim().is_empty()
        || request.group_ids.is_empty()
        || request
            .group_ids
            .iter()
            .any(|group| group.trim().is_empty())
        || request.memory_domains.is_empty()
    {
        return Err(
            "provisional KV grant requires model, pool, group, and memory-domain identity"
                .to_string(),
        );
    }
    request
        .adapter_profile
        .validate()
        .map_err(|error| format!("invalid provisional KV adapter profile: {error}"))?;
    KvProvisioningGrant {
        token: "validation".to_string(),
        geometry_digest: request.geometry_digest.clone(),
        authority_generation: 1,
        expires_at_unix_ms: 1,
    }
    .validate()
    .map_err(|error| format!("invalid provisional KV geometry digest: {error}"))?;
    if request.ttl < Duration::from_secs(1) || request.ttl > Duration::from_secs(3600) {
        return Err("provisional KV grant TTL must be between 1 and 3600 seconds".to_string());
    }
    if request.candidates.is_empty() {
        return Err("provisional KV grant requires at least one sizing candidate".to_string());
    }
    let bytes_per_block = request.candidates[0].bytes_per_block;
    if request.maximum_block_count.is_some_and(|maximum| {
        maximum < request.candidates[0].block_count
            || maximum.checked_mul(bytes_per_block).is_none()
    }) {
        return Err(
            "elastic KV maximum must be overflow-safe and no smaller than the preferred initial candidate"
                .to_string(),
        );
    }
    let mut previous: Option<(u64, u64)> = None;
    for candidate in &request.candidates {
        if candidate.block_count == 0
            || candidate.bytes_per_block == 0
            || candidate.effective_target_concurrency == 0
            || candidate.bytes_per_block != bytes_per_block
            || previous.is_some_and(|(blocks, concurrency)| {
                candidate.block_count >= blocks
                    || candidate.effective_target_concurrency > concurrency
            })
            || candidate
                .block_count
                .checked_mul(candidate.bytes_per_block)
                .is_none()
        {
            return Err(
                "provisional KV candidates must be positive, overflow-safe, share one stride, and strictly decrease by whole block count"
                    .to_string(),
            );
        }
        previous = Some((
            candidate.block_count,
            candidate.effective_target_concurrency,
        ));
    }
    for domain in &request.memory_domains {
        runtime_memory_domain(domain)
            .map_err(|error| format!("invalid provisional KV memory domain: {error}"))?;
    }
    Ok(())
}

fn provisional_memory_plan(
    request: &ProvisionalKvGrantRequest,
    candidate: &ProvisionalKvCandidate,
    owner: MemoryOwner,
    token: &str,
) -> Result<MemoryPlan, String> {
    let bytes = candidate
        .block_count
        .checked_mul(candidate.bytes_per_block)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| "provisional KV candidate byte size exceeds this runtime".to_string())?;
    let mut plan = MemoryPlan::new();
    for (index, domain) in request.memory_domains.iter().enumerate() {
        let domain = runtime_memory_domain(domain)
            .map_err(|error| format!("invalid provisional KV memory domain: {error}"))?;
        plan.push(MemoryClaim::external(
            domain,
            owner,
            MemoryAllocationClass::KvCache,
            format!("provisional:{token}:{}:{index}", request.capacity_pool_id),
            bytes,
        ));
    }
    Ok(plan)
}

fn new_provisioning_token() -> String {
    let mut bytes = [0_u8; 32];
    OsRng.fill_bytes(&mut bytes);
    format!("kvg1_{}", BASE64_URL_SAFE_NO_PAD.encode(bytes))
}

#[derive(Default)]
struct CoordinatorState {
    participants: HashMap<String, ParticipantRecord>,
    leases: HashMap<String, ExternalLeaseRecord>,
    sequences: HashMap<(String, KvSequenceKey), String>,
    managed_readiness_fences: HashMap<String, Weak<ManagedVllmKvReadinessFence>>,
    provisional_grants: HashMap<String, ProvisionalKvGrantRecord>,
}

fn participant_matches_managed_base(participant_id: &str, participant_base: &str) -> bool {
    participant_id == participant_base
        || participant_id
            .strip_prefix(participant_base)
            .is_some_and(|suffix| suffix.starts_with(':'))
}

fn managed_bases_overlap(left: &str, right: &str) -> bool {
    participant_matches_managed_base(left, right) || participant_matches_managed_base(right, left)
}

fn managed_participant_id_locked(
    state: &CoordinatorState,
    participant_base: &str,
) -> Result<Option<String>, String> {
    let mut matches = state.participants.keys().filter(|participant_id| {
        participant_matches_managed_base(participant_id, participant_base)
    });
    let Some(participant_id) = matches.next().cloned() else {
        return Ok(None);
    };
    if matches.next().is_some() {
        return Err(format!(
            "multiple live KV participants match managed base '{participant_base}'"
        ));
    }
    Ok(Some(participant_id))
}

/// Parse exact adapter/backend tuples produced by the hardware conformance
/// suite. This is an operator allowlist, not remote attestation: the local
/// control socket remains the trust boundary for participant identity.
#[cfg(any(test, all(feature = "gpu-device-pool", target_os = "linux")))]
pub(crate) fn parse_shared_pool_profiles(
    values: &[String],
) -> Result<BTreeSet<KvAdapterProfile>, String> {
    values
        .iter()
        .map(|value| {
            let fields = value.split(',').map(str::trim).collect::<Vec<_>>();
            if fields.len() != 4 {
                return Err(format!(
                    "invalid --kv-shared-pool-profile '{value}': expected adapter_id,adapter_version,backend_version,profile_id"
                ));
            }
            let profile = KvAdapterProfile {
                adapter_id: fields[0].to_string(),
                adapter_version: fields[1].to_string(),
                backend_version: fields[2].to_string(),
                profile_id: fields[3].to_string(),
            };
            profile.validate().map_err(|error| {
                format!("invalid --kv-shared-pool-profile '{value}': {error}")
            })?;
            Ok(profile)
        })
        .collect()
}

fn validate_shared_attachment(
    record: &ParticipantRecord,
    attachment: &KvSharedPoolAttachment,
) -> Result<(), KvContractError> {
    attachment.validate()?;
    if record.receipt.participant_epoch != attachment.participant_epoch {
        return Err(KvContractError::invalid_request(
            "shared-pool attachment epoch does not match the registration receipt",
        ));
    }
    if !record
        .registration
        .capabilities
        .features
        .contains(&KvFeature::ExternalPoolAttachment)
    {
        return Err(KvContractError::invalid_capabilities(
            "participant did not advertise external shared-pool attachment",
        ));
    }
    let binding = record
        .receipt
        .shared_pools
        .iter()
        .find(|binding| binding.binding_id == attachment.binding_id)
        .ok_or_else(|| {
            KvContractError::invalid_request(
                "attachment references a binding outside the registration receipt",
            )
        })?;
    let expected_bytes = binding
        .block_count
        .checked_mul(binding.bytes_per_block)
        .ok_or_else(|| KvContractError::Internal {
            message: "validated shared-pool byte size overflowed".to_string(),
        })?;
    if attachment.imported_bytes != expected_bytes {
        return Err(KvContractError::invalid_request(format!(
            "attachment imported {} bytes for binding '{}', expected {expected_bytes}",
            attachment.imported_bytes, binding.binding_id
        )));
    }
    let expected_mapped_bytes = binding.elastic.as_ref().map(|elastic| {
        elastic
            .mapped_block_count
            .checked_mul(binding.bytes_per_block)
            .expect("validated elastic binding byte size")
    });
    if attachment.mapped_bytes != expected_mapped_bytes {
        return Err(KvContractError::invalid_request(format!(
            "attachment mapped-byte evidence for binding '{}' does not match the provisioned physical prefix",
            binding.binding_id
        )));
    }

    let topology = record
        .registration
        .topology
        .as_ref()
        .expect("validated shared-pool registration has structured topology");
    if attachment.shard.tensor_parallel_world_size != topology.shard.tensor_parallel_world_size
        || attachment.shard.pipeline_parallel_world_size
            != topology.shard.pipeline_parallel_world_size
    {
        return Err(KvContractError::invalid_topology(
            "attachment shard world sizes do not match the registered topology",
        ));
    }

    let bound_groups = binding.group_ids.iter().collect::<BTreeSet<_>>();
    let expected_layers = topology
        .cache_groups
        .iter()
        .filter(|group| bound_groups.contains(&group.group_id))
        .flat_map(|group| {
            group
                .layers
                .iter()
                .map(|layer| (group.group_id.as_str(), layer.index, layer.name.as_deref()))
        })
        .collect::<BTreeSet<_>>();
    let mut attached_layers = BTreeSet::new();
    for view in &attachment.views {
        let end = view
            .offset_bytes
            .checked_add(view.length_bytes)
            .expect("attachment validation checked view overflow");
        if end > attachment.imported_bytes {
            return Err(KvContractError::invalid_request(format!(
                "attachment view for group '{}' layer {} exceeds binding '{}'",
                view.group_id, view.layer.index, binding.binding_id
            )));
        }
        attached_layers.insert((
            view.group_id.as_str(),
            view.layer.index,
            view.layer.name.as_deref(),
        ));
    }
    if attached_layers != expected_layers {
        return Err(KvContractError::invalid_request(
            "attachment views do not cover the binding's registered cache layers",
        ));
    }

    let activation = record
        .shared_activation
        .as_ref()
        .expect("shared participant has activation state");
    if activation
        .attachments
        .values()
        .any(|existing| existing.profile != attachment.profile)
    {
        return Err(KvContractError::invalid_request(
            "all participant bindings must use the same certified adapter profile",
        ));
    }
    if record.registration.adapter_profile.as_ref() != Some(&attachment.profile) {
        return Err(KvContractError::invalid_request(
            "attachment profile does not match the profile declared before provisioning",
        ));
    }
    if activation.attachments.values().any(|existing| {
        existing.binding_id != attachment.binding_id
            && record
                .receipt
                .shared_pools
                .iter()
                .find(|binding| binding.binding_id == existing.binding_id)
                .is_some_and(|existing_binding| {
                    existing_binding.capacity_pool_id == binding.capacity_pool_id
                        && existing.shard.tensor_parallel_rank
                            == attachment.shard.tensor_parallel_rank
                        && existing.shard.pipeline_parallel_rank
                            == attachment.shard.pipeline_parallel_rank
                })
    }) {
        return Err(KvContractError::invalid_request(
            "replicas of one capacity pool must attach from distinct parallel ranks",
        ));
    }
    Ok(())
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
    next_provisioning_generation: AtomicU64,
    next_resize_generation: AtomicU64,
    next_lease_id: AtomicU64,
    maximum_lease_ttl: Duration,
    shared_pool_provisioner: Option<Arc<dyn SharedPoolProvisioner>>,
    allowed_shared_pool_profiles: BTreeSet<KvAdapterProfile>,
}

impl ExternalKvCoordinator {
    pub(crate) fn new(
        memory: Arc<MemoryAuthority>,
        maximum_lease_ttl: Duration,
    ) -> Result<Arc<Self>, String> {
        Self::new_with_shared_pool_provisioner(memory, maximum_lease_ttl, None, BTreeSet::new())
    }

    pub(crate) fn new_with_shared_pool_provisioner(
        memory: Arc<MemoryAuthority>,
        maximum_lease_ttl: Duration,
        shared_pool_provisioner: Option<Arc<dyn SharedPoolProvisioner>>,
        allowed_shared_pool_profiles: BTreeSet<KvAdapterProfile>,
    ) -> Result<Arc<Self>, String> {
        if maximum_lease_ttl.is_zero() {
            return Err("KV control lease TTL must be non-zero".to_string());
        }
        Ok(Arc::new(Self {
            memory,
            state: Mutex::new(CoordinatorState::default()),
            next_participant_slot: AtomicU32::new(0),
            next_participant_epoch: AtomicU64::new(1),
            next_provisioning_generation: AtomicU64::new(1),
            next_resize_generation: AtomicU64::new(1),
            next_lease_id: AtomicU64::new(1),
            maximum_lease_ttl,
            shared_pool_provisioner,
            allowed_shared_pool_profiles,
        }))
    }

    /// Atomically select and reserve one exact external KV candidate before
    /// the managed backend process is allowed to start.
    pub(crate) fn reserve_provisional_kv_grant(
        &self,
        request: &ProvisionalKvGrantRequest,
    ) -> Result<ProvisionalKvGrant, String> {
        validate_provisional_grant_request(request)?;
        self.expire_provisional_grants();

        let slot = self.next_participant_slot.fetch_add(1, Ordering::Relaxed);
        let owner = MemoryOwner::external_kv(slot)
            .ok_or_else(|| "external KV participant owner space exhausted".to_string())?;
        let generation = self
            .next_provisioning_generation
            .fetch_add(1, Ordering::Relaxed);
        if generation == 0 {
            return Err("KV provisioning generation space exhausted".to_string());
        }
        let proof = KvProvisioningGrant {
            token: new_provisioning_token(),
            geometry_digest: request.geometry_digest.clone(),
            authority_generation: generation,
            expires_at_unix_ms: unix_ms_after(request.ttl),
        };
        proof
            .validate()
            .map_err(|error| format!("invalid generated KV provisioning proof: {error}"))?;

        let candidates = request
            .candidates
            .iter()
            .map(|candidate| provisional_memory_plan(request, candidate, owner, &proof.token))
            .collect::<Result<Vec<_>, _>>()?;

        // Coordinator state serializes participant/grant namespaces while
        // MemoryAuthority serializes the snapshot, candidate decision, and
        // physical adapter reservations. No child can register between them.
        let mut state = self.state.lock();
        if state.participants.keys().any(|participant_id| {
            participant_matches_managed_base(participant_id, &request.participant_base)
        }) || state
            .provisional_grants
            .values()
            .any(|grant| managed_bases_overlap(&grant.participant_base, &request.participant_base))
        {
            return Err(format!(
                "managed participant base '{}' already has a live participant or provisional grant",
                request.participant_base
            ));
        }
        if state.provisional_grants.contains_key(&proof.token) {
            return Err("cryptographic KV provisioning token collision".to_string());
        }
        let (lease, selected_candidate_index, authority_snapshot) =
            self.memory.admit_first_fitting(&candidates)?;
        let selected_candidate = request.candidates[selected_candidate_index].clone();
        let minimum_block_count = request
            .candidates
            .last()
            .expect("validated provisional candidates are non-empty")
            .block_count;
        state.provisional_grants.insert(
            proof.token.clone(),
            ProvisionalKvGrantRecord {
                proof: proof.clone(),
                participant_base: request.participant_base.clone(),
                model_fingerprint: request.model_fingerprint.clone(),
                adapter_profile: request.adapter_profile.clone(),
                capacity_pool_id: request.capacity_pool_id.clone(),
                group_ids: request.group_ids.clone(),
                memory_domains: request.memory_domains.clone(),
                candidate: selected_candidate.clone(),
                minimum_block_count,
                maximum_block_count: request.maximum_block_count,
                owner,
                lease: Some(lease),
                expires_at: Instant::now() + request.ttl,
            },
        );
        Ok(ProvisionalKvGrant {
            proof,
            selected_candidate,
            selected_candidate_index,
            authority_snapshot,
        })
    }

    /// Release every unused provisional charge for one supervised namespace.
    pub(crate) fn cancel_provisional_kv_grants(&self, participant_base: &str) -> usize {
        let grants = {
            let mut state = self.state.lock();
            let tokens = state
                .provisional_grants
                .iter()
                .filter_map(|(token, grant)| {
                    managed_bases_overlap(&grant.participant_base, participant_base)
                        .then_some(token.clone())
                })
                .collect::<Vec<_>>();
            tokens
                .into_iter()
                .filter_map(|token| state.provisional_grants.remove(&token))
                .collect::<Vec<_>>()
        };
        let count = grants.len();
        drop(grants);
        count
    }

    fn expire_provisional_grants(&self) -> usize {
        let now = Instant::now();
        let grants = {
            let mut state = self.state.lock();
            let tokens = state
                .provisional_grants
                .iter()
                .filter_map(|(token, grant)| (grant.expires_at <= now).then_some(token.clone()))
                .collect::<Vec<_>>();
            tokens
                .into_iter()
                .filter_map(|token| state.provisional_grants.remove(&token))
                .collect::<Vec<_>>()
        };
        let count = grants.len();
        if count != 0 {
            log::warn!("[kv-control] expired {count} unused KV provisioning grant(s)");
        }
        drop(grants);
        count
    }

    pub(crate) fn expire_stale(&self) -> usize {
        self.expire_provisional_grants();
        let now = Instant::now();
        let (mut expired, resize_timeouts) = {
            let mut state = self.state.lock();
            let resize_timeouts = state
                .participants
                .iter()
                .filter_map(|(participant_id, participant)| {
                    participant.resize.as_ref().and_then(|resize| {
                        (resize.failure.is_none() && resize.deadline <= now)
                            .then_some(participant_id.clone())
                    })
                })
                .collect::<Vec<_>>();
            for participant_id in &resize_timeouts {
                Self::fence_managed_readiness_locked(&mut state, participant_id);
                if let Some(participant) = state.participants.get_mut(participant_id) {
                    if let Some(activation) = participant.shared_activation.as_mut() {
                        activation.active = false;
                    }
                    if let Some(resize) = participant.resize.as_mut() {
                        resize.failure =
                            Some("live KV resize acknowledgement deadline expired".to_string());
                    }
                }
            }
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
            (expired, resize_timeouts.len())
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
        if resize_timeouts > 0 {
            log::error!(
                "[kv-control] fenced {resize_timeouts} participant(s) after a live KV resize timeout; mapped capacity is retained until supervised restart"
            );
        }
        count.saturating_add(resize_timeouts)
    }

    /// Bind the stable participant namespace configured for one managed vLLM
    /// process to its lock-free readiness fence. The coordinator retains only
    /// a weak reference so participant bookkeeping cannot extend the process
    /// lifecycle or form a coordinator/process ownership cycle.
    pub(crate) fn register_managed_readiness_fence(
        &self,
        participant_base: &str,
        fence: Weak<ManagedVllmKvReadinessFence>,
    ) -> Result<(), String> {
        if participant_base.is_empty() {
            return Err("managed participant readiness base must not be empty".to_string());
        }
        if fence.upgrade().is_none() {
            return Err(format!(
                "managed participant readiness fence for '{participant_base}' has expired"
            ));
        }

        let mut state = self.state.lock();
        state
            .managed_readiness_fences
            .retain(|_, candidate| candidate.strong_count() > 0);
        if let Some((conflict, _)) = state.managed_readiness_fences.iter().find(|(base, _)| {
            base.as_str() != participant_base
                && managed_bases_overlap(base.as_str(), participant_base)
        }) {
            return Err(format!(
                "managed participant readiness base '{participant_base}' overlaps registered base '{conflict}'"
            ));
        }
        if let Some(current) = state.managed_readiness_fences.get(participant_base) {
            if Weak::ptr_eq(current, &fence) {
                return Ok(());
            }
            return Err(format!(
                "managed participant readiness base '{participant_base}' already has a live owner"
            ));
        }
        state
            .managed_readiness_fences
            .insert(participant_base.to_string(), fence);
        Ok(())
    }

    /// Advance the matching managed process's lock-free activation epoch.
    ///
    /// This is deliberately called while coordinator state is held, at the
    /// linearization point immediately before activation/backing mutation. It
    /// must remain atomic-only: taking a process lifecycle lock here would
    /// introduce a coordinator-state -> process-lifecycle lock edge.
    fn fence_managed_readiness_locked(state: &mut CoordinatorState, participant_id: &str) {
        let mut matched = 0usize;
        state.managed_readiness_fences.retain(|base, candidate| {
            let Some(fence) = candidate.upgrade() else {
                return false;
            };
            if participant_matches_managed_base(participant_id, base) {
                fence.advance();
                matched += 1;
            }
            true
        });
        debug_assert!(
            matched <= 1,
            "managed readiness base registration must be delimiter-unambiguous"
        );
    }

    /// Retire every control-plane object owned by a participant after its
    /// backend process tree has been reaped.
    ///
    /// This is the runtime-side lifecycle fence for supervised backends. A
    /// worker cannot reliably send the ordinary detach RPC after process exit,
    /// but once the supervisor has proved that no importer survives, retaining
    /// its registration would leak the isolated CUDA IPC backing indefinitely.
    #[cfg(test)]
    pub(crate) fn retire_participant_after_backend_exit(
        &self,
        participant_id: &str,
    ) -> Result<bool, String> {
        self.retire_one_participant_after_backend_exit(participant_id)
    }

    /// Retire every concrete participant identity derived from one supervised
    /// base ID. vLLM appends its generated engine UUID to the configured Kapsl
    /// participant ID, so the parent knows the stable namespace rather than
    /// the final registration key. The delimiter check prevents one model's
    /// base (for example `model-1`) from matching an unrelated `model-10`.
    pub(crate) fn retire_participants_after_backend_exit(
        &self,
        participant_base: &str,
    ) -> Result<usize, String> {
        let cancelled_grants = self.cancel_provisional_kv_grants(participant_base);
        let participant_ids = self
            .state
            .lock()
            .participants
            .keys()
            .filter(|participant_id| {
                participant_matches_managed_base(participant_id, participant_base)
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut retired = 0usize;
        for participant_id in &participant_ids {
            if self.retire_one_participant_after_backend_exit(participant_id)? {
                retired += 1;
            }
        }
        if cancelled_grants != 0 {
            log::info!(
                "[kv-control] cancelled {cancelled_grants} unused KV provisioning grant(s) for managed base '{participant_base}'"
            );
        }
        Ok(retired)
    }

    /// Return whether the one concrete participant derived from a supervised
    /// managed-backend base ID has completed shared-pool activation.
    ///
    /// A missing participant is ordinary while the child starts. More than
    /// one live generation is a lifecycle violation and fails closed rather
    /// than allowing an ambiguous generation to become routable.
    pub(crate) fn managed_participant_is_active(
        &self,
        participant_base: &str,
    ) -> Result<bool, String> {
        let state = self.state.lock();
        let mut matches = state.participants.iter().filter(|(participant_id, _)| {
            participant_matches_managed_base(participant_id, participant_base)
        });
        let Some((_, participant)) = matches.next() else {
            return Ok(false);
        };
        if matches.next().is_some() {
            return Err(format!(
                "multiple live KV participants match managed base '{participant_base}'"
            ));
        }
        let activation = participant.shared_activation.as_ref().ok_or_else(|| {
            format!(
                "managed KV participant base '{participant_base}' registered without shared-pool activation state"
            )
        })?;
        Ok(activation.active)
    }

    /// Return a bounded per-device view of physical, logical, idle, and
    /// quarantined capacity for one supervised managed-vLLM namespace.
    /// Metrics are sampled after cloning the pool reference, so collection
    /// never holds coordinator state while waiting on allocator state.
    pub(crate) fn managed_vllm_kv_snapshot(
        &self,
        participant_base: &str,
    ) -> Result<Vec<ManagedVllmKvDeviceSnapshot>, String> {
        let (pools, active, active_sequences) = {
            let state = self.state.lock();
            let mut matches = state.participants.iter().filter(|(participant_id, _)| {
                participant_matches_managed_base(participant_id, participant_base)
            });
            let Some((participant_id, participant)) = matches.next() else {
                return Ok(Vec::new());
            };
            if matches.next().is_some() {
                return Err(format!(
                    "multiple live KV participants match managed base '{participant_base}'"
                ));
            }
            let pools = participant.shared_pools.clone().ok_or_else(|| {
                format!(
                    "managed KV participant base '{participant_base}' has no shared-pool backing"
                )
            })?;
            let active = participant
                .shared_activation
                .as_ref()
                .is_some_and(|activation| activation.active);
            let active_sequences = state
                .sequences
                .keys()
                .filter(|(candidate, _)| candidate == participant_id.as_str())
                .count() as u64;
            (pools, active, active_sequences)
        };
        Ok(pools.managed_device_snapshots(active_sequences, active))
    }

    pub(crate) fn managed_vllm_resize_snapshot(
        &self,
        participant_base: &str,
    ) -> Result<Option<ManagedVllmKvResizeSnapshot>, String> {
        let (pools, pending_generation, pending_target, failure) = {
            let state = self.state.lock();
            let Some(participant_id) = managed_participant_id_locked(&state, participant_base)?
            else {
                return Ok(None);
            };
            let participant = state
                .participants
                .get(&participant_id)
                .expect("managed participant ID came from coordinator state");
            if !participant
                .registration
                .capabilities
                .features
                .contains(&KvFeature::LivePoolResize)
            {
                return Ok(None);
            }
            let pools = participant.shared_pools.clone().ok_or_else(|| {
                format!("managed KV participant '{participant_id}' has no shared-pool backing")
            })?;
            let (pending_generation, pending_target, failure) = participant
                .resize
                .as_ref()
                .map(|resize| {
                    (
                        Some(resize.generation),
                        Some(resize.target_block_count),
                        resize.failure.clone(),
                    )
                })
                .unwrap_or((None, None, None));
            (pools, pending_generation, pending_target, failure)
        };
        let (_pool_id, _bindings, minimum, current, maximum, alignment) =
            pools.elastic_shape().map_err(|error| error.to_string())?;
        Ok(Some(ManagedVllmKvResizeSnapshot {
            minimum_block_count: minimum,
            current_block_count: current,
            maximum_block_count: maximum,
            resize_alignment_blocks: alignment,
            pending_generation,
            pending_target_block_count: pending_target,
            failure,
        }))
    }

    /// Begin one ordered live CUDA VMM resize. Growth precharges authority and
    /// maps/zeros physical pages before workers are notified. Shrink first
    /// proves that the complete tail is logically free; physical release is
    /// deferred until scheduler and worker acknowledgements complete.
    pub(crate) fn request_managed_vllm_resize(
        &self,
        participant_base: &str,
        target_block_count: u64,
    ) -> Result<u64, String> {
        let mut state = self.state.lock();
        let participant_id =
            managed_participant_id_locked(&state, participant_base)?.ok_or_else(|| {
                format!("managed KV participant '{participant_base}' is not registered")
            })?;
        let (pools, current, maximum, existing) = {
            let participant = state
                .participants
                .get(&participant_id)
                .expect("managed participant ID came from coordinator state");
            if !participant
                .registration
                .capabilities
                .features
                .contains(&KvFeature::LivePoolResize)
            {
                return Err(format!(
                    "managed KV participant '{participant_id}' did not certify live pool resizing"
                ));
            }
            if !participant
                .shared_activation
                .as_ref()
                .is_some_and(|activation| activation.active)
            {
                return Err(format!(
                    "managed KV participant '{participant_id}' is not active"
                ));
            }
            let pools = participant.shared_pools.clone().ok_or_else(|| {
                format!("managed KV participant '{participant_id}' has no shared-pool backing")
            })?;
            let (_pool_id, _bindings, _minimum, current, maximum, _alignment) =
                pools.elastic_shape().map_err(|error| error.to_string())?;
            (
                pools,
                current,
                maximum,
                participant
                    .resize
                    .as_ref()
                    .map(|resize| (resize.generation, resize.target_block_count)),
            )
        };
        if let Some((generation, existing_target)) = existing {
            return if existing_target == target_block_count {
                Ok(generation)
            } else {
                Err(format!(
                    "managed KV participant '{participant_id}' already has resize generation {generation} targeting {existing_target} blocks"
                ))
            };
        }
        let target_block_count = if target_block_count < current {
            pools
                .normalize_shrink_target(target_block_count)
                .map_err(|error| format!("managed KV resize was not admitted: {error}"))?
        } else {
            target_block_count
        };
        if target_block_count == current || target_block_count > maximum {
            return Err(format!(
                "managed KV resize target {target_block_count} must differ from mapped capacity {current} and not exceed maximum {maximum}"
            ));
        }
        let generation = self.next_resize_generation.fetch_add(1, Ordering::Relaxed);
        if generation == 0 {
            return Err("managed KV resize generation space exhausted".to_string());
        }
        let prepared = if target_block_count > current {
            pools.prepare_grow(target_block_count, generation)
        } else {
            pools.prepare_shrink(target_block_count, generation)
        };
        let physical_operations = match prepared {
            Ok(operations) => operations,
            Err(error) => {
                if target_block_count < current
                    || matches!(error, KvContractError::CapacityExhausted { .. })
                {
                    return Err(format!("managed KV resize was not admitted: {error}"));
                }
                Self::fence_managed_readiness_locked(&mut state, &participant_id);
                if let Some(activation) = state
                    .participants
                    .get_mut(&participant_id)
                    .and_then(|participant| participant.shared_activation.as_mut())
                {
                    activation.active = false;
                }
                return Err(format!(
                    "failed to prepare managed KV resize; participant was fenced: {error}"
                ));
            }
        };
        let stage = if target_block_count > current {
            KvPoolResizeStage::MapWorkers
        } else {
            KvPoolResizeStage::RetireScheduler
        };
        state
            .participants
            .get_mut(&participant_id)
            .expect("managed participant remained registered under coordinator lock")
            .resize = Some(SharedPoolResizeRecord {
            generation,
            from_block_count: current,
            target_block_count,
            stage,
            physical_operations,
            worker_acks: BTreeSet::new(),
            scheduler_acks: BTreeSet::new(),
            deadline: Instant::now() + RESIZE_ACK_TIMEOUT,
            failure: None,
        });
        log::info!(
            "[kv-control] started live KV resize for participant '{}' generation={} blocks={}->{} stage={:?}",
            participant_id,
            generation,
            current,
            target_block_count,
            stage,
        );
        Ok(generation)
    }

    pub(crate) fn managed_vllm_live_resize_alignment_blocks(
        &self,
        memory_domains: &BTreeSet<KvMemoryDomain>,
        bytes_per_block: u64,
    ) -> Result<u64, String> {
        self.shared_pool_provisioner
            .as_ref()
            .ok_or_else(|| "no shared-pool provisioner is installed".to_string())?
            .live_resize_alignment_blocks(memory_domains, bytes_per_block)
            .map_err(|error| error.to_string())
    }

    fn retire_one_participant_after_backend_exit(
        &self,
        participant_id: &str,
    ) -> Result<bool, String> {
        let (pools, participant_epoch, mut leases) = {
            let mut state = self.state.lock();
            let Some(participant) = state.participants.get(participant_id) else {
                return Ok(false);
            };
            let pools = participant.shared_pools.clone();
            let participant_epoch = participant.receipt.participant_epoch;
            Self::fence_managed_readiness_locked(&mut state, participant_id);
            if let Some(activation) = state
                .participants
                .get_mut(participant_id)
                .and_then(|participant| participant.shared_activation.as_mut())
            {
                activation.active = false;
            }
            let lease_ids = state
                .leases
                .iter()
                .filter(|(_, lease)| lease.participant_id == participant_id)
                .map(|(lease_id, _)| lease_id.clone())
                .collect::<Vec<_>>();
            let leases = lease_ids
                .into_iter()
                .filter_map(|lease_id| state.leases.remove(&lease_id))
                .collect::<Vec<_>>();
            state
                .sequences
                .retain(|(candidate, _), _| candidate != participant_id);
            (pools, participant_epoch, leases)
        };

        for lease in &mut leases {
            if let (Some(pools), Some(allocation)) =
                (lease.shared_pools.as_ref(), lease.shared_allocation.take())
            {
                pools.release(allocation);
            }
        }
        let lease_count = leases.len();
        drop(leases);
        if let Some(pools) = pools {
            pools.release_backing_after_fence().map_err(|error| {
                format!(
                    "participant '{participant_id}' backing release failed; authority charge is quarantined: {error}"
                )
            })?;
        }
        let participant = {
            let mut state = self.state.lock();
            let current_epoch = state
                .participants
                .get(participant_id)
                .map(|participant| participant.receipt.participant_epoch);
            if current_epoch != Some(participant_epoch) {
                return Err(format!(
                    "participant '{participant_id}' generation changed during supervised retirement"
                ));
            }
            state.participants.remove(participant_id)
        };
        drop(participant);
        log::info!(
            "[kv-control] retired participant '{}' after supervised backend exit (released_leases={})",
            participant_id,
            lease_count,
        );
        Ok(true)
    }

    #[cfg(test)]
    fn participant_count(&self) -> usize {
        self.state.lock().participants.len()
    }

    #[cfg(test)]
    fn provisional_grant_count(&self) -> usize {
        self.state.lock().provisional_grants.len()
    }

    #[cfg(test)]
    fn lease_count(&self) -> usize {
        self.state.lock().leases.len()
    }

    fn participant_registration(
        &self,
        participant_id: &str,
    ) -> Result<ParticipantSnapshot, KvContractError> {
        self.state
            .lock()
            .participants
            .get(participant_id)
            .map(|record| ParticipantSnapshot {
                registration: record.registration.clone(),
                owner: record.owner,
                shared_pools: record.shared_pools.clone(),
                active: record
                    .shared_activation
                    .as_ref()
                    .is_none_or(|activation| activation.active),
            })
            .ok_or_else(|| KvContractError::NotFound {
                message: format!("KV participant '{participant_id}' is not registered"),
            })
    }

    #[cfg(unix)]
    fn response_vmm_handles(
        &self,
        request: &KvControlRequest,
        response: &KvControlResponse,
    ) -> Result<Vec<OwnedFd>, KvContractError> {
        let (participant_id, segments) = match (request, response) {
            (
                KvControlRequest::Register { registration },
                KvControlResponse::Registered { receipt },
            ) if registration.participant_id == receipt.participant_id => {
                let segments = receipt
                    .shared_pools
                    .iter()
                    .filter_map(|binding| binding.elastic.as_ref())
                    .flat_map(|elastic| elastic.segments.iter().cloned())
                    .collect::<Vec<_>>();
                (registration.participant_id.as_str(), segments)
            }
            (
                KvControlRequest::ResizePoll { participant_id, .. },
                KvControlResponse::Resize { operations, .. },
            ) => {
                let segments = operations
                    .iter()
                    .filter(|operation| operation.stage == KvPoolResizeStage::MapWorkers)
                    .flat_map(|operation| operation.segments.iter().cloned())
                    .collect::<Vec<_>>();
                (participant_id.as_str(), segments)
            }
            _ => return Ok(Vec::new()),
        };
        if segments.is_empty() {
            return Ok(Vec::new());
        }
        let pools = self
            .state
            .lock()
            .participants
            .get(participant_id)
            .and_then(|participant| participant.shared_pools.clone())
            .ok_or_else(|| KvContractError::Internal {
                message: format!(
                    "successful CUDA VMM response for '{participant_id}' has no live backing"
                ),
            })?;
        pools.backing.export_vmm_segments(&segments)
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
        if is_shared && self.allowed_shared_pool_profiles.is_empty() {
            return Err(KvContractError::invalid_capabilities(
                "shared_pool requires at least one deployment-allowlisted conformance profile",
            ));
        }
        if is_shared {
            if !registration
                .capabilities
                .features
                .contains(&KvFeature::ExternalPoolAttachment)
            {
                return Err(KvContractError::invalid_capabilities(
                    "external shared_pool requires the external_pool_attachment feature",
                ));
            }
            let profile = registration.adapter_profile.as_ref().ok_or_else(|| {
                KvContractError::invalid_capabilities(
                    "external shared_pool requires an adapter profile before provisioning",
                )
            })?;
            if !self.allowed_shared_pool_profiles.contains(profile) {
                return Err(KvContractError::invalid_capabilities(format!(
                    "shared-pool adapter profile '{}:{}:{}:{}' is not allowlisted by this deployment",
                    profile.adapter_id,
                    profile.adapter_version,
                    profile.backend_version,
                    profile.profile_id,
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
                    shared_activation: None,
                    resize: None,
                },
            );
            return Ok(receipt);
        }

        let (owner, precharged, minimum_block_count) =
            if let Some(proof) = registration.provisioning_grant.as_ref() {
                let record = state.provisional_grants.get(&proof.token).ok_or_else(|| {
                    KvContractError::invalid_request(
                        "provisioning grant is unknown, expired, or already consumed",
                    )
                })?;
                record.validate_registration(registration, proof, Instant::now())?;
                let mut record = state
                    .provisional_grants
                    .remove(&proof.token)
                    .expect("validated provisioning grant remained under coordinator state");
                let lease = record
                    .lease
                    .take()
                    .ok_or_else(|| KvContractError::Internal {
                        message: "validated provisioning grant has no authority lease".to_string(),
                    })?;
                (record.owner, Some(lease), Some(record.minimum_block_count))
            } else {
                let slot = self.next_participant_slot.fetch_add(1, Ordering::Relaxed);
                let owner =
                    MemoryOwner::external_kv(slot).ok_or_else(|| KvContractError::Internal {
                        message: "external KV participant owner space exhausted".to_string(),
                    })?;
                (owner, None, None)
            };
        let participant_epoch = self.next_participant_epoch.fetch_add(1, Ordering::Relaxed);
        let (receipt, shared_pools) = if is_shared {
            let provisioner = self.shared_pool_provisioner.as_ref().ok_or_else(|| {
                KvContractError::invalid_capabilities(
                    "shared_pool was requested but no isolated data-plane provisioner is configured",
                )
            })?;
            let provisioned = provisioner.provision(
                registration,
                owner,
                participant_epoch,
                precharged,
                minimum_block_count,
            )?;
            let receipt = KvRegistrationReceipt {
                participant_id: registration.participant_id.clone(),
                participant_epoch,
                shared_pools: provisioned.descriptors.clone(),
            };
            let pools = SharedPoolSet::new(registration, &receipt, owner, provisioned)?;
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
                shared_activation: is_shared.then(|| SharedPoolActivation {
                    attachments: BTreeMap::new(),
                    active: false,
                }),
                resize: None,
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

    fn attach(
        &self,
        participant_id: &str,
        attachment: &KvSharedPoolAttachment,
    ) -> Result<(), KvContractError> {
        attachment.validate()?;
        let mut state = self.state.lock();
        let record =
            state
                .participants
                .get(participant_id)
                .ok_or_else(|| KvContractError::NotFound {
                    message: format!("KV participant '{participant_id}' is not registered"),
                })?;
        if !self
            .allowed_shared_pool_profiles
            .contains(&attachment.profile)
        {
            return Err(KvContractError::invalid_capabilities(format!(
                "shared-pool adapter profile '{}:{}:{}:{}' is not allowlisted by this deployment",
                attachment.profile.adapter_id,
                attachment.profile.adapter_version,
                attachment.profile.backend_version,
                attachment.profile.profile_id,
            )));
        }
        validate_shared_attachment(record, attachment)?;
        let activation = record.shared_activation.as_ref().ok_or_else(|| {
            KvContractError::invalid_request(
                "opaque participants cannot attach shared-pool bindings",
            )
        })?;
        if let Some(existing) = activation.attachments.get(&attachment.binding_id) {
            if existing == attachment {
                return Ok(());
            }
            return Err(KvContractError::invalid_request(format!(
                "binding '{}' is already attached with different evidence",
                attachment.binding_id
            )));
        }
        if activation.active {
            return Err(KvContractError::invalid_request(
                "an active participant cannot add or replace attachments",
            ));
        }

        state
            .participants
            .get_mut(participant_id)
            .and_then(|record| record.shared_activation.as_mut())
            .expect("shared activation checked above")
            .attachments
            .insert(attachment.binding_id.clone(), attachment.clone());
        log::info!(
            "[kv-control] attached binding '{}' for participant '{}' rank={}/{} epoch={}",
            attachment.binding_id,
            participant_id,
            attachment.shard.tensor_parallel_rank,
            attachment.shard.tensor_parallel_world_size,
            attachment.participant_epoch,
        );
        Ok(())
    }

    fn activate(
        &self,
        participant_id: &str,
        participant_epoch: u64,
    ) -> Result<(), KvContractError> {
        let mut state = self.state.lock();
        let record = state.participants.get_mut(participant_id).ok_or_else(|| {
            KvContractError::NotFound {
                message: format!("KV participant '{participant_id}' is not registered"),
            }
        })?;
        if record.receipt.participant_epoch != participant_epoch {
            return Err(KvContractError::invalid_request(
                "activation epoch does not match the registration receipt",
            ));
        }
        let activation = record.shared_activation.as_mut().ok_or_else(|| {
            KvContractError::invalid_request(
                "opaque participants do not require shared-pool activation",
            )
        })?;
        if activation.active {
            return Ok(());
        }
        let expected_bindings = record
            .receipt
            .shared_pools
            .iter()
            .map(|binding| binding.binding_id.as_str())
            .collect::<BTreeSet<_>>();
        let attached_bindings = activation
            .attachments
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if attached_bindings != expected_bindings {
            let missing = expected_bindings
                .difference(&attached_bindings)
                .copied()
                .collect::<Vec<_>>()
                .join(", ");
            return Err(KvContractError::invalid_request(format!(
                "shared-pool participant '{participant_id}' cannot activate before every binding attaches; missing: {missing}"
            )));
        }

        activation.active = true;
        log::info!(
            "[kv-control] activated shared-pool participant '{}' epoch={} bindings={}",
            participant_id,
            participant_epoch,
            activation.attachments.len(),
        );
        Ok(())
    }

    fn reserve(
        &self,
        participant_id: &str,
        request: &KvReserveRequest,
    ) -> Result<KvLease, KvContractError> {
        request.validate()?;
        self.expire_stale();

        let ParticipantSnapshot {
            registration,
            owner,
            shared_pools,
            active,
        } = self.participant_registration(participant_id)?;
        if shared_pools.is_some() && !active {
            return Err(KvContractError::invalid_request(format!(
                "shared-pool participant '{participant_id}' is provisioned but not active"
            )));
        }
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
            .is_none_or(|current| {
                current.registration != registration
                    || (shared_pools.is_some()
                        && current
                            .shared_activation
                            .as_ref()
                            .is_none_or(|activation| !activation.active))
            })
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

    fn poll_resize(
        &self,
        participant_id: &str,
        request: &KvResizePollRequest,
    ) -> Result<KvResizePollResult, KvContractError> {
        request.validate()?;
        let state = self.state.lock();
        let participant =
            state
                .participants
                .get(participant_id)
                .ok_or_else(|| KvContractError::NotFound {
                    message: format!("KV participant '{participant_id}' is not registered"),
                })?;
        if participant.receipt.participant_epoch != request.participant_epoch {
            return Err(KvContractError::invalid_request(
                "resize poll epoch does not match the registration receipt",
            ));
        }
        if !participant
            .registration
            .capabilities
            .features
            .contains(&KvFeature::LivePoolResize)
        {
            return Err(KvContractError::unsupported(
                "poll_fixed_shared_pool_resize",
            ));
        }
        let Some(resize) = participant.resize.as_ref() else {
            return Ok(KvResizePollResult {
                pending: false,
                operations: Vec::new(),
            });
        };
        if let Some(failure) = resize.failure.as_ref() {
            return Err(KvContractError::invalid_request(format!(
                "live KV resize generation {} is fenced: {failure}",
                resize.generation
            )));
        }
        if request.applied_generation >= resize.generation {
            return Ok(KvResizePollResult {
                pending: true,
                operations: Vec::new(),
            });
        }
        let mut operations = match request.actor {
            KvResizeActor::Scheduler
                if matches!(
                    resize.stage,
                    KvPoolResizeStage::ActivateScheduler | KvPoolResizeStage::RetireScheduler
                ) =>
            {
                SharedPoolSet::scheduler_operations(&resize.physical_operations, resize.stage)
                    .into_iter()
                    .filter(|operation| !resize.scheduler_acks.contains(&operation.binding_id))
                    .collect()
            }
            KvResizeActor::Worker { shard }
                if matches!(
                    resize.stage,
                    KvPoolResizeStage::MapWorkers | KvPoolResizeStage::UnmapWorkers
                ) =>
            {
                let activation = participant.shared_activation.as_ref().ok_or_else(|| {
                    KvContractError::Internal {
                        message: "elastic participant has no activation evidence".to_string(),
                    }
                })?;
                let matching = resize
                    .physical_operations
                    .iter()
                    .filter(|operation| {
                        activation
                            .attachments
                            .get(&operation.binding_id)
                            .is_some_and(|attachment| attachment.shard == shard)
                    })
                    .count();
                if matching == 0 {
                    return Err(KvContractError::invalid_request(
                        "resize worker shard has no attached elastic binding",
                    ));
                }
                SharedPoolSet::worker_operations(&resize.physical_operations, resize.stage)
                    .into_iter()
                    .filter(|operation| {
                        activation
                            .attachments
                            .get(&operation.binding_id)
                            .is_some_and(|attachment| attachment.shard == shard)
                            && !resize.worker_acks.contains(&operation.binding_id)
                    })
                    .collect()
            }
            _ => Vec::new(),
        };
        if matches!(request.actor, KvResizeActor::Worker { .. }) {
            densify_worker_resize_handle_indices(&mut operations)?;
        }
        let result = KvResizePollResult {
            pending: true,
            operations,
        };
        result.validate()?;
        Ok(result)
    }

    fn ack_resize(
        &self,
        participant_id: &str,
        request: &KvResizeAckRequest,
    ) -> Result<(), KvContractError> {
        request.validate()?;
        let mut state = self.state.lock();
        let commit = {
            let participant = state.participants.get_mut(participant_id).ok_or_else(|| {
                KvContractError::NotFound {
                    message: format!("KV participant '{participant_id}' is not registered"),
                }
            })?;
            if participant.receipt.participant_epoch != request.participant_epoch {
                return Err(KvContractError::invalid_request(
                    "resize acknowledgement epoch does not match the registration receipt",
                ));
            }
            let resize = participant.resize.as_mut().ok_or_else(|| {
                KvContractError::invalid_request(
                    "participant has no pending live KV resize generation",
                )
            })?;
            if let Some(failure) = resize.failure.as_ref() {
                return Err(KvContractError::invalid_request(format!(
                    "live KV resize generation {} is fenced: {failure}",
                    resize.generation
                )));
            }
            if request.resize_generation != resize.generation
                || request.stage != resize.stage
                || request.applied_block_count != resize.target_block_count
                || !resize
                    .physical_operations
                    .iter()
                    .any(|operation| operation.binding_id == request.binding_id)
            {
                return Err(KvContractError::invalid_request(
                    "resize acknowledgement does not exactly match the pending operation",
                ));
            }

            let expected_bindings = resize
                .physical_operations
                .iter()
                .map(|operation| operation.binding_id.clone())
                .collect::<BTreeSet<_>>();
            let stage_complete = match request.actor {
                KvResizeActor::Scheduler
                    if matches!(
                        resize.stage,
                        KvPoolResizeStage::ActivateScheduler | KvPoolResizeStage::RetireScheduler
                    ) =>
                {
                    resize.scheduler_acks.insert(request.binding_id.clone());
                    resize.scheduler_acks == expected_bindings
                }
                KvResizeActor::Worker { shard }
                    if matches!(
                        resize.stage,
                        KvPoolResizeStage::MapWorkers | KvPoolResizeStage::UnmapWorkers
                    ) =>
                {
                    let attachment = participant
                        .shared_activation
                        .as_ref()
                        .and_then(|activation| activation.attachments.get(&request.binding_id))
                        .ok_or_else(|| {
                            KvContractError::invalid_request(
                                "resize acknowledgement binding is not attached",
                            )
                        })?;
                    if attachment.shard != shard {
                        return Err(KvContractError::invalid_request(
                            "resize acknowledgement worker shard does not own the binding",
                        ));
                    }
                    resize.worker_acks.insert(request.binding_id.clone());
                    resize.worker_acks == expected_bindings
                }
                _ => {
                    return Err(KvContractError::invalid_request(
                        "resize actor cannot acknowledge the pending stage",
                    ));
                }
            };
            if !stage_complete {
                return Ok(());
            }

            match resize.stage {
                KvPoolResizeStage::MapWorkers => {
                    resize.stage = KvPoolResizeStage::ActivateScheduler;
                    resize.scheduler_acks.clear();
                    resize.deadline = Instant::now() + RESIZE_ACK_TIMEOUT;
                    log::info!(
                        "[kv-control] live KV resize participant='{}' generation={} advanced to activate_scheduler",
                        participant_id,
                        resize.generation,
                    );
                    return Ok(());
                }
                KvPoolResizeStage::RetireScheduler => {
                    resize.stage = KvPoolResizeStage::UnmapWorkers;
                    resize.worker_acks.clear();
                    resize.deadline = Instant::now() + RESIZE_ACK_TIMEOUT;
                    log::info!(
                        "[kv-control] live KV resize participant='{}' generation={} advanced to unmap_workers",
                        participant_id,
                        resize.generation,
                    );
                    return Ok(());
                }
                KvPoolResizeStage::ActivateScheduler | KvPoolResizeStage::UnmapWorkers => Some((
                    participant
                        .shared_pools
                        .clone()
                        .expect("elastic participant has shared-pool backing"),
                    resize.generation,
                    resize.from_block_count,
                    resize.target_block_count,
                    resize.stage == KvPoolResizeStage::ActivateScheduler,
                )),
            }
        };

        let Some((pools, generation, from, target, growing)) = commit else {
            return Ok(());
        };
        let committed = if growing {
            pools.commit_grow(from, target)
        } else {
            pools.commit_shrink(from, target)
        };
        match committed {
            Ok(()) => {
                state
                    .participants
                    .get_mut(participant_id)
                    .expect("resize participant remained registered under coordinator lock")
                    .resize = None;
                log::info!(
                    "[kv-control] completed live KV resize for participant '{}' generation={} blocks={}->{}",
                    participant_id,
                    generation,
                    from,
                    target,
                );
                Ok(())
            }
            Err(error) => {
                Self::fence_managed_readiness_locked(&mut state, participant_id);
                let participant = state
                    .participants
                    .get_mut(participant_id)
                    .expect("resize participant remained registered under coordinator lock");
                if let Some(activation) = participant.shared_activation.as_mut() {
                    activation.active = false;
                }
                if let Some(resize) = participant.resize.as_mut() {
                    resize.failure = Some(error.to_string());
                }
                Err(KvContractError::Internal {
                    message: format!(
                        "live KV resize commit failed; participant was fenced and capacity retained: {error}"
                    ),
                })
            }
        }
    }

    fn detach(
        &self,
        participant_id: &str,
        request: &KvSharedPoolDetachRequest,
    ) -> Result<(), KvContractError> {
        request.validate()?;
        if !matches!(request.completion, KvReleaseCompletion::BackendSynchronized) {
            return Err(KvContractError::unsupported(
                "shared_pool_detach_transport_fence",
            ));
        }
        let mut state = self.state.lock();
        if state
            .leases
            .values()
            .any(|lease| lease.participant_id == participant_id)
        {
            return Err(KvContractError::invalid_request(format!(
                "shared-pool participant '{participant_id}' cannot detach while leases are active"
            )));
        }
        {
            let record = state.participants.get(participant_id).ok_or_else(|| {
                KvContractError::NotFound {
                    message: format!("KV participant '{participant_id}' is not registered"),
                }
            })?;
            if record.receipt.participant_epoch != request.participant_epoch {
                return Err(KvContractError::invalid_request(
                    "detach epoch does not match the registration receipt",
                ));
            }
            let activation = record.shared_activation.as_ref().ok_or_else(|| {
                KvContractError::invalid_request(
                    "opaque participants cannot detach shared-pool bindings",
                )
            })?;
            for binding_id in &request.binding_ids {
                let attachment = activation.attachments.get(binding_id).ok_or_else(|| {
                    KvContractError::invalid_request(format!(
                        "binding '{binding_id}' is not attached"
                    ))
                })?;
                if attachment.shard != request.shard {
                    return Err(KvContractError::invalid_request(format!(
                        "binding '{binding_id}' was attached by a different shard"
                    )));
                }
            }
        }

        // All caller-controlled detach fields have now been validated. Fence
        // stale managed-process health publications before changing active
        // state, removing attachments, or dropping the participant backing.
        // The fence operation is atomic-only and never enters process
        // lifecycle locking while coordinator state is held.
        Self::fence_managed_readiness_locked(&mut state, participant_id);
        let release = {
            let record = state
                .participants
                .get_mut(participant_id)
                .expect("detach participant was validated while coordinator state was held");
            let activation = record
                .shared_activation
                .as_mut()
                .expect("shared activation was validated while coordinator state was held");
            activation.active = false;
            for binding_id in &request.binding_ids {
                activation.attachments.remove(binding_id);
            }
            activation
                .attachments
                .is_empty()
                .then(|| record.shared_pools.clone())
                .flatten()
        };
        drop(state);
        let should_remove = release.is_some();
        if let Some(pools) = release.as_ref() {
            pools.release_backing_after_fence().map_err(|error| {
                KvContractError::Internal {
                    message: format!(
                        "shared-pool backing release failed; participant remains quarantined: {error}"
                    ),
                }
            })?;
        }
        let removed = if should_remove {
            let mut state = self.state.lock();
            let removable = state
                .participants
                .get(participant_id)
                .is_some_and(|record| {
                    record.receipt.participant_epoch == request.participant_epoch
                        && record
                            .shared_activation
                            .as_ref()
                            .is_some_and(|activation| activation.attachments.is_empty())
                });
            if !removable {
                return Err(KvContractError::invalid_request(
                    "shared-pool participant changed while its backing was released",
                ));
            }
            state.participants.remove(participant_id)
        } else {
            None
        };
        drop(removed);
        log::info!(
            "[kv-control] detached {} shared-pool binding(s) for participant '{}' epoch={}",
            request.binding_ids.len(),
            participant_id,
            request.participant_epoch,
        );
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
    use std::mem;
    use std::os::fd::{AsRawFd, RawFd};
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
        let (response, handles) = match tokio::time::timeout(
            Duration::from_secs(5),
            read_frame(&mut stream, max_frame_bytes),
        )
        .await
        {
            Ok(Ok(frame)) => match serde_json::from_slice::<KvControlRequestEnvelope>(&frame) {
                Ok(envelope) => {
                    let response = dispatch_control_request(coordinator.as_ref(), envelope.clone());
                    match coordinator.response_vmm_handles(&envelope.request, &response.response) {
                        Ok(handles) => (response, handles),
                        Err(error) => (error_response(envelope.request_id, error), Vec::new()),
                    }
                }
                Err(error) => (
                    error_response(
                        request_id_from_invalid_frame(&frame),
                        KvContractError::invalid_request(format!(
                            "invalid KV control envelope: {error}"
                        )),
                    ),
                    Vec::new(),
                ),
            },
            Ok(Err(error)) => (
                error_response(
                    String::new(),
                    KvContractError::Transport {
                        message: error.to_string(),
                    },
                ),
                Vec::new(),
            ),
            Err(_) => (
                error_response(
                    String::new(),
                    KvContractError::Transport {
                        message: "timed out reading KV control frame".to_string(),
                    },
                ),
                Vec::new(),
            ),
        };
        let mut encoded = serde_json::to_vec(&response).map_err(io::Error::other)?;
        encoded.push(b'\n');
        write_response_with_handles(&mut stream, &encoded, &handles).await?;
        match stream.shutdown().await {
            Err(error) if error.kind() == io::ErrorKind::NotConnected => Ok(()),
            result => result,
        }
    }

    async fn write_response_with_handles(
        stream: &mut UnixStream,
        encoded: &[u8],
        handles: &[OwnedFd],
    ) -> io::Result<()> {
        if handles.is_empty() {
            return stream.write_all(encoded).await;
        }
        let descriptors = handles.iter().map(AsRawFd::as_raw_fd).collect::<Vec<_>>();
        let descriptor_bytes = descriptors
            .len()
            .checked_mul(mem::size_of::<RawFd>())
            .ok_or_else(|| io::Error::other("CUDA VMM descriptor array overflowed"))?;
        let control_bytes = unsafe { libc::CMSG_SPACE(descriptor_bytes as _) as usize };
        let control_words = control_bytes.div_ceil(mem::size_of::<usize>());
        let mut control = vec![0usize; control_words];

        let sent = loop {
            stream.writable().await?;
            let mut iovec = libc::iovec {
                iov_base: encoded.as_ptr().cast_mut().cast(),
                iov_len: encoded.len(),
            };
            let mut message = unsafe { mem::zeroed::<libc::msghdr>() };
            message.msg_iov = &mut iovec;
            message.msg_iovlen = 1;
            message.msg_control = control.as_mut_ptr().cast();
            message.msg_controllen = control_bytes as _;
            unsafe {
                let header = libc::CMSG_FIRSTHDR(&message);
                if header.is_null() {
                    return Err(io::Error::other(
                        "failed to construct CUDA VMM descriptor control message",
                    ));
                }
                (*header).cmsg_level = libc::SOL_SOCKET;
                (*header).cmsg_type = libc::SCM_RIGHTS;
                (*header).cmsg_len = libc::CMSG_LEN(descriptor_bytes as _) as _;
                std::ptr::copy_nonoverlapping(
                    descriptors.as_ptr().cast::<u8>(),
                    libc::CMSG_DATA(header),
                    descriptor_bytes,
                );
            }
            let sent = unsafe { libc::sendmsg(stream.as_raw_fd(), &message, libc::MSG_NOSIGNAL) };
            if sent >= 0 {
                break sent as usize;
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::WouldBlock {
                return Err(error);
            }
        };
        if sent == 0 || sent > encoded.len() {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "failed to write CUDA VMM descriptor response",
            ));
        }
        if sent < encoded.len() {
            stream.write_all(&encoded[sent..]).await?;
        }
        Ok(())
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

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::os::fd::FromRawFd;

        #[tokio::test]
        async fn response_writer_transfers_every_descriptor_with_scm_rights() {
            let (mut sender, receiver) = UnixStream::pair().unwrap();
            let handles = vec![
                OwnedFd::from(std::fs::File::open("/dev/null").unwrap()),
                OwnedFd::from(std::fs::File::open("/dev/null").unwrap()),
            ];
            let encoded = b"{\"result\":\"registered\"}\n".to_vec();
            let expected = encoded.clone();
            let writer = tokio::spawn(async move {
                write_response_with_handles(&mut sender, &encoded, &handles).await
            });

            receiver.readable().await.unwrap();
            let mut payload = [0u8; 128];
            let mut iovec = libc::iovec {
                iov_base: payload.as_mut_ptr().cast(),
                iov_len: payload.len(),
            };
            let descriptor_bytes = 2 * mem::size_of::<RawFd>();
            let control_bytes = unsafe { libc::CMSG_SPACE(descriptor_bytes as _) as usize };
            let mut control = vec![0usize; control_bytes.div_ceil(mem::size_of::<usize>())];
            let mut message = unsafe { mem::zeroed::<libc::msghdr>() };
            message.msg_iov = &mut iovec;
            message.msg_iovlen = 1;
            message.msg_control = control.as_mut_ptr().cast();
            message.msg_controllen = control_bytes as _;
            let received = unsafe { libc::recvmsg(receiver.as_raw_fd(), &mut message, 0) };
            assert!(
                received > 0,
                "recvmsg failed: {}",
                io::Error::last_os_error()
            );
            assert_eq!(&payload[..received as usize], expected.as_slice());

            let header = unsafe { libc::CMSG_FIRSTHDR(&message) };
            assert!(!header.is_null());
            assert_eq!(unsafe { (*header).cmsg_level }, libc::SOL_SOCKET);
            assert_eq!(unsafe { (*header).cmsg_type }, libc::SCM_RIGHTS);
            let actual_descriptor_bytes =
                unsafe { (*header).cmsg_len as usize - libc::CMSG_LEN(0) as usize };
            assert_eq!(actual_descriptor_bytes, descriptor_bytes);
            let descriptors = unsafe {
                std::slice::from_raw_parts(
                    libc::CMSG_DATA(header).cast::<RawFd>(),
                    actual_descriptor_bytes / mem::size_of::<RawFd>(),
                )
            };
            assert_eq!(descriptors.len(), 2);
            for descriptor in descriptors {
                let owned = unsafe { OwnedFd::from_raw_fd(*descriptor) };
                assert!(unsafe { libc::fcntl(owned.as_raw_fd(), libc::F_GETFD) } >= 0);
            }
            writer.await.unwrap().unwrap();
        }
    }
}

#[cfg(unix)]
pub(crate) use unix::Server as KvControlServer;

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_hal::device::{Device, DeviceBackend, DeviceInfo};
    use kapsl_kv_abi::{
        KvAttachmentView, KvBackendCapabilities, KvCacheGeometry, KvCacheGroup, KvCachePolicy,
        KvCapacityGroup, KvCapacityModel, KvControlRequest, KvElasticPoolDescriptor, KvElementType,
        KvGroupReservation, KvLayerId, KvMemoryDomain, KvPoolResizeStage, KvResizeAckRequest,
        KvResizeActor, KvResizePollRequest, KvSequenceKey, KvShard, KvTensorLayout, KvTopology,
        KvTransport,
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

    #[derive(Default)]
    struct TestElasticBacking {
        mapped_blocks: AtomicU64,
        released_to_blocks: AtomicU64,
        release_attempts: AtomicU64,
        fail_grow: AtomicBool,
        fail_release: AtomicBool,
        fail_tail_release: AtomicBool,
        segments: Mutex<Vec<KvVmmSegmentDescriptor>>,
    }

    impl SharedPoolBacking for TestElasticBacking {
        fn zero_blocks(
            &self,
            _binding: &KvSharedPoolDescriptor,
            _block_indices: &[u64],
        ) -> Result<(), KvContractError> {
            Ok(())
        }

        fn release_after_fence(&self) -> Result<(), KvContractError> {
            self.release_attempts.fetch_add(1, Ordering::AcqRel);
            if self.fail_release.load(Ordering::Acquire) {
                return Err(KvContractError::Internal {
                    message: "injected backing release failure".to_string(),
                });
            }
            self.mapped_blocks.store(0, Ordering::Release);
            self.segments.lock().clear();
            Ok(())
        }

        fn grow_binding(
            &self,
            binding: &KvSharedPoolDescriptor,
            target_block_count: u64,
            resize_generation: u64,
        ) -> Result<Vec<KvVmmSegmentDescriptor>, KvContractError> {
            if self.fail_grow.load(Ordering::Acquire) {
                return Err(KvContractError::Internal {
                    message: "injected backing growth failure".to_string(),
                });
            }
            let current = self.mapped_blocks.load(Ordering::Acquire);
            if target_block_count <= current {
                return Err(KvContractError::invalid_request(
                    "test elastic backing cannot grow backwards",
                ));
            }
            let segment = KvVmmSegmentDescriptor {
                segment_id: format!("test-grow-{resize_generation}"),
                offset_bytes: current * binding.bytes_per_block,
                length_bytes: (target_block_count - current) * binding.bytes_per_block,
                handle_index: 0,
            };
            self.segments.lock().push(segment.clone());
            self.mapped_blocks
                .store(target_block_count, Ordering::Release);
            Ok(vec![segment])
        }

        fn shrink_target_boundary(
            &self,
            binding: &KvSharedPoolDescriptor,
            requested_block_count: u64,
        ) -> Result<u64, KvContractError> {
            self.segments
                .lock()
                .iter()
                .map(|segment| segment.offset_bytes / binding.bytes_per_block)
                .filter(|boundary| *boundary <= requested_block_count)
                .max()
                .ok_or_else(|| KvContractError::Internal {
                    message: "test elastic backing has no shrink boundary".to_string(),
                })
        }

        fn shrink_segments(
            &self,
            binding: &KvSharedPoolDescriptor,
            target_block_count: u64,
        ) -> Result<Vec<KvVmmSegmentDescriptor>, KvContractError> {
            let target_bytes = target_block_count * binding.bytes_per_block;
            let segments = self
                .segments
                .lock()
                .iter()
                .filter(|segment| segment.offset_bytes >= target_bytes)
                .cloned()
                .collect::<Vec<_>>();
            if segments.is_empty() {
                return Err(KvContractError::invalid_request(
                    "test shrink is not on a physical segment boundary",
                ));
            }
            Ok(segments)
        }

        fn release_binding_tail(
            &self,
            binding: &KvSharedPoolDescriptor,
            target_block_count: u64,
        ) -> Result<(), KvContractError> {
            if self.fail_tail_release.load(Ordering::Acquire) {
                return Err(KvContractError::Internal {
                    message: "injected backing tail-release failure".to_string(),
                });
            }
            let target_bytes = target_block_count * binding.bytes_per_block;
            self.segments
                .lock()
                .retain(|segment| segment.offset_bytes < target_bytes);
            self.mapped_blocks
                .store(target_block_count, Ordering::Release);
            self.released_to_blocks
                .store(target_block_count, Ordering::Release);
            Ok(())
        }

        #[cfg(unix)]
        fn export_vmm_segments(
            &self,
            segments: &[KvVmmSegmentDescriptor],
        ) -> Result<Vec<OwnedFd>, KvContractError> {
            segments
                .iter()
                .map(|_| {
                    std::fs::File::open("/dev/null")
                        .map(OwnedFd::from)
                        .map_err(|error| KvContractError::Internal {
                            message: format!("open test VMM descriptor: {error}"),
                        })
                })
                .collect()
        }
    }

    struct TestElasticProvisioner {
        backing: Arc<TestElasticBacking>,
        initial_block_count: u64,
    }

    impl SharedPoolProvisioner for TestElasticProvisioner {
        fn provision(
            &self,
            registration: &KvParticipantRegistration,
            _owner: MemoryOwner,
            participant_epoch: u64,
            precharged: Option<MemoryLease>,
            minimum_block_count: Option<u64>,
        ) -> Result<ProvisionedSharedPools, KvContractError> {
            let group = registration
                .capacity_model
                .groups
                .first()
                .expect("test elastic registration group");
            let minimum = minimum_block_count.ok_or_else(|| {
                KvContractError::invalid_capabilities("test elastic minimum is missing")
            })?;
            let maximum = group.max_allocations.expect("test elastic block cap");
            if minimum > self.initial_block_count || self.initial_block_count > maximum {
                return Err(KvContractError::invalid_capabilities(
                    "test elastic block capacities are unordered",
                ));
            }
            let stride = group.bytes_per_allocation;
            let mut segments = vec![KvVmmSegmentDescriptor {
                segment_id: "test-minimum".to_string(),
                offset_bytes: 0,
                length_bytes: minimum * stride,
                handle_index: 0,
            }];
            if self.initial_block_count > minimum {
                segments.push(KvVmmSegmentDescriptor {
                    segment_id: "test-initial-headroom".to_string(),
                    offset_bytes: minimum * stride,
                    length_bytes: (self.initial_block_count - minimum) * stride,
                    handle_index: 1,
                });
            }
            *self.backing.segments.lock() = segments.clone();
            self.backing
                .mapped_blocks
                .store(self.initial_block_count, Ordering::Release);
            Ok(ProvisionedSharedPools {
                descriptors: vec![KvSharedPoolDescriptor {
                    binding_id: format!("test-vmm-{participant_epoch}"),
                    capacity_pool_id: group.pool_id.clone(),
                    generation: participant_epoch,
                    group_ids: vec![group.group_id.clone()],
                    memory_domain: group.memory_domains[0].clone(),
                    block_count: maximum,
                    bytes_per_block: stride,
                    allocation_mode: KvSharedPoolAllocationMode::ParticipantManaged,
                    transport: KvTransport::CudaVmm,
                    descriptor: "scm_rights:test-vmm-v1".to_string(),
                    elastic: Some(KvElasticPoolDescriptor {
                        minimum_block_count: minimum,
                        mapped_block_count: self.initial_block_count,
                        maximum_block_count: maximum,
                        allocation_granularity_bytes: stride * 2,
                        resize_alignment_blocks: 2,
                        segments,
                    }),
                }],
                backing: self.backing.clone(),
                memory_lease: precharged,
            })
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
            precharged: Option<MemoryLease>,
            _minimum_block_count: Option<u64>,
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
                    memory_domain: group.memory_domains[0].clone(),
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
                    elastic: None,
                }],
                backing: self.backing.clone(),
                memory_lease: precharged,
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
            adapter_profile: None,
            topology: None,
            provisioning_grant: None,
        }
    }

    fn shared_registration() -> KvParticipantRegistration {
        let mut capabilities = KvBackendCapabilities::in_process_shared_pool();
        capabilities.transports.clear();
        capabilities.transports.insert(KvTransport::Custom {
            name: "test_direct".to_string(),
        });
        capabilities
            .features
            .insert(KvFeature::ExternalPoolAttachment);
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
            adapter_profile: Some(KvAdapterProfile {
                adapter_id: "kapsl-test-adapter".to_string(),
                adapter_version: "1.0.0".to_string(),
                backend_version: "test-backend-1".to_string(),
                profile_id: "test-direct-v1".to_string(),
            }),
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
            provisioning_grant: None,
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

    fn shared_attachment(receipt: &KvRegistrationReceipt) -> KvSharedPoolAttachment {
        let binding = receipt.shared_pools.first().expect("test shared binding");
        KvSharedPoolAttachment {
            participant_epoch: receipt.participant_epoch,
            binding_id: binding.binding_id.clone(),
            shard: KvShard::default(),
            profile: KvAdapterProfile {
                adapter_id: "kapsl-test-adapter".to_string(),
                adapter_version: "1.0.0".to_string(),
                backend_version: "test-backend-1".to_string(),
                profile_id: "test-direct-v1".to_string(),
            },
            imported_bytes: binding.block_count * binding.bytes_per_block,
            mapped_bytes: binding
                .elastic
                .as_ref()
                .map(|elastic| elastic.mapped_block_count * binding.bytes_per_block),
            views: vec![KvAttachmentView {
                group_id: "vllm.group.0".to_string(),
                layer: KvLayerId::indexed(0),
                offset_bytes: 0,
                length_bytes: binding.block_count * binding.bytes_per_block,
            }],
        }
    }

    fn allowed_test_profiles() -> BTreeSet<KvAdapterProfile> {
        BTreeSet::from([KvAdapterProfile {
            adapter_id: "kapsl-test-adapter".to_string(),
            adapter_version: "1.0.0".to_string(),
            backend_version: "test-backend-1".to_string(),
            profile_id: "test-direct-v1".to_string(),
        }])
    }

    fn provisional_request(candidates: Vec<ProvisionalKvCandidate>) -> ProvisionalKvGrantRequest {
        ProvisionalKvGrantRequest {
            participant_base: "vllm".to_string(),
            model_fingerprint: "sha256:test".to_string(),
            geometry_digest: format!("sha256:{}", "ab".repeat(32)),
            adapter_profile: allowed_test_profiles()
                .into_iter()
                .next()
                .expect("test profile"),
            capacity_pool_id: "vllm.pool.0".to_string(),
            group_ids: BTreeSet::from(["vllm.group.0".to_string()]),
            memory_domains: BTreeSet::from([KvMemoryDomain::Host]),
            candidates,
            maximum_block_count: None,
            ttl: Duration::from_secs(30),
        }
    }

    fn registration_with_grant(proof: KvProvisioningGrant) -> KvParticipantRegistration {
        let mut registration = shared_registration();
        registration.participant_id = "vllm:engine-1".to_string();
        registration.provisioning_grant = Some(proof);
        registration
            .capabilities
            .features
            .insert(KvFeature::ProvisioningGrant);
        registration
    }

    fn elastic_registration_with_grant(proof: KvProvisioningGrant) -> KvParticipantRegistration {
        let mut registration = registration_with_grant(proof);
        registration.capabilities = KvBackendCapabilities::cuda_vmm_shared_pool();
        registration
            .capabilities
            .features
            .insert(KvFeature::ParticipantBlockSelection);
        registration
            .capabilities
            .features
            .insert(KvFeature::ProvisioningGrant);
        registration.capacity_model.groups[0].max_allocations = Some(8);
        registration
    }

    fn elastic_coordinator() -> (
        Arc<MemoryAuthority>,
        Arc<ExternalKvCoordinator>,
        Arc<TestElasticBacking>,
        KvParticipantRegistration,
        KvRegistrationReceipt,
    ) {
        use kapsl_kv_abi::KvCoordinator as _;

        let memory = test_memory();
        let backing = Arc::new(TestElasticBacking::default());
        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            memory.clone(),
            Duration::from_secs(30),
            Some(Arc::new(TestElasticProvisioner {
                backing: backing.clone(),
                initial_block_count: 4,
            })),
            allowed_test_profiles(),
        )
        .unwrap();
        let mut request = provisional_request(vec![
            ProvisionalKvCandidate {
                block_count: 4,
                bytes_per_block: 4096,
                effective_target_concurrency: 2,
            },
            ProvisionalKvCandidate {
                block_count: 2,
                bytes_per_block: 4096,
                effective_target_concurrency: 1,
            },
        ]);
        request.maximum_block_count = Some(8);
        let grant = coordinator.reserve_provisional_kv_grant(&request).unwrap();
        let registration = elastic_registration_with_grant(grant.proof);
        let receipt = coordinator.register(&registration).unwrap();
        let attachment = shared_attachment(&receipt);
        coordinator
            .attach(&registration.participant_id, &attachment)
            .unwrap();
        coordinator
            .activate(&registration.participant_id, receipt.participant_epoch)
            .unwrap();
        (memory, coordinator, backing, registration, receipt)
    }

    fn resize_poll(
        receipt: &KvRegistrationReceipt,
        actor: KvResizeActor,
        applied_generation: u64,
    ) -> KvResizePollRequest {
        KvResizePollRequest {
            participant_epoch: receipt.participant_epoch,
            actor,
            applied_generation,
        }
    }

    fn resize_ack(operation: &KvPoolResizeOperation, actor: KvResizeActor) -> KvResizeAckRequest {
        KvResizeAckRequest {
            participant_epoch: operation.participant_epoch,
            actor,
            binding_id: operation.binding_id.clone(),
            resize_generation: operation.resize_generation,
            stage: operation.stage,
            applied_block_count: operation.target_block_count,
        }
    }

    #[test]
    fn worker_resize_response_renumbers_rank_local_handles_densely() {
        let operation = |binding_id: &str, handle_index: u32| KvPoolResizeOperation {
            participant_epoch: 1,
            resize_generation: 2,
            binding_id: binding_id.to_string(),
            stage: KvPoolResizeStage::MapWorkers,
            from_block_count: 4,
            target_block_count: 8,
            bytes_per_block: 4096,
            allocation_granularity_bytes: 8192,
            segments: vec![KvVmmSegmentDescriptor {
                segment_id: format!("{binding_id}-grow"),
                offset_bytes: 4 * 4096,
                length_bytes: 4 * 4096,
                handle_index,
            }],
        };
        let mut operations = vec![operation("rank-1", 7), operation("rank-3", 19)];

        densify_worker_resize_handle_indices(&mut operations).unwrap();

        assert_eq!(operations[0].segments[0].handle_index, 0);
        assert_eq!(operations[1].segments[0].handle_index, 1);
        operations
            .iter()
            .for_each(|operation| operation.validate().unwrap());
    }

    fn kv_snapshot_bytes(memory: &MemoryAuthority) -> (usize, usize) {
        memory
            .snapshot()
            .rows
            .iter()
            .filter(|row| row.class == MemoryAllocationClass::KvCache)
            .fold((0, 0), |current, row| {
                (
                    current.0 + row.reserved_bytes,
                    current.1 + row.committed_bytes,
                )
            })
    }

    #[test]
    fn provisional_grant_transfers_without_release_reacquire_or_double_charge() {
        use kapsl_kv_abi::KvCoordinator as _;

        let memory = test_memory();
        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            memory.clone(),
            Duration::from_secs(30),
            Some(Arc::new(TestSharedProvisioner {
                backing: Arc::new(TestSharedBacking::default()),
            })),
            allowed_test_profiles(),
        )
        .unwrap();
        let grant = coordinator
            .reserve_provisional_kv_grant(&provisional_request(vec![
                ProvisionalKvCandidate {
                    block_count: 4,
                    bytes_per_block: 4096,
                    effective_target_concurrency: 2,
                },
                ProvisionalKvCandidate {
                    block_count: 2,
                    bytes_per_block: 4096,
                    effective_target_concurrency: 1,
                },
            ]))
            .unwrap();
        assert_eq!(grant.selected_candidate_index, 0);
        assert_eq!(grant.selected_candidate.block_count, 4);
        assert_eq!(coordinator.provisional_grant_count(), 1);
        assert_eq!(kv_snapshot_bytes(&memory), (4 * 4096, 0));

        let registration = registration_with_grant(grant.proof.clone());
        let receipt = coordinator.register(&registration).unwrap();
        assert_eq!(receipt.shared_pools[0].block_count, 4);
        assert_eq!(coordinator.provisional_grant_count(), 0);
        assert_eq!(kv_snapshot_bytes(&memory), (4 * 4096, 4 * 4096));

        // Repeated scheduler/worker registration is idempotent and does not
        // consume or create another authority charge.
        assert_eq!(coordinator.register(&registration).unwrap(), receipt);
        assert_eq!(kv_snapshot_bytes(&memory), (4 * 4096, 4 * 4096));

        let mut replay = registration;
        replay.participant_id = "vllm:engine-2".to_string();
        assert!(matches!(
            coordinator.register(&replay),
            Err(KvContractError::InvalidRequest { .. })
        ));
        assert_eq!(kv_snapshot_bytes(&memory), (4 * 4096, 4 * 4096));
    }

    #[test]
    fn live_resize_orders_grow_and_shrink_and_releases_to_certified_minimum() {
        use kapsl_kv_abi::KvCoordinator as _;

        let (memory, coordinator, backing, registration, receipt) = elastic_coordinator();
        assert_eq!(kv_snapshot_bytes(&memory), (4 * 4096, 4 * 4096));
        let initial = coordinator
            .managed_vllm_resize_snapshot("vllm")
            .unwrap()
            .expect("elastic snapshot");
        assert_eq!(initial.minimum_block_count, 2);
        assert_eq!(initial.current_block_count, 4);
        assert_eq!(initial.maximum_block_count, 8);
        #[cfg(unix)]
        {
            let handles = coordinator
                .response_vmm_handles(
                    &KvControlRequest::Register {
                        registration: registration.clone(),
                    },
                    &KvControlResponse::Registered {
                        receipt: receipt.clone(),
                    },
                )
                .unwrap();
            assert_eq!(handles.len(), 2);
        }

        let grow_generation = coordinator.request_managed_vllm_resize("vllm", 8).unwrap();
        let worker = KvResizeActor::Worker {
            shard: KvShard::default(),
        };
        let worker_grow = coordinator
            .poll_resize(
                &registration.participant_id,
                &resize_poll(&receipt, worker, 0),
            )
            .unwrap();
        assert!(worker_grow.pending);
        assert_eq!(worker_grow.operations.len(), 1);
        let worker_grow = &worker_grow.operations[0];
        assert_eq!(worker_grow.stage, KvPoolResizeStage::MapWorkers);
        assert_eq!(worker_grow.target_block_count, 8);
        assert_eq!(worker_grow.resize_generation, grow_generation);
        #[cfg(unix)]
        {
            let handles = coordinator
                .response_vmm_handles(
                    &KvControlRequest::ResizePoll {
                        participant_id: registration.participant_id.clone(),
                        request: resize_poll(&receipt, worker, 0),
                    },
                    &KvControlResponse::Resize {
                        pending: true,
                        operations: vec![worker_grow.clone()],
                    },
                )
                .unwrap();
            assert_eq!(handles.len(), 1);
        }

        // Neither the wrong actor nor an inexact applied count may advance the
        // transaction after physical pages have been mapped.
        assert!(coordinator
            .ack_resize(
                &registration.participant_id,
                &resize_ack(worker_grow, KvResizeActor::Scheduler),
            )
            .is_err());
        let mut wrong_count = resize_ack(worker_grow, worker);
        wrong_count.applied_block_count -= 1;
        assert!(coordinator
            .ack_resize(&registration.participant_id, &wrong_count)
            .is_err());
        coordinator
            .ack_resize(
                &registration.participant_id,
                &resize_ack(worker_grow, worker),
            )
            .unwrap();

        let scheduler_grow = coordinator
            .poll_resize(
                &registration.participant_id,
                &resize_poll(&receipt, KvResizeActor::Scheduler, 0),
            )
            .unwrap();
        assert_eq!(scheduler_grow.operations.len(), 1);
        assert_eq!(
            scheduler_grow.operations[0].stage,
            KvPoolResizeStage::ActivateScheduler
        );
        assert!(scheduler_grow.operations[0].segments.is_empty());
        coordinator
            .ack_resize(
                &registration.participant_id,
                &resize_ack(&scheduler_grow.operations[0], KvResizeActor::Scheduler),
            )
            .unwrap();
        assert_eq!(kv_snapshot_bytes(&memory), (8 * 4096, 8 * 4096));
        assert_eq!(
            coordinator
                .managed_vllm_resize_snapshot("vllm")
                .unwrap()
                .unwrap()
                .current_block_count,
            8
        );

        // Six blocks is VMM-aligned but lies inside the single 4->8 growth
        // allocation. The coordinator resolves it to the preceding committed
        // boundary before asking the scheduler to retire blocks.
        let shrink_generation = coordinator.request_managed_vllm_resize("vllm", 6).unwrap();
        let pending = coordinator
            .managed_vllm_resize_snapshot("vllm")
            .unwrap()
            .unwrap();
        assert_eq!(pending.pending_target_block_count, Some(4));
        let scheduler_shrink = coordinator
            .poll_resize(
                &registration.participant_id,
                &resize_poll(&receipt, KvResizeActor::Scheduler, grow_generation),
            )
            .unwrap();
        assert_eq!(scheduler_shrink.operations.len(), 1);
        assert_eq!(
            scheduler_shrink.operations[0].stage,
            KvPoolResizeStage::RetireScheduler
        );
        assert_eq!(scheduler_shrink.operations[0].target_block_count, 4);
        assert_eq!(
            scheduler_shrink.operations[0].resize_generation,
            shrink_generation
        );
        coordinator
            .ack_resize(
                &registration.participant_id,
                &resize_ack(&scheduler_shrink.operations[0], KvResizeActor::Scheduler),
            )
            .unwrap();
        let worker_shrink = coordinator
            .poll_resize(
                &registration.participant_id,
                &resize_poll(&receipt, worker, grow_generation),
            )
            .unwrap();
        assert_eq!(worker_shrink.operations.len(), 1);
        assert_eq!(
            worker_shrink.operations[0].stage,
            KvPoolResizeStage::UnmapWorkers
        );
        assert_eq!(
            worker_shrink.operations[0].segments[0].offset_bytes,
            4 * 4096
        );
        coordinator
            .ack_resize(
                &registration.participant_id,
                &resize_ack(&worker_shrink.operations[0], worker),
            )
            .unwrap();
        assert_eq!(kv_snapshot_bytes(&memory), (4 * 4096, 4 * 4096));
        assert_eq!(backing.released_to_blocks.load(Ordering::Acquire), 4);

        let minimum_generation = coordinator.request_managed_vllm_resize("vllm", 2).unwrap();
        let scheduler_minimum = coordinator
            .poll_resize(
                &registration.participant_id,
                &resize_poll(&receipt, KvResizeActor::Scheduler, shrink_generation),
            )
            .unwrap();
        assert_eq!(
            scheduler_minimum.operations[0].resize_generation,
            minimum_generation
        );
        coordinator
            .ack_resize(
                &registration.participant_id,
                &resize_ack(&scheduler_minimum.operations[0], KvResizeActor::Scheduler),
            )
            .unwrap();
        let worker_minimum = coordinator
            .poll_resize(
                &registration.participant_id,
                &resize_poll(&receipt, worker, shrink_generation),
            )
            .unwrap();
        coordinator
            .ack_resize(
                &registration.participant_id,
                &resize_ack(&worker_minimum.operations[0], worker),
            )
            .unwrap();

        let final_snapshot = coordinator
            .managed_vllm_resize_snapshot("vllm")
            .unwrap()
            .unwrap();
        assert_eq!(final_snapshot.current_block_count, 2);
        assert_eq!(final_snapshot.minimum_block_count, 2);
        assert_eq!(backing.released_to_blocks.load(Ordering::Acquire), 2);
        assert_eq!(kv_snapshot_bytes(&memory), (2 * 4096, 2 * 4096));
    }

    #[test]
    fn failed_physical_growth_rolls_back_the_provisional_authority_charge() {
        let (memory, coordinator, backing, _registration, _receipt) = elastic_coordinator();
        assert_eq!(kv_snapshot_bytes(&memory), (4 * 4096, 4 * 4096));
        assert_eq!(backing.mapped_blocks.load(Ordering::Acquire), 4);
        backing.fail_grow.store(true, Ordering::Release);

        let error = coordinator
            .request_managed_vllm_resize("vllm", 8)
            .expect_err("injected physical growth must fail");
        assert!(error
            .to_string()
            .contains("injected backing growth failure"));
        assert_eq!(backing.mapped_blocks.load(Ordering::Acquire), 4);
        assert_eq!(backing.released_to_blocks.load(Ordering::Acquire), 4);
        assert_eq!(kv_snapshot_bytes(&memory), (4 * 4096, 4 * 4096));
        assert!(coordinator
            .managed_vllm_resize_snapshot("vllm")
            .unwrap()
            .is_some_and(|snapshot| snapshot.pending_target_block_count.is_none()));
    }

    #[test]
    fn ambiguous_growth_rollback_retains_the_authority_charge_until_retirement() {
        let (memory, coordinator, backing, _registration, _receipt) = elastic_coordinator();
        backing.fail_grow.store(true, Ordering::Release);
        backing.fail_tail_release.store(true, Ordering::Release);

        let error = coordinator
            .request_managed_vllm_resize("vllm", 8)
            .expect_err("ambiguous physical growth must fail closed");
        assert!(error.contains("rollback retained the charge/backing"));
        assert_eq!(backing.mapped_blocks.load(Ordering::Acquire), 4);
        assert_eq!(kv_snapshot_bytes(&memory), (8 * 4096, 4 * 4096));
        assert!(!coordinator.managed_participant_is_active("vllm").unwrap());

        backing.fail_tail_release.store(false, Ordering::Release);
        assert_eq!(
            coordinator
                .retire_participants_after_backend_exit("vllm")
                .unwrap(),
            1
        );
        assert_eq!(kv_snapshot_bytes(&memory), (0, 0));
    }

    #[test]
    fn live_resize_timeout_fences_participant_and_retains_ambiguous_capacity() {
        let (memory, coordinator, backing, registration, _receipt) = elastic_coordinator();
        let fence = Arc::new(ManagedVllmKvReadinessFence::new());
        coordinator
            .register_managed_readiness_fence("vllm", Arc::downgrade(&fence))
            .unwrap();
        coordinator.request_managed_vllm_resize("vllm", 8).unwrap();
        {
            let mut state = coordinator.state.lock();
            state
                .participants
                .get_mut(&registration.participant_id)
                .and_then(|participant| participant.resize.as_mut())
                .expect("pending resize")
                .deadline = Instant::now();
        }

        assert_eq!(coordinator.expire_stale(), 1);
        let failed = coordinator
            .managed_vllm_resize_snapshot("vllm")
            .unwrap()
            .unwrap();
        assert!(failed
            .failure
            .as_deref()
            .is_some_and(|message| message.contains("deadline")));
        assert!(!coordinator.managed_participant_is_active("vllm").unwrap());
        assert_eq!(fence.snapshot(), 1);
        assert_eq!(backing.mapped_blocks.load(Ordering::Acquire), 8);
        assert_eq!(kv_snapshot_bytes(&memory), (8 * 4096, 4 * 4096));
    }

    #[test]
    fn failed_backing_release_quarantines_authority_charge_until_retry() {
        let (memory, coordinator, backing, _registration, _receipt) = elastic_coordinator();
        backing.fail_release.store(true, Ordering::Release);

        let error = coordinator
            .retire_participants_after_backend_exit("vllm")
            .unwrap_err();
        assert!(error.contains("quarantined"));
        assert_eq!(coordinator.participant_count(), 1);
        assert_eq!(kv_snapshot_bytes(&memory), (4 * 4096, 4 * 4096));
        assert_eq!(backing.release_attempts.load(Ordering::Acquire), 1);

        backing.fail_release.store(false, Ordering::Release);
        assert_eq!(
            coordinator
                .retire_participants_after_backend_exit("vllm")
                .unwrap(),
            1
        );
        assert_eq!(coordinator.participant_count(), 0);
        assert_eq!(kv_snapshot_bytes(&memory), (0, 0));
        assert_eq!(backing.release_attempts.load(Ordering::Acquire), 2);
    }

    #[test]
    fn provisional_grant_falls_back_to_the_first_whole_block_candidate_that_fits() {
        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_secs(30),
            Some(Arc::new(TestSharedProvisioner {
                backing: Arc::new(TestSharedBacking::default()),
            })),
            allowed_test_profiles(),
        )
        .unwrap();
        let request = provisional_request(vec![
            ProvisionalKvCandidate {
                block_count: 1_000_000_000_000,
                bytes_per_block: 4096,
                effective_target_concurrency: 16,
            },
            ProvisionalKvCandidate {
                block_count: 4,
                bytes_per_block: 4096,
                effective_target_concurrency: 1,
            },
        ]);

        let grant = coordinator.reserve_provisional_kv_grant(&request).unwrap();

        assert_eq!(grant.selected_candidate_index, 1);
        assert_eq!(grant.selected_candidate.block_count, 4);
        assert_eq!(grant.selected_candidate.effective_target_concurrency, 1);
    }

    #[test]
    fn provisional_candidate_validation_allows_the_full_u64_block_count_shape() {
        let request = provisional_request(vec![ProvisionalKvCandidate {
            block_count: u64::MAX,
            bytes_per_block: 1,
            effective_target_concurrency: 1,
        }]);

        validate_provisional_grant_request(&request).unwrap();
    }

    #[test]
    fn mismatched_registration_does_not_consume_provisional_grant() {
        use kapsl_kv_abi::KvCoordinator as _;

        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_secs(30),
            Some(Arc::new(TestSharedProvisioner {
                backing: Arc::new(TestSharedBacking::default()),
            })),
            allowed_test_profiles(),
        )
        .unwrap();
        let grant = coordinator
            .reserve_provisional_kv_grant(&provisional_request(vec![ProvisionalKvCandidate {
                block_count: 4,
                bytes_per_block: 4096,
                effective_target_concurrency: 2,
            }]))
            .unwrap();
        let mut mismatched = registration_with_grant(grant.proof.clone());
        mismatched.capacity_model.groups[0].max_allocations = Some(3);
        assert!(matches!(
            coordinator.register(&mismatched),
            Err(KvContractError::InvalidCapabilities { .. })
        ));
        assert_eq!(coordinator.provisional_grant_count(), 1);

        coordinator
            .register(&registration_with_grant(grant.proof))
            .expect("the intended registration still consumes the grant");
        assert_eq!(coordinator.provisional_grant_count(), 0);
    }

    #[test]
    fn provisional_grant_expiry_and_supervised_cancel_release_authority() {
        let memory = test_memory();
        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            memory.clone(),
            Duration::from_secs(30),
            Some(Arc::new(TestSharedProvisioner {
                backing: Arc::new(TestSharedBacking::default()),
            })),
            allowed_test_profiles(),
        )
        .unwrap();
        let request = provisional_request(vec![ProvisionalKvCandidate {
            block_count: 4,
            bytes_per_block: 4096,
            effective_target_concurrency: 1,
        }]);
        let first = coordinator.reserve_provisional_kv_grant(&request).unwrap();
        {
            let mut state = coordinator.state.lock();
            state
                .provisional_grants
                .get_mut(&first.proof.token)
                .expect("provisional record")
                .expires_at = Instant::now() - Duration::from_millis(1);
        }
        assert_eq!(coordinator.expire_provisional_grants(), 1);
        assert_eq!(kv_snapshot_bytes(&memory), (0, 0));

        coordinator.reserve_provisional_kv_grant(&request).unwrap();
        assert_eq!(kv_snapshot_bytes(&memory), (4 * 4096, 0));
        assert_eq!(
            coordinator
                .retire_participants_after_backend_exit("vllm")
                .unwrap(),
            0
        );
        assert_eq!(coordinator.provisional_grant_count(), 0);
        assert_eq!(kv_snapshot_bytes(&memory), (0, 0));
    }

    #[test]
    fn shared_pool_profile_allowlist_is_exact_and_validated() {
        use kapsl_kv_abi::KvCoordinator as _;

        let values = vec!["kapsl-vllm-connector,0.4.0,0.10.2,vllm-v1-packed-cuda-ipc".to_string()];
        let profiles = parse_shared_pool_profiles(&values).unwrap();
        assert!(profiles.contains(&KvAdapterProfile {
            adapter_id: "kapsl-vllm-connector".to_string(),
            adapter_version: "0.4.0".to_string(),
            backend_version: "0.10.2".to_string(),
            profile_id: "vllm-v1-packed-cuda-ipc".to_string(),
        }));
        assert!(parse_shared_pool_profiles(&["missing,fields".to_string()]).is_err());
        assert!(parse_shared_pool_profiles(&["adapter,,backend,profile".to_string()]).is_err());

        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_secs(30),
            Some(Arc::new(TestSharedProvisioner {
                backing: Arc::new(TestSharedBacking::default()),
            })),
            BTreeSet::new(),
        )
        .unwrap();
        assert!(matches!(
            coordinator.register(&shared_registration()),
            Err(KvContractError::InvalidCapabilities { .. })
        ));

        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_secs(30),
            Some(Arc::new(TestSharedProvisioner {
                backing: Arc::new(TestSharedBacking::default()),
            })),
            profiles,
        )
        .unwrap();
        assert!(matches!(
            coordinator.register(&shared_registration()),
            Err(KvContractError::InvalidCapabilities { .. })
        ));
        assert_eq!(coordinator.participant_count(), 0);
    }

    fn activate_shared(coordinator: &ExternalKvCoordinator, receipt: &KvRegistrationReceipt) {
        use kapsl_kv_abi::KvCoordinator as _;

        coordinator
            .attach("vllm:test:scheduler", &shared_attachment(receipt))
            .unwrap();
        coordinator
            .activate("vllm:test:scheduler", receipt.participant_epoch)
            .unwrap();
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
    fn shared_pool_is_not_reservable_until_every_binding_is_activated() {
        use kapsl_kv_abi::KvCoordinator as _;

        let provisioner = Arc::new(TestSharedProvisioner {
            backing: Arc::new(TestSharedBacking::default()),
        });
        let mut allowed_profiles = allowed_test_profiles();
        allowed_profiles.insert(KvAdapterProfile {
            adapter_id: "kapsl-test-adapter".to_string(),
            adapter_version: "1.0.0".to_string(),
            backend_version: "untested-build".to_string(),
            profile_id: "test-direct-v1".to_string(),
        });
        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_secs(30),
            Some(provisioner),
            allowed_profiles,
        )
        .unwrap();
        let receipt = coordinator.register(&shared_registration()).unwrap();

        assert!(matches!(
            coordinator.reserve("vllm:test:scheduler", &reservation(None)),
            Err(KvContractError::InvalidRequest { .. })
        ));
        assert!(matches!(
            coordinator.activate("vllm:test:scheduler", receipt.participant_epoch),
            Err(KvContractError::InvalidRequest { .. })
        ));

        let attachment = shared_attachment(&receipt);
        let mut untested_attachment = attachment.clone();
        untested_attachment.profile.backend_version = "untested-build".to_string();
        assert!(matches!(
            coordinator.attach("vllm:test:scheduler", &untested_attachment),
            Err(KvContractError::InvalidRequest { .. })
        ));
        coordinator
            .attach("vllm:test:scheduler", &attachment)
            .unwrap();
        coordinator
            .activate("vllm:test:scheduler", receipt.participant_epoch)
            .unwrap();
        let lease = coordinator
            .reserve("vllm:test:scheduler", &reservation(None))
            .unwrap();
        assert!(matches!(
            coordinator.detach(
                "vllm:test:scheduler",
                &KvSharedPoolDetachRequest {
                    participant_epoch: receipt.participant_epoch,
                    binding_ids: vec![attachment.binding_id.clone()],
                    shard: attachment.shard,
                    completion: KvReleaseCompletion::BackendSynchronized,
                },
            ),
            Err(KvContractError::InvalidRequest { .. })
        ));
        coordinator
            .release(
                "vllm:test:scheduler",
                &lease.lease_id,
                Some(&KvReleaseCompletion::BackendSynchronized),
            )
            .unwrap();
        coordinator
            .detach(
                "vllm:test:scheduler",
                &KvSharedPoolDetachRequest {
                    participant_epoch: receipt.participant_epoch,
                    binding_ids: vec![attachment.binding_id],
                    shard: attachment.shard,
                    completion: KvReleaseCompletion::BackendSynchronized,
                },
            )
            .unwrap();
        assert_eq!(coordinator.participant_count(), 0);
    }

    #[test]
    fn supervised_backend_retirement_releases_participant_pool_and_leases() {
        use kapsl_kv_abi::KvCoordinator as _;

        let backing = Arc::new(TestSharedBacking::default());
        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_secs(30),
            Some(Arc::new(TestSharedProvisioner {
                backing: backing.clone(),
            })),
            allowed_test_profiles(),
        )
        .unwrap();
        let baseline_refs = Arc::strong_count(&backing);
        let receipt = coordinator.register(&shared_registration()).unwrap();
        activate_shared(&coordinator, &receipt);
        coordinator
            .reserve("vllm:test:scheduler", &reservation(None))
            .unwrap();
        assert_eq!(coordinator.participant_count(), 1);
        assert_eq!(coordinator.lease_count(), 1);
        assert!(Arc::strong_count(&backing) > baseline_refs);

        assert!(coordinator
            .retire_participant_after_backend_exit("vllm:test:scheduler")
            .unwrap());
        assert_eq!(coordinator.participant_count(), 0);
        assert_eq!(coordinator.lease_count(), 0);
        assert_eq!(Arc::strong_count(&backing), baseline_refs);
        assert!(!coordinator
            .retire_participant_after_backend_exit("vllm:test:scheduler")
            .unwrap());
    }

    #[test]
    fn supervised_base_retirement_matches_engine_suffix_but_not_prefix_collision() {
        use kapsl_kv_abi::KvCoordinator as _;

        let backing = Arc::new(TestSharedBacking::default());
        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_secs(30),
            Some(Arc::new(TestSharedProvisioner {
                backing: backing.clone(),
            })),
            allowed_test_profiles(),
        )
        .unwrap();
        let baseline_refs = Arc::strong_count(&backing);
        let target_fence = Arc::new(ManagedVllmKvReadinessFence::new());
        let collision_fence = Arc::new(ManagedVllmKvReadinessFence::new());
        coordinator
            .register_managed_readiness_fence("kapsl-model-1", Arc::downgrade(&target_fence))
            .unwrap();
        coordinator
            .register_managed_readiness_fence("kapsl-model-10", Arc::downgrade(&collision_fence))
            .unwrap();

        let mut target = shared_registration();
        target.participant_id = "kapsl-model-1:engine-uuid".to_string();
        coordinator.register(&target).unwrap();
        let mut collision = shared_registration();
        collision.participant_id = "kapsl-model-10:engine-uuid".to_string();
        coordinator.register(&collision).unwrap();
        assert_eq!(coordinator.participant_count(), 2);

        assert_eq!(
            coordinator
                .retire_participants_after_backend_exit("kapsl-model-1")
                .unwrap(),
            1
        );
        assert_eq!(coordinator.participant_count(), 1);
        assert!(coordinator
            .state
            .lock()
            .participants
            .contains_key("kapsl-model-10:engine-uuid"));
        assert_eq!(target_fence.snapshot(), 1);
        assert_eq!(collision_fence.snapshot(), 0);
        assert_eq!(Arc::strong_count(&backing), baseline_refs + 1);
    }

    #[test]
    fn validated_detach_fences_managed_readiness_but_invalid_detach_does_not() {
        use kapsl_kv_abi::KvCoordinator as _;

        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_secs(30),
            Some(Arc::new(TestSharedProvisioner {
                backing: Arc::new(TestSharedBacking::default()),
            })),
            allowed_test_profiles(),
        )
        .unwrap();
        let fence = Arc::new(ManagedVllmKvReadinessFence::new());
        coordinator
            .register_managed_readiness_fence("kapsl-model-1", Arc::downgrade(&fence))
            .unwrap();

        let mut registration = shared_registration();
        registration.participant_id = "kapsl-model-1:engine-a".to_string();
        let receipt = coordinator.register(&registration).unwrap();
        let attachment = shared_attachment(&receipt);
        coordinator
            .attach(&registration.participant_id, &attachment)
            .unwrap();
        coordinator
            .activate(&registration.participant_id, receipt.participant_epoch)
            .unwrap();

        let valid = KvSharedPoolDetachRequest {
            participant_epoch: receipt.participant_epoch,
            binding_ids: vec![attachment.binding_id.clone()],
            shard: attachment.shard,
            completion: KvReleaseCompletion::BackendSynchronized,
        };
        let mut stale_epoch = valid.clone();
        stale_epoch.participant_epoch += 1;
        assert!(coordinator
            .detach(&registration.participant_id, &stale_epoch)
            .is_err());
        assert_eq!(fence.snapshot(), 0);
        assert!(coordinator
            .managed_participant_is_active("kapsl-model-1")
            .unwrap());

        let mut unknown_binding = valid.clone();
        unknown_binding.binding_ids = vec!["not-attached".to_string()];
        assert!(coordinator
            .detach(&registration.participant_id, &unknown_binding)
            .is_err());
        assert_eq!(fence.snapshot(), 0);
        assert!(coordinator
            .managed_participant_is_active("kapsl-model-1")
            .unwrap());

        coordinator
            .detach(&registration.participant_id, &valid)
            .unwrap();
        assert_eq!(fence.snapshot(), 1);
        assert_eq!(coordinator.participant_count(), 0);
    }

    #[test]
    fn managed_readiness_registration_is_weak_and_reuses_an_expired_base() {
        let coordinator = ExternalKvCoordinator::new(test_memory(), Duration::from_secs(30))
            .expect("test coordinator");
        let original = Arc::new(ManagedVllmKvReadinessFence::new());
        coordinator
            .register_managed_readiness_fence("kapsl-model-1", Arc::downgrade(&original))
            .unwrap();
        assert_eq!(Arc::strong_count(&original), 1);
        drop(original);

        let replacement = Arc::new(ManagedVllmKvReadinessFence::new());
        coordinator
            .register_managed_readiness_fence("kapsl-model-1", Arc::downgrade(&replacement))
            .unwrap();
        assert_eq!(Arc::strong_count(&replacement), 1);
    }

    #[test]
    fn managed_participant_readiness_requires_one_active_generation() {
        use kapsl_kv_abi::KvCoordinator as _;

        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_secs(30),
            Some(Arc::new(TestSharedProvisioner {
                backing: Arc::new(TestSharedBacking::default()),
            })),
            allowed_test_profiles(),
        )
        .unwrap();
        assert!(!coordinator
            .managed_participant_is_active("kapsl-model-1")
            .unwrap());

        let mut registration = shared_registration();
        registration.participant_id = "kapsl-model-1:engine-a".to_string();
        let receipt = coordinator.register(&registration).unwrap();
        assert!(!coordinator
            .managed_participant_is_active("kapsl-model-1")
            .unwrap());
        coordinator
            .attach(&registration.participant_id, &shared_attachment(&receipt))
            .unwrap();
        coordinator
            .activate(&registration.participant_id, receipt.participant_epoch)
            .unwrap();
        assert!(coordinator
            .managed_participant_is_active("kapsl-model-1")
            .unwrap());

        let mut second = shared_registration();
        second.participant_id = "kapsl-model-1:engine-b".to_string();
        coordinator.register(&second).unwrap();
        assert!(coordinator
            .managed_participant_is_active("kapsl-model-1")
            .unwrap_err()
            .contains("multiple live"));
    }

    #[test]
    fn managed_vllm_snapshot_distinguishes_backing_active_and_idle_blocks() {
        use kapsl_kv_abi::KvCoordinator as _;

        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            test_memory(),
            Duration::from_secs(30),
            Some(Arc::new(TestSharedProvisioner {
                backing: Arc::new(TestSharedBacking::default()),
            })),
            allowed_test_profiles(),
        )
        .unwrap();
        let mut registration = shared_registration();
        registration.participant_id = "kapsl-model-7:engine-a".to_string();
        registration.capacity_model.groups[0].memory_domains =
            vec![KvMemoryDomain::Cuda { device_id: 0 }];
        let receipt = coordinator.register(&registration).unwrap();
        coordinator
            .attach(&registration.participant_id, &shared_attachment(&receipt))
            .unwrap();
        coordinator
            .activate(&registration.participant_id, receipt.participant_epoch)
            .unwrap();

        let initial = coordinator
            .managed_vllm_kv_snapshot("kapsl-model-7")
            .unwrap();
        assert_eq!(initial.len(), 1);
        assert_eq!(initial[0].backing_bytes, 4 * 4096);
        assert_eq!(initial[0].total_blocks, 4);
        assert_eq!(initial[0].idle_blocks, 4);
        assert_eq!(initial[0].active_blocks, 0);
        assert!(initial[0].participant_active);

        let lease = coordinator
            .reserve(&registration.participant_id, &reservation(None))
            .unwrap();
        let occupied = coordinator
            .managed_vllm_kv_snapshot("kapsl-model-7")
            .unwrap();
        assert_eq!(occupied[0].active_blocks, 2);
        assert_eq!(occupied[0].idle_blocks, 2);
        assert_eq!(occupied[0].logical_leased_bytes, 2 * 4096);
        assert_eq!(occupied[0].active_sequences, 1);

        coordinator
            .release(
                &registration.participant_id,
                &lease.lease_id,
                Some(&KvReleaseCompletion::BackendSynchronized),
            )
            .unwrap();
        let released = coordinator
            .managed_vllm_kv_snapshot("kapsl-model-7")
            .unwrap();
        assert_eq!(released[0].active_blocks, 0);
        assert_eq!(released[0].idle_blocks, 4);
        assert_eq!(released[0].active_sequences, 0);
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
            allowed_test_profiles(),
        )
        .unwrap();
        let receipt = coordinator.register(&shared_registration()).unwrap();
        assert_eq!(receipt.shared_pools.len(), 1);
        activate_shared(&coordinator, &receipt);
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
            allowed_test_profiles(),
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
        activate_shared(&coordinator, &receipt);
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
            allowed_test_profiles(),
        )
        .unwrap();
        let receipt = coordinator.register(&shared_registration()).unwrap();
        activate_shared(&coordinator, &receipt);
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
