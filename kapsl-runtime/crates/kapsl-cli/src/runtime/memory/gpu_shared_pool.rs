//! Isolated GPU pool backing for out-of-process shared KV participants.
//!
//! The common per-device pool owns HAL arena, IPC and VMM regions. This module
//! adapts isolated region leases to KV descriptors and the worker protocol.
//!
//! `GpuSharedPoolProvisioner` admits the isolated backings through the same
//! `MemoryAuthority` that budgets the general arena. They use separate physical
//! allocations because CUDA IPC exports the entire allocation: exporting the
//! general arena would expose unrelated models and sessions to the importer.

use super::gpu_regions::GpuRegionIsolation;
use super::{
    MemoryAllocationClass, MemoryAuthority, MemoryClaim, MemoryClaimSource, MemoryDomain,
    MemoryLease, MemoryOwner, MemoryPlan,
};
use crate::runtime::kv::{ProvisionedSharedPools, SharedPoolBacking, SharedPoolProvisioner};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use cudarc::driver::CudaDevice;
use kapsl_kv_abi::{
    KvContractError, KvElasticPoolDescriptor, KvFeature, KvMemoryDomain, KvParticipantRegistration,
    KvSharedPoolAllocationMode, KvSharedPoolDescriptor, KvTransport, KvVmmSegmentDescriptor,
};
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet, HashMap};
#[cfg(unix)]
use std::os::fd::OwnedFd;
use std::sync::Arc;

/// Creates fixed IPC or elastic VMM GPU pools under runtime memory authority.
pub(crate) struct GpuSharedPoolProvisioner {
    memory: Arc<MemoryAuthority>,
    failed_setup: Arc<Mutex<Vec<FailedGpuSetup>>>,
}

impl GpuSharedPoolProvisioner {
    pub(crate) fn new(memory: Arc<MemoryAuthority>) -> Arc<Self> {
        Arc::new(Self {
            memory,
            failed_setup: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn retry_failed_setup(&self) {
        // These handles were never sent to an importer. Driver operations run
        // outside the queue lock and successful cleanup precedes grant release.
        let failed = std::mem::take(&mut *self.failed_setup.lock());
        for mut setup in failed {
            if setup
                .backing
                .as_ref()
                .expect("retained backing")
                .release_after_fence()
                .is_ok()
            {
                drop(setup.backing.take());
                drop(setup.grant.take());
            } else {
                self.failed_setup.lock().push(setup);
            }
        }
    }
}

struct FailedGpuSetup {
    backing: Option<Arc<dyn SharedPoolBacking>>,
    grant: Option<MemoryLease>,
}

impl Drop for FailedGpuSetup {
    fn drop(&mut self) {
        let Some(backing) = self.backing.take() else {
            return;
        };
        let grant = self.grant.take();
        if let Err(error) = backing.release_after_fence() {
            log::error!(
                "[gpu-pool] setup cleanup failed at shutdown; retaining backing and grant: {error}"
            );
            std::mem::forget((backing, grant));
        } else {
            drop(backing);
            drop(grant);
        }
    }
}

struct GpuSetup {
    backing: Option<GpuSharedPoolBacking>,
    grant: Option<MemoryLease>,
    failed: Arc<Mutex<Vec<FailedGpuSetup>>>,
}

impl GpuSetup {
    fn new(grant: MemoryLease, failed: Arc<Mutex<Vec<FailedGpuSetup>>>) -> Self {
        Self {
            backing: Some(GpuSharedPoolBacking {
                allocations: HashMap::new(),
                vmm_allocations: HashMap::new(),
            }),
            grant: Some(grant),
            failed,
        }
    }

    fn finish(mut self, descriptors: Vec<KvSharedPoolDescriptor>) -> ProvisionedSharedPools {
        ProvisionedSharedPools {
            descriptors,
            backing: Arc::new(self.backing.take().expect("setup owns backing")),
            memory_lease: self.grant.take(),
        }
    }
}

impl Drop for GpuSetup {
    fn drop(&mut self) {
        let Some(backing) = self.backing.take() else {
            return;
        };
        if let Err(error) = backing.release_after_fence() {
            if let Some(grant) = self.grant.as_mut() {
                grant.commit_capacity();
            }
            log::error!("[gpu-pool] failed setup retained backing and budget for retry: {error}");
            self.failed.lock().push(FailedGpuSetup {
                backing: Some(Arc::new(backing)),
                grant: self.grant.take(),
            });
        }
        // Successful physical release occurs before the grant field drops.
    }
}

#[derive(Debug)]
struct PlannedBinding {
    descriptor: KvSharedPoolDescriptor,
    device_id: usize,
    allocation_bytes: usize,
    live_resize: bool,
}

#[derive(Debug)]
struct LogicalPool {
    group_ids: BTreeSet<String>,
    domains: BTreeSet<KvMemoryDomain>,
    block_count: u64,
    bytes_per_block: u64,
}

fn plan_bindings(
    registration: &KvParticipantRegistration,
    participant_epoch: u64,
) -> Result<Vec<PlannedBinding>, KvContractError> {
    registration.validate()?;
    let live_resize = registration
        .capabilities
        .features
        .contains(&KvFeature::LivePoolResize);
    let expected_transport = if live_resize {
        KvTransport::CudaVmm
    } else {
        KvTransport::CudaIpc
    };
    if registration.capabilities.transports != BTreeSet::from([expected_transport.clone()]) {
        return Err(KvContractError::invalid_capabilities(
            "Linux CUDA provisioner requires exactly the transport selected by the shared-pool profile",
        ));
    }

    let allocation_mode = if registration
        .capabilities
        .features
        .contains(&KvFeature::ParticipantBlockSelection)
    {
        KvSharedPoolAllocationMode::ParticipantManaged
    } else {
        KvSharedPoolAllocationMode::RuntimeLeased
    };
    let mut pools = BTreeMap::<String, LogicalPool>::new();
    for group in &registration.capacity_model.groups {
        let block_count = group.max_allocations.ok_or_else(|| {
            KvContractError::invalid_capabilities(format!(
                "CUDA IPC group '{}' has no maximum allocation count",
                group.group_id
            ))
        })?;
        if group
            .memory_domains
            .iter()
            .any(|domain| !matches!(domain, KvMemoryDomain::Cuda { .. }))
        {
            return Err(KvContractError::invalid_capabilities(format!(
                "CUDA IPC group '{}' contains a non-CUDA memory domain",
                group.group_id
            )));
        }
        let pool = pools
            .entry(group.pool_id.clone())
            .or_insert_with(|| LogicalPool {
                group_ids: BTreeSet::new(),
                domains: group.memory_domains.iter().cloned().collect(),
                block_count,
                bytes_per_block: group.bytes_per_allocation,
            });
        if pool.block_count != block_count
            || pool.bytes_per_block != group.bytes_per_allocation
            || pool.domains != group.memory_domains.iter().cloned().collect()
        {
            return Err(KvContractError::invalid_capabilities(format!(
                "CUDA IPC groups aliasing '{}' do not share one physical shape and placement",
                group.pool_id
            )));
        }
        pool.group_ids.insert(group.group_id.clone());
    }

    let mut planned = Vec::new();
    for (pool_id, pool) in pools {
        let allocation_bytes_u64 = pool
            .block_count
            .checked_mul(pool.bytes_per_block)
            .ok_or_else(|| {
                KvContractError::invalid_capabilities(format!(
                    "CUDA IPC pool '{pool_id}' byte size overflows"
                ))
            })?;
        let allocation_bytes = usize::try_from(allocation_bytes_u64).map_err(|_| {
            KvContractError::invalid_capabilities(format!(
                "CUDA IPC pool '{pool_id}' is too large for this runtime"
            ))
        })?;
        for domain in pool.domains {
            let KvMemoryDomain::Cuda { device_id } = domain else {
                unreachable!("non-CUDA domains were rejected above");
            };
            let runtime_device_id = usize::try_from(device_id).map_err(|_| {
                KvContractError::invalid_capabilities(
                    "CUDA IPC device ID does not fit the runtime address space",
                )
            })?;
            let binding_id = format!(
                "{}:{}:{}:{}:{}",
                if live_resize { "cuda-vmm" } else { "cuda-ipc" },
                registration.participant_id,
                participant_epoch,
                pool_id,
                device_id
            );
            planned.push(PlannedBinding {
                descriptor: KvSharedPoolDescriptor {
                    binding_id,
                    capacity_pool_id: pool_id.clone(),
                    generation: participant_epoch,
                    group_ids: pool.group_ids.iter().cloned().collect(),
                    memory_domain: KvMemoryDomain::Cuda { device_id },
                    block_count: pool.block_count,
                    bytes_per_block: pool.bytes_per_block,
                    allocation_mode,
                    transport: expected_transport.clone(),
                    descriptor: String::new(),
                    elastic: None,
                },
                device_id: runtime_device_id,
                allocation_bytes,
                live_resize,
            });
        }
    }
    Ok(planned)
}

mod regions;
use regions::{GpuIpcPoolAllocation, GpuVmmPoolAllocation};

struct GpuSharedPoolBacking {
    allocations: HashMap<String, GpuIpcPoolAllocation>,
    vmm_allocations: HashMap<String, GpuVmmPoolAllocation>,
}

impl SharedPoolBacking for GpuSharedPoolBacking {
    fn zero_blocks(
        &self,
        binding: &KvSharedPoolDescriptor,
        block_indices: &[u64],
    ) -> Result<(), KvContractError> {
        if let Some(allocation) = self.allocations.get(&binding.binding_id) {
            return allocation.zero_blocks(binding.bytes_per_block, block_indices);
        }
        if self.vmm_allocations.contains_key(&binding.binding_id) {
            // Participant-managed vLLM owns block selection. Every newly
            // mapped VMM segment is zeroed before publication, so logical
            // leases never prescribe individual block indices here.
            return Ok(());
        }
        Err(KvContractError::Internal {
            message: format!(
                "CUDA shared-pool backing has no allocation for binding '{}'",
                binding.binding_id
            ),
        })
    }

    fn release_after_fence(&self) -> Result<(), KvContractError> {
        for allocation in self.allocations.values() {
            allocation.release_after_fence()?;
        }
        for allocation in self.vmm_allocations.values() {
            allocation.release_after_fence()?;
        }
        Ok(())
    }

    fn grow_binding(
        &self,
        binding: &KvSharedPoolDescriptor,
        target_block_count: u64,
        resize_generation: u64,
    ) -> Result<Vec<KvVmmSegmentDescriptor>, KvContractError> {
        let allocation = self
            .vmm_allocations
            .get(&binding.binding_id)
            .ok_or_else(|| {
                KvContractError::invalid_request("only CUDA VMM bindings can grow live")
            })?;
        let target_bytes = target_block_count
            .checked_mul(binding.bytes_per_block)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| KvContractError::invalid_request("CUDA VMM growth size overflows"))?;
        allocation
            .grow_to(
                target_bytes,
                format!("{}:resize:{resize_generation}", binding.binding_id),
            )
            .map(|segment| vec![segment])
    }

    fn shrink_segments(
        &self,
        binding: &KvSharedPoolDescriptor,
        target_block_count: u64,
    ) -> Result<Vec<KvVmmSegmentDescriptor>, KvContractError> {
        let allocation = self
            .vmm_allocations
            .get(&binding.binding_id)
            .ok_or_else(|| {
                KvContractError::invalid_request("only CUDA VMM bindings can shrink live")
            })?;
        let target_bytes = target_block_count
            .checked_mul(binding.bytes_per_block)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| KvContractError::invalid_request("CUDA VMM shrink size overflows"))?;
        allocation.shrink_segments(target_bytes)
    }

    fn shrink_target_boundary(
        &self,
        binding: &KvSharedPoolDescriptor,
        requested_block_count: u64,
    ) -> Result<u64, KvContractError> {
        let allocation = self
            .vmm_allocations
            .get(&binding.binding_id)
            .ok_or_else(|| {
                KvContractError::invalid_request("only CUDA VMM bindings can shrink live")
            })?;
        let requested_bytes = requested_block_count
            .checked_mul(binding.bytes_per_block)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| KvContractError::invalid_request("CUDA VMM shrink size overflows"))?;
        let target_bytes = allocation.shrink_boundary(requested_bytes)?;
        let stride = usize::try_from(binding.bytes_per_block).map_err(|_| {
            KvContractError::invalid_capabilities("CUDA VMM block stride is too large")
        })?;
        if target_bytes % stride != 0 {
            return Err(KvContractError::Internal {
                message: "CUDA VMM segment boundary is not a whole block count".to_string(),
            });
        }
        Ok((target_bytes / stride) as u64)
    }

    fn release_binding_tail(
        &self,
        binding: &KvSharedPoolDescriptor,
        target_block_count: u64,
    ) -> Result<(), KvContractError> {
        let allocation = self
            .vmm_allocations
            .get(&binding.binding_id)
            .ok_or_else(|| {
                KvContractError::invalid_request("only CUDA VMM bindings can release a live tail")
            })?;
        let target_bytes = target_block_count
            .checked_mul(binding.bytes_per_block)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| KvContractError::invalid_request("CUDA VMM shrink size overflows"))?;
        allocation.release_tail(target_bytes)
    }

    #[cfg(unix)]
    fn export_vmm_segments(
        &self,
        segments: &[KvVmmSegmentDescriptor],
    ) -> Result<Vec<OwnedFd>, KvContractError> {
        let mut ordered = segments.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|segment| segment.handle_index);
        ordered
            .into_iter()
            .map(|segment| {
                self.vmm_allocations
                    .values()
                    .find(|allocation| allocation.contains_segment(&segment.segment_id))
                    .ok_or_else(|| KvContractError::NotFound {
                        message: format!(
                            "CUDA VMM segment '{}' has no live backing",
                            segment.segment_id
                        ),
                    })?
                    .export_segment(&segment.segment_id)
            })
            .collect()
    }
}

impl SharedPoolProvisioner for GpuSharedPoolProvisioner {
    fn provision(
        &self,
        registration: &KvParticipantRegistration,
        owner: MemoryOwner,
        participant_epoch: u64,
        precharged: Option<MemoryLease>,
        minimum_block_count: Option<u64>,
    ) -> Result<ProvisionedSharedPools, KvContractError> {
        self.retry_failed_setup();
        let planned = plan_bindings(registration, participant_epoch)?;
        let live_resize = planned.first().is_some_and(|binding| binding.live_resize);
        if live_resize
            && (planned.iter().any(|binding| !binding.live_resize)
                || planned
                    .iter()
                    .map(|binding| binding.descriptor.capacity_pool_id.as_str())
                    .collect::<BTreeSet<_>>()
                    .len()
                    != 1)
        {
            return Err(KvContractError::invalid_capabilities(
                "live CUDA VMM currently requires one physical vLLM capacity pool",
            ));
        }
        let mut precharged_bytes = HashMap::new();
        let memory_lease = if let Some(lease) = precharged {
            precharged_bytes = validate_precharged_lease(&planned, owner, &lease, !live_resize)?;
            lease
        } else if live_resize {
            return Err(KvContractError::invalid_capabilities(
                "live CUDA VMM startup requires an exact precharged provisioning grant",
            ));
        } else {
            let mut memory_plan = MemoryPlan::new();
            for binding in &planned {
                // `external` means outside the general CUDA suballocator here;
                // the runtime still owns and frees the allocation below.
                memory_plan.push(MemoryClaim::external(
                    MemoryDomain::Cuda {
                        device_id: binding.device_id,
                    },
                    owner,
                    MemoryAllocationClass::KvCache,
                    binding.descriptor.binding_id.clone(),
                    binding.allocation_bytes,
                ));
            }
            self.memory.admit(&memory_plan).map_err(|message| {
                KvContractError::CapacityExhausted {
                    message: format!("CUDA IPC shared-pool admission failed: {message}"),
                }
            })?
        };

        let mut descriptors = Vec::with_capacity(planned.len());
        let mut setup = GpuSetup::new(memory_lease, self.failed_setup.clone());
        let backing = setup.backing.as_mut().expect("setup owns backing");
        let mut next_handle_index = 0u32;
        for binding in planned {
            let pool = self
                .memory
                .gpu_device_pool(binding.device_id)
                .map_err(|message| KvContractError::Internal { message })?;
            let isolation = GpuRegionIsolation::Participant {
                owner,
                participant_id: registration.participant_id.clone(),
                binding_id: binding.descriptor.binding_id.clone(),
                generation: binding.descriptor.generation,
            };
            let mut descriptor = binding.descriptor;
            if binding.live_resize {
                let domain = MemoryDomain::Cuda {
                    device_id: binding.device_id,
                };
                let initial_bytes = precharged_bytes.get(&domain).copied().ok_or_else(|| {
                    KvContractError::invalid_capabilities(
                        "precharged CUDA VMM lease omitted a required device",
                    )
                })?;
                let stride = usize::try_from(descriptor.bytes_per_block).map_err(|_| {
                    KvContractError::invalid_capabilities(
                        "CUDA VMM block stride is too large for this runtime",
                    )
                })?;
                if initial_bytes % stride != 0 {
                    return Err(KvContractError::invalid_capabilities(
                        "precharged CUDA VMM bytes are not a whole block count",
                    ));
                }
                let initial_blocks = u64::try_from(initial_bytes / stride).map_err(|_| {
                    KvContractError::invalid_capabilities(
                        "precharged CUDA VMM block count exceeds uint64",
                    )
                })?;
                let minimum_blocks = minimum_block_count.ok_or_else(|| {
                    KvContractError::invalid_capabilities(
                        "live CUDA VMM startup omitted its certified minimum block count",
                    )
                })?;
                if minimum_blocks > initial_blocks {
                    return Err(KvContractError::invalid_capabilities(
                        "live CUDA VMM minimum exceeds the initial physical grant",
                    ));
                }
                let minimum_bytes = usize::try_from(
                    minimum_blocks
                        .checked_mul(descriptor.bytes_per_block)
                        .ok_or_else(|| {
                            KvContractError::invalid_capabilities(
                                "live CUDA VMM minimum byte count overflowed",
                            )
                        })?,
                )
                .map_err(|_| {
                    KvContractError::invalid_capabilities(
                        "live CUDA VMM minimum is too large for this runtime",
                    )
                })?;
                let allocation = GpuVmmPoolAllocation::reserve(
                    pool,
                    isolation,
                    binding.allocation_bytes,
                    minimum_bytes,
                )?;
                backing
                    .vmm_allocations
                    .insert(descriptor.binding_id.clone(), allocation);
                let allocation = &backing.vmm_allocations[&descriptor.binding_id];
                allocation.initialize(initial_bytes, &descriptor.binding_id)?;
                let mut segments = allocation.segments();
                for segment in &mut segments {
                    segment.handle_index = next_handle_index;
                    next_handle_index = next_handle_index.checked_add(1).ok_or_else(|| {
                        KvContractError::Internal {
                            message: "CUDA VMM handle index overflowed".to_string(),
                        }
                    })?;
                }
                let alignment_blocks = allocation.granularity / gcd(allocation.granularity, stride);
                let allocation_granularity_bytes =
                    u64::try_from(allocation.granularity).map_err(|_| {
                        KvContractError::invalid_capabilities(
                            "CUDA VMM allocation granularity exceeds uint64",
                        )
                    })?;
                let resize_alignment_blocks = u64::try_from(alignment_blocks).map_err(|_| {
                    KvContractError::invalid_capabilities(
                        "CUDA VMM resize alignment exceeds uint64",
                    )
                })?;
                descriptor.descriptor = "scm_rights:cuda-vmm-v1".to_string();
                descriptor.elastic = Some(KvElasticPoolDescriptor {
                    minimum_block_count: minimum_blocks,
                    mapped_block_count: initial_blocks,
                    maximum_block_count: descriptor.block_count,
                    allocation_granularity_bytes,
                    resize_alignment_blocks,
                    segments,
                });
            } else {
                let allocation =
                    GpuIpcPoolAllocation::allocate(pool, isolation, binding.allocation_bytes)?;
                backing
                    .allocations
                    .insert(descriptor.binding_id.clone(), allocation);
                let allocation = &backing.allocations[&descriptor.binding_id];
                allocation.initialize()?;
                descriptor.descriptor = allocation.export_handle()?;
            };
            descriptors.push(descriptor);
        }
        log::info!(
            "[kv-control] provisioned {} isolated CUDA {} KV binding(s) for participant '{}' epoch={}",
            descriptors.len(),
            if live_resize { "VMM" } else { "IPC" },
            registration.participant_id,
            participant_epoch,
        );
        Ok(setup.finish(descriptors))
    }

    fn live_resize_alignment_blocks(
        &self,
        memory_domains: &BTreeSet<KvMemoryDomain>,
        bytes_per_block: u64,
    ) -> Result<u64, KvContractError> {
        let stride = usize::try_from(bytes_per_block).map_err(|_| {
            KvContractError::invalid_capabilities(
                "CUDA VMM block stride is too large for this runtime",
            )
        })?;
        if stride == 0 || memory_domains.is_empty() {
            return Err(KvContractError::invalid_capabilities(
                "CUDA VMM alignment requires a non-zero stride and CUDA domains",
            ));
        }
        let mut alignment = None;
        for domain in memory_domains {
            let KvMemoryDomain::Cuda { device_id } = domain else {
                return Err(KvContractError::invalid_capabilities(
                    "CUDA VMM alignment accepts only CUDA domains",
                ));
            };
            let device_id = usize::try_from(*device_id).map_err(|_| {
                KvContractError::invalid_capabilities(
                    "CUDA VMM device ID does not fit this runtime",
                )
            })?;
            let device = self
                .memory
                .cuda_device(device_id)
                .map_err(|message| KvContractError::Internal { message })?;
            let granularity = GpuVmmPoolAllocation::granularity(&device)?;
            let blocks = u64::try_from(granularity / gcd(granularity, stride)).map_err(|_| {
                KvContractError::invalid_capabilities("CUDA VMM block alignment exceeds uint64")
            })?;
            if alignment
                .replace(blocks)
                .is_some_and(|current| current != blocks)
            {
                return Err(KvContractError::invalid_capabilities(
                    "tensor-parallel CUDA devices require different VMM block alignment",
                ));
            }
        }
        alignment.ok_or_else(|| {
            KvContractError::invalid_capabilities("CUDA VMM alignment found no device")
        })
    }
}

fn validate_precharged_lease(
    planned: &[PlannedBinding],
    owner: MemoryOwner,
    lease: &MemoryLease,
    require_full_allocation: bool,
) -> Result<HashMap<MemoryDomain, usize>, KvContractError> {
    let mut expected = HashMap::<MemoryDomain, usize>::new();
    for binding in planned {
        let bytes = expected
            .entry(MemoryDomain::Cuda {
                device_id: binding.device_id,
            })
            .or_default();
        *bytes = bytes.checked_add(binding.allocation_bytes).ok_or_else(|| {
            KvContractError::Internal {
                message: "precharged CUDA IPC binding bytes overflowed".to_string(),
            }
        })?;
    }
    let mut actual = HashMap::<MemoryDomain, usize>::new();
    for claim in lease.claims() {
        if claim.owner != owner
            || claim.class != MemoryAllocationClass::KvCache
            || !matches!(claim.source, MemoryClaimSource::External { .. })
            || !matches!(claim.domain, MemoryDomain::Cuda { .. })
        {
            return Err(KvContractError::invalid_capabilities(
                "precharged CUDA IPC lease contains a claim outside its exact KV scope",
            ));
        }
        let bytes = actual.entry(claim.domain.clone()).or_default();
        *bytes = bytes
            .checked_add(claim.bytes)
            .ok_or_else(|| KvContractError::Internal {
                message: "precharged CUDA IPC lease bytes overflowed".to_string(),
            })?;
    }
    let exact_domains =
        actual.len() == expected.len() && actual.keys().all(|domain| expected.contains_key(domain));
    let bounded = actual.iter().all(|(domain, bytes)| {
        *bytes > 0 && expected.get(domain).is_some_and(|maximum| bytes <= maximum)
    });
    if !exact_domains || !bounded || (require_full_allocation && actual != expected) {
        return Err(KvContractError::invalid_capabilities(
            if require_full_allocation {
                "precharged CUDA IPC lease does not exactly match planned bindings"
            } else {
                "precharged CUDA VMM lease is outside the planned device or virtual-capacity bounds"
            },
        ));
    }
    Ok(actual)
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_hal::device::DeviceInfo;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Weak;

    struct RetainedBacking {
        memory: Weak<MemoryAuthority>,
        fail: AtomicBool,
        charges_at_release: Mutex<Vec<usize>>,
    }

    fn committed(memory: &MemoryAuthority) -> usize {
        memory
            .snapshot()
            .rows
            .iter()
            .map(|row| row.committed_bytes)
            .sum()
    }

    impl SharedPoolBacking for RetainedBacking {
        fn zero_blocks(
            &self,
            _: &KvSharedPoolDescriptor,
            _: &[u64],
        ) -> Result<(), KvContractError> {
            Ok(())
        }

        fn release_after_fence(&self) -> Result<(), KvContractError> {
            self.charges_at_release
                .lock()
                .push(committed(&self.memory.upgrade().unwrap()));
            if self.fail.load(Ordering::Acquire) {
                Err(KvContractError::Internal {
                    message: "injected cleanup failure".into(),
                })
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn failed_setup_keeps_its_grant_until_a_successful_cleanup_retry() {
        let memory = MemoryAuthority::new(&DeviceInfo {
            cpu_cores: 1,
            total_memory: 1024 * 1024,
            os_type: "test".into(),
            os_release: "test".into(),
            has_cuda: false,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: Vec::new(),
        })
        .unwrap();
        let mut plan = MemoryPlan::new();
        plan.push(MemoryClaim::external(
            MemoryDomain::Host,
            MemoryOwner::new(9, 1),
            MemoryAllocationClass::KvCache,
            "failed-gpu-setup",
            4096,
        ));
        let mut grant = memory.admit(&plan).unwrap();
        grant.commit_capacity();
        let backing = Arc::new(RetainedBacking {
            memory: Arc::downgrade(&memory),
            fail: AtomicBool::new(true),
            charges_at_release: Mutex::new(Vec::new()),
        });
        let provisioner = GpuSharedPoolProvisioner::new(memory.clone());
        provisioner.failed_setup.lock().push(FailedGpuSetup {
            backing: Some(backing.clone()),
            grant: Some(grant),
        });

        provisioner.retry_failed_setup();
        assert_eq!(provisioner.failed_setup.lock().len(), 1);
        assert_eq!(committed(&memory), 4096);

        backing.fail.store(false, Ordering::Release);
        provisioner.retry_failed_setup();
        assert!(provisioner.failed_setup.lock().is_empty());
        assert_eq!(committed(&memory), 0);
        assert_eq!(*backing.charges_at_release.lock(), vec![4096, 4096]);
    }
}
