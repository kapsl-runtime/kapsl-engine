//! Isolated CUDA IPC backing for out-of-process shared KV participants.
//!
//! These allocations deliberately bypass the runtime's general CUDA
//! suballocator. CUDA IPC exports the entire allocation, so exporting a shared
//! allocator slab would expose unrelated models and sessions to the importer.

use super::control::{ProvisionedSharedPools, SharedPoolBacking, SharedPoolProvisioner};
use crate::runtime::memory::{
    MemoryAllocationClass, MemoryAuthority, MemoryClaim, MemoryDomain, MemoryLease, MemoryOwner,
    MemoryPlan,
};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use cudarc::driver::{result as cuda_result, sys as cuda_sys, CudaDevice};
use kapsl_kv_abi::{
    KvContractError, KvFeature, KvMemoryDomain, KvParticipantRegistration,
    KvSharedPoolAllocationMode, KvSharedPoolDescriptor, KvTransport,
};
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

pub(crate) struct CudaIpcSharedPoolProvisioner {
    memory: Arc<MemoryAuthority>,
}

impl CudaIpcSharedPoolProvisioner {
    pub(crate) fn new(memory: Arc<MemoryAuthority>) -> Arc<Self> {
        Arc::new(Self { memory })
    }
}

#[derive(Debug)]
struct PlannedBinding {
    descriptor: KvSharedPoolDescriptor,
    device_id: usize,
    allocation_bytes: usize,
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
    if !registration
        .capabilities
        .transports
        .contains(&KvTransport::CudaIpc)
    {
        return Err(KvContractError::invalid_capabilities(
            "Linux CUDA IPC provisioner requires the cuda_ipc transport",
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
                "cuda-ipc:{}:{}:{}:{}",
                registration.participant_id, participant_epoch, pool_id, device_id
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
                    transport: KvTransport::CudaIpc,
                    descriptor: String::new(),
                },
                device_id: runtime_device_id,
                allocation_bytes,
            });
        }
    }
    Ok(planned)
}

struct CudaIpcAllocation {
    device: Arc<CudaDevice>,
    pointer: cuda_sys::CUdeviceptr,
    bytes: usize,
    operations: Mutex<()>,
}

impl CudaIpcAllocation {
    fn allocate(device: Arc<CudaDevice>, bytes: usize) -> Result<Self, KvContractError> {
        device
            .bind_to_thread()
            .map_err(cuda_internal("bind CUDA IPC allocation context"))?;
        // Legacy synchronous allocations are intentional: CUDA IPC memory
        // handles are not guaranteed for stream-ordered cudaMallocAsync memory.
        let pointer = unsafe { cuda_result::malloc_sync(bytes) }
            .map_err(cuda_internal("allocate isolated CUDA IPC KV region"))?;
        if let Err(error) = unsafe { cuda_result::memset_d8_sync(pointer, 0, bytes) } {
            let _ = unsafe { cuda_result::free_sync(pointer) };
            return Err(cuda_internal("zero isolated CUDA IPC KV region")(error));
        }
        Ok(Self {
            device,
            pointer,
            bytes,
            operations: Mutex::new(()),
        })
    }

    fn export_handle(&self) -> Result<String, KvContractError> {
        self.device
            .bind_to_thread()
            .map_err(cuda_internal("bind CUDA IPC export context"))?;
        let mut handle = cuda_sys::CUipcMemHandle::default();
        unsafe {
            cuda_sys::lib()
                .cuIpcGetMemHandle(&mut handle, self.pointer)
                .result()
        }
        .map_err(cuda_internal("export CUDA IPC memory handle"))?;
        let bytes = unsafe {
            std::slice::from_raw_parts(handle.reserved.as_ptr().cast::<u8>(), handle.reserved.len())
        };
        Ok(BASE64.encode(bytes))
    }

    fn zero_blocks(
        &self,
        bytes_per_block: u64,
        block_indices: &[u64],
    ) -> Result<(), KvContractError> {
        let bytes_per_block = usize::try_from(bytes_per_block).map_err(|_| {
            KvContractError::invalid_capabilities(
                "CUDA IPC block stride does not fit the runtime address space",
            )
        })?;
        let _operations = self.operations.lock();
        self.device
            .bind_to_thread()
            .map_err(cuda_internal("bind CUDA IPC zeroing context"))?;
        for &block_index in block_indices {
            let block_index = usize::try_from(block_index).map_err(|_| {
                KvContractError::invalid_request("CUDA IPC block index is too large")
            })?;
            let offset = block_index.checked_mul(bytes_per_block).ok_or_else(|| {
                KvContractError::invalid_request("CUDA IPC block offset overflows")
            })?;
            let end = offset.checked_add(bytes_per_block).ok_or_else(|| {
                KvContractError::invalid_request("CUDA IPC block range overflows")
            })?;
            if end > self.bytes {
                return Err(KvContractError::invalid_request(
                    "CUDA IPC block index is outside the exported allocation",
                ));
            }
            let pointer = self.pointer.checked_add(offset as u64).ok_or_else(|| {
                KvContractError::invalid_request("CUDA IPC device pointer overflows")
            })?;
            unsafe { cuda_result::memset_d8_sync(pointer, 0, bytes_per_block) }
                .map_err(cuda_internal("zero leased CUDA IPC KV block"))?;
        }
        Ok(())
    }
}

impl Drop for CudaIpcAllocation {
    fn drop(&mut self) {
        let _operations = self.operations.lock();
        if let Err(error) = self.device.bind_to_thread() {
            log::error!("[kv-control] failed to bind CUDA IPC context during drop: {error}");
            return;
        }
        if let Err(error) = self.device.synchronize() {
            log::error!("[kv-control] failed to synchronize CUDA IPC allocation: {error}");
        }
        if let Err(error) = unsafe { cuda_result::free_sync(self.pointer) } {
            log::error!("[kv-control] failed to free CUDA IPC allocation: {error}");
        }
    }
}

struct CudaIpcSharedPoolBacking {
    allocations: HashMap<String, CudaIpcAllocation>,
    // The memory authority lease must outlive every physical allocation. The
    // mutex provides Sync without exposing mutation after construction.
    _memory_lease: Mutex<MemoryLease>,
}

impl SharedPoolBacking for CudaIpcSharedPoolBacking {
    fn zero_blocks(
        &self,
        binding: &KvSharedPoolDescriptor,
        block_indices: &[u64],
    ) -> Result<(), KvContractError> {
        let allocation =
            self.allocations
                .get(&binding.binding_id)
                .ok_or_else(|| KvContractError::Internal {
                    message: format!(
                        "CUDA IPC backing has no allocation for binding '{}'",
                        binding.binding_id
                    ),
                })?;
        allocation.zero_blocks(binding.bytes_per_block, block_indices)
    }
}

impl SharedPoolProvisioner for CudaIpcSharedPoolProvisioner {
    fn provision(
        &self,
        registration: &KvParticipantRegistration,
        owner: MemoryOwner,
        participant_epoch: u64,
    ) -> Result<ProvisionedSharedPools, KvContractError> {
        let planned = plan_bindings(registration, participant_epoch)?;
        let mut memory_plan = MemoryPlan::new();
        for binding in &planned {
            // `external` means outside the general CUDA suballocator here; the
            // runtime still owns and frees the physical allocation below.
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
        let mut memory_lease = self.memory.admit(&memory_plan).map_err(|message| {
            KvContractError::CapacityExhausted {
                message: format!("CUDA IPC shared-pool admission failed: {message}"),
            }
        })?;

        let mut descriptors = Vec::with_capacity(planned.len());
        let mut allocations = HashMap::with_capacity(planned.len());
        for binding in planned {
            let device = self
                .memory
                .cuda_device(binding.device_id)
                .map_err(|message| KvContractError::Internal { message })?;
            let allocation = CudaIpcAllocation::allocate(device, binding.allocation_bytes)?;
            let mut descriptor = binding.descriptor;
            descriptor.descriptor = allocation.export_handle()?;
            allocations.insert(descriptor.binding_id.clone(), allocation);
            descriptors.push(descriptor);
        }
        memory_lease.commit_capacity();
        log::info!(
            "[kv-control] provisioned {} isolated CUDA IPC KV binding(s) for participant '{}' epoch={}",
            descriptors.len(),
            registration.participant_id,
            participant_epoch,
        );
        Ok(ProvisionedSharedPools {
            descriptors,
            backing: Arc::new(CudaIpcSharedPoolBacking {
                allocations,
                _memory_lease: Mutex::new(memory_lease),
            }),
        })
    }
}

fn cuda_internal(
    operation: &'static str,
) -> impl FnOnce(cuda_result::DriverError) -> KvContractError {
    move |error| KvContractError::Internal {
        message: format!("failed to {operation}: {error}"),
    }
}
