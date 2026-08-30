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
    KvContractError, KvElasticPoolDescriptor, KvFeature, KvMemoryDomain, KvParticipantRegistration,
    KvSharedPoolAllocationMode, KvSharedPoolDescriptor, KvTransport, KvVmmSegmentDescriptor,
};
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet, HashMap};
#[cfg(unix)]
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, Ordering};
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

struct CudaIpcAllocation {
    device: Arc<CudaDevice>,
    pointer: cuda_sys::CUdeviceptr,
    bytes: usize,
    operations: Mutex<()>,
    released: AtomicBool,
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
            released: AtomicBool::new(false),
        })
    }

    fn release_after_fence(&self) -> Result<(), KvContractError> {
        let _operations = self.operations.lock();
        if self.released.load(Ordering::Acquire) {
            return Ok(());
        }
        self.device
            .bind_to_thread()
            .map_err(cuda_internal("bind CUDA IPC release context"))?;
        self.device
            .synchronize()
            .map_err(cuda_internal("synchronize CUDA IPC release"))?;
        unsafe { cuda_result::free_sync(self.pointer) }
            .map_err(cuda_internal("free isolated CUDA IPC KV region"))?;
        self.released.store(true, Ordering::Release);
        Ok(())
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
        if self.released.load(Ordering::Acquire) {
            return Err(KvContractError::invalid_request(
                "CUDA IPC backing was already released",
            ));
        }
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
        if let Err(error) = self.release_after_fence() {
            log::error!(
                "[kv-control] failed to release CUDA IPC allocation during final drop: {error}"
            );
        }
    }
}

struct CudaVmmSegment {
    descriptor: KvVmmSegmentDescriptor,
    handle: cuda_sys::CUmemGenericAllocationHandle,
    mapped: bool,
}

struct CudaVmmState {
    segments: Vec<CudaVmmSegment>,
}

struct CudaVmmAllocation {
    device: Arc<CudaDevice>,
    pointer: cuda_sys::CUdeviceptr,
    virtual_bytes: usize,
    minimum_bytes: usize,
    granularity: usize,
    state: Mutex<CudaVmmState>,
    released: AtomicBool,
}

impl CudaVmmAllocation {
    fn allocation_properties(
        device_id: usize,
    ) -> Result<cuda_sys::CUmemAllocationProp, KvContractError> {
        let device_id = i32::try_from(device_id).map_err(|_| {
            KvContractError::invalid_capabilities("CUDA VMM device ordinal is too large")
        })?;
        Ok(cuda_sys::CUmemAllocationProp {
            type_: cuda_sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes:
                cuda_sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            location: cuda_sys::CUmemLocation {
                type_: cuda_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_id,
            },
            ..Default::default()
        })
    }

    fn allocation_granularity(device: &Arc<CudaDevice>) -> Result<usize, KvContractError> {
        device
            .bind_to_thread()
            .map_err(cuda_internal("bind CUDA VMM granularity context"))?;
        let properties = Self::allocation_properties(device.ordinal())?;
        let mut granularity = 0usize;
        unsafe {
            cuda_sys::lib().cuMemGetAllocationGranularity(
                &mut granularity,
                &properties,
                cuda_sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        }
        .result()
        .map_err(cuda_internal("query CUDA VMM allocation granularity"))?;
        if granularity == 0 {
            return Err(KvContractError::Internal {
                message: "CUDA VMM returned zero allocation granularity".to_string(),
            });
        }
        Ok(granularity)
    }

    fn allocate(
        device: Arc<CudaDevice>,
        virtual_bytes: usize,
        minimum_bytes: usize,
        initial_bytes: usize,
        segment_prefix: String,
    ) -> Result<Self, KvContractError> {
        device
            .bind_to_thread()
            .map_err(cuda_internal("bind CUDA VMM allocation context"))?;
        let granularity = Self::allocation_granularity(&device)?;
        if minimum_bytes == 0
            || minimum_bytes > initial_bytes
            || initial_bytes > virtual_bytes
            || !minimum_bytes.is_multiple_of(granularity)
            || !initial_bytes.is_multiple_of(granularity)
            || !virtual_bytes.is_multiple_of(granularity)
        {
            return Err(KvContractError::invalid_capabilities(format!(
                "CUDA VMM minimum ({minimum_bytes}), initial ({initial_bytes}), and virtual ({virtual_bytes}) bytes must be ordered and align to granularity {granularity}"
            )));
        }
        let mut pointer = 0;
        unsafe {
            cuda_sys::lib().cuMemAddressReserve(&mut pointer, virtual_bytes, granularity, 0, 0)
        }
        .result()
        .map_err(cuda_internal("reserve CUDA VMM address range"))?;
        let allocation = Self {
            device,
            pointer,
            virtual_bytes,
            minimum_bytes,
            granularity,
            state: Mutex::new(CudaVmmState {
                segments: Vec::new(),
            }),
            released: AtomicBool::new(false),
        };
        allocation.grow_to(minimum_bytes, format!("{segment_prefix}:minimum"))?;
        if initial_bytes > minimum_bytes {
            allocation.grow_to(initial_bytes, format!("{segment_prefix}:initial-headroom"))?;
        }
        Ok(allocation)
    }

    fn grow_to(
        &self,
        target_bytes: usize,
        segment_id: String,
    ) -> Result<KvVmmSegmentDescriptor, KvContractError> {
        if self.released.load(Ordering::Acquire) {
            return Err(KvContractError::invalid_request(
                "CUDA VMM backing was already released",
            ));
        }
        self.device
            .bind_to_thread()
            .map_err(cuda_internal("bind CUDA VMM growth context"))?;
        let mut state = self.state.lock();
        let current_bytes = state
            .segments
            .last()
            .map(|segment| {
                usize::try_from(segment.descriptor.offset_bytes + segment.descriptor.length_bytes)
                    .expect("validated VMM segment range fits usize")
            })
            .unwrap_or(0);
        if target_bytes <= current_bytes
            || target_bytes > self.virtual_bytes
            || !target_bytes.is_multiple_of(self.granularity)
        {
            return Err(KvContractError::invalid_request(
                "CUDA VMM growth target is outside or misaligned with virtual capacity",
            ));
        }
        let length = target_bytes - current_bytes;
        let descriptor = KvVmmSegmentDescriptor {
            segment_id,
            offset_bytes: current_bytes as u64,
            length_bytes: length as u64,
            handle_index: 0,
        };
        let properties = Self::allocation_properties(self.device.ordinal())?;
        let mut handle = 0;
        unsafe { cuda_sys::lib().cuMemCreate(&mut handle, length, &properties, 0) }
            .result()
            .map_err(cuda_internal("create CUDA VMM physical segment"))?;
        let address = self
            .pointer
            .checked_add(current_bytes as u64)
            .ok_or_else(|| KvContractError::Internal {
                message: "CUDA VMM address overflowed".to_string(),
            })?;
        if let Err(error) =
            unsafe { cuda_sys::lib().cuMemMap(address, length, 0, handle, 0) }.result()
        {
            let release = unsafe { cuda_sys::lib().cuMemRelease(handle).result() };
            if let Err(release_error) = release {
                // Retain an unmapped handle so release_binding_tail can retry.
                // The authority growth must remain charged unless that retry
                // succeeds.
                state.segments.push(CudaVmmSegment {
                    descriptor,
                    handle,
                    mapped: false,
                });
                return Err(KvContractError::Internal {
                    message: format!(
                        "failed to map CUDA VMM physical segment ({error}) and could not release its handle ({release_error})"
                    ),
                });
            }
            return Err(cuda_internal("map CUDA VMM physical segment")(error));
        }
        let access = cuda_sys::CUmemAccessDesc {
            location: properties.location,
            flags: cuda_sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        let initialized = unsafe {
            cuda_sys::lib()
                .cuMemSetAccess(address, length, &access, 1)
                .result()
                .and_then(|()| cuda_result::memset_d8_sync(address, 0, length))
        };
        if let Err(error) = initialized {
            let unmap = unsafe { cuda_sys::lib().cuMemUnmap(address, length).result() };
            match unmap {
                Ok(()) => {
                    if let Err(release_error) =
                        unsafe { cuda_sys::lib().cuMemRelease(handle).result() }
                    {
                        // The address is clean, but the physical handle still
                        // consumes capacity. Preserve it for transactional
                        // rollback instead of silently lowering accounting.
                        state.segments.push(CudaVmmSegment {
                            descriptor,
                            handle,
                            mapped: false,
                        });
                        return Err(KvContractError::Internal {
                            message: format!(
                                "failed to initialize CUDA VMM physical segment ({error}) and could not release its handle ({release_error})"
                            ),
                        });
                    }
                }
                Err(unmap_error) => {
                    // Keep both the mapping and handle represented. The
                    // coordinator will retain the authority charge unless its
                    // rollback can conclusively unmap and release this tail.
                    state.segments.push(CudaVmmSegment {
                        descriptor,
                        handle,
                        mapped: true,
                    });
                    return Err(KvContractError::Internal {
                        message: format!(
                            "failed to initialize CUDA VMM physical segment ({error}) and could not unmap it ({unmap_error})"
                        ),
                    });
                }
            }
            return Err(cuda_internal("initialize CUDA VMM physical segment")(error));
        }
        state.segments.push(CudaVmmSegment {
            descriptor: descriptor.clone(),
            handle,
            mapped: true,
        });
        Ok(descriptor)
    }

    fn shrink_segments(
        &self,
        target_bytes: usize,
    ) -> Result<Vec<KvVmmSegmentDescriptor>, KvContractError> {
        if self.released.load(Ordering::Acquire) {
            return Err(KvContractError::invalid_request(
                "CUDA VMM backing was already released",
            ));
        }
        let state = self.state.lock();
        let current = state
            .segments
            .last()
            .map(|segment| segment.descriptor.offset_bytes + segment.descriptor.length_bytes)
            .unwrap_or(0);
        if target_bytes < self.minimum_bytes
            || target_bytes as u64 >= current
            || !target_bytes.is_multiple_of(self.granularity)
        {
            return Err(KvContractError::invalid_request(
                "CUDA VMM shrink target is outside the releasable mapped tail",
            ));
        }
        let segments = state
            .segments
            .iter()
            .filter(|segment| segment.descriptor.offset_bytes >= target_bytes as u64)
            .map(|segment| segment.descriptor.clone())
            .collect::<Vec<_>>();
        if segments.is_empty()
            || segments
                .first()
                .is_none_or(|segment| segment.offset_bytes != target_bytes as u64)
        {
            return Err(KvContractError::invalid_request(
                "CUDA VMM shrink target is not a committed segment boundary",
            ));
        }
        Ok(segments)
    }

    fn shrink_boundary(&self, requested_bytes: usize) -> Result<usize, KvContractError> {
        if self.released.load(Ordering::Acquire) {
            return Err(KvContractError::invalid_request(
                "CUDA VMM backing was already released",
            ));
        }
        let state = self.state.lock();
        let current = state
            .segments
            .last()
            .map(|segment| segment.descriptor.offset_bytes + segment.descriptor.length_bytes)
            .unwrap_or(0);
        if requested_bytes < self.minimum_bytes
            || requested_bytes as u64 >= current
            || !requested_bytes.is_multiple_of(self.granularity)
        {
            return Err(KvContractError::invalid_request(
                "CUDA VMM shrink request is outside the releasable mapped tail",
            ));
        }
        state
            .segments
            .iter()
            .map(|segment| segment.descriptor.offset_bytes as usize)
            .filter(|offset| *offset >= self.minimum_bytes && *offset <= requested_bytes)
            .max()
            .ok_or_else(|| KvContractError::Internal {
                message: "CUDA VMM allocation has no certified minimum segment boundary"
                    .to_string(),
            })
    }

    fn release_tail(&self, target_bytes: usize) -> Result<(), KvContractError> {
        let mut state = self.state.lock();
        self.device
            .bind_to_thread()
            .map_err(cuda_internal("bind CUDA VMM shrink context"))?;
        self.device
            .synchronize()
            .map_err(cuda_internal("synchronize CUDA VMM shrink"))?;
        while state
            .segments
            .last()
            .is_some_and(|segment| segment.descriptor.offset_bytes >= target_bytes as u64)
        {
            let segment = state.segments.last_mut().expect("tail existence checked");
            let address = self.pointer + segment.descriptor.offset_bytes;
            let length = segment.descriptor.length_bytes as usize;
            if segment.mapped {
                unsafe { cuda_sys::lib().cuMemUnmap(address, length) }
                    .result()
                    .map_err(cuda_internal("unmap CUDA VMM tail segment"))?;
                segment.mapped = false;
            }
            unsafe { cuda_sys::lib().cuMemRelease(segment.handle) }
                .result()
                .map_err(cuda_internal("release CUDA VMM tail segment"))?;
            state.segments.pop();
        }
        let mapped = state
            .segments
            .last()
            .map(|segment| segment.descriptor.offset_bytes + segment.descriptor.length_bytes)
            .unwrap_or(0);
        if mapped != target_bytes as u64 {
            return Err(KvContractError::Internal {
                message: "CUDA VMM tail release ended at the wrong boundary".to_string(),
            });
        }
        Ok(())
    }

    fn release_after_fence(&self) -> Result<(), KvContractError> {
        if self.released.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut state = self.state.lock();
        self.device
            .bind_to_thread()
            .map_err(cuda_internal("bind CUDA VMM release context"))?;
        self.device
            .synchronize()
            .map_err(cuda_internal("synchronize CUDA VMM release"))?;
        while let Some(segment) = state.segments.last_mut() {
            if segment.mapped {
                let address = self.pointer + segment.descriptor.offset_bytes;
                unsafe {
                    cuda_sys::lib()
                        .cuMemUnmap(address, segment.descriptor.length_bytes as usize)
                        .result()
                }
                .map_err(cuda_internal("unmap CUDA VMM segment during release"))?;
                segment.mapped = false;
            }
            unsafe { cuda_sys::lib().cuMemRelease(segment.handle) }
                .result()
                .map_err(cuda_internal("release CUDA VMM physical handle"))?;
            state.segments.pop();
        }
        unsafe {
            cuda_sys::lib()
                .cuMemAddressFree(self.pointer, self.virtual_bytes)
                .result()
        }
        .map_err(cuda_internal("free CUDA VMM address range"))?;
        self.released.store(true, Ordering::Release);
        Ok(())
    }

    #[cfg(unix)]
    fn export_segment(&self, segment_id: &str) -> Result<OwnedFd, KvContractError> {
        if self.released.load(Ordering::Acquire) {
            return Err(KvContractError::invalid_request(
                "CUDA VMM backing was already released",
            ));
        }
        self.device
            .bind_to_thread()
            .map_err(cuda_internal("bind CUDA VMM export context"))?;
        let state = self.state.lock();
        let segment = state
            .segments
            .iter()
            .find(|segment| segment.descriptor.segment_id == segment_id)
            .ok_or_else(|| KvContractError::NotFound {
                message: format!("CUDA VMM segment '{segment_id}' is not live"),
            })?;
        if !segment.mapped {
            return Err(KvContractError::invalid_request(
                "CUDA VMM segment is no longer mapped",
            ));
        }
        let mut descriptor = -1i32;
        unsafe {
            cuda_sys::lib().cuMemExportToShareableHandle(
                (&mut descriptor as *mut i32).cast(),
                segment.handle,
                cuda_sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            )
        }
        .result()
        .map_err(cuda_internal("export CUDA VMM POSIX handle"))?;
        if descriptor < 0 {
            return Err(KvContractError::Internal {
                message: "CUDA VMM export returned an invalid descriptor".to_string(),
            });
        }
        Ok(unsafe { OwnedFd::from_raw_fd(descriptor) })
    }
}

impl Drop for CudaVmmAllocation {
    fn drop(&mut self) {
        if let Err(error) = self.release_after_fence() {
            log::error!(
                "[kv-control] failed to release CUDA VMM allocation during final drop: {error}"
            );
        }
    }
}

struct CudaIpcSharedPoolBacking {
    allocations: HashMap<String, CudaIpcAllocation>,
    vmm_allocations: HashMap<String, CudaVmmAllocation>,
}

impl SharedPoolBacking for CudaIpcSharedPoolBacking {
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
                    .find(|allocation| {
                        allocation
                            .state
                            .lock()
                            .segments
                            .iter()
                            .any(|candidate| candidate.descriptor.segment_id == segment.segment_id)
                    })
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

impl SharedPoolProvisioner for CudaIpcSharedPoolProvisioner {
    fn provision(
        &self,
        registration: &KvParticipantRegistration,
        owner: MemoryOwner,
        participant_epoch: u64,
        precharged: Option<MemoryLease>,
        minimum_block_count: Option<u64>,
    ) -> Result<ProvisionedSharedPools, KvContractError> {
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
        let mut allocations = HashMap::with_capacity(planned.len());
        let mut vmm_allocations = HashMap::with_capacity(planned.len());
        let mut next_handle_index = 0u32;
        for binding in planned {
            let device = self
                .memory
                .cuda_device(binding.device_id)
                .map_err(|message| KvContractError::Internal { message })?;
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
                let allocation = CudaVmmAllocation::allocate(
                    device,
                    binding.allocation_bytes,
                    minimum_bytes,
                    initial_bytes,
                    descriptor.binding_id.clone(),
                )?;
                let mut segments = allocation
                    .state
                    .lock()
                    .segments
                    .iter()
                    .map(|segment| segment.descriptor.clone())
                    .collect::<Vec<_>>();
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
                vmm_allocations.insert(descriptor.binding_id.clone(), allocation);
            } else {
                let allocation = CudaIpcAllocation::allocate(device, binding.allocation_bytes)?;
                descriptor.descriptor = allocation.export_handle()?;
                allocations.insert(descriptor.binding_id.clone(), allocation);
            }
            descriptors.push(descriptor);
        }
        log::info!(
            "[kv-control] provisioned {} isolated CUDA {} KV binding(s) for participant '{}' epoch={}",
            descriptors.len(),
            if live_resize { "VMM" } else { "IPC" },
            registration.participant_id,
            participant_epoch,
        );
        Ok(ProvisionedSharedPools {
            descriptors,
            backing: Arc::new(CudaIpcSharedPoolBacking {
                allocations,
                vmm_allocations,
            }),
            memory_lease: Some(memory_lease),
        })
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
            let granularity = CudaVmmAllocation::allocation_granularity(&device)?;
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
            || !matches!(
                claim.source,
                super::memory::MemoryClaimSource::External { .. }
            )
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

fn cuda_internal(
    operation: &'static str,
) -> impl FnOnce(cuda_result::DriverError) -> KvContractError {
    move |error| KvContractError::Internal {
        message: format!("failed to {operation}: {error}"),
    }
}
