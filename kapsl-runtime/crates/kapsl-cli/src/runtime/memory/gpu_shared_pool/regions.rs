//! KV descriptor adapters over backend-neutral HAL physical regions.

use super::*;
use crate::runtime::memory::gpu_pool::{
    ExportCapacity, GpuDevicePool, GpuRegionLease, GpuRegionRequest,
};
use kapsl_hal::gpu_ipc_region::GpuIpcRegion;
use kapsl_hal::gpu_region::{GpuRegion, GpuRegionError};
use kapsl_hal::gpu_vmm_region::{GpuVmmRegion, GpuVmmSegment};

fn region_error(message: String) -> KvContractError {
    KvContractError::Internal { message }
}

pub(super) fn hal_error(error: GpuRegionError) -> KvContractError {
    KvContractError::Internal {
        message: error.to_string(),
    }
}

pub(super) struct GpuIpcPoolAllocation {
    region: GpuRegionLease,
}

impl GpuIpcPoolAllocation {
    /// Return owned backing before initialization so setup failure can retain
    /// the allocation alongside its authority grant.
    pub(super) fn allocate(
        pool: Arc<GpuDevicePool>,
        isolation: GpuRegionIsolation,
        bytes: usize,
    ) -> Result<Self, KvContractError> {
        Ok(Self {
            region: pool
                .acquire_region(GpuRegionRequest::Exported {
                    isolation,
                    capacity: ExportCapacity::Fixed { bytes },
                })
                .map_err(region_error)?,
        })
    }

    fn backing(&self) -> Result<&GpuIpcRegion, KvContractError> {
        self.region.ipc().map_err(region_error)
    }

    pub(super) fn initialize(&self) -> Result<(), KvContractError> {
        self.backing()?.initialize_zeroed().map_err(hal_error)
    }

    pub(super) fn export_handle(&self) -> Result<String, KvContractError> {
        self.backing()?
            .export_handle()
            .map(|handle| BASE64.encode(handle.as_bytes()))
            .map_err(hal_error)
    }

    pub(super) fn release_after_fence(&self) -> Result<(), KvContractError> {
        // SAFETY: the coordinator retires every importer before invoking the
        // backing release contract. During failed setup no handle was sent.
        unsafe { self.region.release_export_after_fence() }.map_err(region_error)
    }

    pub(super) fn zero_blocks(
        &self,
        bytes_per_block: u64,
        block_indices: &[u64],
    ) -> Result<(), KvContractError> {
        let stride = usize::try_from(bytes_per_block)
            .map_err(|_| KvContractError::invalid_request("IPC block stride is too large"))?;
        let capacity = self.backing()?.physical_snapshot().committed_bytes;
        // Validate the complete request before clearing any block.
        let offsets = block_indices
            .iter()
            .map(|index| {
                usize::try_from(*index)
                    .ok()
                    .and_then(|index| index.checked_mul(stride))
                    .filter(|offset| {
                        stride != 0
                            && offset
                                .checked_add(stride)
                                .is_some_and(|end| end <= capacity)
                    })
                    .ok_or_else(|| KvContractError::invalid_request("IPC block is outside backing"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for offset in offsets {
            // SAFETY: runtime-leased blocks are selected only from the fenced
            // free list. HAL waits for clearing before the lease is published.
            unsafe { self.backing()?.zero_range_after_fence(offset, stride) }.map_err(hal_error)?;
        }
        Ok(())
    }
}

pub(super) struct GpuVmmPoolAllocation {
    region: GpuRegionLease,
    minimum_bytes: usize,
    pub(super) granularity: usize,
    descriptors: Mutex<HashMap<String, GpuVmmSegment>>,
}

impl GpuVmmPoolAllocation {
    pub(super) fn granularity(device: &Arc<CudaDevice>) -> Result<usize, KvContractError> {
        GpuVmmRegion::allocation_granularity(device).map_err(hal_error)
    }

    pub(super) fn reserve(
        pool: Arc<GpuDevicePool>,
        isolation: GpuRegionIsolation,
        virtual_bytes: usize,
        minimum_bytes: usize,
    ) -> Result<Self, KvContractError> {
        let granularity = Self::granularity(pool.device())?;
        if minimum_bytes == 0
            || minimum_bytes > virtual_bytes
            || !minimum_bytes.is_multiple_of(granularity)
        {
            return Err(KvContractError::invalid_capabilities(
                "CUDA VMM minimum must be nonzero, aligned and within virtual capacity",
            ));
        }
        let region = pool
            .acquire_region(GpuRegionRequest::Exported {
                isolation,
                capacity: ExportCapacity::Elastic {
                    maximum_bytes: virtual_bytes,
                },
            })
            .map_err(region_error)?;
        Ok(Self {
            region,
            minimum_bytes,
            granularity,
            descriptors: Mutex::new(HashMap::new()),
        })
    }

    fn backing(&self) -> Result<&GpuVmmRegion, KvContractError> {
        self.region.vmm().map_err(region_error)
    }

    pub(super) fn initialize(
        &self,
        initial_bytes: usize,
        prefix: &str,
    ) -> Result<(), KvContractError> {
        if initial_bytes < self.minimum_bytes
            || initial_bytes > self.backing()?.physical_snapshot().virtual_reserved_bytes
            || !initial_bytes.is_multiple_of(self.granularity)
        {
            return Err(KvContractError::invalid_capabilities(
                "invalid initial VMM capacity",
            ));
        }
        self.grow_to(self.minimum_bytes, format!("{prefix}:minimum"))?;
        if initial_bytes > self.minimum_bytes {
            self.grow_to(initial_bytes, format!("{prefix}:headroom"))?;
        }
        Ok(())
    }

    pub(super) fn grow_to(
        &self,
        target_bytes: usize,
        segment_id: String,
    ) -> Result<KvVmmSegmentDescriptor, KvContractError> {
        let mut descriptors = self.descriptors.lock();
        if descriptors.contains_key(&segment_id) {
            return Err(KvContractError::invalid_request(
                "VMM segment generation is already live",
            ));
        }
        let segment = self.backing()?.grow_to(target_bytes).map_err(hal_error)?;
        let descriptor = wire_segment(&segment_id, segment);
        descriptors.insert(segment_id, segment);
        Ok(descriptor)
    }

    pub(super) fn segments(&self) -> Vec<KvVmmSegmentDescriptor> {
        let mut result = self
            .descriptors
            .lock()
            .iter()
            .map(|(id, segment)| wire_segment(id, *segment))
            .collect::<Vec<_>>();
        result.sort_by_key(|segment| segment.offset_bytes);
        result
    }

    pub(super) fn shrink_segments(
        &self,
        target: usize,
    ) -> Result<Vec<KvVmmSegmentDescriptor>, KvContractError> {
        if target < self.minimum_bytes {
            return Err(KvContractError::invalid_request(
                "VMM shrink is below certified minimum",
            ));
        }
        let descriptors = self.descriptors.lock();
        self.backing()?
            .tail_segments(target)
            .map_err(hal_error)?
            .into_iter()
            .map(|segment| {
                descriptors
                    .iter()
                    .find(|(_, candidate)| candidate.id == segment.id)
                    .map(|(id, _)| wire_segment(id, segment))
                    .ok_or_else(|| KvContractError::invalid_request("VMM cleanup is incomplete"))
            })
            .collect()
    }

    pub(super) fn shrink_boundary(&self, requested: usize) -> Result<usize, KvContractError> {
        if requested < self.minimum_bytes {
            return Err(KvContractError::invalid_request(
                "VMM shrink is below certified minimum",
            ));
        }
        let boundary = self
            .backing()?
            .shrink_boundary(requested)
            .map_err(hal_error)?;
        if boundary < self.minimum_bytes {
            return Err(KvContractError::invalid_request(
                "VMM minimum boundary is unavailable",
            ));
        }
        Ok(boundary)
    }

    pub(super) fn release_tail(&self, target: usize) -> Result<(), KvContractError> {
        if target < self.minimum_bytes {
            return Err(KvContractError::invalid_request(
                "VMM shrink is below certified minimum",
            ));
        }
        let mut descriptors = self.descriptors.lock();
        // SAFETY: the coordinator invokes this only for an unpublished growth
        // rollback or after worker unmap/handle-close acknowledgments and fences.
        unsafe { self.backing()?.release_tail_after_fence(target) }.map_err(hal_error)?;
        descriptors.retain(|_, segment| segment.offset_bytes < target);
        Ok(())
    }

    pub(super) fn release_after_fence(&self) -> Result<(), KvContractError> {
        let mut descriptors = self.descriptors.lock();
        // SAFETY: the coordinator proved importer retirement, or setup failed
        // before descriptors were sent. All exported FDs are scoped to sends.
        unsafe { self.region.release_export_after_fence() }.map_err(region_error)?;
        descriptors.clear();
        Ok(())
    }

    pub(super) fn contains_segment(&self, id: &str) -> bool {
        self.descriptors.lock().contains_key(id)
    }

    pub(super) fn export_segment(&self, id: &str) -> Result<OwnedFd, KvContractError> {
        let descriptors = self.descriptors.lock();
        let segment = descriptors
            .get(id)
            .ok_or_else(|| KvContractError::invalid_request("unknown VMM segment generation"))?;
        self.backing()?
            .export_segment(segment.id)
            .map_err(hal_error)
    }
}

fn wire_segment(id: &str, segment: GpuVmmSegment) -> KvVmmSegmentDescriptor {
    KvVmmSegmentDescriptor {
        segment_id: id.to_owned(),
        offset_bytes: segment.offset_bytes as u64,
        length_bytes: segment.length_bytes as u64,
        handle_index: 0,
    }
}
