//! NCCL Communication Backend
//!
//! Provides real GPU-based distributed communication using NVIDIA NCCL
//! via the cudarc crate. This module is only available when the `nccl`
//! feature is enabled.
//!
//! # Requirements
//! - NVIDIA GPU with CUDA support
//! - CUDA toolkit installed
//! - NCCL library installed

use crate::device_mesh::{DType, MeshComm, ReduceOp};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut};
use cudarc::nccl::{Comm, Id, ReduceOp as NcclReduceOp};
use std::sync::Arc;

/// NCCL-based communication backend for real GPU collective operations.
#[derive(Debug)]
pub struct NcclComm {
    /// The NCCL communicator
    comm: Comm,
    /// CUDA device for memory operations
    device: Arc<CudaDevice>,
    /// This rank's ID
    pub rank: usize,
    /// Total number of ranks
    pub world_size: usize,
}

impl NcclComm {
    /// Create a new NCCL communicator
    ///
    /// # Arguments
    /// * `device` - The CUDA device to use
    /// * `nccl_id` - Unique NCCL ID (must be same across all ranks)
    /// * `rank` - This process's rank
    /// * `world_size` - Total number of processes
    ///
    /// # Example
    /// ```ignore
    /// // On rank 0: generate ID and broadcast to other ranks
    /// let id = NcclComm::generate_id();
    /// // ... broadcast id bytes to other ranks ...
    ///
    /// // On all ranks: create communicator
    /// let comm = NcclComm::new(device, &id, rank, world_size)?;
    /// ```
    pub fn new(
        device: Arc<CudaDevice>,
        nccl_id: &Id,
        rank: usize,
        world_size: usize,
    ) -> Result<Self, String> {
        let comm = Comm::from_rank(device.clone(), rank, world_size, *nccl_id)
            .map_err(|e| format!("Failed to create NCCL communicator: {:?}", e))?;

        Ok(Self {
            comm,
            device,
            rank,
            world_size,
        })
    }

    /// Generate a unique NCCL ID
    ///
    /// This should be called on rank 0 and the resulting ID should be
    /// broadcast to all other ranks before calling `new()`.
    pub fn generate_id() -> Result<Id, String> {
        Id::new().map_err(|e| format!("Failed to generate NCCL ID: {:?}", e))
    }

    /// Convert our DType to NCCL type info
    fn dtype_size(dtype: DType) -> usize {
        match dtype {
            DType::Float16 => 2,
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int8 => 1,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::UInt8 => 1,
            DType::BFloat16 => 2,
        }
    }

    /// Convert our ReduceOp to NCCL ReduceOp
    fn to_nccl_op(op: ReduceOp) -> NcclReduceOp {
        match op {
            ReduceOp::Sum => NcclReduceOp::Sum,
            ReduceOp::Product => NcclReduceOp::Prod,
            ReduceOp::Min => NcclReduceOp::Min,
            ReduceOp::Max => NcclReduceOp::Max,
            ReduceOp::Average => NcclReduceOp::Avg,
        }
    }

    /// Helper to copy host buffer to device
    fn host_to_device(&self, data: &[u8]) -> Result<CudaSlice<u8>, String> {
        self.device
            .htod_sync_copy(data)
            .map_err(|e| format!("Failed to copy to device: {:?}", e))
    }

    /// Helper to copy device buffer to host
    fn device_to_host(
        &self,
        device_buf: &CudaSlice<u8>,
        host_buf: &mut [u8],
    ) -> Result<(), String> {
        self.device
            .dtoh_sync_copy_into(device_buf, host_buf)
            .map_err(|e| format!("Failed to copy to host: {:?}", e))
    }
}

impl MeshComm for NcclComm {
    fn all_reduce(
        &self,
        buf: &mut [u8],
        dtype: DType,
        op: ReduceOp,
        _group: &str,
    ) -> Result<(), String> {
        let elem_size = Self::dtype_size(dtype);
        let count = buf.len() / elem_size;

        // Copy to device
        let mut device_buf = self.host_to_device(buf)?;

        // Perform all-reduce based on dtype
        match dtype {
            DType::Float32 => {
                // Reinterpret as f32 slice
                let f32_count = buf.len() / 4;
                unsafe {
                    // Create a view of the device buffer as f32
                    let send_ptr = *device_buf.device_ptr() as *const f32;
                    let recv_ptr = *device_buf.device_ptr_mut() as *mut f32;

                    self.comm
                        .all_reduce(send_ptr, recv_ptr, f32_count, Self::to_nccl_op(op))
                        .map_err(|e| format!("NCCL all_reduce failed: {:?}", e))?;
                }
            }
            DType::Float64 => {
                let f64_count = buf.len() / 8;
                unsafe {
                    let send_ptr = *device_buf.device_ptr() as *const f64;
                    let recv_ptr = *device_buf.device_ptr_mut() as *mut f64;

                    self.comm
                        .all_reduce(send_ptr, recv_ptr, f64_count, Self::to_nccl_op(op))
                        .map_err(|e| format!("NCCL all_reduce failed: {:?}", e))?;
                }
            }
            DType::Int32 => {
                let i32_count = buf.len() / 4;
                unsafe {
                    let send_ptr = *device_buf.device_ptr() as *const i32;
                    let recv_ptr = *device_buf.device_ptr_mut() as *mut i32;

                    self.comm
                        .all_reduce(send_ptr, recv_ptr, i32_count, Self::to_nccl_op(op))
                        .map_err(|e| format!("NCCL all_reduce failed: {:?}", e))?;
                }
            }
            _ => {
                return Err(format!("Unsupported dtype for all_reduce: {:?}", dtype));
            }
        }

        // Sync and copy back to host
        self.device
            .synchronize()
            .map_err(|e| format!("Sync failed: {:?}", e))?;
        self.device_to_host(&device_buf, buf)?;

        Ok(())
    }

    fn all_gather(
        &self,
        local: &[u8],
        out: &mut [u8],
        dtype: DType,
        _group: &str,
    ) -> Result<(), String> {
        let elem_size = Self::dtype_size(dtype);
        let local_count = local.len() / elem_size;

        // Copy local to device
        let device_local = self.host_to_device(local)?;
        let mut device_out = self.host_to_device(out)?;

        match dtype {
            DType::Float32 => unsafe {
                let send_ptr = *device_local.device_ptr() as *const f32;
                let recv_ptr = *device_out.device_ptr_mut() as *mut f32;

                self.comm
                    .all_gather(send_ptr, recv_ptr, local_count)
                    .map_err(|e| format!("NCCL all_gather failed: {:?}", e))?;
            },
            _ => {
                return Err(format!("Unsupported dtype for all_gather: {:?}", dtype));
            }
        }

        self.device
            .synchronize()
            .map_err(|e| format!("Sync failed: {:?}", e))?;
        self.device_to_host(&device_out, out)?;

        Ok(())
    }

    fn broadcast(&self, buf: &mut [u8], root_rank: usize, _group: &str) -> Result<(), String> {
        // Copy to device
        let mut device_buf = self.host_to_device(buf)?;
        let count = buf.len();

        unsafe {
            let ptr = *device_buf.device_ptr_mut() as *mut u8;
            self.comm
                .broadcast(ptr, ptr, count, root_rank)
                .map_err(|e| format!("NCCL broadcast failed: {:?}", e))?;
        }

        self.device
            .synchronize()
            .map_err(|e| format!("Sync failed: {:?}", e))?;
        self.device_to_host(&device_buf, buf)?;

        Ok(())
    }

    fn reduce_scatter(
        &self,
        buf: &mut [u8],
        out: &mut [u8],
        op: ReduceOp,
        _group: &str,
    ) -> Result<(), String> {
        let device_in = self.host_to_device(buf)?;
        let mut device_out = self.host_to_device(out)?;
        let out_count = out.len() / 4; // Assuming f32

        unsafe {
            let send_ptr = *device_in.device_ptr() as *const f32;
            let recv_ptr = *device_out.device_ptr_mut() as *mut f32;

            self.comm
                .reduce_scatter(send_ptr, recv_ptr, out_count, Self::to_nccl_op(op))
                .map_err(|e| format!("NCCL reduce_scatter failed: {:?}", e))?;
        }

        self.device
            .synchronize()
            .map_err(|e| format!("Sync failed: {:?}", e))?;
        self.device_to_host(&device_out, out)?;

        Ok(())
    }

    fn barrier(&self, _group: &str) -> Result<(), String> {
        // NCCL doesn't have a native barrier, so we use a zero-byte all-reduce
        // This is a common pattern for NCCL barriers
        let dummy: [f32; 1] = [0.0];
        let device_dummy = self
            .device
            .htod_sync_copy(&dummy)
            .map_err(|e| format!("Barrier failed: {:?}", e))?;

        unsafe {
            let ptr = *device_dummy.device_ptr() as *const f32;
            let mut_ptr = *device_dummy.device_ptr() as *mut f32;
            self.comm
                .all_reduce(ptr, mut_ptr, 1, NcclReduceOp::Sum)
                .map_err(|e| format!("Barrier all_reduce failed: {:?}", e))?;
        }

        self.device
            .synchronize()
            .map_err(|e| format!("Barrier sync failed: {:?}", e))?;
        Ok(())
    }

    fn send(&self, buf: &[u8], dest_rank: usize) -> Result<(), String> {
        let device_buf = self.host_to_device(buf)?;

        unsafe {
            let ptr = *device_buf.device_ptr() as *const u8;
            self.comm
                .send(ptr, buf.len(), dest_rank)
                .map_err(|e| format!("NCCL send failed: {:?}", e))?;
        }

        self.device
            .synchronize()
            .map_err(|e| format!("Send sync failed: {:?}", e))?;
        Ok(())
    }

    fn recv(&self, buf: &mut [u8], src_rank: usize) -> Result<(), String> {
        let mut device_buf = self.host_to_device(buf)?;

        unsafe {
            let ptr = *device_buf.device_ptr_mut() as *mut u8;
            self.comm
                .recv(ptr, buf.len(), src_rank)
                .map_err(|e| format!("NCCL recv failed: {:?}", e))?;
        }

        self.device
            .synchronize()
            .map_err(|e| format!("Recv sync failed: {:?}", e))?;
        self.device_to_host(&device_buf, buf)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(NcclComm::dtype_size(DType::Float32), 4);
        assert_eq!(NcclComm::dtype_size(DType::Float64), 8);
        assert_eq!(NcclComm::dtype_size(DType::Int32), 4);
        assert_eq!(NcclComm::dtype_size(DType::Float16), 2);
        assert_eq!(NcclComm::dtype_size(DType::UInt8), 1);
    }

    #[test]
    fn test_reduce_op_conversion() {
        // Just test that conversion doesn't panic
        let _ = NcclComm::to_nccl_op(ReduceOp::Sum);
        let _ = NcclComm::to_nccl_op(ReduceOp::Product);
        let _ = NcclComm::to_nccl_op(ReduceOp::Min);
        let _ = NcclComm::to_nccl_op(ReduceOp::Max);
        let _ = NcclComm::to_nccl_op(ReduceOp::Average);
    }

    // Note: Real NCCL tests require multiple GPUs and multi-process setup
    // These would typically be integration tests run separately
}
