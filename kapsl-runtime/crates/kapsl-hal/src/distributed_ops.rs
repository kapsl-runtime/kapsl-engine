//! Distributed Operations for Tensor Computations
//!
//! High-level distributed tensor operations that use the `MeshComm` trait.
//! These operations enable distributed inference across multiple devices.

use crate::device_mesh::{DType, DeviceMesh, MeshComm, ReduceOp};
use std::sync::Arc;

/// Result type for distributed operations
pub type DistResult<T> = Result<T, String>;

/// Distributed tensor operations
pub struct DistributedOps<'a> {
    mesh: &'a DeviceMesh,
}

impl<'a> DistributedOps<'a> {
    /// Create a new distributed operations context
    pub fn new(mesh: &'a DeviceMesh) -> Self {
        Self { mesh }
    }

    /// Get the communication backend
    fn comm(&self) -> DistResult<&Arc<dyn MeshComm + Send + Sync>> {
        self.mesh
            .comm
            .as_ref()
            .ok_or_else(|| "No communication backend configured".to_string())
    }

    /// All-reduce a f32 tensor across all ranks in a group
    ///
    /// After this operation, all ranks will have the same reduced values.
    pub fn all_reduce_f32(&self, data: &mut [f32], op: ReduceOp, group: &str) -> DistResult<()> {
        let comm = self.comm()?;

        // Convert f32 slice to bytes for the communication layer
        let byte_slice =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) };

        comm.all_reduce(byte_slice, DType::Float32, op, group)
    }

    /// All-reduce a i32 tensor across all ranks in a group
    pub fn all_reduce_i32(&self, data: &mut [i32], op: ReduceOp, group: &str) -> DistResult<()> {
        let comm = self.comm()?;

        let byte_slice =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) };

        comm.all_reduce(byte_slice, DType::Int32, op, group)
    }

    /// Broadcast f32 tensor from root rank to all ranks in group
    pub fn broadcast_f32(&self, data: &mut [f32], root_rank: usize, group: &str) -> DistResult<()> {
        let comm = self.comm()?;

        let byte_slice =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) };

        comm.broadcast(byte_slice, root_rank, group)
    }

    /// All-gather f32 tensors from all ranks
    ///
    /// Each rank contributes `local.len()` elements, and receives
    /// `local.len() * world_size` elements in the output.
    pub fn all_gather_f32(&self, local: &[f32], output: &mut [f32], group: &str) -> DistResult<()> {
        let comm = self.comm()?;

        let local_bytes =
            unsafe { std::slice::from_raw_parts(local.as_ptr() as *const u8, local.len() * 4) };

        let output_bytes = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u8, output.len() * 4)
        };

        comm.all_gather(local_bytes, output_bytes, DType::Float32, group)
    }

    /// Scatter a tensor: divide data among ranks
    ///
    /// Only the root rank's `data` is used for input. After this operation,
    /// each rank's `chunk` will contain its portion of the data.
    pub fn scatter_f32(&self, data: &[f32], chunk: &mut [f32], root_rank: usize) -> DistResult<()> {
        let comm = self.comm()?;

        // Root broadcasts, then each rank picks its chunk
        if self.mesh.rank == root_rank {
            // Convert to bytes for broadcast
            let data_bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };

            // Broadcast the entire data
            let mut broadcast_buf = data_bytes.to_vec();
            comm.broadcast(&mut broadcast_buf, root_rank, "world")?;

            // Extract our chunk
            let chunk_size = chunk.len();
            let offset = self.mesh.rank * chunk_size;
            for (i, val) in chunk.iter_mut().enumerate() {
                let src_idx = offset + i;
                if src_idx < data.len() {
                    *val = data[src_idx];
                }
            }
        } else {
            // Non-root receives broadcast and extracts chunk
            let total_size = data.len();
            let mut broadcast_buf = vec![0u8; total_size * 4];
            comm.broadcast(&mut broadcast_buf, root_rank, "world")?;

            // Convert bytes back to f32 and extract our chunk
            let chunk_size = chunk.len();
            let offset = self.mesh.rank * chunk_size;
            for (i, val) in chunk.iter_mut().enumerate() {
                let idx = (offset + i) * 4;
                if idx + 4 <= broadcast_buf.len() {
                    let bytes: [u8; 4] = broadcast_buf[idx..idx + 4].try_into().unwrap();
                    *val = f32::from_le_bytes(bytes);
                }
            }
        }

        Ok(())
    }

    /// Gather tensors from all ranks to root
    ///
    /// Each rank's `local` data is gathered to the root rank's `output`.
    /// Only the root rank's output will contain the complete gathered data.
    pub fn gather_f32(
        &self,
        local: &[f32],
        output: &mut [f32],
        root_rank: usize,
    ) -> DistResult<()> {
        // Use all_gather first, then root takes the result
        self.all_gather_f32(local, output, "world")?;

        // For non-root ranks, output is partially filled but that's okay
        // since only root is expected to use the complete result
        if self.mesh.rank != root_rank {
            // Clear output for non-root ranks (optional, for clarity)
            // In practice, callers should only use root's output
        }

        Ok(())
    }

    /// Reduce-scatter: reduce and distribute results
    ///
    /// Combines reduction and scatter in one operation.
    /// After this operation, each rank has a portion of the reduced result.
    pub fn reduce_scatter_f32(
        &self,
        data: &mut [f32],
        output: &mut [f32],
        op: ReduceOp,
        group: &str,
    ) -> DistResult<()> {
        let comm = self.comm()?;

        let data_bytes =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) };

        let output_bytes = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u8, output.len() * 4)
        };

        comm.reduce_scatter(data_bytes, output_bytes, op, group)
    }

    /// Barrier synchronization across all ranks in a group
    pub fn barrier(&self, group: &str) -> DistResult<()> {
        let comm = self.comm()?;
        comm.barrier(group)
    }

    /// Point-to-point send of f32 tensor
    pub fn send_f32(&self, data: &[f32], dest_rank: usize) -> DistResult<()> {
        let comm = self.comm()?;
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        comm.send(bytes, dest_rank)
    }

    /// Point-to-point receive of f32 tensor
    pub fn recv_f32(&self, data: &mut [f32], src_rank: usize) -> DistResult<()> {
        let comm = self.comm()?;
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) };
        comm.recv(bytes, src_rank)
    }
}

/// Convenience function to create distributed ops from a mesh
pub fn dist_ops(mesh: &DeviceMesh) -> DistributedOps<'_> {
    DistributedOps::new(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, DeviceBackend};
    use crate::device_mesh::DeviceMesh;

    fn create_test_devices(count: usize) -> Vec<Device> {
        (0..count)
            .map(|i| Device {
                id: i,
                name: format!("GPU_{}", i),
                backend: DeviceBackend::Cuda,
                memory_mb: 16000,
                compute_units: 80,
                pci_bus_id: None,
                driver_version: None,
                compute_capability: None,
                utilization_gpu_pct: None,
                temperature_c: None,
                supports_fp16: true,
                supports_int8: true,
                cuda_version: Some("12.0".to_string()),
            })
            .collect()
    }

    #[test]
    fn test_all_reduce_f32_single_rank() {
        let devices = create_test_devices(1);
        let mesh = DeviceMesh::new_with_mock_comm(devices, 0);

        let ops = dist_ops(&mesh);
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];

        let result = ops.all_reduce_f32(&mut data, ReduceOp::Sum, "world");
        assert!(result.is_ok());

        // Single rank, data unchanged
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_broadcast_f32() {
        let devices = create_test_devices(2);
        let mesh = DeviceMesh::new_with_mock_comm(devices, 0);

        let ops = dist_ops(&mesh);
        let mut data = vec![42.0f32, 24.0];

        let result = ops.broadcast_f32(&mut data, 0, "world");
        assert!(result.is_ok());
    }

    #[test]
    fn test_barrier() {
        let devices = create_test_devices(4);
        let mesh = DeviceMesh::new_with_mock_comm(devices, 0);

        let ops = dist_ops(&mesh);
        let result = ops.barrier("world");
        assert!(result.is_ok());
    }

    #[test]
    fn test_send_recv_f32() {
        use crate::mock_comm::MockComm;
        use std::sync::{Arc, RwLock};

        // Create shared state for 2 ranks
        let state = Arc::new(RwLock::new(crate::mock_comm::MockCommState::new(2)));

        // Create two meshes with shared comm state
        let devices = create_test_devices(2);
        let mut mesh0 = DeviceMesh::new(devices.clone());
        mesh0.rank = 0;
        mesh0.comm = Some(Arc::new(MockComm::with_shared_state(0, state.clone())));

        let mut mesh1 = DeviceMesh::new(devices);
        mesh1.rank = 1;
        mesh1.comm = Some(Arc::new(MockComm::with_shared_state(1, state)));

        // Rank 0 sends to rank 1
        let ops0 = dist_ops(&mesh0);
        let send_data = vec![1.0f32, 2.0, 3.0];
        ops0.send_f32(&send_data, 1).unwrap();

        // Rank 1 receives from rank 0
        let ops1 = dist_ops(&mesh1);
        let mut recv_data = vec![0.0f32; 3];
        ops1.recv_f32(&mut recv_data, 0).unwrap();

        assert_eq!(recv_data, send_data);
    }

    #[test]
    fn test_no_comm_backend_error() {
        let devices = create_test_devices(2);
        let mesh = DeviceMesh::new(devices); // No mock comm attached

        let ops = dist_ops(&mesh);
        let mut data = vec![1.0f32];

        let result = ops.all_reduce_f32(&mut data, ReduceOp::Sum, "world");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No communication backend"));
    }
}
