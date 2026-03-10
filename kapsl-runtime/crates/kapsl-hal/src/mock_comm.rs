//! Mock Communication Backend for Testing
//!
//! Provides a single-process implementation of `MeshComm` that simulates
//! distributed operations for testing without requiring real multi-GPU setup.

use crate::device_mesh::{DType, MeshComm, ReduceOp};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Shared state for simulating multi-rank communication in a single process.
///
/// Each "rank" has its own buffer space, and operations coordinate through
/// this shared state.
#[derive(Debug, Default)]
pub struct MockCommState {
    /// Per-rank buffers for simulating communication
    pub rank_buffers: HashMap<usize, Vec<u8>>,
    /// World size for this communication group
    pub world_size: usize,
    /// Barrier counter for synchronization
    barrier_count: usize,
}

impl MockCommState {
    /// Create new shared state for a given world size
    pub fn new(world_size: usize) -> Self {
        Self {
            rank_buffers: HashMap::new(),
            world_size,
            barrier_count: 0,
        }
    }
}

/// Mock communication backend for testing distributed operations.
///
/// This implementation simulates collective operations in-memory,
/// allowing testing of distributed code paths on a single machine.
#[derive(Debug)]
pub struct MockComm {
    /// This rank's ID in the mesh
    pub rank: usize,
    /// Total number of ranks
    pub world_size: usize,
    /// Shared state for coordinating between simulated ranks
    state: Arc<RwLock<MockCommState>>,
}

impl MockComm {
    /// Create a new MockComm for a specific rank
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self {
            rank,
            world_size,
            state: Arc::new(RwLock::new(MockCommState::new(world_size))),
        }
    }

    /// Create a new MockComm with shared state (for multi-rank simulation)
    pub fn with_shared_state(rank: usize, state: Arc<RwLock<MockCommState>>) -> Self {
        let world_size = state.read().unwrap().world_size;
        Self {
            rank,
            world_size,
            state,
        }
    }

    /// Create a group of MockComm instances that share state
    pub fn create_group(world_size: usize) -> Vec<Self> {
        let state = Arc::new(RwLock::new(MockCommState::new(world_size)));
        (0..world_size)
            .map(|rank| Self::with_shared_state(rank, state.clone()))
            .collect()
    }

    /// Helper to apply a reduction operation on f32 values
    fn reduce_f32(values: &[f32], op: ReduceOp) -> f32 {
        match op {
            ReduceOp::Sum => values.iter().sum(),
            ReduceOp::Product => values.iter().product(),
            ReduceOp::Min => values.iter().cloned().fold(f32::INFINITY, f32::min),
            ReduceOp::Max => values.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            ReduceOp::Average => values.iter().sum::<f32>() / values.len() as f32,
        }
    }

    /// Helper to apply a reduction operation on i32 values
    fn reduce_i32(values: &[i32], op: ReduceOp) -> i32 {
        match op {
            ReduceOp::Sum => values.iter().sum(),
            ReduceOp::Product => values.iter().product(),
            ReduceOp::Min => values.iter().cloned().min().unwrap_or(0),
            ReduceOp::Max => values.iter().cloned().max().unwrap_or(0),
            ReduceOp::Average => values.iter().sum::<i32>() / values.len().max(1) as i32,
        }
    }
}

impl MeshComm for MockComm {
    fn all_reduce(
        &self,
        buf: &mut [u8],
        dtype: DType,
        op: ReduceOp,
        _group: &str,
    ) -> Result<(), String> {
        // In single-rank mode, the buffer is already the result
        if self.world_size == 1 {
            return Ok(());
        }

        // Store our data in shared state
        {
            let mut state = self.state.write().map_err(|e| e.to_string())?;
            state.rank_buffers.insert(self.rank, buf.to_vec());
        }

        // Wait for all ranks (in real impl, this would be a barrier)
        // For mock, we simulate by checking if all buffers are present
        let all_buffers: Vec<Vec<u8>> = {
            let state = self.state.read().map_err(|e| e.to_string())?;
            if state.rank_buffers.len() < self.world_size {
                // Not all ranks have contributed yet - in real impl we'd wait
                // For mock, just use our own buffer
                return Ok(());
            }
            (0..self.world_size)
                .filter_map(|r| state.rank_buffers.get(&r).cloned())
                .collect()
        };

        // Perform reduction based on dtype
        match dtype {
            DType::Float32 => {
                let elem_count = buf.len() / 4;
                for i in 0..elem_count {
                    let values: Vec<f32> = all_buffers
                        .iter()
                        .map(|b| {
                            let bytes: [u8; 4] = b[i * 4..(i + 1) * 4].try_into().unwrap();
                            f32::from_le_bytes(bytes)
                        })
                        .collect();
                    let result = Self::reduce_f32(&values, op);
                    buf[i * 4..(i + 1) * 4].copy_from_slice(&result.to_le_bytes());
                }
            }
            DType::Int32 => {
                let elem_count = buf.len() / 4;
                for i in 0..elem_count {
                    let values: Vec<i32> = all_buffers
                        .iter()
                        .map(|b| {
                            let bytes: [u8; 4] = b[i * 4..(i + 1) * 4].try_into().unwrap();
                            i32::from_le_bytes(bytes)
                        })
                        .collect();
                    let result = Self::reduce_i32(&values, op);
                    buf[i * 4..(i + 1) * 4].copy_from_slice(&result.to_le_bytes());
                }
            }
            _ => {
                // For other dtypes, just keep our buffer (no-op reduction)
            }
        }

        Ok(())
    }

    fn all_gather(
        &self,
        local: &[u8],
        out: &mut [u8],
        _dtype: DType,
        _group: &str,
    ) -> Result<(), String> {
        // In single-rank mode, just copy local to output
        if self.world_size == 1 {
            out[..local.len()].copy_from_slice(local);
            return Ok(());
        }

        // Store our local data
        {
            let mut state = self.state.write().map_err(|e| e.to_string())?;
            state.rank_buffers.insert(self.rank, local.to_vec());
        }

        // Gather from all ranks
        let chunk_size = local.len();
        let state = self.state.read().map_err(|e| e.to_string())?;

        for rank in 0..self.world_size {
            let offset = rank * chunk_size;
            if let Some(data) = state.rank_buffers.get(&rank) {
                let copy_len = data.len().min(chunk_size);
                out[offset..offset + copy_len].copy_from_slice(&data[..copy_len]);
            }
        }

        Ok(())
    }

    fn broadcast(&self, buf: &mut [u8], root_rank: usize, _group: &str) -> Result<(), String> {
        if self.rank == root_rank {
            // Root stores its data
            let mut state = self.state.write().map_err(|e| e.to_string())?;
            state.rank_buffers.insert(root_rank, buf.to_vec());
        } else {
            // Non-root reads from root's buffer
            let state = self.state.read().map_err(|e| e.to_string())?;
            if let Some(root_data) = state.rank_buffers.get(&root_rank) {
                let copy_len = root_data.len().min(buf.len());
                buf[..copy_len].copy_from_slice(&root_data[..copy_len]);
            }
        }
        Ok(())
    }

    fn reduce_scatter(
        &self,
        buf: &mut [u8],
        out: &mut [u8],
        op: ReduceOp,
        group: &str,
    ) -> Result<(), String> {
        // First do all-reduce
        self.all_reduce(buf, DType::Float32, op, group)?;

        // Then scatter - each rank gets its chunk
        let chunk_size = buf.len() / self.world_size;
        let offset = self.rank * chunk_size;
        let copy_len = chunk_size.min(out.len());
        out[..copy_len].copy_from_slice(&buf[offset..offset + copy_len]);

        Ok(())
    }

    fn barrier(&self, _group: &str) -> Result<(), String> {
        // Increment barrier counter
        let mut state = self.state.write().map_err(|e| e.to_string())?;
        state.barrier_count += 1;
        // In a real implementation, we'd wait for all ranks to reach the barrier
        Ok(())
    }

    fn send(&self, buf: &[u8], dest_rank: usize) -> Result<(), String> {
        if dest_rank >= self.world_size {
            return Err(format!("Invalid dest rank {}", dest_rank));
        }
        // Store in shared state with a special key
        let key = self.rank * 1000 + dest_rank; // Unique key for src->dst
        let mut state = self.state.write().map_err(|e| e.to_string())?;
        state.rank_buffers.insert(key, buf.to_vec());
        Ok(())
    }

    fn recv(&self, buf: &mut [u8], src_rank: usize) -> Result<(), String> {
        if src_rank >= self.world_size {
            return Err(format!("Invalid src rank {}", src_rank));
        }
        let key = src_rank * 1000 + self.rank;
        let state = self.state.read().map_err(|e| e.to_string())?;
        if let Some(data) = state.rank_buffers.get(&key) {
            let copy_len = data.len().min(buf.len());
            buf[..copy_len].copy_from_slice(&data[..copy_len]);
            Ok(())
        } else {
            Err(format!("No data from rank {}", src_rank))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_comm_single_rank() {
        let comm = MockComm::new(0, 1);
        let buf = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = buf.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut byte_buf = bytes;

        comm.all_reduce(&mut byte_buf, DType::Float32, ReduceOp::Sum, "world")
            .unwrap();

        // Single rank, buffer unchanged
        let result: Vec<f32> = byte_buf
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mock_comm_broadcast() {
        let comms = MockComm::create_group(2);

        // Root broadcasts
        let mut root_buf = vec![0u8; 8];
        root_buf[..4].copy_from_slice(&42.0f32.to_le_bytes());
        root_buf[4..8].copy_from_slice(&24.0f32.to_le_bytes());

        comms[0].broadcast(&mut root_buf, 0, "world").unwrap();

        // Non-root receives
        let mut recv_buf = vec![0u8; 8];
        comms[1].broadcast(&mut recv_buf, 0, "world").unwrap();

        assert_eq!(root_buf, recv_buf);
    }

    #[test]
    fn test_mock_comm_send_recv() {
        let comms = MockComm::create_group(2);

        let send_data = vec![1u8, 2, 3, 4];
        comms[0].send(&send_data, 1).unwrap();

        let mut recv_buf = vec![0u8; 4];
        comms[1].recv(&mut recv_buf, 0).unwrap();

        assert_eq!(recv_buf, send_data);
    }

    #[test]
    fn test_mock_comm_barrier() {
        let comm = MockComm::new(0, 4);
        assert!(comm.barrier("world").is_ok());
    }

    #[test]
    fn test_mock_comm_all_gather() {
        let comms = MockComm::create_group(2);

        // Each rank contributes its data
        let local0 = vec![1u8, 2];
        let local1 = vec![3u8, 4];

        let mut out0 = vec![0u8; 4];
        let mut out1 = vec![0u8; 4];

        comms[0]
            .all_gather(&local0, &mut out0, DType::UInt8, "world")
            .unwrap();
        comms[1]
            .all_gather(&local1, &mut out1, DType::UInt8, "world")
            .unwrap();

        // Both should have gathered data
        assert_eq!(&out0[0..2], &local0[..]);
        assert_eq!(&out1[2..4], &local1[..]);
    }
}
