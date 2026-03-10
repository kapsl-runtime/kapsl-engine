use crate::device::Device;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum MeshTopology {
    DataParallel,
    TensorParallel {
        degree: usize,
        mesh_shape: (usize, usize),
    },
    PipelineParallel {
        stages: usize,
    },
    Mixed {
        tp: usize,
        pp: usize,
        dp: usize,
    },
}

impl MeshTopology {
    /// Get the expected world size for this topology
    pub fn expected_world_size(&self) -> usize {
        match self {
            MeshTopology::DataParallel => 1, // Can be any size
            MeshTopology::TensorParallel { degree, .. } => *degree,
            MeshTopology::PipelineParallel { stages } => *stages,
            MeshTopology::Mixed { tp, pp, dp } => tp * pp * dp,
        }
    }

    /// Validate if the current world size matches the topology
    pub fn validate(&self, world_size: usize) -> Result<(), String> {
        match self {
            MeshTopology::DataParallel => Ok(()), // Any size works
            MeshTopology::TensorParallel { degree, .. } => {
                if world_size < *degree {
                    Err(format!(
                        "Tensor parallel degree {} requires at least {} devices, got {}",
                        degree, degree, world_size
                    ))
                } else {
                    Ok(())
                }
            }
            MeshTopology::PipelineParallel { stages } => {
                if world_size < *stages {
                    Err(format!(
                        "Pipeline parallel requires {} stages, got {} devices",
                        stages, world_size
                    ))
                } else {
                    Ok(())
                }
            }
            MeshTopology::Mixed { tp, pp, dp } => {
                let required = tp * pp * dp;
                if world_size < required {
                    Err(format!(
                        "Mixed parallelism (TP={}, PP={}, DP={}) requires {} devices, got {}",
                        tp, pp, dp, required, world_size
                    ))
                } else {
                    Ok(())
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProcessGroup {
    pub name: String,
    pub ranks: Vec<usize>,
    pub backend: GroupBackend,
}

#[derive(Debug, Clone, Copy)]
pub enum GroupBackend {
    Nccl, // NVIDIA GPUs
    Gloo, // CPU/generic
    MPI,  // HPC environments
    Mock, // Testing
}

/// Device Mesh for distributed ML inference
///
/// Manages a set of devices arranged in a logical mesh topology for
/// data/tensor/pipeline parallelism. Optimized for memory efficiency.
#[derive(Debug, Clone)]
pub struct DeviceMesh {
    /// Devices ordered by mesh coordinate (using Arc for memory efficiency)
    devices: Arc<Vec<Arc<Device>>>,

    /// Total number of devices in the mesh
    pub world_size: usize,

    /// This process's global rank
    pub rank: usize,

    /// Mesh topology configuration
    pub topology: MeshTopology,

    /// Process groups (map group name -> group definition)
    groups: Arc<HashMap<String, ProcessGroup>>,

    /// Communication backend handle
    pub comm: Option<Arc<dyn MeshComm + Send + Sync>>,
}

impl DeviceMesh {
    /// Create a new device mesh from a list of devices
    pub fn new(devices: Vec<Device>) -> Self {
        let world_size = devices.len();
        let devices: Vec<Arc<Device>> = devices.into_iter().map(Arc::new).collect();

        // Default to DataParallel topology
        let topology = MeshTopology::DataParallel;

        // Default groups: "world" contains all ranks
        let mut groups = HashMap::new();
        groups.insert(
            "world".to_string(),
            ProcessGroup {
                name: "world".to_string(),
                ranks: (0..world_size).collect(),
                backend: GroupBackend::Mock,
            },
        );

        Self {
            devices: Arc::new(devices),
            world_size,
            rank: 0, // Default to rank 0 for single-node
            topology,
            groups: Arc::new(groups),
            comm: None,
        }
    }

    /// Create a new device mesh with a mock communication backend for testing
    pub fn new_with_mock_comm(devices: Vec<Device>, rank: usize) -> Self {
        use crate::mock_comm::MockComm;

        let world_size = devices.len();
        let mut mesh = Self::new(devices);
        mesh.rank = rank;
        mesh.comm = Some(Arc::new(MockComm::new(rank, world_size)));
        mesh
    }

    /// Create a new device mesh with NCCL communication backend for real GPU communication
    ///
    /// # Arguments
    /// * `devices` - List of devices in the mesh
    /// * `nccl_id` - NCCL unique ID (generate with `NcclComm::generate_id()` on rank 0)
    /// * `rank` - This process's rank
    ///
    /// # Requirements
    /// - CUDA device available
    /// - NCCL library installed
    /// - All ranks must call this with the same `nccl_id`
    #[cfg(feature = "nccl")]
    pub fn new_with_nccl(
        devices: Vec<Device>,
        nccl_id: &crate::nccl_comm::cudarc::nccl::Id,
        rank: usize,
    ) -> Result<Self, String> {
        use crate::nccl_comm::NcclComm;
        use cudarc::driver::CudaDevice;

        let world_size = devices.len();

        // Get the CUDA device for this rank
        let cuda_device = CudaDevice::new(rank)
            .map_err(|e| format!("Failed to get CUDA device {}: {:?}", rank, e))?;

        let nccl_comm = NcclComm::new(cuda_device, nccl_id, rank, world_size)?;

        let mut mesh = Self::new(devices);
        mesh.rank = rank;
        mesh.comm = Some(Arc::new(nccl_comm));
        Ok(mesh)
    }

    /// Create a mesh with specific topology
    pub fn with_topology(devices: Vec<Device>, topology: MeshTopology) -> Result<Self, String> {
        let world_size = devices.len();

        // Validate topology against world size
        topology.validate(world_size)?;

        let devices: Vec<Arc<Device>> = devices.into_iter().map(Arc::new).collect();

        let mut groups = HashMap::new();

        // Create world group
        groups.insert(
            "world".to_string(),
            ProcessGroup {
                name: "world".to_string(),
                ranks: (0..world_size).collect(),
                backend: GroupBackend::Mock,
            },
        );

        // Create topology-specific groups
        match &topology {
            MeshTopology::TensorParallel { degree, .. } => {
                // Create TP groups
                for i in 0..world_size / degree {
                    let start = i * degree;
                    let ranks: Vec<usize> = (start..start + degree).collect();
                    groups.insert(
                        format!("tp_{}", i),
                        ProcessGroup {
                            name: format!("tp_{}", i),
                            ranks,
                            backend: GroupBackend::Nccl,
                        },
                    );
                }
            }
            MeshTopology::PipelineParallel { stages } => {
                // Each stage is a group
                for stage in 0..*stages {
                    groups.insert(
                        format!("pp_stage_{}", stage),
                        ProcessGroup {
                            name: format!("pp_stage_{}", stage),
                            ranks: vec![stage],
                            backend: GroupBackend::Gloo,
                        },
                    );
                }
            }
            MeshTopology::Mixed { tp, pp, dp } => {
                // Create TP groups
                let tp_size = *tp;
                for dp_idx in 0..*dp {
                    for pp_idx in 0..*pp {
                        let base = (dp_idx * pp + pp_idx) * tp_size;
                        let ranks: Vec<usize> = (base..base + tp_size).collect();
                        groups.insert(
                            format!("tp_dp{}_pp{}", dp_idx, pp_idx),
                            ProcessGroup {
                                name: format!("tp_dp{}_pp{}", dp_idx, pp_idx),
                                ranks,
                                backend: GroupBackend::Nccl,
                            },
                        );
                    }
                }
            }
            _ => {}
        }

        Ok(Self {
            devices: Arc::new(devices),
            world_size,
            rank: 0,
            topology,
            groups: Arc::new(groups),
            comm: None,
        })
    }

    /// Set the rank for this process
    pub fn set_rank(&mut self, rank: usize) -> Result<(), String> {
        if rank >= self.world_size {
            return Err(format!(
                "Rank {} out of bounds for world size {}",
                rank, self.world_size
            ));
        }
        self.rank = rank;
        Ok(())
    }

    /// Get device by rank (memory efficient - returns Arc clone)
    pub fn get_device(&self, rank: usize) -> Option<Arc<Device>> {
        self.devices.get(rank).cloned()
    }

    /// Get the local device for this process
    pub fn local_device(&self) -> Option<Arc<Device>> {
        self.get_device(self.rank)
    }

    /// Get all devices (returns Arc to avoid cloning the entire Vec)
    pub fn all_devices(&self) -> Arc<Vec<Arc<Device>>> {
        self.devices.clone()
    }

    /// Get devices for a specific backend type (memory efficient)
    pub fn devices_by_backend(&self, backend: crate::device::DeviceBackend) -> Vec<Arc<Device>> {
        self.devices
            .iter()
            .filter(|d| std::mem::discriminant(&d.backend) == std::mem::discriminant(&backend))
            .cloned()
            .collect()
    }

    /// Get devices in a specific process group
    pub fn devices_in_group(&self, group_name: &str) -> Result<Vec<Arc<Device>>, String> {
        let group = self
            .groups
            .get(group_name)
            .ok_or_else(|| format!("Group '{}' not found", group_name))?;

        Ok(group
            .ranks
            .iter()
            .filter_map(|&rank| self.get_device(rank))
            .collect())
    }

    /// Add a custom process group
    pub fn add_group(
        &mut self,
        name: String,
        ranks: Vec<usize>,
        backend: GroupBackend,
    ) -> Result<(), String> {
        // Validate ranks
        for &rank in &ranks {
            if rank >= self.world_size {
                return Err(format!("Rank {} out of bounds", rank));
            }
        }

        let groups = Arc::make_mut(&mut self.groups);
        groups.insert(
            name.clone(),
            ProcessGroup {
                name,
                ranks,
                backend,
            },
        );

        Ok(())
    }

    /// Get a process group by name
    pub fn get_group(&self, name: &str) -> Option<&ProcessGroup> {
        self.groups.get(name)
    }

    /// List all group names
    pub fn group_names(&self) -> Vec<String> {
        self.groups.keys().cloned().collect()
    }

    /// Check if this rank is in a specific group
    pub fn in_group(&self, group_name: &str) -> bool {
        self.groups
            .get(group_name)
            .map(|g| g.ranks.contains(&self.rank))
            .unwrap_or(false)
    }

    /// Get rank within a group (local rank)
    pub fn group_rank(&self, group_name: &str) -> Option<usize> {
        self.groups
            .get(group_name)
            .and_then(|g| g.ranks.iter().position(|&r| r == self.rank))
    }

    /// Set the communication backend
    pub fn set_comm(&mut self, comm: Arc<dyn MeshComm + Send + Sync>) {
        self.comm = Some(comm);
    }

    /// Get total memory across all devices
    pub fn total_memory_mb(&self) -> u64 {
        self.devices.iter().map(|d| d.memory_mb).sum()
    }

    /// Get total compute units across all devices
    pub fn total_compute_units(&self) -> u32 {
        self.devices.iter().map(|d| d.compute_units).sum()
    }

    /// Reshape the mesh to a different topology
    pub fn reshape(&mut self, new_topology: MeshTopology) -> Result<(), String> {
        new_topology.validate(self.world_size)?;
        self.topology = new_topology;
        Ok(())
    }

    /// Get mesh statistics
    pub fn stats(&self) -> MeshStats {
        let backend_counts = self.count_backends();

        MeshStats {
            world_size: self.world_size,
            total_memory_mb: self.total_memory_mb(),
            total_compute_units: self.total_compute_units(),
            backend_distribution: backend_counts,
            group_count: self.groups.len(),
        }
    }

    fn count_backends(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for device in self.devices.iter() {
            let backend_name = format!("{:?}", device.backend);
            *counts.entry(backend_name).or_insert(0) += 1;
        }
        counts
    }
}

/// Statistics about the device mesh
#[derive(Debug, Clone)]
pub struct MeshStats {
    pub world_size: usize,
    pub total_memory_mb: u64,
    pub total_compute_units: u32,
    pub backend_distribution: HashMap<String, usize>,
    pub group_count: usize,
}

/// Communication operations for distributed execution
pub trait MeshComm: std::fmt::Debug {
    /// All-reduce operation: reduce values across all ranks in a group
    fn all_reduce(
        &self,
        buf: &mut [u8],
        dtype: DType,
        op: ReduceOp,
        group: &str,
    ) -> Result<(), String>;

    /// All-gather: gather data from all ranks
    fn all_gather(
        &self,
        local: &[u8],
        out: &mut [u8],
        dtype: DType,
        group: &str,
    ) -> Result<(), String>;

    /// Broadcast from root rank to all ranks in group
    fn broadcast(&self, buf: &mut [u8], root_rank: usize, group: &str) -> Result<(), String>;

    /// Reduce-scatter: reduce and distribute results
    fn reduce_scatter(
        &self,
        buf: &mut [u8],
        out: &mut [u8],
        op: ReduceOp,
        group: &str,
    ) -> Result<(), String>;

    /// Barrier synchronization
    fn barrier(&self, group: &str) -> Result<(), String>;

    /// Send data to a specific rank
    fn send(&self, buf: &[u8], dest_rank: usize) -> Result<(), String>;

    /// Receive data from a specific rank
    fn recv(&self, buf: &mut [u8], src_rank: usize) -> Result<(), String>;
}

/// Data type for communication operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int64,
    UInt8,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float16 => 2,
            DType::BFloat16 => 2,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::UInt8 => 1,
        }
    }
}

/// Reduction operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
    Average,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceBackend;

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
    fn test_mesh_creation() {
        let devices = create_test_devices(4);
        let mesh = DeviceMesh::new(devices);

        assert_eq!(mesh.world_size, 4);
        assert_eq!(mesh.rank, 0);
        assert!(mesh.get_group("world").is_some());
    }

    #[test]
    fn test_tensor_parallel_topology() {
        let devices = create_test_devices(8);
        let topology = MeshTopology::TensorParallel {
            degree: 4,
            mesh_shape: (2, 4),
        };

        let mesh = DeviceMesh::with_topology(devices, topology).unwrap();
        assert_eq!(mesh.world_size, 8);

        // Should have 2 TP groups (8 devices / 4 degree)
        assert!(mesh.get_group("tp_0").is_some());
        assert!(mesh.get_group("tp_1").is_some());
    }

    #[test]
    fn test_group_operations() {
        let devices = create_test_devices(4);
        let mut mesh = DeviceMesh::new(devices);

        // Add custom group
        mesh.add_group("custom".to_string(), vec![0, 2], GroupBackend::Gloo)
            .unwrap();

        assert!(mesh.get_group("custom").is_some());
        assert_eq!(mesh.get_group("custom").unwrap().ranks, vec![0, 2]);
    }

    #[test]
    fn test_mesh_stats() {
        let devices = create_test_devices(4);
        let mesh = DeviceMesh::new(devices);

        let stats = mesh.stats();
        assert_eq!(stats.world_size, 4);
        assert_eq!(stats.total_memory_mb, 64000); // 4 * 16000
        assert_eq!(stats.total_compute_units, 320); // 4 * 80
    }
}
