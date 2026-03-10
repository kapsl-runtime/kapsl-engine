//! Mesh Routing Module
//!
//! Provides topology-aware routing strategies for distributed inference.
//! Routes requests to appropriate workers based on the mesh topology.

use kapsl_hal::device_mesh::{DeviceMesh, MeshTopology};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Routing strategy based on mesh topology
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Round-robin across all workers (default for DataParallel)
    RoundRobin,
    /// Sticky routing based on session ID hash
    SessionAffinity,
    /// Route to specific TP group based on request metadata
    TensorParallel,
    /// Route to first pipeline stage
    PipelineParallel,
}

/// Router that selects workers based on mesh topology
#[derive(Debug)]
pub struct MeshRouter {
    /// Device mesh configuration
    mesh: Option<Arc<DeviceMesh>>,
    /// Number of workers available
    num_workers: usize,
    /// Counter for round-robin routing
    rr_counter: AtomicUsize,
}

impl MeshRouter {
    /// Create a new mesh router
    pub fn new(mesh: Option<Arc<DeviceMesh>>, num_workers: usize) -> Self {
        Self {
            mesh,
            num_workers,
            rr_counter: AtomicUsize::new(0),
        }
    }

    /// Get the routing strategy based on mesh topology
    pub fn strategy(&self) -> RoutingStrategy {
        match &self.mesh {
            None => RoutingStrategy::RoundRobin,
            Some(mesh) => match &mesh.topology {
                MeshTopology::DataParallel => RoutingStrategy::RoundRobin,
                MeshTopology::TensorParallel { .. } => RoutingStrategy::TensorParallel,
                MeshTopology::PipelineParallel { .. } => RoutingStrategy::PipelineParallel,
                MeshTopology::Mixed { .. } => RoutingStrategy::TensorParallel, // Default to TP routing for mixed
            },
        }
    }

    /// Route a request to an appropriate worker index
    ///
    /// # Arguments
    /// * `session_id` - Optional session ID for sticky routing
    /// * `tp_group_hint` - Optional hint for which TP group to target
    ///
    /// # Returns
    /// Worker index to route the request to
    pub fn route(&self, session_id: &Option<String>, tp_group_hint: Option<usize>) -> usize {
        if self.num_workers == 0 {
            return 0;
        }

        // For any strategy, if a session_id is provided, use sticky routing
        // This ensures consistent routing for stateful sessions regardless of topology
        if let Some(ref id) = session_id {
            return self.route_by_session(id);
        }

        match self.strategy() {
            RoutingStrategy::RoundRobin | RoutingStrategy::SessionAffinity => {
                self.route_round_robin()
            }
            RoutingStrategy::TensorParallel => {
                self.route_tensor_parallel(session_id, tp_group_hint)
            }
            RoutingStrategy::PipelineParallel => self.route_pipeline_parallel(),
        }
    }

    /// Round-robin routing across all workers
    fn route_round_robin(&self) -> usize {
        self.rr_counter.fetch_add(1, Ordering::Relaxed) % self.num_workers
    }

    /// Route based on session ID hash for sticky sessions
    fn route_by_session(&self, session_id: &str) -> usize {
        let mut hasher = DefaultHasher::new();
        session_id.hash(&mut hasher);
        (hasher.finish() as usize) % self.num_workers
    }

    /// Route for tensor parallelism - requests go to workers in same TP group
    fn route_tensor_parallel(
        &self,
        session_id: &Option<String>,
        tp_group_hint: Option<usize>,
    ) -> usize {
        let mesh = match &self.mesh {
            Some(m) => m,
            None => return self.route_round_robin(),
        };

        // Get TP degree from topology
        let tp_degree = match &mesh.topology {
            MeshTopology::TensorParallel { degree, .. } => *degree,
            MeshTopology::Mixed { tp, .. } => *tp,
            _ => 1,
        };

        if tp_degree <= 1 {
            // No tensor parallelism, use round-robin
            return self.route_round_robin();
        }

        // Determine which TP group to use
        let num_tp_groups = self.num_workers / tp_degree;
        if num_tp_groups == 0 {
            return self.route_round_robin();
        }

        let tp_group = if let Some(hint) = tp_group_hint {
            hint % num_tp_groups
        } else if let Some(ref id) = session_id {
            // Use session ID to pick a consistent TP group
            let mut hasher = DefaultHasher::new();
            id.hash(&mut hasher);
            (hasher.finish() as usize) % num_tp_groups
        } else {
            // Round-robin across TP groups
            (self.rr_counter.fetch_add(1, Ordering::Relaxed) / tp_degree) % num_tp_groups
        };

        // Return the first worker in the selected TP group
        // (in a real TP implementation, all workers in the group would be used)
        tp_group * tp_degree
    }

    /// Route for pipeline parallelism - always route to stage 0
    fn route_pipeline_parallel(&self) -> usize {
        // Pipeline parallel routes to the first stage (rank 0)
        // The pipeline execution will handle forwarding to subsequent stages
        0
    }

    /// Get the TP group index for a given worker
    pub fn get_tp_group(&self, worker_idx: usize) -> Option<usize> {
        let mesh = self.mesh.as_ref()?;
        let tp_degree = match &mesh.topology {
            MeshTopology::TensorParallel { degree, .. } => *degree,
            MeshTopology::Mixed { tp, .. } => *tp,
            _ => return None,
        };

        if tp_degree <= 1 {
            return None;
        }

        Some(worker_idx / tp_degree)
    }

    /// Get all worker indices in a TP group
    pub fn get_tp_group_workers(&self, tp_group: usize) -> Vec<usize> {
        let mesh = match &self.mesh {
            Some(m) => m,
            None => return vec![],
        };

        let tp_degree = match &mesh.topology {
            MeshTopology::TensorParallel { degree, .. } => *degree,
            MeshTopology::Mixed { tp, .. } => *tp,
            _ => return vec![],
        };

        let start = tp_group * tp_degree;
        (start..start + tp_degree)
            .filter(|&i| i < self.num_workers)
            .collect()
    }

    /// Get mesh statistics if available
    pub fn mesh_stats(&self) -> Option<MeshRouterStats> {
        let mesh = self.mesh.as_ref()?;
        Some(MeshRouterStats {
            world_size: mesh.world_size,
            topology: format!("{:?}", mesh.topology),
            num_workers: self.num_workers,
            strategy: self.strategy(),
        })
    }
}

/// Statistics about the mesh router
#[derive(Debug, Clone)]
pub struct MeshRouterStats {
    pub world_size: usize,
    pub topology: String,
    pub num_workers: usize,
    pub strategy: RoutingStrategy,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_hal::device::{Device, DeviceBackend};

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
    fn test_round_robin_routing() {
        let router = MeshRouter::new(None, 4);

        let indices: Vec<usize> = (0..8).map(|_| router.route(&None, None)).collect();

        // Should cycle through workers
        assert_eq!(indices, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn test_session_affinity_routing() {
        let devices = create_test_devices(4);
        let mesh = DeviceMesh::new(devices);
        let router = MeshRouter::new(Some(Arc::new(mesh)), 4);

        let session = Some("user-123".to_string());
        let first = router.route(&session, None);
        let second = router.route(&session, None);

        // Same session should route to same worker
        assert_eq!(first, second);
    }

    #[test]
    fn test_tensor_parallel_routing() {
        let devices = create_test_devices(8);
        let topology = MeshTopology::TensorParallel {
            degree: 4,
            mesh_shape: (2, 4),
        };
        let mesh = DeviceMesh::with_topology(devices, topology).unwrap();
        let router = MeshRouter::new(Some(Arc::new(mesh)), 8);

        assert_eq!(router.strategy(), RoutingStrategy::TensorParallel);

        // Routing with TP group hint should respect the hint
        let idx = router.route(&None, Some(0));
        assert!(idx < 4); // Should be in first TP group

        let idx = router.route(&None, Some(1));
        assert!((4..8).contains(&idx)); // Should be in second TP group
    }

    #[test]
    fn test_pipeline_parallel_routing() {
        let devices = create_test_devices(4);
        let topology = MeshTopology::PipelineParallel { stages: 4 };
        let mesh = DeviceMesh::with_topology(devices, topology).unwrap();
        let router = MeshRouter::new(Some(Arc::new(mesh)), 4);

        assert_eq!(router.strategy(), RoutingStrategy::PipelineParallel);

        // Pipeline parallel should always route to stage 0
        for _ in 0..10 {
            assert_eq!(router.route(&None, None), 0);
        }
    }

    #[test]
    fn test_get_tp_group_workers() {
        let devices = create_test_devices(8);
        let topology = MeshTopology::TensorParallel {
            degree: 4,
            mesh_shape: (2, 4),
        };
        let mesh = DeviceMesh::with_topology(devices, topology).unwrap();
        let router = MeshRouter::new(Some(Arc::new(mesh)), 8);

        let group0 = router.get_tp_group_workers(0);
        let group1 = router.get_tp_group_workers(1);

        assert_eq!(group0, vec![0, 1, 2, 3]);
        assert_eq!(group1, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_mesh_stats() {
        let devices = create_test_devices(4);
        let mesh = DeviceMesh::new(devices);
        let router = MeshRouter::new(Some(Arc::new(mesh)), 4);

        let stats = router.mesh_stats().unwrap();
        assert_eq!(stats.world_size, 4);
        assert_eq!(stats.num_workers, 4);
        assert_eq!(stats.strategy, RoutingStrategy::RoundRobin);
    }
}
