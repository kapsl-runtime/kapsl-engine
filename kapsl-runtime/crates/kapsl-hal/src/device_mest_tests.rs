use crate::device::{Device, DeviceBackend};
use crate::device_mesh::{DeviceMesh, GroupBackend, MeshTopology};

fn make_device(id: usize, backend: DeviceBackend, memory_mb: u64) -> Device {
    Device {
        id,
        name: format!("dev_{id}"),
        backend,
        memory_mb,
        compute_units: 8,
        pci_bus_id: None,
        driver_version: None,
        supports_fp16: true,
        supports_int8: true,
        cuda_version: None,
        compute_capability: None,
        utilization_gpu_pct: None,
        temperature_c: None,
    }
}

fn make_devices() -> Vec<Device> {
    vec![
        make_device(0, DeviceBackend::Cpu, 1024),
        make_device(1, DeviceBackend::Cuda, 8192),
        make_device(2, DeviceBackend::Cuda, 4096),
        make_device(3, DeviceBackend::Metal, 2048),
    ]
}

#[test]
fn topology_expected_world_size() {
    assert_eq!(MeshTopology::DataParallel.expected_world_size(), 1);
    assert_eq!(
        MeshTopology::TensorParallel {
            degree: 4,
            mesh_shape: (2, 2)
        }
        .expected_world_size(),
        4
    );
    assert_eq!(
        MeshTopology::PipelineParallel { stages: 3 }.expected_world_size(),
        3
    );
    assert_eq!(
        MeshTopology::Mixed {
            tp: 2,
            pp: 2,
            dp: 3
        }
        .expected_world_size(),
        12
    );
}

#[test]
fn topology_validate_rejects_too_small_world() {
    let topology = MeshTopology::TensorParallel {
        degree: 4,
        mesh_shape: (2, 2),
    };

    let err = topology.validate(2).unwrap_err();
    assert!(err.contains("requires at least"));
}

#[test]
fn mesh_with_topology_rejects_invalid_world_size() {
    let devices = vec![
        make_device(0, DeviceBackend::Cpu, 1024),
        make_device(1, DeviceBackend::Cuda, 4096),
    ];

    let topology = MeshTopology::TensorParallel {
        degree: 4,
        mesh_shape: (2, 2),
    };

    assert!(DeviceMesh::with_topology(devices, topology).is_err());
}

#[test]
fn custom_group_membership_and_rank() {
    let devices = make_devices();
    let mut mesh = DeviceMesh::new(devices);

    mesh.add_group("custom".to_string(), vec![1, 3], GroupBackend::Gloo)
        .unwrap();

    mesh.set_rank(3).unwrap();
    assert!(mesh.in_group("custom"));
    assert_eq!(mesh.group_rank("custom"), Some(1));

    mesh.set_rank(0).unwrap();
    assert!(!mesh.in_group("custom"));
    assert_eq!(mesh.group_rank("custom"), None);
}

#[test]
fn devices_by_backend_filters_variants() {
    let devices = make_devices();
    let mesh = DeviceMesh::new(devices);

    let cuda = mesh.devices_by_backend(DeviceBackend::Cuda);
    assert_eq!(cuda.len(), 2);
    assert!(cuda
        .iter()
        .all(|d| matches!(d.backend, DeviceBackend::Cuda)));

    let metal = mesh.devices_by_backend(DeviceBackend::Metal);
    assert_eq!(metal.len(), 1);
}

#[test]
fn stats_reflect_mesh_totals() {
    let devices = make_devices();
    let mesh = DeviceMesh::new(devices);

    let stats = mesh.stats();
    assert_eq!(stats.world_size, 4);
    assert_eq!(stats.total_memory_mb, 1024 + 8192 + 4096 + 2048);
    assert_eq!(stats.total_compute_units, 4 * 8);
    assert!(stats.group_count >= 1);
}
