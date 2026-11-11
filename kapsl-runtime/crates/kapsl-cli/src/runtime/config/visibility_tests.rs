//! Placement over fake CUDA inventories where `CUDA_VISIBLE_DEVICES` hides a
//! device. No GPU: the CUDA runtime's view is the `ordinal_pci` list handed to
//! `CudaOrdinalMap::build`, exactly what the engine enumerates at startup.

use super::*;
use kapsl_hal::device::{Device, DeviceBackend};
use kapsl_llm::cuda_ordinal::{CudaOrdinalMap, EngineCudaDevice};

const PCI_0: &str = "00000000:02:00.0";
const PCI_1: &str = "00000000:03:00.0";

fn cuda(id: usize, pci: &str) -> Device {
    Device {
        id,
        name: format!("fake-cuda-{id}"),
        backend: DeviceBackend::Cuda,
        memory_mb: 24 * 1024,
        compute_units: 1,
        pci_bus_id: Some(pci.to_string()),
        partition_id: None,
        driver_version: None,
        cuda_version: Some("12.6".to_string()),
        compute_capability: Some("8.6".to_string()),
        utilization_gpu_pct: None,
        temperature_c: None,
        supports_fp16: true,
        supports_int8: true,
    }
}

fn host_with_two_gpus() -> DeviceInfo {
    DeviceInfo {
        cpu_cores: 8,
        total_memory: 64 * 1024 * 1024 * 1024,
        os_type: "linux".to_string(),
        os_release: "test".to_string(),
        has_cuda: true,
        has_metal: false,
        has_rocm: false,
        has_directml: false,
        devices: vec![cuda(0, PCI_0), cuda(1, PCI_1)],
    }
}

/// The map the engine installs when the CUDA runtime enumerates `visible`
/// (the PCI ids `CUDA_VISIBLE_DEVICES` leaves, in ordinal order).
fn map_for(info: &DeviceInfo, visible: &[&str]) -> CudaOrdinalMap {
    let engine: Vec<_> = info
        .devices
        .iter()
        .map(|device| EngineCudaDevice {
            id: device.id,
            pci_bus_id: device.pci_bus_id.clone(),
        })
        .collect();
    // The CUDA runtime spells the domain with four digits.
    let ordinal_pci: Vec<_> = visible
        .iter()
        .map(|pci| Some(pci.replacen("00000000:", "0000:", 1)))
        .collect();
    CudaOrdinalMap::build(&engine, &ordinal_pci)
}

fn requirements(
    device_id: Option<i32>,
    strategy: Option<&str>,
) -> kapsl_core::HardwareRequirements {
    kapsl_core::HardwareRequirements {
        preferred_provider: Some("cuda".to_string()),
        device_id,
        strategy: strategy.map(str::to_string),
        ..kapsl_core::HardwareRequirements::default()
    }
}

fn selected_ids(selection: &MeshDeviceSelection) -> Vec<usize> {
    selection.devices.iter().map(|device| device.id).collect()
}

#[test]
fn cvd_0_denies_a_request_for_the_hidden_device_1_with_a_clear_error() {
    let info = host_with_two_gpus();
    let map = map_for(&info, &[PCI_0]);
    for strategy in [None, Some("pool"), Some("data-parallel")] {
        let error = select_mesh_devices_with(&requirements(Some(1), strategy), &info, Some(&map))
            .expect_err("device 1 is hidden");
        assert!(error.contains("CUDA device 1"), "{error}");
        assert!(error.contains("not visible to this process"), "{error}");
        assert!(error.contains("CUDA_VISIBLE_DEVICES"), "{error}");
        assert!(error.contains("[0]"), "names the usable devices: {error}");
    }
    let typed = check_requested_device_usable(&requirements(Some(1), None), &info, Some(&map))
        .expect_err("typed denial for admission");
    assert_eq!(typed.device_id, 1);
}

#[test]
fn cvd_0_request_for_device_0_loads_on_ordinal_0() {
    let info = host_with_two_gpus();
    let map = map_for(&info, &[PCI_0]);
    let selection = select_mesh_devices_with(&requirements(Some(0), None), &info, Some(&map))
        .expect("device 0 is visible");
    assert_eq!(selected_ids(&selection), vec![0]);
    assert_eq!(map.ordinal_or_error(0), Ok(0));
}

#[test]
fn cvd_0_with_no_requested_device_is_placed_on_visible_devices_only() {
    // The reported bug: with no device named, the mesh held both inventory
    // devices and the primary load created an engine on hidden device 1.
    let info = host_with_two_gpus();
    let map = map_for(&info, &[PCI_0]);
    for strategy in [None, Some("pool"), Some("data-parallel"), Some("auto")] {
        let selection = select_mesh_devices_with(&requirements(None, strategy), &info, Some(&map))
            .expect("a visible device exists");
        assert_eq!(selected_ids(&selection), vec![0], "strategy {strategy:?}");
        assert_eq!(selection.logical_provider, "cuda");
    }
    // Without any manifest preference (provider auto-selection) too.
    let selection = select_mesh_devices_with(
        &kapsl_core::HardwareRequirements::default(),
        &info,
        Some(&map),
    )
    .expect("auto selection");
    assert!(selection
        .devices
        .iter()
        .filter(|device| device.backend.to_string() == "cuda")
        .all(|device| device.id == 0));
    // Without a map nothing is filtered (library use, non-CUDA hosts).
    let all = select_mesh_devices_with(&requirements(None, None), &info, None).unwrap();
    assert_eq!(selected_ids(&all), vec![0, 1]);
}

#[test]
fn cvd_1_maps_engine_device_1_to_ordinal_0_and_denies_device_0() {
    let info = host_with_two_gpus();
    let map = map_for(&info, &[PCI_1]);
    assert_eq!(map.ordinal_or_error(1), Ok(0), "engine 1 is CUDA ordinal 0");

    let selection = select_mesh_devices_with(&requirements(Some(1), None), &info, Some(&map))
        .expect("device 1 is the visible one");
    assert_eq!(selected_ids(&selection), vec![1]);

    let error = select_mesh_devices_with(&requirements(Some(0), None), &info, Some(&map))
        .expect_err("device 0 is hidden");
    assert!(error.contains("CUDA device 0"), "{error}");
    assert!(error.contains("[1]"), "names the usable devices: {error}");
    assert!(
        map.ordinal_or_error(0).is_err(),
        "hidden device 0 never becomes ordinal 0"
    );
}

#[test]
fn cvd_1_with_no_requested_device_is_placed_on_engine_device_1_only() {
    let info = host_with_two_gpus();
    let map = map_for(&info, &[PCI_1]);
    for strategy in [None, Some("pool"), Some("data-parallel")] {
        let selection = select_mesh_devices_with(&requirements(None, strategy), &info, Some(&map))
            .expect("a visible device exists");
        assert_eq!(selected_ids(&selection), vec![1], "strategy {strategy:?}");
    }
}

#[test]
fn both_devices_visible_keep_both_in_the_mesh_and_honour_the_request() {
    let info = host_with_two_gpus();
    let map = map_for(&info, &[PCI_0, PCI_1]);
    let pool = select_mesh_devices_with(&requirements(None, Some("pool")), &info, Some(&map))
        .expect("pool");
    assert_eq!(selected_ids(&pool), vec![0, 1]);
    let one = select_mesh_devices_with(&requirements(Some(1), None), &info, Some(&map)).unwrap();
    assert_eq!(selected_ids(&one), vec![1]);
}

#[test]
fn a_process_that_sees_no_gpu_denies_every_cuda_request_and_falls_back_to_cpu_otherwise() {
    // CUDA_VISIBLE_DEVICES="": the runtime enumerates nothing.
    let mut info = host_with_two_gpus();
    let map = map_for(&info, &[]);
    assert!(select_mesh_devices_with(&requirements(Some(0), None), &info, Some(&map)).is_err());
    info.devices.push(Device {
        id: 0,
        name: "cpu".to_string(),
        backend: DeviceBackend::Cpu,
        ..cuda(0, PCI_0)
    });
    let selection = select_mesh_devices_with(
        &kapsl_core::HardwareRequirements {
            preferred_provider: Some("cpu".to_string()),
            ..kapsl_core::HardwareRequirements::default()
        },
        &info,
        Some(&map),
    )
    .expect("a CPU model does not need a CUDA device");
    assert!(selection
        .devices
        .iter()
        .all(|device| device.backend == DeviceBackend::Cpu));
}

#[test]
fn an_unmatched_device_is_never_a_placement_target() {
    // Two engine devices with one PCI id (for example MIG instances) cannot be
    // told apart, so neither is managed or selectable.
    let mut info = host_with_two_gpus();
    info.devices[1].pci_bus_id = Some(PCI_0.to_string());
    let map = map_for(&info, &[PCI_0]);
    let error = select_mesh_devices_with(&requirements(Some(1), None), &info, Some(&map))
        .expect_err("unmatched device 1");
    assert!(error.contains("could not be matched"), "{error}");
}
