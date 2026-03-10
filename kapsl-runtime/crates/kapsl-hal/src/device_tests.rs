use crate::device::{Device, DeviceBackend, DeviceInfo};

fn make_device(id: usize, backend: DeviceBackend, memory_mb: u64) -> Device {
    Device {
        id,
        name: format!("dev_{id}"),
        backend,
        memory_mb,
        compute_units: 16,
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

fn make_info(devices: Vec<Device>) -> DeviceInfo {
    DeviceInfo {
        cpu_cores: 8,
        total_memory: 32 * 1024,
        os_type: "test".to_string(),
        os_release: "1.0".to_string(),
        has_cuda: devices
            .iter()
            .any(|d| matches!(d.backend, DeviceBackend::Cuda)),
        has_metal: devices
            .iter()
            .any(|d| matches!(d.backend, DeviceBackend::Metal)),
        has_rocm: devices
            .iter()
            .any(|d| matches!(d.backend, DeviceBackend::Rocm)),
        has_directml: devices
            .iter()
            .any(|d| matches!(d.backend, DeviceBackend::DirectML)),
        devices,
    }
}

#[test]
fn device_backend_display_matches_expected_names() {
    assert_eq!(DeviceBackend::Cpu.to_string(), "cpu");
    assert_eq!(DeviceBackend::Cuda.to_string(), "cuda");
    assert_eq!(DeviceBackend::Metal.to_string(), "metal");
    assert_eq!(DeviceBackend::Rocm.to_string(), "rocm");
    assert_eq!(DeviceBackend::DirectML.to_string(), "directml");
    assert_eq!(DeviceBackend::OpenCL.to_string(), "opencl");
    assert_eq!(DeviceBackend::Vulkan.to_string(), "vulkan");
    assert_eq!(DeviceBackend::WebGpu.to_string(), "webgpu");
    assert_eq!(DeviceBackend::OneApi.to_string(), "oneapi");
    assert_eq!(DeviceBackend::Custom("foo".to_string()).to_string(), "foo");
}

#[test]
fn best_gpu_picks_highest_memory_non_cpu() {
    let devices = vec![
        make_device(0, DeviceBackend::Cpu, 1024),
        make_device(1, DeviceBackend::Cuda, 12_000),
        make_device(2, DeviceBackend::Metal, 8_000),
    ];
    let info = make_info(devices);

    let best = info.best_gpu().expect("expected a gpu");
    assert_eq!(best.id, 1);
}

#[test]
fn best_gpu_none_when_only_cpu() {
    let devices = vec![make_device(0, DeviceBackend::Cpu, 1024)];
    let info = make_info(devices);

    assert!(info.best_gpu().is_none());
}

#[test]
fn cuda_devices_filters_only_cuda() {
    let devices = vec![
        make_device(0, DeviceBackend::Cpu, 1024),
        make_device(1, DeviceBackend::Cuda, 12_000),
        make_device(2, DeviceBackend::Cuda, 10_000),
        make_device(3, DeviceBackend::Metal, 8_000),
    ];
    let info = make_info(devices);

    let cuda = info.cuda_devices();
    assert_eq!(cuda.len(), 2);
    assert!(cuda
        .iter()
        .all(|d| matches!(d.backend, DeviceBackend::Cuda)));
}

#[test]
fn provider_helpers_use_flags_and_defaults() {
    let devices = vec![make_device(0, DeviceBackend::Cpu, 1024)];
    let mut info = make_info(devices);
    info.has_cuda = true;
    info.has_metal = true;

    assert_eq!(info.get_best_provider(), "cuda");
    assert!(info.has_provider("cuda"));
    assert!(info.has_provider("metal"));
    assert!(info.has_provider("coreml"));
    assert!(info.has_provider("cpu"));
    assert!(!info.has_provider("rocm"));
}
