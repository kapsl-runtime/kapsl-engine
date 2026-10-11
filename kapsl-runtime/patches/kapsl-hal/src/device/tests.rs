//! Tests for hardware discovery data and device selection.

use crate::device::{Device, DeviceBackend, DeviceInfo, GpuPreference};

fn make_device(id: usize, backend: DeviceBackend, memory_mb: u64) -> Device {
    Device {
        id,
        name: format!("dev_{id}"),
        backend,
        memory_mb,
        compute_units: 16,
        pci_bus_id: None,
        partition_id: None,
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

// ── GpuPreference::parse ─────────────────────────────────────────────────────

#[test]
fn gpu_preference_parse_backend_id() {
    let pref = GpuPreference::parse("cuda:2").unwrap();
    assert_eq!(
        pref,
        GpuPreference::BackendId {
            backend: "cuda".to_string(),
            id: 2
        }
    );
}

#[test]
fn gpu_preference_parse_pci_bus_id() {
    let pref = GpuPreference::parse("0000:01:00.0").unwrap();
    assert_eq!(pref, GpuPreference::PciBusId("0000:01:00.0".to_string()));
}

#[test]
fn gpu_preference_parse_name_contains() {
    let pref = GpuPreference::parse("A100").unwrap();
    assert_eq!(pref, GpuPreference::NameContains("a100".to_string()));
}

#[test]
fn gpu_preference_parse_mig_prefix() {
    let pref = GpuPreference::parse("mig:GPU-abc123/0/0").unwrap();
    assert_eq!(pref, GpuPreference::Partition("GPU-abc123/0/0".to_string()));
}

#[test]
fn gpu_preference_parse_partition_prefix() {
    let pref = GpuPreference::parse("partition:GPU-xyz/1/0").unwrap();
    assert_eq!(pref, GpuPreference::Partition("GPU-xyz/1/0".to_string()));
}

#[test]
fn gpu_preference_parse_empty_returns_none() {
    assert!(GpuPreference::parse("").is_none());
    assert!(GpuPreference::parse("  ").is_none());
}

// ── GpuPreference::matches ────────────────────────────────────────────────────

#[test]
fn gpu_preference_matches_backend_id() {
    let mut dev = make_device(1, DeviceBackend::Cuda, 8_000);
    dev.pci_bus_id = Some("0000:01:00.0".to_string());

    let pref = GpuPreference::BackendId {
        backend: "cuda".to_string(),
        id: 1,
    };
    assert!(pref.matches(&dev));

    let wrong_id = GpuPreference::BackendId {
        backend: "cuda".to_string(),
        id: 2,
    };
    assert!(!wrong_id.matches(&dev));
}

#[test]
fn gpu_preference_matches_pci_bus_id() {
    let mut dev = make_device(0, DeviceBackend::Cuda, 8_000);
    dev.pci_bus_id = Some("0000:03:00.0".to_string());

    let pref = GpuPreference::PciBusId("0000:03:00.0".to_string());
    assert!(pref.matches(&dev));

    // Case-insensitive
    let pref_upper = GpuPreference::PciBusId("0000:03:00.0".to_ascii_uppercase());
    assert!(pref_upper.matches(&dev));

    // Different bus
    assert!(!GpuPreference::PciBusId("0000:04:00.0".to_string()).matches(&dev));

    // Missing pci_bus_id
    let no_pci = make_device(0, DeviceBackend::Cuda, 8_000);
    assert!(!pref.matches(&no_pci));
}

#[test]
fn gpu_preference_matches_name_contains() {
    let mut dev = make_device(0, DeviceBackend::Cuda, 8_000);
    dev.name = "NVIDIA A100 80GB PCIe".to_string();

    assert!(GpuPreference::NameContains("a100".to_string()).matches(&dev));
    assert!(GpuPreference::NameContains("80GB".to_string()).matches(&dev));
    assert!(!GpuPreference::NameContains("H100".to_string()).matches(&dev));
}

#[test]
fn gpu_preference_matches_partition() {
    let mut dev = make_device(0, DeviceBackend::Cuda, 8_000);
    dev.partition_id = Some("GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".to_string());

    let pref = GpuPreference::Partition("GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".to_string());
    assert!(pref.matches(&dev));

    // Case-insensitive
    let pref_upper =
        GpuPreference::Partition("GPU-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX".to_string());
    assert!(pref_upper.matches(&dev));

    assert!(!GpuPreference::Partition("GPU-other".to_string()).matches(&dev));

    // No partition_id on device
    let no_part = make_device(0, DeviceBackend::Cuda, 8_000);
    assert!(!pref.matches(&no_part));
}

#[test]
fn best_gpu_with_preference_partition() {
    let mut dev0 = make_device(0, DeviceBackend::Cuda, 8_000);
    dev0.partition_id = Some("GPU-aaa".to_string());
    let mut dev1 = make_device(1, DeviceBackend::Cuda, 8_000);
    dev1.partition_id = Some("GPU-bbb".to_string());
    let info = make_info(vec![make_device(0, DeviceBackend::Cpu, 1024), dev0, dev1]);

    let found = info
        .best_gpu_with_preference(&GpuPreference::Partition("GPU-bbb".to_string()))
        .expect("should find partition");
    assert_eq!(found.id, 1);

    assert!(info
        .best_gpu_with_preference(&GpuPreference::Partition("GPU-ccc".to_string()))
        .is_none());
}
