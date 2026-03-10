#[cfg(test)]
mod tests {
    use super::super::BackendFactory;
    use kapsl_core::HardwareRequirements;
    use kapsl_hal::device::{Device, DeviceBackend, DeviceInfo};

    fn mock_device_info() -> DeviceInfo {
        DeviceInfo {
            cpu_cores: 8,
            total_memory: 16 * 1024 * 1024 * 1024, // 16GB
            os_type: "linux".to_string(),
            os_release: "5.15".to_string(),
            has_cuda: true,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: vec![
                Device {
                    id: 0,
                    name: "CPU".to_string(),
                    backend: DeviceBackend::Cpu,
                    memory_mb: 16 * 1024,
                    compute_units: 8,
                    pci_bus_id: None,
                    driver_version: None,
                    compute_capability: None,
                    utilization_gpu_pct: None,
                    temperature_c: None,
                    supports_fp16: true,
                    supports_int8: true,
                    cuda_version: None,
                },
                Device {
                    id: 1,
                    name: "NVIDIA RTX 3080".to_string(),
                    backend: DeviceBackend::Cuda,
                    memory_mb: 10 * 1024, // 10GB
                    compute_units: 0,
                    pci_bus_id: None,
                    driver_version: None,
                    compute_capability: None,
                    utilization_gpu_pct: None,
                    temperature_c: None,
                    supports_fp16: true,
                    supports_int8: true,
                    cuda_version: Some("12.0".to_string()),
                },
                Device {
                    id: 2,
                    name: "NVIDIA RTX 3060".to_string(),
                    backend: DeviceBackend::Cuda,
                    memory_mb: 8 * 1024, // 8GB
                    compute_units: 0,
                    pci_bus_id: None,
                    driver_version: None,
                    compute_capability: None,
                    utilization_gpu_pct: None,
                    temperature_c: None,
                    supports_fp16: true,
                    supports_int8: true,
                    cuda_version: Some("12.0".to_string()),
                },
            ],
        }
    }

    #[test]
    fn test_validate_memory_success() {
        let info = mock_device_info();
        let req = HardwareRequirements {
            min_memory_mb: Some(8 * 1024), // 8GB
            ..Default::default()
        };
        assert!(BackendFactory::validate_requirements(&req, &info).is_ok());
    }

    #[test]
    fn test_validate_memory_failure() {
        let info = mock_device_info();
        let req = HardwareRequirements {
            min_memory_mb: Some(32 * 1024), // 32GB (more than available)
            ..Default::default()
        };
        assert!(BackendFactory::validate_requirements(&req, &info).is_err());
    }

    #[test]
    fn test_validate_vram_success() {
        let info = mock_device_info();
        let req = HardwareRequirements {
            preferred_provider: Some("cuda".to_string()),
            device_id: Some(1),
            min_vram_mb: Some(8 * 1024), // 8GB < 10GB
            ..Default::default()
        };
        assert!(BackendFactory::validate_requirements(&req, &info).is_ok());
    }

    #[test]
    fn test_validate_vram_failure() {
        let info = mock_device_info();
        let req = HardwareRequirements {
            preferred_provider: Some("cuda".to_string()),
            device_id: Some(1),
            min_vram_mb: Some(12 * 1024), // 12GB > 10GB
            ..Default::default()
        };
        let res = BackendFactory::validate_requirements(&req, &info);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("insufficient VRAM"));
    }

    #[test]
    fn test_validate_cuda_version_success() {
        let info = mock_device_info();
        let req = HardwareRequirements {
            preferred_provider: Some("cuda".to_string()),
            device_id: Some(1),
            min_cuda_version: Some("11.8".to_string()), // 11.8 < 12.0
            ..Default::default()
        };
        assert!(BackendFactory::validate_requirements(&req, &info).is_ok());
    }

    #[test]
    fn test_validate_cuda_version_failure() {
        let info = mock_device_info();
        let req = HardwareRequirements {
            preferred_provider: Some("cuda".to_string()),
            device_id: Some(1),
            min_cuda_version: Some("12.1".to_string()), // 12.1 > 12.0
            ..Default::default()
        };
        let res = BackendFactory::validate_requirements(&req, &info);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("CUDA version too old"));
    }

    #[test]
    fn test_validate_missing_provider_failure() {
        let info = mock_device_info();
        let req = HardwareRequirements {
            preferred_provider: Some("metal".to_string()),
            ..Default::default()
        };
        let res = BackendFactory::validate_requirements(&req, &info);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Provider metal not available"));
    }

    #[test]
    fn test_validate_fallback_provider_success() {
        let info = mock_device_info();
        let req = HardwareRequirements {
            preferred_provider: Some("metal".to_string()),
            fallback_providers: vec!["cpu".to_string()],
            ..Default::default()
        };
        assert!(BackendFactory::validate_requirements(&req, &info).is_ok());
    }

    #[test]
    fn test_validate_device_id_not_found() {
        let info = mock_device_info();
        let req = HardwareRequirements {
            preferred_provider: Some("cuda".to_string()),
            device_id: Some(99),
            ..Default::default()
        };
        let res = BackendFactory::validate_requirements(&req, &info);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Device ID 99 not found"));
    }

    #[test]
    fn test_validate_multi_gpu_strategy_accepts_any_matching_device() {
        let info = mock_device_info();
        let req = HardwareRequirements {
            preferred_provider: Some("cuda".to_string()),
            min_vram_mb: Some(9 * 1024), // 9GB, only the 3080 qualifies
            strategy: Some("pool".to_string()),
            ..Default::default()
        };
        assert!(BackendFactory::validate_requirements(&req, &info).is_ok());
    }
}
