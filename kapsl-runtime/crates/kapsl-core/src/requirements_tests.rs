#[cfg(test)]
mod tests {
    use super::super::HardwareRequirements;

    #[test]
    fn test_default_requirements_empty() {
        let req = HardwareRequirements::default();
        assert!(req.preferred_provider.is_none());
        assert!(req.fallback_providers.is_empty());
        assert!(req.min_memory_mb.is_none());
        assert!(req.min_vram_mb.is_none());
        assert!(req.min_cuda_version.is_none());
        assert!(req.required_precision.is_none());
        assert!(req.optimized_for.is_empty());
        assert!(req.graph_optimization_level.is_none());
        assert!(req.device_id.is_none());
        assert!(req.strategy.is_none());
    }

    #[test]
    fn test_cpu_only_requirements() {
        let req = HardwareRequirements::cpu_only();
        assert_eq!(req.preferred_provider.as_deref(), Some("cpu"));
        assert!(req.fallback_providers.is_empty());
        assert_eq!(req.required_precision.as_deref(), Some("fp32"));
        assert_eq!(req.graph_optimization_level.as_deref(), Some("all"));
        assert_eq!(req.device_id, Some(0));
        assert_eq!(req.strategy.as_deref(), Some("round-robin"));
    }

    #[test]
    fn test_gpu_with_cpu_fallback_requirements() {
        let req = HardwareRequirements::gpu_with_cpu_fallback("cuda");
        assert_eq!(req.preferred_provider.as_deref(), Some("cuda"));
        assert_eq!(req.fallback_providers, vec!["cpu".to_string()]);
        assert_eq!(req.required_precision.as_deref(), Some("fp32"));
        assert_eq!(req.graph_optimization_level.as_deref(), Some("all"));
        assert_eq!(req.device_id, Some(0));
        assert_eq!(req.strategy.as_deref(), Some("round-robin"));
    }
}
