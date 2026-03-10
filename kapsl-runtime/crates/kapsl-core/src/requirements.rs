use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareRequirements {
    /// Preferred execution provider (e.g., "cuda", "metal", "coreml", "cpu")
    #[serde(default)]
    pub preferred_provider: Option<String>,

    /// Fallback providers in order of preference
    #[serde(default)]
    pub fallback_providers: Vec<String>,

    /// Minimum memory required in MB
    #[serde(default)]
    pub min_memory_mb: Option<u64>,

    /// Minimum VRAM required in MB (for GPU providers)
    #[serde(default)]
    pub min_vram_mb: Option<u64>,

    /// Minimum CUDA version required (e.g., "11.8")
    #[serde(default)]
    pub min_cuda_version: Option<String>,

    /// Required precision (e.g., "fp32", "fp16", "int8")
    #[serde(default)]
    pub required_precision: Option<String>,

    /// Optimizations this model has (e.g., "cuda_11.8", "tensorrt_8.6")
    #[serde(default)]
    pub optimized_for: Vec<String>,

    /// Graph optimization level for ONNX Runtime
    /// Valid values: "disable" (0), "basic" (1), "extended" (2), "all" (99)
    /// Maps to ort::GraphOptimizationLevel
    #[serde(default)]
    pub graph_optimization_level: Option<String>,

    #[serde(default)]
    pub device_id: Option<i32>,

    #[serde(default)]
    pub strategy: Option<String>,
}

impl HardwareRequirements {
    /// Create CPU-only requirements (default fallback)
    pub fn cpu_only() -> Self {
        Self {
            preferred_provider: Some("cpu".to_string()),
            fallback_providers: vec![],
            min_memory_mb: None,
            min_vram_mb: None,
            min_cuda_version: None,
            required_precision: Some("fp32".to_string()),
            optimized_for: vec![],
            graph_optimization_level: Some("all".to_string()), // Default to max optimization
            device_id: Some(0),
            strategy: Some("round-robin".to_string()),
        }
    }

    /// Create requirements with GPU preference and CPU fallback
    pub fn gpu_with_cpu_fallback(gpu_type: &str) -> Self {
        Self {
            preferred_provider: Some(gpu_type.to_string()),
            fallback_providers: vec!["cpu".to_string()],
            min_memory_mb: None,
            min_vram_mb: None,
            min_cuda_version: None,
            required_precision: Some("fp32".to_string()),
            optimized_for: vec![],
            graph_optimization_level: Some("all".to_string()), // Default to max optimization
            device_id: Some(0),
            strategy: Some("round-robin".to_string()),
        }
    }
}

#[path = "requirements_tests.rs"]
mod requirements_tests;
