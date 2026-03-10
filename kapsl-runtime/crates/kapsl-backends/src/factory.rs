use crate::onnx::{ExecutionProvider, OnnxBackend, OnnxBackendBuilder};
use kapsl_core::loader::Manifest;
use kapsl_core::HardwareRequirements;
use kapsl_engine_api::Engine;
use kapsl_hal::device::DeviceInfo;
use kapsl_llm::llm_backend::LLMBackend;
use kapsl_llm::GgufBackend;
#[cfg(target_os = "windows")]
use ort::execution_providers::DirectMLExecutionProvider;
use ort::execution_providers::ExecutionProvider as _;
use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, OpenVINOExecutionProvider,
    ROCmExecutionProvider, TensorRTExecutionProvider,
};
use ort::session::builder::GraphOptimizationLevel;

pub struct BackendFactory;

#[derive(Debug, Clone, Default)]
pub struct OnnxRuntimeTuning {
    pub memory_pattern: Option<bool>,
    pub disable_cpu_mem_arena: Option<bool>,
    pub session_buckets: Option<usize>,
    pub bucket_dim_granularity: Option<usize>,
    pub bucket_max_dims: Option<usize>,
    pub peak_concurrency_hint: Option<u32>,
}

pub fn parse_optimization_level(level: Option<&String>) -> Result<GraphOptimizationLevel, String> {
    match level.as_ref().map(|s| s.as_str()) {
        Some("disable") | Some("0") => Ok(GraphOptimizationLevel::Disable),
        Some("basic") | Some("1") => Ok(GraphOptimizationLevel::Level1),
        Some("extended") | Some("2") => Ok(GraphOptimizationLevel::Level2),
        Some("all") | Some("3") | Some("99") | None => Ok(GraphOptimizationLevel::Level3),
        _ => Err("Unknown optimization level".to_string()),
    }
}

impl BackendFactory {
    fn apply_onnx_tuning(
        mut builder: OnnxBackendBuilder,
        tuning: &OnnxRuntimeTuning,
    ) -> OnnxBackendBuilder {
        if let Some(v) = tuning.memory_pattern {
            builder = builder.with_memory_pattern(v);
        }
        if let Some(v) = tuning.disable_cpu_mem_arena {
            builder = builder.with_disable_cpu_mem_arena(v);
        }
        if let Some(v) = tuning.session_buckets {
            builder = builder.with_max_bucket_sessions(v);
        }
        if let Some(v) = tuning.bucket_dim_granularity {
            builder = builder.with_bucket_dim_granularity(v);
        }
        if let Some(v) = tuning.bucket_max_dims {
            builder = builder.with_bucket_max_dims(v);
        }
        if let Some(v) = tuning.peak_concurrency_hint {
            builder = builder.with_peak_concurrency_hint(v);
        }
        builder
    }

    fn build_onnx_backend(
        provider: ExecutionProvider,
        opt_level: GraphOptimizationLevel,
        device_id: i32,
        tuning: &OnnxRuntimeTuning,
    ) -> Result<Box<dyn Engine>, String> {
        let mut builder = OnnxBackend::builder()
            .with_provider(provider)
            .with_optimization_level(opt_level);
        if !matches!(provider, ExecutionProvider::CPU) {
            builder = builder.with_device_id(device_id)?;
        }
        builder = Self::apply_onnx_tuning(builder, tuning);
        Ok(Box::new(builder.build()))
    }

    fn push_unique_provider(providers: &mut Vec<String>, provider: &str) {
        if providers
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(provider))
        {
            return;
        }
        providers.push(provider.to_string());
    }

    fn provider_policy() -> String {
        std::env::var("KAPSL_PROVIDER_POLICY")
            .or_else(|_| std::env::var("KAPSL_PROVIDER_POLICY"))
            .unwrap_or_else(|_| "fastest".to_string())
            .trim()
            .to_ascii_lowercase()
    }

    fn should_append_fastest_candidates(providers: &[String]) -> bool {
        if Self::provider_policy() == "manifest" {
            return false;
        }

        providers.is_empty()
            || providers
                .iter()
                .all(|provider| matches!(provider.trim().to_ascii_lowercase().as_str(), "" | "cpu"))
    }

    fn append_fastest_candidates(device_info: &DeviceInfo, providers: &mut Vec<String>) {
        if device_info.has_cuda {
            Self::push_unique_provider(providers, "tensorrt");
            Self::push_unique_provider(providers, "cuda");
        }
        if device_info.has_metal {
            Self::push_unique_provider(providers, "coreml");
        }
        if device_info.has_rocm {
            Self::push_unique_provider(providers, "rocm");
        }
        if device_info.has_directml {
            Self::push_unique_provider(providers, "directml");
        }
        Self::push_unique_provider(providers, "cpu");
    }

    /// Create the best available backend based on manifest requirements and available hardware
    pub fn create_best_backend(
        manifest: &Manifest,
        device_info: &DeviceInfo,
    ) -> Result<Box<dyn Engine>, String> {
        Self::create_best_backend_with_tuning(manifest, device_info, &OnnxRuntimeTuning::default())
    }

    pub fn create_best_backend_with_tuning(
        manifest: &Manifest,
        device_info: &DeviceInfo,
        tuning: &OnnxRuntimeTuning,
    ) -> Result<Box<dyn Engine>, String> {
        // GGUF: route to the llama.cpp-backed GgufBackend
        if manifest.framework == "gguf" {
            log::info!("✓ Using GgufBackend (llama.cpp)");
            return Ok(Box::new(GgufBackend::new()));
        }

        // Check for LLM framework
        if manifest.framework == "llm" {
            let requirements = &manifest.hardware_requirements;
            if Self::provider_policy() == "manifest" {
                if let Some(provider) = requirements.preferred_provider.clone() {
                    let device_id = requirements.device_id.unwrap_or(0);
                    log::info!(
                        "✓ Using LLMBackend with manifest provider override: {}",
                        provider
                    );
                    return Ok(Box::new(LLMBackend::with_device(provider, device_id)));
                }
            }
            log::info!("✓ Using LLMBackend with runtime fastest-provider selection");
            return Ok(Box::new(LLMBackend::new()));
        }

        let requirements = &manifest.hardware_requirements;

        log::info!("🔍 Selecting backend based on requirements:");
        log::info!("   Preferred: {:?}", requirements.preferred_provider);
        log::info!("   Fallbacks: {:?}", requirements.fallback_providers);
        log::info!(
            "   Graph Optimization: {:?}",
            requirements.graph_optimization_level
        );

        // Parse and validate optimization level early (fail-fast)
        let opt_level = parse_optimization_level(requirements.graph_optimization_level.as_ref())
            .map_err(|e| format!("Invalid graph optimization level in manifest: {}", e))?;

        log::info!("   Graph Optimization: {:?}", opt_level);

        let mut providers_to_try = Vec::new();
        if let Some(preferred) = &requirements.preferred_provider {
            Self::push_unique_provider(&mut providers_to_try, preferred);
        }
        for provider in &requirements.fallback_providers {
            Self::push_unique_provider(&mut providers_to_try, provider);
        }

        if Self::should_append_fastest_candidates(&providers_to_try) {
            log::info!("⚡ Provider policy `fastest`: appending hardware-accelerated providers");
            Self::append_fastest_candidates(device_info, &mut providers_to_try);
        }

        for provider in &providers_to_try {
            let device_id = requirements.device_id.unwrap_or(0);
            match Self::try_create_provider(provider, device_info, opt_level, device_id, tuning) {
                Ok(backend) => {
                    log::info!("✓ Using provider: {}", provider);
                    return Ok(backend);
                }
                Err(err) => {
                    log::warn!("⚠ Provider '{}' not available: {}", provider, err);
                }
            }
        }

        // Last resort: CPU
        log::info!("⚠ Using last-resort CPU backend");
        let opt_cpu = parse_optimization_level(requirements.graph_optimization_level.as_ref())
            .unwrap_or(GraphOptimizationLevel::Level3);
        Self::build_onnx_backend(ExecutionProvider::CPU, opt_cpu, 0, tuning)
    }

    /// Create a backend for a specific device
    pub fn create_backend_for_device(
        manifest: &Manifest,
        provider: &str,
        device_id: usize,
        device_info: &DeviceInfo,
    ) -> Result<Box<dyn Engine>, String> {
        Self::create_backend_for_device_with_tuning(
            manifest,
            provider,
            device_id,
            device_info,
            &OnnxRuntimeTuning::default(),
        )
    }

    pub fn create_backend_for_device_with_tuning(
        manifest: &Manifest,
        provider: &str,
        device_id: usize,
        device_info: &DeviceInfo,
        tuning: &OnnxRuntimeTuning,
    ) -> Result<Box<dyn Engine>, String> {
        // GGUF: route to the llama.cpp-backed GgufBackend
        if manifest.framework == "gguf" {
            log::info!("✓ Using GgufBackend (llama.cpp)");
            return Ok(Box::new(GgufBackend::new()));
        }

        // Check for LLM framework
        if manifest.framework == "llm" {
            if Self::provider_policy() == "manifest" {
                log::info!(
                    "✓ Using LLMBackend with manifest provider override: {}",
                    provider
                );
                return Ok(Box::new(LLMBackend::with_device(
                    provider.to_string(),
                    device_id as i32,
                )));
            }

            log::info!(
                "✓ Using LLMBackend with device pinning and runtime provider auto-selection"
            );
            return Ok(Box::new(LLMBackend::with_device_id(device_id as i32)));
        }

        let requirements = &manifest.hardware_requirements;
        let opt_level = parse_optimization_level(requirements.graph_optimization_level.as_ref())
            .map_err(|e| format!("Invalid graph optimization level in manifest: {}", e))?;

        Self::try_create_provider(provider, device_info, opt_level, device_id as i32, tuning)
    }

    fn try_create_provider(
        provider: &str,
        device_info: &DeviceInfo,
        opt_level: GraphOptimizationLevel,
        device_id: i32,
        tuning: &OnnxRuntimeTuning,
    ) -> Result<Box<dyn Engine>, String> {
        let provider_lower = provider.to_lowercase();

        match provider_lower.as_str() {
            "cuda" => {
                if !device_info.has_cuda {
                    return Err("CUDA not available on this system".to_string());
                }
                if !CUDAExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err(
                        "CUDA execution provider is not available in ONNX Runtime".to_string()
                    );
                }
                let cuda_version = device_info
                    .devices
                    .iter()
                    .find(|d| matches!(d.backend, kapsl_hal::device::DeviceBackend::Cuda))
                    .and_then(|d| d.cuda_version.as_ref())
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                log::info!("   CUDA available: version {:?}", cuda_version);
                Self::build_onnx_backend(ExecutionProvider::CUDA, opt_level, device_id, tuning)
            }

            "tensorrt" => {
                if !device_info.has_cuda {
                    return Err("TensorRT requires CUDA-capable GPU".to_string());
                }
                if !TensorRTExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err(
                        "TensorRT execution provider is not available in ONNX Runtime".to_string(),
                    );
                }
                log::info!("   TensorRT requested (requires CUDA)");
                Self::build_onnx_backend(ExecutionProvider::TensorRT, opt_level, device_id, tuning)
            }

            "metal" => {
                if !device_info.has_metal {
                    return Err("Metal not available on this system".to_string());
                }
                if !CoreMLExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err("CoreML execution provider is not available".to_string());
                }
                log::info!("   Metal available on macOS");
                log::info!("   Using CoreML execution provider for Metal");
                Self::build_onnx_backend(ExecutionProvider::CoreML, opt_level, device_id, tuning)
            }
            "coreml" => {
                if !device_info.has_metal {
                    return Err("CoreML not available on this system".to_string());
                }
                if !CoreMLExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err("CoreML execution provider is not available".to_string());
                }
                log::info!("   CoreML available on macOS");
                Self::build_onnx_backend(ExecutionProvider::CoreML, opt_level, device_id, tuning)
            }
            "rocm" => {
                if !device_info.has_rocm {
                    return Err("ROCm not available on this system".to_string());
                }
                if !ROCmExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err("ROCm execution provider is not available".to_string());
                }
                log::info!("   ROCm available");
                Self::build_onnx_backend(ExecutionProvider::ROCm, opt_level, device_id, tuning)
            }
            "directml" => {
                #[cfg(target_os = "windows")]
                {
                    if !device_info.has_directml {
                        return Err("DirectML not available on this system".to_string());
                    }
                    if !DirectMLExecutionProvider::default()
                        .is_available()
                        .unwrap_or(false)
                    {
                        return Err("DirectML execution provider is not available".to_string());
                    }
                    log::info!("   DirectML available");
                    Self::build_onnx_backend(
                        ExecutionProvider::DirectML,
                        opt_level,
                        device_id,
                        tuning,
                    )
                }
                #[cfg(not(target_os = "windows"))]
                {
                    Err("DirectML is only supported on Windows".to_string())
                }
            }
            "openvino" => {
                if !OpenVINOExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err("OpenVINO execution provider is not available".to_string());
                }
                log::info!("   OpenVINO available");
                Self::build_onnx_backend(ExecutionProvider::OpenVINO, opt_level, device_id, tuning)
            }

            "cpu" => {
                log::info!("   Using CPU execution");
                Self::build_onnx_backend(ExecutionProvider::CPU, opt_level, 0, tuning)
            }

            _ => Err(format!("Unknown provider: {}", provider)),
        }
    }

    /// Validate that hardware meets minimum requirements
    pub fn validate_requirements(
        requirements: &HardwareRequirements,
        device_info: &DeviceInfo,
    ) -> Result<(), String> {
        // Validation logic for CPU memory
        if let Some(min_mem_mb) = requirements.min_memory_mb {
            let available_mb = device_info.total_memory / (1024 * 1024);
            if available_mb < min_mem_mb {
                return Err(format!(
                    "Insufficient memory: need {}MB, have {}MB",
                    min_mem_mb, available_mb
                ));
            }
        }

        // Collect all providers to check (preferred + fallbacks)
        let mut providers_to_check = Vec::new();
        if let Some(preferred) = &requirements.preferred_provider {
            providers_to_check.push(preferred.clone());
        }
        providers_to_check.extend(requirements.fallback_providers.clone());

        // We only fail if NONE of the providers are valid/present
        let mut reasons = Vec::new();
        let mut has_valid_provider = false;

        let strategy = requirements
            .strategy
            .as_deref()
            .unwrap_or("")
            .to_ascii_lowercase();
        let allow_multi = matches!(
            strategy.as_str(),
            "pool"
                | "round-robin"
                | "data-parallel"
                | "pipeline"
                | "pipeline-parallel"
                | "tensor-parallel"
                | "auto"
        );

        for provider in &providers_to_check {
            let provider_lower = provider.to_lowercase();
            let backend_key = match provider_lower.as_str() {
                "tensorrt" => "cuda",
                "coreml" => "metal",
                other => other,
            };
            let is_cpu = backend_key == "cpu";

            if is_cpu {
                // CPU is always valid if memory check passed (which is global above, though strictly
                // memory check should maybe be per-provider if requirements differed, but here it's global)
                has_valid_provider = true;
                break;
            }

            // GPU checks
            if !device_info.has_provider(backend_key) {
                reasons.push(format!("Provider {} not available", provider));
                continue;
            }

            let device_meets = |device: &kapsl_hal::device::Device| -> bool {
                if backend_key != "cpu" {
                    if let Some(min_vram) = requirements.min_vram_mb {
                        if device.memory_mb < min_vram {
                            return false;
                        }
                    }
                    if backend_key == "cuda" {
                        if let Some(min_ver) = &requirements.min_cuda_version {
                            if let Some(dev_ver) = &device.cuda_version {
                                if dev_ver < min_ver {
                                    return false;
                                }
                            } else {
                                return false;
                            }
                        }
                    }
                }
                true
            };

            if allow_multi {
                let mut candidates = device_info
                    .devices
                    .iter()
                    .filter(|d| d.backend.to_string().to_lowercase() == backend_key);

                if candidates.any(device_meets) {
                    has_valid_provider = true;
                    break;
                }

                reasons.push(format!(
                    "No devices meet requirements for provider {}",
                    provider
                ));
                continue;
            }

            // Find the device
            let dev_id = requirements.device_id.unwrap_or(0) as usize;
            // Note: device_id 0 is usually the first GPU if provider is GPU.
            if let Some(device) = device_info
                .devices
                .iter()
                .find(|d| d.id == dev_id && d.backend.to_string().to_lowercase() == backend_key)
            {
                // Check VRAM
                if let Some(min_vram) = requirements.min_vram_mb {
                    if device.memory_mb < min_vram {
                        reasons.push(format!(
                            "Provider {} (Device {}) has insufficient VRAM: {}MB < required {}MB",
                            provider, dev_id, device.memory_mb, min_vram
                        ));
                        continue;
                    }
                }

                // Check CUDA version
                if backend_key == "cuda" {
                    if let Some(min_ver) = &requirements.min_cuda_version {
                        if let Some(dev_ver) = &device.cuda_version {
                            if dev_ver < min_ver {
                                reasons.push(format!(
                                    "CUDA version too old: {} < required {}",
                                    dev_ver, min_ver
                                ));
                                continue;
                            }
                        } else {
                            reasons.push("Unknown CUDA version on device".to_string());
                            continue;
                        }
                    }
                }

                has_valid_provider = true;
                break;
            } else {
                reasons.push(format!(
                    "Device ID {} not found for provider {}",
                    dev_id, provider
                ));
            }
        }

        if !has_valid_provider {
            if providers_to_check.is_empty() {
                // No requirements?
                return Ok(());
            }
            // If we have CPU in list and it wasn't caught above, it means something weird happened.
            // But usually CPU works.
            return Err(format!(
                "No compatible provider found. Reasons: {:?}",
                reasons
            ));
        }

        Ok(())
    }
}

#[path = "factory_tests.rs"]
mod factory_tests;
