use async_trait::async_trait;
use half::f16;
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo, InferenceRequest,
    TensorDtype,
};
use ndarray::ArrayD;
use ort::execution_providers::ExecutionProvider as OrtExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::tensor::TensorElementType;
use ort::value::Value;
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;

// TODO: Consider adding support for OpenVINO and other backends
#[derive(Debug, Clone, Copy)]
pub enum ExecutionProvider {
    CPU,
    CUDA,
    TensorRT,
    DirectML,
    ROCm,
    OpenVINO,
    CoreML,
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub input_shapes: Vec<Vec<i64>>,
    pub output_shapes: Vec<Vec<i64>>,
    pub input_dtypes: Vec<Option<TensorDtype>>,
    pub output_dtypes: Vec<Option<TensorDtype>>,
}

pub struct OnnxBackend {
    session: Arc<RwLock<Option<Session>>>,
    bucket_sessions: Arc<RwLock<BucketSessionState>>,
    model_path: Arc<RwLock<Option<PathBuf>>>,
    provider: ExecutionProvider,
    optimization_level: u8,
    device_id: i32,
    memory_pattern: bool,
    disable_cpu_mem_arena: bool,
    max_bucket_sessions: usize,
    bucket_dim_granularity: usize,
    bucket_max_dims: usize,
    peak_concurrency_hint: Option<u32>,
    metrics: Arc<RwLock<EngineMetrics>>,
    metadata: Arc<RwLock<Option<ModelMetadata>>>,
    warmed_up: Arc<AtomicBool>,
}

#[derive(Default)]
struct BucketSessionState {
    primary_bucket_key: Option<String>,
    sessions: HashMap<String, Session>,
    lru: VecDeque<String>,
}

const ORT_MEMORY_PATTERN_ENV: &str = "KAPSL_ORT_MEMORY_PATTERN";
const ORT_DISABLE_CPU_MEM_ARENA_ENV: &str = "KAPSL_ORT_DISABLE_CPU_MEM_ARENA";
const ORT_SESSION_BUCKETS_ENV: &str = "KAPSL_ORT_SESSION_BUCKETS";
const ORT_BUCKET_DIM_GRANULARITY_ENV: &str = "KAPSL_ORT_BUCKET_DIM_GRANULARITY";
const ORT_BUCKET_MAX_DIMS_ENV: &str = "KAPSL_ORT_BUCKET_MAX_DIMS";
const MODEL_PEAK_CONCURRENCY_ENV: &str = "KAPSL_MODEL_PEAK_CONCURRENCY";
const ORT_SESSION_BUCKETS_MAX: usize = 64;

fn read_env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

fn read_env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

fn read_env_u32(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|value| *value > 0)
}

fn expose_sensitive_ids_in_logs() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("KAPSL_LOG_SENSITIVE_IDS")
            .or_else(|_| std::env::var("KAPSL_LOG_SENSITIVE_IDS"))
            .ok()
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn redact_session_id_for_log(session_id: &str) -> String {
    if expose_sensitive_ids_in_logs() || session_id.is_empty() {
        return session_id.to_string();
    }
    let prefix: String = session_id.chars().take(4).collect();
    format!("{}...[redacted]", prefix)
}

fn map_ort_dtype(dtype: TensorElementType) -> Option<TensorDtype> {
    match dtype {
        TensorElementType::Float32 => Some(TensorDtype::Float32),
        TensorElementType::Float64 => Some(TensorDtype::Float64),
        TensorElementType::Float16 => Some(TensorDtype::Float16),
        TensorElementType::Int32 => Some(TensorDtype::Int32),
        TensorElementType::Int64 => Some(TensorDtype::Int64),
        TensorElementType::Uint8 => Some(TensorDtype::Uint8),
        _ => None,
    }
}

// TODO: Review if manual unsafe impl is necessary - Session should already be Send + Sync
// TODO: Add documentation explaining thread safety guarantees
// Session is Send + Sync, so OnnxBackend is Send + Sync
unsafe impl Send for OnnxBackend {}
unsafe impl Sync for OnnxBackend {}

#[derive(Debug)]
pub struct OnnxBackendBuilder {
    provider: ExecutionProvider,
    optimization_level: GraphOptimizationLevel,
    device_id: i32,
    memory_pattern: bool,
    disable_cpu_mem_arena: bool,
    max_bucket_sessions: usize,
    bucket_dim_granularity: usize,
    bucket_max_dims: usize,
    peak_concurrency_hint: Option<u32>,
}

impl OnnxBackendBuilder {
    pub fn new() -> Self {
        let memory_pattern = read_env_flag(ORT_MEMORY_PATTERN_ENV, true);
        let disable_cpu_mem_arena = read_env_flag(ORT_DISABLE_CPU_MEM_ARENA_ENV, false);
        let max_bucket_sessions = read_env_usize(ORT_SESSION_BUCKETS_ENV)
            .unwrap_or(1)
            .clamp(1, ORT_SESSION_BUCKETS_MAX);
        let bucket_dim_granularity = read_env_usize(ORT_BUCKET_DIM_GRANULARITY_ENV)
            .unwrap_or(64)
            .max(1);
        let bucket_max_dims = read_env_usize(ORT_BUCKET_MAX_DIMS_ENV).unwrap_or(4).max(1);
        let peak_concurrency_hint = read_env_u32(MODEL_PEAK_CONCURRENCY_ENV);
        Self {
            provider: ExecutionProvider::CPU,
            optimization_level: GraphOptimizationLevel::Level3,
            device_id: 0,
            memory_pattern,
            disable_cpu_mem_arena,
            max_bucket_sessions,
            bucket_dim_granularity,
            bucket_max_dims,
            peak_concurrency_hint,
        }
    }

    pub fn with_provider(mut self, provider: ExecutionProvider) -> Self {
        self.provider = provider;
        self
    }

    pub fn with_optimization_level(mut self, opt_level: GraphOptimizationLevel) -> Self {
        self.optimization_level = opt_level;
        self
    }

    pub fn with_device_id(mut self, device_id: i32) -> Result<Self, String> {
        if device_id < 0 {
            return Err("Device ID must be non-negative".to_string());
        }
        self.device_id = device_id;
        Ok(self)
    }

    pub fn with_memory_pattern(mut self, enabled: bool) -> Self {
        self.memory_pattern = enabled;
        self
    }

    pub fn with_disable_cpu_mem_arena(mut self, disabled: bool) -> Self {
        self.disable_cpu_mem_arena = disabled;
        self
    }

    pub fn with_max_bucket_sessions(mut self, max_bucket_sessions: usize) -> Self {
        self.max_bucket_sessions = max_bucket_sessions.clamp(1, ORT_SESSION_BUCKETS_MAX);
        self
    }

    pub fn with_bucket_dim_granularity(mut self, bucket_dim_granularity: usize) -> Self {
        self.bucket_dim_granularity = bucket_dim_granularity.max(1);
        self
    }

    pub fn with_bucket_max_dims(mut self, bucket_max_dims: usize) -> Self {
        self.bucket_max_dims = bucket_max_dims.max(1);
        self
    }

    pub fn with_peak_concurrency_hint(mut self, peak_concurrency_hint: u32) -> Self {
        self.peak_concurrency_hint = Some(peak_concurrency_hint.max(1));
        self
    }

    pub fn build(self) -> OnnxBackend {
        let level_value = match self.optimization_level {
            GraphOptimizationLevel::Disable => 0,
            GraphOptimizationLevel::Level1 => 1,
            GraphOptimizationLevel::Level2 => 2,
            GraphOptimizationLevel::Level3 => 3,
            GraphOptimizationLevel::All => 4,
        };
        OnnxBackend {
            session: Arc::new(RwLock::new(None)),
            bucket_sessions: Arc::new(RwLock::new(BucketSessionState::default())),
            model_path: Arc::new(RwLock::new(None)),
            provider: self.provider,
            optimization_level: level_value,
            device_id: self.device_id,
            memory_pattern: self.memory_pattern,
            disable_cpu_mem_arena: self.disable_cpu_mem_arena,
            max_bucket_sessions: self.max_bucket_sessions,
            bucket_dim_granularity: self.bucket_dim_granularity,
            bucket_max_dims: self.bucket_max_dims,
            peak_concurrency_hint: self.peak_concurrency_hint,
            metrics: Arc::new(RwLock::new(EngineMetrics::default())),
            metadata: Arc::new(RwLock::new(None)),
            warmed_up: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Default for OnnxBackendBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxBackend {
    /// Convert stored u8 optimization level to GraphOptimizationLevel
    fn get_opt_level(&self) -> GraphOptimizationLevel {
        match self.optimization_level {
            0 => GraphOptimizationLevel::Disable,
            1 => GraphOptimizationLevel::Level1,
            2 => GraphOptimizationLevel::Level2,
            3 => GraphOptimizationLevel::Level3,
            _ => GraphOptimizationLevel::All,
        }
    }

    pub fn builder() -> OnnxBackendBuilder {
        OnnxBackendBuilder::new()
    }

    pub fn new_cpu() -> Self {
        Self::builder().build()
    }

    pub fn new_cpu_with_optimization(opt_level: GraphOptimizationLevel) -> Self {
        Self::builder().with_optimization_level(opt_level).build()
    }

    pub fn new_cuda(device_id: i32) -> Result<Self, String> {
        Self::new_cuda_with_optimization(GraphOptimizationLevel::Level3, device_id)
    }

    pub fn new_cuda_with_optimization(
        opt_level: GraphOptimizationLevel,
        device_id: i32,
    ) -> Result<Self, String> {
        Ok(Self::builder()
            .with_provider(ExecutionProvider::CUDA)
            .with_optimization_level(opt_level)
            .with_device_id(device_id)?
            .build())
    }

    pub fn new_tensorrt(device_id: i32) -> Result<Self, String> {
        Self::new_tensorrt_with_optimization(GraphOptimizationLevel::Level3, device_id)
    }

    pub fn new_tensorrt_with_optimization(
        opt_level: GraphOptimizationLevel,
        device_id: i32,
    ) -> Result<Self, String> {
        Ok(Self::builder()
            .with_provider(ExecutionProvider::TensorRT)
            .with_optimization_level(opt_level)
            .with_device_id(device_id)?
            .build())
    }

    pub fn new_directml(device_id: i32) -> Result<Self, String> {
        Self::new_directml_with_optimization(GraphOptimizationLevel::Level3, device_id)
    }

    pub fn new_directml_with_optimization(
        opt_level: GraphOptimizationLevel,
        device_id: i32,
    ) -> Result<Self, String> {
        Ok(Self::builder()
            .with_provider(ExecutionProvider::DirectML)
            .with_optimization_level(opt_level)
            .with_device_id(device_id)?
            .build())
    }

    pub fn new_rocm(device_id: i32) -> Result<Self, String> {
        Self::new_rocm_with_optimization(GraphOptimizationLevel::Level3, device_id)
    }

    pub fn new_rocm_with_optimization(
        opt_level: GraphOptimizationLevel,
        device_id: i32,
    ) -> Result<Self, String> {
        Ok(Self::builder()
            .with_provider(ExecutionProvider::ROCm)
            .with_optimization_level(opt_level)
            .with_device_id(device_id)?
            .build())
    }

    pub fn new_openvino_with_optimiation(
        opt_level: GraphOptimizationLevel,
        device_id: i32,
    ) -> Result<Self, String> {
        Ok(Self::builder()
            .with_provider(ExecutionProvider::OpenVINO)
            .with_optimization_level(opt_level)
            .with_device_id(device_id)?
            .build())
    }

    pub fn new_coreml_with_optimiation(
        opt_level: GraphOptimizationLevel,
        device_id: i32,
    ) -> Result<Self, String> {
        Ok(Self::builder()
            .with_provider(ExecutionProvider::CoreML)
            .with_optimization_level(opt_level)
            .with_device_id(device_id)?
            .build())
    }

    fn bucket_key_for_request(&self, request: &InferenceRequest) -> Option<String> {
        if self.max_bucket_sessions <= 1 {
            return None;
        }

        let mut key = format!(
            "{}:r{}",
            request.input.dtype.as_str(),
            request.input.shape.len()
        );
        for (index, dim) in request
            .input
            .shape
            .iter()
            .take(self.bucket_max_dims)
            .copied()
            .enumerate()
        {
            let rounded = if dim <= 0 {
                -1
            } else if index == 0 {
                dim
            } else {
                let granularity = self.bucket_dim_granularity as i64;
                ((dim + granularity - 1) / granularity) * granularity
            };
            key.push(':');
            key.push_str(&rounded.to_string());
        }
        if request.input.shape.len() > self.bucket_max_dims {
            key.push_str(":*");
        }
        Some(key)
    }

    fn touch_bucket_lru(state: &mut BucketSessionState, bucket_key: &str) {
        if let Some(pos) = state.lru.iter().position(|existing| existing == bucket_key) {
            state.lru.remove(pos);
        }
        state.lru.push_back(bucket_key.to_string());
    }

    fn get_or_create_bucket_session<'a>(
        &self,
        state: &'a mut BucketSessionState,
        bucket_key: &str,
    ) -> Result<&'a mut Session, EngineError> {
        if !state.sessions.contains_key(bucket_key) {
            let secondary_capacity = self.max_bucket_sessions.saturating_sub(1).max(1);
            while state.sessions.len() >= secondary_capacity {
                let Some(evict_key) = state.lru.pop_front() else {
                    break;
                };
                state.sessions.remove(&evict_key);
            }

            let model_path = self
                .model_path
                .read()
                .map_err(|_| EngineError::Backend {
                    message: "Lock poisoned".to_string(),
                    source: None,
                })?
                .clone()
                .ok_or(EngineError::ModelNotLoaded)?;
            let session = self.create_session(&model_path, self.get_opt_level())?;
            state.sessions.insert(bucket_key.to_string(), session);
        }

        Self::touch_bucket_lru(state, bucket_key);
        state
            .sessions
            .get_mut(bucket_key)
            .ok_or(EngineError::ModelNotLoaded)
    }

    fn create_session(
        &self,
        model_path: &Path,
        opt_level: GraphOptimizationLevel,
    ) -> Result<Session, EngineError> {
        // Common builder setup
        let mut builder = Session::builder()
            .map_err(|e| EngineError::ModelLoadError {
                path: model_path.to_string_lossy().into_owned(),
                source: Box::new(e),
            })?
            .with_optimization_level(opt_level)
            .map_err(|e| EngineError::ModelLoadError {
                path: model_path.to_string_lossy().into_owned(),
                source: Box::new(e),
            })?
            .with_memory_pattern(self.memory_pattern)
            .map_err(|e| EngineError::ModelLoadError {
                path: model_path.to_string_lossy().into_owned(),
                source: Box::new(e),
            })?;
        if self.disable_cpu_mem_arena {
            builder = builder
                .with_config_entry("session.disable_cpu_mem_arena", "1")
                .map_err(|e| EngineError::ModelLoadError {
                    path: model_path.to_string_lossy().into_owned(),
                    source: Box::new(e),
                })?;
        }

        // Configure execution providers based on the selected backend
        let builder = match self.provider {
            ExecutionProvider::CUDA => {
                if !ort::execution_providers::CUDAExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err(EngineError::Backend {
                        message:
                            "CUDA execution provider is not available. Please check your CUDA installation."
                                .to_string(),
                        source: None,
                    });
                }
                builder
                    .with_execution_providers([
                        ort::execution_providers::CUDAExecutionProvider::default()
                            .with_device_id(self.device_id)
                            .build(),
                    ])
                    .map_err(|e| EngineError::ModelLoadError {
                        path: model_path.to_string_lossy().into_owned(),
                        source: Box::new(e),
                    })?
            }
            ExecutionProvider::TensorRT => {
                if !ort::execution_providers::TensorRTExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err(EngineError::Backend {
                        message: "TensorRT execution provider is not available.".to_string(),
                        source: None,
                    });
                }
                builder
                    .with_execution_providers([
                        ort::execution_providers::TensorRTExecutionProvider::default()
                            .with_device_id(self.device_id)
                            .build(),
                        // Fallback to CUDA if TensorRT has issues with some nodes
                        ort::execution_providers::CUDAExecutionProvider::default()
                            .with_device_id(self.device_id)
                            .build(),
                    ])
                    .map_err(|e| EngineError::ModelLoadError {
                        path: model_path.to_string_lossy().into_owned(),
                        source: Box::new(e),
                    })?
            }
            ExecutionProvider::DirectML => {
                #[cfg(target_os = "windows")]
                {
                    if !ort::execution_providers::DirectMLExecutionProvider::default()
                        .is_available()
                        .unwrap_or(false)
                    {
                        return Err(EngineError::Backend {
                            message: "DirectML execution provider is not available.".to_string(),
                            source: None,
                        });
                    }
                    builder
                        .with_execution_providers([
                            ort::execution_providers::DirectMLExecutionProvider::default()
                                .with_device_id(self.device_id)
                                .build(),
                        ])
                        .map_err(|e| EngineError::ModelLoadError {
                            path: model_path.to_string_lossy().into_owned(),
                            source: Box::new(e),
                        })?
                }
                #[cfg(not(target_os = "windows"))]
                {
                    return Err(EngineError::Backend {
                        message: "DirectML execution provider is only supported on Windows."
                            .to_string(),
                        source: None,
                    });
                }
            }
            ExecutionProvider::ROCm => {
                if !ort::execution_providers::ROCmExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err(EngineError::Backend {
                        message: "ROCm execution provider is not available.".to_string(),
                        source: None,
                    });
                }
                builder
                    .with_execution_providers([
                        ort::execution_providers::ROCmExecutionProvider::default()
                            .with_device_id(self.device_id)
                            .build(),
                    ])
                    .map_err(|e| EngineError::ModelLoadError {
                        path: model_path.to_string_lossy().into_owned(),
                        source: Box::new(e),
                    })?
            }
            ExecutionProvider::OpenVINO => {
                if !ort::execution_providers::OpenVINOExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err(EngineError::Backend {
                        message: "OpenVINO execution provider is not available.".to_string(),
                        source: None,
                    });
                }
                builder
                    .with_execution_providers([
                        ort::execution_providers::OpenVINOExecutionProvider::default().build(),
                    ])
                    .map_err(|e| EngineError::ModelLoadError {
                        path: model_path.to_string_lossy().into_owned(),
                        source: Box::new(e),
                    })?
            }
            ExecutionProvider::CoreML => {
                if !ort::execution_providers::CoreMLExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    return Err(EngineError::Backend {
                        message: "CoreML execution provider is not available.".to_string(),
                        source: None,
                    });
                }
                builder
                    .with_execution_providers([
                        ort::execution_providers::CoreMLExecutionProvider::default().build(),
                        // Keep CPU registered as a fallback for nodes CoreML cannot handle. Without
                        // this, ONNXRuntime will hard-fail on models that partially compile/run
                        // under CoreML (ex: plan-building failures for certain ops).
                        ort::execution_providers::CPUExecutionProvider::default().build(),
                    ])
                    .map_err(|e| EngineError::ModelLoadError {
                        path: model_path.to_string_lossy().into_owned(),
                        source: Box::new(e),
                    })?
            }
            ExecutionProvider::CPU => {
                // CPU is always available, nothing special to add
                builder
            }
        };

        // Finalize session from file
        builder
            .commit_from_file(model_path)
            .map_err(|e| EngineError::ModelLoadError {
                path: model_path.to_string_lossy().into_owned(),
                source: Box::new(e),
            })
    }
}

/// PreparedInput is returned by the input validation helper. It contains the
/// typed vector of elements parsed from raw bytes.
///
/// Note: kept at module scope so tests can access it.
#[derive(Debug)]
enum PreparedInput {
    F32(Vec<f32>),
    F64(Vec<f64>),
    F16(Vec<f16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
}

fn validate_byte_len(
    input: &BinaryTensorPacket,
    num_elements: usize,
    elem_size: usize,
    dtype_label: &str,
) -> Result<(), EngineError> {
    let expected =
        num_elements
            .checked_mul(elem_size)
            .ok_or_else(|| EngineError::InvalidInput {
                message: "Data size overflow".to_string(),
                source: None,
            })?;
    if input.data.len() != expected {
        return Err(EngineError::InvalidInput {
            message: format!(
                "Data length mismatch: expected {} bytes ({} {} values) but got {} bytes",
                expected,
                num_elements,
                dtype_label,
                input.data.len()
            ),
            source: None,
        });
    }
    Ok(())
}

fn parse_ne_f32(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    if let Some(values) = try_aligned_copy::<f32>(bytes) {
        return values;
    }
    let mut values = Vec::with_capacity(num_elements);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    values
}

fn parse_ne_f64(bytes: &[u8], num_elements: usize) -> Vec<f64> {
    if let Some(values) = try_aligned_copy::<f64>(bytes) {
        return values;
    }
    let mut values = Vec::with_capacity(num_elements);
    for chunk in bytes.chunks_exact(8) {
        values.push(f64::from_ne_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    values
}

fn parse_ne_f16(bytes: &[u8], num_elements: usize) -> Vec<f16> {
    if let Some(values) = try_aligned_copy::<u16>(bytes) {
        let mut out = Vec::with_capacity(num_elements);
        out.extend(values.into_iter().map(f16::from_bits));
        return out;
    }
    let mut values = Vec::with_capacity(num_elements);
    for chunk in bytes.chunks_exact(2) {
        values.push(f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])));
    }
    values
}

fn parse_ne_i32(bytes: &[u8], num_elements: usize) -> Vec<i32> {
    if let Some(values) = try_aligned_copy::<i32>(bytes) {
        return values;
    }
    let mut values = Vec::with_capacity(num_elements);
    for chunk in bytes.chunks_exact(4) {
        values.push(i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    values
}

fn parse_ne_i64(bytes: &[u8], num_elements: usize) -> Vec<i64> {
    if let Some(values) = try_aligned_copy::<i64>(bytes) {
        return values;
    }
    let mut values = Vec::with_capacity(num_elements);
    for chunk in bytes.chunks_exact(8) {
        values.push(i64::from_ne_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    values
}

fn find_additional_input_by_name<'a>(
    additional_inputs: &'a [kapsl_engine_api::NamedTensor],
    name: &str,
) -> Option<&'a BinaryTensorPacket> {
    additional_inputs
        .iter()
        .find(|entry| entry.name == name)
        .map(|entry| &entry.tensor)
}

fn ensure_unique_additional_input_names(
    additional_inputs: &[kapsl_engine_api::NamedTensor],
) -> Result<(), EngineError> {
    for i in 0..additional_inputs.len() {
        for j in (i + 1)..additional_inputs.len() {
            if additional_inputs[i].name == additional_inputs[j].name {
                return Err(EngineError::InvalidInput {
                    message: format!(
                        "Duplicate additional input name: {}",
                        additional_inputs[i].name
                    ),
                    source: None,
                });
            }
        }
    }
    Ok(())
}

fn try_aligned_copy<T: Copy>(bytes: &[u8]) -> Option<Vec<T>> {
    // SAFETY: The concrete call sites only use plain numeric POD types.
    let (prefix, aligned, suffix) = unsafe { bytes.align_to::<T>() };
    if prefix.is_empty() && suffix.is_empty() {
        Some(aligned.to_vec())
    } else {
        None
    }
}

/// Validate an incoming BinaryTensorPacket and return the shape and a typed
/// vector if valid. This performs:
///  - dtype support checking
///  - shape -> element count computation
///  - buffer length validation (must equal element_count * dtype_size)
///  - safe byte->value conversion
fn validate_and_prepare_input(
    input: &kapsl_engine_api::BinaryTensorPacket,
) -> Result<(Vec<i64>, PreparedInput), EngineError> {
    // Compute number of elements from shape; treat empty shape as scalar (1 element)
    let num_elements: usize = if input.shape.is_empty() {
        1
    } else {
        // If any dimension is <= 0, reject as invalid shape
        let mut prod: usize = 1;
        for &d in &input.shape {
            if d <= 0 {
                return Err(EngineError::InvalidInput {
                    message: format!("Invalid shape dimension: {}", d),
                    source: None,
                });
            }
            prod = prod
                .checked_mul(d as usize)
                .ok_or_else(|| EngineError::InvalidInput {
                    message: "Shape multiplication overflow".to_string(),
                    source: None,
                })?;
        }
        prod
    };

    // Determine element size and branch by dtype
    match input.dtype {
        TensorDtype::Float32 => {
            validate_byte_len(input, num_elements, 4, "float32")?;
            let values = parse_ne_f32(&input.data, num_elements);
            Ok((input.shape.clone(), PreparedInput::F32(values)))
        }
        TensorDtype::Float64 => {
            validate_byte_len(input, num_elements, 8, "float64")?;
            let values = parse_ne_f64(&input.data, num_elements);
            Ok((input.shape.clone(), PreparedInput::F64(values)))
        }
        TensorDtype::Float16 => {
            validate_byte_len(input, num_elements, 2, "float16")?;
            let values = parse_ne_f16(&input.data, num_elements);
            Ok((input.shape.clone(), PreparedInput::F16(values)))
        }
        TensorDtype::Int32 => {
            validate_byte_len(input, num_elements, 4, "int32")?;
            let values = parse_ne_i32(&input.data, num_elements);
            Ok((input.shape.clone(), PreparedInput::I32(values)))
        }
        TensorDtype::Int64 => {
            validate_byte_len(input, num_elements, 8, "int64")?;
            let values = parse_ne_i64(&input.data, num_elements);
            Ok((input.shape.clone(), PreparedInput::I64(values)))
        }
        TensorDtype::Uint8 => {
            validate_byte_len(input, num_elements, 1, "uint8")?;
            Ok((input.shape.clone(), PreparedInput::U8(input.data.clone())))
        }
        other => Err(EngineError::InvalidInput {
            message: format!(
                "Unsupported dtype: {}. Supported: float32, float64, float16, int32, int64, uint8",
                other
            ),
            source: None,
        }),
    }
}

fn tensor_packet_to_session_input(
    input: &BinaryTensorPacket,
) -> Result<(Vec<usize>, SessionInputValue<'_>), EngineError> {
    let (shape_i64, prepared) = validate_and_prepare_input(input)?;
    let shape_usize = get_shape_usize(&shape_i64);

    let value: SessionInputValue = match prepared {
        PreparedInput::F32(v) => Value::from_array((shape_usize.clone(), v)).map(|v| v.into()),
        PreparedInput::F64(v) => Value::from_array((shape_usize.clone(), v)).map(|v| v.into()),
        PreparedInput::F16(v) => Value::from_array((shape_usize.clone(), v)).map(|v| v.into()),
        PreparedInput::I32(v) => Value::from_array((shape_usize.clone(), v)).map(|v| v.into()),
        PreparedInput::I64(v) => Value::from_array((shape_usize.clone(), v)).map(|v| v.into()),
        PreparedInput::U8(v) => Value::from_array((shape_usize.clone(), v)).map(|v| v.into()),
    }
    .map_err(|e| EngineError::InferenceError {
        reason: "Failed to create input tensor".to_string(),
        source: Some(Box::new(e)),
    })?;

    Ok((shape_usize, value))
}

fn run_inference_with_session(
    session: &mut Session,
    request: &InferenceRequest,
    metadata: &ModelMetadata,
    shape_usize: Vec<usize>,
    main_input_tensor: SessionInputValue<'_>,
) -> Result<BinaryTensorPacket, EngineError> {
    // We assume the first input maps to the provided input packet
    if metadata.input_names.is_empty() {
        return Err(EngineError::InferenceError {
            reason: "Model has no inputs defined".to_string(),
            source: None,
        });
    }

    let outputs = if metadata.input_names.len() == 1 && request.additional_inputs.is_empty() {
        session.run([main_input_tensor]).map_err(|e| {
            log::error!("ONNX Runtime inference error: {:?}", e);
            EngineError::InferenceError {
                reason: format!("Inference failed: {}", e),
                source: Some(Box::new(e)),
            }
        })?
    } else {
        // Prepare named input map only when required by multi-input models.
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue)> =
            Vec::with_capacity(metadata.input_names.len());
        inputs.push((
            Cow::Borrowed(metadata.input_names[0].as_str()),
            main_input_tensor,
        ));
        ensure_unique_additional_input_names(&request.additional_inputs)?;

        // Get batch size from main input (assume dim 0 is batch)
        let batch_size = if !shape_usize.is_empty() {
            shape_usize[0]
        } else {
            1
        };
        let seq_len = if shape_usize.len() > 1 {
            shape_usize[1]
        } else {
            1
        };

        // Fill other inputs
        // Workaround: ORT Value::from_array rejects 0-dimension (e.g. past_seq_len=0).
        // Try to use ndarray which might handle it better, or maybe it was just a vector check.
        // We set workaround length to 0 (correct behavior).
        let workaround_past_len = 0;

        for (i, name) in metadata.input_names.iter().enumerate().skip(1) {
            let shape_def = &metadata.input_shapes[i];
            if let Some(named_input) =
                find_additional_input_by_name(&request.additional_inputs, name)
            {
                let (_, value) = tensor_packet_to_session_input(named_input)?;
                inputs.push((Cow::Borrowed(name.as_str()), value));
                continue;
            }

            if name.contains("attention_mask") {
                // Attention mask: [batch, total_seq_len] -> 1s for active, 0 for past workaround
                // Total len = seq_len + workaround_past_len
                let total_len = seq_len + workaround_past_len;
                // Attention mask: [batch, total_seq_len] -> 1s for active
                let mask_shape = vec![batch_size as i64, total_len as i64];

                let mut mask_data = Vec::with_capacity(batch_size * total_len);
                for _ in 0..batch_size {
                    // Mask out the dummy past
                    mask_data.extend(std::iter::repeat_n(0i64, workaround_past_len));
                    // Active sequence
                    mask_data.extend(std::iter::repeat_n(1i64, seq_len));
                }

                log::debug!(
                    "Creating attention_mask tensor for {} with shape {:?}",
                    name,
                    mask_shape
                );

                let mask_tensor = Value::from_array((get_shape_usize(&mask_shape), mask_data))
                    .map_err(|e| EngineError::InferenceError {
                        reason: format!("Failed to create attention_mask tensor for {}", name),
                        source: Some(Box::new(e)),
                    })?;
                inputs.push((Cow::Borrowed(name.as_str()), mask_tensor.into()));
            } else if name.contains("position_ids") {
                let pos_shape = vec![batch_size as i64, seq_len as i64];
                let mut pos_data = Vec::with_capacity(batch_size * seq_len);
                // Position IDs should likely start after past? Or 0?
                // If we have dummy past at 0..1, then real tokens start at 0?
                // Actually if past is masked, it's effectively like it doesn't exist.
                // So we keep pos ids 0-based.
                for _ in 0..batch_size {
                    for s in 0..seq_len {
                        pos_data.push(s as i64);
                    }
                }
                let pos_tensor = Value::from_array((get_shape_usize(&pos_shape), pos_data))
                    .map_err(|e| EngineError::InferenceError {
                        reason: format!("Failed to create position_ids tensor for {}", name),
                        source: Some(Box::new(e)),
                    })?;
                inputs.push((Cow::Borrowed(name.as_str()), pos_tensor.into()));
            } else if name.starts_with("past_key_values") {
                let mut new_shape = Vec::new();
                new_shape.push(batch_size); // dim 0

                if shape_def.len() == 4 {
                    let dim1 = if shape_def[1] > 0 {
                        shape_def[1] as usize
                    } else {
                        1
                    };
                    new_shape.push(dim1);

                    new_shape.push(workaround_past_len); // dim 2: 0

                    let dim3 = if shape_def[3] > 0 {
                        shape_def[3] as usize
                    } else {
                        64
                    };
                    new_shape.push(dim3);

                    log::debug!("Creating KV tensor for {} with shape {:?}", name, new_shape);

                    let count: usize = new_shape.iter().product();
                    let empty_data: Vec<f16> = vec![f16::ZERO; count];

                    // Use ndarray to construct possibly 0-dim tensor
                    let kv_array = ArrayD::from_shape_vec(new_shape, empty_data).map_err(|e| {
                        EngineError::InferenceError {
                            reason: format!("Failed to create ndarray for {}: {:?}", name, e),
                            source: Some(Box::new(e)),
                        }
                    })?;

                    let kv_tensor =
                        Value::from_array(kv_array).map_err(|e| EngineError::InferenceError {
                            reason: format!(
                                "Failed to create empty KV tensor for {}: {:?}",
                                name, e
                            ),
                            source: Some(Box::new(e)),
                        })?;
                    inputs.push((Cow::Borrowed(name.as_str()), kv_tensor.into()));
                } else {
                    log::warn!(
                        "Skipping input {} due to unknown shape pattern {:?}",
                        name,
                        shape_def
                    );
                }
            } else {
                log::warn!(
                    "Skipping input {} as it is not recognized as auto-fillable",
                    name
                );
            }
        }

        session.run(inputs).map_err(|e| {
            log::error!("ONNX Runtime inference error: {:?}", e);
            EngineError::InferenceError {
                reason: format!("Inference failed: {}", e),
                source: Some(Box::new(e)),
            }
        })?
    };

    // For LLMs, we often get multiple outputs (logits + KV cache).
    // We currently only ignore the KV cache return values and just use the first output (logits).
    if outputs.len() > 1 {
        log::debug!(
            "Backend received {} outputs, using only the first one (logits)",
            outputs.len()
        );
    }

    // Handle output - try f32 first, otherwise return an error.
    let output_value = &outputs[0];
    let output_packet = if let Ok((shape_ref, data)) = output_value.try_extract_tensor::<f32>() {
        BinaryTensorPacket {
            shape: shape_ref.to_vec(),
            dtype: TensorDtype::Float32,
            data: data.iter().flat_map(|&x| x.to_ne_bytes()).collect(),
        }
    } else if let Ok((shape_ref, data)) = output_value.try_extract_tensor::<f64>() {
        BinaryTensorPacket {
            shape: shape_ref.to_vec(),
            dtype: TensorDtype::Float64,
            data: data.iter().flat_map(|&x| x.to_ne_bytes()).collect(),
        }
    } else if let Ok((shape_ref, data)) = output_value.try_extract_tensor::<f16>() {
        BinaryTensorPacket {
            shape: shape_ref.to_vec(),
            dtype: TensorDtype::Float16,
            data: data
                .iter()
                .flat_map(|x| x.to_bits().to_ne_bytes())
                .collect(),
        }
    } else if let Ok((shape_ref, data)) = output_value.try_extract_tensor::<i32>() {
        BinaryTensorPacket {
            shape: shape_ref.to_vec(),
            dtype: TensorDtype::Int32,
            data: data.iter().flat_map(|&x| x.to_ne_bytes()).collect(),
        }
    } else if let Ok((shape_ref, data)) = output_value.try_extract_tensor::<i64>() {
        BinaryTensorPacket {
            shape: shape_ref.to_vec(),
            dtype: TensorDtype::Int64,
            data: data.iter().flat_map(|&x| x.to_ne_bytes()).collect(),
        }
    } else if let Ok((shape_ref, data)) = output_value.try_extract_tensor::<u8>() {
        BinaryTensorPacket {
            shape: shape_ref.to_vec(),
            dtype: TensorDtype::Uint8,
            data: data.to_vec(),
        }
    } else {
        return Err(EngineError::InferenceError {
            reason: "Failed to extract output tensor. Supported output dtypes: float32, float64, float16, int32, int64, uint8"
                .to_string(),
            source: None,
        });
    };

    Ok(output_packet)
}

#[async_trait]
impl Engine for OnnxBackend {
    // TODO: Extract session creation to a helper method to reduce duplication
    // TODO: Add better error messages with execution provider fallback information
    // TODO: Add capability checking before attempting to use hardware accelerators
    // TODO: Consider async loading for large models
    // TODO: Add progress callback for large model loads
    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        let opt_level = self.get_opt_level();
        log::info!(
            "Loading ONNX model with optimization level: {:?} on provider {:?}",
            opt_level,
            self.provider
        );
        log::info!(
            "ORT memory config: memory_pattern={} disable_cpu_mem_arena={} session_buckets={} bucket_dim_granularity={} bucket_max_dims={} peak_concurrency_hint={:?}",
            self.memory_pattern,
            self.disable_cpu_mem_arena,
            self.max_bucket_sessions,
            self.bucket_dim_granularity,
            self.bucket_max_dims,
            self.peak_concurrency_hint
        );

        let session = self.create_session(model_path, opt_level)?;

        log::info!("Model Inputs:");
        for (i, input) in session.inputs().iter().enumerate() {
            log::info!("  Input {}: {} ({:?})", i, input.name(), input.dtype());
        }

        // Extract metadata
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let mut input_shapes = Vec::new();
        let mut input_dtypes = Vec::new();
        for input in session.inputs() {
            let (shape, dtype) = match input.dtype() {
                ort::value::ValueType::Tensor { ty, shape, .. } => {
                    (shape.iter().copied().collect(), map_ort_dtype(*ty))
                }
                _ => (vec![], None),
            };
            input_shapes.push(shape);
            input_dtypes.push(dtype);
        }

        let mut output_shapes = Vec::new();
        let mut output_dtypes = Vec::new();
        for output in session.outputs() {
            let (shape, dtype) = match output.dtype() {
                ort::value::ValueType::Tensor { ty, shape, .. } => {
                    (shape.iter().copied().collect(), map_ort_dtype(*ty))
                }
                _ => (vec![], None),
            };
            output_shapes.push(shape);
            output_dtypes.push(dtype);
        }

        let metadata = ModelMetadata {
            input_names,
            output_names,
            input_shapes,
            output_shapes,
            input_dtypes,
            output_dtypes,
        };

        // Store metadata
        if let Ok(mut meta_guard) = self.metadata.write() {
            *meta_guard = Some(metadata);
        }

        let mut session_guard = self.session.write().map_err(|_| EngineError::Backend {
            message: "Lock poisoned".to_string(),
            source: None,
        })?;
        *session_guard = Some(session);
        drop(session_guard);

        if let Ok(mut model_path_guard) = self.model_path.write() {
            *model_path_guard = Some(model_path.to_path_buf());
        }
        if let Ok(mut bucket_guard) = self.bucket_sessions.write() {
            bucket_guard.primary_bucket_key = None;
            bucket_guard.sessions.clear();
            bucket_guard.lru.clear();
        }

        // Reset warmup state
        self.warmed_up.store(false, Ordering::SeqCst);

        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let start_time = Instant::now();
        if let Some(session_id) = &request.session_id {
            log::debug!(
                "Processing request for session: {}",
                redact_session_id_for_log(session_id)
            );
        }

        let metadata = self.metadata.read().map_err(|_| EngineError::Backend {
            message: "Lock poisoned".to_string(),
            source: None,
        })?;
        let metadata = metadata
            .as_ref()
            .cloned()
            .ok_or(EngineError::ModelNotLoaded)?;

        // Validate and convert primary input.
        let (shape_usize, main_input_tensor) = tensor_packet_to_session_input(&request.input)?;
        let mut prepared_input = Some((shape_usize, main_input_tensor));

        let output_packet = if let Some(bucket_key) = self.bucket_key_for_request(request) {
            let use_primary = {
                let mut bucket_guard =
                    self.bucket_sessions
                        .write()
                        .map_err(|_| EngineError::Backend {
                            message: "Lock poisoned".to_string(),
                            source: None,
                        })?;
                let primary_key = bucket_guard
                    .primary_bucket_key
                    .get_or_insert_with(|| bucket_key.clone());
                *primary_key == bucket_key
            };

            if use_primary {
                let mut session_guard = self.session.write().map_err(|_| EngineError::Backend {
                    message: "Lock poisoned".to_string(),
                    source: None,
                })?;
                let session = session_guard.as_mut().ok_or(EngineError::ModelNotLoaded)?;
                let (shape_usize, main_input_tensor) = prepared_input
                    .take()
                    .ok_or_else(|| EngineError::backend("input already consumed".to_string()))?;
                run_inference_with_session(
                    session,
                    request,
                    &metadata,
                    shape_usize,
                    main_input_tensor,
                )?
            } else {
                let mut bucket_guard =
                    self.bucket_sessions
                        .write()
                        .map_err(|_| EngineError::Backend {
                            message: "Lock poisoned".to_string(),
                            source: None,
                        })?;
                let session = self.get_or_create_bucket_session(&mut bucket_guard, &bucket_key)?;
                let (shape_usize, main_input_tensor) = prepared_input
                    .take()
                    .ok_or_else(|| EngineError::backend("input already consumed".to_string()))?;
                run_inference_with_session(
                    session,
                    request,
                    &metadata,
                    shape_usize,
                    main_input_tensor,
                )?
            }
        } else {
            let mut session_guard = self.session.write().map_err(|_| EngineError::Backend {
                message: "Lock poisoned".to_string(),
                source: None,
            })?;
            let session = session_guard.as_mut().ok_or(EngineError::ModelNotLoaded)?;
            let (shape_usize, main_input_tensor) = prepared_input
                .take()
                .ok_or_else(|| EngineError::backend("input already consumed".to_string()))?;
            run_inference_with_session(session, request, &metadata, shape_usize, main_input_tensor)?
        };

        // Update metrics
        let duration = start_time.elapsed().as_secs_f64();
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.inference_time = duration;
            // We can't easily get exact memory usage per inference from ONNX Runtime easily here without allocator hooks,
            // so we leave it as is or update if we had a way.
        }

        // Mark as warmed up
        self.warmed_up.store(true, Ordering::SeqCst);

        Ok(output_packet)
    }

    // TODO: Implement proper streaming for LLM models with token-by-token generation
    // TODO: This is a placeholder - real streaming should yield tokens as they're generated
    fn infer_stream(
        &self,
        request: &InferenceRequest,
    ) -> std::pin::Pin<
        Box<dyn futures::stream::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
    > {
        // Call infer immediately to avoid lifetime issues
        let result = self.infer(request);
        // Wrap the single result in a stream using futures::stream::once
        Box::pin(futures::stream::once(async move { result }))
    }

    // TODO: Add proper cleanup (free GPU memory, release resources)
    // TODO: Log unload operations for debugging
    fn unload(&mut self) {
        if let Ok(mut session_guard) = self.session.write() {
            *session_guard = None;
        }
        if let Ok(mut bucket_guard) = self.bucket_sessions.write() {
            bucket_guard.primary_bucket_key = None;
            bucket_guard.sessions.clear();
            bucket_guard.lru.clear();
        }
        if let Ok(mut model_path_guard) = self.model_path.write() {
            *model_path_guard = None;
        }
        if let Ok(mut meta_guard) = self.metadata.write() {
            *meta_guard = None;
        }
    }

    fn metrics(&self) -> kapsl_engine_api::EngineMetrics {
        if let Ok(metrics) = self.metrics.read() {
            metrics.clone()
        } else {
            kapsl_engine_api::EngineMetrics::default()
        }
    }

    fn health_check(&self) -> Result<(), EngineError> {
        // Check if session is loaded and lock is not poisoned
        let session_guard = self.session.read().map_err(|_| EngineError::Backend {
            message: "Session lock poisoned".to_string(),
            source: None,
        })?;
        let bucket_guard = self
            .bucket_sessions
            .read()
            .map_err(|_| EngineError::Backend {
                message: "Session cache lock poisoned".to_string(),
                source: None,
            })?;

        if session_guard.is_some() || !bucket_guard.sessions.is_empty() {
            Ok(())
        } else {
            Err(EngineError::ModelNotLoaded)
        }
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        let metadata_guard = self.metadata.read().ok()?;
        let metadata = metadata_guard.as_ref()?;
        Some(EngineModelInfo {
            input_names: metadata.input_names.clone(),
            output_names: metadata.output_names.clone(),
            input_shapes: metadata.input_shapes.clone(),
            output_shapes: metadata.output_shapes.clone(),
            input_dtypes: metadata
                .input_dtypes
                .iter()
                .map(|dtype| {
                    dtype
                        .as_ref()
                        .map(TensorDtype::as_str)
                        .unwrap_or("unknown")
                        .to_string()
                })
                .collect(),
            output_dtypes: metadata
                .output_dtypes
                .iter()
                .map(|dtype| {
                    dtype
                        .as_ref()
                        .map(TensorDtype::as_str)
                        .unwrap_or("unknown")
                        .to_string()
                })
                .collect(),
            framework: Some("onnx".to_string()),
            model_version: None,
            peak_concurrency: self.peak_concurrency_hint,
        })
    }
}

fn get_shape_usize(shape: &[i64]) -> Vec<usize> {
    shape.iter().map(|&v| v as usize).collect()
}

#[path = "onnx_tests.rs"]
mod onnx_tests;
