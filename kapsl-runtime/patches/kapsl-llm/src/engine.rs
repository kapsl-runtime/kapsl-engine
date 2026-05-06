/*
Reworked LLMEngine to detect model KV geometry at load time and use dynamic
num_layers / num_heads / head_dim instead of hard-coded constants.

This file is adapted from the original implementation and preserves the engine's
behavior while making the KV geometry configurable at model-load time.
*/

use crate::block_manager::{BlockManager, SharedBlockAllocator};
use crate::kv_cache::{KvCache, KvCacheConfig, KvCacheMode, KvEvictionPolicy};
use crate::llm_metrics::LLMMetrics;
use crate::model_paths::{find_model_asset, find_model_root};
use crate::scheduler::{LLMScheduler, SchedulerConfig};
use crate::sequence::{
    FinishReason, SamplingParams, Sequence, SequenceGroup, SequenceGroupOutput, SequenceStatus,
};
use half::f16;
use kapsl_engine_api::EngineError;
use kapsl_hal::kernel::KernelBackend;
use ndarray::{Array2, Array4, ArrayD, IxDyn};
#[cfg(target_os = "windows")]
use ort::execution_providers::DirectMLExecutionProvider;
use ort::execution_providers::ExecutionProvider as OrtExecutionProvider;
use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, OpenVINOExecutionProvider,
    ROCmExecutionProvider, TensorRTExecutionProvider,
};
use ort::session::{
    builder::GraphOptimizationLevel, builder::SessionBuilder, Session, SessionInputValue,
};
use ort::tensor::TensorElementType;
use ort::value::{DynValue, TensorRef, Value};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

// === DEFAULTS (used until model loads and detection happens) ===
// These are conservative defaults; they will be overridden at load() time when
// the model declares its input shapes.
const DEFAULT_NUM_LAYERS: usize = 32;
const DEFAULT_NUM_HEADS: usize = 32;
const DEFAULT_HEAD_DIM: usize = 128;
const MAX_SEQ_LEN: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KvLayout {
    SeqFirst,
    HeadDimFirst,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SafeLoadSetting {
    ForceOn,
    ForceOff,
    Auto,
}

struct PipelineStage {
    session: Session,
    input_names: HashSet<String>,
    input_shapes: HashMap<String, Vec<i64>>,
    input_types: HashMap<String, TensorElementType>,
}

type SessionIo = (
    HashSet<String>,
    HashMap<String, Vec<i64>>,
    HashMap<String, TensorElementType>,
    HashSet<String>,
);

fn parse_safe_load_setting(value: &serde_json::Value) -> Option<SafeLoadSetting> {
    if let Some(enabled) = value.as_bool() {
        return Some(if enabled {
            SafeLoadSetting::ForceOn
        } else {
            SafeLoadSetting::ForceOff
        });
    }
    let value = value.as_str()?.trim().to_ascii_lowercase();
    match value.as_str() {
        "1" | "true" | "on" | "yes" => Some(SafeLoadSetting::ForceOn),
        "0" | "false" | "off" | "no" => Some(SafeLoadSetting::ForceOff),
        "auto" => Some(SafeLoadSetting::Auto),
        _ => None,
    }
}

fn env_var_alias(primary: &str, legacy: &str) -> Option<String> {
    std::env::var(primary)
        .or_else(|_| std::env::var(legacy))
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn llm_decode_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KAPSL_LLM_PROFILE_DECODE")
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

fn parse_byte_size(value: &str) -> Option<usize> {
    let lower = value.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }

    let (number, multiplier) = if let Some(number) = lower.strip_suffix("gib") {
        (number, 1024usize.pow(3))
    } else if let Some(number) = lower.strip_suffix("mib") {
        (number, 1024usize.pow(2))
    } else if let Some(number) = lower.strip_suffix("kib") {
        (number, 1024usize)
    } else if let Some(number) = lower.strip_suffix("gb") {
        (number, 1000usize.pow(3))
    } else if let Some(number) = lower.strip_suffix("mb") {
        (number, 1000usize.pow(2))
    } else if let Some(number) = lower.strip_suffix("kb") {
        (number, 1000usize)
    } else if let Some(number) = lower.strip_suffix('b') {
        (number, 1usize)
    } else {
        (lower.as_str(), 1usize)
    };

    number
        .trim()
        .parse::<usize>()
        .ok()
        .and_then(|parsed| parsed.checked_mul(multiplier))
}

fn parse_byte_size_json(value: &serde_json::Value) -> Option<usize> {
    match value {
        serde_json::Value::Number(number) => number.as_u64().and_then(|v| usize::try_from(v).ok()),
        serde_json::Value::String(text) => parse_byte_size(text),
        _ => None,
    }
}

fn llm_decode_bucket_granularity() -> usize {
    static GRANULARITY: OnceLock<usize> = OnceLock::new();
    *GRANULARITY.get_or_init(|| {
        std::env::var("KAPSL_LLM_DECODE_BUCKET_GRANULARITY")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 1)
            .unwrap_or(1)
    })
}

fn round_up_to_granularity(value: usize, granularity: usize) -> usize {
    if value == 0 || granularity <= 1 {
        return value;
    }
    value
        .saturating_add(granularity - 1)
        .checked_div(granularity)
        .unwrap_or(0)
        .saturating_mul(granularity)
}

fn llm_provider_policy() -> String {
    env_var_alias("KAPSL_PROVIDER_POLICY", "KAPSL_PROVIDER_POLICY")
        .unwrap_or_else(|| "fastest".to_string())
        .to_ascii_lowercase()
}

fn llm_provider_available(provider: &str) -> bool {
    match provider.trim().to_ascii_lowercase().as_str() {
        "tensorrt" => TensorRTExecutionProvider::default()
            .is_available()
            .unwrap_or(false),
        "cuda" => CUDAExecutionProvider::default()
            .is_available()
            .unwrap_or(false),
        "coreml" | "metal" => CoreMLExecutionProvider::default()
            .is_available()
            .unwrap_or(false),
        "rocm" => ROCmExecutionProvider::default()
            .is_available()
            .unwrap_or(false),
        "directml" => directml_provider_available(),
        "openvino" => OpenVINOExecutionProvider::default()
            .is_available()
            .unwrap_or(false),
        "cpu" => true,
        _ => false,
    }
}

#[cfg(target_os = "windows")]
fn directml_provider_available() -> bool {
    DirectMLExecutionProvider::default()
        .is_available()
        .unwrap_or(false)
}

#[cfg(not(target_os = "windows"))]
fn directml_provider_available() -> bool {
    false
}

fn fastest_llm_provider() -> &'static str {
    if llm_provider_available("tensorrt") {
        "tensorrt"
    } else if llm_provider_available("cuda") {
        "cuda"
    } else if llm_provider_available("coreml") {
        "coreml"
    } else if llm_provider_available("rocm") {
        "rocm"
    } else if llm_provider_available("directml") {
        "directml"
    } else if llm_provider_available("openvino") {
        "openvino"
    } else {
        "cpu"
    }
}

fn ensure_external_data_near_model(model_path: &Path, model_root: &Path) {
    let file_name = match model_path.file_name().and_then(|name| name.to_str()) {
        Some(name) => name,
        None => return,
    };
    let model_dir = match model_path.parent() {
        Some(dir) => dir,
        None => return,
    };

    let candidates = [format!("{file_name}_data"), format!("{file_name}.data")];

    for candidate in candidates {
        let expected = model_dir.join(&candidate);
        if expected.exists() {
            continue;
        }

        let src = find_model_asset(model_path, &candidate)
            .or_else(|| {
                let onnx_dir = model_root.join("onnx");
                let alt = onnx_dir.join(&candidate);
                alt.exists().then_some(alt)
            })
            .or_else(|| {
                let export_dir = model_root.join("onnx-export");
                let alt = export_dir.join(&candidate);
                alt.exists().then_some(alt)
            });

        let Some(src) = src else {
            continue;
        };

        if src == expected {
            continue;
        }

        if let Some(parent) = expected.parent() {
            if let Err(err) = std::fs::create_dir_all(parent) {
                log::warn!(
                    "Failed to create external data directory {}: {}",
                    parent.display(),
                    err
                );
                continue;
            }
        }

        let link_result = std::fs::hard_link(&src, &expected);
        if link_result.is_err() {
            if let Err(err) = std::fs::copy(&src, &expected) {
                log::warn!(
                    "Failed to place external data file {} (from {}): {}",
                    expected.display(),
                    src.display(),
                    err
                );
            } else {
                log::info!(
                    "Copied external data file {} from {}",
                    expected.display(),
                    src.display()
                );
            }
        } else {
            log::info!(
                "Linked external data file {} to {}",
                expected.display(),
                src.display()
            );
        }
    }
}

fn has_dot_external_data(model_path: &Path) -> bool {
    let file_name = match model_path.file_name().and_then(|name| name.to_str()) {
        Some(name) => name,
        None => return false,
    };
    let model_dir = match model_path.parent() {
        Some(dir) => dir,
        None => return false,
    };
    model_dir.join(format!("{file_name}.data")).exists()
}

fn resolve_stage_path(model_root: &Path, model_path: &Path, stage: &str) -> Option<PathBuf> {
    let candidate = PathBuf::from(stage);
    if candidate.is_absolute() {
        return candidate.exists().then_some(candidate);
    }

    let mut bases = Vec::new();
    if let Some(parent) = model_path.parent() {
        bases.push(parent.to_path_buf());
    }
    bases.push(model_root.to_path_buf());
    bases.push(model_root.join("onnx-export"));

    for base in bases {
        let path = base.join(stage);
        if path.exists() {
            return Some(path);
        }
    }

    None
}

fn infer_kv_layout(
    input_shapes: &HashMap<String, Vec<i64>>,
    num_heads: usize,
    head_dim: usize,
) -> KvLayout {
    let key_name = "past_key_values.0.key";
    if let Some(shape) = input_shapes.get(key_name) {
        if shape.len() >= 4 {
            let head_dim_axis = shape.iter().position(|&d| d == head_dim as i64);
            if head_dim_axis == Some(2) {
                return KvLayout::HeadDimFirst;
            }
            if head_dim_axis == Some(3) {
                return KvLayout::SeqFirst;
            }
            let num_heads_axis = shape.iter().position(|&d| d == num_heads as i64);
            if num_heads_axis == Some(1) {
                if shape.get(2) == Some(&(head_dim as i64)) {
                    return KvLayout::HeadDimFirst;
                }
                if shape.get(3) == Some(&(head_dim as i64)) {
                    return KvLayout::SeqFirst;
                }
            }
        }
    }
    KvLayout::SeqFirst
}

fn capture_session_io(session: &Session) -> SessionIo {
    let mut input_names = HashSet::new();
    let mut input_shapes = HashMap::new();
    let mut input_types = HashMap::new();
    for input in session.inputs() {
        let name = input.name().to_string();
        input_names.insert(name.clone());
        let shape = match input.dtype() {
            ort::value::ValueType::Tensor { ty, shape, .. } => {
                input_types.insert(name.clone(), *ty);
                shape.iter().copied().collect()
            }
            _ => vec![],
        };
        input_shapes.insert(name, shape);
    }

    let mut output_names = HashSet::new();
    for output in session.outputs() {
        output_names.insert(output.name().to_string());
    }

    (input_names, input_shapes, input_types, output_names)
}

fn build_kv_array_f16(
    data: &[f16],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    layout: KvLayout,
    label: &str,
) -> Result<Array4<f16>, EngineError> {
    let stride = num_heads * head_dim;
    if stride == 0 {
        return Err(EngineError::backend(format!(
            "Invalid KV tensor config for {}: num_heads={}, head_dim={}",
            label, num_heads, head_dim
        )));
    }
    if !data.len().is_multiple_of(stride) {
        return Err(EngineError::backend(format!(
            "KV tensor data length {} not divisible by head stride {} for {}",
            data.len(),
            stride,
            label
        )));
    }
    let max_seq_len = data.len() / stride;
    if seq_len > max_seq_len {
        return Err(EngineError::backend(format!(
            "KV tensor seq_len {} exceeds cache capacity {} for {}",
            seq_len, max_seq_len, label
        )));
    }

    let total = num_heads * seq_len * head_dim;
    let mut packed = vec![f16::ZERO; total];
    for h in 0..num_heads {
        let head_offset = h * max_seq_len * head_dim;
        let packed_offset = h * seq_len * head_dim;
        for pos in 0..seq_len {
            let src = head_offset + pos * head_dim;
            let dst = packed_offset + pos * head_dim;
            let src_end = src + head_dim;
            let dst_end = dst + head_dim;
            if src_end <= data.len() {
                packed[dst..dst_end].copy_from_slice(&data[src..src_end]);
            }
        }
    }
    match layout {
        KvLayout::SeqFirst => Array4::from_shape_vec((1, num_heads, seq_len, head_dim), packed)
            .map_err(|e| EngineError::backend(format!("Failed to make {} array: {:?}", label, e))),
        KvLayout::HeadDimFirst => {
            let mut reordered = vec![f16::ZERO; total];
            for h in 0..num_heads {
                for pos in 0..seq_len {
                    for d in 0..head_dim {
                        let src = (h * seq_len * head_dim) + (pos * head_dim) + d;
                        let dst = (h * head_dim * seq_len) + (d * seq_len) + pos;
                        reordered[dst] = packed[src];
                    }
                }
            }
            Array4::from_shape_vec((1, num_heads, head_dim, seq_len), reordered).map_err(|e| {
                EngineError::backend(format!("Failed to make {} array: {:?}", label, e))
            })
        }
    }
}

fn build_kv_array_f32_from_f16(
    data: &[f16],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    layout: KvLayout,
    label: &str,
) -> Result<Array4<f32>, EngineError> {
    let stride = num_heads * head_dim;
    if stride == 0 {
        return Err(EngineError::backend(format!(
            "Invalid KV tensor config for {}: num_heads={}, head_dim={}",
            label, num_heads, head_dim
        )));
    }
    if !data.len().is_multiple_of(stride) {
        return Err(EngineError::backend(format!(
            "KV tensor data length {} not divisible by head stride {} for {}",
            data.len(),
            stride,
            label
        )));
    }
    let max_seq_len = data.len() / stride;
    if seq_len > max_seq_len {
        return Err(EngineError::backend(format!(
            "KV tensor seq_len {} exceeds cache capacity {} for {}",
            seq_len, max_seq_len, label
        )));
    }

    let total = num_heads * seq_len * head_dim;
    let mut packed = vec![0f32; total];
    for h in 0..num_heads {
        let head_offset = h * max_seq_len * head_dim;
        let packed_offset = h * seq_len * head_dim;
        for pos in 0..seq_len {
            let src = head_offset + pos * head_dim;
            let dst = packed_offset + pos * head_dim;
            let src_end = src + head_dim;
            let dst_end = dst + head_dim;
            if src_end <= data.len() {
                for (out, val) in packed[dst..dst_end].iter_mut().zip(&data[src..src_end]) {
                    *out = val.to_f32();
                }
            }
        }
    }
    match layout {
        KvLayout::SeqFirst => Array4::from_shape_vec((1, num_heads, seq_len, head_dim), packed)
            .map_err(|e| EngineError::backend(format!("Failed to make {} array: {:?}", label, e))),
        KvLayout::HeadDimFirst => {
            let mut reordered = vec![0f32; total];
            for h in 0..num_heads {
                for pos in 0..seq_len {
                    for d in 0..head_dim {
                        let src = (h * seq_len * head_dim) + (pos * head_dim) + d;
                        let dst = (h * head_dim * seq_len) + (d * seq_len) + pos;
                        reordered[dst] = packed[src];
                    }
                }
            }
            Array4::from_shape_vec((1, num_heads, head_dim, seq_len), reordered).map_err(|e| {
                EngineError::backend(format!("Failed to make {} array: {:?}", label, e))
            })
        }
    }
}

fn build_kv_array_f16_from_packed(
    data: &[f16],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    layout: KvLayout,
    label: &str,
) -> Result<Array4<f16>, EngineError> {
    let expected = num_heads * seq_len * head_dim;
    if expected == 0 {
        return Err(EngineError::backend(format!(
            "Invalid KV tensor config for {}: num_heads={}, seq_len={}, head_dim={}",
            label, num_heads, seq_len, head_dim
        )));
    }
    if data.len() != expected {
        return Err(EngineError::backend(format!(
            "Packed KV length {} does not match expected {} for {}",
            data.len(),
            expected,
            label
        )));
    }

    match layout {
        KvLayout::SeqFirst => {
            Array4::from_shape_vec((1, num_heads, seq_len, head_dim), data.to_vec()).map_err(|e| {
                EngineError::backend(format!("Failed to make {} array: {:?}", label, e))
            })
        }
        KvLayout::HeadDimFirst => {
            let mut reordered = vec![f16::ZERO; expected];
            for h in 0..num_heads {
                for pos in 0..seq_len {
                    for d in 0..head_dim {
                        let src = (h * seq_len * head_dim) + (pos * head_dim) + d;
                        let dst = (h * head_dim * seq_len) + (d * seq_len) + pos;
                        reordered[dst] = data[src];
                    }
                }
            }
            Array4::from_shape_vec((1, num_heads, head_dim, seq_len), reordered).map_err(|e| {
                EngineError::backend(format!("Failed to make {} array: {:?}", label, e))
            })
        }
    }
}

fn build_kv_array_f32_from_packed(
    data: &[f16],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    layout: KvLayout,
    label: &str,
) -> Result<Array4<f32>, EngineError> {
    let expected = num_heads * seq_len * head_dim;
    if expected == 0 {
        return Err(EngineError::backend(format!(
            "Invalid KV tensor config for {}: num_heads={}, seq_len={}, head_dim={}",
            label, num_heads, seq_len, head_dim
        )));
    }
    if data.len() != expected {
        return Err(EngineError::backend(format!(
            "Packed KV length {} does not match expected {} for {}",
            data.len(),
            expected,
            label
        )));
    }
    let mut packed = Vec::with_capacity(expected);
    packed.extend(data.iter().map(|v| v.to_f32()));
    match layout {
        KvLayout::SeqFirst => Array4::from_shape_vec((1, num_heads, seq_len, head_dim), packed)
            .map_err(|e| EngineError::backend(format!("Failed to make {} array: {:?}", label, e))),
        KvLayout::HeadDimFirst => Array4::from_shape_vec((1, num_heads, head_dim, seq_len), packed)
            .map_err(|e| EngineError::backend(format!("Failed to make {} array: {:?}", label, e))),
    }
}

fn extract_tensor_f16<'a>(
    value: &'a DynValue,
    label: &str,
) -> Result<(Vec<usize>, Cow<'a, [f16]>), EngineError> {
    match value.try_extract_tensor::<f16>() {
        Ok((shape, data)) => Ok((
            shape.iter().map(|&dim| dim as usize).collect(),
            Cow::Borrowed(data),
        )),
        Err(f16_err) => {
            let (shape, data_f32) = value.try_extract_tensor::<f32>().map_err(|f32_err| {
                EngineError::backend(format!(
                    "Failed to extract {} as f16 ({:?}) or f32 ({:?})",
                    label, f16_err, f32_err
                ))
            })?;
            Ok((
                shape.iter().map(|&dim| dim as usize).collect(),
                Cow::Owned(data_f32.iter().map(|v| f16::from_f32(*v)).collect()),
            ))
        }
    }
}

fn extract_tensor_f32<'a>(
    value: &'a DynValue,
    label: &str,
) -> Result<(Vec<usize>, Cow<'a, [f32]>), EngineError> {
    match value.try_extract_tensor::<f32>() {
        Ok((shape, data)) => Ok((
            shape.iter().map(|&dim| dim as usize).collect(),
            Cow::Borrowed(data),
        )),
        Err(f32_err) => {
            let (shape, data_f16) = value.try_extract_tensor::<f16>().map_err(|f16_err| {
                EngineError::backend(format!(
                    "Failed to extract {} as f32 ({:?}) or f16 ({:?})",
                    label, f32_err, f16_err
                ))
            })?;
            Ok((
                shape.iter().map(|&dim| dim as usize).collect(),
                Cow::Owned(data_f16.iter().map(|v| v.to_f32()).collect()),
            ))
        }
    }
}

fn build_packed_kv_input_f16<'a>(
    data: Arc<[f16]>,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    layout: KvLayout,
    label: &str,
) -> Result<SessionInputValue<'a>, EngineError> {
    match layout {
        KvLayout::SeqFirst => {
            TensorRef::from_array_view(([1usize, num_heads, seq_len, head_dim], data))
                .map(|tensor| tensor.into())
                .map_err(|e| {
                    EngineError::backend(format!("Failed to make {} tensor view: {}", label, e))
                })
        }
        KvLayout::HeadDimFirst => {
            let arr = build_kv_array_f16_from_packed(
                data.as_ref(),
                num_heads,
                seq_len,
                head_dim,
                layout,
                label,
            )?;
            Value::from_array(arr)
                .map(|value| value.into())
                .map_err(|e| EngineError::backend(e.to_string()))
        }
    }
}

#[derive(Clone)]
enum EmptyKvArray {
    F16(Arc<ArrayD<f16>>),
    F32(Arc<ArrayD<f32>>),
}

fn empty_kv_shape(
    shape_def: Option<&Vec<i64>>,
    kv_layout: KvLayout,
    num_heads: usize,
    head_dim: usize,
) -> Vec<usize> {
    // Use seq_len=1 (not 0) so CoreML EP doesn't reject zero-element tensors.
    // The single dummy timestep is masked out by a leading 0 in the attention mask.
    if let Some(def) = shape_def {
        if def.len() == 4 {
            return match kv_layout {
                KvLayout::SeqFirst => vec![1, num_heads, 1, head_dim],
                KvLayout::HeadDimFirst => vec![1, num_heads, head_dim, 1],
            };
        }

        let mut shape: Vec<usize> = def
            .iter()
            .map(|&d| if d > 0 { d as usize } else { 1 })
            .collect();
        if !shape.is_empty() {
            shape[0] = 1;
        }
        return shape;
    }

    match kv_layout {
        KvLayout::SeqFirst => vec![1, num_heads, 1, head_dim],
        KvLayout::HeadDimFirst => vec![1, num_heads, head_dim, 1],
    }
}

fn empty_kv_cache_key(name: &str, dtype: TensorElementType, shape: &[usize]) -> String {
    format!("{}|{:?}|{:?}", name, dtype, shape)
}

fn get_empty_kv_value_cached(
    empty_kv_cache: &mut HashMap<String, EmptyKvArray>,
    name: &str,
    dtype: TensorElementType,
    shape: &[usize],
) -> Result<DynValue, EngineError> {
    let key = empty_kv_cache_key(name, dtype, shape);
    let entry = if let Some(cached) = empty_kv_cache.get(&key) {
        cached.clone()
    } else {
        let total_elems: usize = shape.iter().product();
        let new_entry = match dtype {
            TensorElementType::Float16 => EmptyKvArray::F16(Arc::new(
                ArrayD::<f16>::from_shape_vec(IxDyn(shape), vec![f16::ZERO; total_elems]).map_err(
                    |e| {
                        EngineError::backend(format!(
                            "Failed to build empty {} tensor: {:?}",
                            name, e
                        ))
                    },
                )?,
            )),
            TensorElementType::Float32 => EmptyKvArray::F32(Arc::new(
                ArrayD::<f32>::from_shape_vec(IxDyn(shape), vec![0f32; total_elems]).map_err(
                    |e| {
                        EngineError::backend(format!(
                            "Failed to build empty {} tensor: {:?}",
                            name, e
                        ))
                    },
                )?,
            )),
            other => {
                return Err(EngineError::backend(format!(
                    "Unsupported KV tensor dtype for {}: {:?}",
                    name, other
                )))
            }
        };
        empty_kv_cache.insert(key, new_entry.clone());
        new_entry
    };

    match &entry {
        EmptyKvArray::F16(arr) => Value::from_array(arr.as_ref().clone())
            .map_err(|e| EngineError::backend(e.to_string()))
            .map(|v| v.into_dyn()),
        EmptyKvArray::F32(arr) => Value::from_array(arr.as_ref().clone())
            .map_err(|e| EngineError::backend(e.to_string()))
            .map(|v| v.into_dyn()),
    }
}

/// Item used during batched prefill steps; lives at module level so the pool Vec
/// can be stored on LLMEngine and reused across calls without reallocating.
struct PrefillItem {
    group_arc: Arc<Mutex<SequenceGroup>>,
    seq_arc: Arc<Mutex<Sequence>>,
    seq_id: u64,
    input_tokens: Vec<u32>,
    input_len: usize,
    total_len: usize,
}

/// Item used during batched decode steps; lives at module level for the same reason.
struct BatchItem {
    group_arc: Arc<Mutex<SequenceGroup>>,
    seq_arc: Arc<Mutex<Sequence>>,
    seq_id: u64,
    input_token: u32,
    total_len: usize,
}

/// LLM engine: prepares inputs for ONNX Runtime and manages KV cache.
/// Captures model-declared input names and input shapes at load time and ensures
/// required past_key_values inputs exist (even as zero-length tensors) to satisfy graph ops.
pub struct LLMEngine {
    scheduler: LLMScheduler,
    _kernel_backend: Box<dyn KernelBackend>,
    request_rx: mpsc::Receiver<SequenceGroup>,
    sessions: HashMap<String, Arc<Mutex<Sequence>>>,
    session: Option<Session>,
    decode_session: Option<Session>,
    tokenizer: Option<Tokenizer>,
    tokenizer_path: Option<PathBuf>,
    model_path: Option<PathBuf>,
    provider_override: Option<String>,
    device_id_override: Option<i32>,
    device_ids_override: Option<Vec<i32>>,
    pipeline_stages: Option<Vec<PipelineStage>>,

    // KV cache (recreated when model geometry detected)
    kv_cache: KvCache,
    kv_cache_config: KvCacheConfig,
    metrics: Arc<Mutex<LLMMetrics>>,

    _block_size: usize,
    next_sequence_id: AtomicU64,

    // Keep the set of input names the model declares: used to avoid sending unknown inputs.
    // Wrapped in Arc so per-step clone is O(1) pointer bump instead of deep copy.
    model_input_names: Arc<HashSet<String>>,
    // Keep the model-declared input shapes (if available) so we can construct zero-length tensors
    // that match the model's expected ranks when needed.
    model_input_shapes: Arc<HashMap<String, Vec<i64>>>,
    // Keep model-declared input element types so we can match expected dtypes.
    model_input_types: Arc<HashMap<String, TensorElementType>>,
    // Optional dedicated decode graph metadata. When present, single-token decode can run through
    // a smaller ONNX/CoreML session than the primary prefill graph.
    decode_model_input_names: Arc<HashSet<String>>,
    decode_model_input_shapes: Arc<HashMap<String, Vec<i64>>>,
    decode_model_input_types: Arc<HashMap<String, TensorElementType>>,
    // Cache of zero-length KV arrays keyed by name/dtype/shape to avoid reallocs.
    empty_kv_cache: HashMap<String, EmptyKvArray>,

    // Detected model geometry (set at load time). Defaults are used until detection.
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    max_session_tokens: usize,
    kv_layout: KvLayout,
    kv_admission_soft_limit_bytes: Option<usize>,
    /// Per-replica cap on KV cache total_blocks set by the auto-scaler.
    /// Overrides the value from metadata/env when set, so that memory is divided
    /// fairly across replicas sharing the same device.
    kv_blocks_cap: Option<usize>,
    kv_admission_hard_limit_bytes: Option<usize>,
    // Detected vocabulary size (optional). Populated at load() when available.
    vocab_size: Option<usize>,
    // Detected BOS token id (optional).
    bos_token_id: Option<u32>,
    // Whether to use KV cache inputs/outputs. Some models export these but don't support non-zero past.
    use_kv_cache: bool,
    // Guard to avoid repeated in-loop recovery churn.
    coreml_cpu_fallback_attempted: bool,
    // Reusable workspace for KV packing in batched decode (avoids per-layer allocation).
    kv_workspace_key: Vec<f16>,
    kv_workspace_val: Vec<f16>,
    // Reusable workspace for logits modification (avoids per-sequence clone).
    logits_workspace: Vec<f32>,
    // Reusable Vec storage for per-batch items (avoids realloc on each batched step).
    prefill_items_pool: Vec<PrefillItem>,
    decode_items_pool: Vec<BatchItem>,
}

impl LLMEngine {
    fn get_empty_kv_value(
        &mut self,
        name: &str,
        dtype: TensorElementType,
        shape: &[usize],
    ) -> Result<DynValue, EngineError> {
        get_empty_kv_value_cached(&mut self.empty_kv_cache, name, dtype, shape)
    }

    fn is_coreml_runtime_error(err: &EngineError) -> bool {
        let msg = err.to_string().to_ascii_lowercase();
        msg.contains("coremlexecutionprovider")
            || msg.contains("coreml_execution_provider")
            || msg.contains("coreml ep")
    }

    async fn try_recover_from_coreml_failure(&mut self, err: &EngineError) -> bool {
        if !Self::is_coreml_runtime_error(err) || self.coreml_cpu_fallback_attempted {
            return false;
        }
        if self
            .provider_override
            .as_deref()
            .map(|provider| provider.eq_ignore_ascii_case("cpu"))
            .unwrap_or(false)
        {
            return false;
        }

        let Some(model_path) = self.model_path.clone() else {
            return false;
        };

        self.coreml_cpu_fallback_attempted = true;
        let prev_provider_override = self.provider_override.clone();
        let prev_device_id_override = self.device_id_override;
        let prev_device_ids_override = self.device_ids_override.clone();

        self.provider_override = Some("cpu".to_string());
        self.device_id_override = Some(0);
        self.device_ids_override = None;

        log::warn!(
            "Detected CoreML runtime failure. Reloading model on CPU for subsequent requests."
        );

        match self.load(&model_path).await {
            Ok(_) => {
                self.coreml_cpu_fallback_attempted = true;
                log::warn!("LLM engine recovered on CPU provider.");
                true
            }
            Err(load_err) => {
                log::error!(
                    "CPU recovery load failed after CoreML runtime error: {}",
                    load_err
                );
                self.provider_override = prev_provider_override;
                self.device_id_override = prev_device_id_override;
                self.device_ids_override = prev_device_ids_override;
                self.coreml_cpu_fallback_attempted = true;
                false
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scheduler_config: SchedulerConfig,
        block_size: usize,
        num_gpu_blocks: usize,
        request_rx: mpsc::Receiver<SequenceGroup>,
        metrics: Arc<Mutex<LLMMetrics>>,
        provider_override: Option<String>,
        device_id_override: Option<i32>,
        device_ids_override: Option<Vec<i32>>,
    ) -> Self {
        let block_manager = BlockManager::new(num_gpu_blocks, block_size, 0);
        let scheduler = LLMScheduler::new(scheduler_config, block_manager);
        let kernel_backend = kapsl_kernels::create_backend();

        // Initialize kv_cache with defaults; will be replaced in load() if we detect geometry.
        let kv_cache = KvCache::new(
            DEFAULT_NUM_LAYERS,
            DEFAULT_NUM_HEADS,
            MAX_SEQ_LEN,
            DEFAULT_HEAD_DIM,
        );

        Self {
            scheduler,
            _kernel_backend: kernel_backend,
            request_rx,
            sessions: HashMap::new(),
            session: None,
            decode_session: None,
            tokenizer: None,
            tokenizer_path: None,
            model_path: None,
            provider_override,
            device_id_override,
            device_ids_override,
            pipeline_stages: None,
            kv_cache,
            kv_cache_config: KvCacheConfig::default(),
            metrics,
            _block_size: block_size,
            next_sequence_id: AtomicU64::new(1),
            model_input_names: Arc::new(HashSet::new()),
            model_input_shapes: Arc::new(HashMap::new()),
            model_input_types: Arc::new(HashMap::new()),
            decode_model_input_names: Arc::new(HashSet::new()),
            decode_model_input_shapes: Arc::new(HashMap::new()),
            decode_model_input_types: Arc::new(HashMap::new()),
            empty_kv_cache: HashMap::new(),
            num_layers: DEFAULT_NUM_LAYERS,
            num_heads: DEFAULT_NUM_HEADS,
            head_dim: DEFAULT_HEAD_DIM,
            max_seq_len: MAX_SEQ_LEN,
            max_session_tokens: MAX_SEQ_LEN * 4,
            kv_layout: KvLayout::SeqFirst,
            kv_admission_soft_limit_bytes: None,
            kv_admission_hard_limit_bytes: None,
            kv_blocks_cap: None,
            vocab_size: None,
            bos_token_id: None,
            use_kv_cache: true,
            coreml_cpu_fallback_attempted: false,
            kv_workspace_key: Vec::new(),
            kv_workspace_val: Vec::new(),
            logits_workspace: Vec::new(),
            prefill_items_pool: Vec::new(),
            decode_items_pool: Vec::new(),
        }
    }

    /// Replace the private block allocator with a shared one so this engine
    /// draws from the same GPU KV block pool as other engines.
    ///
    /// Must be called **before** `load()`.  Engines sharing the same
    /// `SharedBlockAllocator` form a unified KV block pool: blocks freed by
    /// any engine become immediately available to all others.
    pub fn with_shared_pool(mut self, allocator: SharedBlockAllocator) -> Self {
        let block_size = self._block_size;
        let block_manager = BlockManager::new_shared(allocator, block_size);
        let scheduler = LLMScheduler::new(
            SchedulerConfig {
                max_num_batched_tokens: self.scheduler.config_max_num_batched_tokens(),
                max_num_seqs: self.scheduler.config_max_num_seqs(),
                max_paddings: self.scheduler.config_max_paddings(),
            },
            block_manager,
        );
        self.scheduler = scheduler;
        self
    }

    /// Cap the KV cache `total_blocks` to `cap` during the next `load()` call.
    /// Called by the runtime when creating additional replicas so that each
    /// replica receives a proportional share of the device's block budget.
    pub fn set_kv_blocks_cap(&mut self, cap: usize) {
        self.kv_blocks_cap = Some(cap);
    }

    /// Load model and capture its input names and shapes. Detect KV geometry
    /// (num_layers, num_heads, head_dim) from declared inputs when possible and
    /// reinitialize kv_cache accordingly.
    pub async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        if !model_path.exists() {
            return Err(EngineError::backend(format!(
                "Model file does not exist: {}",
                model_path.display()
            )));
        }

        self.model_path = Some(model_path.to_path_buf());
        self.coreml_cpu_fallback_attempted = false;
        // Reset to default each load; metadata and model capabilities may override.
        self.use_kv_cache = true;
        let model_root = find_model_root(model_path);
        ensure_external_data_near_model(model_path, &model_root);
        let mut metadata_disable_kv_cache = false;
        let mut metadata_safe_load: Option<SafeLoadSetting> = None;
        let mut preferred_provider: Option<String> = None;
        let mut preferred_device_id: i32 = 0;
        let mut optimization_level: GraphOptimizationLevel = GraphOptimizationLevel::Level2;
        let mut preferred_intra_threads: Option<usize> = None;
        let mut kv_cache_config = KvCacheConfig::default();
        let mut pipeline_stage_files: Vec<String> = Vec::new();
        // Manifest hints for model architecture — used as fallbacks when ONNX / config.json
        // detection doesn't find geometry, and to right-size max_seq_len.
        let mut manifest_num_layers: Option<usize> = None;
        let mut manifest_num_kv_heads: Option<usize> = None;
        let mut manifest_head_dim: Option<usize> = None;
        let mut manifest_max_seq_len: Option<usize> = None;
        let mut manifest_vocab_size: Option<usize> = None;
        let mut manifest_max_session_tokens: Option<usize> = None;
        let mut decode_model_file: Option<String> = None;
        let mut kv_admission_soft_limit_bytes: Option<usize> = None;
        let mut kv_admission_hard_limit_bytes: Option<usize> = None;

        let meta_path = model_root.join("metadata.json");
        if let Ok(file) = std::fs::File::open(meta_path) {
            if let Ok(meta) = serde_json::from_reader::<_, serde_json::Value>(file) {
                if let Some(hr) = meta.get("hardware_requirements") {
                    preferred_provider = hr
                        .get("preferred_provider")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    preferred_device_id =
                        hr.get("device_id").and_then(|v| v.as_i64()).unwrap_or(0) as i32;

                    if let Some(opt) = hr.get("graph_optimization_level").and_then(|v| v.as_str()) {
                        optimization_level = match opt.to_lowercase().as_str() {
                            "none" | "disabled" => GraphOptimizationLevel::Disable,
                            "basic" | "level1" => GraphOptimizationLevel::Level1,
                            "extended" | "level2" => GraphOptimizationLevel::Level2,
                            "all" | "level3" => GraphOptimizationLevel::Level3,
                            _ => GraphOptimizationLevel::Level2,
                        };
                    }
                    if let Some(t) = hr.get("intra_op_threads").and_then(|v| v.as_u64()) {
                        if t > 0 {
                            preferred_intra_threads = Some(t as usize);
                        }
                    }
                }

                let disable = meta
                    .get("metadata")
                    .and_then(|m| m.get("llm"))
                    .and_then(|m| m.get("disable_kv_cache"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if disable {
                    metadata_disable_kv_cache = true;
                }
                let safe_load_value = meta
                    .get("metadata")
                    .and_then(|m| m.get("llm"))
                    .and_then(|m| m.get("safe_load"));
                if let Some(value) = safe_load_value {
                    metadata_safe_load = parse_safe_load_setting(value);
                    if metadata_safe_load.is_none() {
                        log::warn!("Unknown metadata.llm.safe_load value; expected bool or 'auto'");
                    }
                }
                if let Some(value) = meta
                    .get("metadata")
                    .and_then(|m| m.get("llm"))
                    .and_then(|m| m.get("decode_model_file"))
                    .and_then(|v| v.as_str())
                {
                    let trimmed = value.trim();
                    if !trimmed.is_empty() {
                        decode_model_file = Some(trimmed.to_string());
                    }
                }

                if let Some(kv) = meta
                    .get("metadata")
                    .and_then(|m| m.get("llm"))
                    .and_then(|m| m.get("kv_cache"))
                {
                    if let Some(mode) = kv.get("mode").and_then(|v| v.as_str()) {
                        match mode.to_ascii_lowercase().as_str() {
                            "paged" => kv_cache_config.mode = KvCacheMode::Paged,
                            "dense" => kv_cache_config.mode = KvCacheMode::Dense,
                            other => {
                                log::warn!("Unknown kv_cache mode '{}', defaulting to dense", other)
                            }
                        }
                    }
                    if let Some(block_size) = kv.get("block_size").and_then(|v| v.as_u64()) {
                        if block_size > 0 {
                            kv_cache_config.block_size = block_size as usize;
                        }
                    }
                    if let Some(total_blocks) = kv.get("total_blocks").and_then(|v| v.as_u64()) {
                        if total_blocks > 0 {
                            kv_cache_config.total_blocks = total_blocks as usize;
                        }
                    }
                    if let Some(eviction) = kv.get("eviction").and_then(|v| v.as_str()) {
                        match eviction.to_ascii_lowercase().as_str() {
                            "none" => kv_cache_config.eviction_policy = KvEvictionPolicy::None,
                            "lru" | "lru_inactive" => {
                                kv_cache_config.eviction_policy = KvEvictionPolicy::LruInactive
                            }
                            other => log::warn!(
                                "Unknown kv_cache eviction policy '{}', defaulting to none",
                                other
                            ),
                        }
                    }
                    if let Some(v) = kv.get("initial_seq_len").and_then(|v| v.as_u64()) {
                        if v > 0 {
                            kv_cache_config.initial_seq_len = v as usize;
                        }
                    }
                    if let Some(v) = kv.get("dense_free_list_cap").and_then(|v| v.as_u64()) {
                        if v > 0 {
                            kv_cache_config.dense_free_list_cap = v as usize;
                        }
                    }
                    if let Some(v) = kv
                        .get("admission_soft_limit_bytes")
                        .and_then(parse_byte_size_json)
                    {
                        if v > 0 {
                            kv_admission_soft_limit_bytes = Some(v);
                        }
                    }
                    if let Some(v) = kv
                        .get("admission_hard_limit_bytes")
                        .and_then(parse_byte_size_json)
                    {
                        if v > 0 {
                            kv_admission_hard_limit_bytes = Some(v);
                        }
                    }
                }

                if let Some(stages) = meta
                    .get("metadata")
                    .and_then(|m| m.get("llm"))
                    .and_then(|m| m.get("pipeline"))
                    .and_then(|m| m.get("stages"))
                    .and_then(|v| v.as_array())
                {
                    for stage in stages {
                        if let Some(name) = stage.as_str() {
                            pipeline_stage_files.push(name.to_string());
                        }
                    }
                }

                let llm_meta = meta.get("metadata").and_then(|m| m.get("llm"));
                if let Some(llm) = llm_meta {
                    if let Some(v) = llm.get("num_layers").and_then(|v| v.as_u64()) {
                        if v > 0 {
                            manifest_num_layers = Some(v as usize);
                        }
                    }
                    if let Some(v) = llm
                        .get("num_kv_heads")
                        .or_else(|| llm.get("num_key_value_heads"))
                        .and_then(|v| v.as_u64())
                    {
                        if v > 0 {
                            manifest_num_kv_heads = Some(v as usize);
                        }
                    }
                    if let Some(v) = llm.get("head_dim").and_then(|v| v.as_u64()) {
                        if v > 0 {
                            manifest_head_dim = Some(v as usize);
                        }
                    }
                    if let Some(v) = llm
                        .get("max_sequence_length")
                        .or_else(|| llm.get("max_seq_len"))
                        .and_then(|v| v.as_u64())
                    {
                        if v > 0 {
                            manifest_max_seq_len = Some(v as usize);
                        }
                    }
                    if let Some(v) = llm.get("vocab_size").and_then(|v| v.as_u64()) {
                        if v > 0 {
                            manifest_vocab_size = Some(v as usize);
                        }
                    }
                    if let Some(v) = llm.get("max_session_tokens").and_then(|v| v.as_u64()) {
                        if v > 0 {
                            manifest_max_session_tokens = Some(v as usize);
                        }
                    }
                }
            }
        }

        if let Some(provider) = &self.provider_override {
            preferred_provider = Some(provider.clone());
        }
        if let Some(device_id) = self.device_id_override {
            preferred_device_id = device_id;
        }
        if self.provider_override.is_some() || self.device_id_override.is_some() {
            log::info!(
                "LLM using overridden device selection: provider={:?}, device_id={}",
                preferred_provider,
                preferred_device_id
            );
        }
        if self.provider_override.is_none() && llm_provider_policy() != "manifest" {
            let provider_is_cpu_or_missing = preferred_provider
                .as_deref()
                .map(|provider| provider.trim().eq_ignore_ascii_case("cpu"))
                .unwrap_or(true);
            if provider_is_cpu_or_missing {
                let fastest = fastest_llm_provider().to_string();
                if fastest != "cpu" {
                    log::info!(
                        "LLM auto-selecting fastest provider `{}` (set KAPSL_PROVIDER_POLICY=manifest to keep package provider)",
                        fastest
                    );
                }
                preferred_provider = Some(fastest);
            }
        }
        if let Some(provider) = preferred_provider.as_ref() {
            if !llm_provider_available(provider) {
                log::warn!(
                    "Requested LLM provider `{}` is unavailable, falling back to CPU",
                    provider
                );
                preferred_provider = Some("cpu".to_string());
            }
        }
        if preferred_provider
            .as_deref()
            .map(|provider| {
                matches!(
                    provider.trim().to_ascii_lowercase().as_str(),
                    "coreml" | "metal"
                )
            })
            .unwrap_or(false)
            && has_dot_external_data(model_path)
        {
            log::warn!(
                "Model uses `{}` external tensor data; forcing CPU because CoreML may fail to resolve this layout.",
                model_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| format!("{name}.data"))
                    .unwrap_or_else(|| "model .data".to_string())
            );
            preferred_provider = Some("cpu".to_string());
        }

        if let Some(mode) = env_var_alias("KAPSL_LLM_KV_CACHE_MODE", "KAPSL_LLM_KV_CACHE_MODE") {
            match mode.to_ascii_lowercase().as_str() {
                "paged" => kv_cache_config.mode = KvCacheMode::Paged,
                "dense" => kv_cache_config.mode = KvCacheMode::Dense,
                other => log::warn!(
                    "Unknown KAPSL_LLM_KV_CACHE_MODE '{}', using metadata/default",
                    other
                ),
            }
        }
        if let Some(block_size) = env_var_alias(
            "KAPSL_LLM_KV_CACHE_BLOCK_SIZE",
            "KAPSL_LLM_KV_CACHE_BLOCK_SIZE",
        ) {
            if let Ok(parsed) = block_size.parse::<usize>() {
                if parsed > 0 {
                    kv_cache_config.block_size = parsed;
                }
            }
        }
        if let Some(total_blocks) = env_var_alias(
            "KAPSL_LLM_KV_CACHE_TOTAL_BLOCKS",
            "KAPSL_LLM_KV_CACHE_TOTAL_BLOCKS",
        ) {
            if let Ok(parsed) = total_blocks.parse::<usize>() {
                if parsed > 0 {
                    kv_cache_config.total_blocks = parsed;
                }
            }
        }
        // Apply per-replica block cap set by the auto-scaler. The env var above
        // acts as an absolute override; this cap is a soft upper bound that the
        // auto-scaler uses to distribute the block budget across replicas.
        if let Some(cap) = self.kv_blocks_cap {
            if cap > 0 && kv_cache_config.total_blocks > cap {
                log::info!(
                    "KV cache total_blocks reduced from {} to {} (per-replica cap)",
                    kv_cache_config.total_blocks,
                    cap,
                );
                kv_cache_config.total_blocks = cap;
            }
        }
        if let Some(eviction) =
            env_var_alias("KAPSL_LLM_KV_CACHE_EVICTION", "KAPSL_LLM_KV_CACHE_EVICTION")
        {
            match eviction.to_ascii_lowercase().as_str() {
                "none" => kv_cache_config.eviction_policy = KvEvictionPolicy::None,
                "lru" | "lru_inactive" => {
                    kv_cache_config.eviction_policy = KvEvictionPolicy::LruInactive
                }
                other => log::warn!(
                    "Unknown KAPSL_LLM_KV_CACHE_EVICTION '{}', using metadata/default",
                    other
                ),
            }
        }
        if let Some(v) = env_var_alias(
            "KAPSL_LLM_KV_INITIAL_SEQ_LEN",
            "KAPSL_LLM_KV_INITIAL_SEQ_LEN",
        ) {
            if let Ok(parsed) = v.parse::<usize>() {
                if parsed > 0 {
                    kv_cache_config.initial_seq_len = parsed;
                }
            }
        }
        if let Some(v) = env_var_alias("KAPSL_LLM_KV_FREE_LIST_CAP", "KAPSL_LLM_KV_FREE_LIST_CAP") {
            if let Ok(parsed) = v.parse::<usize>() {
                if parsed > 0 {
                    kv_cache_config.dense_free_list_cap = parsed;
                }
            }
        }
        if let Some(v) = env_var_alias(
            "KAPSL_LLM_KV_ADMISSION_SOFT_LIMIT_BYTES",
            "KAPSL_LLM_KV_ADMISSION_SOFT_LIMIT_BYTES",
        ) {
            if let Some(parsed) = parse_byte_size(&v) {
                if parsed > 0 {
                    kv_admission_soft_limit_bytes = Some(parsed);
                }
            }
        }
        if let Some(v) = env_var_alias(
            "KAPSL_LLM_KV_ADMISSION_HARD_LIMIT_BYTES",
            "KAPSL_LLM_KV_ADMISSION_HARD_LIMIT_BYTES",
        ) {
            if let Some(parsed) = parse_byte_size(&v) {
                if parsed > 0 {
                    kv_admission_hard_limit_bytes = Some(parsed);
                }
            }
        }
        let tokenizer_path = find_model_asset(model_path, "tokenizer.json").ok_or_else(|| {
            EngineError::backend(format!(
                "Tokenizer file not found near model or package root for {}",
                model_path.display()
            ))
        })?;
        self.tokenizer_path = Some(tokenizer_path.clone());
        self.tokenizer = Some(
            Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| EngineError::backend(e.to_string()))?,
        );

        // Build the ONNX session so we can inspect its declared inputs.
        let make_builder = |provider: Option<&str>,
                            device_id: i32|
         -> Result<SessionBuilder, EngineError> {
            let mut builder =
                Session::builder().map_err(|e| EngineError::backend(e.to_string()))?;

            if let Some(provider) = provider {
                let p_lower = provider.to_lowercase();
                match p_lower.as_str() {
                    "coreml" | "metal" => {
                        if CoreMLExecutionProvider::default()
                            .is_available()
                            .unwrap_or(false)
                        {
                            builder = builder
                                .with_execution_providers([
                                    CoreMLExecutionProvider::default().build()
                                ])
                                .map_err(|e| {
                                    EngineError::backend(format!("Failed to enable CoreML: {}", e))
                                })?;
                            log::info!("LLM using CoreML execution provider");
                        } else {
                            log::warn!("CoreML requested but not available, falling back to CPU");
                        }
                    }
                    "directml" => {
                        #[cfg(target_os = "windows")]
                        {
                            if DirectMLExecutionProvider::default()
                                .is_available()
                                .unwrap_or(false)
                            {
                                builder = builder
                                    .with_execution_providers([DirectMLExecutionProvider::default(
                                    )
                                    .with_device_id(device_id)
                                    .build()])
                                    .map_err(|e| {
                                        EngineError::backend(format!(
                                            "Failed to enable DirectML: {}",
                                            e
                                        ))
                                    })?;
                                log::info!("LLM using DirectML on device {}", device_id);
                            } else {
                                log::warn!(
                                    "DirectML requested but not available, falling back to CPU"
                                );
                            }
                        }
                        #[cfg(not(target_os = "windows"))]
                        {
                            log::warn!(
                                "DirectML requested on non-Windows host, falling back to CPU"
                            );
                        }
                    }
                    "cuda" => {
                        if CUDAExecutionProvider::default()
                            .is_available()
                            .unwrap_or(false)
                        {
                            builder = builder
                                .with_execution_providers([CUDAExecutionProvider::default()
                                    .with_device_id(device_id)
                                    .build()])
                                .map_err(|e| {
                                    EngineError::backend(format!("Failed to enable CUDA: {}", e))
                                })?;
                            log::info!("LLM using CUDA on device {}", device_id);
                        } else {
                            log::warn!("CUDA requested but not available, falling back to CPU");
                        }
                    }
                    "tensorrt" => {
                        if TensorRTExecutionProvider::default()
                            .is_available()
                            .unwrap_or(false)
                        {
                            builder = builder
                                .with_execution_providers([
                                    TensorRTExecutionProvider::default()
                                        .with_device_id(device_id)
                                        .build(),
                                    CUDAExecutionProvider::default()
                                        .with_device_id(device_id)
                                        .build(),
                                ])
                                .map_err(|e| {
                                    EngineError::backend(format!(
                                        "Failed to enable TensorRT: {}",
                                        e
                                    ))
                                })?;
                            log::info!(
                                "LLM using TensorRT (+ CUDA fallback) on device {}",
                                device_id
                            );
                        } else {
                            log::warn!("TensorRT requested but not available, falling back to CPU");
                        }
                    }
                    "rocm" => {
                        if ROCmExecutionProvider::default()
                            .is_available()
                            .unwrap_or(false)
                        {
                            builder = builder
                                .with_execution_providers([ROCmExecutionProvider::default()
                                    .with_device_id(device_id)
                                    .build()])
                                .map_err(|e| {
                                    EngineError::backend(format!("Failed to enable ROCm: {}", e))
                                })?;
                            log::info!("LLM using ROCm on device {}", device_id);
                        } else {
                            log::warn!("ROCm requested but not available, falling back to CPU");
                        }
                    }
                    "openvino" => {
                        if OpenVINOExecutionProvider::default()
                            .is_available()
                            .unwrap_or(false)
                        {
                            builder = builder
                                .with_execution_providers([
                                    OpenVINOExecutionProvider::default().build()
                                ])
                                .map_err(|e| {
                                    EngineError::backend(format!(
                                        "Failed to enable OpenVINO: {}",
                                        e
                                    ))
                                })?;
                            log::info!("LLM using OpenVINO execution provider");
                        } else {
                            log::warn!("OpenVINO requested but not available, falling back to CPU");
                        }
                    }
                    "cpu" => {
                        log::info!("LLM using CPU execution");
                    }
                    _ => {
                        log::warn!("Unknown provider '{}', falling back to CPU", provider);
                    }
                }
            }

            builder = builder
                .with_optimization_level(optimization_level)
                .map_err(|e| EngineError::backend(e.to_string()))?;
            builder = builder
                .with_memory_pattern(false)
                .map_err(|e| EngineError::backend(e.to_string()))?;
            if let Some(threads) = preferred_intra_threads {
                builder = builder
                    .with_intra_threads(threads)
                    .map_err(|e| EngineError::backend(e.to_string()))?;
            }
            Ok(builder)
        };
        let apply_safe_load = |mut builder: SessionBuilder| -> Result<SessionBuilder, EngineError> {
            builder = builder
                .with_prepacking(false)
                .map_err(|e| EngineError::backend(e.to_string()))?
                .with_intra_threads(1)
                .map_err(|e| EngineError::backend(e.to_string()))?
                .with_inter_threads(1)
                .map_err(|e| EngineError::backend(e.to_string()))?
                .with_parallel_execution(false)
                .map_err(|e| EngineError::backend(e.to_string()))?
                .with_config_entry("session.disable_cpu_mem_arena", "1")
                .map_err(|e| EngineError::backend(e.to_string()))?;
            Ok(builder)
        };

        let safe_load_env = env_var_alias("KAPSL_LLM_SAFE_LOAD", "KAPSL_LLM_SAFE_LOAD");
        let safe_load_setting = if let Some(value) = safe_load_env.as_deref() {
            if value == "0" {
                SafeLoadSetting::ForceOff
            } else {
                SafeLoadSetting::ForceOn
            }
        } else if let Some(metadata_setting) = metadata_safe_load {
            metadata_setting
        } else {
            SafeLoadSetting::Auto
        };
        let safe_load = match safe_load_setting {
            SafeLoadSetting::ForceOn => true,
            SafeLoadSetting::ForceOff => false,
            SafeLoadSetting::Auto => false,
        };

        let allow_cpu_fallback = self.provider_override.is_none();
        let try_build_session_with_provider = |path: &Path,
                                               provider: Option<&str>,
                                               device_id: i32|
         -> Result<Session, EngineError> {
            let mut builder = make_builder(provider, device_id)?;
            if safe_load {
                builder = apply_safe_load(builder)?;
                log::info!(
                    "LLM safe-load enabled: limiting ORT threads and disabling CPU mem arena."
                );
            }

            let session = match builder.commit_from_file(path) {
                Ok(session) => session,
                Err(err) => {
                    if !safe_load && safe_load_setting == SafeLoadSetting::Auto {
                        log::warn!(
                            "LLM safe-load disabled by default, but session creation failed; retrying with safe-load settings."
                        );
                        let mut builder = make_builder(provider, device_id)?;
                        builder = apply_safe_load(builder)?;
                        log::info!(
                            "LLM safe-load enabled: limiting ORT threads and disabling CPU mem arena."
                        );
                        builder
                            .commit_from_file(path)
                            .map_err(|e| EngineError::backend(e.to_string()))?
                    } else {
                        return Err(EngineError::backend(err.to_string()));
                    }
                }
            };

            Ok(session)
        };
        let build_session = |path: &Path, device_id: i32| -> Result<Session, EngineError> {
            let primary_provider = preferred_provider.as_deref();
            match try_build_session_with_provider(path, primary_provider, device_id) {
                Ok(session) => Ok(session),
                Err(primary_error) => {
                    let provider_name = primary_provider.unwrap_or("cpu");
                    if !allow_cpu_fallback || provider_name.eq_ignore_ascii_case("cpu") {
                        return Err(primary_error);
                    }

                    let primary_error_message = primary_error.to_string();
                    log::warn!(
                        "LLM provider `{}` failed to create session ({}). Retrying on CPU.",
                        provider_name,
                        primary_error_message
                    );
                    try_build_session_with_provider(path, Some("cpu"), 0).map_err(|cpu_error| {
                        EngineError::backend(format!(
                            "Failed to create session with provider `{}` ({}); CPU fallback also failed ({})",
                            provider_name,
                            primary_error_message,
                            cpu_error
                        ))
                    })
                }
            }
        };

        let mut input_names_set: HashSet<String> = HashSet::new();
        let mut input_shapes_map: HashMap<String, Vec<i64>> = HashMap::new();
        let mut input_types_map: HashMap<String, TensorElementType> = HashMap::new();
        let mut output_names_set: HashSet<String> = HashSet::new();
        let mut decode_input_names_set: HashSet<String> = HashSet::new();
        let mut decode_input_shapes_map: HashMap<String, Vec<i64>> = HashMap::new();
        let mut decode_input_types_map: HashMap<String, TensorElementType> = HashMap::new();
        let mut pipeline_stages: Option<Vec<PipelineStage>> = None;
        let mut session: Option<Session> = None;
        let mut decode_session: Option<Session> = None;

        if !pipeline_stage_files.is_empty() {
            if decode_model_file.is_some() {
                log::warn!(
                    "Ignoring metadata.llm.decode_model_file because pipeline execution uses staged sessions."
                );
            }
            let device_ids = if let Some(ids) = &self.device_ids_override {
                ids.clone()
            } else {
                vec![preferred_device_id; pipeline_stage_files.len()]
            };

            if device_ids.len() < pipeline_stage_files.len() {
                return Err(EngineError::backend(format!(
                    "Pipeline stages ({}) exceed available device ids ({})",
                    pipeline_stage_files.len(),
                    device_ids.len()
                )));
            }

            let mut stages = Vec::new();
            for (idx, stage_name) in pipeline_stage_files.iter().enumerate() {
                let stage_path = resolve_stage_path(&model_root, model_path, stage_name)
                    .ok_or_else(|| {
                        EngineError::backend(format!("Pipeline stage not found: {}", stage_name))
                    })?;
                let stage_device_id = device_ids.get(idx).copied().unwrap_or(preferred_device_id);
                log::info!(
                    "Loading pipeline stage {} from {} on device {}",
                    idx,
                    stage_path.display(),
                    stage_device_id
                );

                let stage_session = build_session(&stage_path, stage_device_id)?;
                let (stage_inputs, stage_shapes, stage_types, stage_outputs) =
                    capture_session_io(&stage_session);

                for name in stage_inputs.iter() {
                    input_names_set.insert(name.clone());
                }
                for (name, shape) in stage_shapes.iter() {
                    input_shapes_map
                        .entry(name.clone())
                        .or_insert_with(|| shape.clone());
                }
                for (name, ty) in stage_types.iter() {
                    input_types_map.entry(name.clone()).or_insert(*ty);
                }
                if idx + 1 == pipeline_stage_files.len() {
                    output_names_set = stage_outputs.clone();
                }

                stages.push(PipelineStage {
                    session: stage_session,
                    input_names: stage_inputs,
                    input_shapes: stage_shapes,
                    input_types: stage_types,
                });
            }

            if stages.is_empty() {
                return Err(EngineError::backend(
                    "Pipeline stages were requested but none were loaded".to_string(),
                ));
            }

            pipeline_stages = Some(stages);
        } else {
            let built_session = build_session(model_path, preferred_device_id)?;
            let (names, shapes, types, outputs) = capture_session_io(&built_session);
            input_names_set = names;
            input_shapes_map = shapes;
            input_types_map = types;
            output_names_set = outputs;

            if let Some(decode_name) = decode_model_file.as_deref() {
                if let Some(decode_path) = resolve_stage_path(&model_root, model_path, decode_name)
                {
                    if decode_path != model_path {
                        ensure_external_data_near_model(&decode_path, &model_root);
                        match build_session(&decode_path, preferred_device_id) {
                            Ok(built_decode_session) => {
                                let (dn, dsh, dt, _) = capture_session_io(&built_decode_session);
                                decode_input_names_set = dn;
                                decode_input_shapes_map = dsh;
                                decode_input_types_map = dt;
                                log::info!(
                                    "Loaded dedicated decode session from {}",
                                    decode_path.display()
                                );
                                decode_session = Some(built_decode_session);
                            }
                            Err(e) => {
                                log::warn!(
                                    "Failed to load dedicated decode model {}: {}. Falling back to primary session.",
                                    decode_path.display(),
                                    e
                                );
                            }
                        }
                    }
                } else {
                    log::warn!(
                        "Dedicated decode model '{}' was configured but could not be resolved; falling back to primary session.",
                        decode_name
                    );
                }
            }

            session = Some(built_session);
        }

        log::info!(
            "Model declared inputs: {:?}",
            input_names_set.iter().collect::<Vec<_>>()
        );

        let mut config_num_layers: Option<usize> = None;
        let mut config_num_heads: Option<usize> = None;
        let mut config_head_dim: Option<usize> = None;
        let mut config_bos_token_id: Option<u32> = None;
        let mut cfg_paths: Vec<PathBuf> = Vec::new();
        if let Some(parent) = model_path.parent() {
            cfg_paths.push(parent.join("config.json"));
            cfg_paths.push(parent.join("onnx-export").join("config.json"));
        }
        if model_path
            .parent()
            .map(|dir| dir != model_root.as_path())
            .unwrap_or(true)
        {
            cfg_paths.push(model_root.join("config.json"));
            cfg_paths.push(model_root.join("onnx-export").join("config.json"));
        }
        for cfg_path in cfg_paths {
            if let Ok(file) = std::fs::File::open(&cfg_path) {
                if let Ok(cfg) = serde_json::from_reader::<_, serde_json::Value>(file) {
                    if let Some(nl) = cfg.get("num_hidden_layers").and_then(|v| v.as_u64()) {
                        config_num_layers = Some(nl as usize);
                    }
                    if let Some(nkv) = cfg.get("num_key_value_heads").and_then(|v| v.as_u64()) {
                        config_num_heads = Some(nkv as usize);
                    } else if config_num_heads.is_none() {
                        if let Some(nh) = cfg.get("num_attention_heads").and_then(|v| v.as_u64()) {
                            config_num_heads = Some(nh as usize);
                        }
                    }
                    if let Some(hd) = cfg.get("head_dim").and_then(|v| v.as_u64()) {
                        config_head_dim = Some(hd as usize);
                    } else if config_head_dim.is_none() {
                        let hidden = cfg.get("hidden_size").and_then(|v| v.as_u64());
                        let attn = cfg.get("num_attention_heads").and_then(|v| v.as_u64());
                        if let (Some(hidden), Some(attn)) = (hidden, attn) {
                            if attn > 0 && hidden % attn == 0 {
                                config_head_dim = Some((hidden / attn) as usize);
                            }
                        }
                    }
                    if let Some(bos) = cfg.get("bos_token_id").and_then(|v| v.as_u64()) {
                        config_bos_token_id = Some(bos as u32);
                    }
                }
            }
        }

        let has_kv_inputs = input_names_set
            .iter()
            .any(|name| name.starts_with("past_key_values."));
        let has_kv_outputs = output_names_set
            .iter()
            .any(|name| name.starts_with("present."));
        if metadata_disable_kv_cache {
            log::info!("Disabling KV cache via metadata.json");
            self.use_kv_cache = false;
        } else if !has_kv_inputs || !has_kv_outputs {
            if !has_kv_inputs {
                log::info!("Model declares no past_key_values inputs; disabling KV cache");
            }
            if !has_kv_outputs {
                log::info!("Model declares no present KV outputs; disabling KV cache");
            }
            self.use_kv_cache = false;
        }

        // Detect KV geometry heuristically from declared past_key_values inputs.
        // Typical shape: [batch, num_heads, seq_len, head_dim]
        // Look for any "past_key_values.{layer}.key" entry and use its positive dims.
        let mut detected_num_layers: Option<usize> = None;
        let mut detected_num_heads: Option<usize> = None;
        let mut detected_head_dim: Option<usize> = None;

        for (name, shape) in &input_shapes_map {
            // try to match "past_key_values.{n}.key"
            if name.starts_with("past_key_values.") && name.ends_with(".key") {
                // shape is Vec<i64>, convert to Vec<i64> for checking
                if shape.len() >= 4 {
                    let heads = shape[1];
                    let hd = shape[3];
                    if heads > 0 {
                        detected_num_heads = Some(heads as usize);
                    }
                    if hd > 0 {
                        detected_head_dim = Some(hd as usize);
                    }
                } else if shape.len() == 3 {
                    // some exported graphs collapse batch dim; try best-effort
                    // possible layouts: [1, NUM_HEADS, HEAD_DIM] or [NUM_HEADS, seq, HEAD_DIM]
                    // we prefer to extract heads and head_dim if available
                    if shape[0] > 0 && shape[1] > 0 {
                        // ambiguous - pick conservative values only if not already detected.
                        // Use explicit Some(...) assignments instead of get_or_insert to avoid
                        // mutable-borrow / inference issues in certain toolchains.
                        if detected_num_heads.is_none() {
                            detected_num_heads = Some(shape[0] as usize);
                        }
                        if detected_head_dim.is_none() {
                            detected_head_dim = Some(shape[shape.len() - 1] as usize);
                        }
                    }
                }
            }

            // Also consider "present.{n}.key" outputs if they were present as inputs (some graphs declare them)
            if name.starts_with("present.") && name.ends_with(".key") && shape.len() >= 4 {
                let heads = shape[1];
                let hd = shape[3];
                if detected_num_heads.is_none() && heads > 0 {
                    detected_num_heads = Some(heads as usize);
                }
                if detected_head_dim.is_none() && hd > 0 {
                    detected_head_dim = Some(hd as usize);
                }
            }

            // try to infer number of layers from model-declared count (count of past_key_values.*.key)
            if name.starts_with("past_key_values.") && name.ends_with(".key") {
                // extract layer index
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() >= 3 {
                    if let Ok(idx) = parts[1].parse::<usize>() {
                        detected_num_layers =
                            Some(detected_num_layers.map_or(idx + 1, |v| v.max(idx + 1)));
                    }
                }
            }
        }

        // Apply detections, falling back to manifest hints, then hardcoded defaults.
        if let Some(nh) = config_num_heads {
            self.num_heads = nh;
        } else if let Some(nh) = detected_num_heads {
            self.num_heads = nh;
        } else if let Some(nh) = manifest_num_kv_heads {
            self.num_heads = nh;
        } else {
            self.num_heads = DEFAULT_NUM_HEADS;
        }
        if let Some(hd) = config_head_dim {
            self.head_dim = hd;
        } else if let Some(hd) = detected_head_dim {
            self.head_dim = hd;
        } else if let Some(hd) = manifest_head_dim {
            self.head_dim = hd;
        } else {
            self.head_dim = DEFAULT_HEAD_DIM;
        }
        if let Some(nl) = config_num_layers {
            self.num_layers = nl;
        } else if let Some(nl) = detected_num_layers {
            self.num_layers = nl;
        } else if let Some(nl) = manifest_num_layers {
            self.num_layers = nl;
        } else {
            self.num_layers = DEFAULT_NUM_LAYERS;
        }
        self.max_seq_len = manifest_max_seq_len.unwrap_or(MAX_SEQ_LEN);
        self.max_session_tokens =
            manifest_max_session_tokens.unwrap_or(self.max_seq_len.saturating_mul(4));

        self.kv_layout = infer_kv_layout(&input_shapes_map, self.num_heads, self.head_dim);

        log::info!(
            "Detected KV geometry -> layers: {}, heads: {}, head_dim: {}, max_seq_len: {}",
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.max_seq_len,
        );
        log::info!("Detected KV layout: {:?}", self.kv_layout);

        // Validate KV cache config when paged mode is selected.
        if kv_cache_config.mode == KvCacheMode::Paged
            && (kv_cache_config.block_size == 0 || kv_cache_config.total_blocks == 0)
        {
            log::warn!(
                "Invalid paged KV cache config (block_size={}, total_blocks={}); falling back to dense.",
                kv_cache_config.block_size,
                kv_cache_config.total_blocks
            );
            kv_cache_config.mode = KvCacheMode::Dense;
        }

        self.kv_cache_config = kv_cache_config.clone();
        self.kv_admission_soft_limit_bytes = kv_admission_soft_limit_bytes;
        self.kv_admission_hard_limit_bytes = kv_admission_hard_limit_bytes;
        // Recreate kv_cache using detected geometry (max_seq_len from manifest or default).
        self.kv_cache = KvCache::new_with_config(
            self.num_layers,
            self.num_heads,
            self.max_seq_len,
            self.head_dim,
            kv_cache_config,
        );
        log::info!(
            "KV cache mode: {:?} (block_size={}, total_blocks={}, eviction={:?})",
            self.kv_cache_config.mode,
            self.kv_cache_config.block_size,
            self.kv_cache_config.total_blocks,
            self.kv_cache_config.eviction_policy
        );
        if self.kv_admission_soft_limit_bytes.is_some()
            || self.kv_admission_hard_limit_bytes.is_some()
        {
            log::info!(
                "KV admission budgets: soft_limit_bytes={:?}, hard_limit_bytes={:?}",
                self.kv_admission_soft_limit_bytes,
                self.kv_admission_hard_limit_bytes
            );
        }

        // Attempt to detect model vocab size from declared outputs (logits) as a best-effort.
        // Typical logits output shape: [batch, seq_len, vocab_size]
        let mut detected_vocab: Option<usize> = None;
        let final_outputs = if let Some(stages) = pipeline_stages.as_ref() {
            stages
                .last()
                .map(|stage| stage.session.outputs())
                .unwrap_or_default()
        } else {
            session
                .as_ref()
                .map(|sess| sess.outputs())
                .unwrap_or_default()
        };

        for output in final_outputs {
            // prefer outputs named 'logits' or containing 'logit' substring
            let out_name = output.name().to_lowercase();
            if out_name == "logits" || out_name.contains("logit") || out_name.contains("output") {
                let out_shape: Vec<i64> = match output.dtype() {
                    ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
                    _ => vec![],
                };
                if out_shape.len() >= 3 {
                    let vocab_dim = out_shape[2];
                    if vocab_dim > 0 {
                        detected_vocab = Some(vocab_dim as usize);
                        break;
                    }
                }
            }
        }

        let resolved_vocab = detected_vocab.or(manifest_vocab_size);
        if let Some(vs) = resolved_vocab {
            if detected_vocab.is_some() {
                log::info!("Detected model vocab_size from outputs: {}", vs);
            } else {
                log::info!("Using manifest vocab_size hint: {}", vs);
            }
            self.vocab_size = Some(vs);
            // Pre-allocate logits workspace once so inference steps never need to grow the Vec.
            // This eliminates per-step reallocs for repetition penalty (extends vocab_size f32s).
            self.logits_workspace.reserve(vs);
        } else {
            log::info!("Could not detect model vocab_size; vocab_size remains unset");
            self.vocab_size = None;
        }
        self.bos_token_id = config_bos_token_id;

        // Persist captured names & shapes and session.
        self.model_input_names = Arc::new(input_names_set);
        self.model_input_shapes = Arc::new(input_shapes_map);
        self.model_input_types = Arc::new(input_types_map);
        self.decode_model_input_names = Arc::new(decode_input_names_set);
        self.decode_model_input_shapes = Arc::new(decode_input_shapes_map);
        self.decode_model_input_types = Arc::new(decode_input_types_map);
        self.empty_kv_cache.clear();
        self.pipeline_stages = pipeline_stages;
        self.session = session;
        self.decode_session = decode_session;
        self.update_kv_cache_metrics();

        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        (self.session.is_some() || self.pipeline_stages.is_some()) && self.tokenizer.is_some()
    }

    fn decode_model_input_names(&self) -> Arc<HashSet<String>> {
        if self.decode_session.is_some() {
            self.decode_model_input_names.clone()
        } else {
            self.model_input_names.clone()
        }
    }

    fn decode_model_input_shapes(&self) -> Arc<HashMap<String, Vec<i64>>> {
        if self.decode_session.is_some() {
            self.decode_model_input_shapes.clone()
        } else {
            self.model_input_shapes.clone()
        }
    }

    fn decode_model_input_types(&self) -> Arc<HashMap<String, TensorElementType>> {
        if self.decode_session.is_some() {
            self.decode_model_input_types.clone()
        } else {
            self.model_input_types.clone()
        }
    }

    fn update_kv_cache_metrics(&self) {
        let stats = self.kv_cache.stats();
        let mut metrics = self.metrics.lock().unwrap();
        metrics.kv_cache_bytes_used = stats.bytes_used;
        metrics.kv_cache_bytes_capacity = stats.bytes_capacity;
        metrics.kv_cache_blocks_total = stats.blocks_total;
        metrics.kv_cache_blocks_free = stats.blocks_free;
        metrics.kv_cache_sequences = stats.sequences;
        metrics.kv_cache_evicted_blocks = stats.evicted_blocks;
        metrics.kv_cache_evicted_sequences = stats.evicted_sequences;
        metrics.kv_cache_packed_layers = stats.packed_layers;
        metrics.kv_cache_cpu_offloaded_blocks = stats.cpu_offloaded_blocks;
    }

    fn kv_bytes_per_token(&self) -> Option<usize> {
        if self.num_layers == 0 || self.num_heads == 0 || self.head_dim == 0 {
            return None;
        }
        Some(
            self.num_layers
                .saturating_mul(self.num_heads)
                .saturating_mul(self.head_dim)
                .saturating_mul(2)
                .saturating_mul(std::mem::size_of::<f16>()),
        )
    }

    fn kv_admission_limits(&self, current_capacity_bytes: usize) -> (Option<usize>, Option<usize>) {
        let mut hard = self.kv_admission_hard_limit_bytes;
        if hard.is_none()
            && self.kv_cache_config.mode == KvCacheMode::Paged
            && current_capacity_bytes > 0
        {
            hard = Some(current_capacity_bytes);
        }

        let mut soft = self
            .kv_admission_soft_limit_bytes
            .or_else(|| hard.map(|hard_limit| hard_limit.saturating_mul(9) / 10));
        if let (Some(soft_limit), Some(hard_limit)) = (soft, hard) {
            soft = Some(soft_limit.min(hard_limit));
        }
        (soft, hard)
    }

    fn projected_kv_reservation_bytes(
        &self,
        prompt_tokens: usize,
        max_new_tokens: usize,
        has_live_sequence: bool,
    ) -> Option<usize> {
        match self.kv_cache_config.mode {
            KvCacheMode::Dense => {
                if has_live_sequence {
                    return Some(0);
                }
                Some(
                    self.num_layers
                        .saturating_mul(self.num_heads)
                        .saturating_mul(self.max_seq_len)
                        .saturating_mul(self.head_dim)
                        .saturating_mul(2)
                        .saturating_mul(std::mem::size_of::<f16>()),
                )
            }
            KvCacheMode::Paged => {
                let bytes_per_token = self.kv_bytes_per_token()?;
                Some(
                    prompt_tokens
                        .saturating_add(max_new_tokens)
                        .saturating_mul(bytes_per_token),
                )
            }
        }
    }

    fn enforce_kv_admission(
        &self,
        request_id: &str,
        session_id: Option<&str>,
        prompt_tokens: usize,
        max_new_tokens: usize,
        has_live_sequence: bool,
    ) -> Result<(), EngineError> {
        if !self.use_kv_cache {
            return Ok(());
        }

        let additional_reserved = match self.projected_kv_reservation_bytes(
            prompt_tokens,
            max_new_tokens,
            has_live_sequence,
        ) {
            Some(bytes) if bytes > 0 => bytes,
            _ => return Ok(()),
        };

        let stats = self.kv_cache.stats();
        let projected = stats.bytes_used.saturating_add(additional_reserved);
        let (soft_limit, hard_limit) = self.kv_admission_limits(stats.bytes_capacity);
        let session_suffix = session_id
            .map(|id| format!(" session='{}'", id))
            .unwrap_or_default();

        if let Some(limit) = hard_limit {
            if projected > limit {
                return Err(EngineError::resource_exhausted(format!(
                    "KV admission rejected for request '{}'{}: projected {} bytes exceeds hard limit {} (current used {}, additional reserved {})",
                    request_id,
                    session_suffix,
                    projected,
                    limit,
                    stats.bytes_used,
                    additional_reserved,
                )));
            }
        }

        if let Some(limit) = soft_limit {
            if projected > limit {
                return Err(EngineError::overloaded(format!(
                    "KV admission deferred for request '{}'{}: projected {} bytes exceeds soft limit {} (current used {}, additional reserved {})",
                    request_id,
                    session_suffix,
                    projected,
                    limit,
                    stats.bytes_used,
                    additional_reserved,
                )));
            }
        }

        Ok(())
    }

    pub async fn run_loop(&mut self) {
        loop {
            while let Ok(mut req) = self.request_rx.try_recv() {
                let session_id = req.session_id.clone();
                let (prompt, req_seq_arc) = {
                    let seq_arc = req.sequences.values().next().unwrap();
                    let seq = seq_arc.lock().unwrap();
                    (seq.prompt.clone(), seq_arc.clone())
                };

                let existing_seq_arc = session_id.as_ref().and_then(|session_key| {
                    let arc = self.sessions.get(session_key).cloned()?;
                    let cur_len = arc.lock().unwrap().get_len();
                    if cur_len >= self.max_session_tokens {
                        log::info!(
                            "Session '{}' reached max context ({} tokens); starting fresh.",
                            session_key,
                            cur_len
                        );
                        self.sessions.remove(session_key);
                        None
                    } else {
                        Some(arc)
                    }
                });

                let tokenizer = self.tokenizer.as_ref().unwrap();
                let encoded = tokenizer.encode(prompt, false).unwrap();
                let mut token_ids = encoded.get_ids().to_vec();
                if let Some(bos_id) = self.bos_token_id {
                    if existing_seq_arc.is_some() {
                        if token_ids.first().copied() == Some(bos_id) {
                            token_ids.remove(0);
                        }
                    } else if token_ids.first().copied() != Some(bos_id) {
                        token_ids.insert(0, bos_id);
                    }
                }

                // Fail-fast validation: ensure tokenizer ids are within detected model vocab_size (if known).
                if let Some(vocab) = self.vocab_size {
                    if let Some(&max_id) = token_ids.iter().max() {
                        if (max_id as usize) >= vocab {
                            log::error!("Tokenizer produced out-of-range token id {} >= model vocab_size {}. Rejecting request.", max_id, vocab);
                            // Notify client via response channel with helpful message (best-effort)
                            let _ = req.response_tx.try_send(SequenceGroupOutput {
                                request_id: req.request_id.clone(),
                                text: format!(
                                    "Tokenizer produced out-of-range token id {} (model vocab {})",
                                    max_id, vocab
                                ),
                                finish_reason: Some(FinishReason::Error),
                            });
                            // Skip scheduling this request
                            continue;
                        }
                    }
                }

                let prompt_len = token_ids.len();
                let max_new_tokens = req.sampling_params.max_tokens;
                if let Some(existing_seq_arc) = existing_seq_arc {
                    let existing_seq_id = existing_seq_arc.lock().unwrap().sequence_id;
                    let has_live_sequence =
                        self.use_kv_cache && self.kv_cache.has_sequence(existing_seq_id);
                    if let Err(engine_err) = self.enforce_kv_admission(
                        &req.request_id,
                        session_id.as_deref(),
                        prompt_len,
                        max_new_tokens,
                        has_live_sequence,
                    ) {
                        let group = Arc::new(Mutex::new(req));
                        self.fail_groups_with_error(&[group], &engine_err);
                        continue;
                    }

                    let (seq_id, new_len) = {
                        let mut seq = existing_seq_arc.lock().unwrap();
                        seq.output_token_ids.extend(token_ids.clone());
                        seq.generated_this_turn = 0;
                        seq.status = SequenceStatus::Waiting;
                        (seq.sequence_id, seq.get_len())
                    };
                    let base_seed = req
                        .sampling_params
                        .seed
                        .unwrap_or_else(|| Self::fnv1a_hash64(req.request_id.as_bytes()));
                    let rng_state =
                        Self::splitmix64(base_seed ^ seq_id.wrapping_mul(0x9e3779b97f4a7c15));
                    {
                        let mut seq = existing_seq_arc.lock().unwrap();
                        seq.rng_state = rng_state;
                    }

                    if self.use_kv_cache && !self.kv_cache.has_sequence(seq_id) {
                        // Pass token_ids for prefix matching
                        match self.kv_cache.allocate_sequence(seq_id, &token_ids) {
                            Ok(cached_len) => {
                                // If we reused KV cache, we should update the sequence state
                                // so that the next step skips these tokens.
                                let mut seq = existing_seq_arc.lock().unwrap();
                                seq.kv_cached_len = cached_len;
                            }
                            Err(err) => {
                                let engine_err = EngineError::resource_exhausted(err.to_string());
                                if let Some(session_key) = session_id.clone() {
                                    self.sessions.remove(&session_key);
                                }
                                let group = Arc::new(Mutex::new(req));
                                self.fail_groups_with_error(&[group], &engine_err);
                                continue;
                            }
                        }
                    }

                    req.sequences.clear();
                    req.sequences.insert(seq_id, existing_seq_arc);
                    req.reset_cache_for_single_seq(seq_id, new_len, SequenceStatus::Waiting);
                    self.scheduler.add_sequence_group(req);
                    continue;
                }

                if let Err(engine_err) = self.enforce_kv_admission(
                    &req.request_id,
                    session_id.as_deref(),
                    prompt_len,
                    max_new_tokens,
                    false,
                ) {
                    let group = Arc::new(Mutex::new(req));
                    self.fail_groups_with_error(&[group], &engine_err);
                    continue;
                }

                let seq_id = self.next_sequence_id.fetch_add(1, Ordering::SeqCst);
                let base_seed = req
                    .sampling_params
                    .seed
                    .unwrap_or_else(|| Self::fnv1a_hash64(req.request_id.as_bytes()));
                let rng_state =
                    Self::splitmix64(base_seed ^ seq_id.wrapping_mul(0x9e3779b97f4a7c15));
                {
                    let mut seq = req_seq_arc.lock().unwrap();
                    seq.sequence_id = seq_id;
                    seq.prompt_token_ids = token_ids.clone();
                    seq.output_token_ids.clear();
                    seq.generated_this_turn = 0;
                    seq.kv_cached_len = 0;
                    seq.rng_state = rng_state;
                    seq.status = SequenceStatus::Waiting;
                }

                if let Some(session_key) = session_id.as_ref() {
                    {
                        let mut seq = req_seq_arc.lock().unwrap();
                        seq.generated_this_turn = 0;
                        seq.output_token_ids.clear();
                    }
                    self.sessions
                        .insert(session_key.clone(), req_seq_arc.clone());
                }
                if self.use_kv_cache {
                    match self.kv_cache.allocate_sequence(seq_id, &token_ids) {
                        Ok(cached_len) => {
                            let mut seq = req_seq_arc.lock().unwrap();
                            seq.kv_cached_len = cached_len;
                        }
                        Err(err) => {
                            let engine_err = EngineError::resource_exhausted(err.to_string());
                            if let Some(session_key) = session_id.as_ref() {
                                self.sessions.remove(session_key);
                            }
                            let group = Arc::new(Mutex::new(req));
                            self.fail_groups_with_error(&[group], &engine_err);
                            continue;
                        }
                    }
                }

                // Move token_ids after using them
                {
                    let mut seq = req_seq_arc.lock().unwrap();
                    seq.prompt_token_ids = token_ids;
                }

                req.sequences.clear();
                req.sequences.insert(seq_id, req_seq_arc);
                req.reset_cache_for_single_seq(seq_id, prompt_len, SequenceStatus::Waiting);
                self.scheduler.add_sequence_group(req);
            }

            let active_ids = self.scheduler.active_sequence_ids();
            if self.use_kv_cache {
                self.kv_cache.set_active_sequences(&active_ids);
            }

            let outputs = self.scheduler.schedule();
            if outputs.scheduled_seq_groups.is_empty() {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                continue;
            }

            if let Err(e) = self.execute_step(&outputs.scheduled_seq_groups).await {
                log::error!("Execution error: {}", e);
                if self.try_recover_from_coreml_failure(&e).await {
                    log::warn!(
                        "Current batch failed due to provider error; future requests will run on CPU."
                    );
                }
                self.fail_groups_with_error(&outputs.scheduled_seq_groups, &e);
            }

            let finished_ids = self.scheduler.free_finished_sequences();
            if self.use_kv_cache {
                for seq_id in finished_ids {
                    self.kv_cache.remove_sequence(seq_id);
                }
                let evicted = self.kv_cache.drain_evicted_sequences();
                if !evicted.is_empty() {
                    let evicted_set: HashSet<u64> = evicted.into_iter().collect();
                    self.sessions.retain(|_, seq_arc| {
                        let seq = seq_arc.lock().unwrap();
                        !evicted_set.contains(&seq.sequence_id)
                    });
                }
                self.kv_cache.clear_active_sequences();
            }

            self.update_kv_cache_metrics();

            // Ensure streaming consumers can make progress even when decode loops are hot.
            tokio::task::yield_now().await;
        }
    }

    fn fail_groups_with_error(&self, groups: &[Arc<Mutex<SequenceGroup>>], err: &EngineError) {
        let err_text = format!("LLM execution error: {}", err);
        for group_arc in groups {
            let (request_id, response_tx, seqs) = {
                let group = group_arc.lock().unwrap();
                (
                    group.request_id.clone(),
                    group.response_tx.clone(),
                    group.sequences.values().cloned().collect::<Vec<_>>(),
                )
            };

            for seq_arc in seqs {
                let (old_status, new_status) = {
                    let mut seq = seq_arc.lock().unwrap();
                    let old_status = seq.status;
                    if !seq.is_finished() {
                        seq.status = SequenceStatus::Finished(FinishReason::Error);
                    }
                    (old_status, seq.status)
                };
                if old_status != new_status {
                    let mut group = group_arc.lock().unwrap();
                    group.update_seq_status(old_status, new_status);
                }
            }

            let _ = response_tx.try_send(SequenceGroupOutput {
                request_id,
                text: err_text.clone(),
                finish_reason: Some(FinishReason::Error),
            });
        }
    }

    fn cancel_group_if_needed(group_arc: &Arc<Mutex<SequenceGroup>>) -> bool {
        let cancellation = {
            let group = group_arc.lock().unwrap();
            group.cancellation.clone()
        };

        let Some(token) = cancellation else {
            return false;
        };
        if !token.is_cancelled() {
            return false;
        }

        let (request_id, response_tx, seqs, already_finished) = {
            let group = group_arc.lock().unwrap();
            (
                group.request_id.clone(),
                group.response_tx.clone(),
                group.sequences.values().cloned().collect::<Vec<_>>(),
                group.is_finished(),
            )
        };
        if already_finished {
            return true;
        }

        let mut status_updates = Vec::new();
        for seq_arc in &seqs {
            let (old_status, new_status) = {
                let mut seq = seq_arc.lock().unwrap();
                let old_status = seq.status;
                if !seq.is_finished() {
                    seq.status = SequenceStatus::Finished(FinishReason::Cancelled);
                }
                (old_status, seq.status)
            };
            if old_status != new_status {
                status_updates.push((old_status, new_status));
            }
        }

        if !status_updates.is_empty() {
            let mut group = group_arc.lock().unwrap();
            for (old_status, new_status) in status_updates {
                group.update_seq_status(old_status, new_status);
            }
        }

        // Wake any waiting response stream so it can observe cancellation promptly.
        let _ = response_tx.try_send(SequenceGroupOutput {
            request_id,
            text: String::new(),
            finish_reason: Some(FinishReason::Cancelled),
        });

        true
    }

    fn compute_input_tokens(seq: &Sequence, use_kv_cache: bool) -> Vec<u32> {
        if use_kv_cache {
            let prompt_len = seq.prompt_token_ids.len();
            let start = seq.kv_cached_len;
            if start < prompt_len {
                let mut tokens = seq.prompt_token_ids[start..].to_vec();
                tokens.extend_from_slice(&seq.output_token_ids);
                tokens
            } else {
                let start_out = start.saturating_sub(prompt_len);
                seq.output_token_ids[start_out..].to_vec()
            }
        } else {
            let mut tokens = seq.prompt_token_ids.clone();
            tokens.extend(seq.output_token_ids.iter().copied());
            tokens
        }
    }

    fn try_execute_batched_prefill_step(
        &mut self,
        groups: &[Arc<Mutex<SequenceGroup>>],
        model_input_names: &HashSet<String>,
        model_input_types: &HashMap<String, TensorElementType>,
    ) -> Result<bool, EngineError> {
        if !self.use_kv_cache {
            return Ok(false);
        }
        // Only the SeqFirst fast path is implemented here.
        if self.kv_layout != KvLayout::SeqFirst {
            return Ok(false);
        }

        let mut items = std::mem::take(&mut self.prefill_items_pool);
        items.clear();

        for group_arc in groups {
            if Self::cancel_group_if_needed(group_arc) {
                continue;
            }
            let seqs: Vec<_> = {
                let group = group_arc.lock().unwrap();
                group.sequences.values().cloned().collect()
            };
            for seq_arc in seqs {
                let (input_tokens, seq_id, total_len, kv_cached_len) = {
                    let seq = seq_arc.lock().unwrap();
                    if seq.is_finished() {
                        continue;
                    }
                    let total_len = seq.get_len();
                    let kv_cached_len = seq.kv_cached_len;
                    let input_tokens = Self::compute_input_tokens(&seq, true);
                    (input_tokens, seq.sequence_id, total_len, kv_cached_len)
                };
                // Only batch fresh prefill: full prompt, no prior KV
                if input_tokens.len() <= 1 || kv_cached_len != 0 {
                    items.clear();
                    self.prefill_items_pool = items;
                    return Ok(false);
                }
                items.push(PrefillItem {
                    group_arc: group_arc.clone(),
                    seq_arc,
                    seq_id,
                    input_len: input_tokens.len(),
                    total_len,
                    input_tokens,
                });
            }
        }

        if items.len() < 2 {
            items.clear();
            self.prefill_items_pool = items;
            return Ok(false);
        }

        // All must share the same prompt length for a rectangular batch.
        let input_len = items[0].input_len;
        if items.iter().any(|item| item.input_len != input_len) {
            items.clear();
            self.prefill_items_pool = items;
            return Ok(false);
        }

        let batch_size = items.len();

        // --- Build input tensors ---
        let input_ids_flat: Vec<i64> = items
            .iter()
            .flat_map(|item| item.input_tokens.iter().map(|&t| t as i64))
            .collect();
        let input_ids_i64 =
            Array2::from_shape_vec((batch_size, input_len), input_ids_flat).unwrap();
        // CoreML rejects zero-element KV tensors, so we feed a dummy past KV of length 1 (zeros).
        // The attention mask must cover past + present positions: prepend a 0 to mask out the dummy.
        let mask_len = 1 + input_len;
        let mask_data: Vec<i64> = (0..batch_size)
            .flat_map(|_| std::iter::once(0i64).chain(std::iter::repeat(1i64).take(input_len)))
            .collect();
        let mask_i64 = Array2::from_shape_vec((batch_size, mask_len), mask_data).unwrap();
        let pos_ids_flat: Vec<i64> = (0..batch_size).flat_map(|_| 0..input_len as i64).collect();
        let pos_ids_i64 = Array2::from_shape_vec((batch_size, input_len), pos_ids_flat).unwrap();

        if !model_input_names.contains("input_ids") {
            return Err(EngineError::backend(
                "Model input 'input_ids' not found".to_string(),
            ));
        }

        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(3 + self.num_layers * 2);

        let input_ids_type = model_input_types
            .get("input_ids")
            .copied()
            .unwrap_or(TensorElementType::Int64);
        let input_ids_value: DynValue = match input_ids_type {
            TensorElementType::Int64 => Value::from_array(input_ids_i64)
                .map_err(|e| EngineError::backend(e.to_string()))?
                .into_dyn(),
            TensorElementType::Int32 => {
                let arr = Array2::from_shape_vec(
                    (batch_size, input_len),
                    items
                        .iter()
                        .flat_map(|item| item.input_tokens.iter().map(|&t| t as i32))
                        .collect::<Vec<i32>>(),
                )
                .unwrap();
                Value::from_array(arr)
                    .map_err(|e| EngineError::backend(e.to_string()))?
                    .into_dyn()
            }
            other => {
                return Err(EngineError::backend(format!(
                    "Unsupported input_ids dtype: {:?}",
                    other
                )))
            }
        };
        inputs.push(("input_ids".into(), input_ids_value.into()));

        if model_input_names.contains("attention_mask") {
            let mask_type = model_input_types
                .get("attention_mask")
                .copied()
                .unwrap_or(TensorElementType::Int64);
            let mask_value: DynValue = match mask_type {
                TensorElementType::Int64 => Value::from_array(mask_i64)
                    .map_err(|e| EngineError::backend(e.to_string()))?
                    .into_dyn(),
                TensorElementType::Int32 => {
                    let data: Vec<i32> = (0..batch_size)
                        .flat_map(|_| {
                            std::iter::once(0i32).chain(std::iter::repeat(1i32).take(input_len))
                        })
                        .collect();
                    Value::from_array(Array2::from_shape_vec((batch_size, mask_len), data).unwrap())
                        .map_err(|e| EngineError::backend(e.to_string()))?
                        .into_dyn()
                }
                TensorElementType::Bool => {
                    let data: Vec<bool> = (0..batch_size)
                        .flat_map(|_| {
                            std::iter::once(false).chain(std::iter::repeat(true).take(input_len))
                        })
                        .collect();
                    Value::from_array(Array2::from_shape_vec((batch_size, mask_len), data).unwrap())
                        .map_err(|e| EngineError::backend(e.to_string()))?
                        .into_dyn()
                }
                TensorElementType::Float32 => {
                    let data: Vec<f32> = (0..batch_size)
                        .flat_map(|_| {
                            std::iter::once(0.0f32).chain(std::iter::repeat(1.0f32).take(input_len))
                        })
                        .collect();
                    Value::from_array(Array2::from_shape_vec((batch_size, mask_len), data).unwrap())
                        .map_err(|e| EngineError::backend(e.to_string()))?
                        .into_dyn()
                }
                other => {
                    return Err(EngineError::backend(format!(
                        "Unsupported attention_mask dtype: {:?}",
                        other
                    )))
                }
            };
            inputs.push(("attention_mask".into(), mask_value.into()));
        }

        if model_input_names.contains("position_ids") {
            let pos_type = model_input_types
                .get("position_ids")
                .copied()
                .unwrap_or(TensorElementType::Int64);
            let pos_value: DynValue = match pos_type {
                TensorElementType::Int64 => Value::from_array(pos_ids_i64)
                    .map_err(|e| EngineError::backend(e.to_string()))?
                    .into_dyn(),
                TensorElementType::Int32 => {
                    let pos_i32: Vec<i32> =
                        (0..batch_size).flat_map(|_| 0..input_len as i32).collect();
                    Value::from_array(
                        Array2::from_shape_vec((batch_size, input_len), pos_i32).unwrap(),
                    )
                    .map_err(|e| EngineError::backend(e.to_string()))?
                    .into_dyn()
                }
                other => {
                    return Err(EngineError::backend(format!(
                        "Unsupported position_ids dtype: {:?}",
                        other
                    )))
                }
            };
            inputs.push(("position_ids".into(), pos_value.into()));
        }

        // Past KV: all sequences are fresh (kv_cached_len == 0), so send empty tensors.
        // Shape [batch, num_heads, 0, head_dim] has zero elements — valid for any batch size.
        for layer in 0..self.num_layers {
            let key_name = format!("past_key_values.{}.key", layer);
            let val_name = format!("past_key_values.{}.value", layer);
            let needs_key = model_input_names.contains(&key_name);
            let needs_val = model_input_names.contains(&val_name);
            if !needs_key && !needs_val {
                continue;
            }
            let key_type = model_input_types
                .get(&key_name)
                .copied()
                .unwrap_or(TensorElementType::Float16);
            let val_type = model_input_types
                .get(&val_name)
                .copied()
                .unwrap_or(TensorElementType::Float16);
            // Dummy past KV of length 1 (zeros) — CoreML rejects zero-element tensors.
            // The attention mask already masks out this dummy position (leading 0).
            let empty_shape = (batch_size, self.num_heads, 1usize, self.head_dim);
            let kv_num_elems = batch_size * self.num_heads * self.head_dim;
            if needs_key {
                let k_input: SessionInputValue<'_> = match key_type {
                    TensorElementType::Float16 => Value::from_array(
                        Array4::<f16>::from_shape_vec(empty_shape, vec![f16::ZERO; kv_num_elems])
                            .map_err(|e| EngineError::backend(e.to_string()))?,
                    )
                    .map_err(|e| EngineError::backend(e.to_string()))?
                    .into(),
                    TensorElementType::Float32 => Value::from_array(
                        Array4::<f32>::from_shape_vec(empty_shape, vec![0.0f32; kv_num_elems])
                            .map_err(|e| EngineError::backend(e.to_string()))?,
                    )
                    .map_err(|e| EngineError::backend(e.to_string()))?
                    .into(),
                    other => {
                        return Err(EngineError::backend(format!(
                            "Unsupported KV dtype for {}: {:?}",
                            key_name, other
                        )))
                    }
                };
                inputs.push((key_name.clone().into(), k_input));
            }
            if needs_val {
                let v_input: SessionInputValue<'_> = match val_type {
                    TensorElementType::Float16 => Value::from_array(
                        Array4::<f16>::from_shape_vec(empty_shape, vec![f16::ZERO; kv_num_elems])
                            .map_err(|e| EngineError::backend(e.to_string()))?,
                    )
                    .map_err(|e| EngineError::backend(e.to_string()))?
                    .into(),
                    TensorElementType::Float32 => Value::from_array(
                        Array4::<f32>::from_shape_vec(empty_shape, vec![0.0f32; kv_num_elems])
                            .map_err(|e| EngineError::backend(e.to_string()))?,
                    )
                    .map_err(|e| EngineError::backend(e.to_string()))?
                    .into(),
                    other => {
                        return Err(EngineError::backend(format!(
                            "Unsupported KV dtype for {}: {:?}",
                            val_name, other
                        )))
                    }
                };
                inputs.push((val_name.clone().into(), v_input));
            }
        }

        let outputs = self
            .session
            .as_mut()
            .ok_or(EngineError::ModelNotLoaded)?
            .run(inputs)
            .map_err(|e| EngineError::backend(e.to_string()))?;

        // === Store present KV for each sequence ===
        // Expected present shape: [batch, num_heads, prompt_len, head_dim] (SeqFirst)
        for layer in 0..self.num_layers {
            let key_name = format!("present.{}.key", layer);
            let val_name = format!("present.{}.value", layer);
            let (Some(k_out), Some(v_out)) = (outputs.get(&key_name), outputs.get(&val_name))
            else {
                continue;
            };

            let (k_shape, k_data) = match k_out.try_extract_tensor::<f16>() {
                Ok((shape, data)) => (shape.to_vec(), data.to_vec()),
                Err(_) => {
                    let (shape, data_f32) =
                        k_out.try_extract_tensor::<f32>().map_err(|f32_err| {
                            EngineError::backend(format!(
                                "Failed to extract present key for prefill layer {}: {:?}",
                                layer, f32_err
                            ))
                        })?;
                    (
                        shape.to_vec(),
                        data_f32.iter().map(|v| f16::from_f32(*v)).collect(),
                    )
                }
            };
            let (_v_shape, v_data) = match v_out.try_extract_tensor::<f16>() {
                Ok((shape, data)) => (shape.to_vec(), data.to_vec()),
                Err(_) => {
                    let (shape, data_f32) =
                        v_out.try_extract_tensor::<f32>().map_err(|f32_err| {
                            EngineError::backend(format!(
                                "Failed to extract present value for prefill layer {}: {:?}",
                                layer, f32_err
                            ))
                        })?;
                    (
                        shape.to_vec(),
                        data_f32.iter().map(|v| f16::from_f32(*v)).collect(),
                    )
                }
            };

            if k_shape.len() < 4 {
                log::warn!(
                    "Batched prefill: unexpected present KV rank {} for layer {}",
                    k_shape.len(),
                    layer
                );
                continue;
            }
            let k_out_heads = k_shape[1] as usize;
            let k_out_seq = k_shape[2] as usize;
            let k_out_dim = k_shape[3] as usize;

            for (batch_idx, item) in items.iter().enumerate() {
                // k_out_seq = 1 + input_len (dummy past position 0 + actual tokens 1..input_len+1).
                // Skip position 0 (dummy) and store only the actual input_len positions at cache pos 0.
                let actual_seq = k_out_seq.saturating_sub(1); // = input_len
                let skip_elems = k_out_dim; // one dummy position per head to skip
                let packed_len = self.num_heads * actual_seq * k_out_dim;
                let mut packed_key = Vec::with_capacity(packed_len);
                let mut packed_val = Vec::with_capacity(packed_len);
                for h in 0..self.num_heads {
                    let head_start = (batch_idx * k_out_heads + h) * k_out_seq * k_out_dim;
                    let actual_start = head_start + skip_elems;
                    let actual_end = head_start + k_out_seq * k_out_dim;
                    if actual_end > k_data.len() || actual_end > v_data.len() {
                        log::warn!(
                            "Batched prefill: KV out-of-bounds layer {} batch {} head {}",
                            layer,
                            batch_idx,
                            h
                        );
                        continue;
                    }
                    if actual_seq == 0 {
                        continue;
                    }
                    packed_key.extend_from_slice(&k_data[actual_start..actual_end]);
                    packed_val.extend_from_slice(&v_data[actual_start..actual_end]);
                    self.kv_cache
                        .append_head_range_seq_first(
                            item.seq_id,
                            layer,
                            h,
                            0,
                            &k_data[actual_start..actual_end],
                            &v_data[actual_start..actual_end],
                        )
                        .map_err(|e| EngineError::resource_exhausted(e.to_string()))?;
                }
                if actual_seq > 0
                    && packed_key.len() == packed_len
                    && packed_val.len() == packed_len
                {
                    self.kv_cache.set_packed_layer(
                        item.seq_id,
                        layer,
                        actual_seq,
                        &packed_key,
                        &packed_val,
                    );
                } else {
                    self.kv_cache.clear_packed_layer(item.seq_id, layer);
                }
            }
        }

        // Advance KV cursors.
        for item in &items {
            if let Some(current_len) = self.kv_cache.sequence_length(item.seq_id) {
                if item.total_len > current_len {
                    self.kv_cache
                        .advance_sequence_by(item.seq_id, item.total_len - current_len);
                }
            } else {
                self.kv_cache
                    .advance_sequence_by(item.seq_id, item.total_len);
            }
        }

        // === Logits → sample next token ===
        let (logits_shape, logits_data) = {
            let logits = outputs
                .get("logits")
                .ok_or_else(|| EngineError::backend("Missing logits output".to_string()))?;
            match logits.try_extract_tensor::<f32>() {
                Ok((shape, data)) => (shape.to_vec(), data.to_vec()),
                Err(_) => {
                    let (shape, data_f16) =
                        logits.try_extract_tensor::<f16>().map_err(|f16_err| {
                            EngineError::backend(format!(
                                "Failed to extract logits as f32 or f16: {:?}",
                                f16_err
                            ))
                        })?;
                    (
                        shape.to_vec(),
                        data_f16.iter().map(|v| v.to_f32()).collect(),
                    )
                }
            }
        };
        drop(outputs);

        let vocab = logits_shape
            .last()
            .copied()
            .filter(|v| *v > 0)
            .map(|v| v as usize)
            .or(self.vocab_size)
            .ok_or_else(|| EngineError::backend("Unable to infer vocab size".to_string()))?;
        if vocab == 0 || logits_data.len() < vocab {
            return Err(EngineError::backend(
                "Invalid logits output length".to_string(),
            ));
        }

        let seq_len = if logits_shape.len() >= 2 {
            logits_shape
                .get(logits_shape.len() - 2)
                .copied()
                .filter(|v| *v > 0)
                .map(|v| v as usize)
                .unwrap_or(1)
        } else {
            1
        };
        let rows = logits_data.len() / vocab;
        if rows < batch_size {
            return Err(EngineError::backend(format!(
                "Unexpected logits rows={} < batch_size={}",
                rows, batch_size
            )));
        }
        let rows_per_batch = (rows / batch_size).max(1);

        for (batch_idx, item) in items.iter().enumerate() {
            // Take the last position's logits for each batch element.
            let row = batch_idx * rows_per_batch + (seq_len.min(rows_per_batch).saturating_sub(1));
            let start = row * vocab;
            let end = start + vocab;
            if end > logits_data.len() {
                return Err(EngineError::backend(format!(
                    "Logits slice out of bounds for batch {}",
                    batch_idx
                )));
            }

            let (request_id, response_tx, sampling_params) = {
                let group = item.group_arc.lock().unwrap();
                (
                    group.request_id.clone(),
                    group.response_tx.clone(),
                    group.sampling_params.clone(),
                )
            };

            let (token_ids_for_penalty, mut rng_state) = {
                let seq = item.seq_arc.lock().unwrap();
                let token_ids = (sampling_params.repetition_penalty > 1.0).then(|| {
                    seq.prompt_token_ids
                        .iter()
                        .chain(seq.output_token_ids.iter())
                        .copied()
                        .collect::<Vec<u32>>()
                });
                (token_ids, seq.rng_state)
            };
            // Only clone into workspace when repetition penalty is active; otherwise
            // sample directly from logits_data to avoid a ~600 KB/vocab copy per step.
            let next = if let Some(token_ids) = token_ids_for_penalty {
                let penalty = sampling_params.repetition_penalty;
                self.logits_workspace.clear();
                self.logits_workspace
                    .extend_from_slice(&logits_data[start..end]);
                for token_id in token_ids {
                    let idx = token_id as usize;
                    if idx < self.logits_workspace.len() {
                        let val = self.logits_workspace[idx];
                        self.logits_workspace[idx] = if val > 0.0 {
                            val / penalty
                        } else {
                            val * penalty
                        };
                    }
                }
                Self::sample_next_token(&self.logits_workspace, &sampling_params, &mut rng_state)
            } else {
                Self::sample_next_token(&logits_data[start..end], &sampling_params, &mut rng_state)
            };
            let mut finish_reason = None;
            let (old_len, old_status, new_len, new_status, text) = {
                let mut seq = item.seq_arc.lock().unwrap();
                let old_len = seq.get_len();
                let old_status = seq.status;
                seq.kv_cached_len = seq.kv_cached_len.saturating_add(item.input_len);
                seq.rng_state = rng_state;
                seq.append_token_id(next, 0.0);
                if sampling_params.stop_token_ids.contains(&next) {
                    seq.status = SequenceStatus::Finished(FinishReason::Stop);
                    finish_reason = Some(FinishReason::Stop);
                } else if seq.generated_this_turn >= sampling_params.max_tokens {
                    seq.status = SequenceStatus::Finished(FinishReason::Length);
                    finish_reason = Some(FinishReason::Length);
                }
                let text = self
                    .tokenizer
                    .as_ref()
                    .unwrap()
                    .decode(&[next], true)
                    .unwrap_or_default();
                (old_len, old_status, seq.get_len(), seq.status, text)
            };

            if old_len != new_len || old_status != new_status {
                let mut group = item.group_arc.lock().unwrap();
                if old_len != new_len {
                    group.update_seq_len(item.seq_id, new_len);
                }
                if old_status != new_status {
                    group.update_seq_status(old_status, new_status);
                }
            }

            let _ = response_tx.try_send(SequenceGroupOutput {
                request_id,
                text,
                finish_reason,
            });
        }

        items.clear();
        self.prefill_items_pool = items;
        Ok(true)
    }

    fn try_execute_batched_decode_step(
        &mut self,
        groups: &[Arc<Mutex<SequenceGroup>>],
        model_input_names: &HashSet<String>,
        model_input_types: &HashMap<String, TensorElementType>,
    ) -> Result<bool, EngineError> {
        if !self.use_kv_cache {
            return Ok(false);
        }

        let mut items = std::mem::take(&mut self.decode_items_pool);
        items.clear();

        for group_arc in groups {
            if Self::cancel_group_if_needed(group_arc) {
                continue;
            }
            let seqs: Vec<_> = {
                let group = group_arc.lock().unwrap();
                group.sequences.values().cloned().collect()
            };
            for seq_arc in seqs {
                let (input_tokens, seq_id, total_len) = {
                    let seq = seq_arc.lock().unwrap();
                    if seq.is_finished() {
                        continue;
                    }
                    (
                        Self::compute_input_tokens(&seq, self.use_kv_cache),
                        seq.sequence_id,
                        seq.get_len(),
                    )
                };
                if input_tokens.len() != 1 || total_len == 0 {
                    items.clear();
                    self.decode_items_pool = items;
                    return Ok(false);
                }
                items.push(BatchItem {
                    group_arc: group_arc.clone(),
                    seq_arc,
                    seq_id,
                    input_token: input_tokens[0],
                    total_len,
                });
            }
        }

        if items.is_empty() {
            items.clear();
            self.decode_items_pool = items;
            return Ok(false);
        }

        let actual_max_total_len = items.iter().map(|item| item.total_len).max().unwrap_or(0);
        if actual_max_total_len <= 1 {
            items.clear();
            self.decode_items_pool = items;
            return Ok(false);
        }

        let decode_bucket_granularity = llm_decode_bucket_granularity();
        let max_total_len =
            round_up_to_granularity(actual_max_total_len, decode_bucket_granularity)
                .max(actual_max_total_len);
        let max_past_len = max_total_len - 1;
        let batch_size = items.len();
        let profile_decode = llm_decode_profile_enabled();
        let step_started = profile_decode.then(Instant::now);
        let input_prep_started = profile_decode.then(Instant::now);
        let input_tokens_i64 = items
            .iter()
            .map(|item| item.input_token as i64)
            .collect::<Vec<i64>>();
        let attention_mask_i64 = items
            .iter()
            .flat_map(|item| {
                let pad_len = max_total_len.saturating_sub(item.total_len);
                std::iter::repeat_n(0i64, pad_len).chain(std::iter::repeat_n(1i64, item.total_len))
            })
            .collect::<Vec<i64>>();
        let position_ids_i64 = items
            .iter()
            .map(|item| item.total_len.saturating_sub(1) as i64)
            .collect::<Vec<i64>>();

        let input_ids_i64 =
            Array2::from_shape_vec((batch_size, 1), input_tokens_i64.clone()).unwrap();
        let mask_i64 =
            Array2::from_shape_vec((batch_size, max_total_len), attention_mask_i64.clone())
                .unwrap();
        let pos_ids_i64 =
            Array2::from_shape_vec((batch_size, 1), position_ids_i64.clone()).unwrap();

        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(3 + (self.num_layers * 2));

        let input_ids_type = model_input_types
            .get("input_ids")
            .copied()
            .unwrap_or(TensorElementType::Int64);
        let input_ids_value: DynValue = match input_ids_type {
            TensorElementType::Int64 => Value::from_array(input_ids_i64)
                .map_err(|e| EngineError::backend(e.to_string()))?
                .into_dyn(),
            TensorElementType::Int32 => {
                let input_ids_i32 = Array2::from_shape_vec(
                    (batch_size, 1),
                    input_tokens_i64
                        .iter()
                        .map(|token| *token as i32)
                        .collect::<Vec<i32>>(),
                )
                .unwrap();
                Value::from_array(input_ids_i32)
                    .map_err(|e| EngineError::backend(e.to_string()))?
                    .into_dyn()
            }
            other => {
                return Err(EngineError::backend(format!(
                    "Unsupported input_ids dtype: {:?}",
                    other
                )))
            }
        };
        inputs.push(("input_ids".into(), input_ids_value.into()));

        if model_input_names.contains("attention_mask") {
            let mask_type = model_input_types
                .get("attention_mask")
                .copied()
                .unwrap_or(TensorElementType::Int64);
            let mask_value: DynValue = match mask_type {
                TensorElementType::Int64 => Value::from_array(mask_i64)
                    .map_err(|e| EngineError::backend(e.to_string()))?
                    .into_dyn(),
                TensorElementType::Int32 => {
                    let mask_i32 = Array2::from_shape_vec(
                        (batch_size, max_total_len),
                        attention_mask_i64
                            .iter()
                            .map(|value| *value as i32)
                            .collect::<Vec<i32>>(),
                    )
                    .unwrap();
                    Value::from_array(mask_i32)
                        .map_err(|e| EngineError::backend(e.to_string()))?
                        .into_dyn()
                }
                TensorElementType::Bool => {
                    let mask_bool = Array2::from_shape_vec(
                        (batch_size, max_total_len),
                        attention_mask_i64
                            .iter()
                            .map(|value| *value != 0)
                            .collect::<Vec<bool>>(),
                    )
                    .unwrap();
                    Value::from_array(mask_bool)
                        .map_err(|e| EngineError::backend(e.to_string()))?
                        .into_dyn()
                }
                TensorElementType::Float32 => {
                    let mask_f32 = Array2::from_shape_vec(
                        (batch_size, max_total_len),
                        attention_mask_i64
                            .iter()
                            .map(|value| *value as f32)
                            .collect::<Vec<f32>>(),
                    )
                    .unwrap();
                    Value::from_array(mask_f32)
                        .map_err(|e| EngineError::backend(e.to_string()))?
                        .into_dyn()
                }
                other => {
                    return Err(EngineError::backend(format!(
                        "Unsupported attention_mask dtype: {:?}",
                        other
                    )))
                }
            };
            inputs.push(("attention_mask".into(), mask_value.into()));
        }

        if model_input_names.contains("position_ids") {
            let pos_type = model_input_types
                .get("position_ids")
                .copied()
                .unwrap_or(TensorElementType::Int64);
            let pos_value: DynValue = match pos_type {
                TensorElementType::Int64 => Value::from_array(pos_ids_i64)
                    .map_err(|e| EngineError::backend(e.to_string()))?
                    .into_dyn(),
                TensorElementType::Int32 => {
                    let pos_i32 = Array2::from_shape_vec(
                        (batch_size, 1),
                        position_ids_i64
                            .iter()
                            .map(|pos| *pos as i32)
                            .collect::<Vec<i32>>(),
                    )
                    .unwrap();
                    Value::from_array(pos_i32)
                        .map_err(|e| EngineError::backend(e.to_string()))?
                        .into_dyn()
                }
                other => {
                    return Err(EngineError::backend(format!(
                        "Unsupported position_ids dtype: {:?}",
                        other
                    )))
                }
            };
            inputs.push(("position_ids".into(), pos_value.into()));
        }

        let input_prep_ms = input_prep_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let kv_pack_started = profile_decode.then(Instant::now);

        for layer in 0..self.num_layers {
            let key_name = format!("past_key_values.{}.key", layer);
            let val_name = format!("past_key_values.{}.value", layer);
            let needs_key = model_input_names.contains(&key_name);
            let needs_val = model_input_names.contains(&val_name);
            if !needs_key && !needs_val {
                continue;
            }

            let key_type = model_input_types
                .get(&key_name)
                .copied()
                .unwrap_or(TensorElementType::Float16);
            let val_type = model_input_types
                .get(&val_name)
                .copied()
                .unwrap_or(TensorElementType::Float16);

            let seq_first_size = batch_size * self.num_heads * max_past_len * self.head_dim;
            // Reuse workspace buffers to avoid per-layer heap allocation.
            self.kv_workspace_key.clear();
            self.kv_workspace_key.resize(seq_first_size, f16::ZERO);
            self.kv_workspace_val.clear();
            self.kv_workspace_val.resize(seq_first_size, f16::ZERO);
            let key_seq_first = &mut self.kv_workspace_key;
            let val_seq_first = &mut self.kv_workspace_val;

            for (batch_idx, item) in items.iter().enumerate() {
                let past_len = item.total_len.saturating_sub(1);
                let pad_len = max_past_len.saturating_sub(past_len);

                if let Some(packed) = self.kv_cache.get_packed_layer(item.seq_id, layer) {
                    if packed.length < past_len {
                        return Ok(false);
                    }
                    let packed_head_stride = packed.length * self.head_dim;
                    let copy_len = past_len * self.head_dim;
                    for h in 0..self.num_heads {
                        let src_start = h * packed_head_stride;
                        let dst_start = ((batch_idx * self.num_heads + h) * max_past_len + pad_len)
                            * self.head_dim;
                        key_seq_first[dst_start..dst_start + copy_len]
                            .copy_from_slice(&packed.key[src_start..src_start + copy_len]);
                        val_seq_first[dst_start..dst_start + copy_len]
                            .copy_from_slice(&packed.value[src_start..src_start + copy_len]);
                    }
                    continue;
                }

                let view = match self.kv_cache.get_layer_view(item.seq_id, layer) {
                    Some(view) => view,
                    None => return Ok(false),
                };
                if view.length < past_len {
                    return Ok(false);
                }

                let stride = self.num_heads * self.head_dim;
                if stride == 0
                    || !view.key.len().is_multiple_of(stride)
                    || !view.value.len().is_multiple_of(stride)
                {
                    return Ok(false);
                }
                let max_seq_len = view.key.len() / stride;
                if past_len > max_seq_len {
                    return Ok(false);
                }

                let copy_len = past_len * self.head_dim;
                for h in 0..self.num_heads {
                    let src_start = h * max_seq_len * self.head_dim;
                    let dst_start =
                        ((batch_idx * self.num_heads + h) * max_past_len + pad_len) * self.head_dim;
                    key_seq_first[dst_start..dst_start + copy_len]
                        .copy_from_slice(&view.key[src_start..src_start + copy_len]);
                    val_seq_first[dst_start..dst_start + copy_len]
                        .copy_from_slice(&view.value[src_start..src_start + copy_len]);
                }
            }

            let reorder_to_head_dim_first = |src: &[f16]| -> Vec<f16> {
                let mut dst = vec![f16::ZERO; src.len()];
                for b in 0..batch_size {
                    for h in 0..self.num_heads {
                        for pos in 0..max_past_len {
                            for d in 0..self.head_dim {
                                let src_idx = (((b * self.num_heads + h) * max_past_len + pos)
                                    * self.head_dim)
                                    + d;
                                let dst_idx = (((b * self.num_heads + h) * self.head_dim + d)
                                    * max_past_len)
                                    + pos;
                                dst[dst_idx] = src[src_idx];
                            }
                        }
                    }
                }
                dst
            };

            let (key_data, val_data, shape): (Vec<f16>, Vec<f16>, (usize, usize, usize, usize)) =
                match self.kv_layout {
                    // Clone from workspace — workspace retains capacity for next layer.
                    KvLayout::SeqFirst => (
                        key_seq_first.clone(),
                        val_seq_first.clone(),
                        (batch_size, self.num_heads, max_past_len, self.head_dim),
                    ),
                    KvLayout::HeadDimFirst => (
                        reorder_to_head_dim_first(key_seq_first),
                        reorder_to_head_dim_first(val_seq_first),
                        (batch_size, self.num_heads, self.head_dim, max_past_len),
                    ),
                };

            if needs_key {
                // key_data is moved into Array4 — no extra clone.
                let key_input: SessionInputValue<'_> = match key_type {
                    TensorElementType::Float16 => {
                        Value::from_array(Array4::from_shape_vec(shape, key_data).map_err(|e| {
                            EngineError::backend(format!("KV key shape error: {e}"))
                        })?)
                        .map_err(|e| EngineError::backend(e.to_string()))?
                        .into()
                    }
                    TensorElementType::Float32 => {
                        let key_f32: Vec<f32> = key_data.iter().map(|v| v.to_f32()).collect();
                        Value::from_array(Array4::from_shape_vec(shape, key_f32).map_err(|e| {
                            EngineError::backend(format!("KV key shape error: {e}"))
                        })?)
                        .map_err(|e| EngineError::backend(e.to_string()))?
                        .into()
                    }
                    other => {
                        return Err(EngineError::backend(format!(
                            "Unsupported KV key dtype for {}: {:?}",
                            key_name, other
                        )))
                    }
                };
                inputs.push((key_name.clone().into(), key_input));
            }
            if needs_val {
                // val_data is moved into Array4 — no extra clone.
                let val_input: SessionInputValue<'_> = match val_type {
                    TensorElementType::Float16 => {
                        Value::from_array(Array4::from_shape_vec(shape, val_data).map_err(|e| {
                            EngineError::backend(format!("KV value shape error: {e}"))
                        })?)
                        .map_err(|e| EngineError::backend(e.to_string()))?
                        .into()
                    }
                    TensorElementType::Float32 => {
                        let val_f32: Vec<f32> = val_data.iter().map(|v| v.to_f32()).collect();
                        Value::from_array(Array4::from_shape_vec(shape, val_f32).map_err(|e| {
                            EngineError::backend(format!("KV value shape error: {e}"))
                        })?)
                        .map_err(|e| EngineError::backend(e.to_string()))?
                        .into()
                    }
                    other => {
                        return Err(EngineError::backend(format!(
                            "Unsupported KV value dtype for {}: {:?}",
                            val_name, other
                        )))
                    }
                };
                inputs.push((val_name.clone().into(), val_input));
            }
        }

        let kv_pack_ms = kv_pack_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let ort_started = profile_decode.then(Instant::now);
        let outputs = if self.decode_session.is_some() {
            self.decode_session
                .as_mut()
                .unwrap()
                .run(inputs)
                .map_err(|e| EngineError::backend(e.to_string()))?
        } else {
            self.session
                .as_mut()
                .ok_or(EngineError::ModelNotLoaded)?
                .run(inputs)
                .map_err(|e| EngineError::backend(e.to_string()))?
        };
        let ort_run_ms = ort_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let kv_writeback_started = profile_decode.then(Instant::now);
        let mut first_present_shape: Option<Vec<usize>> = None;

        for layer in 0..self.num_layers {
            let key_name = format!("present.{}.key", layer);
            let val_name = format!("present.{}.value", layer);
            let (Some(k_out), Some(v_out)) = (outputs.get(&key_name), outputs.get(&val_name))
            else {
                continue;
            };

            let (k_shape, k_data) = match k_out.try_extract_tensor::<f16>() {
                Ok((shape, data)) => (shape.to_vec(), data.to_vec()),
                Err(f16_err) => {
                    let (shape, data_f32) =
                        k_out.try_extract_tensor::<f32>().map_err(|f32_err| {
                            EngineError::backend(format!(
                                "Failed to extract present key tensor as f16 ({:?}) or f32 ({:?})",
                                f16_err, f32_err
                            ))
                        })?;
                    (
                        shape.to_vec(),
                        data_f32.iter().map(|v| f16::from_f32(*v)).collect(),
                    )
                }
            };
            let (v_shape, v_data) = match v_out.try_extract_tensor::<f16>() {
                Ok((shape, data)) => (shape.to_vec(), data.to_vec()),
                Err(f16_err) => {
                    let (shape, data_f32) =
                        v_out.try_extract_tensor::<f32>().map_err(|f32_err| {
                            EngineError::backend(format!(
                                "Failed to extract present value tensor as f16 ({:?}) or f32 ({:?})",
                                f16_err, f32_err
                            ))
                        })?;
                    (
                        shape.to_vec(),
                        data_f32.iter().map(|v| f16::from_f32(*v)).collect(),
                    )
                }
            };

            if first_present_shape.is_none() {
                first_present_shape = Some(k_shape.iter().map(|&dim| dim as usize).collect());
            }
            if k_shape.len() < 4 || v_shape.len() < 4 {
                return Err(EngineError::backend(format!(
                    "Unsupported present KV rank for layer {}: key={:?} value={:?}",
                    layer, k_shape, v_shape
                )));
            }

            let (k_batch, k_heads, k_seq, k_head_dim) = match self.kv_layout {
                KvLayout::SeqFirst => (
                    k_shape[0].max(1) as usize,
                    k_shape[1].max(1) as usize,
                    k_shape[2].max(1) as usize,
                    k_shape[3].max(1) as usize,
                ),
                KvLayout::HeadDimFirst => (
                    k_shape[0].max(1) as usize,
                    k_shape[1].max(1) as usize,
                    k_shape[3].max(1) as usize,
                    k_shape[2].max(1) as usize,
                ),
            };
            if k_batch < batch_size || k_heads < self.num_heads || k_head_dim < self.head_dim {
                return Err(EngineError::backend(format!(
                    "Present KV shape too small for batch decode on layer {}: {:?}",
                    layer, k_shape
                )));
            }

            for (batch_idx, item) in items.iter().enumerate() {
                let past_len = item.total_len.saturating_sub(1);
                let pad_len = max_past_len.saturating_sub(past_len);
                let abs_pos = past_len;
                let output_pos = if k_seq == max_total_len {
                    let pos = pad_len + past_len;
                    if pos >= k_seq {
                        return Err(EngineError::backend(format!(
                            "Present KV sequence dim too small on layer {}: output_pos={} seq_dim={}",
                            layer, pos, k_seq
                        )));
                    }
                    pos
                } else if k_seq == 1 {
                    0
                } else {
                    return Err(EngineError::backend(format!(
                        "Unsupported batched decode present seq dim on layer {}: seq_dim={} expected {} (full history) or 1 (delta-only)",
                        layer, k_seq, max_total_len
                    )));
                };

                let mut key_token = vec![f16::ZERO; self.num_heads * self.head_dim];
                let mut val_token = vec![f16::ZERO; self.num_heads * self.head_dim];
                for h in 0..self.num_heads {
                    for d in 0..self.head_dim {
                        let key_idx = match self.kv_layout {
                            KvLayout::SeqFirst => {
                                (((batch_idx * k_heads + h) * k_seq + output_pos) * k_head_dim) + d
                            }
                            KvLayout::HeadDimFirst => {
                                (((batch_idx * k_heads + h) * k_head_dim + d) * k_seq) + output_pos
                            }
                        };
                        let val_idx = key_idx;
                        if key_idx >= k_data.len() || val_idx >= v_data.len() {
                            return Err(EngineError::backend(format!(
                                "Present KV index out of bounds on layer {} (key_idx={}, val_idx={})",
                                layer, key_idx, val_idx
                            )));
                        }
                        key_token[h * self.head_dim + d] = k_data[key_idx];
                        val_token[h * self.head_dim + d] = v_data[val_idx];
                    }
                }

                self.kv_cache
                    .append_token(item.seq_id, layer, abs_pos, &key_token, &val_token, None)
                    .map_err(|e| EngineError::resource_exhausted(e.to_string()))?;
                let can_store_packed = k_seq == max_total_len
                    && batch_size == 1
                    && self.kv_layout == KvLayout::SeqFirst
                    && item.total_len == max_total_len;
                if can_store_packed {
                    self.kv_cache
                        .set_packed_layer(item.seq_id, layer, k_seq, &k_data, &v_data);
                } else {
                    self.kv_cache.clear_packed_layer(item.seq_id, layer);
                }
            }
        }

        for item in &items {
            if let Some(current_len) = self.kv_cache.sequence_length(item.seq_id) {
                if item.total_len > current_len {
                    self.kv_cache
                        .advance_sequence_by(item.seq_id, item.total_len - current_len);
                }
            } else {
                self.kv_cache
                    .advance_sequence_by(item.seq_id, item.total_len);
            }
        }

        let kv_writeback_ms = kv_writeback_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let sampling_started = profile_decode.then(Instant::now);
        let (logits_shape, logits_data) = {
            let logits = outputs
                .get("logits")
                .ok_or_else(|| EngineError::backend("Missing logits output".to_string()))?;
            match logits.try_extract_tensor::<f32>() {
                Ok((shape, data)) => (shape.to_vec(), data.to_vec()),
                Err(f32_err) => {
                    let (shape, data_f16) =
                        logits.try_extract_tensor::<f16>().map_err(|f16_err| {
                            EngineError::backend(format!(
                                "Failed to extract logits as f32 ({:?}) or f16 ({:?})",
                                f32_err, f16_err
                            ))
                        })?;
                    (
                        shape.to_vec(),
                        data_f16.iter().map(|v| v.to_f32()).collect(),
                    )
                }
            }
        };
        drop(outputs);

        let vocab = logits_shape
            .last()
            .copied()
            .filter(|v| *v > 0)
            .map(|v| v as usize)
            .or(self.vocab_size)
            .ok_or_else(|| EngineError::backend("Unable to infer logits vocab size".to_string()))?;
        if vocab == 0 || logits_data.len() < vocab {
            return Err(EngineError::backend(
                "Invalid logits output length".to_string(),
            ));
        }

        let seq_len = if logits_shape.len() >= 2 {
            logits_shape
                .get(logits_shape.len() - 2)
                .copied()
                .filter(|v| *v > 0)
                .map(|v| v as usize)
                .unwrap_or(1)
        } else {
            1
        };
        let rows = logits_data.len() / vocab;
        if rows == 0 || rows < batch_size {
            return Err(EngineError::backend(format!(
                "Unexpected logits row count: rows={} batch_size={}",
                rows, batch_size
            )));
        }
        let rows_per_batch = (rows / batch_size).max(1);

        for (batch_idx, item) in items.iter().enumerate() {
            let row = batch_idx * rows_per_batch + (seq_len.min(rows_per_batch).saturating_sub(1));
            let start = row * vocab;
            let end = start + vocab;
            if end > logits_data.len() {
                return Err(EngineError::backend(format!(
                    "Logits slice out of bounds for batch {} (start={}, end={}, len={})",
                    batch_idx,
                    start,
                    end,
                    logits_data.len()
                )));
            }

            let (request_id, response_tx, sampling_params) = {
                let group = item.group_arc.lock().unwrap();
                (
                    group.request_id.clone(),
                    group.response_tx.clone(),
                    group.sampling_params.clone(),
                )
            };

            let (token_ids_for_penalty, mut rng_state) = {
                let seq = item.seq_arc.lock().unwrap();
                let token_ids = (sampling_params.repetition_penalty > 1.0).then(|| {
                    seq.prompt_token_ids
                        .iter()
                        .chain(seq.output_token_ids.iter())
                        .copied()
                        .collect::<Vec<u32>>()
                });
                (token_ids, seq.rng_state)
            };
            let next = if let Some(token_ids) = token_ids_for_penalty {
                let penalty = sampling_params.repetition_penalty;
                self.logits_workspace.clear();
                self.logits_workspace
                    .extend_from_slice(&logits_data[start..end]);
                for token_id in token_ids {
                    let idx = token_id as usize;
                    if idx < self.logits_workspace.len() {
                        let val = self.logits_workspace[idx];
                        self.logits_workspace[idx] = if val > 0.0 {
                            val / penalty
                        } else {
                            val * penalty
                        };
                    }
                }
                Self::sample_next_token(&self.logits_workspace, &sampling_params, &mut rng_state)
            } else {
                Self::sample_next_token(&logits_data[start..end], &sampling_params, &mut rng_state)
            };
            let mut finish_reason = None;
            let (old_len, old_status, new_len, new_status, text) = {
                let mut seq = item.seq_arc.lock().unwrap();
                let old_len = seq.get_len();
                let old_status = seq.status;
                seq.kv_cached_len = seq.kv_cached_len.saturating_add(1);
                seq.rng_state = rng_state;
                seq.append_token_id(next, 0.0);
                if sampling_params.stop_token_ids.contains(&next) {
                    seq.status = SequenceStatus::Finished(FinishReason::Stop);
                    finish_reason = Some(FinishReason::Stop);
                } else if seq.generated_this_turn >= sampling_params.max_tokens {
                    seq.status = SequenceStatus::Finished(FinishReason::Length);
                    finish_reason = Some(FinishReason::Length);
                }
                let text = self
                    .tokenizer
                    .as_ref()
                    .unwrap()
                    .decode(&[next], true)
                    .unwrap_or_default();
                (old_len, old_status, seq.get_len(), seq.status, text)
            };

            if old_len != new_len || old_status != new_status {
                let mut group = item.group_arc.lock().unwrap();
                if old_len != new_len {
                    group.update_seq_len(item.seq_id, new_len);
                }
                if old_status != new_status {
                    group.update_seq_status(old_status, new_status);
                }
            }

            let _ = response_tx.try_send(SequenceGroupOutput {
                request_id,
                text,
                finish_reason,
            });
        }

        let sampling_ms = sampling_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let total_ms = step_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        if profile_decode {
            eprintln!(
                "[decode-profile] batch_size={} actual_max_total_len={} padded_max_total_len={} bucket_granularity={} input_prep_ms={:.3} kv_pack_ms={:.3} ort_run_ms={:.3} kv_writeback_ms={:.3} sampling_ms={:.3} total_ms={:.3} present0={:?} logits_shape={:?}",
                batch_size,
                actual_max_total_len,
                max_total_len,
                decode_bucket_granularity,
                input_prep_ms,
                kv_pack_ms,
                ort_run_ms,
                kv_writeback_ms,
                sampling_ms,
                total_ms,
                first_present_shape,
                logits_shape,
            );
        }

        items.clear();
        self.decode_items_pool = items;
        Ok(true)
    }

    async fn execute_step(
        &mut self,
        groups: &[Arc<Mutex<SequenceGroup>>],
    ) -> Result<(), EngineError> {
        if self.pipeline_stages.is_some() {
            return self.execute_pipeline_step(groups).await;
        }
        // Arc-clone model-declared inputs & shapes (O(1) pointer bump) to avoid borrowing self
        // while mutably borrowing session.
        let model_input_names = self.model_input_names.clone();
        let model_input_shapes = self.model_input_shapes.clone();
        let model_input_types = self.model_input_types.clone();
        let decode_model_input_names = self.decode_model_input_names();
        let decode_model_input_shapes = self.decode_model_input_shapes();
        let decode_model_input_types = self.decode_model_input_types();
        let use_kv_cache = self.use_kv_cache;

        if self.try_execute_batched_prefill_step(groups, &model_input_names, &model_input_types)? {
            return Ok(());
        }

        if self.try_execute_batched_decode_step(
            groups,
            &decode_model_input_names,
            &decode_model_input_types,
        )? {
            return Ok(());
        }

        for group_arc in groups {
            if Self::cancel_group_if_needed(group_arc) {
                continue;
            }
            let seqs: Vec<_> = {
                let group = group_arc.lock().unwrap();
                group.sequences.values().cloned().collect()
            };

            for seq_arc in seqs {
                let (input_tokens, seq_id, total_len, kv_cached_len) = {
                    let seq = seq_arc.lock().unwrap();
                    if seq.is_finished() {
                        continue;
                    }

                    let total_len = seq.get_len();
                    let kv_cached_len = seq.kv_cached_len;
                    let input_tokens = if use_kv_cache {
                        let prompt_len = seq.prompt_token_ids.len();
                        let start = kv_cached_len;
                        if start < prompt_len {
                            let mut tokens = seq.prompt_token_ids[start..].to_vec();
                            tokens.extend_from_slice(&seq.output_token_ids);
                            tokens
                        } else {
                            let start_out = start.saturating_sub(prompt_len);
                            seq.output_token_ids[start_out..].to_vec()
                        }
                    } else {
                        let mut tokens = seq.prompt_token_ids.clone();
                        tokens.extend(seq.output_token_ids.iter().copied());
                        tokens
                    };

                    (input_tokens, seq.sequence_id, total_len, kv_cached_len)
                };

                let input_len = input_tokens.len();
                if input_len == 0 {
                    continue;
                }

                let input_tokens_i64: Vec<i64> = input_tokens.iter().map(|&x| x as i64).collect();
                let input_ids_i64 =
                    Array2::from_shape_vec((1, input_len), input_tokens_i64).unwrap();

                // attention_mask: use total sequence length (past + current).
                // When kv_cached_len == 0 we send a dummy past KV of length 1 to avoid
                // zero-element tensors rejected by CoreML; prepend a 0 to mask it out.
                let uses_dummy_past = use_kv_cache && kv_cached_len == 0;
                let mask_len = if uses_dummy_past {
                    1 + total_len
                } else {
                    total_len
                };
                let mask_data_i64: Vec<i64> = if uses_dummy_past {
                    std::iter::once(0i64)
                        .chain((0..total_len).map(|_| 1i64))
                        .collect()
                } else {
                    (0..total_len).map(|_| 1i64).collect()
                };
                let mask_i64 = Array2::from_shape_vec((1, mask_len), mask_data_i64).unwrap();

                // position_ids: [batch, input_len] with absolute positions into total sequence.
                let pos_start = total_len.saturating_sub(input_len);
                let pos_ids_vec: Vec<i64> = (pos_start..total_len).map(|v| v as i64).collect();
                let pos_ids_i64 = Array2::from_shape_vec((1, input_len), pos_ids_vec).unwrap();

                let use_dedicated_decode_session = self.decode_session.is_some()
                    && use_kv_cache
                    && kv_cached_len > 0
                    && input_len == 1;
                let active_input_names = if use_dedicated_decode_session {
                    &decode_model_input_names
                } else {
                    &model_input_names
                };
                let active_input_shapes = if use_dedicated_decode_session {
                    &decode_model_input_shapes
                } else {
                    &model_input_shapes
                };
                let active_input_types = if use_dedicated_decode_session {
                    &decode_model_input_types
                } else {
                    &model_input_types
                };

                if !active_input_names.contains("input_ids") {
                    return Err(EngineError::backend(
                        "Model input 'input_ids' not found".to_string(),
                    ));
                }

                let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = Vec::with_capacity(3);
                let input_ids_type = active_input_types
                    .get("input_ids")
                    .copied()
                    .unwrap_or(TensorElementType::Int64);
                let input_ids_value: DynValue = match input_ids_type {
                    TensorElementType::Int64 => Value::from_array(input_ids_i64)
                        .map_err(|e| EngineError::backend(e.to_string()))?
                        .into_dyn(),
                    TensorElementType::Int32 => {
                        let input_ids_i32 = Array2::from_shape_vec(
                            (1, input_len),
                            input_tokens.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
                        )
                        .unwrap();
                        Value::from_array(input_ids_i32)
                            .map_err(|e| EngineError::backend(e.to_string()))?
                            .into_dyn()
                    }
                    other => {
                        return Err(EngineError::backend(format!(
                            "Unsupported input_ids dtype: {:?}",
                            other
                        )))
                    }
                };
                inputs.push(("input_ids".into(), input_ids_value.into()));

                if active_input_names.contains("attention_mask") {
                    let mask_type = active_input_types
                        .get("attention_mask")
                        .copied()
                        .unwrap_or(TensorElementType::Int64);
                    let mask_value: DynValue = match mask_type {
                        TensorElementType::Int64 => Value::from_array(mask_i64)
                            .map_err(|e| EngineError::backend(e.to_string()))?
                            .into_dyn(),
                        TensorElementType::Int32 => {
                            let data: Vec<i32> = if uses_dummy_past {
                                std::iter::once(0i32)
                                    .chain((0..total_len).map(|_| 1i32))
                                    .collect()
                            } else {
                                (0..total_len).map(|_| 1i32).collect()
                            };
                            let mask_i32 = Array2::from_shape_vec((1, mask_len), data).unwrap();
                            Value::from_array(mask_i32)
                                .map_err(|e| EngineError::backend(e.to_string()))?
                                .into_dyn()
                        }
                        TensorElementType::Bool => {
                            let data: Vec<bool> = if uses_dummy_past {
                                std::iter::once(false)
                                    .chain((0..total_len).map(|_| true))
                                    .collect()
                            } else {
                                (0..total_len).map(|_| true).collect()
                            };
                            let mask_bool = Array2::from_shape_vec((1, mask_len), data).unwrap();
                            Value::from_array(mask_bool)
                                .map_err(|e| EngineError::backend(e.to_string()))?
                                .into_dyn()
                        }
                        TensorElementType::Float32 => {
                            let data: Vec<f32> = if uses_dummy_past {
                                std::iter::once(0.0f32)
                                    .chain((0..total_len).map(|_| 1.0f32))
                                    .collect()
                            } else {
                                (0..total_len).map(|_| 1.0f32).collect()
                            };
                            let mask_f32 = Array2::from_shape_vec((1, mask_len), data).unwrap();
                            Value::from_array(mask_f32)
                                .map_err(|e| EngineError::backend(e.to_string()))?
                                .into_dyn()
                        }
                        other => {
                            return Err(EngineError::backend(format!(
                                "Unsupported attention_mask dtype: {:?}",
                                other
                            )))
                        }
                    };
                    inputs.push(("attention_mask".into(), mask_value.into()));
                }

                if active_input_names.contains("position_ids") {
                    let pos_type = active_input_types
                        .get("position_ids")
                        .copied()
                        .unwrap_or(TensorElementType::Int64);
                    let pos_value: DynValue = match pos_type {
                        TensorElementType::Int64 => Value::from_array(pos_ids_i64)
                            .map_err(|e| EngineError::backend(e.to_string()))?
                            .into_dyn(),
                        TensorElementType::Int32 => {
                            let pos_i32 = Array2::from_shape_vec(
                                (1, input_len),
                                (pos_start..total_len)
                                    .map(|v| v as i32)
                                    .collect::<Vec<i32>>(),
                            )
                            .unwrap();
                            Value::from_array(pos_i32)
                                .map_err(|e| EngineError::backend(e.to_string()))?
                                .into_dyn()
                        }
                        other => {
                            return Err(EngineError::backend(format!(
                                "Unsupported position_ids dtype: {:?}",
                                other
                            )))
                        }
                    };
                    inputs.push(("position_ids".into(), pos_value.into()));
                }

                // === Feed KV from production cache ===
                // Only send past_key_values inputs if the model declares the names.
                // If the model declares them but our view length is zero, create and send a zero-length
                // tensor matching the model-declared rank (using captured shapes) so runtime ops like Concat
                // that expect the inputs won't fail due to missing inputs.
                if use_kv_cache {
                    enum KvPlan {
                        Packed,
                        View,
                        Empty,
                        Skip,
                    }

                    let mut plans = Vec::with_capacity(self.num_layers);
                    let mut present_names: HashSet<String> = HashSet::new();

                    for layer in 0..self.num_layers {
                        let key_name = format!("past_key_values.{}.key", layer);
                        let val_name = format!("past_key_values.{}.value", layer);

                        if !active_input_names.contains(&key_name)
                            && !active_input_names.contains(&val_name)
                        {
                            plans.push(KvPlan::Skip);
                            continue;
                        }

                        let packed_len = self
                            .kv_cache
                            .get_packed_layer(seq_id, layer)
                            .map(|v| v.length)
                            .unwrap_or(0);
                        if packed_len > 0 {
                            plans.push(KvPlan::Packed);
                            if active_input_names.contains(&key_name) {
                                present_names.insert(key_name.clone());
                            }
                            if active_input_names.contains(&val_name) {
                                present_names.insert(val_name.clone());
                            }
                            continue;
                        }

                        let view_len = self
                            .kv_cache
                            .get_layer_view(seq_id, layer)
                            .map(|v| v.length)
                            .unwrap_or(0);
                        if view_len > 0 {
                            plans.push(KvPlan::View);
                            if active_input_names.contains(&key_name) {
                                present_names.insert(key_name.clone());
                            }
                            if active_input_names.contains(&val_name) {
                                present_names.insert(val_name.clone());
                            }
                        } else {
                            plans.push(KvPlan::Empty);
                        }
                    }

                    let mut empty_inputs: Vec<(String, DynValue)> = Vec::new();
                    for layer in 0..self.num_layers {
                        let key_name = format!("past_key_values.{}.key", layer);
                        let val_name = format!("past_key_values.{}.value", layer);
                        if active_input_names.contains(&key_name)
                            && !present_names.contains(&key_name)
                        {
                            let kv_type = active_input_types
                                .get(&key_name)
                                .copied()
                                .unwrap_or(TensorElementType::Float16);
                            let target_shape = empty_kv_shape(
                                active_input_shapes.get(&key_name),
                                self.kv_layout,
                                self.num_heads,
                                self.head_dim,
                            );
                            let v = self.get_empty_kv_value(&key_name, kv_type, &target_shape)?;
                            empty_inputs.push((key_name, v));
                        }
                        if active_input_names.contains(&val_name)
                            && !present_names.contains(&val_name)
                        {
                            let kv_type = active_input_types
                                .get(&val_name)
                                .copied()
                                .unwrap_or(TensorElementType::Float16);
                            let target_shape = empty_kv_shape(
                                active_input_shapes.get(&val_name),
                                self.kv_layout,
                                self.num_heads,
                                self.head_dim,
                            );
                            let v = self.get_empty_kv_value(&val_name, kv_type, &target_shape)?;
                            empty_inputs.push((val_name, v));
                        }
                    }

                    for (layer, plan) in plans.iter().enumerate().take(self.num_layers) {
                        let key_name = format!("past_key_values.{}.key", layer);
                        let val_name = format!("past_key_values.{}.value", layer);

                        match plan {
                            KvPlan::Skip | KvPlan::Empty => {}
                            KvPlan::Packed => {
                                if let Some(packed) = self.kv_cache.get_packed_layer(seq_id, layer)
                                {
                                    if packed.length > 0 {
                                        if active_input_names.contains(&key_name) {
                                            let key_type = active_input_types
                                                .get(&key_name)
                                                .copied()
                                                .unwrap_or(TensorElementType::Float16);
                                            let k_input: SessionInputValue<'_> = match key_type {
                                                TensorElementType::Float16 => {
                                                    build_packed_kv_input_f16(
                                                        packed.key.clone(),
                                                        self.num_heads,
                                                        packed.length,
                                                        self.head_dim,
                                                        self.kv_layout,
                                                        "key",
                                                    )?
                                                }
                                                TensorElementType::Float32 => {
                                                    let arr = build_kv_array_f32_from_packed(
                                                        packed.key.as_ref(),
                                                        self.num_heads,
                                                        packed.length,
                                                        self.head_dim,
                                                        self.kv_layout,
                                                        "key",
                                                    )
                                                    .and_then(|arr| {
                                                        Value::from_array(arr).map_err(|e| {
                                                            EngineError::backend(e.to_string())
                                                        })
                                                    })?;
                                                    arr.into()
                                                }
                                                other => {
                                                    return Err(EngineError::backend(format!(
                                                        "Unsupported KV key dtype for {}: {:?}",
                                                        key_name, other
                                                    )))
                                                }
                                            };
                                            inputs.push((key_name.clone().into(), k_input));
                                        }

                                        if active_input_names.contains(&val_name) {
                                            let val_type = active_input_types
                                                .get(&val_name)
                                                .copied()
                                                .unwrap_or(TensorElementType::Float16);
                                            let v_input: SessionInputValue<'_> = match val_type {
                                                TensorElementType::Float16 => {
                                                    build_packed_kv_input_f16(
                                                        packed.value.clone(),
                                                        self.num_heads,
                                                        packed.length,
                                                        self.head_dim,
                                                        self.kv_layout,
                                                        "value",
                                                    )?
                                                }
                                                TensorElementType::Float32 => {
                                                    let arr = build_kv_array_f32_from_packed(
                                                        packed.value.as_ref(),
                                                        self.num_heads,
                                                        packed.length,
                                                        self.head_dim,
                                                        self.kv_layout,
                                                        "value",
                                                    )
                                                    .and_then(|arr| {
                                                        Value::from_array(arr).map_err(|e| {
                                                            EngineError::backend(e.to_string())
                                                        })
                                                    })?;
                                                    arr.into()
                                                }
                                                other => {
                                                    return Err(EngineError::backend(format!(
                                                        "Unsupported KV value dtype for {}: {:?}",
                                                        val_name, other
                                                    )))
                                                }
                                            };
                                            inputs.push((val_name.clone().into(), v_input));
                                        }
                                    }
                                }
                            }
                            KvPlan::View => {
                                if let Some(view) = self.kv_cache.get_layer_view(seq_id, layer) {
                                    if view.length > 0 {
                                        let key_type = active_input_types
                                            .get(&key_name)
                                            .copied()
                                            .unwrap_or(TensorElementType::Float16);
                                        let val_type = active_input_types
                                            .get(&val_name)
                                            .copied()
                                            .unwrap_or(TensorElementType::Float16);

                                        if active_input_names.contains(&key_name) {
                                            let k: DynValue = match key_type {
                                                TensorElementType::Float16 => build_kv_array_f16(
                                                    view.key.as_ref(),
                                                    self.num_heads,
                                                    view.length,
                                                    self.head_dim,
                                                    self.kv_layout,
                                                    "key",
                                                )
                                                .and_then(|arr| {
                                                    Value::from_array(arr).map_err(|e| {
                                                        EngineError::backend(e.to_string())
                                                    })
                                                })?
                                                .into_dyn(),
                                                TensorElementType::Float32 => {
                                                    build_kv_array_f32_from_f16(
                                                        view.key.as_ref(),
                                                        self.num_heads,
                                                        view.length,
                                                        self.head_dim,
                                                        self.kv_layout,
                                                        "key",
                                                    )
                                                    .and_then(|arr| {
                                                        Value::from_array(arr).map_err(|e| {
                                                            EngineError::backend(e.to_string())
                                                        })
                                                    })?
                                                    .into_dyn()
                                                }
                                                other => {
                                                    return Err(EngineError::backend(format!(
                                                        "Unsupported KV key dtype for {}: {:?}",
                                                        key_name, other
                                                    )))
                                                }
                                            };
                                            inputs.push((key_name.clone().into(), k.into()));
                                        }

                                        if active_input_names.contains(&val_name) {
                                            let v: DynValue = match val_type {
                                                TensorElementType::Float16 => build_kv_array_f16(
                                                    view.value.as_ref(),
                                                    self.num_heads,
                                                    view.length,
                                                    self.head_dim,
                                                    self.kv_layout,
                                                    "value",
                                                )
                                                .and_then(|arr| {
                                                    Value::from_array(arr).map_err(|e| {
                                                        EngineError::backend(e.to_string())
                                                    })
                                                })?
                                                .into_dyn(),
                                                TensorElementType::Float32 => {
                                                    build_kv_array_f32_from_f16(
                                                        view.value.as_ref(),
                                                        self.num_heads,
                                                        view.length,
                                                        self.head_dim,
                                                        self.kv_layout,
                                                        "value",
                                                    )
                                                    .and_then(|arr| {
                                                        Value::from_array(arr).map_err(|e| {
                                                            EngineError::backend(e.to_string())
                                                        })
                                                    })?
                                                    .into_dyn()
                                                }
                                                other => {
                                                    return Err(EngineError::backend(format!(
                                                        "Unsupported KV value dtype for {}: {:?}",
                                                        val_name, other
                                                    )))
                                                }
                                            };
                                            inputs.push((val_name.clone().into(), v.into()));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (name, value) in empty_inputs {
                        inputs.push((name.into(), value.into()));
                    }
                }

                // Run inference.
                let outputs = if use_dedicated_decode_session {
                    self.decode_session
                        .as_mut()
                        .ok_or(EngineError::ModelNotLoaded)?
                        .run(inputs)
                        .map_err(|e| EngineError::backend(e.to_string()))?
                } else {
                    self.session
                        .as_mut()
                        .ok_or(EngineError::ModelNotLoaded)?
                        .run(inputs)
                        .map_err(|e| EngineError::backend(e.to_string()))?
                };

                // === Write delta KV into production cache ===
                if use_kv_cache {
                    let mut delta_positions: Option<usize> = None;
                    for layer in 0..self.num_layers {
                        if let (Some(k_out), Some(v_out)) = (
                            outputs.get(format!("present.{}.key", layer)),
                            outputs.get(format!("present.{}.value", layer)),
                        ) {
                            let (k_shape_vec, k_data) = extract_tensor_f16(
                                k_out,
                                &format!("present key tensor for layer {}", layer),
                            )?;
                            let (_v_shape, v_data) = extract_tensor_f16(
                                v_out,
                                &format!("present value tensor for layer {}", layer),
                            )?;

                            // Robust handling:
                            // Determine dimensionality and layout heuristically. We try to find
                            // axes corresponding to num_heads and head_dim; if not found, fall back
                            // to the conventional layout [1, num_heads, delta_len, head_dim].
                            // Find axis indices
                            let mut idx_head_dim: Option<usize> = None;
                            let mut idx_num_heads: Option<usize> = None;
                            let mut idx_seq_dim: Option<usize> = None;

                            for (i, _) in k_shape_vec.iter().enumerate() {
                                if k_shape_vec[i] == self.head_dim {
                                    idx_head_dim = Some(i);
                                }
                                if k_shape_vec[i] == self.num_heads {
                                    idx_num_heads = Some(i);
                                }
                            }

                            // Choose seq axis as an axis that is not batch (0) and not head dim or num_heads
                            for (i, _) in k_shape_vec.iter().enumerate() {
                                if i == 0 {
                                    continue;
                                }
                                if Some(i) == idx_head_dim || Some(i) == idx_num_heads {
                                    continue;
                                }
                                // pick first candidate
                                idx_seq_dim = Some(i);
                                break;
                            }

                            // If any index missing, try conventional fallback positions for common layouts
                            if idx_head_dim.is_none() || idx_num_heads.is_none() {
                                // common layout: [1, num_heads, delta_len, head_dim]
                                if k_shape_vec.len() >= 4 {
                                    idx_num_heads = Some(1);
                                    idx_seq_dim = Some(2);
                                    idx_head_dim = Some(3);
                                } else if k_shape_vec.len() == 3 {
                                    // Maybe [1, num_heads, head_dim] or [num_heads, seq, head_dim]
                                    idx_head_dim =
                                        k_shape_vec.iter().position(|&d| d == self.head_dim);
                                    idx_num_heads =
                                        k_shape_vec.iter().position(|&d| d == self.num_heads);
                                    if idx_seq_dim.is_none() {
                                        // pick the remaining axis if possible
                                        for i in 0..k_shape_vec.len() {
                                            if Some(i) != idx_head_dim && Some(i) != idx_num_heads {
                                                idx_seq_dim = Some(i);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }

                            // If still missing critical indices, skip this layer to avoid panics
                            if idx_head_dim.is_none()
                                || idx_num_heads.is_none()
                                || idx_seq_dim.is_none()
                            {
                                log::warn!(
                                "Skipping present KV append for layer {} due to ambiguous tensor shape: {:?}",
                                layer,
                                k_shape_vec
                            );
                                continue;
                            }

                            let idx_head_dim = idx_head_dim.unwrap();
                            let idx_num_heads = idx_num_heads.unwrap();
                            let idx_seq_dim = idx_seq_dim.unwrap();

                            // Compute C-order strides for k_shape_vec
                            let ndim = k_shape_vec.len();
                            let mut strides = vec![0usize; ndim];
                            let mut acc = 1usize;
                            for i in (0..ndim).rev() {
                                strides[i] = acc;
                                acc *= k_shape_vec[i];
                            }

                            let num_heads = k_shape_vec[idx_num_heads];
                            let seq_dim = k_shape_vec[idx_seq_dim];
                            // Map present seq axis to absolute cache positions (handles delta-only outputs).
                            // When uses_dummy_past, position 0 in the present KV is the dummy zero-filled
                            // timestep; append_start=1 skips it. abs_pos_start is always 0 so actual tokens
                            // are cached starting at position 0 (not 1).
                            let base_pos = total_len.saturating_sub(seq_dim);
                            let append_start = if uses_dummy_past {
                                1
                            } else {
                                seq_dim.saturating_sub(input_len)
                            };
                            let appended = seq_dim.saturating_sub(append_start);
                            if let Some(prev) = delta_positions {
                                if prev != appended {
                                    log::warn!(
                                        "Inconsistent present seq_len across layers: {} vs {}",
                                        prev,
                                        appended
                                    );
                                }
                            } else {
                                delta_positions = Some(appended);
                            }
                            let head_dim = k_shape_vec[idx_head_dim];

                            // Validate reasonable sizes
                            if num_heads == 0 || head_dim == 0 {
                                log::warn!(
                                    "Invalid present tensor dims for layer {}: {:?}",
                                    layer,
                                    k_shape_vec
                                );
                                continue;
                            }

                            let delta_len = seq_dim.saturating_sub(append_start);
                            // When uses_dummy_past, the first real token belongs at cache position 0.
                            let abs_pos_start = if uses_dummy_past {
                                0
                            } else {
                                base_pos + append_start
                            };

                            let seq_first_fast = ndim == 4
                                && idx_num_heads == 1
                                && idx_seq_dim == 2
                                && idx_head_dim == 3;
                            let head_dim_fast = ndim == 4
                                && idx_num_heads == 1
                                && idx_head_dim == 2
                                && idx_seq_dim == 3;

                            if seq_first_fast && delta_len > 0 {
                                // Fast path for [1, num_heads, seq_len, head_dim]
                                for h in 0..num_heads {
                                    let src_base = (h * seq_dim + append_start) * head_dim;
                                    let src_end = src_base + delta_len * head_dim;
                                    if src_end > k_data.len() || src_end > v_data.len() {
                                        log::warn!(
                                            "Present tensor out of bounds for layer {}: src_end={}, k_len={}, v_len={}",
                                            layer,
                                            src_end,
                                            k_data.len(),
                                            v_data.len()
                                        );
                                        break;
                                    }
                                    self.kv_cache
                                        .append_head_range_seq_first(
                                            seq_id,
                                            layer,
                                            h,
                                            abs_pos_start,
                                            &k_data[src_base..src_end],
                                            &v_data[src_base..src_end],
                                        )
                                        .map_err(|e| {
                                            EngineError::resource_exhausted(e.to_string())
                                        })?;
                                }
                            } else {
                                // Fallback: append only new positions (past is already cached).
                                for pos in append_start..seq_dim {
                                    let abs_pos = base_pos + pos;
                                    // Build per-token flattened key/value slice in engine's expected format:
                                    // concatenation of heads, each of head_dim length: [h0_d0..dN, h1_d0..dN, ...]
                                    let mut key_token: Vec<f16> =
                                        Vec::with_capacity(num_heads * head_dim);
                                    let mut val_token: Vec<f16> =
                                        Vec::with_capacity(num_heads * head_dim);

                                    for h in 0..num_heads {
                                        for d in 0..head_dim {
                                            // Build coords vector for k_shape_vec dims
                                            // Default 0 for axes we don't set
                                            let mut coords = vec![0usize; ndim];
                                            coords[idx_num_heads] = h;
                                            coords[idx_seq_dim] = pos;
                                            coords[idx_head_dim] = d;
                                            // compute flat index
                                            let mut flat = 0usize;
                                            for i in 0..ndim {
                                                flat += coords[i] * strides[i];
                                            }
                                            // bounds check
                                            if flat >= k_data.len() || flat >= v_data.len() {
                                                log::warn!("Present tensor index out of bounds (layer={}, flat={}, k_len={}, v_len={})", layer, flat, k_data.len(), v_data.len());
                                                continue;
                                            }
                                            key_token.push(k_data[flat]);
                                            val_token.push(v_data[flat]);
                                        }
                                    }

                                    // Append assembled token to KV cache
                                    // Note: append_token expects slice length == num_heads * head_dim
                                    if key_token.len() == num_heads * head_dim
                                        && val_token.len() == num_heads * head_dim
                                    {
                                        self.kv_cache
                                            .append_token(
                                                seq_id, layer, abs_pos, &key_token, &val_token,
                                                None,
                                            )
                                            .map_err(|e| {
                                                EngineError::resource_exhausted(e.to_string())
                                            })?;
                                    } else {
                                        log::warn!("Assembled token KV length mismatch for layer {}: key_token={}, val_token={}, expected={}", layer, key_token.len(), val_token.len(), num_heads * head_dim);
                                    }
                                }
                            }

                            let can_store_packed = (seq_first_fast
                                && self.kv_layout == KvLayout::SeqFirst
                                && seq_dim == total_len)
                                || (head_dim_fast
                                    && self.kv_layout == KvLayout::HeadDimFirst
                                    && seq_dim == total_len);
                            if can_store_packed {
                                self.kv_cache
                                    .set_packed_layer(seq_id, layer, seq_dim, &k_data, &v_data);
                            } else {
                                self.kv_cache.clear_packed_layer(seq_id, layer);
                            }
                        }
                    }

                    // Advance sequence cursor
                    if let Some(current_len) = self.kv_cache.sequence_length(seq_id) {
                        if total_len > current_len {
                            self.kv_cache
                                .advance_sequence_by(seq_id, total_len - current_len);
                        } else if total_len < current_len {
                            log::warn!(
                                "Sequence length regressed: current_len={} target_len={}",
                                current_len,
                                total_len
                            );
                        }
                    } else {
                        // Fallback: advance by total sequence length if cache is missing.
                        let fallback = delta_positions.unwrap_or(total_len);
                        self.kv_cache.advance_sequence_by(seq_id, fallback);
                    }
                }

                // === Sampling ===
                let (logits_shape, logits_data) = {
                    let logits = outputs
                        .get("logits")
                        .ok_or_else(|| EngineError::backend("Missing logits output".to_string()))?;
                    extract_tensor_f32(logits, "logits")?
                };
                let mut vocab = logits_shape
                    .last()
                    .copied()
                    .filter(|v| *v > 0)
                    .map(|v| v as usize);
                if vocab.is_none() {
                    if let Some(vs) = self.vocab_size {
                        if vs > 0 && logits_data.len() % vs == 0 {
                            vocab = Some(vs);
                        }
                    }
                }
                let seq_len = if logits_shape.len() >= 2 {
                    logits_shape
                        .get(logits_shape.len() - 2)
                        .copied()
                        .filter(|v| *v > 0)
                        .map(|v| v as usize)
                } else {
                    None
                };
                let (start, end) = if let Some(vocab) = vocab {
                    let seq = seq_len.unwrap_or_else(|| logits_data.len() / vocab).max(1);
                    let start = (seq - 1) * vocab;
                    (start, start + vocab)
                } else {
                    (0, logits_data.len())
                };
                let (request_id, response_tx, sampling_params) = {
                    let group = group_arc.lock().unwrap();
                    (
                        group.request_id.clone(),
                        group.response_tx.clone(),
                        group.sampling_params.clone(),
                    )
                };
                let mut logits_slice = if end <= logits_data.len() {
                    logits_data[start..end].to_vec()
                } else {
                    logits_data.to_vec()
                };
                drop(outputs);
                let (token_ids_for_penalty, mut rng_state) = {
                    let seq = seq_arc.lock().unwrap();
                    let token_ids = (sampling_params.repetition_penalty > 1.0).then(|| {
                        seq.prompt_token_ids
                            .iter()
                            .chain(seq.output_token_ids.iter())
                            .copied()
                            .collect::<Vec<u32>>()
                    });
                    (token_ids, seq.rng_state)
                };
                if let Some(token_ids) = token_ids_for_penalty {
                    let penalty = sampling_params.repetition_penalty;
                    for token_id in token_ids {
                        let idx = token_id as usize;
                        if idx < logits_slice.len() {
                            let val = logits_slice[idx];
                            logits_slice[idx] = if val > 0.0 {
                                val / penalty
                            } else {
                                val * penalty
                            };
                        }
                    }
                }
                let next = Self::sample_next_token(&logits_slice, &sampling_params, &mut rng_state);

                {
                    let mut finish_reason = None;
                    let (old_len, old_status, new_len, new_status, text) = {
                        let mut seq = seq_arc.lock().unwrap();
                        let old_len = seq.get_len();
                        let old_status = seq.status;

                        if use_kv_cache {
                            seq.kv_cached_len = seq.kv_cached_len.saturating_add(input_len);
                        }

                        seq.rng_state = rng_state;
                        seq.append_token_id(next, 0.0);
                        if sampling_params.stop_token_ids.contains(&next) {
                            seq.status = SequenceStatus::Finished(FinishReason::Stop);
                            finish_reason = Some(FinishReason::Stop);
                        } else if seq.generated_this_turn >= sampling_params.max_tokens {
                            seq.status = SequenceStatus::Finished(FinishReason::Length);
                            finish_reason = Some(FinishReason::Length);
                        }

                        let text = self
                            .tokenizer
                            .as_ref()
                            .unwrap()
                            .decode(&[next], true)
                            .unwrap_or_default();
                        (old_len, old_status, seq.get_len(), seq.status, text)
                    };

                    if old_len != new_len || old_status != new_status {
                        let mut group = group_arc.lock().unwrap();
                        if old_len != new_len {
                            group.update_seq_len(seq_id, new_len);
                        }
                        if old_status != new_status {
                            group.update_seq_status(old_status, new_status);
                        }
                    }

                    let _ = response_tx.try_send(SequenceGroupOutput {
                        request_id,
                        text,
                        finish_reason,
                    });
                }
            }
        }

        Ok(())
    }

    async fn execute_pipeline_step(
        &mut self,
        groups: &[Arc<Mutex<SequenceGroup>>],
    ) -> Result<(), EngineError> {
        let stages = match self.pipeline_stages.as_mut() {
            Some(stages) => stages,
            None => return Err(EngineError::ModelNotLoaded),
        };
        let use_kv_cache = self.use_kv_cache;
        let empty_kv_cache = &mut self.empty_kv_cache;

        for group_arc in groups {
            if Self::cancel_group_if_needed(group_arc) {
                continue;
            }
            let seqs: Vec<_> = {
                let group = group_arc.lock().unwrap();
                group.sequences.values().cloned().collect()
            };

            for seq_arc in seqs {
                let (input_tokens, seq_id, total_len, kv_cached_len) = {
                    let seq = seq_arc.lock().unwrap();
                    if seq.is_finished() {
                        continue;
                    }

                    let total_len = seq.get_len();
                    let kv_cached_len = seq.kv_cached_len;
                    let input_tokens = if use_kv_cache {
                        let prompt_len = seq.prompt_token_ids.len();
                        let start = kv_cached_len;
                        if start < prompt_len {
                            let mut tokens = seq.prompt_token_ids[start..].to_vec();
                            tokens.extend_from_slice(&seq.output_token_ids);
                            tokens
                        } else {
                            let start_out = start.saturating_sub(prompt_len);
                            seq.output_token_ids[start_out..].to_vec()
                        }
                    } else {
                        let mut tokens = seq.prompt_token_ids.clone();
                        tokens.extend(seq.output_token_ids.iter().copied());
                        tokens
                    };

                    (input_tokens, seq.sequence_id, total_len, kv_cached_len)
                };

                let input_len = input_tokens.len();
                if input_len == 0 {
                    continue;
                }

                let input_tokens_i64: Vec<i64> = input_tokens.iter().map(|&x| x as i64).collect();
                let input_ids_i64 =
                    Array2::from_shape_vec((1, input_len), input_tokens_i64).unwrap();

                let uses_dummy_past = use_kv_cache && kv_cached_len == 0;
                let mask_len = if uses_dummy_past {
                    1 + total_len
                } else {
                    total_len
                };
                let mask_data_i64: Vec<i64> = if uses_dummy_past {
                    std::iter::once(0i64)
                        .chain((0..total_len).map(|_| 1i64))
                        .collect()
                } else {
                    (0..total_len).map(|_| 1i64).collect()
                };
                let mask_i64 = Array2::from_shape_vec((1, mask_len), mask_data_i64).unwrap();

                let pos_start = total_len.saturating_sub(input_len);
                let pos_ids_vec: Vec<i64> = (pos_start..total_len).map(|v| v as i64).collect();
                let pos_ids_i64 = Array2::from_shape_vec((1, input_len), pos_ids_vec).unwrap();

                let mut prev_outputs: Option<ort::session::SessionOutputs<'_>> = None;

                for (stage_idx, stage) in stages.iter_mut().enumerate() {
                    let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
                        Vec::with_capacity(stage.input_names.len());
                    let mut provided: HashSet<String> = HashSet::new();

                    if stage.input_names.contains("input_ids") {
                        let input_ids_type = stage
                            .input_types
                            .get("input_ids")
                            .copied()
                            .unwrap_or(TensorElementType::Int64);
                        let input_ids_value: DynValue = match input_ids_type {
                            TensorElementType::Int64 => Value::from_array(input_ids_i64.clone())
                                .map_err(|e| EngineError::backend(e.to_string()))?
                                .into_dyn(),
                            TensorElementType::Int32 => {
                                let input_ids_i32 = Array2::from_shape_vec(
                                    (1, input_len),
                                    input_tokens.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
                                )
                                .unwrap();
                                Value::from_array(input_ids_i32)
                                    .map_err(|e| EngineError::backend(e.to_string()))?
                                    .into_dyn()
                            }
                            other => {
                                return Err(EngineError::backend(format!(
                                    "Unsupported input_ids dtype: {:?}",
                                    other
                                )))
                            }
                        };
                        inputs.push(("input_ids".into(), input_ids_value.into()));
                        provided.insert("input_ids".to_string());
                    } else if stage_idx == 0 {
                        return Err(EngineError::backend(
                            "Model input 'input_ids' not found".to_string(),
                        ));
                    }

                    if stage.input_names.contains("attention_mask") {
                        let mask_type = stage
                            .input_types
                            .get("attention_mask")
                            .copied()
                            .unwrap_or(TensorElementType::Int64);
                        let mask_value: DynValue = match mask_type {
                            TensorElementType::Int64 => Value::from_array(mask_i64.clone())
                                .map_err(|e| EngineError::backend(e.to_string()))?
                                .into_dyn(),
                            TensorElementType::Int32 => {
                                let data: Vec<i32> = if uses_dummy_past {
                                    std::iter::once(0i32)
                                        .chain((0..total_len).map(|_| 1i32))
                                        .collect()
                                } else {
                                    (0..total_len).map(|_| 1i32).collect()
                                };
                                let mask_i32 = Array2::from_shape_vec((1, mask_len), data).unwrap();
                                Value::from_array(mask_i32)
                                    .map_err(|e| EngineError::backend(e.to_string()))?
                                    .into_dyn()
                            }
                            TensorElementType::Bool => {
                                let data: Vec<bool> = if uses_dummy_past {
                                    std::iter::once(false)
                                        .chain((0..total_len).map(|_| true))
                                        .collect()
                                } else {
                                    (0..total_len).map(|_| true).collect()
                                };
                                let mask_bool =
                                    Array2::from_shape_vec((1, mask_len), data).unwrap();
                                Value::from_array(mask_bool)
                                    .map_err(|e| EngineError::backend(e.to_string()))?
                                    .into_dyn()
                            }
                            TensorElementType::Float32 => {
                                let data: Vec<f32> = if uses_dummy_past {
                                    std::iter::once(0.0f32)
                                        .chain((0..total_len).map(|_| 1.0f32))
                                        .collect()
                                } else {
                                    (0..total_len).map(|_| 1.0f32).collect()
                                };
                                let mask_f32 = Array2::from_shape_vec((1, mask_len), data).unwrap();
                                Value::from_array(mask_f32)
                                    .map_err(|e| EngineError::backend(e.to_string()))?
                                    .into_dyn()
                            }
                            other => {
                                return Err(EngineError::backend(format!(
                                    "Unsupported attention_mask dtype: {:?}",
                                    other
                                )))
                            }
                        };
                        inputs.push(("attention_mask".into(), mask_value.into()));
                        provided.insert("attention_mask".to_string());
                    }

                    if stage.input_names.contains("position_ids") {
                        let pos_type = stage
                            .input_types
                            .get("position_ids")
                            .copied()
                            .unwrap_or(TensorElementType::Int64);
                        let pos_value: DynValue = match pos_type {
                            TensorElementType::Int64 => Value::from_array(pos_ids_i64.clone())
                                .map_err(|e| EngineError::backend(e.to_string()))?
                                .into_dyn(),
                            TensorElementType::Int32 => {
                                let pos_i32 = Array2::from_shape_vec(
                                    (1, input_len),
                                    (pos_start..total_len)
                                        .map(|v| v as i32)
                                        .collect::<Vec<i32>>(),
                                )
                                .unwrap();
                                Value::from_array(pos_i32)
                                    .map_err(|e| EngineError::backend(e.to_string()))?
                                    .into_dyn()
                            }
                            other => {
                                return Err(EngineError::backend(format!(
                                    "Unsupported position_ids dtype: {:?}",
                                    other
                                )))
                            }
                        };
                        inputs.push(("position_ids".into(), pos_value.into()));
                        provided.insert("position_ids".to_string());
                    }

                    if use_kv_cache {
                        enum KvPlan {
                            Packed,
                            View,
                            Empty,
                            Skip,
                        }

                        let mut plans = Vec::with_capacity(self.num_layers);
                        let mut present_names: HashSet<String> = HashSet::new();

                        for layer in 0..self.num_layers {
                            let key_name = format!("past_key_values.{}.key", layer);
                            let val_name = format!("past_key_values.{}.value", layer);

                            if !stage.input_names.contains(&key_name)
                                && !stage.input_names.contains(&val_name)
                            {
                                plans.push(KvPlan::Skip);
                                continue;
                            }

                            let packed_len = self
                                .kv_cache
                                .get_packed_layer(seq_id, layer)
                                .map(|v| v.length)
                                .unwrap_or(0);
                            if packed_len > 0 {
                                plans.push(KvPlan::Packed);
                                if stage.input_names.contains(&key_name) {
                                    present_names.insert(key_name.clone());
                                }
                                if stage.input_names.contains(&val_name) {
                                    present_names.insert(val_name.clone());
                                }
                                continue;
                            }

                            let view_len = self
                                .kv_cache
                                .get_layer_view(seq_id, layer)
                                .map(|v| v.length)
                                .unwrap_or(0);
                            if view_len > 0 {
                                plans.push(KvPlan::View);
                                if stage.input_names.contains(&key_name) {
                                    present_names.insert(key_name.clone());
                                }
                                if stage.input_names.contains(&val_name) {
                                    present_names.insert(val_name.clone());
                                }
                            } else {
                                plans.push(KvPlan::Empty);
                            }
                        }

                        let mut empty_inputs: Vec<(String, DynValue)> = Vec::new();
                        for layer in 0..self.num_layers {
                            let key_name = format!("past_key_values.{}.key", layer);
                            let val_name = format!("past_key_values.{}.value", layer);
                            if stage.input_names.contains(&key_name)
                                && !present_names.contains(&key_name)
                            {
                                let kv_type = stage
                                    .input_types
                                    .get(&key_name)
                                    .copied()
                                    .unwrap_or(TensorElementType::Float16);
                                let target_shape = empty_kv_shape(
                                    stage.input_shapes.get(&key_name),
                                    self.kv_layout,
                                    self.num_heads,
                                    self.head_dim,
                                );
                                let v = get_empty_kv_value_cached(
                                    empty_kv_cache,
                                    &key_name,
                                    kv_type,
                                    &target_shape,
                                )?;
                                empty_inputs.push((key_name, v));
                            }
                            if stage.input_names.contains(&val_name)
                                && !present_names.contains(&val_name)
                            {
                                let kv_type = stage
                                    .input_types
                                    .get(&val_name)
                                    .copied()
                                    .unwrap_or(TensorElementType::Float16);
                                let target_shape = empty_kv_shape(
                                    stage.input_shapes.get(&val_name),
                                    self.kv_layout,
                                    self.num_heads,
                                    self.head_dim,
                                );
                                let v = get_empty_kv_value_cached(
                                    empty_kv_cache,
                                    &val_name,
                                    kv_type,
                                    &target_shape,
                                )?;
                                empty_inputs.push((val_name, v));
                            }
                        }

                        for (layer, plan) in plans.iter().enumerate().take(self.num_layers) {
                            let key_name = format!("past_key_values.{}.key", layer);
                            let val_name = format!("past_key_values.{}.value", layer);

                            match plan {
                                KvPlan::Skip | KvPlan::Empty => {}
                                KvPlan::Packed => {
                                    if let Some(packed) =
                                        self.kv_cache.get_packed_layer(seq_id, layer)
                                    {
                                        if packed.length > 0 {
                                            if stage.input_names.contains(&key_name) {
                                                let key_type = stage
                                                    .input_types
                                                    .get(&key_name)
                                                    .copied()
                                                    .unwrap_or(TensorElementType::Float16);
                                                let k_input: SessionInputValue<'_> = match key_type
                                                {
                                                    TensorElementType::Float16 => {
                                                        build_packed_kv_input_f16(
                                                            packed.key.clone(),
                                                            self.num_heads,
                                                            packed.length,
                                                            self.head_dim,
                                                            self.kv_layout,
                                                            "key",
                                                        )?
                                                    }
                                                    TensorElementType::Float32 => {
                                                        let arr = build_kv_array_f32_from_packed(
                                                            packed.key.as_ref(),
                                                            self.num_heads,
                                                            packed.length,
                                                            self.head_dim,
                                                            self.kv_layout,
                                                            "key",
                                                        )
                                                        .and_then(|arr| {
                                                            Value::from_array(arr).map_err(|e| {
                                                                EngineError::backend(e.to_string())
                                                            })
                                                        })?;
                                                        arr.into()
                                                    }
                                                    other => {
                                                        return Err(EngineError::backend(format!(
                                                            "Unsupported KV key dtype for {}: {:?}",
                                                            key_name, other
                                                        )))
                                                    }
                                                };
                                                inputs.push((key_name.clone().into(), k_input));
                                                provided.insert(key_name.clone());
                                            }

                                            if stage.input_names.contains(&val_name) {
                                                let val_type = stage
                                                    .input_types
                                                    .get(&val_name)
                                                    .copied()
                                                    .unwrap_or(TensorElementType::Float16);
                                                let v_input: SessionInputValue<'_> =
                                                    match val_type {
                                                        TensorElementType::Float16 => {
                                                            let arr = build_kv_array_f16(
                                                                packed.value.as_ref(),
                                                                self.num_heads,
                                                                packed.length,
                                                                self.head_dim,
                                                                self.kv_layout,
                                                                "value",
                                                            )
                                                            .and_then(|arr| {
                                                                Value::from_array(arr).map_err(
                                                                    |e| {
                                                                        EngineError::backend(
                                                                            e.to_string(),
                                                                        )
                                                                    },
                                                                )
                                                            })?;
                                                            arr.into()
                                                        }
                                                        TensorElementType::Float32 => {
                                                            let arr =
                                                                build_kv_array_f32_from_packed(
                                                                    packed.value.as_ref(),
                                                                    self.num_heads,
                                                                    packed.length,
                                                                    self.head_dim,
                                                                    self.kv_layout,
                                                                    "value",
                                                                )
                                                                .and_then(|arr| {
                                                                    Value::from_array(arr).map_err(
                                                                        |e| {
                                                                            EngineError::backend(
                                                                                e.to_string(),
                                                                            )
                                                                        },
                                                                    )
                                                                })?;
                                                            arr.into()
                                                        }
                                                        other => {
                                                            return Err(EngineError::backend(
                                                                format!(
                                                                    "Unsupported KV value dtype for {}: {:?}",
                                                                    val_name, other
                                                                ),
                                                            ))
                                                        }
                                                    };
                                                inputs.push((val_name.clone().into(), v_input));
                                                provided.insert(val_name.clone());
                                            }
                                        }
                                    }
                                }
                                KvPlan::View => {
                                    if let Some(view) = self.kv_cache.get_layer_view(seq_id, layer)
                                    {
                                        if view.length > 0 {
                                            let key_type = stage
                                                .input_types
                                                .get(&key_name)
                                                .copied()
                                                .unwrap_or(TensorElementType::Float16);
                                            let val_type = stage
                                                .input_types
                                                .get(&val_name)
                                                .copied()
                                                .unwrap_or(TensorElementType::Float16);

                                            if stage.input_names.contains(&key_name) {
                                                let k: DynValue = match key_type {
                                                    TensorElementType::Float16 => {
                                                        build_kv_array_f16(
                                                            view.key.as_ref(),
                                                            self.num_heads,
                                                            view.length,
                                                            self.head_dim,
                                                            self.kv_layout,
                                                            "key",
                                                        )
                                                        .and_then(|arr| {
                                                            Value::from_array(arr).map_err(|e| {
                                                                EngineError::backend(e.to_string())
                                                            })
                                                        })?
                                                        .into_dyn()
                                                    }
                                                    TensorElementType::Float32 => {
                                                        build_kv_array_f32_from_f16(
                                                            view.key.as_ref(),
                                                            self.num_heads,
                                                            view.length,
                                                            self.head_dim,
                                                            self.kv_layout,
                                                            "key",
                                                        )
                                                        .and_then(|arr| {
                                                            Value::from_array(arr).map_err(|e| {
                                                                EngineError::backend(e.to_string())
                                                            })
                                                        })?
                                                        .into_dyn()
                                                    }
                                                    other => {
                                                        return Err(EngineError::backend(format!(
                                                            "Unsupported KV key dtype for {}: {:?}",
                                                            key_name, other
                                                        )))
                                                    }
                                                };
                                                inputs.push((key_name.clone().into(), k.into()));
                                                provided.insert(key_name.clone());
                                            }

                                            if stage.input_names.contains(&val_name) {
                                                let v: DynValue = match val_type {
                                                    TensorElementType::Float16 => {
                                                        build_kv_array_f16(
                                                            view.value.as_ref(),
                                                            self.num_heads,
                                                            view.length,
                                                            self.head_dim,
                                                            self.kv_layout,
                                                            "value",
                                                        )
                                                        .and_then(|arr| {
                                                            Value::from_array(arr).map_err(|e| {
                                                                EngineError::backend(
                                                                    e.to_string(),
                                                                )
                                                            })
                                                        })?
                                                        .into_dyn()
                                                    }
                                                    TensorElementType::Float32 => {
                                                        build_kv_array_f32_from_f16(
                                                            view.value.as_ref(),
                                                            self.num_heads,
                                                            view.length,
                                                            self.head_dim,
                                                            self.kv_layout,
                                                            "value",
                                                        )
                                                        .and_then(|arr| {
                                                            Value::from_array(arr).map_err(|e| {
                                                                EngineError::backend(
                                                                    e.to_string(),
                                                                )
                                                            })
                                                        })?
                                                        .into_dyn()
                                                    }
                                                    other => {
                                                        return Err(EngineError::backend(
                                                            format!(
                                                                "Unsupported KV value dtype for {}: {:?}",
                                                                val_name, other
                                                            ),
                                                        ))
                                                    }
                                                };
                                                inputs.push((val_name.clone().into(), v.into()));
                                                provided.insert(val_name.clone());
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        for (name, value) in empty_inputs {
                            inputs.push((name.clone().into(), value.into()));
                            provided.insert(name);
                        }
                    }

                    let mut prev_map = prev_outputs.take();
                    for name in stage.input_names.iter() {
                        if provided.contains(name) {
                            continue;
                        }
                        if let Some(prev) = prev_map.as_mut() {
                            if let Some(value) = prev.remove(name) {
                                inputs.push((name.clone().into(), value.into()));
                                provided.insert(name.clone());
                                continue;
                            }
                        }
                        return Err(EngineError::backend(format!(
                            "Missing pipeline input '{}' for stage {}",
                            name, stage_idx
                        )));
                    }

                    let stage_outputs = stage
                        .session
                        .run(inputs)
                        .map_err(|e| EngineError::backend(e.to_string()))?;
                    prev_outputs = Some(stage_outputs);
                }

                let outputs = prev_outputs.ok_or(EngineError::ModelNotLoaded)?;

                // === Write delta KV into production cache ===
                if use_kv_cache {
                    let mut delta_positions: Option<usize> = None;
                    for layer in 0..self.num_layers {
                        if let (Some(k_out), Some(v_out)) = (
                            outputs.get(format!("present.{}.key", layer)),
                            outputs.get(format!("present.{}.value", layer)),
                        ) {
                            let (k_shape_vec, k_data) = extract_tensor_f16(
                                k_out,
                                &format!("pipeline present key tensor for layer {}", layer),
                            )?;
                            let (_v_shape, v_data) = extract_tensor_f16(
                                v_out,
                                &format!("pipeline present value tensor for layer {}", layer),
                            )?;
                            let mut idx_head_dim: Option<usize> = None;
                            let mut idx_num_heads: Option<usize> = None;
                            let mut idx_seq_dim: Option<usize> = None;

                            for (i, _) in k_shape_vec.iter().enumerate() {
                                if k_shape_vec[i] == self.head_dim {
                                    idx_head_dim = Some(i);
                                }
                                if k_shape_vec[i] == self.num_heads {
                                    idx_num_heads = Some(i);
                                }
                            }

                            for (i, _) in k_shape_vec.iter().enumerate() {
                                if i == 0 {
                                    continue;
                                }
                                if Some(i) == idx_head_dim || Some(i) == idx_num_heads {
                                    continue;
                                }
                                idx_seq_dim = Some(i);
                                break;
                            }

                            if idx_head_dim.is_none() || idx_num_heads.is_none() {
                                if k_shape_vec.len() >= 4 {
                                    idx_num_heads = Some(1);
                                    idx_seq_dim = Some(2);
                                    idx_head_dim = Some(3);
                                } else if k_shape_vec.len() == 3 {
                                    idx_head_dim =
                                        k_shape_vec.iter().position(|&d| d == self.head_dim);
                                    idx_num_heads =
                                        k_shape_vec.iter().position(|&d| d == self.num_heads);
                                    if idx_seq_dim.is_none() {
                                        for i in 0..k_shape_vec.len() {
                                            if Some(i) != idx_head_dim && Some(i) != idx_num_heads {
                                                idx_seq_dim = Some(i);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }

                            if idx_head_dim.is_none()
                                || idx_num_heads.is_none()
                                || idx_seq_dim.is_none()
                            {
                                log::warn!(
                                "Skipping present KV append for layer {} due to ambiguous tensor shape: {:?}",
                                layer,
                                k_shape_vec
                            );
                                continue;
                            }

                            let idx_head_dim = idx_head_dim.unwrap();
                            let idx_num_heads = idx_num_heads.unwrap();
                            let idx_seq_dim = idx_seq_dim.unwrap();

                            let ndim = k_shape_vec.len();
                            let mut strides = vec![0usize; ndim];
                            let mut acc = 1usize;
                            for i in (0..ndim).rev() {
                                strides[i] = acc;
                                acc *= k_shape_vec[i];
                            }

                            let num_heads = k_shape_vec[idx_num_heads];
                            let seq_dim = k_shape_vec[idx_seq_dim];
                            let base_pos = total_len.saturating_sub(seq_dim);
                            let append_start = seq_dim.saturating_sub(input_len);
                            let appended = seq_dim.saturating_sub(append_start);
                            if let Some(prev) = delta_positions {
                                if prev != appended {
                                    log::warn!(
                                        "Inconsistent present seq_len across layers: {} vs {}",
                                        prev,
                                        appended
                                    );
                                }
                            } else {
                                delta_positions = Some(appended);
                            }
                            let head_dim = k_shape_vec[idx_head_dim];

                            if num_heads == 0 || head_dim == 0 {
                                log::warn!(
                                    "Invalid present tensor dims for layer {}: {:?}",
                                    layer,
                                    k_shape_vec
                                );
                                continue;
                            }

                            let delta_len = seq_dim.saturating_sub(append_start);
                            // When uses_dummy_past, the first real token belongs at cache position 0.
                            let abs_pos_start = if uses_dummy_past {
                                0
                            } else {
                                base_pos + append_start
                            };

                            let seq_first_fast = ndim == 4
                                && idx_num_heads == 1
                                && idx_seq_dim == 2
                                && idx_head_dim == 3;
                            let head_dim_fast = ndim == 4
                                && idx_num_heads == 1
                                && idx_head_dim == 2
                                && idx_seq_dim == 3;

                            if seq_first_fast && delta_len > 0 {
                                for h in 0..num_heads {
                                    let src_base = (h * seq_dim + append_start) * head_dim;
                                    let src_end = src_base + delta_len * head_dim;
                                    if src_end > k_data.len() || src_end > v_data.len() {
                                        log::warn!(
                                            "Present tensor out of bounds for layer {}: src_end={}, k_len={}, v_len={}",
                                            layer,
                                            src_end,
                                            k_data.len(),
                                            v_data.len()
                                        );
                                        break;
                                    }
                                    self.kv_cache
                                        .append_head_range_seq_first(
                                            seq_id,
                                            layer,
                                            h,
                                            abs_pos_start,
                                            &k_data[src_base..src_end],
                                            &v_data[src_base..src_end],
                                        )
                                        .map_err(|e| {
                                            EngineError::resource_exhausted(e.to_string())
                                        })?;
                                }
                            } else {
                                for pos in append_start..seq_dim {
                                    let abs_pos = base_pos + pos;
                                    let mut key_token: Vec<f16> =
                                        Vec::with_capacity(num_heads * head_dim);
                                    let mut val_token: Vec<f16> =
                                        Vec::with_capacity(num_heads * head_dim);

                                    for h in 0..num_heads {
                                        for d in 0..head_dim {
                                            let mut coords = vec![0usize; ndim];
                                            coords[idx_num_heads] = h;
                                            coords[idx_seq_dim] = pos;
                                            coords[idx_head_dim] = d;
                                            let mut flat = 0usize;
                                            for i in 0..ndim {
                                                flat += coords[i] * strides[i];
                                            }
                                            if flat >= k_data.len() || flat >= v_data.len() {
                                                log::warn!("Present tensor index out of bounds (layer={}, flat={}, k_len={}, v_len={})", layer, flat, k_data.len(), v_data.len());
                                                continue;
                                            }
                                            key_token.push(k_data[flat]);
                                            val_token.push(v_data[flat]);
                                        }
                                    }

                                    if key_token.len() == num_heads * head_dim
                                        && val_token.len() == num_heads * head_dim
                                    {
                                        self.kv_cache
                                            .append_token(
                                                seq_id, layer, abs_pos, &key_token, &val_token,
                                                None,
                                            )
                                            .map_err(|e| {
                                                EngineError::resource_exhausted(e.to_string())
                                            })?;
                                    } else {
                                        log::warn!("Assembled token KV length mismatch for layer {}: key_token={}, val_token={}, expected={}", layer, key_token.len(), val_token.len(), num_heads * head_dim);
                                    }
                                }
                            }

                            let can_store_packed = (seq_first_fast
                                && self.kv_layout == KvLayout::SeqFirst
                                && seq_dim == total_len)
                                || (head_dim_fast
                                    && self.kv_layout == KvLayout::HeadDimFirst
                                    && seq_dim == total_len);
                            if can_store_packed {
                                self.kv_cache
                                    .set_packed_layer(seq_id, layer, seq_dim, &k_data, &v_data);
                            } else {
                                self.kv_cache.clear_packed_layer(seq_id, layer);
                            }
                        }
                    }

                    if let Some(current_len) = self.kv_cache.sequence_length(seq_id) {
                        if total_len > current_len {
                            self.kv_cache
                                .advance_sequence_by(seq_id, total_len - current_len);
                        } else if total_len < current_len {
                            log::warn!(
                                "Sequence length regressed: current_len={} target_len={}",
                                current_len,
                                total_len
                            );
                        }
                    } else {
                        let fallback = delta_positions.unwrap_or(total_len);
                        self.kv_cache.advance_sequence_by(seq_id, fallback);
                    }
                }

                // === Sampling ===
                let (logits_shape, logits_data) = {
                    let logits = outputs
                        .get("logits")
                        .ok_or_else(|| EngineError::backend("Missing logits output".to_string()))?;
                    extract_tensor_f32(logits, "logits")?
                };
                let mut vocab = logits_shape
                    .last()
                    .copied()
                    .filter(|v| *v > 0)
                    .map(|v| v as usize);
                if vocab.is_none() {
                    if let Some(vs) = self.vocab_size {
                        if vs > 0 && logits_data.len() % vs == 0 {
                            vocab = Some(vs);
                        }
                    }
                }
                let seq_len = if logits_shape.len() >= 2 {
                    logits_shape
                        .get(logits_shape.len() - 2)
                        .copied()
                        .filter(|v| *v > 0)
                        .map(|v| v as usize)
                } else {
                    None
                };
                let (start, end) = if let Some(vocab) = vocab {
                    let seq = seq_len.unwrap_or_else(|| logits_data.len() / vocab).max(1);
                    let start = (seq - 1) * vocab;
                    (start, start + vocab)
                } else {
                    (0, logits_data.len())
                };
                let (request_id, response_tx, sampling_params) = {
                    let group = group_arc.lock().unwrap();
                    (
                        group.request_id.clone(),
                        group.response_tx.clone(),
                        group.sampling_params.clone(),
                    )
                };
                let mut logits_slice = if end <= logits_data.len() {
                    logits_data[start..end].to_vec()
                } else {
                    logits_data.to_vec()
                };
                drop(outputs);
                let (token_ids_for_penalty, mut rng_state) = {
                    let seq = seq_arc.lock().unwrap();
                    let token_ids = (sampling_params.repetition_penalty > 1.0).then(|| {
                        seq.prompt_token_ids
                            .iter()
                            .chain(seq.output_token_ids.iter())
                            .copied()
                            .collect::<Vec<u32>>()
                    });
                    (token_ids, seq.rng_state)
                };
                if let Some(token_ids) = token_ids_for_penalty {
                    let penalty = sampling_params.repetition_penalty;
                    for token_id in token_ids {
                        let idx = token_id as usize;
                        if idx < logits_slice.len() {
                            let val = logits_slice[idx];
                            logits_slice[idx] = if val > 0.0 {
                                val / penalty
                            } else {
                                val * penalty
                            };
                        }
                    }
                }
                let next = Self::sample_next_token(&logits_slice, &sampling_params, &mut rng_state);

                {
                    let mut finish_reason = None;
                    let (old_len, old_status, new_len, new_status, text) = {
                        let mut seq = seq_arc.lock().unwrap();
                        let old_len = seq.get_len();
                        let old_status = seq.status;

                        if use_kv_cache {
                            seq.kv_cached_len = seq.kv_cached_len.saturating_add(input_len);
                        }

                        seq.rng_state = rng_state;
                        seq.append_token_id(next, 0.0);
                        if sampling_params.stop_token_ids.contains(&next) {
                            seq.status = SequenceStatus::Finished(FinishReason::Stop);
                            finish_reason = Some(FinishReason::Stop);
                        } else if seq.generated_this_turn >= sampling_params.max_tokens {
                            seq.status = SequenceStatus::Finished(FinishReason::Length);
                            finish_reason = Some(FinishReason::Length);
                        }

                        let text = self
                            .tokenizer
                            .as_ref()
                            .unwrap()
                            .decode(&[next], true)
                            .unwrap_or_default();
                        (old_len, old_status, seq.get_len(), seq.status, text)
                    };

                    if old_len != new_len || old_status != new_status {
                        let mut group = group_arc.lock().unwrap();
                        if old_len != new_len {
                            group.update_seq_len(seq_id, new_len);
                        }
                        if old_status != new_status {
                            group.update_seq_status(old_status, new_status);
                        }
                    }

                    let _ = response_tx.try_send(SequenceGroupOutput {
                        request_id,
                        text,
                        finish_reason,
                    });
                }
            }
        }

        Ok(())
    }

    fn fnv1a_hash64(bytes: &[u8]) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET_BASIS;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    fn splitmix64(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f32(rng_state: &mut u64) -> f32 {
        *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (*rng_state >> 32) as u32;
        (val as f32) / (u32::MAX as f32)
    }

    fn sample_next_token(logits: &[f32], params: &SamplingParams, rng_state: &mut u64) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        let temperature = params.temperature;
        if temperature <= 0.0 {
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }

        let mut top_k = if params.top_k == 0 {
            logits.len()
        } else {
            params.top_k.min(logits.len())
        };
        if top_k == 0 {
            top_k = 1;
        }

        let mut top: Vec<(usize, f32)> = Vec::with_capacity(top_k);
        for (idx, &logit) in logits.iter().enumerate() {
            let scaled = logit / temperature;
            if top.len() < top_k {
                top.push((idx, scaled));
                if top.len() == top_k {
                    top.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                }
            } else if scaled > top[0].1 {
                top[0] = (idx, scaled);
                top.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            }
        }

        if top.is_empty() {
            return 0;
        }

        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_p = params.top_p.clamp(0.0, 1.0);
        if top_p < 1.0 && top.len() > 1 {
            let max_logit = top[0].1;
            let mut exp_vals: Vec<f32> = Vec::with_capacity(top.len());
            let mut sum = 0.0;
            for &(_, l) in &top {
                let v = (l - max_logit).exp();
                exp_vals.push(v);
                sum += v;
            }

            let mut cumulative = 0.0;
            let mut cutoff = top.len();
            for (i, v) in exp_vals.iter().enumerate() {
                cumulative += v / sum;
                if cumulative >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            top.truncate(cutoff);
        }

        let max_logit = top[0].1;
        let mut exp_vals: Vec<f32> = Vec::with_capacity(top.len());
        let mut sum = 0.0;
        for &(_, l) in &top {
            let v = (l - max_logit).exp();
            exp_vals.push(v);
            sum += v;
        }

        let mut draw = Self::next_f32(rng_state) * sum;
        for ((idx, _), v) in top.iter().zip(exp_vals.iter()) {
            if draw <= *v {
                return *idx as u32;
            }
            draw -= v;
        }

        top[0].0 as u32
    }
}

#[path = "engine_tests.rs"]
mod engine_tests;
