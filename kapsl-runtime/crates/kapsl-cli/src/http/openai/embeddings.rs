//! Text and token `POST /v1/embeddings` compatibility.
//!
//! Kapsl's ONNX embedding backend already owns forward execution, masked-mean
//! pooling, and normalization. This adapter supplies the missing protocol edge:
//! OpenAI input parsing, package tokenizer lookup, integer tensor construction,
//! exact token accounting, and float/base64 response encoding.

use super::*;
use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE_NO_PAD};
use kapsl_engine_api::{EngineModelInfo, NamedTensor, RequestMetadata};
use kapsl_llm::model_paths::find_model_asset;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::SystemTime;
use tokenizers::Tokenizer;

const MAX_INPUTS: usize = 2_048;
const MAX_TOKENS_PER_INPUT: usize = 8_192;
const MAX_TOKENS_PER_REQUEST: usize = 300_000;

pub(crate) struct EmbeddingsConfig {
    pub(crate) models: Arc<ModelManager>,
    pub(crate) inference: Arc<InferenceService>,
    pub(crate) log_sensitive_ids: bool,
}

pub(crate) fn build_embeddings_route(
    config: EmbeddingsConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let EmbeddingsConfig {
        models,
        inference,
        log_sensitive_ids,
    } = config;
    let tokenizer_cache = Arc::new(TokenizerCache::default());

    warp::path!("v1" / "embeddings")
        .and(warp::post())
        .and(warp::body::bytes())
        .and(warp::header::optional::<String>("x-kapsl-session"))
        .and(warp::header::optional::<String>("authorization"))
        .and_then(
            move |body: warp::hyper::body::Bytes,
                  session_header: Option<String>,
                  authorization: Option<String>| {
                let models = models.clone();
                let inference = inference.clone();
                let tokenizer_cache = tokenizer_cache.clone();
                async move {
                    Ok::<_, warp::Rejection>(
                        handle_embeddings(
                            body,
                            session_header,
                            authorization,
                            &models,
                            &inference,
                            &tokenizer_cache,
                            log_sensitive_ids,
                        )
                        .await,
                    )
                }
            },
        )
        .map(reply_into_response)
        .boxed()
}

#[derive(Debug, Deserialize)]
struct CreateEmbeddingRequest {
    model: String,
    #[serde(default)]
    input: serde_json::Value,
    #[serde(default)]
    encoding_format: Option<String>,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    user: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbeddingEncoding {
    Float,
    Base64,
}

impl EmbeddingEncoding {
    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value
            .unwrap_or("float")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "float" => Ok(Self::Float),
            "base64" => Ok(Self::Base64),
            other => Err(format!(
                "encoding_format must be 'float' or 'base64', got '{other}'"
            )),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ParsedEmbeddingInput {
    Text(Vec<String>),
    Tokens(Vec<Vec<u32>>),
}

impl ParsedEmbeddingInput {
    fn parse(value: &serde_json::Value) -> Result<Self, String> {
        if let Some(text) = value.as_str() {
            validate_text(text, 0)?;
            return Ok(Self::Text(vec![text.to_string()]));
        }

        let Some(items) = value.as_array() else {
            return Err(
                "input must be a string, an array of strings, an array of token IDs, or an array of token-ID arrays"
                    .to_string(),
            );
        };
        if items.is_empty() {
            return Err("input must contain at least one item".to_string());
        }

        let parsed = if items.iter().all(serde_json::Value::is_string) {
            let texts = items
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let text = value.as_str().expect("checked string");
                    validate_text(text, index)?;
                    Ok(text.to_string())
                })
                .collect::<Result<Vec<_>, String>>()?;
            Self::Text(texts)
        } else if items.iter().all(serde_json::Value::is_number) {
            Self::Tokens(vec![parse_token_array(items, 0)?])
        } else if items.iter().all(serde_json::Value::is_array) {
            let token_arrays = items
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    parse_token_array(value.as_array().expect("checked array"), index)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Self::Tokens(token_arrays)
        } else {
            return Err(
                "input arrays must contain only strings, only integer token IDs, or only token-ID arrays"
                    .to_string(),
            );
        };

        if parsed.len() > MAX_INPUTS {
            return Err(format!(
                "input contains {} items; at most {MAX_INPUTS} are supported",
                parsed.len()
            ));
        }
        Ok(parsed)
    }

    fn len(&self) -> usize {
        match self {
            Self::Text(values) => values.len(),
            Self::Tokens(values) => values.len(),
        }
    }
}

fn validate_text(text: &str, index: usize) -> Result<(), String> {
    if text.is_empty() {
        return Err(format!("input[{index}] must not be an empty string"));
    }
    Ok(())
}

fn parse_token_array(items: &[serde_json::Value], input_index: usize) -> Result<Vec<u32>, String> {
    if items.is_empty() {
        return Err(format!(
            "input[{input_index}] must contain at least one token ID"
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(token_index, value)| {
            let token = value.as_u64().ok_or_else(|| {
                format!(
                    "input[{input_index}][{token_index}] must be a non-negative integer token ID"
                )
            })?;
            u32::try_from(token).map_err(|_| {
                format!("input[{input_index}][{token_index}] exceeds the supported token-ID range")
            })
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileFingerprint {
    len: u64,
    modified: Option<SystemTime>,
}

impl FileFingerprint {
    fn read(path: &Path) -> Result<Self, String> {
        let metadata = std::fs::metadata(path)
            .map_err(|error| format!("Failed to inspect {}: {error}", path.display()))?;
        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenizerFingerprint {
    tokenizer: FileFingerprint,
    config: Option<FileFingerprint>,
}

struct EmbeddingTokenizer {
    tokenizer: Tokenizer,
    pad_token_id: Option<u32>,
    max_length: Option<usize>,
}

struct CachedTokenizer {
    fingerprint: TokenizerFingerprint,
    tokenizer: Arc<EmbeddingTokenizer>,
}

#[derive(Default)]
struct TokenizerCache {
    entries: RwLock<HashMap<PathBuf, CachedTokenizer>>,
}

impl TokenizerCache {
    fn load(&self, model_path: &Path) -> Result<Arc<EmbeddingTokenizer>, String> {
        let tokenizer_path = find_model_asset(model_path, "tokenizer.json").ok_or_else(|| {
            format!(
                "Embedding model '{}' does not include tokenizer.json; provide token IDs or rebuild the package with tokenizer assets",
                model_path.display()
            )
        })?;
        let config_path = find_model_asset(model_path, "tokenizer_config.json");
        let fingerprint = TokenizerFingerprint {
            tokenizer: FileFingerprint::read(&tokenizer_path)?,
            config: config_path
                .as_deref()
                .map(FileFingerprint::read)
                .transpose()?,
        };

        if let Some(cached) = self.entries.read().get(&tokenizer_path) {
            if cached.fingerprint == fingerprint {
                return Ok(cached.tokenizer.clone());
            }
        }

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            format!(
                "Failed to load embedding tokenizer {}: {error}",
                tokenizer_path.display()
            )
        })?;
        let tokenizer_padding_id = tokenizer.get_padding().map(|padding| padding.pad_id);
        let tokenizer_max_length = tokenizer
            .get_truncation()
            .map(|truncation| truncation.max_length);
        let config = config_path
            .as_deref()
            .map(read_tokenizer_config)
            .transpose()?
            .flatten();

        let config_pad_id = config
            .as_ref()
            .and_then(|config| config.pad_token.as_deref())
            .and_then(|token| tokenizer.token_to_id(token));
        let fallback_pad_id = ["[PAD]", "<pad>"]
            .iter()
            .find_map(|token| tokenizer.token_to_id(token));
        let pad_token_id = tokenizer_padding_id.or(config_pad_id).or(fallback_pad_id);
        let config_max_length = config.and_then(|config| config.max_length);
        let max_length = min_option(tokenizer_max_length, config_max_length);

        // The endpoint validates limits explicitly. Built-in tokenizer
        // truncation would silently change the caller's input, and built-in
        // padding would make exact token accounting impossible.
        tokenizer
            .with_truncation(None)
            .map_err(|error| format!("Failed to disable tokenizer truncation: {error}"))?;
        tokenizer.with_padding(None);

        let loaded = Arc::new(EmbeddingTokenizer {
            tokenizer,
            pad_token_id,
            max_length,
        });
        self.entries.write().insert(
            tokenizer_path,
            CachedTokenizer {
                fingerprint,
                tokenizer: loaded.clone(),
            },
        );
        Ok(loaded)
    }
}

struct TokenizerConfig {
    pad_token: Option<String>,
    max_length: Option<usize>,
}

fn read_tokenizer_config(path: &Path) -> Result<Option<TokenizerConfig>, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Invalid tokenizer config {}: {error}", path.display()))?;
    let pad_token = value.get("pad_token").and_then(|value| {
        value.as_str().map(str::to_string).or_else(|| {
            value
                .get("content")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
    });
    // Hugging Face uses enormous sentinel values for "unbounded". Ignore
    // those and retain the endpoint's explicit 8192-token ceiling.
    let max_length = value
        .get("model_max_length")
        .and_then(serde_json::Value::as_u64)
        .and_then(|length| usize::try_from(length).ok())
        .filter(|length| *length > 0 && *length <= 1_000_000);
    Ok(Some(TokenizerConfig {
        pad_token,
        max_length,
    }))
}

fn min_option(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum AuxiliaryInputKind {
    AttentionMask,
    TokenTypeIds,
    PositionIds,
}

#[derive(Debug, Clone)]
struct AuxiliaryInput {
    name: String,
    dtype: TensorDtype,
    kind: AuxiliaryInputKind,
}

#[derive(Debug, Clone)]
struct EmbeddingModelContract {
    input_dtype: TensorDtype,
    fixed_sequence_length: Option<usize>,
    native_output_dimensions: Option<usize>,
    auxiliary_inputs: Vec<AuxiliaryInput>,
}

impl EmbeddingModelContract {
    fn from_engine_info(info: &EngineModelInfo) -> Result<Self, String> {
        let Some(primary_name) = info.input_names.first() else {
            return Err("Embedding model reports no input tensors".to_string());
        };
        let primary_name_lower = primary_name.to_ascii_lowercase();
        if !primary_name_lower.contains("input_ids") && primary_name_lower != "input" {
            return Err(format!(
                "Embedding model's first input is '{primary_name}', expected an input_ids tensor"
            ));
        }

        let primary_dtype = model_input_dtype(info, 0)?;
        ensure_integer_dtype(primary_dtype, primary_name)?;
        let primary_shape = info.input_shapes.first().map(Vec::as_slice).unwrap_or(&[]);
        validate_rank_two(primary_name, primary_shape)?;
        if primary_shape
            .first()
            .copied()
            .is_some_and(|batch| batch > 0 && batch != 1)
        {
            return Err(format!(
                "Embedding model requires a fixed batch size of {}; the OpenAI adapter executes one input per scheduled request",
                primary_shape[0]
            ));
        }
        let fixed_sequence_length = primary_shape
            .get(1)
            .copied()
            .filter(|dimension| *dimension > 0)
            .and_then(|dimension| usize::try_from(dimension).ok());

        let mut auxiliary_inputs = Vec::new();
        for (index, name) in info.input_names.iter().enumerate().skip(1) {
            let lower = name.to_ascii_lowercase();
            let kind = if lower.contains("attention_mask") {
                AuxiliaryInputKind::AttentionMask
            } else if lower.contains("token_type_ids") || lower.contains("segment_ids") {
                AuxiliaryInputKind::TokenTypeIds
            } else if lower.contains("position_ids") {
                AuxiliaryInputKind::PositionIds
            } else {
                return Err(format!(
                    "Embedding model input '{name}' is not supported by the text adapter"
                ));
            };
            let dtype = model_input_dtype(info, index)?;
            ensure_integer_dtype(dtype, name)?;
            let shape = info
                .input_shapes
                .get(index)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            validate_rank_two(name, shape)?;
            if shape
                .first()
                .copied()
                .is_some_and(|batch| batch > 0 && batch != 1)
            {
                return Err(format!(
                    "Embedding model input '{name}' requires unsupported fixed batch size {}",
                    shape[0]
                ));
            }
            if let (Some(primary), Some(auxiliary)) = (
                fixed_sequence_length,
                shape
                    .get(1)
                    .copied()
                    .filter(|dimension| *dimension > 0)
                    .and_then(|dimension| usize::try_from(dimension).ok()),
            ) {
                if primary != auxiliary {
                    return Err(format!(
                        "Embedding model inputs disagree on fixed sequence length ({primary} vs {auxiliary} for '{name}')"
                    ));
                }
            }
            auxiliary_inputs.push(AuxiliaryInput {
                name: name.clone(),
                dtype,
                kind,
            });
        }

        if let Some(dtype) = info.output_dtypes.first() {
            if TensorDtype::from_str(dtype).ok() != Some(TensorDtype::Float32) {
                return Err(format!(
                    "Embedding model output dtype '{dtype}' is unsupported; expected float32"
                ));
            }
        }
        let native_output_dimensions = info
            .output_shapes
            .first()
            .and_then(|shape| shape.last())
            .copied()
            .filter(|dimension| *dimension > 0)
            .and_then(|dimension| usize::try_from(dimension).ok());

        Ok(Self {
            input_dtype: primary_dtype,
            fixed_sequence_length,
            native_output_dimensions,
            auxiliary_inputs,
        })
    }
}

fn model_input_dtype(info: &EngineModelInfo, index: usize) -> Result<TensorDtype, String> {
    let Some(dtype) = info.input_dtypes.get(index) else {
        // Older engine implementations did not report dtypes. ONNX text
        // encoders overwhelmingly use int64, which is the backend's own
        // auto-fill default for masks and position IDs.
        return Ok(TensorDtype::Int64);
    };
    TensorDtype::from_str(dtype).map_err(|_| {
        format!(
            "Embedding model input '{}' reports unsupported dtype '{dtype}'",
            info.input_names
                .get(index)
                .map(String::as_str)
                .unwrap_or("unknown")
        )
    })
}

fn ensure_integer_dtype(dtype: TensorDtype, name: &str) -> Result<(), String> {
    if matches!(dtype, TensorDtype::Int32 | TensorDtype::Int64) {
        Ok(())
    } else {
        Err(format!(
            "Embedding model input '{name}' uses {dtype}; only int32 and int64 token tensors are supported"
        ))
    }
}

fn validate_rank_two(name: &str, shape: &[i64]) -> Result<(), String> {
    if !shape.is_empty() && shape.len() != 2 {
        return Err(format!(
            "Embedding model input '{name}' has shape {shape:?}; expected [batch, sequence]"
        ));
    }
    Ok(())
}

struct PreparedInputs {
    token_ids: Vec<Vec<u32>>,
    pad_token_id: Option<u32>,
    tokenizer_max_length: Option<usize>,
}

fn tokenize_text_inputs(
    cache: &TokenizerCache,
    model_path: &Path,
    texts: Vec<String>,
) -> Result<PreparedInputs, String> {
    let tokenizer = cache.load(model_path)?;
    let token_ids = texts
        .iter()
        .enumerate()
        .map(|(index, text)| {
            tokenizer
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|error| format!("Failed to tokenize input[{index}]: {error}"))
                .and_then(|encoding| {
                    let ids = encoding.get_ids().to_vec();
                    if ids.is_empty() {
                        Err(format!("input[{index}] produced no tokens"))
                    } else {
                        Ok(ids)
                    }
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PreparedInputs {
        token_ids,
        pad_token_id: tokenizer.pad_token_id,
        tokenizer_max_length: tokenizer.max_length,
    })
}

fn validate_token_limits(
    inputs: &[Vec<u32>],
    tokenizer_max_length: Option<usize>,
    fixed_sequence_length: Option<usize>,
) -> Result<u64, String> {
    let per_input_limit = [
        Some(MAX_TOKENS_PER_INPUT),
        tokenizer_max_length,
        fixed_sequence_length,
    ]
    .into_iter()
    .flatten()
    .min()
    .expect("the API limit is always present");

    let mut total = 0usize;
    for (index, tokens) in inputs.iter().enumerate() {
        if tokens.is_empty() {
            return Err(format!("input[{index}] must contain at least one token ID"));
        }
        if tokens.len() > per_input_limit {
            return Err(format!(
                "input[{index}] contains {} tokens; this model accepts at most {per_input_limit}",
                tokens.len()
            ));
        }
        total = total
            .checked_add(tokens.len())
            .ok_or_else(|| "input token count overflow".to_string())?;
    }
    if total > MAX_TOKENS_PER_REQUEST {
        return Err(format!(
            "input contains {total} tokens in total; at most {MAX_TOKENS_PER_REQUEST} are supported"
        ));
    }
    Ok(total as u64)
}

fn build_inference_requests(
    inputs: Vec<Vec<u32>>,
    pad_token_id: Option<u32>,
    contract: &EmbeddingModelContract,
    session_id: Option<&str>,
) -> Result<Vec<InferenceRequest>, String> {
    inputs
        .into_iter()
        .enumerate()
        .map(|(index, token_ids)| {
            let actual_length = token_ids.len();
            let target_length = contract.fixed_sequence_length.unwrap_or(actual_length);
            let mut padded_ids = token_ids;
            if target_length > actual_length {
                if !contract
                    .auxiliary_inputs
                    .iter()
                    .any(|input| matches!(input.kind, AuxiliaryInputKind::AttentionMask))
                {
                    return Err(format!(
                        "input[{index}] requires padding to the model's fixed sequence length, but the model exposes no attention_mask input"
                    ));
                }
                let pad_token_id = pad_token_id.ok_or_else(|| {
                    format!(
                        "input[{index}] has {actual_length} tokens but the model requires {target_length}; its tokenizer does not define a padding token"
                    )
                })?;
                padded_ids.resize(target_length, pad_token_id);
            }

            let shape = vec![1, target_length as i64];
            let input = integer_packet(shape.clone(), contract.input_dtype, &padded_ids)?;
            let mut request = InferenceRequest::new(input).with_metadata(RequestMetadata {
                request_id: Some(embedding_request_id()),
                ..RequestMetadata::default()
            });
            if let Some(session_id) = session_id {
                request.session_id = Some(session_id.to_string());
            }

            for auxiliary in &contract.auxiliary_inputs {
                let values = match auxiliary.kind {
                    AuxiliaryInputKind::AttentionMask => (0..target_length)
                        .map(|position| u32::from(position < actual_length))
                        .collect(),
                    AuxiliaryInputKind::TokenTypeIds => vec![0; target_length],
                    AuxiliaryInputKind::PositionIds => (0..target_length)
                        .map(|position| {
                            if position < actual_length {
                                position as u32
                            } else {
                                0
                            }
                        })
                        .collect(),
                };
                request.additional_inputs.push(NamedTensor {
                    name: auxiliary.name.clone(),
                    tensor: integer_packet(shape.clone(), auxiliary.dtype, &values)?,
                });
            }
            Ok(request)
        })
        .collect()
}

fn integer_packet(
    shape: Vec<i64>,
    dtype: TensorDtype,
    values: &[u32],
) -> Result<BinaryTensorPacket, String> {
    let mut data = Vec::with_capacity(values.len().saturating_mul(dtype.size_bytes()));
    match dtype {
        TensorDtype::Int64 => {
            for value in values {
                data.extend_from_slice(&i64::from(*value).to_ne_bytes());
            }
        }
        TensorDtype::Int32 => {
            for value in values {
                let value = i32::try_from(*value).map_err(|_| {
                    format!("token ID {value} does not fit the model's int32 input dtype")
                })?;
                data.extend_from_slice(&value.to_ne_bytes());
            }
        }
        _ => return Err(format!("Unsupported embedding input dtype {dtype}")),
    }
    BinaryTensorPacket::new(shape, dtype, data)
        .map_err(|error| format!("Failed to build embedding input tensor: {error}"))
}

fn embedding_request_id() -> String {
    let mut bytes = [0u8; 12];
    OsRng.fill_bytes(&mut bytes);
    format!("embd_{}", URL_SAFE_NO_PAD.encode(bytes))
}

fn decode_embedding_output(
    packet: BinaryTensorPacket,
    expected_batch: usize,
) -> Result<Vec<Vec<f32>>, String> {
    if packet.dtype != TensorDtype::Float32 {
        return Err(format!(
            "Embedding backend returned {}, expected float32",
            packet.dtype
        ));
    }
    let [batch, dimensions] = packet.shape.as_slice() else {
        return Err(format!(
            "Embedding backend returned shape {:?}, expected [batch, dimensions]",
            packet.shape
        ));
    };
    let batch = usize::try_from(*batch)
        .ok()
        .filter(|batch| *batch > 0)
        .ok_or_else(|| format!("Embedding backend returned invalid batch dimension {batch}"))?;
    let dimensions = usize::try_from(*dimensions)
        .ok()
        .filter(|dimensions| *dimensions > 0)
        .ok_or_else(|| {
            format!("Embedding backend returned invalid output dimension {dimensions}")
        })?;
    if batch != expected_batch {
        return Err(format!(
            "Embedding backend returned {batch} rows, expected {expected_batch}"
        ));
    }
    let expected_values = batch
        .checked_mul(dimensions)
        .ok_or_else(|| "Embedding output shape overflow".to_string())?;
    let expected_bytes = expected_values
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Embedding output byte length overflow".to_string())?;
    if packet.data.len() != expected_bytes {
        return Err(format!(
            "Embedding backend returned {} bytes but shape {:?} requires {expected_bytes}",
            packet.data.len(),
            packet.shape
        ));
    }

    let values = packet
        .data
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("Embedding backend returned a non-finite value".to_string());
    }
    Ok(values
        .chunks_exact(dimensions)
        .map(<[f32]>::to_vec)
        .collect())
}

fn apply_dimensions(values: &mut Vec<f32>, dimensions: Option<usize>) -> Result<(), String> {
    let Some(dimensions) = dimensions else {
        return Ok(());
    };
    if dimensions == 0 {
        return Err("dimensions must be at least 1".to_string());
    }
    if dimensions > values.len() {
        return Err(format!(
            "dimensions={dimensions} exceeds the model's native embedding size of {}",
            values.len()
        ));
    }
    if dimensions == values.len() {
        return Ok(());
    }

    values.truncate(dimensions);
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in values {
            *value /= norm;
        }
    }
    Ok(())
}

fn encoded_embedding(values: &[f32], encoding: EmbeddingEncoding) -> serde_json::Value {
    match encoding {
        EmbeddingEncoding::Float => serde_json::json!(values),
        EmbeddingEncoding::Base64 => {
            let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            serde_json::Value::String(BASE64.encode(bytes))
        }
    }
}

fn supports_embeddings(model: &ModelInfo) -> bool {
    model
        .task
        .as_deref()
        .is_some_and(|task| task.eq_ignore_ascii_case("embed"))
        || (model.task.is_none()
            && model
                .model_type
                .as_deref()
                .is_some_and(|kind| kind.eq_ignore_ascii_case("embedding")))
}

#[allow(clippy::too_many_arguments)]
async fn handle_embeddings(
    body: warp::hyper::body::Bytes,
    session_header: Option<String>,
    authorization: Option<String>,
    models: &Arc<ModelManager>,
    inference: &Arc<InferenceService>,
    tokenizer_cache: &Arc<TokenizerCache>,
    log_sensitive_ids: bool,
) -> warp::reply::Response {
    use warp::http::StatusCode;

    let embedding_request: CreateEmbeddingRequest = match serde_json::from_slice(body.as_ref()) {
        Ok(request) => request,
        Err(error) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("Invalid embeddings payload: {error}"),
                "invalid_request_error",
            );
        }
    };
    let encoding = match EmbeddingEncoding::parse(embedding_request.encoding_format.as_deref()) {
        Ok(encoding) => encoding,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
        }
    };
    if embedding_request.dimensions == Some(0) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "dimensions must be at least 1",
            "invalid_request_error",
        );
    }
    let parsed_input = match ParsedEmbeddingInput::parse(&embedding_request.input) {
        Ok(input) => input,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
        }
    };

    let resolved = match resolve_model(models, &embedding_request.model) {
        Ok(resolved) => resolved,
        Err(message) => {
            return openai_error(StatusCode::NOT_FOUND, message, "invalid_request_error");
        }
    };
    let Some(model) = models.registry().get(resolved.id) else {
        return openai_error(
            StatusCode::NOT_FOUND,
            format!("The model '{}' does not exist", embedding_request.model),
            "invalid_request_error",
        );
    };
    if !supports_embeddings(&model) {
        let task = model.task.as_deref().unwrap_or("unspecified");
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "The model '{}' cannot serve embeddings (task={task}); load an ONNX model packaged with task=embed",
                resolved.name
            ),
            "invalid_request_error",
        );
    }

    let Some(pool) = models.pool(resolved.id) else {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("The model '{}' is not currently loaded", resolved.name),
            "server_error",
        );
    };
    let Some(engine_info) = pool.model_info() else {
        return openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "The embedding model '{}' did not report its tensor contract",
                resolved.name
            ),
            "server_error",
        );
    };
    let contract = match EmbeddingModelContract::from_engine_info(&engine_info) {
        Ok(contract) => contract,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
        }
    };
    if let (Some(requested), Some(native)) = (
        embedding_request.dimensions,
        contract.native_output_dimensions,
    ) {
        if requested > native {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "dimensions={requested} exceeds the model's native embedding size of {native}"
                ),
                "invalid_request_error",
            );
        }
    }

    let prepared = match parsed_input {
        ParsedEmbeddingInput::Tokens(token_ids) => PreparedInputs {
            token_ids,
            pad_token_id: None,
            tokenizer_max_length: None,
        },
        ParsedEmbeddingInput::Text(texts) => {
            let cache = tokenizer_cache.clone();
            let model_path = PathBuf::from(&model.model_path);
            match tokio::task::spawn_blocking(move || {
                tokenize_text_inputs(&cache, &model_path, texts)
            })
            .await
            {
                Ok(Ok(prepared)) => prepared,
                Ok(Err(message)) => {
                    return openai_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        message,
                        "server_error",
                    );
                }
                Err(error) => {
                    return openai_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Embedding tokenization task failed: {error}"),
                        "server_error",
                    );
                }
            }
        }
    };
    let prompt_tokens = match validate_token_limits(
        &prepared.token_ids,
        prepared.tokenizer_max_length,
        contract.fixed_sequence_length,
    ) {
        Ok(tokens) => tokens,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
        }
    };

    let client_session_id = session_header
        .map(|session| session.trim().to_string())
        .filter(|session| !session.is_empty())
        .or_else(|| {
            embedding_request
                .user
                .as_ref()
                .map(|user| user.trim().to_string())
                .filter(|user| !user.is_empty())
        });
    let session_id =
        scope_session_id_for_authorization(client_session_id.as_deref(), authorization.as_deref());
    let requests = match build_inference_requests(
        prepared.token_ids,
        prepared.pad_token_id,
        &contract,
        session_id.as_deref(),
    ) {
        Ok(requests) => requests,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
        }
    };
    let session_id_for_log = redact_identifier_for_logs(
        client_session_id.as_deref().unwrap_or("-"),
        log_sensitive_ids,
    );

    let mut embeddings = Vec::with_capacity(requests.len());
    for request in requests {
        let priority = inference.priority_for_request(&request);
        let output = match inference.infer(resolved.id, request, priority, false).await {
            Ok(output) => output,
            Err(error) => {
                let status = status_code_for_engine_error(&error);
                if status == StatusCode::INTERNAL_SERVER_ERROR {
                    log::error!(
                        "Embedding failed: model_id={} session_id={} status={} error={}",
                        resolved.id,
                        session_id_for_log,
                        status.as_u16(),
                        error
                    );
                } else {
                    log::warn!(
                        "Embedding rejected: model_id={} session_id={} status={} error={}",
                        resolved.id,
                        session_id_for_log,
                        status.as_u16(),
                        error
                    );
                }
                let error_type = if status == StatusCode::BAD_REQUEST {
                    "invalid_request_error"
                } else {
                    "server_error"
                };
                return openai_error(status, error.to_string(), error_type);
            }
        };
        let mut rows = match decode_embedding_output(output, 1) {
            Ok(rows) => rows,
            Err(message) => {
                log::error!(
                    "Invalid embedding output: model_id={} error={}",
                    resolved.id,
                    message
                );
                return openai_error(StatusCode::INTERNAL_SERVER_ERROR, message, "server_error");
            }
        };
        let mut vector = rows.pop().expect("one row was validated");
        if let Err(message) = apply_dimensions(&mut vector, embedding_request.dimensions) {
            return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
        }
        embeddings.push(vector);
    }

    let data = embeddings
        .iter()
        .enumerate()
        .map(|(index, embedding)| {
            serde_json::json!({
                "object": "embedding",
                "embedding": encoded_embedding(embedding, encoding),
                "index": index,
            })
        })
        .collect::<Vec<_>>();
    reply_into_response(warp::reply::json(&serde_json::json!({
        "object": "list",
        "data": data,
        "model": resolved.name,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_openai_input_shapes() {
        assert_eq!(
            ParsedEmbeddingInput::parse(&serde_json::json!("hello")).unwrap(),
            ParsedEmbeddingInput::Text(vec!["hello".to_string()])
        );
        assert_eq!(
            ParsedEmbeddingInput::parse(&serde_json::json!(["hello", "world"])).unwrap(),
            ParsedEmbeddingInput::Text(vec!["hello".to_string(), "world".to_string()])
        );
        assert_eq!(
            ParsedEmbeddingInput::parse(&serde_json::json!([1, 2, 3])).unwrap(),
            ParsedEmbeddingInput::Tokens(vec![vec![1, 2, 3]])
        );
        assert_eq!(
            ParsedEmbeddingInput::parse(&serde_json::json!([[1, 2], [3]])).unwrap(),
            ParsedEmbeddingInput::Tokens(vec![vec![1, 2], vec![3]])
        );
    }

    #[test]
    fn rejects_mixed_and_empty_inputs() {
        assert!(ParsedEmbeddingInput::parse(&serde_json::json!([])).is_err());
        assert!(ParsedEmbeddingInput::parse(&serde_json::json!(["text", 1])).is_err());
        assert!(ParsedEmbeddingInput::parse(&serde_json::json!([[1], []])).is_err());
        assert!(ParsedEmbeddingInput::parse(&serde_json::json!([-1])).is_err());
    }

    #[test]
    fn dimensions_truncate_and_renormalize() {
        let mut values = vec![3.0, 4.0, 12.0];
        apply_dimensions(&mut values, Some(2)).unwrap();
        assert_eq!(values, vec![0.6, 0.8]);
    }

    #[test]
    fn base64_is_little_endian_float32() {
        let encoded = encoded_embedding(&[1.0, -2.5], EmbeddingEncoding::Base64)
            .as_str()
            .unwrap()
            .to_string();
        let bytes = BASE64.decode(encoded).unwrap();
        assert_eq!(&bytes[..4], &1.0f32.to_le_bytes());
        assert_eq!(&bytes[4..], &(-2.5f32).to_le_bytes());
    }

    #[test]
    fn fixed_sequence_inputs_are_padded_and_masked() {
        let contract = EmbeddingModelContract {
            input_dtype: TensorDtype::Int64,
            fixed_sequence_length: Some(4),
            native_output_dimensions: Some(3),
            auxiliary_inputs: vec![AuxiliaryInput {
                name: "attention_mask".to_string(),
                dtype: TensorDtype::Int64,
                kind: AuxiliaryInputKind::AttentionMask,
            }],
        };

        let requests =
            build_inference_requests(vec![vec![5, 6]], Some(0), &contract, Some("session"))
                .unwrap();
        let request = &requests[0];
        let decode_i64 = |packet: &BinaryTensorPacket| {
            packet
                .data
                .chunks_exact(8)
                .map(|bytes| i64::from_ne_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };

        assert_eq!(request.input.shape, vec![1, 4]);
        assert_eq!(decode_i64(&request.input), vec![5, 6, 0, 0]);
        assert_eq!(request.session_id.as_deref(), Some("session"));
        assert_eq!(
            decode_i64(&request.additional_inputs[0].tensor),
            vec![1, 1, 0, 0]
        );
    }

    #[test]
    fn token_limits_are_checked_before_inference() {
        let too_long = vec![vec![0; MAX_TOKENS_PER_INPUT + 1]];
        let error = validate_token_limits(&too_long, None, None).unwrap_err();
        assert!(error.contains("at most 8192"));
    }
}
