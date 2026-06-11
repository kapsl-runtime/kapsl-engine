use async_stream::stream;
use async_trait::async_trait;
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, EngineMetrics, EngineModelInfo, EngineStream,
    InferenceRequest, TensorDtype,
};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "gguf")]
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    sampling::LlamaSampler,
};

/// Static weak cache: model path → shared inner.
/// A `Weak` reference is stored so that when all replicas using the model are
/// dropped, the `Arc` refcount falls to zero and memory is freed.
#[cfg(feature = "gguf")]
static GGUF_MODEL_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, std::sync::Weak<GgufInner>>>,
> = std::sync::OnceLock::new();

#[cfg(feature = "gguf")]
fn gguf_model_cache() -> &'static std::sync::Mutex<
    std::collections::HashMap<std::path::PathBuf, std::sync::Weak<GgufInner>>,
> {
    GGUF_MODEL_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// GGUF inference backend backed by llama.cpp via the `llama-cpp-2` crate.
///
/// The model and backend are loaded once and reused across requests. A new
/// `LlamaContext` is created per inference call because `LlamaContext` is not
/// `Send` and cannot be shared across threads.
pub struct GgufBackend {
    #[cfg(feature = "gguf")]
    inner: Option<Arc<GgufInner>>,
    metrics: EngineMetrics,
}

#[cfg(feature = "gguf")]
struct GgufInner {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    /// Effective context length (capped at 4096 for safety).
    max_context: u32,
}

impl GgufBackend {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "gguf")]
            inner: None,
            metrics: EngineMetrics::new(),
        }
    }

    /// Extract the prompt string from an `InferenceRequest`.
    /// The input tensor stores UTF-8 encoded text as raw bytes.
    fn extract_prompt(request: &InferenceRequest) -> Result<String, EngineError> {
        String::from_utf8(request.input.data.clone())
            .map_err(|e| EngineError::invalid_input(format!("Input is not valid UTF-8: {e}")))
    }

    fn max_new_tokens(request: &InferenceRequest) -> i32 {
        request
            .metadata
            .as_ref()
            .and_then(|m| m.max_new_tokens)
            .unwrap_or(512) as i32
    }
}

impl Default for GgufBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Engine impl ────────────────────────────────────────────────────────────

#[cfg(feature = "gguf")]
#[async_trait]
impl Engine for GgufBackend {
    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        let model_path_key = model_path.to_path_buf();

        // Check whether another replica has already loaded this model.
        // If so, reuse its weights — each replica still creates its own
        // LlamaContext per inference call, so inference is not serialized.
        let cached = gguf_model_cache()
            .lock()
            .unwrap()
            .get(&model_path_key)
            .and_then(|w| w.upgrade());

        let inner = if let Some(shared) = cached {
            log::info!(
                "[gguf] Reusing shared model weights for {} (replica path)",
                model_path_key.display()
            );
            shared
        } else {
            let model_path_load = model_path_key.clone();
            let (backend, model, n_ctx) = tokio::task::spawn_blocking(move || {
                let backend = LlamaBackend::init()
                    .map_err(|e| EngineError::backend(format!("llama backend init failed: {e}")))?;
                let params = LlamaModelParams::default().with_n_gpu_layers(99);
                let model = LlamaModel::load_from_file(&backend, &model_path_load, &params)
                    .map_err(|e| EngineError::backend(format!("GGUF load failed: {e}")))?;
                let n_ctx = model.n_ctx_train();
                Ok::<_, EngineError>((backend, model, n_ctx))
            })
            .await
            .map_err(|e| EngineError::backend(format!("spawn_blocking join error: {e}")))??;

            let max_context = (n_ctx as u32).min(4096);
            log::info!(
                "[gguf] Model loaded; train ctx={}, effective ctx={}",
                n_ctx,
                max_context
            );
            let arc_inner = Arc::new(GgufInner {
                backend: Arc::new(backend),
                model: Arc::new(model),
                max_context,
            });
            // Register in cache so subsequent replicas can reuse this graph.
            gguf_model_cache()
                .lock()
                .unwrap()
                .insert(model_path_key, Arc::downgrade(&arc_inner));
            arc_inner
        };

        self.inner = Some(inner);
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let inner = self.inner.as_ref().ok_or(EngineError::ModelNotLoaded)?;
        let prompt = Self::extract_prompt(request)?;
        let max_tokens = Self::max_new_tokens(request);

        let model = Arc::clone(&inner.model);
        let backend = Arc::clone(&inner.backend);
        let max_context = inner.max_context;

        // llama-cpp-2 types are not Send, so we run inference on a dedicated OS thread
        // and join synchronously. `std::thread::scope` cannot be used with async, but
        // since `infer()` is a synchronous method this is fine.
        let (tx, rx) = std::sync::mpsc::channel::<Result<String, EngineError>>();
        std::thread::spawn(move || {
            let result = run_generation_sync(&model, &backend, &prompt, max_tokens, max_context);
            let _ = tx.send(result);
        });

        let text = rx
            .recv()
            .map_err(|_| EngineError::backend("inference thread panicked"))??;

        let data = text.into_bytes();
        let len = data.len() as i64;
        BinaryTensorPacket::new(vec![1, len], TensorDtype::Uint8, data)
            .map_err(|e| EngineError::backend(format!("Failed to build output packet: {e}")))
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        let inner = match self.inner.as_ref() {
            Some(i) => i,
            None => {
                return Box::pin(stream! {
                    yield Err(EngineError::ModelNotLoaded);
                });
            }
        };

        let prompt = match Self::extract_prompt(request) {
            Ok(p) => p,
            Err(e) => {
                return Box::pin(stream! {
                    yield Err(e);
                });
            }
        };

        let max_tokens = Self::max_new_tokens(request);
        let model = Arc::clone(&inner.model);
        let backend = Arc::clone(&inner.backend);
        let max_context = inner.max_context;

        // Channel for streaming tokens from the llama.cpp OS thread
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<String, EngineError>>(64);

        std::thread::spawn(move || {
            let ctx_params = match NonZeroU32::new(max_context) {
                Some(n) => LlamaContextParams::default().with_n_ctx(Some(n)),
                None => LlamaContextParams::default(),
            };

            let mut ctx = match model.new_context(&backend, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.blocking_send(Err(EngineError::backend(format!(
                        "Failed to create llama context: {e}"
                    ))));
                    return;
                }
            };

            // Tokenize prompt
            let tokens = match model.str_to_token(&prompt, AddBos::Always) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.blocking_send(Err(EngineError::backend(format!(
                        "Tokenization failed: {e}"
                    ))));
                    return;
                }
            };

            // Fill initial batch with prompt tokens
            let mut batch = LlamaBatch::new(tokens.len() + max_tokens as usize, 1);
            let n_prompt = tokens.len() as i32;
            for (i, &token) in tokens.iter().enumerate() {
                let is_last = i == tokens.len() - 1;
                if let Err(e) = batch.add(token, i as i32, &[0], is_last) {
                    let _ = tx
                        .blocking_send(Err(EngineError::backend(format!("Batch add error: {e}"))));
                    return;
                }
            }

            if let Err(e) = ctx.decode(&mut batch) {
                let _ = tx.blocking_send(Err(EngineError::backend(format!(
                    "Prefill decode failed: {e}"
                ))));
                return;
            }

            let mut n_cur = n_prompt;

            // Auto-detect EOS token id
            let eos_token = model.token_eos();

            // Decode loop
            for _ in 0..max_tokens {
                let next_token = LlamaSampler::greedy().sample(&ctx, batch.n_tokens() - 1);

                if next_token == eos_token {
                    break;
                }

                // Decode token to text
                let piece = match model.token_to_str(next_token, Special::Tokenize) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(EngineError::backend(format!(
                            "Token decode error: {e}"
                        ))));
                        return;
                    }
                };

                if tx.blocking_send(Ok(piece)).is_err() {
                    // Receiver dropped — client cancelled
                    return;
                }

                // Prepare next batch with just the new token
                batch.clear();
                if let Err(e) = batch.add(next_token, n_cur, &[0], true) {
                    let _ = tx
                        .blocking_send(Err(EngineError::backend(format!("Batch add error: {e}"))));
                    return;
                }

                if let Err(e) = ctx.decode(&mut batch) {
                    let _ = tx.blocking_send(Err(EngineError::backend(format!(
                        "Decode step failed: {e}"
                    ))));
                    return;
                }

                n_cur += 1;
            }
        });

        Box::pin(stream! {
            while let Some(result) = rx.recv().await {
                let piece = result?;
                let data = piece.into_bytes();
                let len = data.len() as i64;
                yield BinaryTensorPacket::new(vec![1, len], TensorDtype::Uint8, data)
                    .map_err(|e| EngineError::backend(format!("Output packet error: {e}")));
            }
        })
    }

    fn unload(&mut self) {
        self.inner = None;
        log::info!("[gguf] Model unloaded");
    }

    fn metrics(&self) -> EngineMetrics {
        self.metrics.clone()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        if self.inner.is_some() {
            Ok(())
        } else {
            Err(EngineError::ModelNotLoaded)
        }
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        let _inner = self.inner.as_ref()?;
        Some(EngineModelInfo {
            input_names: vec!["text".to_string()],
            output_names: vec!["text".to_string()],
            input_shapes: vec![vec![-1]],
            output_shapes: vec![vec![-1]],
            input_dtypes: vec!["uint8".to_string()],
            output_dtypes: vec!["uint8".to_string()],
            framework: Some("gguf".to_string()),
            model_version: None,
            peak_concurrency: Some(1),
        })
    }
}

/// Blocking (non-streaming) generation — runs the full decode loop and returns the
/// complete output string. Called from a dedicated OS thread inside `infer()`.
#[cfg(feature = "gguf")]
fn run_generation_sync(
    model: &LlamaModel,
    backend: &LlamaBackend,
    prompt: &str,
    max_tokens: i32,
    max_context: u32,
) -> Result<String, EngineError> {
    let ctx_params = match NonZeroU32::new(max_context) {
        Some(n) => LlamaContextParams::default().with_n_ctx(Some(n)),
        None => LlamaContextParams::default(),
    };

    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| EngineError::backend(format!("Failed to create llama context: {e}")))?;

    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|e| EngineError::backend(format!("Tokenization failed: {e}")))?;

    let n_prompt = tokens.len() as i32;
    let mut batch = LlamaBatch::new(tokens.len() + max_tokens as usize, 1);
    for (i, &token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch
            .add(token, i as i32, &[0], is_last)
            .map_err(|e| EngineError::backend(format!("Batch add error: {e}")))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| EngineError::backend(format!("Prefill decode failed: {e}")))?;

    let mut output = String::new();
    let mut n_cur = n_prompt;
    let eos_token = model.token_eos();

    for _ in 0..max_tokens {
        let next_token = LlamaSampler::greedy().sample(&ctx, batch.n_tokens() - 1);

        if next_token == eos_token {
            break;
        }

        let piece = model
            .token_to_str(next_token, Special::Tokenize)
            .map_err(|e| EngineError::backend(format!("Token decode error: {e}")))?;
        output.push_str(&piece);

        batch.clear();
        batch
            .add(next_token, n_cur, &[0], true)
            .map_err(|e| EngineError::backend(format!("Batch add error: {e}")))?;
        ctx.decode(&mut batch)
            .map_err(|e| EngineError::backend(format!("Decode step failed: {e}")))?;

        n_cur += 1;
    }

    Ok(output)
}

// ─── Stub impl when gguf feature is disabled ────────────────────────────────

#[cfg(not(feature = "gguf"))]
#[async_trait]
impl Engine for GgufBackend {
    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        Err(EngineError::backend(
            "GGUF support not compiled in (enable the 'gguf' feature)",
        ))
    }

    fn infer(&self, _request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        Err(EngineError::backend(
            "GGUF support not compiled in (enable the 'gguf' feature)",
        ))
    }

    fn infer_stream(&self, _request: &InferenceRequest) -> EngineStream {
        Box::pin(stream! {
            yield Err(EngineError::backend(
                "GGUF support not compiled in (enable the 'gguf' feature)",
            ));
        })
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        self.metrics.clone()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Err(EngineError::backend(
            "GGUF support not compiled in (enable the 'gguf' feature)",
        ))
    }
}
