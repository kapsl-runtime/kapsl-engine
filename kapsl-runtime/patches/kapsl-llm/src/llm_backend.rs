use crate::block_manager::SharedBlockAllocator;
use crate::engine::LLMEngine;
use crate::llm_metrics::LLMMetrics;
use crate::model_paths::{find_model_asset, find_model_root};
use crate::scheduler::SchedulerConfig;
use crate::sequence::{SamplingParams, SequenceGroup};
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, EngineMetrics, InferenceRequest, TensorDtype,
};
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use tokio::runtime::Runtime;
use tokio::runtime::RuntimeFlavor;
use tokio::sync::{mpsc, oneshot};

fn shared_runtime() -> &'static Runtime {
    static SHARED_RT: OnceLock<Runtime> = OnceLock::new();
    SHARED_RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build shared runtime")
    })
}

/// LLM Backend that bridges the Engine trait to the asynchronous LLMEngine loop
pub struct LLMBackend {
    request_tx: RwLock<Option<mpsc::Sender<SequenceGroup>>>,
    metrics: Arc<Mutex<LLMMetrics>>,
    model_config: Arc<Mutex<ModelRuntimeConfig>>,
    provider_override: Option<String>,
    device_id_override: Option<i32>,
    device_ids_override: Option<Vec<i32>>,
    /// Optional shared block pool.  When set, the engine draws from this pool
    /// instead of a private allocator, enabling unified KV memory across models.
    shared_pool: Option<SharedBlockAllocator>,
    /// Optional per-replica cap on KV cache total_blocks. Set by the runtime
    /// when scaling to multiple replicas to divide the block budget fairly.
    kv_blocks_cap: Option<usize>,
}

#[derive(Clone)]
struct ModelRuntimeConfig {
    use_chat_template: bool,
    prompt_prefix: String,
    prompt_suffix: String,
    sampling: SamplingParams,
}

fn default_sampling_params() -> SamplingParams {
    SamplingParams {
        max_tokens: 512,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        stop_token_ids: Vec::new(),
        repetition_penalty: 1.15,
        seed: None,
    }
}

fn read_json(path: &Path) -> Option<Value> {
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn extract_bos_token(tokenizer_json: &Value) -> Option<String> {
    let post_processor = tokenizer_json.get("post_processor")?;
    if let Some(single) = post_processor.get("single").and_then(|v| v.as_array()) {
        if let Some(first) = single.first() {
            if let Some(id) = first
                .get("SpecialToken")
                .and_then(|v| v.get("id"))
                .and_then(|v| v.as_str())
            {
                return Some(id.to_string());
            }
        }
    }
    post_processor
        .get("special_tokens")
        .and_then(|v| v.as_object())
        .and_then(|map| map.keys().next())
        .map(|k| k.to_string())
}

fn extract_added_token_content_by_id(tokenizer_json: &Value, token_id: u32) -> Option<String> {
    tokenizer_json
        .get("added_tokens")
        .and_then(|v| v.as_array())
        .and_then(|tokens| {
            tokens.iter().find_map(|entry| {
                let id = entry.get("id").and_then(|v| v.as_u64())? as u32;
                if id != token_id {
                    return None;
                }
                entry
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
        })
}

fn extract_tag(template: &str, label: &str) -> Option<String> {
    let label_lower = label.to_ascii_lowercase();
    let mut search_start = 0usize;
    while let Some(found) = template[search_start..].find('<') {
        let start = search_start + found;
        let rest = &template[start..];
        let end_rel = rest.find('>')?;
        let end = start + end_rel;
        let tag = &template[start..=end];
        if tag.to_ascii_lowercase().contains(&label_lower) {
            return Some(tag.to_string());
        }
        search_start = end + 1;
    }
    None
}

fn load_model_runtime_config(model_path: &Path) -> ModelRuntimeConfig {
    let mut config = ModelRuntimeConfig {
        use_chat_template: false,
        prompt_prefix: String::new(),
        prompt_suffix: String::new(),
        sampling: default_sampling_params(),
    };

    let mut cfg_json: Option<Value> = None;
    if let Some(gen_path) = find_model_asset(model_path, "generation_config.json") {
        if let Some(gen) = read_json(&gen_path) {
            if let Some(temp) = gen.get("temperature").and_then(|v| v.as_f64()) {
                config.sampling.temperature = temp as f32;
            }
            if let Some(max_new) = gen.get("max_new_tokens").and_then(|v| v.as_u64()) {
                if max_new > 0 {
                    config.sampling.max_tokens = max_new as usize;
                }
            } else if let Some(max_len) = gen.get("max_length").and_then(|v| v.as_u64()) {
                if max_len > 0 {
                    config.sampling.max_tokens = max_len as usize;
                }
            }
            if let Some(top_p) = gen.get("top_p").and_then(|v| v.as_f64()) {
                config.sampling.top_p = top_p as f32;
            }
            if let Some(top_k) = gen.get("top_k").and_then(|v| v.as_u64()) {
                config.sampling.top_k = top_k as usize;
            }
            if let Some(penalty) = gen.get("repetition_penalty").and_then(|v| v.as_f64()) {
                config.sampling.repetition_penalty = penalty as f32;
            }
            let mut stop_ids = Vec::new();

            let mut push_stop_id = |id: u64| {
                let id = id as u32;
                if !stop_ids.contains(&id) {
                    stop_ids.push(id);
                }
            };

            if let Some(ids) = gen.get("eos_token_ids").and_then(|v| v.as_array()) {
                for id in ids {
                    if let Some(val) = id.as_u64() {
                        push_stop_id(val);
                    }
                }
            }

            // HF configs may encode eos_token_id as either a scalar or an array.
            if let Some(eos_token_id) = gen.get("eos_token_id") {
                if let Some(eos) = eos_token_id.as_u64() {
                    push_stop_id(eos);
                } else if let Some(ids) = eos_token_id.as_array() {
                    for id in ids {
                        if let Some(val) = id.as_u64() {
                            push_stop_id(val);
                        }
                    }
                }
            }
            if let Some(bos) = gen.get("bos_token_id").and_then(|v| v.as_u64()) {
                push_stop_id(bos);
            }
            if !stop_ids.is_empty() {
                config.sampling.stop_token_ids = stop_ids;
            }
        }
    }

    if let Some(cfg_path) = find_model_asset(model_path, "config.json") {
        if let Some(cfg) = read_json(&cfg_path) {
            cfg_json = Some(cfg.clone());
        }
    }

    let template_path = find_model_asset(model_path, "chat_template.jinja");
    let template_text = template_path
        .as_ref()
        .and_then(|p| fs::read_to_string(p).ok());

    let tokenizer_path = find_model_asset(model_path, "tokenizer.json");
    let tokenizer_json = tokenizer_path.as_ref().and_then(|p| read_json(p));

    if config.sampling.stop_token_ids.is_empty() {
        if let Some(cfg) = cfg_json.as_ref() {
            let mut stop_ids = Vec::new();

            let mut push_stop_id = |id: u64| {
                let id = id as u32;
                if !stop_ids.contains(&id) {
                    stop_ids.push(id);
                }
            };

            // HF config.json may encode eos_token_id as either a scalar or an array.
            if let Some(eos_token_id) = cfg.get("eos_token_id") {
                if let Some(eos) = eos_token_id.as_u64() {
                    push_stop_id(eos);
                } else if let Some(ids) = eos_token_id.as_array() {
                    for id in ids {
                        if let Some(val) = id.as_u64() {
                            push_stop_id(val);
                        }
                    }
                }
            }

            if let Some(bos) = cfg.get("bos_token_id").and_then(|v| v.as_u64()) {
                push_stop_id(bos);
            }

            if !stop_ids.is_empty() {
                config.sampling.stop_token_ids = stop_ids;
            }
        }
    }

    let use_template = template_text.is_some();

    if use_template {
        let think_suffix = template_text
            .as_deref()
            .filter(|t| t.contains("<think>"))
            .map(|_| "<think>\n".to_string())
            .unwrap_or_default();

        config.use_chat_template = true;

        let template_uses_role_header_format = template_text
            .as_deref()
            .map(|t| {
                t.contains("<|start_header_id|>")
                    && t.contains("<|end_header_id|>")
                    && t.contains("<|eot_id|>")
            })
            .unwrap_or(false);

        if template_uses_role_header_format {
            // Role-header templates use:
            //   <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>
            //   <|start_header_id|>assistant<|end_header_id|>\n\n
            let bos_token_id = cfg_json
                .as_ref()
                .and_then(|cfg| cfg.get("bos_token_id"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let bos_token = bos_token_id
                .and_then(|id| {
                    tokenizer_json
                        .as_ref()
                        .and_then(|tok| extract_added_token_content_by_id(tok, id))
                })
                .unwrap_or_default();

            config.prompt_prefix =
                format!("{}<|start_header_id|>user<|end_header_id|>\n\n", bos_token);
            config.prompt_suffix = format!(
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}",
                think_suffix
            );
        } else {
            let bos_token_id = cfg_json
                .as_ref()
                .and_then(|cfg| cfg.get("bos_token_id"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let bos_token = tokenizer_json
                .as_ref()
                .and_then(extract_bos_token)
                .or_else(|| {
                    bos_token_id.and_then(|id| {
                        tokenizer_json
                            .as_ref()
                            .and_then(|tok| extract_added_token_content_by_id(tok, id))
                    })
                })
                .unwrap_or_default();

            let user_tag = template_text
                .as_deref()
                .and_then(|t| extract_tag(t, "User"))
                .unwrap_or_else(|| "<|user|>".to_string());
            let assistant_tag = template_text
                .as_deref()
                .and_then(|t| extract_tag(t, "Assistant"))
                .unwrap_or_else(|| "<|assistant|>".to_string());

            config.prompt_prefix = format!("{}{}", bos_token, user_tag);
            config.prompt_suffix = format!("{}{}", assistant_tag, think_suffix);
        }
    }

    config
}

impl LLMBackend {
    pub fn new() -> Self {
        Self {
            request_tx: RwLock::new(None),
            metrics: Arc::new(Mutex::new(LLMMetrics::default())),
            model_config: Arc::new(Mutex::new(ModelRuntimeConfig {
                use_chat_template: false,
                prompt_prefix: String::new(),
                prompt_suffix: String::new(),
                sampling: default_sampling_params(),
            })),
            provider_override: None,
            device_id_override: None,
            device_ids_override: None,
            shared_pool: None,
            kv_blocks_cap: None,
        }
    }

    /// Attach a shared block allocator so this backend draws from a unified
    /// KV pool shared with other `LLMBackend` instances on the same device.
    pub fn with_shared_pool(mut self, allocator: SharedBlockAllocator) -> Self {
        self.shared_pool = Some(allocator);
        self
    }

    /// Set a per-replica cap on KV cache total_blocks. The runtime calls this
    /// when spawning additional replicas so the block budget is divided fairly
    /// across all engines on the same device.
    pub fn with_kv_blocks_cap(mut self, cap: usize) -> Self {
        self.kv_blocks_cap = Some(cap);
        self
    }

    pub fn with_device(provider: String, device_id: i32) -> Self {
        let mut backend = Self::new();
        backend.provider_override = Some(provider);
        backend.device_id_override = Some(device_id);
        backend
    }

    pub fn with_devices(provider: String, device_ids: Vec<i32>) -> Self {
        let mut backend = Self::new();
        backend.provider_override = Some(provider);
        backend.device_ids_override = Some(device_ids);
        backend
    }

    pub fn with_device_id(device_id: i32) -> Self {
        let mut backend = Self::new();
        backend.device_id_override = Some(device_id);
        backend
    }

    pub fn with_device_ids(device_ids: Vec<i32>) -> Self {
        let mut backend = Self::new();
        backend.device_ids_override = Some(device_ids);
        backend
    }
}

impl Default for LLMBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Engine for LLMBackend {
    async fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        log::info!("Starting LLMEngine for model: {}", model_path.display());

        let runtime_cfg = load_model_runtime_config(model_path);
        {
            let mut cfg_guard = self.model_config.lock().unwrap();
            *cfg_guard = runtime_cfg;
        }

        let (request_tx, request_rx) = mpsc::channel(100);
        let (load_tx, load_rx) = oneshot::channel::<Result<(), EngineError>>();

        // Read per-model tuning hints from metadata.json before constructing LLMEngine.
        // This ensures BlockManager and SchedulerConfig are sized for the actual model
        // rather than using one-size-fits-all defaults.
        struct ManifestHints {
            max_seq_len: usize,
            max_num_seqs: usize,
            max_paddings: usize,
            block_size: usize,
            num_gpu_blocks: usize,
        }
        let hints: ManifestHints = {
            let model_root = find_model_root(model_path);
            let meta_path = model_root.join("metadata.json");
            let llm_meta = std::fs::File::open(&meta_path)
                .ok()
                .and_then(|f| serde_json::from_reader::<_, serde_json::Value>(f).ok())
                .and_then(|meta| meta.get("metadata").and_then(|m| m.get("llm")).cloned());

            let get_usize = |llm: &serde_json::Value, key: &str| -> Option<usize> {
                llm.get(key)
                    .and_then(|v| v.as_u64())
                    .filter(|&v| v > 0)
                    .map(|v| v as usize)
            };

            if let Some(llm) = llm_meta.as_ref() {
                let max_seq_len = get_usize(llm, "max_sequence_length")
                    .or_else(|| get_usize(llm, "max_seq_len"))
                    .unwrap_or(2048);
                let sched = llm.get("scheduler");
                let kv = llm.get("kv_cache");
                ManifestHints {
                    max_seq_len,
                    max_num_seqs: sched
                        .and_then(|s| get_usize(s, "max_num_seqs"))
                        .unwrap_or(16),
                    max_paddings: sched
                        .and_then(|s| get_usize(s, "max_paddings"))
                        .unwrap_or(32),
                    block_size: kv.and_then(|k| get_usize(k, "block_size")).unwrap_or(16),
                    num_gpu_blocks: kv.and_then(|k| get_usize(k, "total_blocks")).unwrap_or(128),
                }
            } else {
                ManifestHints {
                    max_seq_len: 2048,
                    max_num_seqs: 16,
                    max_paddings: 32,
                    block_size: 16,
                    num_gpu_blocks: 128,
                }
            }
        };
        let config = SchedulerConfig {
            max_num_batched_tokens: hints.max_seq_len,
            max_num_seqs: hints.max_num_seqs,
            max_paddings: hints.max_paddings,
        };

        let engine_path = model_path.to_path_buf();
        let metrics = self.metrics.clone();
        let provider_override = self.provider_override.clone();
        let device_id_override = self.device_id_override;
        let device_ids_override = self.device_ids_override.clone();
        let engine_block_size = hints.block_size;
        let engine_num_gpu_blocks = hints.num_gpu_blocks;
        let shared_pool = self.shared_pool.clone();
        let kv_blocks_cap = self.kv_blocks_cap;
        tokio::spawn(async move {
            let mut engine = LLMEngine::new(
                config,
                engine_block_size,
                engine_num_gpu_blocks,
                request_rx,
                metrics,
                provider_override,
                device_id_override,
                device_ids_override,
            );
            // If a shared pool was attached, replace the private allocator.
            let mut engine = if let Some(pool) = shared_pool {
                engine.with_shared_pool(pool)
            } else {
                engine
            };
            // Apply per-replica block cap if set.
            if let Some(cap) = kv_blocks_cap {
                engine.set_kv_blocks_cap(cap);
            }
            let load_result = engine.load(&engine_path).await;
            if let Err(e) = load_tx.send(load_result) {
                log::error!("Failed to send load result: {:?}", e);
            }

            // Only run the loop if loading was successful
            if engine.is_loaded() {
                engine.run_loop().await;
            }
        });

        // Await the oneshot receiver instead of blocking the current runtime
        match load_rx.await {
            Ok(Ok(_)) => {
                let mut tx_guard = self.request_tx.write().unwrap();
                *tx_guard = Some(request_tx);
                Ok(())
            }
            Ok(Err(e)) => Err(e),
            Err(e) => Err(EngineError::backend(format!(
                "Failed to receive load status: {}",
                e
            ))),
        }
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let stream = self.infer_stream(request);
        let mut pinned_stream = Box::pin(stream);

        // Run the async stream to completion, reusing a shared runtime when possible.
        enum ExecMode {
            BlockInPlace(tokio::runtime::Handle),
            SharedRuntime,
            SpawnThread,
        }

        let exec_mode = match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == RuntimeFlavor::CurrentThread => {
                ExecMode::SpawnThread
            }
            Ok(handle) => ExecMode::BlockInPlace(handle),
            Err(_) => ExecMode::SharedRuntime,
        };

        let run_stream = async move {
            let mut all_text = String::new();
            let mut last_packet = None;
            while let Some(packet_res) = pinned_stream.next().await {
                match packet_res {
                    Ok(packet) => {
                        if let Ok(text) = std::str::from_utf8(&packet.data) {
                            all_text.push_str(text);
                        }
                        last_packet = Some(packet);
                    }
                    Err(err) => return Err(err),
                }
            }
            Ok::<_, EngineError>((all_text, last_packet))
        };

        let result = match exec_mode {
            ExecMode::BlockInPlace(handle) => {
                Ok(tokio::task::block_in_place(|| handle.block_on(run_stream)))
            }
            ExecMode::SharedRuntime => Ok(shared_runtime().block_on(run_stream)),
            ExecMode::SpawnThread => {
                std::thread::spawn(move || shared_runtime().block_on(run_stream))
                    .join()
                    .map_err(|e| EngineError::backend(format!("Failed to join thread: {:?}", e)))
            }
        };
        let (all_text, last_packet) = result??;

        if let Some(mut packet) = last_packet {
            packet.data = all_text.into_bytes();
            packet.shape = vec![1, packet.data.len() as i64];
            Ok(packet)
        } else {
            Err(EngineError::backend("No output from LLM engine"))
        }
    }

    fn infer_stream(
        &self,
        request: &InferenceRequest,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>> {
        let (response_tx, mut response_rx) = mpsc::channel(100);

        if request.input.dtype != TensorDtype::Utf8 {
            return Box::pin(stream::once(async {
                Err(EngineError::invalid_input("LLM backend expects Utf8 input"))
            }));
        }

        let prompt = match String::from_utf8(request.input.data.clone()) {
            Ok(text) => text,
            Err(err) => {
                return Box::pin(stream::once(async {
                    Err(EngineError::invalid_input_with_source(
                        "Invalid UTF-8 input",
                        err,
                    ))
                }))
            }
        };

        let runtime_cfg = self.model_config.lock().unwrap().clone();
        let prompt = if runtime_cfg.use_chat_template {
            format!(
                "{}{}{}",
                runtime_cfg.prompt_prefix, prompt, runtime_cfg.prompt_suffix
            )
        } else {
            prompt
        };

        let mut sampling_params = runtime_cfg.sampling;
        if let Some(meta) = request.metadata.as_ref() {
            if let Some(max_new) = meta.max_new_tokens {
                if max_new > 0 {
                    sampling_params.max_tokens = max_new as usize;
                }
            }
            if let Some(temp) = meta.temperature {
                sampling_params.temperature = temp;
            }
            if let Some(top_p) = meta.top_p {
                sampling_params.top_p = top_p;
            }
            if let Some(top_k) = meta.top_k {
                sampling_params.top_k = top_k as usize;
            }
            if let Some(penalty) = meta.repetition_penalty {
                sampling_params.repetition_penalty = penalty;
            }
            if let Some(seed) = meta.seed {
                sampling_params.seed = Some(seed);
            }
            if let Some(extra_stop) = meta.stop_token_ids.as_ref() {
                if !extra_stop.is_empty() {
                    for id in extra_stop {
                        if !sampling_params.stop_token_ids.contains(id) {
                            sampling_params.stop_token_ids.push(*id);
                        }
                    }
                }
            }
        }
        let request_id = request
            .metadata
            .as_ref()
            .and_then(|meta| meta.request_id.clone())
            .or_else(|| request.session_id.clone())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let cancellation = request.cancellation.clone();

        let seq_group = SequenceGroup::new(
            request_id,
            request.session_id.clone(),
            prompt,
            vec![],
            sampling_params,
            cancellation.clone(),
            response_tx,
        );

        let tx_guard = self.request_tx.read().unwrap();
        if let Some(tx) = tx_guard.as_ref() {
            let tx = tx.clone();
            drop(tx_guard);

            let stream = stream! {
                let mut emitted = String::new();
                if let Some(token) = cancellation.as_ref() {
                    if token.is_cancelled() {
                        yield Err(EngineError::cancelled("Request cancelled"));
                        return;
                    }
                }

                if tx.send(seq_group).await.is_err() {
                    yield Err(EngineError::backend("Failed to send request to engine"));
                    return;
                }

                let mut saw_finish = false;
                loop {
                    if let Some(token) = cancellation.as_ref() {
                        if token.is_cancelled() {
                            yield Err(EngineError::cancelled("Request cancelled"));
                            return;
                        }
                    }

                    let Some(output) = response_rx.recv().await else {
                        break;
                    };

                    if let Some(token) = cancellation.as_ref() {
                        if token.is_cancelled() {
                            yield Err(EngineError::cancelled("Request cancelled"));
                            return;
                        }
                    }

                    let output_text = output.text;
                    let chunk = if output_text.starts_with(&emitted) && output_text.len() >= emitted.len() {
                        let delta = output_text[emitted.len()..].to_string();
                        emitted = output_text;
                        delta
                    } else {
                        emitted.push_str(&output_text);
                        output_text
                    };

                    if !chunk.is_empty() {
                        yield Ok(BinaryTensorPacket {
                            shape: vec![1, (chunk.len() as i64)],
                            dtype: TensorDtype::Utf8,
                            data: chunk.into_bytes(),
                        });
                    }
                    if output.finish_reason.is_some() {
                        saw_finish = true;
                        break;
                    }
                }

                if !saw_finish {
                    if let Some(token) = cancellation.as_ref() {
                        if token.is_cancelled() {
                            yield Err(EngineError::cancelled("Request cancelled"));
                            return;
                        }
                    }
                    yield Err(EngineError::backend("LLM response channel closed"));
                }
            };
            Box::pin(stream)
        } else {
            Box::pin(stream::once(async { Err(EngineError::ModelNotLoaded) }))
        }
    }

    fn unload(&mut self) {
        let mut tx_guard = self.request_tx.write().unwrap();
        *tx_guard = None;
    }

    fn metrics(&self) -> EngineMetrics {
        let m = self.metrics.lock().unwrap();
        EngineMetrics {
            inference_time: m.total_inference_time,
            kv_cache_bytes_used: m.kv_cache_bytes_used,
            kv_cache_bytes_capacity: m.kv_cache_bytes_capacity,
            kv_cache_blocks_total: m.kv_cache_blocks_total,
            kv_cache_blocks_free: m.kv_cache_blocks_free,
            kv_cache_sequences: m.kv_cache_sequences,
            kv_cache_evicted_blocks: m.kv_cache_evicted_blocks,
            kv_cache_evicted_sequences: m.kv_cache_evicted_sequences,
            kv_cache_packed_layers: m.kv_cache_packed_layers,
            ..EngineMetrics::default()
        }
    }

    fn health_check(&self) -> Result<(), EngineError> {
        let tx_guard = self.request_tx.read().unwrap();
        if tx_guard.is_some() {
            Ok(())
        } else {
            Err(EngineError::ModelNotLoaded)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_tokenizer_standalone() {
        let tokenizer_path = PathBuf::from("../../models/deepseek/tokenizer.json");
        if !tokenizer_path.exists() {
            return;
        }
        use tokenizers::Tokenizer;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
        let prompt = "Explain quantum physics in one sentence.";
        let encoded = tokenizer.encode(prompt, true).expect("Failed to tokenize");
        let ids = encoded.get_ids();
        let tokens = encoded.get_tokens();
        println!("Prompt: {}", prompt);
        println!("IDs: {:?}", ids);
        println!("Tokens: {:?}", tokens);
        assert!(ids.len() > 1);
    }

    #[tokio::test]
    async fn test_llm_backend_load() {
        let _ = env_logger::builder().is_test(true).try_init();
        // Only run if model exists
        let model_path = PathBuf::from("../../models/deepseek/model_q4f16.onnx");
        if !model_path.exists() {
            log::warn!(
                "Skipping test because model does not exist at {:?}",
                model_path
            );
            return;
        }

        let mut backend = LLMBackend::new();
        let result = backend.load(&model_path).await;
        assert!(result.is_ok());

        // Test health check
        assert!(backend.health_check().is_ok());

        // Test inference stream
        let request = InferenceRequest {
            input: BinaryTensorPacket {
                shape: vec![1, 1],
                dtype: TensorDtype::Utf8,
                data: "H".as_bytes().to_vec(),
            },
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let mut stream = backend.infer_stream(&request);
        let mut text = String::new();
        while let Some(packet_res) = stream.next().await {
            assert!(packet_res.is_ok());
            let packet = packet_res.unwrap();
            let chunk = String::from_utf8_lossy(&packet.data);
            print!("{}", chunk);
            text.push_str(&chunk);
        }

        assert!(!text.is_empty());
        println!("\nGenerated text: {}", text);
    }
}

#[path = "llm_backend_tests.rs"]
mod llm_backend_tests;
