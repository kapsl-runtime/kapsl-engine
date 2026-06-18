use super::*;

pub(crate) const FORMAT_OPTIONS: &[(&str, &str)] = &[
    ("onnx", "ONNX graph"),
    ("gguf", "GGUF file — tokenizer embedded in the file"),
    ("safetensors", "safetensors weights (custom kernels)"),
];
pub(crate) const MODEL_TYPE_OPTIONS: &[(&str, &str)] = &[
    ("causal-lm", "autoregressive LLM (text generation)"),
    ("embedding", "embedding / encoder model"),
    ("seq-classifier", "sequence classifier"),
    ("seq2seq", "encoder-decoder (seq2seq)"),
    ("opaque", "raw graph — run as-is, tensors in/out"),
];
pub(crate) const TASK_OPTIONS_CAUSAL_LM: &[(&str, &str)] = &[
    ("generate", "autoregressive text generation"),
    ("embed", "embeddings from hidden states"),
    ("forward", "raw forward pass"),
];
pub(crate) const TASK_OPTIONS_SEQ2SEQ: &[(&str, &str)] = &[
    ("generate", "sequence generation"),
    ("forward", "raw forward pass"),
];

/// Model file format inferred from a model path's extension, constrained to the
/// known `format` vocabulary (`onnx`/`gguf`/`safetensors`).
pub(crate) fn infer_format_from_model_path(model_path: &Path) -> &'static str {
    match model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "gguf" => "gguf",
        "safetensors" => "safetensors",
        _ => "onnx",
    }
}

/// Default model type for a format when the user hasn't said otherwise.
pub(crate) fn default_model_type_for_format(format: &str) -> &'static str {
    match format {
        "gguf" => "causal-lm",
        _ => "opaque",
    }
}

/// Default serving task for a model type.
pub(crate) fn default_task_for_model_type(model_type: &str) -> &'static str {
    match model_type {
        "causal-lm" => "generate",
        "embedding" => "embed",
        "seq-classifier" => "classify",
        "seq2seq" => "generate",
        _ => "forward",
    }
}

/// The selectable tasks for a model type. A single-element slice means the task
/// is fixed and is not prompted for.
pub(crate) fn task_options_for_model_type(
    model_type: &str,
) -> &'static [(&'static str, &'static str)] {
    match model_type {
        "causal-lm" => TASK_OPTIONS_CAUSAL_LM,
        "seq2seq" => TASK_OPTIONS_SEQ2SEQ,
        "embedding" => &[("embed", "embeddings")],
        "seq-classifier" => &[("classify", "classification")],
        _ => &[("forward", "raw forward pass")],
    }
}

/// Legacy `framework` value equivalent to a `(format, model_type, task)` triple,
/// kept so packages stay loadable by readers that predate the split.
pub(crate) fn legacy_framework_for(format: &str, _model_type: &str, task: &str) -> String {
    match format {
        "gguf" => "gguf",
        "safetensors" => "safetensors",
        "onnx" if task == "generate" => "llm",
        _ => "onnx",
    }
    .to_string()
}

/// CLI overrides for the orthogonal model axes (`--format` / `--model-type` /
/// `--task`).
#[derive(Default, Clone, Copy)]
pub(crate) struct AxisOverrides<'a> {
    pub(crate) format: Option<&'a str>,
    pub(crate) model_type: Option<&'a str>,
    pub(crate) task: Option<&'a str>,
}

impl AxisOverrides<'_> {
    pub(crate) fn any(&self) -> bool {
        self.format.is_some() || self.model_type.is_some() || self.task.is_some()
    }
}

/// Resolve the full `(format, model_type, task, framework)` from optional axis
/// overrides, filling unspecified axes with format-derived defaults. The legacy
/// `framework` is derived to stay consistent with the chosen axes.
pub(crate) fn resolve_axis_triple(
    default_format: &str,
    axes: AxisOverrides,
) -> (String, String, String, String) {
    let pick = |o: Option<&str>| {
        o.map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let format = pick(axes.format).unwrap_or_else(|| default_format.to_string());
    let task_hint = pick(axes.task);
    // When only a task is given, infer the model type it implies so e.g.
    // `--task embed` alone is a coherent (embedding, embed) pair rather than
    // (opaque, embed).
    let model_type = pick(axes.model_type).unwrap_or_else(|| {
        match task_hint.as_deref() {
            Some("embed") => "embedding",
            Some("classify") => "seq-classifier",
            Some("generate") => "causal-lm",
            _ => default_model_type_for_format(&format),
        }
        .to_string()
    });
    let task = task_hint.unwrap_or_else(|| default_task_for_model_type(&model_type).to_string());
    let framework = legacy_framework_for(&format, &model_type, &task);
    (format, model_type, task, framework)
}
