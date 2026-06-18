use super::*;

pub(crate) type ContextManifest = (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<kapsl_core::HardwareRequirements>,
    Option<serde_json::Value>,
);

pub(crate) fn parse_context_manifest(context_dir: &Path) -> Result<ContextManifest, String> {
    let metadata_path = context_dir.join("metadata.json");
    if !metadata_path.exists() {
        return Ok((None, None, None, None, None, None));
    }

    let raw = fs::read_to_string(&metadata_path)
        .map_err(|e| format!("Failed to read {}: {}", metadata_path.display(), e))?;
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        format!(
            "Invalid metadata.json in {}: {}",
            metadata_path.display(),
            e
        )
    })?;
    let Some(obj) = value.as_object() else {
        return Err(format!(
            "metadata.json in {} must be a JSON object",
            metadata_path.display()
        ));
    };

    let project_name = obj
        .get("project_name")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let framework = obj
        .get("framework")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let version = obj
        .get("version")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let model_file = obj
        .get("model_file")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let hardware_requirements = obj
        .get("hardware_requirements")
        .cloned()
        .map(serde_json::from_value::<kapsl_core::HardwareRequirements>)
        .transpose()
        .map_err(|e| format!("Invalid hardware_requirements in metadata.json: {}", e))?;

    let metadata = obj.get("metadata").cloned();

    Ok((
        project_name,
        framework,
        version,
        model_file,
        hardware_requirements,
        metadata,
    ))
}

pub(crate) fn create_source_metadata_if_missing(
    metadata_path: &Path,
    manifest_bytes: &[u8],
) -> Result<Option<PathBuf>, String> {
    let mut file = match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(metadata_path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Ok(None),
        Err(error) => {
            return Err(format!(
                "Failed to create {}: {}",
                metadata_path.display(),
                error
            ));
        }
    };
    file.write_all(manifest_bytes).map_err(|e| {
        format!(
            "Failed to write generated metadata.json {}: {}",
            metadata_path.display(),
            e
        )
    })?;
    Ok(Some(metadata_path.to_path_buf()))
}

pub(crate) fn prompt_with_default(
    reader: &mut impl BufRead,
    label: &str,
    default: &str,
) -> Result<String, String> {
    let a = Ansi::new();
    eprint!(
        "{} {} {}: ",
        a.teal("?"),
        label,
        a.dim(&format!("({})", default))
    );
    std::io::stderr()
        .flush()
        .map_err(|e| format!("Failed to flush prompt: {}", e))?;

    let mut input = String::new();
    let bytes = reader
        .read_line(&mut input)
        .map_err(|e| format!("Failed to read prompt input: {}", e))?;
    // read_line returns 0 only at end of input. Treat that as an abort rather
    // than as "use the default", so a closed stdin (or Ctrl-D) can never spin a
    // retry loop forever.
    if bytes == 0 {
        return Err("unexpected end of input while reading prompt".to_string());
    }
    let trimmed = input.trim();
    if trimmed.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

pub(crate) fn prompt_non_empty_with_default(
    reader: &mut impl BufRead,
    label: &str,
    default: &str,
) -> Result<String, String> {
    loop {
        let value = prompt_with_default(reader, label, default)?;
        if !value.trim().is_empty() {
            return Ok(value.trim().to_string());
        }
        eprintln!("  {}", Ansi::new().red("Value cannot be empty."));
    }
}

/// (value, description) pairs shown in the interactive selection menus.
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

/// Prompt for the serving task, skipping the prompt when the model type allows
/// only one task.
pub(crate) fn prompt_task_for_model_type(
    reader: &mut impl BufRead,
    model_type: &str,
    preferred: Option<&str>,
) -> Result<String, String> {
    let options = task_options_for_model_type(model_type);
    // Use a preferred task (e.g. from --task) as the default when it's valid for
    // this model type, else the model type's natural default.
    let default = preferred
        .map(str::trim)
        .filter(|p| options.iter().any(|(v, _)| v.eq_ignore_ascii_case(p)))
        .unwrap_or_else(|| default_task_for_model_type(model_type));
    if options.len() <= 1 {
        return Ok(default.to_string());
    }
    prompt_select_with_default(reader, "Task", options, default)
}

pub(crate) const PROVIDER_OPTIONS: &[(&str, &str)] = &[
    ("cpu", "CPU execution"),
    ("cuda", "NVIDIA GPU (CUDA)"),
    ("tensorrt", "NVIDIA TensorRT"),
    ("coreml", "Apple CoreML / Metal"),
    ("rocm", "AMD GPU (ROCm)"),
    ("directml", "Windows DirectML"),
];

/// Prompt the user to pick one of `options` (each a `(value, description)` pair).
/// Accepts the option's number or its value (case-insensitive); an empty line
/// keeps `default`. Re-prompts on an invalid choice so the returned value is
/// always one of `options`.
pub(crate) fn prompt_select_with_default(
    reader: &mut impl BufRead,
    label: &str,
    options: &[(&str, &str)],
    default: &str,
) -> Result<String, String> {
    let a = Ansi::new();
    let default_index = options
        .iter()
        .position(|(value, _)| value.eq_ignore_ascii_case(default));
    let name_width = options
        .iter()
        .map(|(value, _)| value.len())
        .max()
        .unwrap_or(0);

    eprintln!("{} {}:", a.teal("?"), label);
    for (index, (value, description)) in options.iter().enumerate() {
        let suffix = if Some(index) == default_index {
            " (default)"
        } else {
            ""
        };
        eprintln!(
            "    {}) {:<width$}  {}",
            index + 1,
            value,
            a.dim(&format!("{}{}", description, suffix)),
            width = name_width
        );
    }

    loop {
        eprint!("  {} {}: ", a.teal("›"), a.dim(&format!("[{}]", default)));
        std::io::stderr()
            .flush()
            .map_err(|e| format!("Failed to flush prompt: {}", e))?;

        let mut input = String::new();
        reader
            .read_line(&mut input)
            .map_err(|e| format!("Failed to read prompt input: {}", e))?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            return Ok(default.to_string());
        }
        if let Ok(choice) = trimmed.parse::<usize>() {
            if choice >= 1 && choice <= options.len() {
                return Ok(options[choice - 1].0.to_string());
            }
        }
        if let Some((value, _)) = options
            .iter()
            .find(|(value, _)| value.eq_ignore_ascii_case(trimmed))
        {
            return Ok(value.to_string());
        }
        eprintln!(
            "  {}",
            a.red("Enter a number from the list or one of the option names.")
        );
    }
}

pub(crate) fn prompt_provider_with_default(
    reader: &mut impl BufRead,
    default: Option<&str>,
) -> Result<Option<String>, String> {
    let default = default
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("cpu");
    let value =
        prompt_select_with_default(reader, "Preferred provider", PROVIDER_OPTIONS, default)?;
    Ok(Some(value))
}

pub(crate) fn prompt_model_file_with_default(
    reader: &mut impl BufRead,
    context_dir: &Path,
    default: &str,
) -> Result<PathBuf, String> {
    // Compare against the canonicalized context dir: `canonicalize()` below
    // resolves symlinks (e.g. macOS /var -> /private/var), so the bound must be
    // resolved too or every valid in-context file is wrongly rejected.
    let context_canonical = context_dir
        .canonicalize()
        .unwrap_or_else(|_| context_dir.to_path_buf());
    loop {
        let value = prompt_non_empty_with_default(reader, "Model file", default)?;
        let candidate = context_dir.join(&value);
        if candidate.exists() && candidate.is_file() {
            let canonical = candidate.canonicalize().map_err(|e| {
                format!(
                    "Failed to resolve model file path {}: {}",
                    candidate.display(),
                    e
                )
            })?;
            if canonical.starts_with(&context_canonical) {
                return Ok(canonical);
            }
        }
        eprintln!(
            "  {}",
            Ansi::new().red("Model file must exist inside the build context.")
        );
    }
}

pub(crate) fn normalize_output_path_for_context(
    context_dir: &Path,
    output: Option<&Path>,
    project_name: &str,
) -> PathBuf {
    let mut output_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| context_dir.join(format!("{}.aimod", project_name)));
    if output_path.extension().and_then(|v| v.to_str()) != Some("aimod") {
        output_path.set_extension("aimod");
    }
    output_path
}

pub(crate) fn collect_existing_file_references_from_metadata(
    context_dir: &Path,
    value: &serde_yaml::Value,
    out: &mut HashSet<PathBuf>,
) {
    match value {
        serde_yaml::Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return;
            }
            let candidate = context_dir.join(trimmed);
            if !candidate.exists() || !candidate.is_file() {
                return;
            }
            if let Ok(relative) = candidate.strip_prefix(context_dir) {
                out.insert(relative.to_path_buf());
            }
        }
        serde_yaml::Value::Sequence(seq) => {
            for item in seq {
                collect_existing_file_references_from_metadata(context_dir, item, out);
            }
        }
        serde_yaml::Value::Mapping(map) => {
            for (_, item) in map {
                collect_existing_file_references_from_metadata(context_dir, item, out);
            }
        }
        _ => {}
    }
}

pub(crate) fn collect_context_files(
    context_dir: &Path,
    output_path: &Path,
    primary_model_ext: &str,
    keep_primary_model_files: &HashSet<PathBuf>,
) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let output_canonical = output_path.canonicalize().ok();
    let output_in_context_relative = output_path
        .strip_prefix(context_dir)
        .ok()
        .map(PathBuf::from);

    let primary_model_ext = primary_model_ext.trim().to_ascii_lowercase();
    let mut keep_onnx_stems_by_dir: HashMap<PathBuf, HashSet<String>> = HashMap::new();
    if primary_model_ext == "onnx" {
        for rel in keep_primary_model_files {
            if rel
                .extension()
                .and_then(|v| v.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("onnx"))
                .unwrap_or(false)
            {
                let dir = rel.parent().unwrap_or(Path::new("")).to_path_buf();
                let stem = rel
                    .file_stem()
                    .and_then(|v| v.to_str())
                    .unwrap_or("")
                    .to_string();
                if !stem.is_empty() {
                    keep_onnx_stems_by_dir.entry(dir).or_default().insert(stem);
                }
            }
        }
    }

    // First collect all file paths and (when applicable) the ONNX stems per directory, so we can
    // reliably decide which *.onnx_data* files belong to which model file.
    let mut all_files = Vec::new();
    let mut onnx_stems_by_dir: HashMap<PathBuf, Vec<String>> = HashMap::new();
    let mut primary_model_count_by_dir: HashMap<PathBuf, usize> = HashMap::new();
    let mut stack = vec![context_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|e| format!("Failed to read context directory {}: {}", dir.display(), e))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|e| format!("Failed to inspect {}: {}", path.display(), e))?;
            if file_type.is_dir() {
                let rel = path
                    .strip_prefix(context_dir)
                    .map_err(|e| format!("Failed to resolve dir path {}: {}", path.display(), e))?
                    .to_path_buf();
                if rel.components().any(|c| {
                    matches!(
                        c.as_os_str().to_str(),
                        Some(".git" | "__pycache__" | ".pytest_cache")
                    )
                }) {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }

            let relative = path
                .strip_prefix(context_dir)
                .map_err(|e| format!("Failed to resolve file path {}: {}", path.display(), e))?
                .to_path_buf();

            let ext = relative
                .extension()
                .and_then(|v| v.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();

            if !primary_model_ext.is_empty() && ext == primary_model_ext {
                let dir_rel = relative.parent().unwrap_or(Path::new("")).to_path_buf();
                *primary_model_count_by_dir.entry(dir_rel).or_insert(0) += 1;
            }

            if primary_model_ext == "onnx" && ext == "onnx" {
                let dir_rel = relative.parent().unwrap_or(Path::new("")).to_path_buf();
                let stem = relative
                    .file_stem()
                    .and_then(|v| v.to_str())
                    .unwrap_or("")
                    .to_string();
                if !stem.is_empty() {
                    onnx_stems_by_dir.entry(dir_rel).or_default().push(stem);
                }
            }

            all_files.push((path, relative));
        }
    }

    let mut files = Vec::new();
    for (path, relative) in all_files {
        // Skip common local/system noise.
        if relative
            .file_name()
            .and_then(|v| v.to_str())
            .map(|name| name == ".DS_Store")
            .unwrap_or(false)
        {
            continue;
        }
        if relative.components().any(|c| {
            matches!(
                c.as_os_str().to_str(),
                Some(".git" | "__pycache__" | ".pytest_cache")
            )
        }) {
            continue;
        }

        // metadata.json is generated for each build.
        if relative == Path::new("metadata.json") {
            continue;
        }
        // Avoid recursively embedding existing packages.
        if relative.extension().and_then(|ext| ext.to_str()) == Some("aimod") {
            continue;
        }
        if output_in_context_relative
            .as_ref()
            .map(|p| p == &relative)
            .unwrap_or(false)
        {
            continue;
        }
        if output_canonical
            .as_ref()
            .map(|out| out == &path)
            .unwrap_or(false)
        {
            continue;
        }

        let ext = relative
            .extension()
            .and_then(|v| v.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if !primary_model_ext.is_empty()
            && ext == primary_model_ext
            && !keep_primary_model_files.is_empty()
            && primary_model_count_by_dir
                .get(relative.parent().unwrap_or(Path::new("")))
                .copied()
                .unwrap_or(0)
                > 1
            && !keep_primary_model_files.contains(&relative)
        {
            continue;
        }

        if primary_model_ext == "onnx" && !keep_primary_model_files.is_empty() {
            if let Some(name) = relative.file_name().and_then(|v| v.to_str()) {
                if name.contains(".onnx_data") {
                    let dir_rel = relative.parent().unwrap_or(Path::new("")).to_path_buf();
                    if primary_model_count_by_dir
                        .get(&dir_rel)
                        .copied()
                        .unwrap_or(0)
                        <= 1
                    {
                        files.push((path, relative));
                        continue;
                    }
                    let stems = onnx_stems_by_dir.get(&dir_rel);
                    let mut associated_stem: Option<&str> = None;
                    if let Some(stems) = stems {
                        for stem in stems {
                            let needle = format!("{}.onnx_data", stem);
                            if name.starts_with(&needle) {
                                associated_stem = Some(stem.as_str());
                                break;
                            }
                        }
                        if associated_stem.is_none() && stems.len() == 1 {
                            associated_stem = stems.first().map(|s| s.as_str());
                        }
                    }

                    if let Some(stem) = associated_stem {
                        if let Some(keep) = keep_onnx_stems_by_dir.get(&dir_rel) {
                            if !keep.contains(stem) {
                                continue;
                            }
                        } else {
                            // We can associate this data shard with a concrete ONNX model in this
                            // directory, but none of the ONNX models in this directory are kept.
                            continue;
                        }
                    }
                }
            }
        }

        files.push((path, relative));
    }

    files.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(files)
}

pub(crate) fn create_kapsl_package_from_context(
    context_path: &Path,
    model_override: Option<&Path>,
    output_override: Option<&Path>,
    project_name_override: Option<&str>,
    framework_override: Option<&str>,
    version_override: Option<&str>,
    metadata_override: Option<&serde_json::Value>,
    axes: AxisOverrides,
    interactive_metadata_setup: bool,
) -> Result<PackageKapslResponse, String> {
    let context_input = PathBuf::from(context_path);
    if !context_input.exists() {
        return Err(format!(
            "Build context does not exist: {}",
            context_input.display()
        ));
    }
    if !context_input.is_dir() {
        return Err(format!(
            "Build context must be a directory: {}",
            context_input.display()
        ));
    }
    let context_dir = context_input.canonicalize().map_err(|e| {
        format!(
            "Failed to resolve build context {}: {}",
            context_input.display(),
            e
        )
    })?;

    let source_metadata_path = context_dir.join("metadata.json");
    let should_create_source_metadata = !source_metadata_path.exists();

    let (
        project_name_from_manifest,
        framework_from_manifest,
        version_from_manifest,
        model_file_from_manifest,
        hardware_requirements_from_manifest,
        metadata_from_manifest,
    ) = parse_context_manifest(&context_dir)?;

    let mut model_path = if let Some(model_path) = model_override {
        let candidate = if model_path.is_absolute() {
            model_path.to_path_buf()
        } else {
            let in_context = context_dir.join(model_path);
            if in_context.exists() {
                in_context
            } else {
                model_path.to_path_buf()
            }
        };
        if !candidate.exists() || !candidate.is_file() {
            return Err(format!(
                "Model file does not exist: {}",
                candidate.display()
            ));
        }
        candidate.canonicalize().map_err(|e| {
            format!(
                "Failed to resolve model file path {}: {}",
                candidate.display(),
                e
            )
        })?
    } else if let Some(model_file) = model_file_from_manifest.as_deref() {
        let candidate = context_dir.join(model_file);
        if !candidate.exists() || !candidate.is_file() {
            return Err(format!(
                "metadata.json model_file does not exist in context: {}",
                candidate.display()
            ));
        }
        candidate
    } else {
        find_model_file_in_context(&context_dir)?
    };

    if !model_path.starts_with(&context_dir) {
        return Err(format!(
            "Model path {} must be inside build context {}",
            model_path.display(),
            context_dir.display()
        ));
    }

    let mut model_file = model_path
        .strip_prefix(&context_dir)
        .map_err(|e| {
            format!(
                "Failed to compute model path relative to context {}: {}",
                model_path.display(),
                e
            )
        })?
        .to_string_lossy()
        .to_string();

    let mut project_name = project_name_override
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or(project_name_from_manifest)
        .or_else(|| {
            context_dir
                .file_name()
                .and_then(|v| v.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "kapsl-model".to_string());

    let mut framework = framework_override
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or(framework_from_manifest)
        .unwrap_or_else(|| infer_framework_from_model_path(&model_path));

    let mut version = version_override
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or(version_from_manifest)
        .unwrap_or_else(|| "1.0.0".to_string());

    let mut hardware_requirements = hardware_requirements_from_manifest.unwrap_or_default();

    // Orthogonal model axes; populated from CLI overrides and/or the interactive
    // flow, else None so the loader infers them from `framework`.
    let mut format_axis: Option<String> = None;
    let mut model_type_axis: Option<String> = None;
    let mut task_axis: Option<String> = None;

    // Non-interactive: --format / --model-type / --task fill the axes.
    if axes.any() {
        let (fmt, mt, tk, fw) =
            resolve_axis_triple(infer_format_from_model_path(&model_path), axes);
        framework = fw;
        format_axis = Some(fmt);
        model_type_axis = Some(mt);
        task_axis = Some(tk);
    }

    if should_create_source_metadata && interactive_metadata_setup {
        let a = Ansi::new();
        eprintln!(
            "{}",
            a.bold("No metadata.json found. Let's create one for this model.")
        );
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
        model_path = prompt_model_file_with_default(&mut reader, &context_dir, &model_file)?;
        model_file = model_path
            .strip_prefix(&context_dir)
            .map_err(|e| {
                format!(
                    "Failed to compute model path relative to context {}: {}",
                    model_path.display(),
                    e
                )
            })?
            .to_string_lossy()
            .to_string();

        project_name = prompt_non_empty_with_default(&mut reader, "Project name", &project_name)?;
        let format_default = format_axis
            .as_deref()
            .unwrap_or_else(|| infer_format_from_model_path(&model_path));
        let format =
            prompt_select_with_default(&mut reader, "Format", FORMAT_OPTIONS, format_default)?;
        let model_type_default = model_type_axis
            .as_deref()
            .unwrap_or_else(|| default_model_type_for_format(&format));
        let model_type = prompt_select_with_default(
            &mut reader,
            "Model type",
            MODEL_TYPE_OPTIONS,
            model_type_default,
        )?;
        let task = prompt_task_for_model_type(&mut reader, &model_type, task_axis.as_deref())?;
        framework = legacy_framework_for(&format, &model_type, &task);
        format_axis = Some(format);
        model_type_axis = Some(model_type);
        task_axis = Some(task);
        version = prompt_non_empty_with_default(&mut reader, "Version", &version)?;
        hardware_requirements.preferred_provider = prompt_provider_with_default(
            &mut reader,
            hardware_requirements.preferred_provider.as_deref(),
        )?;
        eprintln!();
    }

    let package = move || -> Result<PackageKapslResponse, String> {
        let output_path =
            normalize_output_path_for_context(&context_dir, output_override, &project_name);
        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "Failed to create parent directory {}: {}",
                        parent.display(),
                        e
                    )
                })?;
            }
        }

        let created_at = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("System clock error: {}", e))?
            .as_secs()
            .to_string();

        let metadata_value = metadata_override
            .cloned()
            .or(metadata_from_manifest)
            .map(|value| {
                serde_yaml::to_value(value)
                    .map_err(|e| format!("Failed to convert metadata payload: {}", e))
            })
            .transpose()?;

        let primary_model_ext = model_path
            .extension()
            .and_then(|v| v.to_str())
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();

        let mut referenced_files = HashSet::new();
        referenced_files.insert(PathBuf::from(&model_file));
        if let Some(metadata) = metadata_value.as_ref() {
            collect_existing_file_references_from_metadata(
                &context_dir,
                metadata,
                &mut referenced_files,
            );
        }

        let mut keep_primary_model_files: HashSet<PathBuf> = HashSet::new();
        if !primary_model_ext.is_empty() {
            for rel in &referenced_files {
                let ext = rel
                    .extension()
                    .and_then(|v| v.to_str())
                    .unwrap_or("")
                    .trim()
                    .to_ascii_lowercase();
                if !ext.is_empty() && ext == primary_model_ext {
                    keep_primary_model_files.insert(rel.to_path_buf());
                }
            }
        }

        let manifest = Manifest {
            project_name: project_name.clone(),
            framework: framework.clone(),
            version: version.clone(),
            created_at,
            model_file,
            format: format_axis,
            model_type: model_type_axis,
            task: task_axis,
            metadata: metadata_value,
            hardware_requirements,
            cron_jobs: Vec::new(),
        };
        EngineKind::validate(&manifest)?;

        let output_file = File::create(&output_path).map_err(|e| {
            format!(
                "Failed to create output package {}: {}",
                output_path.display(),
                e
            )
        })?;
        let encoder = GzEncoder::new(
            BufWriter::with_capacity(8 << 20, output_file),
            Compression::fast(),
        );
        let mut archive = Builder::new(encoder);

        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| format!("Failed to encode metadata.json: {}", e))?;
        append_tar_bytes_entry(&mut archive, "metadata.json", &manifest_bytes)?;

        for (absolute, relative) in collect_context_files(
            &context_dir,
            &output_path,
            &primary_model_ext,
            &keep_primary_model_files,
        )? {
            archive
                .append_path_with_name(&absolute, &relative)
                .map_err(|e| format!("Failed to add {} to archive: {}", relative.display(), e))?;
        }

        let encoder = archive
            .into_inner()
            .map_err(|e| format!("Failed to finalize tar archive: {}", e))?;
        let mut buf_writer = encoder
            .finish()
            .map_err(|e| format!("Failed to finalize gzip stream: {}", e))?;
        buf_writer
            .flush()
            .map_err(|e| format!("Failed to flush output package: {}", e))?;

        let created_metadata_path = if should_create_source_metadata {
            create_source_metadata_if_missing(&source_metadata_path, &manifest_bytes)?
        } else {
            None
        };

        let absolute_output_path = output_path.canonicalize().unwrap_or(output_path);
        Ok(PackageKapslResponse {
            status: "ok".to_string(),
            kapsl_path: absolute_output_path.to_string_lossy().to_string(),
            project_name,
            framework,
            version,
            metadata_path: created_metadata_path.map(|path| path.to_string_lossy().to_string()),
        })
    };

    // Prompts (above) ran without a spinner; show it here for the packaging work.
    if interactive_metadata_setup {
        run_with_loading("Building package", package)
    } else {
        package()
    }
}
