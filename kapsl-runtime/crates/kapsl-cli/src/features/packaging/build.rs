use super::*;

pub(crate) fn infer_framework_from_model_path(model_path: &Path) -> String {
    let ext = model_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "onnx" => "onnx".to_string(),
        "gguf" => "gguf".to_string(),
        "safetensors" => "pytorch".to_string(),
        "pt" | "pth" => "pytorch".to_string(),
        "pb" => "tensorflow".to_string(),
        _ => "onnx".to_string(),
    }
}

pub(crate) fn looks_like_model_file_path(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "onnx" | "gguf" | "safetensors" | "pt" | "pth" | "pb"
    )
}

/// Resolve a model path to a `PackageLoader`.
///
/// Priority:
/// 1. Known single-file extension (`.gguf`, `.onnx`, `.safetensors` …) →
///    `from_raw_file`
/// 2. Directory with safetensors shards → `from_directory`
/// 3. `.aimod` archive → `PackageLoader::load`
pub(crate) fn resolve_package_loader(
    path: &Path,
    model_id: impl std::fmt::Display,
) -> Result<PackageLoader, Box<dyn std::error::Error + Send + Sync>> {
    if looks_like_model_file_path(path) {
        return PackageLoader::from_raw_file(path)
            .map_err(|e| format!("Failed to load raw model {model_id}: {e}").into());
    }
    if path.is_dir() {
        return PackageLoader::from_directory(path)
            .map_err(|e| format!("Failed to load model directory {model_id}: {e}").into());
    }
    PackageLoader::load(path).map_err(|e| format!("Failed to load model {model_id}: {e}").into())
}

pub(crate) fn append_tar_bytes_entry<W: Write>(
    builder: &mut Builder<W>,
    entry_path: &str,
    bytes: &[u8],
) -> Result<(), String> {
    let mut header = tar::Header::new_gnu();
    header
        .set_path(entry_path)
        .map_err(|e| format!("Failed to set tar path {}: {}", entry_path, e))?;
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append(&header, bytes)
        .map_err(|e| format!("Failed to append {} to archive: {}", entry_path, e))
}

pub(crate) fn create_kapsl_package(
    request: &PackageKapslRequest,
    interactive_metadata_setup: bool,
) -> Result<PackageKapslResponse, String> {
    let input_model_path = PathBuf::from(request.model_path.trim());
    if !input_model_path.exists() {
        return Err(format!(
            "Model file does not exist: {}",
            input_model_path.display()
        ));
    }

    if !input_model_path.is_file() {
        return Err(format!(
            "Model path must be a file: {}",
            input_model_path.display()
        ));
    }

    let model_path = input_model_path.canonicalize().map_err(|e| {
        format!(
            "Failed to resolve model file path {}: {}",
            input_model_path.display(),
            e
        )
    })?;

    let model_file_name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("Invalid model filename: {}", model_path.display()))?
        .to_string();

    let mut project_name = request
        .project_name
        .as_ref()
        .map(|name| name.trim())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .or_else(|| {
            model_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "kapsl-model".to_string());

    let mut framework = request
        .framework
        .as_ref()
        .map(|framework| framework.trim())
        .filter(|framework| !framework.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| infer_framework_from_model_path(&model_path));

    let mut version = request
        .version
        .as_ref()
        .map(|version| version.trim())
        .filter(|version| !version.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| "1.0.0".to_string());

    let mut hardware_requirements = kapsl_core::HardwareRequirements::default();

    // Orthogonal model axes (format / model_type / task). Populated from CLI
    // overrides and/or the interactive flow; left None otherwise so the loader
    // infers them from `framework`. See kapsl_core::EngineKind.
    let mut format_axis: Option<String> = None;
    let mut model_type_axis: Option<String> = None;
    let mut task_axis: Option<String> = None;

    // Non-interactive: --format / --model-type / --task fill the axes (and a
    // consistent legacy `framework`) without prompting.
    let axes = AxisOverrides {
        format: request.format.as_deref(),
        model_type: request.model_type.as_deref(),
        task: request.task.as_deref(),
    };
    if axes.any() {
        let (fmt, mt, tk, fw) =
            resolve_axis_triple(infer_format_from_model_path(&model_path), axes);
        framework = fw;
        format_axis = Some(fmt);
        model_type_axis = Some(mt);
        task_axis = Some(tk);
    }

    // The model file's directory is where we persist a generated metadata.json so
    // a subsequent build can be reproduced without re-prompting.
    let source_metadata_path = model_path
        .parent()
        .map(|parent| parent.join("metadata.json"))
        .unwrap_or_else(|| PathBuf::from("metadata.json"));
    let should_create_source_metadata = !source_metadata_path.exists();

    if should_create_source_metadata && interactive_metadata_setup {
        let a = Ansi::new();
        eprintln!(
            "{}",
            a.bold("No metadata.json found. Let's create one for this model.")
        );
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
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
        let mut output_path = request
            .output_path
            .as_ref()
            .map(|path| PathBuf::from(path.trim()))
            .unwrap_or_else(|| PathBuf::from(format!("{}.aimod", project_name)));

        if output_path.extension().and_then(|ext| ext.to_str()) != Some("aimod") {
            output_path.set_extension("aimod");
        }

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
            .as_secs();

        let metadata = match request.metadata.as_ref() {
            Some(metadata) => Some(
                serde_yaml::to_value(metadata)
                    .map_err(|e| format!("Failed to convert metadata payload: {}", e))?,
            ),
            None => None,
        };

        let manifest = Manifest {
            project_name: project_name.clone(),
            framework: framework.clone(),
            version: version.clone(),
            created_at: created_at.to_string(),
            model_file: model_file_name.clone(),
            format: format_axis,
            model_type: model_type_axis,
            task: task_axis,
            metadata,
            hardware_requirements,
            cron_jobs: Vec::new(),
        };
        EngineKind::validate(&manifest)?;

        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| format!("Failed to encode metadata.json: {}", e))?;
        let output_file = File::create(&output_path).map_err(|e| {
            format!(
                "Failed to create output package {}: {}",
                output_path.display(),
                e
            )
        })?;
        // Model weights are binary float data — nearly incompressible. Use level 1
        // (fast) instead of the default level 6 to avoid burning CPU for no gain.
        // The 8 MiB BufWriter reduces syscall overhead from one call per 32 KiB
        // GzEncoder output chunk to one call per 8 MiB.
        let encoder = GzEncoder::new(
            BufWriter::with_capacity(8 << 20, output_file),
            Compression::fast(),
        );
        let mut archive = Builder::new(encoder);

        append_tar_bytes_entry(&mut archive, "metadata.json", &manifest_bytes)?;

        archive
            .append_path_with_name(&model_path, &model_file_name)
            .map_err(|e| format!("Failed to add model file to archive: {}", e))?;

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

    // When prompting was shown above, the caller suppressed its spinner so the
    // prompts stayed legible. Drive the spinner here, around the actual packaging.
    if interactive_metadata_setup {
        run_with_loading("Building package", package)
    } else {
        package()
    }
}

pub(crate) fn find_model_file_in_context(context_dir: &Path) -> Result<PathBuf, String> {
    let mut stack = vec![context_dir.to_path_buf()];
    let mut matches = Vec::new();
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
                stack.push(path);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|v| v.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if matches!(
                ext.as_str(),
                "onnx" | "gguf" | "safetensors" | "pt" | "pth" | "pb"
            ) {
                matches.push(path);
            }
        }
    }

    matches.sort();
    if matches.is_empty() {
        return Err(format!(
            "No model file found in context {}. Pass --model explicitly.",
            context_dir.display()
        ));
    }
    if matches.len() > 1 {
        return Err(format!(
            "Multiple model files found in context {}. Pass --model explicitly.",
            context_dir.display()
        ));
    }
    Ok(matches.remove(0))
}
