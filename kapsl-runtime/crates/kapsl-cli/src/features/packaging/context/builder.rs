use super::*;

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
