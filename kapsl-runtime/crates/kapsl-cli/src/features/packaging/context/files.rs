use super::*;

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
