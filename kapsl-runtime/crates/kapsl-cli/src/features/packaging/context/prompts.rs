use super::*;

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
