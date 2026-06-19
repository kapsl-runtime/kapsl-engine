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
