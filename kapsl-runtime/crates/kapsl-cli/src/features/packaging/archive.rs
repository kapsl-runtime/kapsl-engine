use super::*;

/// Fully resolved package data consumed by the archive writer.
///
/// Input discovery, CLI precedence, and interactive prompting happen before
/// this boundary. The writer therefore has one job for both single-file and
/// directory-context builds: validate and persist an `.aimod` archive.
pub(super) struct PackageArchivePlan {
    pub(super) output_path: PathBuf,
    pub(super) manifest: Manifest,
    pub(super) entries: Vec<(PathBuf, PathBuf)>,
    pub(super) source_metadata_path: Option<PathBuf>,
}

fn append_tar_bytes_entry<W: Write>(
    builder: &mut Builder<W>,
    entry_path: &str,
    bytes: &[u8],
) -> Result<(), String> {
    let mut header = tar::Header::new_gnu();
    header
        .set_path(entry_path)
        .map_err(|error| format!("Failed to set tar path {entry_path}: {error}"))?;
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append(&header, bytes)
        .map_err(|error| format!("Failed to append {entry_path} to archive: {error}"))
}

pub(super) fn write_package_archive(
    plan: PackageArchivePlan,
) -> Result<PackageKapslResponse, String> {
    let PackageArchivePlan {
        output_path,
        manifest,
        entries,
        source_metadata_path,
    } = plan;

    validate_model_contract(&manifest)?;
    validate_serving_backend_declaration(&manifest)?;

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "Failed to create parent directory {}: {}",
                    parent.display(),
                    error
                )
            })?;
        }
    }

    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("Failed to encode metadata.json: {error}"))?;
    let output_file = File::create(&output_path).map_err(|error| {
        format!(
            "Failed to create output package {}: {}",
            output_path.display(),
            error
        )
    })?;
    // Model weights are effectively incompressible. Fast gzip plus a large
    // buffer avoids spending CPU and syscalls for negligible size reduction.
    let encoder = GzEncoder::new(
        BufWriter::with_capacity(8 << 20, output_file),
        Compression::fast(),
    );
    let mut archive = Builder::new(encoder);
    append_tar_bytes_entry(&mut archive, "metadata.json", &manifest_bytes)?;
    for (source, archive_path) in entries {
        archive
            .append_path_with_name(&source, &archive_path)
            .map_err(|error| {
                format!(
                    "Failed to add {} to archive: {}",
                    archive_path.display(),
                    error
                )
            })?;
    }

    let encoder = archive
        .into_inner()
        .map_err(|error| format!("Failed to finalize tar archive: {error}"))?;
    let mut writer = encoder
        .finish()
        .map_err(|error| format!("Failed to finalize gzip stream: {error}"))?;
    writer
        .flush()
        .map_err(|error| format!("Failed to flush output package: {error}"))?;

    let metadata_path = source_metadata_path
        .map(|path| create_source_metadata_if_missing(&path, &manifest_bytes))
        .transpose()?
        .flatten();
    let absolute_output_path = output_path.canonicalize().unwrap_or(output_path);

    Ok(PackageKapslResponse {
        status: "ok".to_string(),
        kapsl_path: absolute_output_path.to_string_lossy().to_string(),
        project_name: manifest.project_name,
        framework: manifest.framework,
        version: manifest.version,
        metadata_path: metadata_path.map(|path| path.to_string_lossy().to_string()),
    })
}
