use flate2::read::GzDecoder;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use tar::Archive;

fn collect_manifest_directories(
    directory: &Path,
    matches: &mut Vec<PathBuf>,
) -> Result<(), String> {
    for entry in fs::read_dir(directory).map_err(|error| {
        format!(
            "Failed to inspect extracted extension archive directory {}: {}",
            directory.display(),
            error
        )
    })? {
        let entry = entry.map_err(|error| {
            format!("Failed to read extension archive directory entry: {error}")
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_manifest_directories(&path, matches)?;
        } else if path.file_name().and_then(|name| name.to_str()) == Some("rag-extension.toml") {
            if let Some(parent) = path.parent() {
                matches.push(parent.to_path_buf());
            }
        }
    }

    Ok(())
}

pub(super) fn find_manifest_root(extract_directory: &Path) -> Result<PathBuf, String> {
    let mut matches = Vec::new();
    collect_manifest_directories(extract_directory, &mut matches)?;

    match matches.len() {
        0 => Err(format!(
            "Marketplace archive did not contain rag-extension.toml under {}",
            extract_directory.display()
        )),
        1 => Ok(matches.remove(0)),
        _ => Err(format!(
            "Marketplace archive contained multiple extension manifests under {}",
            extract_directory.display()
        )),
    }
}

pub(super) fn unpack_archive(archive_bytes: &[u8], target_directory: &Path) -> Result<(), String> {
    let decoder = GzDecoder::new(Cursor::new(archive_bytes));
    let mut archive = Archive::new(decoder);
    let entries = archive
        .entries()
        .map_err(|error| format!("Failed to read extension marketplace archive: {error}"))?;

    for entry in entries {
        let mut entry =
            entry.map_err(|error| format!("Failed to read extension archive entry: {error}"))?;
        let unpacked = entry.unpack_in(target_directory).map_err(|error| {
            format!(
                "Failed to unpack extension archive into {}: {}",
                target_directory.display(),
                error
            )
        })?;
        if !unpacked {
            return Err("Extension archive contains invalid paths".to_string());
        }
    }

    Ok(())
}
