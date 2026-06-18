use super::*;

pub(crate) struct TempDirGuard {
    pub(crate) path: PathBuf,
}

impl TempDirGuard {
    pub(crate) fn new(prefix: &str) -> Result<Self, String> {
        Self::new_in(&std::env::temp_dir(), prefix)
    }

    pub(crate) fn new_in(parent: &Path, prefix: &str) -> Result<Self, String> {
        let dir = parent.join(format!(
            "{}-{}-{}",
            prefix,
            std::process::id(),
            temp_nonce()
        ));
        fs::create_dir_all(&dir).map_err(|e| {
            format!(
                "Failed to create temporary directory {}: {}",
                dir.display(),
                e
            )
        })?;
        Ok(Self { path: dir })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub(crate) fn temp_nonce() -> String {
    let mut nonce_bytes = [0u8; 8];
    OsRng.fill_bytes(&mut nonce_bytes);
    hex_encode(&nonce_bytes)
}

pub(crate) fn staged_output_path(output_path: &Path, prefix: &str) -> PathBuf {
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact.aimod");
    parent.join(format!(
        ".{}.{}-{}-{}.part",
        file_name,
        prefix,
        std::process::id(),
        temp_nonce()
    ))
}

pub(crate) fn replace_output_file(staged_path: &Path, output_path: &Path) -> std::io::Result<()> {
    if output_path.exists() {
        fs::remove_file(output_path)?;
    }
    fs::rename(staged_path, output_path)
}

pub(crate) fn stage_link_or_copy_file(
    source_path: &Path,
    output_path: &Path,
    prefix: &str,
) -> Result<u64, String> {
    if source_path == output_path {
        return fs::metadata(source_path)
            .map(|meta| meta.len())
            .map_err(|e| format!("Failed to stat {}: {}", source_path.display(), e));
    }

    let staged_path = staged_output_path(output_path, prefix);
    let stage_result = match fs::hard_link(source_path, &staged_path) {
        Ok(()) => fs::metadata(source_path)
            .map(|meta| meta.len())
            .map_err(|e| {
                format!(
                    "Failed to stat staged linked artifact {}: {}",
                    source_path.display(),
                    e
                )
            }),
        Err(_) => fs::copy(source_path, &staged_path).map_err(|e| {
            format!(
                "Failed to copy artifact {} to staging path {}: {}",
                source_path.display(),
                staged_path.display(),
                e
            )
        }),
    };

    let bytes = match stage_result {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = fs::remove_file(&staged_path);
            return Err(error);
        }
    };

    replace_output_file(&staged_path, output_path).map_err(|e| {
        let _ = fs::remove_file(&staged_path);
        format!(
            "Failed to finalize staged artifact {} -> {}: {}",
            staged_path.display(),
            output_path.display(),
            e
        )
    })?;

    Ok(bytes)
}

pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
