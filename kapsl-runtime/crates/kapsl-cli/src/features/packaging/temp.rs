use super::*;

#[cfg(test)]
pub(crate) struct TempDirGuard {
    pub(crate) path: PathBuf,
}

#[cfg(test)]
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

#[cfg(test)]
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

pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
