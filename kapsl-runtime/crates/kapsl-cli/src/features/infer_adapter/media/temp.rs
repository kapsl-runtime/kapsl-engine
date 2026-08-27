use super::super::error::{InferRequestError, InferResult};
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::{
    io::Write,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
};

pub(super) struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    pub(super) fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[cfg(unix)]
pub(super) fn set_private_dir_permissions(path: &Path) -> InferResult<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|error| {
        InferRequestError::internal(format!(
            "Failed to set private permissions on temporary video directory: {}",
            error
        ))
    })
}

#[cfg(not(unix))]
pub(super) fn set_private_dir_permissions(_path: &Path) -> InferResult<()> {
    Ok(())
}

#[cfg(unix)]
pub(super) fn write_private_temp_file(path: &Path, contents: &[u8]) -> InferResult<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .mode(0o600)
        .open(path)
        .map_err(|error| {
            InferRequestError::internal(format!("Failed to write temporary video input: {}", error))
        })?;
    file.write_all(contents).map_err(|error| {
        InferRequestError::internal(format!("Failed to write temporary video input: {}", error))
    })
}

#[cfg(not(unix))]
pub(super) fn write_private_temp_file(path: &Path, contents: &[u8]) -> InferResult<()> {
    fs::write(path, contents).map_err(|error| {
        InferRequestError::internal(format!("Failed to write temporary video input: {}", error))
    })
}
