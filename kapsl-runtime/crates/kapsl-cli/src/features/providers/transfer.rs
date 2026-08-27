use crate::features::packaging::{
    format_remote_http_error, http_agent_for_transfer, native_tls_http_agent,
};
use crate::DynError;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::process::Command;

/// External operations required to obtain and expand a provider pack.
/// Injecting this boundary keeps installation policy independent of HTTP and
/// PowerShell process execution.
pub(super) trait ProviderPackTransfer {
    fn download_archive(&self, url: &str, path: &Path) -> Result<(), DynError>;
    fn download_checksum(&self, url: &str) -> Result<String, DynError>;
    fn expand_archive(&self, archive: &Path, destination: &Path) -> Result<(), DynError>;
}

pub(super) struct NativeProviderPackTransfer;

impl ProviderPackTransfer for NativeProviderPackTransfer {
    fn download_archive(&self, url: &str, path: &Path) -> Result<(), DynError> {
        let agent = http_agent_for_transfer();
        let mut response = agent.get(url).call().map_err(|error| {
            format!(
                "Failed to download {url}: {}",
                format_remote_http_error(error)
            )
        })?;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let mut reader = response.body_mut().as_reader();
        std::io::copy(&mut reader, &mut writer)?;
        writer.flush()?;
        Ok(())
    }

    fn download_checksum(&self, url: &str) -> Result<String, DynError> {
        let agent = native_tls_http_agent();
        let mut response = agent.get(url).call().map_err(|error| {
            format!(
                "Failed to download checksum {url}: {}",
                format_remote_http_error(error)
            )
        })?;
        let content = response.body_mut().read_to_string()?;
        let checksum = content
            .split_whitespace()
            .next()
            .ok_or_else(|| format!("Checksum file from {url} was empty"))?;
        if checksum.len() != 64 || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(
                format!("Checksum file from {url} did not contain a SHA-256 digest").into(),
            );
        }
        Ok(checksum.to_ascii_lowercase())
    }

    fn expand_archive(&self, archive: &Path, destination: &Path) -> Result<(), DynError> {
        let command = "& { param([string]$Archive, [string]$Destination) Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force }";
        let status = Command::new("powershell.exe")
            .args([
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ])
            .arg(archive)
            .arg(destination)
            .status()
            .map_err(|error| {
                format!("Could not start PowerShell to extract the provider pack: {error}")
            })?;
        if !status.success() {
            return Err(format!(
                "PowerShell failed to extract the provider pack (exit code {}).",
                status.code().unwrap_or(-1)
            )
            .into());
        }
        Ok(())
    }
}

pub(super) fn sha256_file(path: &Path) -> Result<String, DynError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    // Windows executables reserve a 1 MiB main-thread stack by default. Keep
    // this transfer buffer on the heap while hashing large provider archives.
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_archive_checksum_is_sha256() {
        let temporary = tempfile::tempdir().expect("create temporary directory");
        let archive_path = temporary.path().join("provider.zip");
        std::fs::write(&archive_path, b"kapsl provider pack").expect("write provider fixture");

        assert_eq!(
            sha256_file(&archive_path).expect("hash provider fixture"),
            "0fb9ed3f95dbf0485da5119520df57775d39e11b36871dff4092c8118d2ea45f"
        );
    }
}
