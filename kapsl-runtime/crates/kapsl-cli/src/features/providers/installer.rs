use super::activation::activate_staged_provider_pack;
use super::pack::{
    provider_archive_name, provider_release_url, ProviderPack, ProviderPackManifest,
};
use super::transfer::{sha256_file, ProviderPackTransfer};
use crate::DynError;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Installs verified provider packs using injected transfer operations.
pub(super) struct ProviderInstaller<'a> {
    version: &'a str,
    base_url: &'a str,
    install_dir: &'a Path,
    force: bool,
    transfer: &'a dyn ProviderPackTransfer,
}

impl<'a> ProviderInstaller<'a> {
    pub(super) fn new(
        version: &'a str,
        base_url: &'a str,
        install_dir: &'a Path,
        force: bool,
        transfer: &'a dyn ProviderPackTransfer,
    ) -> Self {
        Self {
            version,
            base_url,
            install_dir,
            force,
            transfer,
        }
    }

    pub(super) fn install(&self, pack: ProviderPack) -> Result<(), DynError> {
        if !self.force && installed_manifest_is_complete(pack, self.version, self.install_dir) {
            println!(
                "{} provider pack is already installed.",
                pack.display_name()
            );
            return Ok(());
        }

        let temporary = tempfile::Builder::new()
            .prefix("kapsl-provider-install-")
            .tempdir()?;
        let archive_name = provider_archive_name(pack, self.version);
        let release_url = provider_release_url(self.base_url, self.version);
        let archive_url = format!("{release_url}/{archive_name}");
        let checksum_url = format!("{archive_url}.sha256");
        let archive_path = temporary.path().join(&archive_name);
        let extract_dir = temporary.path().join("extracted");
        fs::create_dir_all(&extract_dir)?;

        println!(
            "Downloading {} provider pack for Kapsl {}...",
            pack.display_name(),
            self.version
        );
        self.transfer
            .download_archive(&archive_url, &archive_path)?;
        let expected_checksum = self.transfer.download_checksum(&checksum_url)?;
        let actual_checksum = sha256_file(&archive_path)?;
        if !actual_checksum.eq_ignore_ascii_case(&expected_checksum) {
            return Err(format!(
                "Checksum verification failed for {archive_name}: expected {expected_checksum}, got {actual_checksum}"
            )
            .into());
        }

        self.transfer.expand_archive(&archive_path, &extract_dir)?;
        let manifest_path = extract_dir.join(pack.manifest_name());
        let manifest = validate_staged_manifest(pack, self.version, &manifest_path, &extract_dir)?;
        activate_staged_provider_pack(
            pack,
            &manifest,
            &manifest_path,
            &extract_dir,
            self.install_dir,
        )?;

        println!(
            "Installed {} into {}.",
            pack.display_name(),
            self.install_dir.display()
        );
        Ok(())
    }
}

fn validate_staged_manifest(
    pack: ProviderPack,
    version: &str,
    manifest_path: &Path,
    directory: &Path,
) -> Result<ProviderPackManifest, DynError> {
    let manifest_metadata = fs::symlink_metadata(manifest_path).map_err(|error| {
        format!(
            "Could not inspect provider manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    if !manifest_metadata.file_type().is_file() {
        return Err(format!(
            "Provider manifest is not a regular file: {}",
            manifest_path.display()
        )
        .into());
    }

    let manifest: ProviderPackManifest = serde_json::from_slice(&fs::read(manifest_path)?)
        .map_err(|error| {
            format!(
                "Invalid provider manifest {}: {error}",
                manifest_path.display()
            )
        })?;
    if !manifest.provider.eq_ignore_ascii_case(pack.provider()) {
        return Err(format!(
            "Provider manifest identifies `{}` instead of `{}`",
            manifest.provider,
            pack.provider()
        )
        .into());
    }
    if manifest.runtime_version != version {
        return Err(format!(
            "Provider pack targets Kapsl {}, but this runtime is {}",
            manifest.runtime_version, version
        )
        .into());
    }
    if manifest.platform != "windows-x86_64" {
        return Err(format!(
            "Provider pack targets {}, not windows-x86_64",
            manifest.platform
        )
        .into());
    }
    if manifest.files.is_empty() {
        return Err("Provider manifest contains no files".into());
    }

    let mut unique_file_names = HashSet::with_capacity(manifest.files.len());
    for file_name in &manifest.files {
        let relative = Path::new(file_name);
        let unsafe_path = file_name.is_empty()
            || file_name.contains(['/', '\\'])
            || matches!(file_name.as_str(), "." | "..")
            || relative.components().count() != 1
            || relative.file_name().is_none();
        if unsafe_path {
            return Err(format!("Unsafe provider manifest path: {file_name}").into());
        }
        if pack.owns_manifest_name(file_name) {
            return Err(format!(
                "Provider manifest reserves `{file_name}` for activation metadata"
            )
            .into());
        }
        if !unique_file_names.insert(file_name.to_ascii_lowercase()) {
            return Err(format!("Provider manifest repeats file: {file_name}").into());
        }

        let staged_file = directory.join(relative);
        let staged_metadata = fs::symlink_metadata(&staged_file).map_err(|error| {
            format!("Provider pack is missing declared file {file_name}: {error}")
        })?;
        if !staged_metadata.file_type().is_file() {
            return Err(format!("Provider pack entry is not a regular file: {file_name}").into());
        }
    }
    Ok(manifest)
}

fn installed_manifest_is_complete(pack: ProviderPack, version: &str, directory: &Path) -> bool {
    let manifest_path = directory.join(pack.manifest_name());
    validate_staged_manifest(pack, version, &manifest_path, directory).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    struct FakeProviderPackTransfer;

    impl ProviderPackTransfer for FakeProviderPackTransfer {
        fn download_archive(&self, _url: &str, path: &Path) -> Result<(), DynError> {
            fs::write(path, b"provider-archive")?;
            Ok(())
        }

        fn download_checksum(&self, _url: &str) -> Result<String, DynError> {
            Ok(format!("{:x}", Sha256::digest(b"provider-archive")))
        }

        fn expand_archive(&self, _archive: &Path, destination: &Path) -> Result<(), DynError> {
            fs::write(destination.join("provider.dll"), b"provider-runtime")?;
            fs::write(
                destination.join(ProviderPack::Cuda12.manifest_name()),
                br#"{"provider":"cuda","runtime_version":"0.1.18","platform":"windows-x86_64","files":["provider.dll"]}"#,
            )?;
            Ok(())
        }
    }

    #[test]
    fn injected_transfer_installs_a_verified_pack() {
        let temporary = tempfile::tempdir().expect("create temporary directory");
        let installer = ProviderInstaller::new(
            "0.1.18",
            "https://downloads.example.test",
            temporary.path(),
            true,
            &FakeProviderPackTransfer,
        );

        installer
            .install(ProviderPack::Cuda12)
            .expect("install provider pack through injected transfer");

        assert_eq!(
            fs::read(temporary.path().join("provider.dll")).unwrap(),
            b"provider-runtime"
        );
        assert!(temporary
            .path()
            .join(ProviderPack::Cuda12.manifest_name())
            .is_file());
    }

    #[test]
    fn staged_manifest_must_match_runtime_and_platform() {
        let temporary = tempfile::tempdir().expect("create temporary directory");
        let directory = temporary.path();
        fs::write(
            directory.join("onnxruntime_providers_cuda.dll"),
            b"provider",
        )
        .expect("write provider fixture");
        let manifest_path = directory.join(ProviderPack::Cuda12.manifest_name());
        fs::write(
            &manifest_path,
            br#"{
                "provider":"cuda",
                "runtime_version":"0.1.18",
                "platform":"windows-x86_64",
                "files":["onnxruntime_providers_cuda.dll"]
            }"#,
        )
        .expect("write provider manifest");

        assert!(validate_staged_manifest(
            ProviderPack::Cuda12,
            "0.1.18",
            &manifest_path,
            directory
        )
        .is_ok());
        assert!(validate_staged_manifest(
            ProviderPack::Cuda12,
            "0.1.19",
            &manifest_path,
            directory
        )
        .is_err());
    }

    #[test]
    fn staged_manifest_rejects_duplicate_and_reserved_file_names() {
        let temporary = tempfile::tempdir().expect("create temporary directory");
        let directory = temporary.path();
        fs::write(directory.join("provider.dll"), b"provider").expect("write provider fixture");
        let manifest_path = directory.join(ProviderPack::Cuda12.manifest_name());

        fs::write(
            &manifest_path,
            br#"{"provider":"cuda","runtime_version":"0.1.18","platform":"windows-x86_64","files":["provider.dll","PROVIDER.DLL"]}"#,
        )
        .expect("write duplicate manifest");
        assert!(validate_staged_manifest(
            ProviderPack::Cuda12,
            "0.1.18",
            &manifest_path,
            directory
        )
        .is_err());

        fs::write(
            &manifest_path,
            br#"{"provider":"cuda","runtime_version":"0.1.18","platform":"windows-x86_64","files":["kapsl-provider-cuda12.json"]}"#,
        )
        .expect("write reserved-name manifest");
        assert!(validate_staged_manifest(
            ProviderPack::Cuda12,
            "0.1.18",
            &manifest_path,
            directory
        )
        .is_err());
    }
}
