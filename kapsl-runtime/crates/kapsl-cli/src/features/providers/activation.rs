use super::pack::{ProviderPack, ProviderPackManifest};
use crate::DynError;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct BackupEntry {
    original: PathBuf,
    backup: PathBuf,
}

pub(super) fn activate_staged_provider_pack(
    pack: ProviderPack,
    manifest: &ProviderPackManifest,
    manifest_path: &Path,
    staged_directory: &Path,
    install_directory: &Path,
) -> Result<(), DynError> {
    activate_staged_provider_pack_with(
        pack,
        manifest,
        manifest_path,
        staged_directory,
        install_directory,
        |source, destination| fs::rename(source, destination),
    )
}

fn activate_staged_provider_pack_with<F>(
    pack: ProviderPack,
    manifest: &ProviderPackManifest,
    manifest_path: &Path,
    staged_directory: &Path,
    install_directory: &Path,
    mut publish: F,
) -> Result<(), DynError>
where
    F: FnMut(&Path, &Path) -> io::Result<()>,
{
    // Staging under the installation directory keeps every final rename on the
    // same filesystem. No active provider file is touched until all incoming
    // bytes have been copied successfully.
    let transaction = tempfile::Builder::new()
        .prefix(".kapsl-provider-activation-")
        .tempdir_in(install_directory)
        .map_err(|error| {
            format!(
                "Could not prepare a provider activation transaction in {}: {}. If Kapsl was installed system-wide, rerun this command from an Administrator PowerShell.",
                install_directory.display(),
                error
            )
        })?;
    let incoming_directory = transaction.path().join("incoming");
    let backup_directory = transaction.path().join("backup");
    fs::create_dir(&incoming_directory)?;
    fs::create_dir(&backup_directory)?;

    let mut incoming_files = Vec::with_capacity(manifest.files.len());
    for file_name in &manifest.files {
        let incoming = incoming_directory.join(file_name);
        fs::copy(staged_directory.join(file_name), &incoming).map_err(|error| {
            format!(
                "Could not stage {file_name} for installation into {}: {error}. If Kapsl was installed system-wide, rerun this command from an Administrator PowerShell.",
                install_directory.display()
            )
        })?;
        incoming_files.push((incoming, install_directory.join(file_name)));
    }
    let incoming_manifest = transaction.path().join("provider-manifest.new");
    fs::copy(manifest_path, &incoming_manifest).map_err(|error| {
        format!(
            "Could not stage the {} provider manifest for activation: {error}",
            pack.display_name()
        )
    })?;

    let mut targets = incoming_files
        .iter()
        .map(|(_, destination)| destination.clone())
        .collect::<Vec<_>>();
    targets.push(install_directory.join(pack.manifest_name()));
    targets.extend(existing_provider_manifest_paths(pack, install_directory)?);
    deduplicate_targets_case_insensitively(&mut targets);

    // Reject directory collisions before the first mutation so rollback never
    // needs to reason about recursive filesystem changes.
    for target in &targets {
        match fs::symlink_metadata(target) {
            Ok(metadata) if metadata.file_type().is_dir() => {
                return Err(format!(
                    "Cannot replace provider target because it is a directory: {}",
                    target.display()
                )
                .into());
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "Could not inspect provider target {}: {error}",
                    target.display()
                )
                .into())
            }
        }
    }

    let mut backups = Vec::new();
    let published = Vec::new();
    for (index, target) in targets.iter().enumerate() {
        let target_exists = match path_exists_without_following_links(target) {
            Ok(exists) => exists,
            Err(error) => {
                return Err(rollback_activation(
                    pack,
                    format!(
                        "Could not recheck provider target {}: {error}",
                        target.display()
                    ),
                    transaction,
                    &published,
                    &backups,
                ))
            }
        };
        if !target_exists {
            continue;
        }
        let backup = backup_directory.join(format!("{index}.backup"));
        if let Err(error) = fs::rename(target, &backup) {
            return Err(rollback_activation(
                pack,
                format!(
                    "Could not preserve existing file {}: {error}",
                    target.display()
                ),
                transaction,
                &published,
                &backups,
            ));
        }
        backups.push(BackupEntry {
            original: target.clone(),
            backup,
        });
    }

    let mut published = Vec::with_capacity(incoming_files.len() + 1);
    for (incoming, destination) in &incoming_files {
        if let Err(error) = publish(incoming, destination) {
            return Err(rollback_activation(
                pack,
                format!(
                    "Could not publish provider file {}: {error}",
                    destination.display()
                ),
                transaction,
                &published,
                &backups,
            ));
        }
        published.push(destination.clone());
    }

    // The manifest is the activation marker and is always published last.
    // Provider discovery therefore never observes a successful new activation
    // until every declared runtime file is in place.
    let active_manifest = install_directory.join(pack.manifest_name());
    if let Err(error) = publish(&incoming_manifest, &active_manifest) {
        return Err(rollback_activation(
            pack,
            format!(
                "Could not activate provider manifest {}: {error}",
                active_manifest.display()
            ),
            transaction,
            &published,
            &backups,
        ));
    }

    Ok(())
}

fn existing_provider_manifest_paths(
    pack: ProviderPack,
    directory: &Path,
) -> Result<Vec<PathBuf>, DynError> {
    let mut manifests = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if !file_type.is_dir() && pack.owns_manifest_name(file_name) {
            manifests.push(entry.path());
        }
    }
    manifests.sort();
    Ok(manifests)
}

fn deduplicate_targets_case_insensitively(targets: &mut Vec<PathBuf>) {
    let mut seen = HashSet::with_capacity(targets.len());
    targets.retain(|target| {
        let key = target
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        seen.insert(key)
    });
}

fn path_exists_without_following_links(path: &Path) -> Result<bool, DynError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn rollback_activation(
    pack: ProviderPack,
    cause: String,
    transaction: tempfile::TempDir,
    published: &[PathBuf],
    backups: &[BackupEntry],
) -> DynError {
    let mut rollback_errors = Vec::new();

    for path in published.iter().rev() {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => rollback_errors.push(format!(
                "could not remove newly published {}: {error}",
                path.display()
            )),
        }
    }

    for entry in backups.iter().rev() {
        if let Err(error) = fs::rename(&entry.backup, &entry.original) {
            rollback_errors.push(format!(
                "could not restore {}: {error}",
                entry.original.display()
            ));
        }
    }

    if rollback_errors.is_empty() {
        format!(
            "Failed to activate the {} provider pack: {cause}. The previous installation state was rolled back.",
            pack.display_name()
        )
        .into()
    } else {
        let recovery_directory = transaction.keep();
        format!(
            "Failed to activate the {} provider pack: {cause}. Rollback was incomplete: {}. Recovery files were retained at {}.",
            pack.display_name(),
            rollback_errors.join("; "),
            recovery_directory.display()
        )
        .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ActivationFixture {
        _temporary: tempfile::TempDir,
        staged: PathBuf,
        installed: PathBuf,
        manifest_path: PathBuf,
        manifest: ProviderPackManifest,
    }

    impl ActivationFixture {
        fn new() -> Self {
            let temporary = tempfile::tempdir().expect("create temporary directory");
            let staged = temporary.path().join("staged");
            let installed = temporary.path().join("installed");
            fs::create_dir(&staged).expect("create staging directory");
            fs::create_dir(&installed).expect("create install directory");

            fs::write(staged.join("provider-a.dll"), b"new-a").expect("write staged file");
            fs::write(staged.join("provider-b.dll"), b"new-b").expect("write staged file");
            let manifest_path = staged.join(ProviderPack::Cuda12.manifest_name());
            fs::write(&manifest_path, b"new-manifest").expect("write staged manifest");

            fs::write(installed.join("provider-a.dll"), b"old-a").expect("write old file");
            fs::write(installed.join("provider-b.dll"), b"old-b").expect("write old file");
            fs::write(
                installed.join(ProviderPack::Cuda12.manifest_name()),
                b"old-manifest",
            )
            .expect("write old manifest");

            Self {
                _temporary: temporary,
                staged,
                installed,
                manifest_path,
                manifest: ProviderPackManifest {
                    provider: "cuda".to_string(),
                    runtime_version: "0.1.18".to_string(),
                    platform: "windows-x86_64".to_string(),
                    files: vec!["provider-a.dll".to_string(), "provider-b.dll".to_string()],
                },
            }
        }

        fn activate_with<F>(&self, publish: F) -> Result<(), DynError>
        where
            F: FnMut(&Path, &Path) -> io::Result<()>,
        {
            activate_staged_provider_pack_with(
                ProviderPack::Cuda12,
                &self.manifest,
                &self.manifest_path,
                &self.staged,
                &self.installed,
                publish,
            )
        }

        fn assert_previous_installation(&self) {
            assert_eq!(
                fs::read(self.installed.join("provider-a.dll")).unwrap(),
                b"old-a"
            );
            assert_eq!(
                fs::read(self.installed.join("provider-b.dll")).unwrap(),
                b"old-b"
            );
            assert_eq!(
                fs::read(self.installed.join(ProviderPack::Cuda12.manifest_name())).unwrap(),
                b"old-manifest"
            );
        }
    }

    #[test]
    fn activation_publishes_the_manifest_after_every_runtime_file() {
        let fixture = ActivationFixture::new();
        fs::write(
            fixture.installed.join("kapsl-provider-cuda12-legacy.json"),
            b"legacy-manifest",
        )
        .expect("write legacy manifest");

        let mut publication_order = Vec::new();
        fixture
            .activate_with(|source, destination| {
                publication_order.push(
                    destination
                        .file_name()
                        .expect("published file name")
                        .to_string_lossy()
                        .to_string(),
                );
                fs::rename(source, destination)
            })
            .expect("activate provider pack");

        assert_eq!(
            fs::read(fixture.installed.join("provider-a.dll")).unwrap(),
            b"new-a"
        );
        assert_eq!(
            fs::read(fixture.installed.join("provider-b.dll")).unwrap(),
            b"new-b"
        );
        assert_eq!(
            publication_order.last().map(String::as_str),
            Some(ProviderPack::Cuda12.manifest_name())
        );
        assert!(!fixture
            .installed
            .join("kapsl-provider-cuda12-legacy.json")
            .exists());
    }

    #[test]
    fn runtime_file_publication_failure_restores_the_previous_installation() {
        let fixture = ActivationFixture::new();
        let mut publish_count = 0usize;
        let error = fixture
            .activate_with(|source, destination| {
                publish_count += 1;
                if publish_count == 2 {
                    return Err(io::Error::other("injected runtime-file failure"));
                }
                fs::rename(source, destination)
            })
            .expect_err("second runtime file publication must fail");

        assert!(error.to_string().contains("was rolled back"));
        fixture.assert_previous_installation();
    }

    #[test]
    fn manifest_publication_failure_restores_the_previous_installation() {
        let fixture = ActivationFixture::new();
        let mut publish_count = 0usize;
        let error = fixture
            .activate_with(|source, destination| {
                publish_count += 1;
                if publish_count == 3 {
                    return Err(io::Error::other("injected manifest failure"));
                }
                fs::rename(source, destination)
            })
            .expect_err("manifest publication must fail");

        assert!(error.to_string().contains("was rolled back"));
        fixture.assert_previous_installation();
    }
}
