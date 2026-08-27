use super::archive::{find_manifest_root, unpack_archive};
use super::is_valid_extension_id;
use super::marketplace::download_extension_archive;
use kapsl_rag::extension::{ExtensionRegistry, InstalledExtension};

/// Downloads and installs an extension inside the running engine process.
pub(crate) fn download_and_install_marketplace_extension(
    registry: &ExtensionRegistry,
    extension_id: &str,
    marketplace_url: Option<&str>,
) -> Result<InstalledExtension, String> {
    let extension_id = extension_id.trim();
    if !is_valid_extension_id(extension_id) {
        return Err(format!("Invalid extension_id `{extension_id}`"));
    }

    let archive_bytes = download_extension_archive(extension_id, marketplace_url)?;
    let temp_directory = tempfile::Builder::new()
        .prefix("kapsl-extension-marketplace-")
        .tempdir()
        .map_err(|error| format!("Failed to prepare temporary extension directory: {error}"))?;

    unpack_archive(&archive_bytes, temp_directory.path())?;
    let extracted_root = find_manifest_root(temp_directory.path())?;
    registry
        .install_from_dir(&extracted_root)
        .map_err(|error| error.to_string())
}
