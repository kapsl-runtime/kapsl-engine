use super::archive::{find_manifest_root, unpack_archive};
use super::is_valid_extension_id;
use super::marketplace::download_verified_extension_archive;
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

    let archive_bytes = download_verified_extension_archive(extension_id, marketplace_url)?;
    install_archive_bytes(registry, &archive_bytes)
}

/// Unpacks an already-verified archive and installs it into the registry.
fn install_archive_bytes(
    registry: &ExtensionRegistry,
    archive_bytes: &[u8],
) -> Result<InstalledExtension, String> {
    let temp_directory = tempfile::Builder::new()
        .prefix("kapsl-extension-marketplace-")
        .tempdir()
        .map_err(|error| format!("Failed to prepare temporary extension directory: {error}"))?;

    unpack_archive(archive_bytes, temp_directory.path())?;
    let extracted_root = find_manifest_root(temp_directory.path())?;
    registry
        .install_from_dir(&extracted_root)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::super::marketplace::serve_for_tests;
    use super::*;
    use sha2::{Digest, Sha256};

    fn extension_archive(module: &[u8]) -> Vec<u8> {
        let manifest = "id = \"connector.demo\"\nname = \"Demo\"\nversion = \"1.0.0\"\nruntime = \"wasm\"\ncapabilities = [\"sync\"]\nauth = [\"none\"]\npermissions = []\nentrypoint = \"connector.wasm\"\n";
        let mut builder = tar::Builder::new(Vec::new());
        for (name, data) in [
            ("rag-extension.toml", manifest.as_bytes()),
            ("connector.wasm", module),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, name, data).unwrap();
        }
        let tar_bytes = builder.into_inner().unwrap();
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, &tar_bytes).unwrap();
        encoder.finish().unwrap()
    }

    fn catalog_pinning(archive: &[u8]) -> String {
        serde_json::json!([{
            "manifest": {"id": "connector.demo"},
            "archive_sha256": format!("{:x}", Sha256::digest(archive)),
        }])
        .to_string()
    }

    #[test]
    fn a_tampered_module_is_refused_and_nothing_is_installed() {
        let genuine = extension_archive(b"\0asm\x01\0\0\0");
        let tampered = extension_archive(b"\0asm\x01\0\0\0 malicious");
        let registry_dir = tempfile::tempdir().unwrap();
        let registry = ExtensionRegistry::new(registry_dir.path());

        // The marketplace pins the genuine archive but serves the tampered one.
        let base = serve_for_tests(catalog_pinning(&genuine), tampered, 2);
        let error =
            download_and_install_marketplace_extension(&registry, "connector.demo", Some(&base))
                .unwrap_err();
        assert!(error.contains("failed SHA-256 verification"), "{error}");
        assert!(registry.discover().unwrap().is_empty());
    }

    #[test]
    fn a_module_matching_its_pin_is_installed() {
        let genuine = extension_archive(b"\0asm\x01\0\0\0");
        let registry_dir = tempfile::tempdir().unwrap();
        let registry = ExtensionRegistry::new(registry_dir.path());

        let base = serve_for_tests(catalog_pinning(&genuine), genuine, 2);
        let installed =
            download_and_install_marketplace_extension(&registry, "connector.demo", Some(&base))
                .unwrap();
        assert_eq!(installed.manifest.id, "connector.demo");
        assert_eq!(registry.discover().unwrap().len(), 1);
    }
}
