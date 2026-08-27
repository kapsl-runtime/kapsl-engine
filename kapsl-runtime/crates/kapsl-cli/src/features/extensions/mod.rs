//! Extension command, marketplace, archive, and installation support.
//!
//! The CLI command sends an install request to a running engine. The engine
//! performs marketplace I/O, safely extracts the archive, and installs it in
//! the configured extension registry.

mod archive;
mod command;
mod installer;
mod marketplace;
mod sync;
mod types;

pub(crate) use command::execute_extension_command;
pub(crate) use installer::download_and_install_marketplace_extension;
pub(crate) use marketplace::fetch_extension_marketplace;
pub(crate) use sync::{extension_key, select_sync_source_id};
pub(crate) use types::{
    ExtensionErrorResponse, ExtensionInstallRequest, ExtensionInstallResponse, SyncExtensionRequest,
};

/// Returns whether an extension ID is non-empty and safe to place in a URL or
/// filesystem-derived registry path.
pub(crate) fn is_valid_extension_id(extension_id: &str) -> bool {
    !extension_id.trim().is_empty()
        && extension_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
}

#[cfg(test)]
mod tests {
    use super::is_valid_extension_id;

    #[test]
    fn validates_extension_ids() {
        assert!(is_valid_extension_id("connector.s3"));
        assert!(is_valid_extension_id("my_connector-2"));
        assert!(!is_valid_extension_id(""));
        assert!(!is_valid_extension_id("connector/s3"));
        assert!(!is_valid_extension_id("connector s3"));
    }
}
