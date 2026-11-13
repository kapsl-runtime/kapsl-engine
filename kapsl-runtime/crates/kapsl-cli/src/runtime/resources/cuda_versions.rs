//! CUDA driver and runtime versions, logged at startup and served by
//! `/api/system/stats` so a version mismatch (`min_cuda_version`, ORT provider
//! builds) is diagnosable without shell access to the host.
//!
//! Three different numbers get called "the CUDA version":
//! - `driver_version`: the NVIDIA kernel driver (`550.54.15`), from NVML.
//! - `driver_cuda_version`: the newest CUDA the driver supports (`12.4`), from
//!   NVML. This is what `nvidia-smi` prints and what `min_cuda_version` is
//!   checked against.
//! - `runtime_version`: the CUDA toolkit/runtime installed with the engine
//!   (`12.4.1`), read from `CUDA_VERSION`, or the toolkit's `version.json` /
//!   `version.txt`. `None` when no toolkit is found (the engine can still run
//!   on bundled provider libraries).

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(crate) struct CudaVersions {
    pub driver_version: Option<String>,
    pub driver_cuda_version: Option<String>,
    pub runtime_version: Option<String>,
}

impl CudaVersions {
    pub(crate) fn describe(&self) -> String {
        let show = |v: &Option<String>| v.clone().unwrap_or_else(|| "unknown".to_string());
        format!(
            "driver {}, driver supports CUDA {}, CUDA runtime {}",
            show(&self.driver_version),
            show(&self.driver_cuda_version),
            show(&self.runtime_version)
        )
    }
}

/// Version from the toolkit's `version.json` (`"cuda_cudart": {"version": ..}`,
/// falling back to the top-level `"cuda"` entry).
pub(crate) fn parse_toolkit_version_json(raw: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    let version = value
        .pointer("/cuda_cudart/version")
        .or_else(|| value.pointer("/cuda/version"))?
        .as_str()?;
    clean_version(version)
}

/// Version from the legacy `version.txt` (`CUDA Version 11.8.89`).
pub(crate) fn parse_toolkit_version_txt(raw: &str) -> Option<String> {
    let rest = raw.trim().strip_prefix("CUDA Version")?;
    clean_version(rest)
}

fn clean_version(raw: &str) -> Option<String> {
    let version = raw.trim();
    let valid = !version.is_empty()
        && version
            .split('.')
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()));
    valid.then(|| version.to_string())
}

fn toolkit_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for key in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(value) = std::env::var_os(key).filter(|v| !v.is_empty()) {
            roots.push(PathBuf::from(value));
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));
    roots
}

/// Runtime version from `CUDA_VERSION` (set by NVIDIA's container images),
/// else a toolkit directory.
pub(crate) fn detect_runtime_version(
    env_version: Option<&str>,
    roots: &[PathBuf],
    read: impl Fn(&Path) -> Option<String>,
) -> Option<String> {
    if let Some(version) = env_version.and_then(clean_version) {
        return Some(version);
    }
    roots.iter().find_map(|root| {
        read(&root.join("version.json"))
            .and_then(|raw| parse_toolkit_version_json(&raw))
            .or_else(|| {
                read(&root.join("version.txt")).and_then(|raw| parse_toolkit_version_txt(&raw))
            })
    })
}

/// Probed once; NVML and the filesystem are not consulted again.
pub(crate) fn cuda_versions() -> &'static CudaVersions {
    static VERSIONS: OnceLock<CudaVersions> = OnceLock::new();
    VERSIONS.get_or_init(|| CudaVersions {
        driver_version: super::nvml_driver_version(),
        driver_cuda_version: super::nvml_cuda_version(),
        runtime_version: detect_runtime_version(
            std::env::var("CUDA_VERSION").ok().as_deref(),
            &toolkit_roots(),
            |path| std::fs::read_to_string(path).ok(),
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolkit_version_json_prefers_cudart() {
        let raw = r#"{"cuda":{"name":"CUDA SDK","version":"12.4.1"},"cuda_cudart":{"name":"CUDA Runtime","version":"12.4.127"}}"#;
        assert_eq!(parse_toolkit_version_json(raw).as_deref(), Some("12.4.127"));
        assert_eq!(
            parse_toolkit_version_json(r#"{"cuda":{"version":"12.4.1"}}"#).as_deref(),
            Some("12.4.1")
        );
        assert_eq!(parse_toolkit_version_json("not json"), None);
        assert_eq!(
            parse_toolkit_version_json(r#"{"cuda":{"version":"x"}}"#),
            None
        );
    }

    #[test]
    fn toolkit_version_txt_is_parsed() {
        assert_eq!(
            parse_toolkit_version_txt("CUDA Version 11.8.89\n").as_deref(),
            Some("11.8.89")
        );
        assert_eq!(parse_toolkit_version_txt("garbage"), None);
    }

    #[test]
    fn runtime_version_prefers_env_then_toolkit_dirs() {
        let roots = [PathBuf::from("/a"), PathBuf::from("/b")];
        let read = |path: &Path| match path.to_str().unwrap() {
            "/b/version.txt" => Some("CUDA Version 11.8.89".to_string()),
            _ => None,
        };
        assert_eq!(
            detect_runtime_version(Some("12.6.3"), &roots, read).as_deref(),
            Some("12.6.3")
        );
        assert_eq!(
            detect_runtime_version(Some(""), &roots, read).as_deref(),
            Some("11.8.89")
        );
        assert_eq!(detect_runtime_version(None, &[], read), None);
    }

    #[test]
    fn describe_names_unknown_parts() {
        let versions = CudaVersions {
            driver_version: Some("550.54.15".into()),
            driver_cuda_version: Some("12.4".into()),
            runtime_version: None,
        };
        assert_eq!(
            versions.describe(),
            "driver 550.54.15, driver supports CUDA 12.4, CUDA runtime unknown"
        );
    }
}
