use super::managed_vllm_bridge::{
    map_ureq_body_error, ManagedVllmBridgeError, ManagedVllmByteStream, ManagedVllmHttpBridge,
    ManagedVllmRequestTimeouts, ManagedVllmSseStream,
};
use super::*;
use kapsl_engine_api::{
    CancellationToken, EngineStream, MemoryAllocation, MemoryAllocationClass,
    MemoryAllocationSource, MemoryDomain, MemoryReport, OpenAiWireFormat, OpenAiWireHeader,
    OpenAiWireHeaderName, OpenAiWireRequest, OpenAiWireResponse, OpenAiWireResponseHead,
    OpenAiWireStream, OpenAiWireStreamResponse,
};
use serde::Deserialize;
use std::fs::{OpenOptions, Permissions};
use std::net::TcpListener as StdTcpListener;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::{ExitStatus, Stdio};

pub(crate) const MANAGED_VLLM_ADAPTER_ID: &str = "kapsl-vllm-connector";
pub(crate) const MANAGED_VLLM_ADAPTER_VERSION: &str = "0.6.0";
pub(crate) const MANAGED_VLLM_BACKEND_VERSION: &str = "0.26.1rc1.dev1130+g2ec6f0d71";
pub(crate) const MANAGED_VLLM_PROFILE_ID: &str = "vllm-v1-packed-cuda-ipc/flash-attn";
pub(crate) const MANAGED_VLLM_PYTHON_VERSION: &str = "3.12.3";
pub(crate) const MANAGED_VLLM_TORCH_VERSION: &str = "2.13.0+cu130";
pub(crate) const MANAGED_VLLM_TORCHVISION_VERSION: &str = "0.28.0+cu130";
pub(crate) const MANAGED_VLLM_TORCHAUDIO_VERSION: &str = "2.11.0+cu130";
pub(crate) const MANAGED_VLLM_CUDA_RUNTIME_VERSION: &str = "13.0";

const MANAGED_VLLM_PYTHON_ENV: &str = "KAPSL_VLLM_PYTHON";
const MANAGED_VLLM_BUNDLE_ENV: &str = "KAPSL_VLLM_BUNDLE";
const MANAGED_VLLM_CHAT_MARKER: &str = "__kapsl_managed_vllm_chat_v1";
const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);
const SUPERVISOR_INTERVAL: Duration = Duration::from_secs(2);
const SUPERVISOR_BACKOFF: Duration = Duration::from_secs(2);
const MAX_RESTARTS: u32 = 5;
const MAX_CONSECUTIVE_HEALTH_FAILURES: u8 = 15;
const DEFAULT_LEGACY_GPU_MEMORY_UTILIZATION: f64 = 0.5;
const DEFAULT_KV_HEADROOM_PERCENT: usize = 20;
const MAX_KV_HEADROOM_PERCENT: usize = 100;
const BRIDGE_HEADER_TIMEOUT: Duration = Duration::from_secs(30);
const BRIDGE_IDLE_BODY_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_OPENAI_WIRE_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ManagedVllmReplicaState {
    Planned = 0,
    Starting = 1,
    Routable = 2,
    Suspect = 3,
    Stopped = 4,
}

impl ManagedVllmReplicaState {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Planned,
            1 => Self::Starting,
            2 => Self::Routable,
            3 => Self::Suspect,
            4 => Self::Stopped,
            _ => Self::Suspect,
        }
    }
}

/// Lock-free activation revision shared weakly with the KV coordinator.
///
/// A validated detach or participant retirement advances this value before
/// changing coordinator activation/backing state. Managed health probes bind
/// their Routable publication to the observed revision, and request dispatch
/// verifies that the published revision is still current. Keeping this object
/// atomic-only avoids a coordinator-state -> process-lifecycle lock edge.
pub(crate) struct ManagedVllmKvReadinessFence {
    epoch: AtomicU64,
}

impl ManagedVllmKvReadinessFence {
    pub(crate) fn new() -> Self {
        Self {
            epoch: AtomicU64::new(0),
        }
    }

    pub(crate) fn snapshot(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    pub(crate) fn advance(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }
}

pub(crate) fn certified_vllm_profile() -> String {
    format!(
        "{MANAGED_VLLM_ADAPTER_ID},{MANAGED_VLLM_ADAPTER_VERSION},{MANAGED_VLLM_BACKEND_VERSION},{MANAGED_VLLM_PROFILE_ID}"
    )
}

pub(crate) struct ManagedVllmDeployment {
    python: PathBuf,
    control_endpoint: String,
    runtime_root: PathBuf,
    lease_ttl_ms: u64,
    runtimes: parking_lot::Mutex<HashMap<u32, Vec<std::sync::Weak<ManagedVllmRuntime>>>>,
    #[cfg(unix)]
    coordinator: parking_lot::RwLock<Option<Arc<ExternalKvCoordinator>>>,
}

/// Deployment plus the KV-control settings derived during preparation.
pub(crate) struct PreparedManagedVllmDeployment {
    pub(crate) deployment: Arc<ManagedVllmDeployment>,
    pub(crate) control_socket: PathBuf,
    pub(crate) shared_pool_profile: String,
}

impl ManagedVllmDeployment {
    pub(crate) fn prepare(
        state_dir: Option<&Path>,
        requested_control_socket: Option<&Path>,
        lease_ttl_ms: u64,
    ) -> Result<PreparedManagedVllmDeployment, String> {
        if !cfg!(all(feature = "gpu-device-pool", target_os = "linux")) {
            return Err(
                "managed vLLM requires a Linux Kapsl build with `gpu-device-pool` enabled"
                    .to_string(),
            );
        }

        let runtime_root = managed_vllm_runtime_root(state_dir);
        std::fs::create_dir_all(&runtime_root).map_err(|error| {
            format!(
                "create managed vLLM runtime directory {}: {error}",
                runtime_root.display()
            )
        })?;
        #[cfg(unix)]
        std::fs::set_permissions(&runtime_root, Permissions::from_mode(0o700)).map_err(
            |error| {
                format!(
                    "secure managed vLLM runtime directory {}: {error}",
                    runtime_root.display()
                )
            },
        )?;

        let socket_path = match requested_control_socket {
            Some(path) => absolute_path(path)?,
            None => runtime_root.join("kv-control.sock"),
        };
        let shared_pool_profile = certified_vllm_profile();

        let python = discover_certified_vllm_python()?;
        log::info!(
            "Managed vLLM bundle: {} (vLLM {}, connector {}, profile {})",
            python.display(),
            MANAGED_VLLM_BACKEND_VERSION,
            MANAGED_VLLM_ADAPTER_VERSION,
            MANAGED_VLLM_PROFILE_ID
        );

        let deployment = Arc::new(Self {
            python,
            control_endpoint: format!("unix://{}", socket_path.display()),
            runtime_root,
            lease_ttl_ms,
            runtimes: parking_lot::Mutex::new(HashMap::new()),
            #[cfg(unix)]
            coordinator: parking_lot::RwLock::new(None),
        });
        Ok(PreparedManagedVllmDeployment {
            deployment,
            control_socket: socket_path,
            shared_pool_profile,
        })
    }

    #[cfg(unix)]
    pub(crate) fn install_coordinator(
        &self,
        coordinator: Arc<ExternalKvCoordinator>,
    ) -> Result<(), String> {
        let mut installed = self.coordinator.write();
        if let Some(current) = installed.as_ref() {
            if Arc::ptr_eq(current, &coordinator) {
                return Ok(());
            }
            return Err("managed vLLM control coordinator is already installed".to_string());
        }
        *installed = Some(coordinator);
        Ok(())
    }

    fn retire_participants_after_backend_exit(
        &self,
        participant_base: &str,
    ) -> Result<usize, String> {
        #[cfg(unix)]
        {
            let coordinator =
                self.coordinator.read().clone().ok_or_else(|| {
                    "managed vLLM control coordinator is not installed".to_string()
                })?;
            Ok(coordinator.retire_participants_after_backend_exit(participant_base))
        }
        #[cfg(not(unix))]
        {
            let _ = participant_base;
            Err("managed vLLM participant retirement requires Unix".to_string())
        }
    }

    fn participant_is_active(&self, participant_base: &str) -> Result<bool, String> {
        #[cfg(unix)]
        {
            let coordinator =
                self.coordinator.read().clone().ok_or_else(|| {
                    "managed vLLM control coordinator is not installed".to_string()
                })?;
            coordinator.managed_participant_is_active(participant_base)
        }
        #[cfg(not(unix))]
        {
            let _ = participant_base;
            Err("managed vLLM participant readiness requires Unix".to_string())
        }
    }

    fn register_participant_readiness_fence(
        &self,
        participant_base: &str,
        process: &Arc<ManagedVllmProcess>,
    ) -> Result<(), String> {
        #[cfg(unix)]
        {
            let coordinator =
                self.coordinator.read().clone().ok_or_else(|| {
                    "managed vLLM control coordinator is not installed".to_string()
                })?;
            coordinator.register_managed_readiness_fence(
                participant_base,
                Arc::downgrade(&process.kv_readiness_fence),
            )
        }
        #[cfg(not(unix))]
        {
            let _ = (participant_base, process);
            Err("managed vLLM participant readiness requires Unix".to_string())
        }
    }

    fn register_runtime(&self, model_id: u32, runtime: &Arc<ManagedVllmRuntime>) {
        let mut runtimes = self.runtimes.lock();
        let model_runtimes = runtimes.entry(model_id).or_default();
        model_runtimes.retain(|candidate| candidate.strong_count() > 0);
        model_runtimes.push(Arc::downgrade(runtime));
    }

    /// Stop all managed replicas for a logical model at the lifecycle
    /// boundary, without waiting for scheduler worker handles to be dropped.
    /// The process-group fence inside `shutdown` makes it safe to retire the
    /// participant and release its CUDA IPC backing immediately afterwards.
    pub(crate) fn shutdown_model(&self, model_id: u32) -> Result<usize, String> {
        let runtimes = self
            .runtimes
            .lock()
            .remove(&model_id)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|runtime| runtime.upgrade())
            .collect::<Vec<_>>();

        let mut failures = Vec::new();
        for runtime in &runtimes {
            if !runtime.shutdown(self) {
                failures.push(runtime.participant_id.clone());
            }
        }
        if failures.is_empty() {
            return Ok(runtimes.len());
        }

        let mut registered = self.runtimes.lock();
        registered
            .entry(model_id)
            .or_default()
            .extend(runtimes.iter().map(Arc::downgrade));
        Err(format!(
            "managed vLLM teardown could not prove backend exit for participant(s): {}",
            failures.join(", ")
        ))
    }

    /// Fence every managed process group before the core runtime exits.
    /// Ordinary model removal uses `shutdown_model`; this aggregate boundary
    /// is for process-wide signals where no managed child may outlive Kapsl.
    pub(crate) fn shutdown_all(&self) -> Result<usize, String> {
        let model_ids = self.runtimes.lock().keys().copied().collect::<Vec<_>>();
        let mut stopped = 0usize;
        let mut failures = Vec::new();
        for model_id in model_ids {
            match self.shutdown_model(model_id) {
                Ok(count) => stopped = stopped.saturating_add(count),
                Err(error) => failures.push(format!("model {model_id}: {error}")),
            }
        }
        if failures.is_empty() {
            Ok(stopped)
        } else {
            Err(failures.join("; "))
        }
    }
}

fn managed_vllm_runtime_root(state_dir: Option<&Path>) -> PathBuf {
    match state_dir {
        Some(state_dir) => state_dir
            .join("managed-vllm")
            .join(std::process::id().to_string()),
        None => std::env::temp_dir().join(format!("kapsl-vllm-{}", std::process::id())),
    }
}

fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .map_err(|error| format!("resolve {}: {error}", path.display()))
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct ManagedVllmEnvironment {
    python: String,
    torch: String,
    torchvision: String,
    torchaudio: String,
    vllm: String,
    connector_distribution: String,
    connector: String,
    profile: String,
    cuda_runtime: String,
    cuda_available: bool,
    planner_schema_version: u64,
    exact_cache_memory_flag: bool,
}

fn discover_certified_vllm_python() -> Result<PathBuf, String> {
    let explicit_python = std::env::var_os(MANAGED_VLLM_PYTHON_ENV).map(PathBuf::from);
    let candidates = if let Some(python) = explicit_python.as_ref() {
        vec![python.clone()]
    } else {
        managed_vllm_python_candidates()
    };

    let mut failures = Vec::new();
    for candidate in candidates {
        match validate_certified_vllm_python(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) => failures.push(format!("{}: {error}", candidate.display())),
        }
    }

    let explicit_note = if explicit_python.is_some() {
        format!(" Fix or remove {MANAGED_VLLM_PYTHON_ENV}.")
    } else {
        format!(
            " Install the certified GPU backend bundle beside Kapsl or set {MANAGED_VLLM_PYTHON_ENV} to its Python executable."
        )
    };
    Err(format!(
        "no certified managed-vLLM environment is available. Expected Python {}, PyTorch {}, torchvision {}, torchaudio {}, CUDA runtime {}, vLLM {}, and {} {} (profile {}).{} Checked: {}",
        MANAGED_VLLM_PYTHON_VERSION,
        MANAGED_VLLM_TORCH_VERSION,
        MANAGED_VLLM_TORCHVISION_VERSION,
        MANAGED_VLLM_TORCHAUDIO_VERSION,
        MANAGED_VLLM_CUDA_RUNTIME_VERSION,
        MANAGED_VLLM_BACKEND_VERSION,
        MANAGED_VLLM_ADAPTER_ID,
        MANAGED_VLLM_ADAPTER_VERSION,
        MANAGED_VLLM_PROFILE_ID,
        explicit_note,
        if failures.is_empty() {
            "no Python candidates found".to_string()
        } else {
            failures.join("; ")
        }
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CertifiedManagedVllmSource {
    Missing,
    LazyCache,
    Other,
}

/// Identify whether the certified environment selected by the normal vLLM
/// discovery order came from the signed lazy-pack cache. Cache candidates must
/// still pass through `BackendManager` on every run so a valid version probe
/// cannot bypass the signed install record and critical-file hashes. In
/// particular, this function never executes a cache entrypoint before the
/// manager has validated it.
pub(crate) fn certified_managed_vllm_source() -> CertifiedManagedVllmSource {
    // An explicit Python is wholly operator-managed. `prepare` still performs
    // the complete version/profile probe and reports an invalid override; the
    // lazy manager must not silently replace it.
    if std::env::var_os(MANAGED_VLLM_PYTHON_ENV).is_some() {
        return CertifiedManagedVllmSource::Other;
    }

    let cached_python = managed_vllm_cache_python_candidate();
    for candidate in managed_vllm_pre_cache_candidates() {
        if cached_python.as_ref() != Some(&candidate)
            && validate_certified_vllm_python(&candidate).is_ok()
        {
            return CertifiedManagedVllmSource::Other;
        }
    }
    if cached_python
        .as_deref()
        .is_some_and(|path| fs::symlink_metadata(path).is_ok())
    {
        return CertifiedManagedVllmSource::LazyCache;
    }
    if validate_certified_vllm_python(Path::new("python3")).is_ok() {
        CertifiedManagedVllmSource::Other
    } else {
        CertifiedManagedVllmSource::Missing
    }
}

fn managed_vllm_python_candidates() -> Vec<PathBuf> {
    let mut candidates = managed_vllm_pre_cache_candidates();
    if let Some(python) = managed_vllm_cache_python_candidate() {
        candidates.push(python);
    }
    candidates.push(PathBuf::from("python3"));

    let mut seen = HashSet::new();
    candidates
        .into_iter()
        .filter(|candidate| seen.insert(candidate.clone()))
        .collect()
}

fn managed_vllm_pre_cache_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(bundle) = std::env::var_os(MANAGED_VLLM_BUNDLE_ENV) {
        candidates.push(PathBuf::from(bundle).join("bin/python"));
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            candidates.push(parent.join("backends/vllm/bin/python"));
            candidates.push(parent.join("../backends/vllm/bin/python"));
        }
    }
    candidates
}

fn managed_vllm_cache_python_candidate() -> Option<PathBuf> {
    backend_cache_root().map(|cache| {
        cache
            .join(runtime_release_version())
            .join("vllm")
            .join(MANAGED_VLLM_PACK_PROFILE)
            .join("bin/python")
    })
}

fn validate_certified_vllm_python(python: &Path) -> Result<(), String> {
    const PROBE: &str = r#"
import argparse
import importlib.metadata as md
import json
import platform
import torch
from kapsl_vllm_connector import ADAPTER_PROFILE_ID, ADAPTER_VERSION
from kapsl_vllm_connector.planning import PLANNER_SCHEMA_VERSION, planner_json_schema
from vllm.engine.arg_utils import EngineArgs
parser = argparse.ArgumentParser(add_help=False)
EngineArgs.add_cli_args(parser)
exact_cache_memory_flag = any(
    "--kv-cache-memory-bytes" in action.option_strings
    for action in parser._actions
)
planner_schema = planner_json_schema()
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": md.version("torchvision"),
    "torchaudio": md.version("torchaudio"),
    "vllm": md.version("vllm"),
    "connector_distribution": md.version("kapsl-vllm-connector"),
    "connector": ADAPTER_VERSION,
    "profile": ADAPTER_PROFILE_ID,
    "cuda_runtime": str(torch.version.cuda),
    "cuda_available": bool(torch.cuda.is_available()),
    "planner_schema_version": PLANNER_SCHEMA_VERSION,
    "planner_schema_const": planner_schema["properties"]["schema_version"]["const"],
    "exact_cache_memory_flag": exact_cache_memory_flag,
}, sort_keys=True))
"#;
    let output = Command::new(python)
        .args(["-c", PROBE])
        .output()
        .map_err(|error| format!("could not execute Python: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            format!("environment probe exited with {}", output.status)
        } else {
            stderr
        });
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("environment probe emitted non-UTF-8 output: {error}"))?;
    let mut value: serde_json::Value = serde_json::from_str(stdout.trim())
        .map_err(|error| format!("environment probe emitted invalid JSON: {error}"))?;
    let schema_const = value
        .get("planner_schema_const")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "environment probe omitted planner schema const".to_string())?;
    let schema_version = value
        .get("planner_schema_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "environment probe omitted planner schema version".to_string())?;
    if schema_const != schema_version {
        return Err(format!(
            "connector planner schema mismatch: module={schema_version} schema={schema_const}"
        ));
    }
    value
        .as_object_mut()
        .expect("the environment probe was validated as a JSON object")
        .remove("planner_schema_const");
    let actual: ManagedVllmEnvironment = serde_json::from_value(value)
        .map_err(|error| format!("environment probe emitted invalid fields: {error}"))?;
    let expected = ManagedVllmEnvironment {
        python: MANAGED_VLLM_PYTHON_VERSION.to_string(),
        torch: MANAGED_VLLM_TORCH_VERSION.to_string(),
        torchvision: MANAGED_VLLM_TORCHVISION_VERSION.to_string(),
        torchaudio: MANAGED_VLLM_TORCHAUDIO_VERSION.to_string(),
        vllm: MANAGED_VLLM_BACKEND_VERSION.to_string(),
        connector_distribution: MANAGED_VLLM_ADAPTER_VERSION.to_string(),
        connector: MANAGED_VLLM_ADAPTER_VERSION.to_string(),
        profile: MANAGED_VLLM_PROFILE_ID.to_string(),
        cuda_runtime: MANAGED_VLLM_CUDA_RUNTIME_VERSION.to_string(),
        cuda_available: true,
        planner_schema_version: 1,
        exact_cache_memory_flag: true,
    };
    if actual != expected {
        return Err(format!(
            "binary environment mismatch: {actual:?} != {expected:?}"
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
enum ManagedVllmKvCachePolicy {
    Auto {
        target_concurrency: Option<usize>,
        headroom_percent: usize,
        min_bytes: Option<usize>,
        max_bytes: Option<usize>,
        strict: bool,
    },
    Fixed {
        bytes: usize,
    },
    LegacyFraction {
        gpu_memory_utilization: f64,
    },
}

#[derive(Clone, Debug)]
struct ManagedVllmSettings {
    /// Compatibility value used by the process command until the certified
    /// geometry planner and provisional reservation handoff are wired. The
    /// parsed policy is retained separately so enabling exact sizing does not
    /// require another manifest migration.
    gpu_memory_utilization: f64,
    kv_cache_policy: ManagedVllmKvCachePolicy,
    legacy_top_level_fraction_authored: bool,
    max_model_len: usize,
    startup_timeout: Duration,
}

impl ManagedVllmSettings {
    fn from_manifest(manifest: &Manifest) -> Result<Self, String> {
        let mut settings = Self {
            gpu_memory_utilization: DEFAULT_LEGACY_GPU_MEMORY_UTILIZATION,
            kv_cache_policy: ManagedVllmKvCachePolicy::LegacyFraction {
                gpu_memory_utilization: DEFAULT_LEGACY_GPU_MEMORY_UTILIZATION,
            },
            legacy_top_level_fraction_authored: false,
            max_model_len: 1024,
            startup_timeout: DEFAULT_STARTUP_TIMEOUT,
        };
        let Some(vllm) = manifest
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("serving"))
            .and_then(|serving| serving.get("vllm"))
        else {
            return Ok(settings);
        };
        let object = vllm
            .as_mapping()
            .ok_or_else(|| "metadata.serving.vllm must be an object when present".to_string())?;
        let value = |name: &str| object.get(serde_yaml::Value::String(name.to_string()));

        let legacy_fraction = value("gpu_memory_utilization");
        let kv_cache = value("kv_cache");
        if legacy_fraction.is_some() && kv_cache.is_some() {
            return Err(
                "metadata.serving.vllm.gpu_memory_utilization conflicts with metadata.serving.vllm.kv_cache"
                    .to_string(),
            );
        }
        if let Some(raw) = legacy_fraction {
            let utilization =
                parse_gpu_memory_utilization(raw, "metadata.serving.vllm.gpu_memory_utilization")?;
            settings.gpu_memory_utilization = utilization;
            settings.kv_cache_policy = ManagedVllmKvCachePolicy::LegacyFraction {
                gpu_memory_utilization: utilization,
            };
            settings.legacy_top_level_fraction_authored = true;
        } else if let Some(raw) = kv_cache {
            let (policy, compatibility_fraction) = parse_kv_cache_policy(raw)?;
            settings.kv_cache_policy = policy;
            if let Some(utilization) = compatibility_fraction {
                settings.gpu_memory_utilization = utilization;
            }
        }

        if let Some(raw) = value("max_model_len") {
            let raw = raw.as_u64().ok_or_else(|| {
                "metadata.serving.vllm.max_model_len must be a positive integer".to_string()
            })?;
            settings.max_model_len = usize::try_from(raw)
                .map_err(|_| "metadata.serving.vllm.max_model_len is too large".to_string())?;
        }
        if settings.max_model_len == 0 {
            return Err(
                "metadata.serving.vllm.max_model_len must be a positive integer".to_string(),
            );
        }

        if let Some(raw) = value("startup_timeout_seconds") {
            let seconds = raw.as_u64().ok_or_else(|| {
                "metadata.serving.vllm.startup_timeout_seconds must be an integer".to_string()
            })?;
            if !(30..=1800).contains(&seconds) {
                return Err(
                    "metadata.serving.vllm.startup_timeout_seconds must be between 30 and 1800"
                        .to_string(),
                );
            }
            settings.startup_timeout = Duration::from_secs(seconds);
        }
        Ok(settings)
    }

    fn validate_launch_policy(&self) -> Result<(), String> {
        match self.kv_cache_policy {
            ManagedVllmKvCachePolicy::LegacyFraction { .. } => Ok(()),
            ManagedVllmKvCachePolicy::Auto { .. } | ManagedVllmKvCachePolicy::Fixed { .. } => Err(
                "managed vLLM exact KV policy is not launchable until the certified runtime planner and provisional MemoryAuthority grant handoff are installed; use kv_cache.mode: legacy_fraction during this compatibility release"
                    .to_string(),
            ),
        }
    }
}

pub(crate) fn validate_managed_vllm_launch_policy(manifest: &Manifest) -> Result<(), String> {
    ManagedVllmSettings::from_manifest(manifest)?.validate_launch_policy()
}

fn parse_gpu_memory_utilization(raw: &serde_yaml::Value, path: &str) -> Result<f64, String> {
    let utilization = raw
        .as_f64()
        .ok_or_else(|| format!("{path} must be a number"))?;
    if !(0.1..=0.9).contains(&utilization) {
        return Err(format!("{path} must be between 0.1 and 0.9"));
    }
    Ok(utilization)
}

fn parse_kv_cache_policy(
    raw: &serde_yaml::Value,
) -> Result<(ManagedVllmKvCachePolicy, Option<f64>), String> {
    const PATH: &str = "metadata.serving.vllm.kv_cache";
    let object = raw
        .as_mapping()
        .ok_or_else(|| format!("{PATH} must be an object"))?;
    let value = |name: &str| object.get(serde_yaml::Value::String(name.to_string()));
    let mode = value("mode")
        .and_then(serde_yaml::Value::as_str)
        .ok_or_else(|| format!("{PATH}.mode must be one of auto, fixed, or legacy_fraction"))?;

    let allowed = match mode {
        "auto" => &[
            "mode",
            "target_concurrency",
            "headroom_percent",
            "min_bytes",
            "max_bytes",
            "strict",
        ][..],
        "fixed" => &["mode", "bytes"][..],
        "legacy_fraction" => &["mode", "gpu_memory_utilization"][..],
        _ => {
            return Err(format!(
                "{PATH}.mode must be one of auto, fixed, or legacy_fraction"
            ));
        }
    };
    for key in object.keys() {
        let key = key
            .as_str()
            .ok_or_else(|| format!("{PATH} field names must be strings"))?;
        if !allowed.contains(&key) {
            return Err(format!("{PATH}.{key} conflicts with mode {mode}"));
        }
    }

    match mode {
        "auto" => {
            let target_concurrency = value("target_concurrency")
                .map(|raw| parse_positive_usize(raw, &format!("{PATH}.target_concurrency")))
                .transpose()?;
            let headroom_percent = value("headroom_percent")
                .map(|raw| parse_usize(raw, &format!("{PATH}.headroom_percent")))
                .transpose()?
                .unwrap_or(DEFAULT_KV_HEADROOM_PERCENT);
            if headroom_percent > MAX_KV_HEADROOM_PERCENT {
                return Err(format!(
                    "{PATH}.headroom_percent must be between 0 and {MAX_KV_HEADROOM_PERCENT}"
                ));
            }
            let min_bytes = value("min_bytes")
                .map(|raw| parse_positive_usize(raw, &format!("{PATH}.min_bytes")))
                .transpose()?;
            let max_bytes = value("max_bytes")
                .map(|raw| parse_positive_usize(raw, &format!("{PATH}.max_bytes")))
                .transpose()?;
            if min_bytes.zip(max_bytes).is_some_and(|(min, max)| min > max) {
                return Err(format!("{PATH}.min_bytes must not exceed {PATH}.max_bytes"));
            }
            let strict = value("strict")
                .map(|raw| {
                    raw.as_bool()
                        .ok_or_else(|| format!("{PATH}.strict must be a boolean"))
                })
                .transpose()?
                .unwrap_or(false);
            Ok((
                ManagedVllmKvCachePolicy::Auto {
                    target_concurrency,
                    headroom_percent,
                    min_bytes,
                    max_bytes,
                    strict,
                },
                None,
            ))
        }
        "fixed" => {
            let bytes = value("bytes")
                .ok_or_else(|| format!("{PATH}.bytes is required when mode is fixed"))
                .and_then(|raw| parse_positive_usize(raw, &format!("{PATH}.bytes")))?;
            Ok((ManagedVllmKvCachePolicy::Fixed { bytes }, None))
        }
        "legacy_fraction" => {
            let utilization = value("gpu_memory_utilization")
                .map(|raw| {
                    parse_gpu_memory_utilization(raw, &format!("{PATH}.gpu_memory_utilization"))
                })
                .transpose()?
                .unwrap_or(DEFAULT_LEGACY_GPU_MEMORY_UTILIZATION);
            Ok((
                ManagedVllmKvCachePolicy::LegacyFraction {
                    gpu_memory_utilization: utilization,
                },
                Some(utilization),
            ))
        }
        _ => unreachable!("mode was validated above"),
    }
}

fn parse_usize(raw: &serde_yaml::Value, path: &str) -> Result<usize, String> {
    let value = raw
        .as_u64()
        .ok_or_else(|| format!("{path} must be a non-negative integer"))?;
    usize::try_from(value).map_err(|_| format!("{path} is too large"))
}

fn parse_positive_usize(raw: &serde_yaml::Value, path: &str) -> Result<usize, String> {
    let value = parse_usize(raw, path)?;
    if value == 0 {
        return Err(format!("{path} must be a positive integer"));
    }
    Ok(value)
}

/// One standard, non-hybrid cache group used only by the Rust diagnostic
/// cross-check. Hybrid/shared-pool geometry remains authoritative in the
/// certified connector planner because its groups alias one physical stride.
/// `prefix_retention_blocks` is a bounded operator policy, not a fraction of
/// device memory.
#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct ManagedVllmKvCacheGroupGeometry {
    group_id: String,
    block_size_tokens: usize,
    bytes_per_block: usize,
    block_alignment: usize,
    prefix_retention_blocks: usize,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct ManagedVllmKvCacheGeometry {
    groups: Vec<ManagedVllmKvCacheGroupGeometry>,
    allocation_alignment_bytes: usize,
    fixed_overhead_bytes: usize,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct ManagedVllmKvCacheGroupSizing {
    group_id: String,
    sequence_blocks: usize,
    granted_blocks: usize,
    granted_bytes: usize,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct ManagedVllmKvCacheSizing {
    requested_bytes: usize,
    granted_bytes: usize,
    minimum_required_bytes: usize,
    target_concurrency: usize,
    effective_target_concurrency: usize,
    headroom_reduced: bool,
    unassigned_bytes: usize,
    groups: Vec<ManagedVllmKvCacheGroupSizing>,
}

#[derive(Clone, Debug)]
struct CalculatedKvCacheSizing {
    bytes: usize,
    groups: Vec<ManagedVllmKvCacheGroupSizing>,
}

/// Resolve an exact initial cache request from certified geometry. This is
/// deliberately host-only for now: the managed process continues using its
/// compatibility fraction until planner output is certified and a
/// `MemoryAuthority` grant can be transferred into connector registration.
#[allow(dead_code)]
fn size_managed_vllm_kv_cache(
    policy: &ManagedVllmKvCachePolicy,
    geometry: &ManagedVllmKvCacheGeometry,
    max_model_len: usize,
    resolved_target_concurrency: usize,
    available_bytes: usize,
) -> Result<ManagedVllmKvCacheSizing, String> {
    validate_kv_cache_geometry(geometry)?;
    if max_model_len == 0 {
        return Err("managed vLLM KV sizing requires a positive max_model_len".to_string());
    }

    let minimum = calculate_kv_cache_sizing(geometry, max_model_len, 1, 0, false)?;
    match policy {
        ManagedVllmKvCachePolicy::LegacyFraction { .. } => Err(
            "legacy_fraction is a vLLM-local policy and does not produce an exact KV byte plan"
                .to_string(),
        ),
        ManagedVllmKvCachePolicy::Fixed { bytes } => {
            if *bytes == 0 {
                return Err("fixed managed vLLM KV bytes must be positive".to_string());
            }
            let group = &geometry.groups[0];
            let block_alignment = effective_kv_block_alignment(geometry, group)?;
            if bytes % group.bytes_per_block != 0 {
                return Err(format!(
                    "fixed managed vLLM KV bytes {} are not an exact multiple of the {}-byte packed block stride",
                    bytes, group.bytes_per_block
                ));
            }
            let granted_blocks = bytes / group.bytes_per_block;
            if !granted_blocks.is_multiple_of(block_alignment) {
                return Err(format!(
                    "fixed managed vLLM KV block count {granted_blocks} is not aligned to the certified {block_alignment}-block quantum"
                ));
            }
            if *bytes < minimum.bytes {
                return Err(format!(
                    "fixed managed vLLM KV bytes {} are below the one-sequence minimum {}",
                    bytes, minimum.bytes
                ));
            }
            if *bytes > available_bytes {
                return Err(format!(
                    "fixed managed vLLM KV bytes {} exceed available authority bytes {}",
                    bytes, available_bytes
                ));
            }
            Ok(ManagedVllmKvCacheSizing {
                requested_bytes: *bytes,
                granted_bytes: *bytes,
                minimum_required_bytes: minimum.bytes,
                target_concurrency: 1,
                effective_target_concurrency: 1,
                headroom_reduced: false,
                unassigned_bytes: 0,
                groups: vec![ManagedVllmKvCacheGroupSizing {
                    group_id: group.group_id.clone(),
                    sequence_blocks: minimum.groups[0].sequence_blocks,
                    granted_blocks,
                    granted_bytes: *bytes,
                }],
            })
        }
        ManagedVllmKvCachePolicy::Auto {
            target_concurrency,
            headroom_percent,
            min_bytes,
            max_bytes,
            strict,
        } => {
            if *headroom_percent > MAX_KV_HEADROOM_PERCENT {
                return Err(format!(
                    "managed vLLM KV headroom_percent must be between 0 and {MAX_KV_HEADROOM_PERCENT}"
                ));
            }
            let target_concurrency = target_concurrency.unwrap_or(resolved_target_concurrency);
            if target_concurrency == 0 {
                return Err(
                    "managed vLLM KV target concurrency must be a positive integer".to_string(),
                );
            }
            if min_bytes.is_some_and(|bytes| bytes == 0)
                || max_bytes.is_some_and(|bytes| bytes == 0)
            {
                return Err("managed vLLM KV min/max bytes must be positive".to_string());
            }
            if min_bytes
                .zip(*max_bytes)
                .is_some_and(|(min, max)| min > max)
            {
                return Err("managed vLLM KV min_bytes must not exceed max_bytes".to_string());
            }
            if min_bytes.is_some_and(|bytes| bytes < minimum.bytes) {
                return Err(format!(
                    "managed vLLM KV min_bytes is below the one-sequence minimum {}",
                    minimum.bytes
                ));
            }
            if max_bytes.is_some_and(|bytes| bytes < minimum.bytes) {
                return Err(format!(
                    "managed vLLM KV max_bytes is below the one-sequence minimum {}",
                    minimum.bytes
                ));
            }

            let group = &geometry.groups[0];
            let block_alignment = effective_kv_block_alignment(geometry, group)?;
            let floor_blocks = kv_blocks_for_byte_floor(
                min_bytes.unwrap_or(minimum.bytes),
                group.bytes_per_block,
                block_alignment,
                "managed vLLM KV min_bytes",
            )?;
            let minimum_floor = floor_blocks
                .checked_mul(group.bytes_per_block)
                .ok_or_else(|| "managed vLLM KV min_bytes overflowed".to_string())?;
            if max_bytes.is_some_and(|bytes| bytes < minimum_floor) {
                return Err(format!(
                    "managed vLLM KV max_bytes is below the aligned minimum floor {minimum_floor}"
                ));
            }
            let authority_cap = available_bytes.min(max_bytes.unwrap_or(usize::MAX));
            let cap_blocks =
                kv_blocks_for_byte_cap(authority_cap, group.bytes_per_block, block_alignment);
            if cap_blocks < floor_blocks {
                return Err(format!(
                    "managed vLLM KV minimum {} does not fit available capped authority bytes {}",
                    minimum_floor, authority_cap
                ));
            }

            let requested = calculate_kv_cache_sizing(
                geometry,
                max_model_len,
                target_concurrency,
                *headroom_percent,
                true,
            )?;
            let requested_blocks = requested.groups[0].granted_blocks.max(floor_blocks);
            let requested_bytes = requested_blocks
                .checked_mul(group.bytes_per_block)
                .ok_or_else(|| "managed vLLM KV requested bytes overflowed".to_string())?;
            let granted_blocks = requested_blocks.min(cap_blocks);
            let granted_bytes = granted_blocks
                .checked_mul(group.bytes_per_block)
                .ok_or_else(|| "managed vLLM KV granted bytes overflowed".to_string())?;

            let fixed_blocks = fixed_kv_overhead_blocks(geometry, group)?;
            let sequence_blocks = requested.groups[0].sequence_blocks;
            // Prefix retention and headroom are optional allowances. A cap
            // may shed either without reducing the number of whole
            // max-length sequences that the remaining pool can serve.
            let usable_blocks = granted_blocks.saturating_sub(fixed_blocks);
            let effective_target_concurrency =
                target_concurrency.min(usable_blocks / sequence_blocks);
            if effective_target_concurrency == 0 {
                return Err(
                    "resolved managed vLLM KV grant cannot hold one maximum-length sequence"
                        .to_string(),
                );
            }
            if *strict && effective_target_concurrency < target_concurrency {
                return Err(format!(
                    "strict managed vLLM KV target concurrency {target_concurrency} requires more than the {granted_bytes}-byte exact grant"
                ));
            }
            Ok(ManagedVllmKvCacheSizing {
                requested_bytes,
                granted_bytes,
                minimum_required_bytes: minimum.bytes,
                target_concurrency,
                effective_target_concurrency,
                headroom_reduced: *headroom_percent > 0 && granted_blocks < requested_blocks,
                unassigned_bytes: 0,
                groups: vec![ManagedVllmKvCacheGroupSizing {
                    group_id: group.group_id.clone(),
                    sequence_blocks,
                    granted_blocks,
                    granted_bytes,
                }],
            })
        }
    }
}

fn validate_kv_cache_geometry(geometry: &ManagedVllmKvCacheGeometry) -> Result<(), String> {
    if geometry.groups.len() != 1 {
        return Err(
            "managed vLLM Rust KV diagnostic supports exactly one standard cache group; hybrid geometry requires the certified connector planner"
                .to_string(),
        );
    }
    if geometry.allocation_alignment_bytes == 0 {
        return Err("managed vLLM KV allocation_alignment_bytes must be positive".to_string());
    }
    let mut group_ids = HashSet::new();
    for group in &geometry.groups {
        if group.group_id.trim().is_empty() || !group_ids.insert(group.group_id.as_str()) {
            return Err("managed vLLM KV group IDs must be non-empty and unique".to_string());
        }
        if group.block_size_tokens == 0 || group.bytes_per_block == 0 || group.block_alignment == 0
        {
            return Err(format!(
                "managed vLLM KV group '{}' geometry values must be positive",
                group.group_id
            ));
        }
        if !geometry
            .fixed_overhead_bytes
            .is_multiple_of(group.bytes_per_block)
        {
            return Err(format!(
                "managed vLLM KV fixed overhead {} is not representable by the {}-byte packed block stride",
                geometry.fixed_overhead_bytes, group.bytes_per_block
            ));
        }
        let _ = effective_kv_block_alignment(geometry, group)?;
    }
    Ok(())
}

fn calculate_kv_cache_sizing(
    geometry: &ManagedVllmKvCacheGeometry,
    max_model_len: usize,
    concurrency: usize,
    headroom_percent: usize,
    include_optional_prefix: bool,
) -> Result<CalculatedKvCacheSizing, String> {
    let group = &geometry.groups[0];
    let sequence_blocks = checked_ceil_div(
        max_model_len,
        group.block_size_tokens,
        "managed vLLM sequence block calculation",
    )?;
    let mut workload_blocks = sequence_blocks.checked_mul(concurrency).ok_or_else(|| {
        format!(
            "managed vLLM KV block count overflow for group '{}'",
            group.group_id
        )
    })?;
    if include_optional_prefix {
        workload_blocks = workload_blocks
            .checked_add(group.prefix_retention_blocks)
            .ok_or_else(|| {
                format!(
                    "managed vLLM KV prefix block count overflow for group '{}'",
                    group.group_id
                )
            })?;
    }
    let headroom = checked_percent_ceil(workload_blocks, headroom_percent, &group.group_id)?;
    let fixed_blocks = fixed_kv_overhead_blocks(geometry, group)?;
    let blocks = fixed_blocks
        .checked_add(workload_blocks)
        .and_then(|blocks| blocks.checked_add(headroom))
        .ok_or_else(|| {
            format!(
                "managed vLLM KV total block count overflow for group '{}'",
                group.group_id
            )
        })?;
    // Match the certified Python planner: add bounded headroom first, then
    // apply the one final block/alocation alignment quantum.
    let blocks = checked_round_up(
        blocks,
        effective_kv_block_alignment(geometry, group)?,
        &format!(
            "managed vLLM KV final block alignment for group '{}'",
            group.group_id
        ),
    )?;
    let bytes = blocks.checked_mul(group.bytes_per_block).ok_or_else(|| {
        format!(
            "managed vLLM KV byte count overflow for group '{}'",
            group.group_id
        )
    })?;
    Ok(CalculatedKvCacheSizing {
        bytes,
        groups: vec![ManagedVllmKvCacheGroupSizing {
            group_id: group.group_id.clone(),
            sequence_blocks,
            granted_blocks: blocks,
            granted_bytes: bytes,
        }],
    })
}

fn fixed_kv_overhead_blocks(
    geometry: &ManagedVllmKvCacheGeometry,
    group: &ManagedVllmKvCacheGroupGeometry,
) -> Result<usize, String> {
    if !geometry
        .fixed_overhead_bytes
        .is_multiple_of(group.bytes_per_block)
    {
        return Err(format!(
            "managed vLLM KV fixed overhead {} is not representable by the {}-byte packed block stride",
            geometry.fixed_overhead_bytes, group.bytes_per_block
        ));
    }
    Ok(geometry.fixed_overhead_bytes / group.bytes_per_block)
}

fn effective_kv_block_alignment(
    geometry: &ManagedVllmKvCacheGeometry,
    group: &ManagedVllmKvCacheGroupGeometry,
) -> Result<usize, String> {
    let allocation_quantum = geometry.allocation_alignment_bytes
        / greatest_common_divisor(geometry.allocation_alignment_bytes, group.bytes_per_block);
    checked_least_common_multiple(group.block_alignment, allocation_quantum).ok_or_else(|| {
        format!(
            "managed vLLM KV alignment overflow for group '{}'",
            group.group_id
        )
    })
}

fn kv_blocks_for_byte_floor(
    bytes: usize,
    bytes_per_block: usize,
    block_alignment: usize,
    context: &str,
) -> Result<usize, String> {
    checked_round_up(
        checked_ceil_div(bytes, bytes_per_block, context)?,
        block_alignment,
        context,
    )
}

fn kv_blocks_for_byte_cap(bytes: usize, bytes_per_block: usize, block_alignment: usize) -> usize {
    (bytes / bytes_per_block) / block_alignment * block_alignment
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn checked_least_common_multiple(left: usize, right: usize) -> Option<usize> {
    left.checked_div(greatest_common_divisor(left, right))?
        .checked_mul(right)
}

fn checked_ceil_div(value: usize, divisor: usize, context: &str) -> Result<usize, String> {
    if divisor == 0 {
        return Err(format!("{context} has a zero divisor"));
    }
    let quotient = value / divisor;
    quotient
        .checked_add(usize::from(!value.is_multiple_of(divisor)))
        .ok_or_else(|| format!("{context} overflowed"))
}

fn checked_round_up(value: usize, alignment: usize, context: &str) -> Result<usize, String> {
    if alignment == 0 {
        return Err(format!("{context} has zero alignment"));
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| format!("{context} overflowed"))
    }
}

fn checked_percent_ceil(value: usize, percent: usize, group_id: &str) -> Result<usize, String> {
    let numerator = (value as u128)
        .checked_mul(percent as u128)
        .ok_or_else(|| format!("managed vLLM KV headroom overflow for group '{group_id}'"))?;
    let rounded = numerator
        .checked_add(99)
        .ok_or_else(|| format!("managed vLLM KV headroom overflow for group '{group_id}'"))?
        / 100;
    usize::try_from(rounded)
        .map_err(|_| format!("managed vLLM KV headroom overflow for group '{group_id}'"))
}

#[derive(Clone, Debug)]
struct ManagedVllmProcessSpec {
    python: PathBuf,
    model_root: PathBuf,
    served_model_name: String,
    endpoint: String,
    port: u16,
    kv_transfer_config: String,
    log_path: PathBuf,
    settings: ManagedVllmSettings,
    tensor_parallel_size: usize,
    cuda_visible_devices: String,
}

struct ManagedVllmProcess {
    spec: ManagedVllmProcessSpec,
    bridge: ManagedVllmHttpBridge,
    lifecycle: Mutex<()>,
    child: Mutex<Option<Child>>,
    shutdown: AtomicBool,
    restarts: AtomicU32,
    readiness: AtomicU8,
    readiness_epoch: AtomicU64,
    kv_readiness_fence: Arc<ManagedVllmKvReadinessFence>,
    published_kv_readiness_epoch: AtomicU64,
    generation: AtomicU64,
}

impl ManagedVllmProcess {
    fn new(spec: ManagedVllmProcessSpec) -> Self {
        let bridge = ManagedVllmHttpBridge::new(&spec.endpoint)
            .expect("managed vLLM process specs always use a private HTTP origin");
        Self {
            spec,
            bridge,
            lifecycle: Mutex::new(()),
            child: Mutex::new(None),
            shutdown: AtomicBool::new(false),
            restarts: AtomicU32::new(0),
            readiness: AtomicU8::new(ManagedVllmReplicaState::Planned as u8),
            readiness_epoch: AtomicU64::new(0),
            kv_readiness_fence: Arc::new(ManagedVllmKvReadinessFence::new()),
            published_kv_readiness_epoch: AtomicU64::new(0),
            generation: AtomicU64::new(0),
        }
    }

    fn build_command(&self) -> Result<Command, EngineError> {
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.spec.log_path)
            .map_err(|error| {
                EngineError::backend(format!(
                    "open managed vLLM log {}: {error}",
                    self.spec.log_path.display()
                ))
            })?;
        let stderr = log_file.try_clone().map_err(|error| {
            EngineError::backend(format!(
                "clone managed vLLM log {}: {error}",
                self.spec.log_path.display()
            ))
        })?;

        let mut command = Command::new(&self.spec.python);
        command
            .arg("-m")
            .arg("vllm.entrypoints.openai.api_server")
            .arg("--model")
            .arg(&self.spec.model_root)
            .args(["--served-model-name", &self.spec.served_model_name])
            .args(["--host", "127.0.0.1"])
            .args(["--port", &self.spec.port.to_string()])
            .args(["--attention-backend", "FLASH_ATTN"])
            .args([
                "--gpu-memory-utilization",
                &self.spec.settings.gpu_memory_utilization.to_string(),
            ])
            .args([
                "--max-model-len",
                &self.spec.settings.max_model_len.to_string(),
            ])
            .arg("--enforce-eager")
            .args(["--kv-transfer-config", &self.spec.kv_transfer_config])
            .env("CUDA_VISIBLE_DEVICES", &self.spec.cuda_visible_devices)
            .env("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(stderr));
        if self.spec.tensor_parallel_size > 1 {
            command.args([
                "--tensor-parallel-size",
                &self.spec.tensor_parallel_size.to_string(),
            ]);
        }
        if let Some(bin_dir) = self
            .spec
            .python
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            let mut paths = vec![bin_dir.to_path_buf()];
            if let Some(existing) = std::env::var_os("PATH") {
                paths.extend(std::env::split_paths(&existing));
            }
            let path = std::env::join_paths(paths).map_err(|error| {
                EngineError::backend(format!("build managed vLLM PATH: {error}"))
            })?;
            command.env("PATH", path);
        }
        #[cfg(unix)]
        command.process_group(0);
        Ok(command)
    }

    fn spawn_child(&self) -> Result<(), EngineError> {
        // Serialize spawn with termination. Otherwise terminate() can observe
        // no child while Command::spawn is in progress and leave the process
        // installed after shutdown.
        let _lifecycle = self.lifecycle.lock();
        if self.shutdown.load(Ordering::Acquire) {
            return Err(EngineError::backend(
                "managed vLLM process has been shut down".to_string(),
            ));
        }
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.readiness
            .store(ManagedVllmReplicaState::Starting as u8, Ordering::Release);
        self.readiness_epoch.fetch_add(1, Ordering::AcqRel);
        let mut command = match self.build_command() {
            Ok(command) => command,
            Err(error) => {
                self.mark_suspect_locked();
                return Err(error);
            }
        };
        let child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                self.mark_suspect_locked();
                return Err(EngineError::backend(format!(
                    "launch managed vLLM with {}: {error}",
                    self.spec.python.display()
                )));
            }
        };
        log::info!(
            "Managed vLLM process started: pid={} endpoint={} log={}",
            child.id(),
            self.spec.endpoint,
            self.spec.log_path.display()
        );
        *self.child.lock() = Some(child);
        Ok(())
    }

    fn state(&self) -> ManagedVllmReplicaState {
        ManagedVllmReplicaState::from_u8(self.readiness.load(Ordering::Acquire))
    }

    fn mark_suspect(&self) {
        let _lifecycle = self.lifecycle.lock();
        self.mark_suspect_locked();
    }

    fn mark_suspect_locked(&self) {
        if self.state() != ManagedVllmReplicaState::Stopped {
            self.readiness
                .store(ManagedVllmReplicaState::Suspect as u8, Ordering::Release);
            // Increment even for Suspect -> Suspect so a transport failure
            // that races an in-flight health probe fences that stale result.
            self.readiness_epoch.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn readiness_snapshot(&self) -> (u64, u64, u64) {
        let _lifecycle = self.lifecycle.lock();
        (
            self.generation.load(Ordering::Acquire),
            self.readiness_epoch.load(Ordering::Acquire),
            self.kv_readiness_fence.snapshot(),
        )
    }

    fn mark_routable(
        &self,
        generation: u64,
        readiness_epoch: u64,
        kv_readiness_epoch: u64,
    ) -> bool {
        let _lifecycle = self.lifecycle.lock();
        if self.shutdown.load(Ordering::Acquire)
            || self.generation.load(Ordering::Acquire) != generation
            || self.readiness_epoch.load(Ordering::Acquire) != readiness_epoch
            || self.kv_readiness_fence.snapshot() != kv_readiness_epoch
        {
            return false;
        }
        match self.state() {
            ManagedVllmReplicaState::Starting
            | ManagedVllmReplicaState::Suspect
            | ManagedVllmReplicaState::Routable => {
                if self.state() == ManagedVllmReplicaState::Routable
                    && self.published_kv_readiness_epoch.load(Ordering::Acquire)
                        == kv_readiness_epoch
                {
                    return true;
                }
                self.published_kv_readiness_epoch
                    .store(kv_readiness_epoch, Ordering::Release);
                self.readiness
                    .store(ManagedVllmReplicaState::Routable as u8, Ordering::Release);
                // Detach advances the shared epoch before changing coordinator
                // state. Recheck after publication so a detach that overlaps
                // this critical section cannot leave a usable Routable state.
                if self.kv_readiness_fence.snapshot() != kv_readiness_epoch {
                    self.mark_suspect_locked();
                    return false;
                }
                self.readiness_epoch.fetch_add(1, Ordering::AcqRel);
                true
            }
            ManagedVllmReplicaState::Planned | ManagedVllmReplicaState::Stopped => false,
        }
    }

    fn ensure_routable(&self) -> Result<(), EngineError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(EngineError::backend(
                "managed vLLM process has been shut down".to_string(),
            ));
        }
        let state = self.state();
        let published_kv_epoch = self.published_kv_readiness_epoch.load(Ordering::Acquire);
        let current_kv_epoch = self.kv_readiness_fence.snapshot();
        if state != ManagedVllmReplicaState::Routable || published_kv_epoch != current_kv_epoch {
            if state == ManagedVllmReplicaState::Routable && published_kv_epoch != current_kv_epoch
            {
                self.mark_suspect();
            }
            return Err(EngineError::backend(format!(
                "managed vLLM replica is not routable (state={:?}, generation={}, kv_readiness_epoch={current_kv_epoch}, published_kv_readiness_epoch={published_kv_epoch})",
                self.state(),
                self.generation.load(Ordering::Acquire),
            )));
        }
        if let Some(status) = self.try_wait()? {
            self.mark_suspect();
            return Err(EngineError::backend(format!(
                "managed vLLM process exited with {status}; log: {}",
                self.spec.log_path.display()
            )));
        }
        Ok(())
    }

    fn try_wait(&self) -> Result<Option<ExitStatus>, EngineError> {
        let mut child = self.child.lock();
        let Some(child) = child.as_mut() else {
            return Err(EngineError::backend(
                "managed vLLM process has not been started".to_string(),
            ));
        };
        child
            .try_wait()
            .map_err(|error| EngineError::backend(format!("inspect managed vLLM process: {error}")))
    }

    async fn probe_health(&self) -> Result<(), EngineError> {
        if let Some(status) = self.try_wait()? {
            self.mark_suspect();
            return Err(EngineError::backend(format!(
                "managed vLLM process exited with {status}; log: {}",
                self.spec.log_path.display()
            )));
        }
        self.bridge
            .check_health(
                "/health",
                ManagedVllmRequestTimeouts::new(HEALTH_TIMEOUT, HEALTH_TIMEOUT, HEALTH_TIMEOUT),
                None,
            )
            .await
            .map_err(|error| bridge_error_to_engine(error, "managed vLLM health check"))
    }

    async fn wait_ready(
        &self,
        deployment: &ManagedVllmDeployment,
        participant_base: &str,
    ) -> Result<(), EngineError> {
        let deadline = Instant::now() + self.spec.settings.startup_timeout;
        let generation = self.generation.load(Ordering::Acquire);
        loop {
            if let Some(status) = self.try_wait()? {
                self.mark_suspect();
                return Err(EngineError::backend(format!(
                    "managed vLLM exited before readiness with {status}; log: {}",
                    self.spec.log_path.display()
                )));
            }
            let (probe_generation, readiness_epoch, kv_readiness_epoch) = self.readiness_snapshot();
            if probe_generation != generation {
                return Err(EngineError::backend(
                    "managed vLLM readiness observed a different process generation".to_string(),
                ));
            }
            if self.probe_health().await.is_ok()
                && deployment
                    .participant_is_active(participant_base)
                    .map_err(|error| {
                        EngineError::backend(format!(
                            "inspect managed vLLM KV activation readiness: {error}"
                        ))
                    })?
            {
                if self.mark_routable(generation, readiness_epoch, kv_readiness_epoch) {
                    return Ok(());
                }
                // A concurrent lifecycle transition fenced this probe. A
                // shutdown or new generation is terminal; a same-generation
                // suspect transition gets a fresh health probe.
                if self.shutdown.load(Ordering::Acquire)
                    || self.generation.load(Ordering::Acquire) != generation
                {
                    return Err(EngineError::backend(
                        "managed vLLM readiness completed for a stale process generation"
                            .to_string(),
                    ));
                }
            }
            if Instant::now() >= deadline {
                self.mark_suspect();
                return Err(EngineError::timeout(format!(
                    "timed out after {}s waiting for managed vLLM; log: {}",
                    self.spec.settings.startup_timeout.as_secs(),
                    self.spec.log_path.display()
                )));
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    fn stop_child(&self) -> bool {
        self.mark_suspect();
        let mut child_guard = self.child.lock();
        let Some(child) = child_guard.as_mut() else {
            return true;
        };
        #[cfg(unix)]
        {
            let process_group = child.id() as i32;
            let _ = child.try_wait();
            if !process_group_alive(process_group) {
                return true;
            }
            unsafe {
                let _ = libc::kill(-process_group, libc::SIGTERM);
            }

            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                let _ = child.try_wait();
                if !process_group_alive(process_group) {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            unsafe {
                let _ = libc::kill(-process_group, libc::SIGKILL);
            }
            let _ = child.kill();
            let _ = child.wait();

            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline {
                if !process_group_alive(process_group) {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            log::error!(
                "managed vLLM process group {} still exists after forced termination; retaining its Kapsl-owned shared pool",
                process_group,
            );
            false
        }
        #[cfg(not(unix))]
        {
            if child.try_wait().ok().flatten().is_some() {
                return true;
            }
            let _ = child.kill();
            let _ = child.wait();
            true
        }
    }

    fn terminate(&self) -> bool {
        {
            let _lifecycle = self.lifecycle.lock();
            self.shutdown.store(true, Ordering::Release);
            self.readiness
                .store(ManagedVllmReplicaState::Stopped as u8, Ordering::Release);
            self.readiness_epoch.fetch_add(1, Ordering::AcqRel);
        }
        self.stop_child()
    }
}

fn managed_vllm_request_timeouts(request: &InferenceRequest) -> ManagedVllmRequestTimeouts {
    managed_vllm_timeouts(
        request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.timeout_ms),
    )
}

fn managed_vllm_wire_request_timeouts(request: &OpenAiWireRequest) -> ManagedVllmRequestTimeouts {
    managed_vllm_timeouts(
        request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.timeout_ms),
    )
}

fn managed_vllm_timeouts(timeout_ms: Option<u64>) -> ManagedVllmRequestTimeouts {
    let total = timeout_ms
        .filter(|timeout_ms| *timeout_ms > 0)
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_REQUEST_TIMEOUT);
    ManagedVllmRequestTimeouts::new(
        BRIDGE_HEADER_TIMEOUT.min(total),
        BRIDGE_IDLE_BODY_TIMEOUT.min(total),
        total,
    )
}

fn managed_vllm_wire_response_head(
    status: hyper::StatusCode,
    headers: &hyper::HeaderMap,
) -> Result<OpenAiWireResponseHead, EngineError> {
    let mut forwarded = Vec::new();
    for (source, target) in [
        (
            hyper::header::CONTENT_TYPE,
            OpenAiWireHeaderName::ContentType,
        ),
        (
            hyper::header::CACHE_CONTROL,
            OpenAiWireHeaderName::CacheControl,
        ),
        (
            hyper::header::HeaderName::from_static("x-request-id"),
            OpenAiWireHeaderName::RequestId,
        ),
        (hyper::header::RETRY_AFTER, OpenAiWireHeaderName::RetryAfter),
        (
            hyper::header::HeaderName::from_static("openai-processing-ms"),
            OpenAiWireHeaderName::ProcessingMilliseconds,
        ),
    ] {
        for value in headers.get_all(source).iter() {
            forwarded.push(OpenAiWireHeader::new(target, value.as_bytes().to_vec())?);
        }
    }
    OpenAiWireResponseHead::new(status.as_u16(), forwarded)
}

fn bridge_error_is_transport_failure(error: &ManagedVllmBridgeError) -> bool {
    matches!(
        error,
        ManagedVllmBridgeError::Request(_) | ManagedVllmBridgeError::Body(_)
    )
}

fn bridge_error_to_engine(error: ManagedVllmBridgeError, context: &str) -> EngineError {
    match error {
        ManagedVllmBridgeError::Cancelled => {
            EngineError::cancelled(format!("{context}: request was cancelled"))
        }
        ManagedVllmBridgeError::HeaderTimeout
        | ManagedVllmBridgeError::IdleBodyTimeout
        | ManagedVllmBridgeError::TotalTimeout => {
            EngineError::timeout(format!("{context}: {error}"))
        }
        ManagedVllmBridgeError::UpstreamStatus { status, body } => EngineError::backend(format!(
            "{context}: upstream HTTP {status}: {}",
            String::from_utf8_lossy(&body),
        )),
        error => EngineError::backend(format!("{context}: {error}")),
    }
}

fn relay_managed_vllm_wire_stream(
    upstream: ManagedVllmByteStream,
    process: Arc<ManagedVllmProcess>,
    errors: Arc<AtomicU64>,
) -> OpenAiWireStream {
    Box::pin(upstream.map(move |item| {
        item.map_err(|error| {
            if bridge_error_is_transport_failure(&error) {
                process.mark_suspect();
            }
            errors.fetch_add(1, Ordering::Relaxed);
            bridge_error_to_engine(error, "read managed vLLM wire stream")
        })
    }))
}

struct ManagedVllmTranslatedStreamState {
    upstream: ManagedVllmSseStream,
    process: Arc<ManagedVllmProcess>,
    errors: Arc<AtomicU64>,
    generated_tokens: Arc<AtomicU64>,
    chat: bool,
    terminal: bool,
}

fn translate_managed_vllm_stream(
    upstream: ManagedVllmSseStream,
    process: Arc<ManagedVllmProcess>,
    errors: Arc<AtomicU64>,
    generated_tokens: Arc<AtomicU64>,
    chat: bool,
) -> EngineStream {
    let state = ManagedVllmTranslatedStreamState {
        upstream,
        process,
        errors,
        generated_tokens,
        chat,
        terminal: false,
    };
    Box::pin(stream::unfold(state, |mut state| async move {
        if state.terminal {
            return None;
        }
        loop {
            let event = match state.upstream.next().await {
                Some(event) => event,
                None => {
                    state.errors.fetch_add(1, Ordering::Relaxed);
                    state.terminal = true;
                    return Some((
                        Err(EngineError::backend(
                            "managed vLLM stream ended before the terminal [DONE] event"
                                .to_string(),
                        )),
                        state,
                    ));
                }
            };
            let data = match event {
                Ok(data) => data,
                Err(error) => {
                    if bridge_error_is_transport_failure(&error) {
                        state.process.mark_suspect();
                    }
                    state.errors.fetch_add(1, Ordering::Relaxed);
                    state.terminal = true;
                    return Some((
                        Err(bridge_error_to_engine(error, "read managed vLLM stream")),
                        state,
                    ));
                }
            };
            if data == b"[DONE]" {
                return None;
            }
            if data.is_empty() {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_slice(&data) {
                Ok(value) => value,
                Err(error) => {
                    state.errors.fetch_add(1, Ordering::Relaxed);
                    state.terminal = true;
                    return Some((
                        Err(EngineError::backend(format!(
                            "decode managed vLLM stream event: {error}"
                        ))),
                        state,
                    ));
                }
            };
            if let Some(upstream_error) = value.get("error") {
                state.errors.fetch_add(1, Ordering::Relaxed);
                state.terminal = true;
                return Some((
                    Err(EngineError::backend(format!(
                        "managed vLLM stream returned an upstream error: {upstream_error}"
                    ))),
                    state,
                ));
            }
            let choice = value
                .get("choices")
                .and_then(serde_json::Value::as_array)
                .and_then(|choices| choices.first());
            let text = if state.chat {
                choice
                    .and_then(|choice| choice.get("delta"))
                    .and_then(|delta| delta.get("content"))
                    .and_then(serde_json::Value::as_str)
            } else {
                choice
                    .and_then(|choice| choice.get("text"))
                    .and_then(serde_json::Value::as_str)
            };
            let Some(text) = text.filter(|text| !text.is_empty()) else {
                continue;
            };
            state
                .generated_tokens
                .fetch_add(estimate_vllm_tokens(text), Ordering::Relaxed);
            return Some((ManagedVllmEngine::output_packet(text.to_string()), state));
        }
    }))
}

impl Drop for ManagedVllmProcess {
    fn drop(&mut self) {
        let _ = self.terminate();
    }
}

#[cfg(unix)]
fn process_group_alive(process_group: i32) -> bool {
    let result = unsafe { libc::kill(-process_group, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

fn start_managed_vllm_supervisor(
    process: Arc<ManagedVllmProcess>,
    deployment: Arc<ManagedVllmDeployment>,
    participant_id: String,
) {
    tokio::spawn(async move {
        let mut consecutive_health_failures = 0u8;
        loop {
            tokio::time::sleep(SUPERVISOR_INTERVAL).await;
            if process.shutdown.load(Ordering::Acquire) {
                break;
            }

            let exited = matches!(process.try_wait(), Ok(Some(_)) | Err(_));
            if !exited {
                // Bind the probe to the exact lifecycle revision observed
                // before dispatch. A response from an old process, or one that
                // overlaps a request transport failure, must not restore
                // Routable on a newer/suspect revision.
                let (generation, readiness_epoch, kv_readiness_epoch) =
                    process.readiness_snapshot();
                if process.probe_health().await.is_ok()
                    && matches!(deployment.participant_is_active(&participant_id), Ok(true))
                {
                    consecutive_health_failures = 0;
                    process.mark_routable(generation, readiness_epoch, kv_readiness_epoch);
                    continue;
                }
                process.mark_suspect();
                consecutive_health_failures = consecutive_health_failures.saturating_add(1);
                if consecutive_health_failures < MAX_CONSECUTIVE_HEALTH_FAILURES {
                    continue;
                }
                log::error!(
                    "[vllm-supervisor] endpoint {} failed {} consecutive HTTP/KV readiness checks",
                    process.spec.endpoint,
                    consecutive_health_failures
                );
            }

            // A restart is a lifecycle boundary: reap the complete old process
            // group, then retire its connector registration and isolated IPC
            // pool before a new vLLM engine UUID can register. If either fence
            // fails, retaining authority is safer than starting a second
            // backend against memory that may still be imported.
            // Process-group fencing may spend several seconds waiting through
            // TERM/KILL grace periods. Keep that synchronous OS lifecycle work
            // off Tokio's async workers while preserving the exact boolean
            // fence before participant retirement and restart.
            let stopped = match tokio::task::spawn_blocking({
                let process = process.clone();
                move || process.stop_child()
            })
            .await
            {
                Ok(stopped) => stopped,
                Err(error) => {
                    log::error!(
                        "[vllm-supervisor] model {} process-group fence task failed; restart is disabled and its shared pool remains charged: {}",
                        process.spec.served_model_name,
                        error,
                    );
                    break;
                }
            };
            if !stopped {
                log::error!(
                    "[vllm-supervisor] model {} process group could not be reaped; restart is disabled and its shared pool remains charged",
                    process.spec.served_model_name,
                );
                break;
            }
            if let Err(error) = deployment.retire_participants_after_backend_exit(&participant_id) {
                log::error!(
                    "[vllm-supervisor] model {} participant retirement failed; restart is disabled: {}",
                    process.spec.served_model_name,
                    error,
                );
                break;
            }

            if process.restarts.load(Ordering::Acquire) >= MAX_RESTARTS {
                log::error!(
                    "[vllm-supervisor] model {} exceeded {} restarts; giving up (log: {})",
                    process.spec.served_model_name,
                    MAX_RESTARTS,
                    process.spec.log_path.display()
                );
                break;
            }
            let attempt = process.restarts.fetch_add(1, Ordering::AcqRel) + 1;
            log::warn!(
                "[vllm-supervisor] restarting model {} (attempt {}/{})",
                process.spec.served_model_name,
                attempt,
                MAX_RESTARTS
            );
            tokio::time::sleep(SUPERVISOR_BACKOFF).await;
            if process.shutdown.load(Ordering::Acquire) {
                break;
            }
            match process.spawn_child() {
                Ok(()) => match process.wait_ready(&deployment, &participant_id).await {
                    Ok(()) => {
                        consecutive_health_failures = 0;
                        log::info!(
                            "[vllm-supervisor] model {} restarted successfully",
                            process.spec.served_model_name
                        );
                    }
                    Err(error) => log::error!(
                        "[vllm-supervisor] model {} did not become ready: {}",
                        process.spec.served_model_name,
                        error
                    ),
                },
                Err(error) => log::error!(
                    "[vllm-supervisor] failed to restart model {}: {}",
                    process.spec.served_model_name,
                    error
                ),
            }
        }
    });
}

struct ManagedVllmRuntime {
    process: Arc<ManagedVllmProcess>,
    participant_id: String,
    participant_retired: parking_lot::Mutex<bool>,
}

impl ManagedVllmRuntime {
    fn shutdown(&self, deployment: &ManagedVllmDeployment) -> bool {
        if !self.process.terminate() {
            return false;
        }

        let mut retired = self.participant_retired.lock();
        if *retired {
            return true;
        }
        match deployment.retire_participants_after_backend_exit(&self.participant_id) {
            Ok(retired_count) => {
                *retired = true;
                if retired_count > 0 {
                    log::info!(
                        "Retired {} managed vLLM KV participant(s) under {} after backend exit",
                        retired_count,
                        self.participant_id,
                    );
                }
                true
            }
            Err(error) => {
                log::error!(
                    "failed to retire managed vLLM KV participant {}: {}",
                    self.participant_id,
                    error,
                );
                false
            }
        }
    }
}

pub(crate) struct ManagedVllmEngine {
    deployment: Arc<ManagedVllmDeployment>,
    process: Arc<ManagedVllmProcess>,
    runtime: Arc<ManagedVllmRuntime>,
    served_model_name: String,
    memory_report: MemoryReport,
    loaded: AtomicBool,
    requests: Arc<AtomicU64>,
    errors: Arc<AtomicU64>,
    generated_tokens: Arc<AtomicU64>,
}

impl ManagedVllmEngine {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create(
        deployment: Arc<ManagedVllmDeployment>,
        manifest: &Manifest,
        model_path: &Path,
        model_id: u32,
        replica_id: u32,
        device_ids: Vec<usize>,
        tensor_parallel_size: usize,
    ) -> Result<Box<dyn Engine>, String> {
        if device_ids.is_empty() {
            return Err("managed vLLM requires at least one CUDA device".to_string());
        }
        if tensor_parallel_size == 0 || tensor_parallel_size != device_ids.len() {
            return Err(format!(
                "managed vLLM tensor-parallel size {} does not match selected CUDA device count {}",
                tensor_parallel_size,
                device_ids.len()
            ));
        }
        let model_root = if model_path.is_dir() {
            model_path.to_path_buf()
        } else {
            model_path.parent().unwrap_or(model_path).to_path_buf()
        };
        if !model_root.join("config.json").is_file() {
            return Err(format!(
                "managed vLLM requires a Hugging Face model directory containing config.json; package cache {} does not contain it",
                model_root.display()
            ));
        }

        let settings = ManagedVllmSettings::from_manifest(manifest)?;
        settings.validate_launch_policy()?;
        if settings.legacy_top_level_fraction_authored {
            log::warn!(
                "metadata.serving.vllm.gpu_memory_utilization is deprecated; use kv_cache.mode: legacy_fraction during this compatibility release. An exact-byte recommendation is unavailable until the certified runtime geometry provider is installed"
            );
        }
        let port = reserve_loopback_port()?;
        let endpoint = format!("http://127.0.0.1:{port}");
        let fingerprint = model_fingerprint(manifest, &model_root)?;
        let participant_id = format!(
            "kapsl-{}-{}-{}",
            sanitize_identifier(&manifest.project_name),
            model_id,
            replica_id
        );
        let kv_transfer_config = build_kv_transfer_config(
            &deployment.control_endpoint,
            &participant_id,
            &fingerprint,
            &device_ids,
            deployment.lease_ttl_ms,
        )?;
        let model_root_log = deployment
            .runtime_root
            .join(format!("model-{model_id}-replica-{replica_id}"));
        std::fs::create_dir_all(&model_root_log).map_err(|error| {
            format!(
                "create managed vLLM model directory {}: {error}",
                model_root_log.display()
            )
        })?;
        let log_path = model_root_log.join("vllm.log");
        let process = Arc::new(ManagedVllmProcess::new(ManagedVllmProcessSpec {
            python: deployment.python.clone(),
            model_root: model_root.clone(),
            served_model_name: manifest.project_name.clone(),
            endpoint,
            port,
            kv_transfer_config,
            log_path,
            settings,
            tensor_parallel_size,
            cuda_visible_devices: child_cuda_visibility(&device_ids)?,
        }));
        let memory_report =
            managed_vllm_memory_report(&model_root, &device_ids, model_id, replica_id)?;
        deployment.register_participant_readiness_fence(&participant_id, &process)?;

        let runtime = Arc::new(ManagedVllmRuntime {
            process: process.clone(),
            participant_id,
            participant_retired: parking_lot::Mutex::new(false),
        });
        deployment.register_runtime(model_id, &runtime);

        Ok(Box::new(Self {
            deployment,
            process,
            runtime,
            served_model_name: manifest.project_name.clone(),
            memory_report,
            loaded: AtomicBool::new(false),
            requests: Arc::new(AtomicU64::new(0)),
            errors: Arc::new(AtomicU64::new(0)),
            generated_tokens: Arc::new(AtomicU64::new(0)),
        }))
    }

    fn shutdown_managed_backend(&self) {
        self.loaded.store(false, Ordering::Release);
        self.runtime.shutdown(&self.deployment);
    }

    fn request_payload(
        &self,
        request: &InferenceRequest,
        stream: bool,
    ) -> Result<(String, String), EngineError> {
        if request.input.dtype != TensorDtype::Utf8 {
            return Err(EngineError::invalid_input(format!(
                "managed vLLM expects a UTF-8 prompt tensor, got {}",
                request.input.dtype
            )));
        }
        let prompt = std::str::from_utf8(&request.input.data)
            .map_err(|error| EngineError::invalid_input(format!("prompt is not UTF-8: {error}")))?;
        let envelope = serde_json::from_str::<serde_json::Value>(prompt).ok();
        let messages = envelope.as_ref().and_then(|value| {
            value
                .get(MANAGED_VLLM_CHAT_MARKER)
                .and_then(serde_json::Value::as_bool)
                .filter(|enabled| *enabled)
                .and_then(|_| value.get("messages"))
                .cloned()
        });
        let mut payload = if let Some(messages) = messages {
            serde_json::json!({
                "model": self.served_model_name,
                "messages": messages,
                "stream": stream,
            })
        } else {
            serde_json::json!({
                "model": self.served_model_name,
                "prompt": prompt,
                "stream": stream,
            })
        };
        let endpoint_path = if payload.get("messages").is_some() {
            if let Some(stop) = envelope
                .as_ref()
                .and_then(|value| value.get("stop"))
                .filter(|stop| !stop.as_array().is_some_and(Vec::is_empty))
            {
                payload["stop"] = stop.clone();
            }
            "/v1/chat/completions"
        } else {
            "/v1/completions"
        };
        let metadata = request.metadata.as_ref();
        payload["max_tokens"] = serde_json::json!(metadata
            .and_then(|value| value.max_new_tokens)
            .unwrap_or(256));
        if let Some(value) = metadata.and_then(|value| value.temperature) {
            payload["temperature"] = serde_json::json!(value);
        }
        if let Some(value) = metadata.and_then(|value| value.top_p) {
            payload["top_p"] = serde_json::json!(value);
        }
        if let Some(value) = metadata.and_then(|value| value.top_k) {
            payload["top_k"] = serde_json::json!(value);
        }
        if let Some(value) = metadata.and_then(|value| value.repetition_penalty) {
            payload["repetition_penalty"] = serde_json::json!(value);
        }
        if let Some(value) = metadata.and_then(|value| value.seed) {
            payload["seed"] = serde_json::json!(value);
        }
        if let Some(value) = metadata.and_then(|value| value.stop_token_ids.as_ref()) {
            payload["stop_token_ids"] = serde_json::json!(value);
        }
        serde_json::to_string(&payload)
            .map(|body| (endpoint_path.to_string(), body))
            .map_err(|error| EngineError::backend(format!("serialize vLLM request: {error}")))
    }

    fn output_packet(text: String) -> Result<BinaryTensorPacket, EngineError> {
        if text.is_empty() {
            return Err(EngineError::backend(
                "managed vLLM returned an empty completion".to_string(),
            ));
        }
        let data = text.into_bytes();
        BinaryTensorPacket::new(vec![1, data.len() as i64], TensorDtype::Utf8, data)
    }

    fn parse_completion(value: &serde_json::Value, chat: bool) -> Result<String, EngineError> {
        let choice = value
            .get("choices")
            .and_then(serde_json::Value::as_array)
            .and_then(|choices| choices.first())
            .ok_or_else(|| EngineError::backend("vLLM response has no choices".to_string()))?;
        let text = if chat {
            choice
                .get("message")
                .and_then(|message| message.get("content"))
                .and_then(serde_json::Value::as_str)
        } else {
            choice.get("text").and_then(serde_json::Value::as_str)
        };
        text.map(str::to_string).ok_or_else(|| {
            EngineError::backend("vLLM response has no textual completion".to_string())
        })
    }

    fn infer_inner(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        self.process.ensure_routable()?;
        if request
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(EngineError::cancelled(
                "managed vLLM request was cancelled before dispatch".to_string(),
            ));
        }
        let (path, payload) = self.request_payload(request, false)?;
        let chat = path.contains("chat");
        let mut response = match self.process.bridge.post_json_sync(
            &path,
            payload.as_bytes(),
            managed_vllm_request_timeouts(request),
        ) {
            Ok(response) => response,
            Err(error) => {
                if bridge_error_is_transport_failure(&error) {
                    self.process.mark_suspect();
                }
                return Err(bridge_error_to_engine(
                    error,
                    "dispatch managed vLLM request",
                ));
            }
        };
        let status = response.status();
        let body = match response.body_mut().read_to_string() {
            Ok(body) => body,
            Err(error) => {
                let error = map_ureq_body_error(error);
                if bridge_error_is_transport_failure(&error) {
                    self.process.mark_suspect();
                }
                return Err(bridge_error_to_engine(error, "read managed vLLM response"));
            }
        };
        if !status.is_success() {
            return Err(EngineError::backend(format!(
                "managed vLLM returned upstream HTTP {status}: {body}"
            )));
        }
        let value: serde_json::Value = serde_json::from_str(&body)
            .map_err(|error| EngineError::backend(format!("decode vLLM response: {error}")))?;
        let text = Self::parse_completion(&value, chat)?;
        self.generated_tokens
            .fetch_add(estimate_vllm_tokens(&text), Ordering::Relaxed);
        Self::output_packet(text)
    }
}

#[async_trait::async_trait]
impl Engine for ManagedVllmEngine {
    fn planned_memory(&self, _path: &Path) -> Result<MemoryReport, EngineError> {
        Ok(self.memory_report.clone())
    }

    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        self.process.spawn_child()?;
        if let Err(error) = self
            .process
            .wait_ready(&self.deployment, &self.runtime.participant_id)
            .await
        {
            self.shutdown_managed_backend();
            return Err(error);
        }
        self.loaded.store(true, Ordering::Release);
        start_managed_vllm_supervisor(
            self.process.clone(),
            self.deployment.clone(),
            self.runtime.participant_id.clone(),
        );
        log::info!(
            "✓ Managed vLLM ready for model {} at {}",
            self.served_model_name,
            self.process.spec.endpoint
        );
        Ok(())
    }

    fn actual_memory(&self) -> MemoryReport {
        if self.loaded.load(Ordering::Acquire) {
            self.memory_report.clone()
        } else {
            MemoryReport::default()
        }
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let result = self.infer_inner(request);
        if result.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn supports_openai_wire(&self) -> bool {
        true
    }

    async fn infer_openai_wire(
        &self,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireResponse, EngineError> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let result = async {
            if request.format != OpenAiWireFormat::Json {
                return Err(EngineError::invalid_input(
                    "managed vLLM unary wire inference requires JSON format",
                ));
            }
            request.validate(usize::MAX)?;
            self.process.ensure_routable()?;
            let response = self
                .process
                .bridge
                .post_json_buffered(
                    request.endpoint.path(),
                    request.body.clone(),
                    managed_vllm_wire_request_timeouts(request),
                    request.cancellation.clone(),
                    MAX_OPENAI_WIRE_RESPONSE_BYTES,
                )
                .await
                .map_err(|error| {
                    if bridge_error_is_transport_failure(&error) {
                        self.process.mark_suspect();
                    }
                    bridge_error_to_engine(error, "dispatch managed vLLM wire request")
                })?;
            if !response.status.is_success() {
                self.errors.fetch_add(1, Ordering::Relaxed);
            }
            Ok(OpenAiWireResponse {
                head: managed_vllm_wire_response_head(response.status, &response.headers)?,
                body: response.body,
            })
        }
        .await;
        if result.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    async fn infer_openai_wire_stream(
        &self,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let result = async {
            if request.format != OpenAiWireFormat::ServerSentEvents {
                return Err(EngineError::invalid_input(
                    "managed vLLM streaming wire inference requires SSE format",
                ));
            }
            request.validate(usize::MAX)?;
            self.process.ensure_routable()?;
            let response = self
                .process
                .bridge
                .post_json_raw(
                    request.endpoint.path(),
                    request.body.clone(),
                    managed_vllm_wire_request_timeouts(request),
                    request.cancellation.clone(),
                )
                .await
                .map_err(|error| {
                    if bridge_error_is_transport_failure(&error) {
                        self.process.mark_suspect();
                    }
                    bridge_error_to_engine(error, "dispatch managed vLLM wire stream")
                })?;
            if !response.status.is_success() {
                self.errors.fetch_add(1, Ordering::Relaxed);
            }
            Ok(OpenAiWireStreamResponse {
                head: managed_vllm_wire_response_head(response.status, &response.headers)?,
                body: relay_managed_vllm_wire_stream(
                    response.body,
                    self.process.clone(),
                    self.errors.clone(),
                ),
            })
        }
        .await;
        if result.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn self_batches(&self) -> bool {
        true
    }

    fn batching_policy(&self) -> BatchingPolicy {
        BatchingPolicy::delegated().with_priority_support()
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let (path, payload) = match self.request_payload(request, true) {
            Ok(request) => request,
            Err(error) => {
                self.errors.fetch_add(1, Ordering::Relaxed);
                return Box::pin(stream::once(async move { Err(error) }));
            }
        };
        let chat = path.contains("chat");
        let process = self.process.clone();
        let cancellation = request.cancellation.clone();
        let timeouts = managed_vllm_request_timeouts(request);
        let errors = self.errors.clone();
        let generated_tokens = self.generated_tokens.clone();
        let start = async move {
            if let Err(error) = process.ensure_routable() {
                errors.fetch_add(1, Ordering::Relaxed);
                return Box::pin(stream::once(async move { Err(error) })) as EngineStream;
            }
            let response = match process
                .bridge
                .post_json_sse(&path, payload.into_bytes(), timeouts, cancellation)
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    if bridge_error_is_transport_failure(&error) {
                        process.mark_suspect();
                    }
                    errors.fetch_add(1, Ordering::Relaxed);
                    let error = bridge_error_to_engine(error, "dispatch managed vLLM stream");
                    return Box::pin(stream::once(async move { Err(error) })) as EngineStream;
                }
            };
            log::trace!(
                "managed vLLM stream opened with status={} content_type={:?}",
                response.status,
                response.headers.get("content-type"),
            );
            translate_managed_vllm_stream(response.events, process, errors, generated_tokens, chat)
        };
        Box::pin(stream::once(start).flatten())
    }

    fn unload(&mut self) {
        self.shutdown_managed_backend();
    }

    fn metrics(&self) -> EngineMetrics {
        let requests = self.requests.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);
        EngineMetrics {
            memory_usage: self
                .actual_memory()
                .allocations
                .iter()
                .map(|allocation| allocation.bytes)
                .sum(),
            error_rate: if requests == 0 {
                0.0
            } else {
                errors as f64 / requests as f64
            },
            generated_tokens_total: self.generated_tokens.load(Ordering::Relaxed),
            ..Default::default()
        }
    }

    fn health_check(&self) -> Result<(), EngineError> {
        self.process.ensure_routable()
    }
}

impl Drop for ManagedVllmEngine {
    fn drop(&mut self) {
        self.shutdown_managed_backend();
    }
}

fn reserve_loopback_port() -> Result<u16, String> {
    let listener = StdTcpListener::bind(("127.0.0.1", 0))
        .map_err(|error| format!("reserve managed vLLM loopback port: {error}"))?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| format!("resolve managed vLLM loopback port: {error}"))
}

fn sanitize_identifier(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    let sanitized = sanitized.trim_matches('-');
    if sanitized.is_empty() {
        "model".to_string()
    } else {
        sanitized.to_string()
    }
}

fn child_cuda_visibility(device_ids: &[usize]) -> Result<String, String> {
    let inherited = std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty());
    match inherited {
        Some(physical_devices) => device_ids
            .iter()
            .map(|device_id| {
                physical_devices.get(*device_id).cloned().ok_or_else(|| {
                    format!(
                        "selected CUDA device {} is outside inherited CUDA_VISIBLE_DEVICES={}",
                        device_id,
                        physical_devices.join(",")
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|values| values.join(",")),
        None => Ok(device_ids
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")),
    }
}

fn model_fingerprint(manifest: &Manifest, model_root: &Path) -> Result<String, String> {
    let mut hasher = Sha256::new();
    let manifest = serde_json::to_vec(manifest)
        .map_err(|error| format!("serialize model manifest for fingerprint: {error}"))?;
    hasher.update(&manifest);
    let entries = std::fs::read_dir(model_root)
        .map_err(|error| format!("read model directory {}: {error}", model_root.display()))?;
    let mut assets = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "read an entry in model directory {}: {error}",
                model_root.display()
            )
        })?;
        let metadata = entry
            .metadata()
            .map_err(|error| format!("stat model asset {}: {error}", entry.path().display()))?;
        if metadata.is_file() {
            assets.push((
                entry.file_name().to_string_lossy().into_owned(),
                metadata.len(),
            ));
        }
    }
    assets.sort();
    for (name, bytes) in assets {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update(bytes.to_le_bytes());
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn build_kv_transfer_config(
    control_endpoint: &str,
    participant_id: &str,
    model_fingerprint: &str,
    device_ids: &[usize],
    lease_ttl_ms: u64,
) -> Result<String, String> {
    let memory_domains = device_ids
        .iter()
        .map(|device_id| serde_json::json!({"kind": "cuda", "device_id": device_id}))
        .collect::<Vec<_>>();
    let rank_device_map = device_ids
        .iter()
        .enumerate()
        .map(|(rank, device_id)| (rank.to_string(), serde_json::json!(device_id)))
        .collect::<serde_json::Map<_, _>>();
    serde_json::to_string(&serde_json::json!({
        "kv_connector": "KapslConnectorV1",
        "kv_role": "kv_both",
        "kv_connector_module_path": "kapsl_vllm_connector",
        "kv_connector_extra_config": {
            "kapsl_control_endpoint": control_endpoint,
            "kapsl_participant_id": participant_id,
            "kapsl_model_fingerprint": model_fingerprint,
            "kapsl_kv_mode": "shared_pool",
            "kapsl_memory_domains": memory_domains,
            "kapsl_rank_device_map": rank_device_map,
            "kapsl_lease_ttl_ms": lease_ttl_ms,
        }
    }))
    .map_err(|error| format!("serialize managed vLLM KV transfer config: {error}"))
}

pub(crate) fn managed_vllm_memory_report(
    model_path: &Path,
    device_ids: &[usize],
    model_id: u32,
    replica_id: u32,
) -> Result<MemoryReport, String> {
    if device_ids.is_empty() {
        return Err("managed vLLM memory reporting requires at least one CUDA device".to_string());
    }
    // PackageLoader exposes the manifest's primary model file, while vLLM
    // consumes the complete Hugging Face directory beside that file. Accept
    // either representation so preliminary admission and the real backend
    // load estimate the same weights.
    let model_root = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent().unwrap_or(model_path)
    };
    let entries = std::fs::read_dir(model_root)
        .map_err(|error| format!("read model directory {}: {error}", model_root.display()))?;
    let mut weight_bytes = 0usize;
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "read an entry in model directory {}: {error}",
                model_root.display()
            )
        })?;
        if entry.path().extension().and_then(|value| value.to_str()) != Some("safetensors") {
            continue;
        }
        let metadata = entry
            .metadata()
            .map_err(|error| format!("stat model shard {}: {error}", entry.path().display()))?;
        let bytes = usize::try_from(metadata.len()).map_err(|_| {
            format!(
                "model shard {} is too large for this host",
                entry.path().display()
            )
        })?;
        weight_bytes = weight_bytes.checked_add(bytes).ok_or_else(|| {
            format!(
                "SafeTensors weight bytes overflowed for model directory {}",
                model_root.display()
            )
        })?;
    }
    if weight_bytes == 0 {
        return Err(format!(
            "managed vLLM model directory {} contains no .safetensors weights",
            model_root.display()
        ));
    }
    let per_device_weights = weight_bytes
        .checked_add(device_ids.len() - 1)
        .ok_or_else(|| "managed vLLM per-device weight rounding overflowed".to_string())?
        / device_ids.len();
    let workspace_bytes = (per_device_weights / 8).max(256 * 1024 * 1024);
    let mut allocations = Vec::with_capacity(device_ids.len() * 2);
    for &device_id in device_ids {
        allocations.push(MemoryAllocation {
            allocation_id: format!("managed-vllm:{model_id}:{replica_id}:weights:{device_id}"),
            domain: MemoryDomain::Cuda { device_id },
            class: MemoryAllocationClass::PersistentWeights,
            source: MemoryAllocationSource::BackendManaged,
            bytes: per_device_weights,
        });
        allocations.push(MemoryAllocation {
            allocation_id: format!("managed-vllm:{model_id}:{replica_id}:workspace:{device_id}"),
            domain: MemoryDomain::Cuda { device_id },
            class: MemoryAllocationClass::TransientWorkspace,
            source: MemoryAllocationSource::BackendManaged,
            bytes: workspace_bytes,
        });
    }
    Ok(MemoryReport { allocations })
}

fn estimate_vllm_tokens(text: &str) -> u64 {
    if text.is_empty() {
        0
    } else {
        ((text.chars().count() as f64) / 4.0).ceil() as u64
    }
}

pub(crate) fn managed_vllm_chat_input(
    messages: &[crate::http::openai::types::ChatMessage],
    stops: &[String],
) -> Result<Vec<u8>, String> {
    if messages.is_empty() {
        return Err("messages must contain at least one item".to_string());
    }
    let messages = messages
        .iter()
        .map(|message| {
            serde_json::json!({
                "role": message.role,
                "content": message.content,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_vec(&serde_json::json!({
        "__kapsl_managed_vllm_chat_v1": true,
        "messages": messages,
        "stop": stops,
    }))
    .map_err(|error| format!("serialize managed vLLM chat request: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn settings_from_metadata(metadata: &str) -> Result<ManagedVllmSettings, String> {
        let metadata = serde_yaml::from_str(metadata).expect("test metadata YAML");
        ManagedVllmSettings::from_manifest(&Manifest {
            project_name: "test".to_string(),
            framework: "safetensors".to_string(),
            version: "1".to_string(),
            created_at: "now".to_string(),
            model_file: "model.safetensors".to_string(),
            format: Some("safetensors".to_string()),
            model_type: Some("causal-lm".to_string()),
            task: Some("generate".to_string()),
            metadata: Some(metadata),
            hardware_requirements: Default::default(),
            cron_jobs: Vec::new(),
        })
    }

    fn simple_kv_geometry() -> ManagedVllmKvCacheGeometry {
        ManagedVllmKvCacheGeometry {
            groups: vec![ManagedVllmKvCacheGroupGeometry {
                group_id: "group.0".to_string(),
                block_size_tokens: 16,
                bytes_per_block: 100,
                block_alignment: 1,
                prefix_retention_blocks: 0,
            }],
            allocation_alignment_bytes: 1,
            fixed_overhead_bytes: 0,
        }
    }

    fn test_managed_process() -> Arc<ManagedVllmProcess> {
        Arc::new(ManagedVllmProcess::new(ManagedVllmProcessSpec {
            python: PathBuf::from("python"),
            model_root: PathBuf::from("model"),
            served_model_name: "test-model".to_string(),
            endpoint: "http://127.0.0.1:12345".to_string(),
            port: 12345,
            kv_transfer_config: "{}".to_string(),
            log_path: PathBuf::from("vllm-test.log"),
            settings: ManagedVllmSettings {
                gpu_memory_utilization: 0.25,
                kv_cache_policy: ManagedVllmKvCachePolicy::LegacyFraction {
                    gpu_memory_utilization: 0.25,
                },
                legacy_top_level_fraction_authored: false,
                max_model_len: 512,
                startup_timeout: Duration::from_secs(30),
            },
            tensor_parallel_size: 1,
            cuda_visible_devices: "0".to_string(),
        }))
    }

    #[test]
    fn certified_profile_matches_certified_tuple() {
        assert_eq!(
            certified_vllm_profile(),
            "kapsl-vllm-connector,0.6.0,0.26.1rc1.dev1130+g2ec6f0d71,vllm-v1-packed-cuda-ipc/flash-attn"
        );
    }

    #[test]
    fn kv_policy_defaults_to_the_legacy_runtime_fraction_until_exact_wiring_exists() {
        let settings = settings_from_metadata("{}\n").unwrap();

        assert_eq!(settings.gpu_memory_utilization, 0.5);
        assert_eq!(
            settings.kv_cache_policy,
            ManagedVllmKvCachePolicy::LegacyFraction {
                gpu_memory_utilization: 0.5,
            }
        );
    }

    #[test]
    fn kv_policy_parses_auto_fixed_and_both_legacy_forms() {
        let auto = settings_from_metadata(
            r#"
serving:
  vllm:
    kv_cache:
      mode: auto
      target_concurrency: 16
      headroom_percent: 25
      min_bytes: 268435456
      max_bytes: 2147483648
      strict: true
"#,
        )
        .unwrap();
        assert_eq!(
            auto.kv_cache_policy,
            ManagedVllmKvCachePolicy::Auto {
                target_concurrency: Some(16),
                headroom_percent: 25,
                min_bytes: Some(268_435_456),
                max_bytes: Some(2_147_483_648),
                strict: true,
            }
        );
        assert_eq!(
            auto.gpu_memory_utilization, DEFAULT_LEGACY_GPU_MEMORY_UTILIZATION,
            "the command compatibility fallback must not change yet"
        );
        assert!(auto.validate_launch_policy().is_err());

        let fixed = settings_from_metadata(
            r#"
serving:
  vllm:
    kv_cache:
      mode: fixed
      bytes: 536870912
"#,
        )
        .unwrap();
        assert_eq!(
            fixed.kv_cache_policy,
            ManagedVllmKvCachePolicy::Fixed { bytes: 536_870_912 }
        );
        assert!(fixed.validate_launch_policy().is_err());

        let explicit_legacy = settings_from_metadata(
            r#"
serving:
  vllm:
    kv_cache:
      mode: legacy_fraction
      gpu_memory_utilization: 0.4
"#,
        )
        .unwrap();
        assert_eq!(explicit_legacy.gpu_memory_utilization, 0.4);
        assert!(!explicit_legacy.legacy_top_level_fraction_authored);
        assert!(explicit_legacy.validate_launch_policy().is_ok());
        assert_eq!(
            explicit_legacy.kv_cache_policy,
            ManagedVllmKvCachePolicy::LegacyFraction {
                gpu_memory_utilization: 0.4,
            }
        );

        let migrated_legacy = settings_from_metadata(
            r#"
serving:
  vllm:
    gpu_memory_utilization: 0.3
"#,
        )
        .unwrap();
        assert_eq!(migrated_legacy.gpu_memory_utilization, 0.3);
        assert!(migrated_legacy.legacy_top_level_fraction_authored);
        assert_eq!(
            migrated_legacy.kv_cache_policy,
            ManagedVllmKvCachePolicy::LegacyFraction {
                gpu_memory_utilization: 0.3,
            }
        );
    }

    #[test]
    fn kv_policy_rejects_cross_mode_and_legacy_conflicts() {
        let conflicts = [
            r#"
serving:
  vllm:
    gpu_memory_utilization: 0.5
    kv_cache:
      mode: auto
"#,
            r#"
serving:
  vllm:
    kv_cache:
      mode: auto
      bytes: 1024
"#,
            r#"
serving:
  vllm:
    kv_cache:
      mode: fixed
      bytes: 1024
      target_concurrency: 2
"#,
            r#"
serving:
  vllm:
    kv_cache:
      mode: legacy_fraction
      min_bytes: 1024
"#,
        ];
        for metadata in conflicts {
            assert!(
                settings_from_metadata(metadata)
                    .unwrap_err()
                    .contains("conflict"),
                "metadata should be rejected as conflicting: {metadata}"
            );
        }
    }

    #[test]
    fn kv_policy_rejects_missing_invalid_and_unknown_modes() {
        let invalid = [
            r#"
serving:
  vllm:
    kv_cache: auto
"#,
            r#"
serving:
  vllm:
    kv_cache: {}
"#,
            r#"
serving:
  vllm:
    kv_cache:
      mode: elastic
"#,
            r#"
serving:
  vllm:
    kv_cache:
      mode: fixed
"#,
        ];
        for metadata in invalid {
            assert!(settings_from_metadata(metadata).is_err(), "{metadata}");
        }
    }

    #[test]
    fn kv_policy_validates_numeric_bounds_and_strict_type() {
        let invalid = [
            ("target_concurrency: 0", "target_concurrency"),
            ("headroom_percent: 101", "headroom_percent"),
            ("min_bytes: 0", "min_bytes"),
            ("max_bytes: 0", "max_bytes"),
            ("min_bytes: 20\n      max_bytes: 10", "must not exceed"),
            ("strict: 'yes'", "strict"),
        ];
        for (fields, expected) in invalid {
            let metadata = format!(
                "serving:\n  vllm:\n    kv_cache:\n      mode: auto\n      {}\n",
                fields
            );
            let error = settings_from_metadata(&metadata).unwrap_err();
            assert!(error.contains(expected), "unexpected error: {error}");
        }

        for value in ["0", "-1"] {
            let metadata = format!(
                "serving:\n  vllm:\n    kv_cache:\n      mode: fixed\n      bytes: {value}\n"
            );
            assert!(settings_from_metadata(&metadata).is_err());
        }
        for value in ["0.09", "0.91", ".nan"] {
            let metadata = format!(
                "serving:\n  vllm:\n    kv_cache:\n      mode: legacy_fraction\n      gpu_memory_utilization: {value}\n"
            );
            assert!(settings_from_metadata(&metadata).is_err());
        }
    }

    #[test]
    fn checked_kv_sizing_applies_block_headroom_and_byte_alignment() {
        let geometry = ManagedVllmKvCacheGeometry {
            groups: vec![ManagedVllmKvCacheGroupGeometry {
                group_id: "group.0".to_string(),
                block_size_tokens: 16,
                bytes_per_block: 1024,
                block_alignment: 4,
                prefix_retention_blocks: 0,
            }],
            allocation_alignment_bytes: 4096,
            fixed_overhead_bytes: 0,
        };
        let policy = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: Some(2),
            headroom_percent: 25,
            min_bytes: None,
            max_bytes: None,
            strict: false,
        };

        let sizing = size_managed_vllm_kv_cache(&policy, &geometry, 33, 99, usize::MAX).unwrap();

        assert_eq!(sizing.minimum_required_bytes, 4096);
        assert_eq!(sizing.requested_bytes, 8192);
        assert_eq!(sizing.granted_bytes, 8192);
        assert_eq!(sizing.effective_target_concurrency, 2);
        assert_eq!(sizing.groups[0].sequence_blocks, 3);
        assert_eq!(sizing.groups[0].granted_blocks, 8);
    }

    #[test]
    fn checked_kv_sizing_rejects_hybrid_groups_without_the_certified_planner() {
        let geometry = ManagedVllmKvCacheGeometry {
            groups: vec![
                ManagedVllmKvCacheGroupGeometry {
                    group_id: "full".to_string(),
                    block_size_tokens: 16,
                    bytes_per_block: 100,
                    block_alignment: 2,
                    prefix_retention_blocks: 2,
                },
                ManagedVllmKvCacheGroupGeometry {
                    group_id: "sliding".to_string(),
                    block_size_tokens: 32,
                    bytes_per_block: 200,
                    block_alignment: 1,
                    prefix_retention_blocks: 1,
                },
            ],
            allocation_alignment_bytes: 256,
            fixed_overhead_bytes: 64,
        };
        let policy = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: Some(2),
            headroom_percent: 50,
            min_bytes: None,
            max_bytes: None,
            strict: false,
        };

        let error = size_managed_vllm_kv_cache(&policy, &geometry, 33, 1, usize::MAX).unwrap_err();
        assert!(error.contains("hybrid geometry requires the certified connector planner"));
    }

    #[test]
    fn checked_kv_sizing_reduces_only_to_whole_sequence_concurrency() {
        let policy = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: Some(4),
            headroom_percent: 0,
            min_bytes: None,
            max_bytes: None,
            strict: false,
        };

        let sizing =
            size_managed_vllm_kv_cache(&policy, &simple_kv_geometry(), 32, 1, 450).unwrap();

        assert_eq!(sizing.requested_bytes, 800);
        assert_eq!(sizing.granted_bytes, 400);
        assert_eq!(sizing.target_concurrency, 4);
        assert_eq!(sizing.effective_target_concurrency, 2);
        assert!(!sizing.headroom_reduced);
    }

    #[test]
    fn strict_kv_sizing_rejects_any_target_reduction() {
        let policy = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: Some(4),
            headroom_percent: 0,
            min_bytes: None,
            max_bytes: None,
            strict: true,
        };

        let error =
            size_managed_vllm_kv_cache(&policy, &simple_kv_geometry(), 32, 1, 450).unwrap_err();
        assert!(error.contains("strict"));
    }

    #[test]
    fn strict_kv_sizing_sheds_headroom_before_reducing_concurrency() {
        let policy = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: Some(2),
            headroom_percent: 100,
            min_bytes: None,
            max_bytes: None,
            strict: true,
        };

        let sizing =
            size_managed_vllm_kv_cache(&policy, &simple_kv_geometry(), 32, 1, 450).unwrap();

        assert_eq!(sizing.requested_bytes, 800);
        assert_eq!(sizing.granted_bytes, 400);
        assert_eq!(sizing.effective_target_concurrency, 2);
        assert!(sizing.headroom_reduced);
    }

    #[test]
    fn strict_kv_sizing_sheds_optional_prefix_before_reducing_concurrency() {
        let mut geometry = simple_kv_geometry();
        geometry.groups[0].prefix_retention_blocks = 2;
        let policy = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: Some(2),
            headroom_percent: 0,
            min_bytes: None,
            max_bytes: Some(400),
            strict: true,
        };

        let sizing = size_managed_vllm_kv_cache(&policy, &geometry, 32, 1, usize::MAX).unwrap();

        assert_eq!(sizing.requested_bytes, 600);
        assert_eq!(sizing.granted_bytes, 400);
        assert_eq!(sizing.effective_target_concurrency, 2);
    }

    #[test]
    fn optional_headroom_is_shed_before_one_sequence_context_capacity() {
        let policy = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: Some(2),
            headroom_percent: 100,
            min_bytes: None,
            max_bytes: None,
            strict: false,
        };

        let sizing =
            size_managed_vllm_kv_cache(&policy, &simple_kv_geometry(), 32, 1, 250).unwrap();

        assert_eq!(sizing.minimum_required_bytes, 200);
        assert_eq!(sizing.requested_bytes, 800);
        assert_eq!(sizing.granted_bytes, 200);
        assert_eq!(sizing.effective_target_concurrency, 1);
        assert!(sizing.headroom_reduced);
    }

    #[test]
    fn min_floor_is_aligned_and_max_cap_reduces_concurrency() {
        let mut geometry = simple_kv_geometry();
        geometry.groups[0].bytes_per_block = 128;
        geometry.allocation_alignment_bytes = 64;
        let floor_policy = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: Some(1),
            headroom_percent: 0,
            min_bytes: Some(500),
            max_bytes: None,
            strict: false,
        };
        let floor =
            size_managed_vllm_kv_cache(&floor_policy, &geometry, 16, 1, usize::MAX).unwrap();
        assert_eq!(floor.minimum_required_bytes, 128);
        assert_eq!(floor.granted_bytes, 512);
        assert_eq!(floor.groups[0].granted_blocks, 4);
        assert_eq!(floor.unassigned_bytes, 0);

        let cap_policy = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: Some(4),
            headroom_percent: 0,
            min_bytes: None,
            max_bytes: Some(450),
            strict: false,
        };
        let capped =
            size_managed_vllm_kv_cache(&cap_policy, &simple_kv_geometry(), 32, 1, usize::MAX)
                .unwrap();
        assert_eq!(capped.granted_bytes, 400);
        assert_eq!(capped.effective_target_concurrency, 2);
    }

    #[test]
    fn min_max_and_available_bytes_cannot_violate_one_sequence_minimum() {
        let cases = [
            ManagedVllmKvCachePolicy::Auto {
                target_concurrency: Some(1),
                headroom_percent: 0,
                min_bytes: Some(199),
                max_bytes: None,
                strict: false,
            },
            ManagedVllmKvCachePolicy::Auto {
                target_concurrency: Some(1),
                headroom_percent: 0,
                min_bytes: None,
                max_bytes: Some(199),
                strict: false,
            },
            ManagedVllmKvCachePolicy::Auto {
                target_concurrency: Some(1),
                headroom_percent: 0,
                min_bytes: Some(300),
                max_bytes: Some(250),
                strict: false,
            },
        ];
        for policy in cases {
            assert!(
                size_managed_vllm_kv_cache(&policy, &simple_kv_geometry(), 32, 1, usize::MAX,)
                    .is_err()
            );
        }

        let normal = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: Some(1),
            headroom_percent: 0,
            min_bytes: None,
            max_bytes: None,
            strict: false,
        };
        assert!(size_managed_vllm_kv_cache(&normal, &simple_kv_geometry(), 32, 1, 199,).is_err());
    }

    #[test]
    fn fixed_kv_sizing_requires_exact_alignment_minimum_and_authority_fit() {
        let mut geometry = simple_kv_geometry();
        geometry.groups[0].bytes_per_block = 128;
        geometry.allocation_alignment_bytes = 64;

        let valid = size_managed_vllm_kv_cache(
            &ManagedVllmKvCachePolicy::Fixed { bytes: 512 },
            &geometry,
            16,
            1,
            1024,
        )
        .unwrap();
        assert_eq!(valid.minimum_required_bytes, 128);
        assert_eq!(valid.granted_bytes, 512);

        for (bytes, available) in [(500, 1024), (64, 1024), (512, 511)] {
            assert!(size_managed_vllm_kv_cache(
                &ManagedVllmKvCachePolicy::Fixed { bytes },
                &geometry,
                16,
                1,
                available,
            )
            .is_err());
        }

        let mut incompatible_stride = simple_kv_geometry();
        incompatible_stride.allocation_alignment_bytes = 64;
        assert!(size_managed_vllm_kv_cache(
            &ManagedVllmKvCachePolicy::Fixed { bytes: 512 },
            &incompatible_stride,
            16,
            1,
            1024,
        )
        .unwrap_err()
        .contains("packed block stride"));
    }

    #[test]
    fn checked_kv_sizing_rejects_invalid_geometry_policy_and_overflow() {
        let auto = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: Some(1),
            headroom_percent: 0,
            min_bytes: None,
            max_bytes: None,
            strict: false,
        };
        let mut geometry = simple_kv_geometry();
        geometry.groups.push(geometry.groups[0].clone());
        assert!(size_managed_vllm_kv_cache(&auto, &geometry, 16, 1, usize::MAX).is_err());

        let mut geometry = simple_kv_geometry();
        geometry.groups[0].block_alignment = 0;
        assert!(size_managed_vllm_kv_cache(&auto, &geometry, 16, 1, usize::MAX).is_err());

        let overflow = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: Some(2),
            headroom_percent: 0,
            min_bytes: None,
            max_bytes: None,
            strict: false,
        };
        assert!(size_managed_vllm_kv_cache(
            &overflow,
            &ManagedVllmKvCacheGeometry {
                groups: vec![ManagedVllmKvCacheGroupGeometry {
                    group_id: "overflow".to_string(),
                    block_size_tokens: 1,
                    bytes_per_block: 1,
                    block_alignment: 1,
                    prefix_retention_blocks: 0,
                }],
                allocation_alignment_bytes: 1,
                fixed_overhead_bytes: 0,
            },
            usize::MAX,
            1,
            usize::MAX,
        )
        .unwrap_err()
        .contains("overflow"));
        assert!(checked_round_up(usize::MAX, 2, "test").is_err());

        let invalid_policy = ManagedVllmKvCachePolicy::Auto {
            target_concurrency: None,
            headroom_percent: 101,
            min_bytes: None,
            max_bytes: None,
            strict: false,
        };
        assert!(size_managed_vllm_kv_cache(
            &invalid_policy,
            &simple_kv_geometry(),
            16,
            0,
            usize::MAX,
        )
        .is_err());
        assert!(size_managed_vllm_kv_cache(
            &ManagedVllmKvCachePolicy::LegacyFraction {
                gpu_memory_utilization: 0.5,
            },
            &simple_kv_geometry(),
            16,
            1,
            usize::MAX,
        )
        .is_err());
    }

    #[test]
    fn generated_kv_config_uses_tagged_cuda_domains_and_rank_map() {
        let encoded = build_kv_transfer_config(
            "unix:///tmp/kapsl.sock",
            "worker",
            "sha256:model",
            &[0, 2],
            30_000,
        )
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        let extra = &value["kv_connector_extra_config"];
        assert_eq!(extra["kapsl_memory_domains"][0]["kind"], "cuda");
        assert_eq!(extra["kapsl_memory_domains"][1]["device_id"], 2);
        assert_eq!(extra["kapsl_rank_device_map"]["0"], 0);
        assert_eq!(extra["kapsl_rank_device_map"]["1"], 2);
        assert_eq!(extra["kapsl_kv_mode"], "shared_pool");
    }

    #[test]
    fn identifiers_are_safe_and_never_empty() {
        assert_eq!(sanitize_identifier("Qwen/Qwen 3"), "Qwen-Qwen-3");
        assert_eq!(sanitize_identifier("///"), "model");
    }

    #[test]
    fn cuda_visibility_without_parent_mapping_uses_logical_ids() {
        if std::env::var_os("CUDA_VISIBLE_DEVICES").is_none() {
            assert_eq!(child_cuda_visibility(&[0, 2]).unwrap(), "0,2");
        }
    }

    #[test]
    fn memory_report_accepts_the_package_primary_model_file() {
        let directory = tempfile::tempdir().unwrap();
        let model_file = directory.path().join("model.safetensors");
        std::fs::write(&model_file, vec![0_u8; 123]).unwrap();

        let from_directory = managed_vllm_memory_report(directory.path(), &[0], 7, 0).unwrap();
        let from_model_file = managed_vllm_memory_report(&model_file, &[0], 7, 0).unwrap();

        assert_eq!(from_model_file.allocations, from_directory.allocations);
        assert_eq!(from_model_file.allocations[0].bytes, 123);
        assert!(managed_vllm_memory_report(&model_file, &[], 7, 0).is_err());
    }

    #[test]
    fn managed_process_passes_the_model_as_a_named_vllm_argument() {
        let directory = tempfile::tempdir().unwrap();
        let model_root = directory.path().join("model");
        std::fs::create_dir(&model_root).unwrap();
        let process = ManagedVllmProcess::new(ManagedVllmProcessSpec {
            python: PathBuf::from("python"),
            model_root: model_root.clone(),
            served_model_name: "test-model".to_string(),
            endpoint: "http://127.0.0.1:12345".to_string(),
            port: 12345,
            kv_transfer_config: "{}".to_string(),
            log_path: directory.path().join("vllm.log"),
            settings: ManagedVllmSettings {
                gpu_memory_utilization: 0.25,
                kv_cache_policy: ManagedVllmKvCachePolicy::LegacyFraction {
                    gpu_memory_utilization: 0.25,
                },
                legacy_top_level_fraction_authored: false,
                max_model_len: 512,
                startup_timeout: Duration::from_secs(30),
            },
            tensor_parallel_size: 1,
            cuda_visible_devices: "0".to_string(),
        });

        let command = process.build_command().unwrap();
        let arguments = command
            .get_args()
            .map(std::ffi::OsStr::to_os_string)
            .collect::<Vec<_>>();
        assert!(arguments
            .windows(2)
            .any(|pair| pair[0] == "--model" && pair[1] == model_root.as_os_str()));
    }

    #[test]
    fn zero_request_timeout_uses_the_default_instead_of_disabling_dispatch() {
        let input = BinaryTensorPacket::new(vec![1], TensorDtype::Uint8, vec![1]).unwrap();
        let request =
            InferenceRequest::new(input).with_metadata(kapsl_engine_api::RequestMetadata {
                timeout_ms: Some(0),
                ..Default::default()
            });

        let timeouts = managed_vllm_request_timeouts(&request);

        assert_eq!(timeouts.total, DEFAULT_REQUEST_TIMEOUT);
        assert_eq!(timeouts.headers, BRIDGE_HEADER_TIMEOUT);
        assert_eq!(timeouts.idle_body, BRIDGE_IDLE_BODY_TIMEOUT);
    }

    #[test]
    fn wire_timeout_uses_internal_metadata_and_zero_retains_default() {
        let mut request = OpenAiWireRequest::new(
            kapsl_engine_api::OpenAiWireEndpoint::ChatCompletions,
            OpenAiWireFormat::Json,
            b"{}".to_vec(),
        );
        request.metadata = Some(kapsl_engine_api::OpenAiWireMetadata {
            timeout_ms: Some(1250),
            ..Default::default()
        });
        let timeouts = managed_vllm_wire_request_timeouts(&request);
        assert_eq!(timeouts.total, Duration::from_millis(1250));
        assert_eq!(timeouts.headers, Duration::from_millis(1250));
        assert_eq!(timeouts.idle_body, Duration::from_millis(1250));

        request.metadata.as_mut().unwrap().timeout_ms = Some(0);
        assert_eq!(
            managed_vllm_wire_request_timeouts(&request).total,
            DEFAULT_REQUEST_TIMEOUT
        );
    }

    #[test]
    fn wire_response_head_forwards_only_allowlisted_headers() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        headers.insert(hyper::header::CACHE_CONTROL, "no-store".parse().unwrap());
        headers.insert("x-request-id", "request-7".parse().unwrap());
        headers.insert("retry-after", "2".parse().unwrap());
        headers.insert("openai-processing-ms", "3.5".parse().unwrap());
        headers.insert("authorization", "secret".parse().unwrap());
        headers.insert("set-cookie", "private=true".parse().unwrap());

        let head = managed_vllm_wire_response_head(hyper::StatusCode::TOO_MANY_REQUESTS, &headers)
            .expect("allowlisted response head");
        assert_eq!(head.status, 429);
        assert_eq!(head.headers.len(), 5);
        assert!(head
            .headers
            .iter()
            .all(|header| header.value != b"secret" && header.value != b"private=true"));
        assert!(head.headers.iter().any(|header| {
            header.name == OpenAiWireHeaderName::RequestId && header.value == b"request-7"
        }));
    }

    #[test]
    fn stale_health_revision_cannot_restore_routability_or_revive_stopped_process() {
        let process = test_managed_process();
        assert!(!process.mark_routable(0, 0, 0), "Planned is not routable");

        process.generation.store(1, Ordering::Release);
        process
            .readiness
            .store(ManagedVllmReplicaState::Starting as u8, Ordering::Release);
        process.readiness_epoch.fetch_add(1, Ordering::AcqRel);
        let stale_probe = process.readiness_snapshot();

        process.mark_suspect();
        assert!(!process.mark_routable(stale_probe.0, stale_probe.1, stale_probe.2));
        let fresh_probe = process.readiness_snapshot();
        assert!(process.mark_routable(fresh_probe.0, fresh_probe.1, fresh_probe.2));
        assert_eq!(process.state(), ManagedVllmReplicaState::Routable);

        let pre_shutdown_probe = process.readiness_snapshot();
        assert!(process.terminate());
        process.mark_suspect();
        assert_eq!(process.state(), ManagedVllmReplicaState::Stopped);
        assert!(!process.mark_routable(
            pre_shutdown_probe.0,
            pre_shutdown_probe.1,
            pre_shutdown_probe.2
        ));
    }

    #[test]
    fn kv_detach_fence_invalidates_routable_and_stale_health_but_not_stopped() {
        let process = test_managed_process();
        process.generation.store(1, Ordering::Release);
        process
            .readiness
            .store(ManagedVllmReplicaState::Starting as u8, Ordering::Release);
        process.readiness_epoch.fetch_add(1, Ordering::AcqRel);

        let initial_probe = process.readiness_snapshot();
        assert!(process.mark_routable(initial_probe.0, initial_probe.1, initial_probe.2));
        let stale_health = process.readiness_snapshot();
        process.kv_readiness_fence.advance();

        assert!(process.ensure_routable().is_err());
        assert_eq!(process.state(), ManagedVllmReplicaState::Suspect);
        assert!(!process.mark_routable(stale_health.0, stale_health.1, stale_health.2));

        let reactivated_probe = process.readiness_snapshot();
        assert!(process.mark_routable(
            reactivated_probe.0,
            reactivated_probe.1,
            reactivated_probe.2
        ));
        assert_eq!(process.state(), ManagedVllmReplicaState::Routable);
        assert_eq!(
            process.published_kv_readiness_epoch.load(Ordering::Acquire),
            process.kv_readiness_fence.snapshot()
        );

        let stopped_probe = process.readiness_snapshot();
        assert!(process.terminate());
        process.kv_readiness_fence.advance();
        assert_eq!(process.state(), ManagedVllmReplicaState::Stopped);
        assert!(!process.mark_routable(stopped_probe.0, stopped_probe.1, stopped_probe.2));
        assert_eq!(process.state(), ManagedVllmReplicaState::Stopped);
    }

    #[test]
    fn oversized_response_is_not_treated_as_process_transport_failure() {
        assert!(!bridge_error_is_transport_failure(
            &ManagedVllmBridgeError::ResponseBodyExceeded { limit: 1 }
        ));
    }

    #[test]
    fn request_deadlines_do_not_evict_an_otherwise_healthy_replica() {
        for error in [
            ManagedVllmBridgeError::HeaderTimeout,
            ManagedVllmBridgeError::IdleBodyTimeout,
            ManagedVllmBridgeError::TotalTimeout,
        ] {
            assert!(!bridge_error_is_transport_failure(&error));
        }
        assert!(bridge_error_is_transport_failure(
            &ManagedVllmBridgeError::Request("connection reset".to_string())
        ));
        assert!(bridge_error_is_transport_failure(
            &ManagedVllmBridgeError::Body("connection reset".to_string())
        ));
    }

    #[tokio::test]
    async fn translated_stream_fails_if_upstream_eof_arrives_before_done() {
        let upstream: ManagedVllmSseStream = Box::pin(stream::iter(vec![Ok(
            br#"{"choices":[{"delta":{"content":"hello"}}]}"#.to_vec(),
        )]));
        let errors = Arc::new(AtomicU64::new(0));
        let generated_tokens = Arc::new(AtomicU64::new(0));
        let output = translate_managed_vllm_stream(
            upstream,
            test_managed_process(),
            errors.clone(),
            generated_tokens,
            true,
        )
        .collect::<Vec<_>>()
        .await;

        assert_eq!(output.len(), 2);
        assert_eq!(output[0].as_ref().unwrap().data, b"hello");
        assert!(output[1]
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("before the terminal [DONE]"));
        assert_eq!(errors.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn translated_stream_accepts_done_and_surfaces_upstream_error_events() {
        let successful: ManagedVllmSseStream = Box::pin(stream::iter(vec![
            Ok(br#"{"choices":[{"delta":{"content":"ok"}}]}"#.to_vec()),
            Ok(b"[DONE]".to_vec()),
        ]));
        let successful_errors = Arc::new(AtomicU64::new(0));
        let output = translate_managed_vllm_stream(
            successful,
            test_managed_process(),
            successful_errors.clone(),
            Arc::new(AtomicU64::new(0)),
            true,
        )
        .collect::<Vec<_>>()
        .await;
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].as_ref().unwrap().data, b"ok");
        assert_eq!(successful_errors.load(Ordering::Relaxed), 0);

        let failed: ManagedVllmSseStream = Box::pin(stream::iter(vec![Ok(
            br#"{"error":{"message":"engine failed"}}"#.to_vec(),
        )]));
        let failed_errors = Arc::new(AtomicU64::new(0));
        let output = translate_managed_vllm_stream(
            failed,
            test_managed_process(),
            failed_errors.clone(),
            Arc::new(AtomicU64::new(0)),
            true,
        )
        .collect::<Vec<_>>()
        .await;
        assert_eq!(output.len(), 1);
        assert!(output[0]
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("engine failed"));
        assert_eq!(failed_errors.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn chat_envelope_uses_the_internal_marker() {
        let messages = vec![crate::http::openai::types::ChatMessage {
            role: "user".to_string(),
            content: serde_json::json!("hello"),
        }];
        let encoded = managed_vllm_chat_input(&messages, &["stop".to_string()]).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(value[MANAGED_VLLM_CHAT_MARKER], true);
        assert_eq!(value["messages"][0]["role"], "user");
        assert_eq!(value["stop"][0], "stop");
    }
}
