use super::managed_vllm_bridge::{
    map_ureq_body_error, ManagedVllmBridgeError, ManagedVllmBridgeTelemetry, ManagedVllmByteStream,
    ManagedVllmHttpBridge, ManagedVllmRequestTimeouts, ManagedVllmSseStream,
};
use super::*;
use kapsl_engine_api::{
    CancellationToken, EngineStream, MemoryAllocation, MemoryAllocationClass,
    MemoryAllocationSource, MemoryDomain, MemoryReport, OpenAiWireFormat, OpenAiWireHeader,
    OpenAiWireHeaderName, OpenAiWireRequest, OpenAiWireResponse, OpenAiWireResponseHead,
    OpenAiWireStream, OpenAiWireStreamResponse,
};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs::{OpenOptions, Permissions};
use std::net::TcpListener as StdTcpListener;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::{ExitStatus, Stdio};

pub(crate) const MANAGED_VLLM_ADAPTER_ID: &str = "kapsl-vllm-connector";
pub(crate) const MANAGED_VLLM_ADAPTER_VERSION: &str = "0.7.0";
pub(crate) const MANAGED_VLLM_BACKEND_VERSION: &str = "0.26.1rc1.dev1130+g2ec6f0d71";
pub(crate) const MANAGED_VLLM_PROFILE_ID: &str = "vllm-v1-packed-cuda-ipc/flash-attn";
pub(crate) const MANAGED_VLLM_ELASTIC_PROFILE_ID: &str = "vllm-v1-packed-cuda-vmm/flash-attn-blnhc";
pub(crate) const MANAGED_VLLM_PYTHON_VERSION: &str = "3.12.3";
pub(crate) const MANAGED_VLLM_TORCH_VERSION: &str = "2.13.0+cu130";
pub(crate) const MANAGED_VLLM_TORCHVISION_VERSION: &str = "0.28.0+cu130";
pub(crate) const MANAGED_VLLM_TORCHAUDIO_VERSION: &str = "2.11.0+cu130";
pub(crate) const MANAGED_VLLM_CUDA_RUNTIME_VERSION: &str = "13.0";

const MANAGED_VLLM_PYTHON_ENV: &str = "KAPSL_VLLM_PYTHON";
const MANAGED_VLLM_BUNDLE_ENV: &str = "KAPSL_VLLM_BUNDLE";
const MANAGED_VLLM_MEMORY_MODE_ENV: &str = "KAPSL_VLLM_MEMORY_MODE";
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
const DEFAULT_KV_GROW_UTILIZATION_PERCENT: usize = 80;
const DEFAULT_KV_SHRINK_UTILIZATION_PERCENT: usize = 25;
const DEFAULT_KV_SHRINK_IDLE_SECONDS: u64 = 60;
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

pub(crate) fn certified_vllm_elastic_profile() -> String {
    format!(
        "{MANAGED_VLLM_ADAPTER_ID},{MANAGED_VLLM_ADAPTER_VERSION},{MANAGED_VLLM_BACKEND_VERSION},{MANAGED_VLLM_ELASTIC_PROFILE_ID}"
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
    pub(crate) shared_pool_profiles: Vec<String>,
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
        let shared_pool_profiles = vec![certified_vllm_profile(), certified_vllm_elastic_profile()];

        let python = discover_certified_vllm_python()?;
        log::info!(
            "Managed vLLM bundle: {} (vLLM {}, connector {}, profiles [{}, {}])",
            python.display(),
            MANAGED_VLLM_BACKEND_VERSION,
            MANAGED_VLLM_ADAPTER_VERSION,
            MANAGED_VLLM_PROFILE_ID,
            MANAGED_VLLM_ELASTIC_PROFILE_ID,
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
            shared_pool_profiles,
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

    fn coordinator(&self) -> Result<Arc<ExternalKvCoordinator>, String> {
        #[cfg(unix)]
        {
            self.coordinator
                .read()
                .clone()
                .ok_or_else(|| "managed vLLM control coordinator is not installed".to_string())
        }
        #[cfg(not(unix))]
        {
            Err("managed vLLM control coordinator requires Unix".to_string())
        }
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
            coordinator.retire_participants_after_backend_exit(participant_base)
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

    pub(crate) fn model_has_live_resize_headroom(&self, model_id: u32) -> bool {
        let runtimes = self
            .runtimes
            .lock()
            .get(&model_id)
            .cloned()
            .unwrap_or_default();
        if runtimes.is_empty() {
            return false;
        }
        let coordinator = match self.coordinator() {
            Ok(coordinator) => coordinator,
            Err(_) => return true,
        };
        runtimes
            .into_iter()
            .filter_map(|runtime| runtime.upgrade())
            .any(|runtime| {
                if runtime.process.spec.settings.live_resize.is_none() {
                    return false;
                }
                coordinator
                    .managed_vllm_resize_snapshot(&runtime.participant_id)
                    .map_or(true, |snapshot| {
                        snapshot.is_none_or(|snapshot| {
                            snapshot.failure.is_none()
                                && (snapshot.pending_generation.is_some()
                                    || snapshot.current_block_count < snapshot.maximum_block_count)
                        })
                    })
            })
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
    elastic_profile: String,
    kv_abi_major: u64,
    kv_abi_minor: u64,
    cuda_runtime: String,
    cuda_available: bool,
    planner_schema_version: u64,
    exact_cache_memory_flag: bool,
    no_async_scheduling_flag: bool,
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
from kapsl_vllm_connector import ADAPTER_PROFILE_ID, ADAPTER_VERSION, ELASTIC_ADAPTER_PROFILE_ID
from kapsl_vllm_connector.contract import ABI_VERSION
from kapsl_vllm_connector.planning import PLANNER_SCHEMA_VERSION, planner_json_schema
from vllm.engine.arg_utils import EngineArgs
parser = argparse.ArgumentParser(add_help=False)
EngineArgs.add_cli_args(parser)
exact_cache_memory_flag = any(
    "--kv-cache-memory-bytes" in action.option_strings
    for action in parser._actions
)
no_async_scheduling_flag = any(
    "--no-async-scheduling" in action.option_strings
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
    "elastic_profile": ELASTIC_ADAPTER_PROFILE_ID,
    "kv_abi_major": ABI_VERSION["major"],
    "kv_abi_minor": ABI_VERSION["minor"],
    "cuda_runtime": str(torch.version.cuda),
    "cuda_available": bool(torch.cuda.is_available()),
    "planner_schema_version": PLANNER_SCHEMA_VERSION,
    "planner_schema_const": planner_schema["properties"]["schema_version"]["const"],
    "exact_cache_memory_flag": exact_cache_memory_flag,
    "no_async_scheduling_flag": no_async_scheduling_flag,
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
        elastic_profile: MANAGED_VLLM_ELASTIC_PROFILE_ID.to_string(),
        kv_abi_major: MANAGED_VLLM_KV_ABI_MAJOR,
        kv_abi_minor: MANAGED_VLLM_KV_ABI_MINOR,
        cuda_runtime: MANAGED_VLLM_CUDA_RUNTIME_VERSION.to_string(),
        cuda_available: true,
        planner_schema_version: 1,
        exact_cache_memory_flag: true,
        no_async_scheduling_flag: true,
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManagedVllmLiveResizeSettings {
    maximum_concurrency: usize,
    grow_utilization_percent: usize,
    shrink_utilization_percent: usize,
    shrink_idle: Duration,
}

#[derive(Clone, Debug)]
struct ManagedVllmSettings {
    /// Compatibility value retained only for explicit legacy-fraction
    /// manifests. Exact auto/fixed launches are driven exclusively by the
    /// certified planner and authority grant.
    gpu_memory_utilization: f64,
    kv_cache_policy: ManagedVllmKvCachePolicy,
    live_resize: Option<ManagedVllmLiveResizeSettings>,
    legacy_top_level_fraction_authored: bool,
    max_model_len: usize,
    startup_timeout: Duration,
}

const MANAGED_VLLM_PLANNER_SCHEMA_VERSION: u64 = 1;
pub(crate) const MANAGED_VLLM_KV_ABI_MAJOR: u64 = 1;
pub(crate) const MANAGED_VLLM_KV_ABI_MINOR: u64 = 5;

#[derive(Clone, Debug)]
struct ManagedVllmPlannerInvocation {
    model_root: PathBuf,
    model_fingerprint: String,
    participant_base: String,
    device_ids: Vec<usize>,
    tensor_parallel_size: usize,
    max_model_len: usize,
    resolved_target_concurrency: usize,
    policy: ManagedVllmKvCachePolicy,
    live_resize: Option<ManagedVllmLiveResizeSettings>,
    timeout: Duration,
    cuda_visible_devices: String,
    output_path: PathBuf,
    log_path: PathBuf,
}

#[derive(Clone, Debug)]
struct ManagedVllmExactPlanTemplate {
    grant_request: ProvisionalKvGrantRequest,
    requested_bytes_per_rank: u64,
    minimum_bytes_per_rank: u64,
    target_concurrency: u64,
    maximum_bytes_per_rank: u64,
    maximum_block_count: u64,
    live_resize: bool,
    fixed_overhead_blocks: u64,
    sequence_blocks: u64,
    maximum_concurrency: Option<u64>,
    maximum_byte_cap: Option<u64>,
    strict_concurrency: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManagedVllmPlannerEnvelope {
    schema_version: u64,
    status: String,
    supported: bool,
    geometry_digest: String,
    geometry: serde_json::Value,
    policy: ManagedVllmPlannerPolicy,
    sizing: ManagedVllmPlannerSizing,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ManagedVllmPlannerPolicy {
    target_concurrency: u64,
    #[serde(default)]
    maximum_concurrency: Option<u64>,
    headroom_percent: u64,
    prefix_blocks: u64,
    alignment_blocks: u64,
    #[serde(default)]
    min_bytes: Option<u64>,
    #[serde(default)]
    max_bytes: Option<u64>,
    strict_concurrency: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManagedVllmPlannerSizing {
    ranks: Vec<ManagedVllmPlannerRankSizing>,
    total_desired_bytes: u64,
    total_maximum_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ManagedVllmPlannerRankSizing {
    rank: u64,
    device_id: u64,
    bytes_per_block: u64,
    sequence_blocks: u64,
    minimum_blocks: u64,
    minimum_bytes: u64,
    base_blocks: u64,
    headroom_blocks: u64,
    desired_blocks: u64,
    desired_bytes: u64,
    maximum_blocks: u64,
    maximum_bytes: u64,
    effective_target_concurrency: u64,
    concurrency_reduced: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ManagedVllmPlannerGeometry {
    identity: ManagedVllmPlannerIdentity,
    model_fingerprint: String,
    max_model_len: u64,
    tensor_parallel_size: u64,
    attention_backend: String,
    layout_id: String,
    total_pool_bytes_per_block: u64,
    ranks: Vec<ManagedVllmPlannerRankGeometry>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ManagedVllmPlannerIdentity {
    adapter_id: String,
    adapter_version: String,
    backend_version: String,
    profile_id: String,
    layout_version: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ManagedVllmPlannerRankGeometry {
    rank: u64,
    device_id: u64,
    pool_bytes_per_block: u64,
    fixed_overhead_blocks: u64,
    required_blocks_per_sequence: u64,
    cache_groups: Vec<ManagedVllmPlannerCacheGroup>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ManagedVllmPlannerCacheGroup {
    group_id: String,
    layers: Vec<String>,
    block_size_tokens: u64,
    bytes_per_group_block: u64,
    required_blocks_per_sequence: u64,
    kv_heads: u64,
    key_head_dim: u64,
    value_head_dim: u64,
    element_type: ManagedVllmPlannerElementType,
    policy: ManagedVllmPlannerCachePolicy,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ManagedVllmPlannerElementType {
    name: String,
    bits: u64,
    bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ManagedVllmPlannerCachePolicy {
    kind: String,
    #[serde(default)]
    window_tokens: Option<u64>,
    #[serde(default)]
    extra_retained_tokens: Option<u64>,
}

impl ManagedVllmSettings {
    fn from_manifest(manifest: &Manifest) -> Result<Self, String> {
        let mut settings = Self {
            gpu_memory_utilization: DEFAULT_LEGACY_GPU_MEMORY_UTILIZATION,
            kv_cache_policy: ManagedVllmKvCachePolicy::Auto {
                target_concurrency: None,
                headroom_percent: DEFAULT_KV_HEADROOM_PERCENT,
                min_bytes: None,
                max_bytes: None,
                strict: false,
            },
            live_resize: None,
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
            settings.apply_memory_mode_override(
                std::env::var(MANAGED_VLLM_MEMORY_MODE_ENV).ok().as_deref(),
            )?;
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
            let (policy, compatibility_fraction, live_resize) = parse_kv_cache_policy(raw)?;
            settings.kv_cache_policy = policy;
            settings.live_resize = live_resize;
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
        settings.apply_memory_mode_override(
            std::env::var(MANAGED_VLLM_MEMORY_MODE_ENV).ok().as_deref(),
        )?;
        Ok(settings)
    }

    fn apply_memory_mode_override(&mut self, mode: Option<&str>) -> Result<(), String> {
        let Some(mode) = mode.map(str::trim).filter(|mode| !mode.is_empty()) else {
            return Ok(());
        };
        match mode.to_ascii_lowercase().as_str() {
            "exact" | "auto" => {
                if matches!(
                    self.kv_cache_policy,
                    ManagedVllmKvCachePolicy::LegacyFraction { .. }
                ) {
                    return Err(format!(
                        "{MANAGED_VLLM_MEMORY_MODE_ENV}={mode} conflicts with explicitly configured legacy_fraction memory sizing"
                    ));
                }
            }
            "legacy" | "legacy-fraction" | "legacy_fraction" => {
                self.kv_cache_policy = ManagedVllmKvCachePolicy::LegacyFraction {
                    gpu_memory_utilization: self.gpu_memory_utilization,
                };
            }
            _ => {
                return Err(format!(
                    "{MANAGED_VLLM_MEMORY_MODE_ENV} must be exact or legacy-fraction"
                ));
            }
        }
        Ok(())
    }

    fn validate_launch_policy(&self) -> Result<(), String> {
        if let Some(live_resize) = &self.live_resize {
            let ManagedVllmKvCachePolicy::Auto {
                target_concurrency, ..
            } = &self.kv_cache_policy
            else {
                return Err(
                    "managed vLLM live_resize is supported only with kv_cache.mode: auto"
                        .to_string(),
                );
            };
            if target_concurrency.is_some_and(|target| live_resize.maximum_concurrency < target) {
                return Err(
                    "managed vLLM live_resize.maximum_concurrency cannot be below target_concurrency"
                        .to_string(),
                );
            }
        }
        Ok(())
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
) -> Result<
    (
        ManagedVllmKvCachePolicy,
        Option<f64>,
        Option<ManagedVllmLiveResizeSettings>,
    ),
    String,
> {
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
            "live_resize",
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
            let live_resize = value("live_resize")
                .map(parse_live_resize_settings)
                .transpose()?
                .flatten();
            Ok((
                ManagedVllmKvCachePolicy::Auto {
                    target_concurrency,
                    headroom_percent,
                    min_bytes,
                    max_bytes,
                    strict,
                },
                None,
                live_resize,
            ))
        }
        "fixed" => {
            let bytes = value("bytes")
                .ok_or_else(|| format!("{PATH}.bytes is required when mode is fixed"))
                .and_then(|raw| parse_positive_usize(raw, &format!("{PATH}.bytes")))?;
            Ok((ManagedVllmKvCachePolicy::Fixed { bytes }, None, None))
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
                None,
            ))
        }
        _ => unreachable!("mode was validated above"),
    }
}

fn parse_live_resize_settings(
    raw: &serde_yaml::Value,
) -> Result<Option<ManagedVllmLiveResizeSettings>, String> {
    const PATH: &str = "metadata.serving.vllm.kv_cache.live_resize";
    let object = raw
        .as_mapping()
        .ok_or_else(|| format!("{PATH} must be an object"))?;
    let value = |name: &str| object.get(serde_yaml::Value::String(name.to_string()));
    let allowed = [
        "enabled",
        "maximum_concurrency",
        "grow_utilization_percent",
        "shrink_utilization_percent",
        "shrink_idle_seconds",
    ];
    for key in object.keys() {
        let key = key
            .as_str()
            .ok_or_else(|| format!("{PATH} field names must be strings"))?;
        if !allowed.contains(&key) {
            return Err(format!("{PATH}.{key} is not supported"));
        }
    }
    let enabled = value("enabled")
        .map(|raw| {
            raw.as_bool()
                .ok_or_else(|| format!("{PATH}.enabled must be a boolean"))
        })
        .transpose()?
        .unwrap_or(true);
    if !enabled {
        if object.len() != usize::from(value("enabled").is_some()) {
            return Err(format!(
                "{PATH} cannot configure resize thresholds when enabled is false"
            ));
        }
        return Ok(None);
    }
    let maximum_concurrency = value("maximum_concurrency")
        .ok_or_else(|| format!("{PATH}.maximum_concurrency is required when enabled"))
        .and_then(|raw| parse_positive_usize(raw, &format!("{PATH}.maximum_concurrency")))?;
    let grow_utilization_percent = value("grow_utilization_percent")
        .map(|raw| parse_positive_usize(raw, &format!("{PATH}.grow_utilization_percent")))
        .transpose()?
        .unwrap_or(DEFAULT_KV_GROW_UTILIZATION_PERCENT);
    let shrink_utilization_percent = value("shrink_utilization_percent")
        .map(|raw| parse_usize(raw, &format!("{PATH}.shrink_utilization_percent")))
        .transpose()?
        .unwrap_or(DEFAULT_KV_SHRINK_UTILIZATION_PERCENT);
    if grow_utilization_percent > 100 || shrink_utilization_percent >= grow_utilization_percent {
        return Err(format!(
            "{PATH} requires 0 <= shrink_utilization_percent < grow_utilization_percent <= 100"
        ));
    }
    let shrink_idle_seconds = value("shrink_idle_seconds")
        .map(|raw| parse_positive_usize(raw, &format!("{PATH}.shrink_idle_seconds")))
        .transpose()?
        .unwrap_or(DEFAULT_KV_SHRINK_IDLE_SECONDS as usize);
    if shrink_idle_seconds > 3600 {
        return Err(format!("{PATH}.shrink_idle_seconds must not exceed 3600"));
    }
    Ok(Some(ManagedVllmLiveResizeSettings {
        maximum_concurrency,
        grow_utilization_percent,
        shrink_utilization_percent,
        shrink_idle: Duration::from_secs(shrink_idle_seconds as u64),
    }))
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

fn planner_policy_for_invocation(
    policy: &ManagedVllmKvCachePolicy,
    resolved_target_concurrency: usize,
    live_resize: Option<&ManagedVllmLiveResizeSettings>,
) -> Result<ManagedVllmPlannerPolicy, String> {
    let to_u64 = |value: usize, field: &str| {
        u64::try_from(value).map_err(|_| format!("managed vLLM {field} exceeds uint64"))
    };
    match policy {
        ManagedVllmKvCachePolicy::LegacyFraction { .. } => {
            Err("legacy_fraction does not invoke the exact managed vLLM planner".to_string())
        }
        ManagedVllmKvCachePolicy::Fixed { bytes } => {
            let bytes = to_u64(*bytes, "fixed KV bytes")?;
            Ok(ManagedVllmPlannerPolicy {
                target_concurrency: 1,
                maximum_concurrency: None,
                headroom_percent: 0,
                prefix_blocks: 0,
                alignment_blocks: 1,
                min_bytes: Some(bytes),
                max_bytes: Some(bytes),
                strict_concurrency: true,
            })
        }
        ManagedVllmKvCachePolicy::Auto {
            target_concurrency,
            headroom_percent,
            min_bytes,
            max_bytes,
            strict,
        } => Ok(ManagedVllmPlannerPolicy {
            target_concurrency: to_u64(
                target_concurrency.unwrap_or(resolved_target_concurrency),
                "target concurrency",
            )?,
            maximum_concurrency: live_resize
                .map(|settings| to_u64(settings.maximum_concurrency, "maximum live KV concurrency"))
                .transpose()?,
            headroom_percent: to_u64(*headroom_percent, "KV headroom percent")?,
            prefix_blocks: 0,
            alignment_blocks: 1,
            min_bytes: min_bytes
                .map(|value| to_u64(value, "minimum KV bytes"))
                .transpose()?,
            max_bytes: max_bytes
                .map(|value| to_u64(value, "maximum KV bytes"))
                .transpose()?,
            strict_concurrency: *strict,
        }),
    }
}

fn parse_managed_vllm_planner_output(
    encoded: &str,
    invocation: &ManagedVllmPlannerInvocation,
) -> Result<ManagedVllmExactPlanTemplate, String> {
    let envelope: ManagedVllmPlannerEnvelope = serde_json::from_str(encoded)
        .map_err(|error| format!("managed vLLM planner emitted invalid JSON: {error}"))?;
    if envelope.schema_version != MANAGED_VLLM_PLANNER_SCHEMA_VERSION
        || envelope.status != "planned"
        || !envelope.supported
    {
        return Err(format!(
            "managed vLLM planner did not emit supported schema {} output",
            MANAGED_VLLM_PLANNER_SCHEMA_VERSION
        ));
    }
    let expected_digest = managed_vllm_geometry_digest(&envelope.geometry)?;
    if envelope.geometry_digest != expected_digest {
        return Err("managed vLLM planner geometry digest does not match its payload".to_string());
    }
    let geometry: ManagedVllmPlannerGeometry = serde_json::from_value(envelope.geometry)
        .map_err(|error| format!("managed vLLM planner geometry is malformed: {error}"))?;
    let expected_policy = planner_policy_for_invocation(
        &invocation.policy,
        invocation.resolved_target_concurrency,
        invocation.live_resize.as_ref(),
    )?;
    if envelope.policy != expected_policy {
        return Err(format!(
            "managed vLLM planner policy drifted from the requested policy: {:?} != {:?}",
            envelope.policy, expected_policy
        ));
    }
    validate_managed_vllm_planner_geometry(
        &geometry,
        &envelope.sizing,
        invocation,
        &envelope.policy,
    )?;
    let candidates = managed_vllm_grant_candidates(
        &geometry.ranks[0],
        &envelope.sizing.ranks[0],
        &envelope.policy,
    )?;
    let memory_domains = invocation
        .device_ids
        .iter()
        .map(|device_id| {
            u32::try_from(*device_id)
                .map(|device_id| kapsl_kv_abi::KvMemoryDomain::Cuda { device_id })
                .map_err(|_| format!("managed vLLM CUDA device {device_id} exceeds uint32"))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let group_ids = geometry.ranks[0]
        .cache_groups
        .iter()
        .map(|group| group.group_id.clone())
        .collect::<BTreeSet<_>>();
    let live_resize = invocation.live_resize.is_some();
    let maximum_block_count = envelope.sizing.ranks[0].maximum_blocks;
    let maximum_bytes_per_rank = envelope.sizing.ranks[0].maximum_bytes;
    Ok(ManagedVllmExactPlanTemplate {
        grant_request: ProvisionalKvGrantRequest {
            participant_base: invocation.participant_base.clone(),
            model_fingerprint: invocation.model_fingerprint.clone(),
            geometry_digest: envelope.geometry_digest,
            adapter_profile: kapsl_kv_abi::KvAdapterProfile {
                adapter_id: MANAGED_VLLM_ADAPTER_ID.to_string(),
                adapter_version: MANAGED_VLLM_ADAPTER_VERSION.to_string(),
                backend_version: MANAGED_VLLM_BACKEND_VERSION.to_string(),
                profile_id: if live_resize {
                    MANAGED_VLLM_ELASTIC_PROFILE_ID
                } else {
                    MANAGED_VLLM_PROFILE_ID
                }
                .to_string(),
            },
            capacity_pool_id: "vllm.pool.0".to_string(),
            group_ids,
            memory_domains,
            candidates,
            maximum_block_count: live_resize.then_some(maximum_block_count),
            ttl: invocation.timeout,
        },
        requested_bytes_per_rank: envelope.sizing.ranks[0].desired_bytes,
        minimum_bytes_per_rank: envelope.sizing.ranks[0].minimum_bytes,
        target_concurrency: envelope.policy.target_concurrency,
        maximum_bytes_per_rank,
        maximum_block_count,
        live_resize,
        fixed_overhead_blocks: geometry.ranks[0].fixed_overhead_blocks,
        sequence_blocks: geometry.ranks[0].required_blocks_per_sequence,
        maximum_concurrency: envelope.policy.maximum_concurrency,
        maximum_byte_cap: envelope.policy.max_bytes,
        strict_concurrency: envelope.policy.strict_concurrency,
    })
}

fn align_managed_vllm_elastic_plan(
    template: &mut ManagedVllmExactPlanTemplate,
    alignment_blocks: u64,
) -> Result<(), String> {
    if !template.live_resize || alignment_blocks == 0 {
        return Err(
            "managed vLLM elastic alignment requires a live plan and non-zero alignment"
                .to_string(),
        );
    }
    let bytes_per_block = template
        .grant_request
        .candidates
        .first()
        .map(|candidate| candidate.bytes_per_block)
        .filter(|stride| *stride > 0)
        .ok_or_else(|| "managed vLLM elastic plan has no block stride".to_string())?;
    let minimum_blocks = checked_u64_round_up(
        checked_u64_ceil_div(
            template.minimum_bytes_per_rank,
            bytes_per_block,
            "elastic minimum blocks",
        )?,
        alignment_blocks,
        "elastic minimum alignment",
    )?;
    let cap_blocks = template
        .maximum_byte_cap
        .map(|cap| (cap / bytes_per_block / alignment_blocks).saturating_mul(alignment_blocks));
    if cap_blocks.is_some_and(|cap| cap < minimum_blocks) {
        return Err(
            "managed vLLM max_bytes cannot satisfy CUDA VMM allocation granularity".to_string(),
        );
    }

    let mut aligned = Vec::new();
    for candidate in &template.grant_request.candidates {
        let rounded_up = checked_u64_round_up(
            candidate.block_count,
            alignment_blocks,
            "elastic grant alignment",
        )?;
        let block_count = cap_blocks
            .filter(|cap| rounded_up > *cap)
            .unwrap_or(rounded_up);
        if block_count < minimum_blocks
            || aligned
                .last()
                .is_some_and(|previous: &ProvisionalKvCandidate| {
                    previous.block_count == block_count
                })
        {
            continue;
        }
        let effective = template.target_concurrency.min(
            block_count.saturating_sub(template.fixed_overhead_blocks) / template.sequence_blocks,
        );
        if effective == 0 {
            continue;
        }
        aligned.push(ProvisionalKvCandidate {
            block_count,
            bytes_per_block,
            effective_target_concurrency: effective,
        });
    }
    let preferred = aligned.first().ok_or_else(|| {
        "managed vLLM CUDA VMM alignment removed every valid grant candidate".to_string()
    })?;
    if template.strict_concurrency
        && preferred.effective_target_concurrency < template.target_concurrency
    {
        return Err(format!(
            "managed vLLM CUDA VMM alignment reduces strict target concurrency from {} to {}",
            template.target_concurrency, preferred.effective_target_concurrency
        ));
    }

    let mut maximum_blocks = checked_u64_round_up(
        template.maximum_block_count,
        alignment_blocks,
        "elastic virtual maximum alignment",
    )?;
    if let Some(cap) = cap_blocks {
        maximum_blocks = maximum_blocks.min(cap);
    }
    if maximum_blocks < preferred.block_count {
        return Err(
            "managed vLLM aligned virtual maximum is below initial physical capacity".to_string(),
        );
    }
    if let Some(maximum_concurrency) = template.maximum_concurrency {
        let effective_maximum = maximum_blocks.saturating_sub(template.fixed_overhead_blocks)
            / template.sequence_blocks;
        if effective_maximum < maximum_concurrency {
            return Err(format!(
                "managed vLLM aligned maximum can hold {effective_maximum} full sequences, below configured maximum_concurrency {maximum_concurrency}"
            ));
        }
    }
    template.requested_bytes_per_rank = preferred
        .block_count
        .checked_mul(bytes_per_block)
        .ok_or_else(|| "managed vLLM aligned desired bytes overflowed".to_string())?;
    template.minimum_bytes_per_rank = minimum_blocks
        .checked_mul(bytes_per_block)
        .ok_or_else(|| "managed vLLM aligned minimum bytes overflowed".to_string())?;
    template.maximum_block_count = maximum_blocks;
    template.maximum_bytes_per_rank = maximum_blocks
        .checked_mul(bytes_per_block)
        .ok_or_else(|| "managed vLLM aligned maximum bytes overflowed".to_string())?;
    template.grant_request.maximum_block_count = Some(maximum_blocks);
    template.grant_request.candidates = aligned;
    Ok(())
}

fn validate_managed_vllm_planner_geometry(
    geometry: &ManagedVllmPlannerGeometry,
    sizing: &ManagedVllmPlannerSizing,
    invocation: &ManagedVllmPlannerInvocation,
    policy: &ManagedVllmPlannerPolicy,
) -> Result<(), String> {
    let expected_tp = u64::try_from(invocation.tensor_parallel_size)
        .map_err(|_| "managed vLLM tensor parallel size exceeds uint64".to_string())?;
    let expected_max_len = u64::try_from(invocation.max_model_len)
        .map_err(|_| "managed vLLM max_model_len exceeds uint64".to_string())?;
    let expected_profile = if invocation.live_resize.is_some() {
        MANAGED_VLLM_ELASTIC_PROFILE_ID
    } else {
        MANAGED_VLLM_PROFILE_ID
    };
    if geometry.identity
        != (ManagedVllmPlannerIdentity {
            adapter_id: MANAGED_VLLM_ADAPTER_ID.to_string(),
            adapter_version: MANAGED_VLLM_ADAPTER_VERSION.to_string(),
            backend_version: MANAGED_VLLM_BACKEND_VERSION.to_string(),
            profile_id: expected_profile.to_string(),
            layout_version: 1,
        })
        || geometry.model_fingerprint != invocation.model_fingerprint
        || geometry.max_model_len != expected_max_len
        || geometry.tensor_parallel_size != expected_tp
        || geometry.attention_backend != "FLASH_ATTN"
        || geometry.layout_id.trim().is_empty()
        || (invocation.live_resize.is_some() && !geometry.layout_id.eq_ignore_ascii_case("BLNHC"))
        || geometry.ranks.len() != invocation.tensor_parallel_size
        || sizing.ranks.len() != invocation.tensor_parallel_size
    {
        return Err(
            "managed vLLM planner identity/model/profile/world does not match the launch"
                .to_string(),
        );
    }
    let mut total_stride = 0_u64;
    let mut total_desired = 0_u64;
    let mut total_maximum = 0_u64;
    let first_rank = geometry
        .ranks
        .first()
        .ok_or_else(|| "managed vLLM planner returned no rank geometry".to_string())?;
    for (index, (rank, rank_sizing)) in geometry.ranks.iter().zip(&sizing.ranks).enumerate() {
        let expected_device = u64::try_from(invocation.device_ids[index])
            .map_err(|_| "managed vLLM device ID exceeds uint64".to_string())?;
        if rank.rank != index as u64
            || rank.device_id != expected_device
            || rank_sizing.rank != index as u64
            || rank_sizing.device_id != expected_device
            || rank.pool_bytes_per_block == 0
            || rank.fixed_overhead_blocks == 0
            || rank.required_blocks_per_sequence == 0
            || rank.cache_groups.is_empty()
            || rank_sizing.bytes_per_block != rank.pool_bytes_per_block
            || rank_sizing.sequence_blocks != rank.required_blocks_per_sequence
            || rank_sizing.desired_bytes
                != checked_u64_mul(
                    rank_sizing.desired_blocks,
                    rank.pool_bytes_per_block,
                    "planner desired bytes",
                )?
            || rank_sizing.minimum_bytes
                != checked_u64_mul(
                    rank_sizing.minimum_blocks,
                    rank.pool_bytes_per_block,
                    "planner minimum bytes",
                )?
            || rank_sizing.maximum_bytes
                != checked_u64_mul(
                    rank_sizing.maximum_blocks,
                    rank.pool_bytes_per_block,
                    "planner maximum bytes",
                )?
            || rank_sizing.maximum_blocks < rank_sizing.desired_blocks
        {
            return Err(format!(
                "managed vLLM planner rank {index} has inconsistent placement or sizing"
            ));
        }
        validate_managed_vllm_rank_sizing(rank, rank_sizing, policy)?;
        let mut group_ids = HashSet::new();
        let mut layers = HashSet::new();
        let mut required_blocks = 0_u64;
        let mut maximum_group_stride = 0_u64;
        for (group_index, group) in rank.cache_groups.iter().enumerate() {
            if group.group_id != format!("vllm.group.{group_index}")
                || !group_ids.insert(group.group_id.as_str())
                || group.layers.is_empty()
                || group
                    .layers
                    .iter()
                    .any(|layer| layer.trim().is_empty() || !layers.insert(layer.as_str()))
                || group.block_size_tokens == 0
                || group.bytes_per_group_block == 0
                || group.bytes_per_group_block > rank.pool_bytes_per_block
                || group.required_blocks_per_sequence == 0
                || group.kv_heads == 0
                || group.key_head_dim == 0
                || group.value_head_dim == 0
                || group.element_type.name.trim().is_empty()
                || group.element_type.bits == 0
                || group.element_type.bytes == 0
                || group.element_type.bits != group.element_type.bytes.saturating_mul(8)
                || !matches!(
                    group.policy.kind.as_str(),
                    "full_attention" | "sliding_window"
                )
                || (group.policy.kind == "full_attention"
                    && (group.policy.window_tokens.is_some()
                        || group.policy.extra_retained_tokens.is_some()))
                || (group.policy.kind == "sliding_window"
                    && (group.policy.window_tokens.is_none_or(|window| window == 0)
                        || group.policy.extra_retained_tokens.is_none()))
            {
                return Err(format!(
                    "managed vLLM planner cache group {group_index} is not a certified packed geometry"
                ));
            }
            required_blocks = checked_u64_add(
                required_blocks,
                group.required_blocks_per_sequence,
                "planner per-sequence group blocks",
            )?;
            maximum_group_stride = maximum_group_stride.max(group.bytes_per_group_block);
        }
        if required_blocks != rank.required_blocks_per_sequence
            || maximum_group_stride != rank.pool_bytes_per_block
        {
            return Err(format!(
                "managed vLLM planner rank {index} group accounting does not match the packed pool"
            ));
        }
        if index != 0
            && (rank.pool_bytes_per_block != first_rank.pool_bytes_per_block
                || rank.fixed_overhead_blocks != first_rank.fixed_overhead_blocks
                || rank.required_blocks_per_sequence != first_rank.required_blocks_per_sequence
                || rank.cache_groups != first_rank.cache_groups
                || rank_sizing.desired_blocks != sizing.ranks[0].desired_blocks
                || rank_sizing.minimum_blocks != sizing.ranks[0].minimum_blocks
                || rank_sizing.maximum_blocks != sizing.ranks[0].maximum_blocks)
        {
            return Err(
                "managed vLLM planner tensor-parallel ranks have divergent cache geometry"
                    .to_string(),
            );
        }
        total_stride = checked_u64_add(
            total_stride,
            rank.pool_bytes_per_block,
            "planner total block stride",
        )?;
        total_desired = checked_u64_add(
            total_desired,
            rank_sizing.desired_bytes,
            "planner total desired bytes",
        )?;
        total_maximum = checked_u64_add(
            total_maximum,
            rank_sizing.maximum_bytes,
            "planner total maximum bytes",
        )?;
    }
    if geometry.total_pool_bytes_per_block != total_stride
        || sizing.total_desired_bytes != total_desired
        || sizing.total_maximum_bytes != total_maximum
    {
        return Err("managed vLLM planner aggregate bytes are inconsistent".to_string());
    }
    Ok(())
}

fn validate_managed_vllm_rank_sizing(
    rank: &ManagedVllmPlannerRankGeometry,
    sizing: &ManagedVllmPlannerRankSizing,
    policy: &ManagedVllmPlannerPolicy,
) -> Result<(), String> {
    if policy.target_concurrency == 0
        || policy.alignment_blocks == 0
        || policy.headroom_percent > 100
    {
        return Err("managed vLLM planner policy contains invalid bounds".to_string());
    }
    let minimum = checked_u64_round_up(
        checked_u64_add(
            rank.fixed_overhead_blocks,
            rank.required_blocks_per_sequence,
            "planner minimum blocks",
        )?,
        policy.alignment_blocks,
        "planner minimum alignment",
    )?;
    let workload = checked_u64_mul(
        rank.required_blocks_per_sequence,
        policy.target_concurrency,
        "planner workload blocks",
    )?;
    let workload_with_prefix =
        checked_u64_add(workload, policy.prefix_blocks, "planner prefix blocks")?;
    let headroom = checked_u64_ceil_div(
        checked_u64_mul(
            workload_with_prefix,
            policy.headroom_percent,
            "planner headroom",
        )?,
        100,
        "planner headroom",
    )?;
    let base = checked_u64_add(
        rank.fixed_overhead_blocks,
        workload_with_prefix,
        "planner base blocks",
    )?;
    let mut desired = checked_u64_round_up(
        checked_u64_add(base, headroom, "planner desired blocks")?,
        policy.alignment_blocks,
        "planner desired alignment",
    )?;
    let floor = policy
        .min_bytes
        .map(|bytes| {
            checked_u64_round_up(
                checked_u64_ceil_div(bytes, rank.pool_bytes_per_block, "planner min bytes")?,
                policy.alignment_blocks,
                "planner min byte alignment",
            )
        })
        .transpose()?;
    if let Some(floor) = floor {
        desired = desired.max(floor);
    }
    if let Some(max_bytes) = policy.max_bytes {
        let cap = (max_bytes / rank.pool_bytes_per_block / policy.alignment_blocks)
            * policy.alignment_blocks;
        if cap < minimum || floor.is_some_and(|floor| cap < floor) {
            return Err("managed vLLM planner byte cap violates its minimum".to_string());
        }
        desired = desired.min(cap);
    }
    let mut maximum = desired;
    if let Some(maximum_concurrency) = policy.maximum_concurrency {
        let maximum_workload = checked_u64_add(
            checked_u64_mul(
                rank.required_blocks_per_sequence,
                maximum_concurrency,
                "planner maximum workload",
            )?,
            policy.prefix_blocks,
            "planner maximum prefix blocks",
        )?;
        let maximum_headroom = checked_u64_ceil_div(
            checked_u64_mul(
                maximum_workload,
                policy.headroom_percent,
                "planner maximum headroom",
            )?,
            100,
            "planner maximum headroom",
        )?;
        maximum = checked_u64_round_up(
            checked_u64_add(
                checked_u64_add(
                    rank.fixed_overhead_blocks,
                    maximum_workload,
                    "planner maximum blocks",
                )?,
                maximum_headroom,
                "planner maximum blocks",
            )?,
            policy.alignment_blocks,
            "planner maximum alignment",
        )?;
        if let Some(max_bytes) = policy.max_bytes {
            let cap = (max_bytes / rank.pool_bytes_per_block / policy.alignment_blocks)
                * policy.alignment_blocks;
            maximum = maximum.min(cap);
        }
        if maximum < desired {
            return Err(
                "managed vLLM planner maximum concurrency cannot contain initial capacity"
                    .to_string(),
            );
        }
    }
    let effective = policy.target_concurrency.min(
        desired.saturating_sub(rank.fixed_overhead_blocks) / rank.required_blocks_per_sequence,
    );
    if sizing.minimum_blocks != minimum
        || sizing.base_blocks != base
        || sizing.headroom_blocks != headroom
        || sizing.desired_blocks != desired
        || sizing.maximum_blocks != maximum
        || sizing.effective_target_concurrency != effective
        || sizing.concurrency_reduced != (effective < policy.target_concurrency)
        || effective == 0
        || (policy.strict_concurrency && effective < policy.target_concurrency)
    {
        return Err("managed vLLM planner rank arithmetic failed the Rust cross-check".to_string());
    }
    Ok(())
}

fn managed_vllm_grant_candidates(
    rank: &ManagedVllmPlannerRankGeometry,
    sizing: &ManagedVllmPlannerRankSizing,
    policy: &ManagedVllmPlannerPolicy,
) -> Result<Vec<ProvisionalKvCandidate>, String> {
    let mut candidates = std::collections::BTreeMap::<u64, u64>::new();
    let mut add_candidate = |blocks: u64| -> Result<(), String> {
        if blocks < sizing.minimum_blocks
            || blocks > sizing.desired_blocks
            || !blocks.is_multiple_of(policy.alignment_blocks)
        {
            return Ok(());
        }
        let effective = policy.target_concurrency.min(
            blocks.saturating_sub(rank.fixed_overhead_blocks) / rank.required_blocks_per_sequence,
        );
        if effective == 0 || (policy.strict_concurrency && effective < policy.target_concurrency) {
            return Ok(());
        }
        candidates
            .entry(blocks)
            .and_modify(|current| *current = (*current).max(effective))
            .or_insert(effective);
        Ok(())
    };
    add_candidate(sizing.desired_blocks)?;
    let floor = policy
        .min_bytes
        .map(|bytes| {
            checked_u64_round_up(
                checked_u64_ceil_div(bytes, rank.pool_bytes_per_block, "candidate min bytes")?,
                policy.alignment_blocks,
                "candidate min alignment",
            )
        })
        .transpose()?
        .unwrap_or(sizing.minimum_blocks);
    let start = if policy.strict_concurrency {
        policy.target_concurrency
    } else {
        sizing.effective_target_concurrency
    };
    for concurrency in (1..=start).rev() {
        let workload = checked_u64_mul(
            rank.required_blocks_per_sequence,
            concurrency,
            "candidate workload blocks",
        )?;
        let bare = checked_u64_round_up(
            checked_u64_add(
                rank.fixed_overhead_blocks,
                workload,
                "candidate bare blocks",
            )?,
            policy.alignment_blocks,
            "candidate bare alignment",
        )?
        .max(floor);
        let optional_base = checked_u64_add(workload, policy.prefix_blocks, "candidate prefix")?;
        let optional_headroom = checked_u64_ceil_div(
            checked_u64_mul(optional_base, policy.headroom_percent, "candidate headroom")?,
            100,
            "candidate headroom",
        )?;
        let full = checked_u64_round_up(
            checked_u64_add(
                checked_u64_add(rank.fixed_overhead_blocks, optional_base, "candidate base")?,
                optional_headroom,
                "candidate full blocks",
            )?,
            policy.alignment_blocks,
            "candidate full alignment",
        )?
        .max(floor)
        .min(sizing.desired_blocks);
        add_candidate(full)?;
        add_candidate(bare)?;
    }
    let candidates = candidates
        .into_iter()
        .rev()
        .map(
            |(block_count, effective_target_concurrency)| ProvisionalKvCandidate {
                block_count,
                bytes_per_block: rank.pool_bytes_per_block,
                effective_target_concurrency,
            },
        )
        .collect::<Vec<_>>();
    if candidates.is_empty()
        || candidates[0].block_count != sizing.desired_blocks
        || candidates
            .last()
            .is_none_or(|candidate| candidate.effective_target_concurrency == 0)
    {
        return Err("managed vLLM planner produced no valid exact grant candidates".to_string());
    }
    Ok(candidates)
}

fn checked_u64_add(left: u64, right: u64, context: &str) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{context} overflowed uint64"))
}

fn checked_u64_mul(left: u64, right: u64, context: &str) -> Result<u64, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{context} overflowed uint64"))
}

fn checked_u64_ceil_div(value: u64, divisor: u64, context: &str) -> Result<u64, String> {
    if divisor == 0 {
        return Err(format!("{context} has a zero divisor"));
    }
    (value / divisor)
        .checked_add(u64::from(!value.is_multiple_of(divisor)))
        .ok_or_else(|| format!("{context} overflowed uint64"))
}

fn checked_u64_round_up(value: u64, alignment: u64, context: &str) -> Result<u64, String> {
    checked_u64_mul(
        checked_u64_ceil_div(value, alignment, context)?,
        alignment,
        context,
    )
}

fn managed_vllm_geometry_digest(geometry: &serde_json::Value) -> Result<String, String> {
    let payload = serde_json::json!({
        "schema_version": MANAGED_VLLM_PLANNER_SCHEMA_VERSION,
        "geometry": geometry,
    });
    let mut canonical = String::new();
    write_canonical_json(&payload, &mut canonical)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn write_canonical_json(value: &serde_json::Value, output: &mut String) -> Result<(), String> {
    match value {
        serde_json::Value::Null => output.push_str("null"),
        serde_json::Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        serde_json::Value::Number(value) => output.push_str(&value.to_string()),
        serde_json::Value::String(value) => output.push_str(
            &serde_json::to_string(value)
                .map_err(|error| format!("canonicalize planner string: {error}"))?,
        ),
        serde_json::Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(']');
        }
        serde_json::Value::Object(values) => {
            output.push('{');
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(key)
                        .map_err(|error| format!("canonicalize planner key: {error}"))?,
                );
                output.push(':');
                write_canonical_json(value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn run_managed_vllm_planner(
    python: &Path,
    invocation: &ManagedVllmPlannerInvocation,
) -> Result<ManagedVllmExactPlanTemplate, String> {
    let expected_policy = planner_policy_for_invocation(
        &invocation.policy,
        invocation.resolved_target_concurrency,
        invocation.live_resize.as_ref(),
    )?;
    let output_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&invocation.output_path)
        .map_err(|error| {
            format!(
                "create managed vLLM planner output {}: {error}",
                invocation.output_path.display()
            )
        })?;
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&invocation.log_path)
        .map_err(|error| {
            format!(
                "open managed vLLM planner log {}: {error}",
                invocation.log_path.display()
            )
        })?;
    let mut command = Command::new(python);
    command
        .arg("-m")
        .arg("kapsl_vllm_connector.plan")
        .args(["--model", invocation.model_root.to_string_lossy().as_ref()])
        .args(["--model-fingerprint", &invocation.model_fingerprint])
        .args(["--max-model-len", &invocation.max_model_len.to_string()])
        .args([
            "--tensor-parallel-size",
            &invocation.tensor_parallel_size.to_string(),
        ])
        .args(["--attention-backend", "FLASH_ATTN"])
        .args([
            "--devices",
            &invocation
                .device_ids
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
        ])
        .args([
            "--target-concurrency",
            &expected_policy.target_concurrency.to_string(),
        ])
        .args([
            "--headroom-percent",
            &expected_policy.headroom_percent.to_string(),
        ])
        .args([
            "--prefix-blocks",
            &expected_policy.prefix_blocks.to_string(),
        ])
        .args([
            "--alignment-blocks",
            &expected_policy.alignment_blocks.to_string(),
        ])
        .env("CUDA_VISIBLE_DEVICES", &invocation.cuda_visible_devices)
        .env("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        .stdout(Stdio::from(output_file))
        .stderr(Stdio::from(log_file));
    if let Some(bytes) = expected_policy.min_bytes {
        command.args(["--min-bytes", &bytes.to_string()]);
    }
    if let Some(bytes) = expected_policy.max_bytes {
        command.args(["--max-bytes", &bytes.to_string()]);
    }
    if let Some(maximum_concurrency) = expected_policy.maximum_concurrency {
        command.args(["--maximum-concurrency", &maximum_concurrency.to_string()]);
    }
    if invocation.live_resize.is_some() {
        command.arg("--live-resize");
    }
    if expected_policy.strict_concurrency {
        command.arg("--strict-concurrency");
    }
    #[cfg(unix)]
    command.process_group(0);

    let mut child = command.spawn().map_err(|error| {
        format!(
            "start managed vLLM geometry planner with {}: {error}",
            python.display()
        )
    })?;
    #[cfg(unix)]
    let process_group = i32::try_from(child.id()).unwrap_or(i32::MAX);
    let deadline = Instant::now() + invocation.timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Ok(None) => {
                stop_managed_vllm_planner(&mut child);
                return Err(format!(
                    "managed vLLM geometry planner exceeded {}s; log: {}",
                    invocation.timeout.as_secs(),
                    invocation.log_path.display()
                ));
            }
            Err(error) => {
                stop_managed_vllm_planner(&mut child);
                return Err(format!("inspect managed vLLM geometry planner: {error}"));
            }
        }
    };
    if !status.success() {
        return Err(format!(
            "managed vLLM geometry planner exited with {status}; log: {}",
            invocation.log_path.display()
        ));
    }
    #[cfg(unix)]
    {
        let descendants_deadline = Instant::now() + Duration::from_secs(3);
        while process_group_alive(process_group) && Instant::now() < descendants_deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        if process_group_alive(process_group) {
            stop_managed_vllm_planner(&mut child);
            return Err(
                "managed vLLM planner parent exited but its process group remained alive; refusing to start the serving generation"
                    .to_string(),
            );
        }
    }
    let output = std::fs::read_to_string(&invocation.output_path).map_err(|error| {
        format!(
            "read managed vLLM planner output {}: {error}",
            invocation.output_path.display()
        )
    })?;
    let template = parse_managed_vllm_planner_output(&output, invocation)?;
    log::info!(
        "[vllm-memory] certified plan model={} desired_per_rank={} minimum_per_rank={} target_concurrency={} candidates={} geometry={} output={}",
        invocation.model_fingerprint,
        template.requested_bytes_per_rank,
        template.minimum_bytes_per_rank,
        template.target_concurrency,
        template.grant_request.candidates.len(),
        template.grant_request.geometry_digest,
        invocation.output_path.display(),
    );
    Ok(template)
}

fn stop_managed_vllm_planner(child: &mut Child) {
    #[cfg(unix)]
    {
        let process_group = i32::try_from(child.id()).unwrap_or(i32::MAX);
        unsafe {
            libc::kill(-process_group, libc::SIGTERM);
        }
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            let _ = child.try_wait();
            if !process_group_alive(process_group) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        unsafe {
            libc::kill(-process_group, libc::SIGKILL);
        }
    }
    #[cfg(not(unix))]
    let _ = child.kill();
    let _ = child.wait();
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

const MANAGED_VLLM_RESERVATION_STATES: &[&str] =
    &["planned", "reserved", "active", "released", "rejected"];

struct ManagedVllmReservationMetricState {
    state: &'static str,
    reserved_at: Option<std::time::Instant>,
}

struct ManagedVllmTelemetry {
    metrics: kapsl_monitor::metrics::KapslMetrics,
    model: String,
    replica: String,
    devices: Vec<String>,
    reservation: Mutex<ManagedVllmReservationMetricState>,
    effective_target_concurrency: AtomicU64,
}

impl ManagedVllmTelemetry {
    fn new(
        metrics: kapsl_monitor::metrics::KapslMetrics,
        model: String,
        replica_id: u32,
        device_ids: &[usize],
        initial_target_concurrency: usize,
    ) -> Arc<Self> {
        let telemetry = Arc::new(Self {
            metrics,
            model,
            replica: replica_id.to_string(),
            devices: device_ids.iter().map(ToString::to_string).collect(),
            reservation: Mutex::new(ManagedVllmReservationMetricState {
                state: "planned",
                reserved_at: None,
            }),
            effective_target_concurrency: AtomicU64::new(
                u64::try_from(initial_target_concurrency.max(1)).unwrap_or(u64::MAX),
            ),
        });
        telemetry.set_reservation_state("planned", None);
        telemetry
    }

    fn replica_labels(&self) -> [&str; 2] {
        [&self.model, &self.replica]
    }

    fn set_reservation_state(&self, state: &'static str, reserved_at: Option<std::time::Instant>) {
        debug_assert!(MANAGED_VLLM_RESERVATION_STATES.contains(&state));
        for &candidate in MANAGED_VLLM_RESERVATION_STATES {
            self.metrics
                .managed_vllm
                .provisional_reservation_state
                .with_label_values(&[self.model.as_str(), self.replica.as_str(), candidate])
                .set(i64::from(candidate == state));
        }
        let mut current = self.reservation.lock();
        current.state = state;
        current.reserved_at = reserved_at;
        drop(current);
        self.refresh_reservation_age();
    }

    fn refresh_reservation_age(&self) {
        let age = self
            .reservation
            .lock()
            .reserved_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64());
        self.metrics
            .managed_vllm
            .provisional_reservation_age_seconds
            .with_label_values(&self.replica_labels())
            .set(age);
    }

    fn observe_grant(&self, plan: &ManagedVllmExactPlanTemplate, grant: &ProvisionalKvGrant) {
        for device in &self.devices {
            let labels = &[self.model.as_str(), self.replica.as_str(), device.as_str()];
            self.metrics
                .managed_vllm
                .kv_requested_bytes
                .with_label_values(labels)
                .set(metric_i64(plan.requested_bytes_per_rank));
            self.metrics
                .managed_vllm
                .kv_granted_bytes
                .with_label_values(labels)
                .set(metric_i64(
                    grant
                        .selected_candidate
                        .block_count
                        .saturating_mul(grant.selected_candidate.bytes_per_block),
                ));
            self.metrics
                .managed_vllm
                .kv_minimum_bytes
                .with_label_values(labels)
                .set(metric_i64(plan.minimum_bytes_per_rank));
        }
        self.metrics
            .managed_vllm
            .effective_target_concurrency
            .with_label_values(&self.replica_labels())
            .set(metric_i64(
                grant.selected_candidate.effective_target_concurrency,
            ));
        self.effective_target_concurrency.store(
            grant.selected_candidate.effective_target_concurrency,
            Ordering::Release,
        );
        if grant.selected_candidate_index != 0 {
            self.metrics
                .managed_vllm
                .planning_reductions_total
                .with_label_values(&[self.model.as_str(), self.replica.as_str(), "authority_cap"])
                .inc();
        }
        if grant.selected_candidate.effective_target_concurrency < plan.target_concurrency {
            self.metrics
                .managed_vllm
                .planning_reductions_total
                .with_label_values(&[self.model.as_str(), self.replica.as_str(), "concurrency"])
                .inc();
        }
        self.set_reservation_state("reserved", Some(std::time::Instant::now()));
    }

    fn planning_rejected(&self, reason: &'static str) {
        self.metrics
            .managed_vllm
            .planning_rejections_total
            .with_label_values(&[self.model.as_str(), self.replica.as_str(), reason])
            .inc();
        self.set_reservation_state("rejected", None);
    }

    fn set_generation(&self, generation: u64) {
        self.metrics
            .managed_vllm
            .restart_generation
            .with_label_values(&self.replica_labels())
            .set(metric_i64(generation));
    }

    fn target_concurrency(&self) -> usize {
        metric_usize(
            self.effective_target_concurrency
                .load(Ordering::Acquire)
                .max(1),
        )
    }

    fn mark_active(&self) {
        self.set_reservation_state("active", None);
    }

    fn mark_released(&self) {
        self.set_reservation_state("released", None);
        for device in &self.devices {
            let labels = &[self.model.as_str(), self.replica.as_str(), device.as_str()];
            self.metrics
                .managed_vllm
                .kv_backing_bytes
                .with_label_values(labels)
                .set(0);
            self.metrics
                .managed_vllm
                .kv_logical_leased_bytes
                .with_label_values(labels)
                .set(0);
            self.metrics
                .managed_vllm
                .kv_blocks_total
                .with_label_values(labels)
                .set(0);
            self.metrics
                .managed_vllm
                .kv_blocks_allocated
                .with_label_values(labels)
                .set(0);
            self.metrics
                .managed_vllm
                .kv_blocks_active
                .with_label_values(labels)
                .set(0);
            self.metrics
                .managed_vllm
                .kv_blocks_idle
                .with_label_values(labels)
                .set(0);
            self.metrics
                .managed_vllm
                .kv_quarantine_bytes
                .with_label_values(labels)
                .set(0);
        }
    }

    fn refresh_live(&self, snapshots: &[ManagedVllmKvDeviceSnapshot]) {
        self.refresh_reservation_age();
        let by_device = snapshots
            .iter()
            .map(|snapshot| (snapshot.device_id.to_string(), snapshot))
            .collect::<HashMap<_, _>>();
        for device in &self.devices {
            let labels = &[self.model.as_str(), self.replica.as_str(), device.as_str()];
            let snapshot = by_device.get(device).copied();
            self.metrics
                .managed_vllm
                .kv_backing_bytes
                .with_label_values(labels)
                .set(metric_i64(snapshot.map_or(0, |row| row.backing_bytes)));
            self.metrics
                .managed_vllm
                .kv_logical_leased_bytes
                .with_label_values(labels)
                .set(metric_i64(
                    snapshot.map_or(0, |row| row.logical_leased_bytes),
                ));
            self.metrics
                .managed_vllm
                .kv_blocks_total
                .with_label_values(labels)
                .set(metric_i64(snapshot.map_or(0, |row| row.total_blocks)));
            self.metrics
                .managed_vllm
                .kv_blocks_allocated
                .with_label_values(labels)
                .set(metric_i64(snapshot.map_or(0, |row| row.allocated_blocks)));
            self.metrics
                .managed_vllm
                .kv_blocks_active
                .with_label_values(labels)
                .set(metric_i64(snapshot.map_or(0, |row| row.active_blocks)));
            self.metrics
                .managed_vllm
                .kv_blocks_idle
                .with_label_values(labels)
                .set(metric_i64(snapshot.map_or(0, |row| row.idle_blocks)));
            self.metrics
                .managed_vllm
                .kv_quarantine_bytes
                .with_label_values(labels)
                .set(metric_i64(snapshot.map_or(0, |row| row.quarantine_bytes)));
        }
    }
}

fn metric_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn metric_usize(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

#[derive(Clone, Debug)]
struct ManagedVllmProcessSpec {
    python: PathBuf,
    model_root: PathBuf,
    served_model_name: String,
    endpoint: String,
    port: u16,
    log_path: PathBuf,
    settings: ManagedVllmSettings,
    tensor_parallel_size: usize,
    cuda_visible_devices: String,
    planner_invocation: Option<ManagedVllmPlannerInvocation>,
}

#[derive(Clone, Debug)]
enum ManagedVllmMemoryArgument {
    LegacyFraction(f64),
    ExactBytes(u64),
}

#[derive(Clone, Debug)]
struct ManagedVllmLaunchSpec {
    kv_transfer_config: String,
    memory_argument: ManagedVllmMemoryArgument,
}

struct ManagedVllmProcess {
    spec: ManagedVllmProcessSpec,
    launch: Mutex<ManagedVllmLaunchSpec>,
    exact_plan: Mutex<Option<ManagedVllmExactPlanTemplate>>,
    launch_preparation: tokio::sync::Mutex<()>,
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
    initial_kv_blocks: AtomicU64,
    telemetry: Option<Arc<ManagedVllmTelemetry>>,
}

impl ManagedVllmProcess {
    fn new(
        spec: ManagedVllmProcessSpec,
        launch: ManagedVllmLaunchSpec,
        telemetry: Option<Arc<ManagedVllmTelemetry>>,
    ) -> Self {
        let bridge = match telemetry.as_ref() {
            Some(telemetry) => ManagedVllmHttpBridge::new_with_telemetry(
                &spec.endpoint,
                ManagedVllmBridgeTelemetry::new(
                    telemetry.metrics.clone(),
                    telemetry.model.clone(),
                    telemetry.replica.clone(),
                ),
            ),
            None => ManagedVllmHttpBridge::new(&spec.endpoint),
        }
        .expect("managed vLLM process specs always use a private HTTP origin");
        Self {
            spec,
            launch: Mutex::new(launch),
            exact_plan: Mutex::new(None),
            launch_preparation: tokio::sync::Mutex::new(()),
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
            initial_kv_blocks: AtomicU64::new(0),
            telemetry,
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

        let launch = self.launch.lock().clone();
        if launch.kv_transfer_config.trim().is_empty() {
            return Err(EngineError::backend(
                "managed vLLM launch has no provisioned KV transfer configuration".to_string(),
            ));
        }
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
                "--max-model-len",
                &self.spec.settings.max_model_len.to_string(),
            ])
            .arg("--enforce-eager")
            .arg("--enable-prefix-caching")
            .args(["--kv-transfer-config", &launch.kv_transfer_config])
            .env("CUDA_VISIBLE_DEVICES", &self.spec.cuda_visible_devices)
            .env("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(stderr));
        match launch.memory_argument {
            ManagedVllmMemoryArgument::LegacyFraction(utilization) => {
                command.args(["--gpu-memory-utilization", &utilization.to_string()]);
            }
            ManagedVllmMemoryArgument::ExactBytes(bytes) => {
                if bytes == 0 {
                    return Err(EngineError::backend(
                        "managed vLLM exact launch has a zero-byte KV grant".to_string(),
                    ));
                }
                command.args(["--kv-cache-memory-bytes", &bytes.to_string()]);
            }
        }
        if self.spec.settings.live_resize.is_some() {
            // Pinned vLLM enables overlapping async batches by default. Safe
            // tail unmapping is certified only after the sole in-flight model
            // batch has settled on every worker.
            command.arg("--no-async-scheduling");
        }
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

    fn install_exact_launch(&self, kv_transfer_config: String, bytes: u64, initial_kv_blocks: u64) {
        *self.launch.lock() = ManagedVllmLaunchSpec {
            kv_transfer_config,
            memory_argument: ManagedVllmMemoryArgument::ExactBytes(bytes),
        };
        self.initial_kv_blocks
            .store(initial_kv_blocks, Ordering::Release);
    }

    async fn prepare_exact_launch(
        &self,
        deployment: &Arc<ManagedVllmDeployment>,
        participant_base: &str,
    ) -> Result<(), EngineError> {
        let Some(invocation) = self.spec.planner_invocation.clone() else {
            return Ok(());
        };
        let _preparation = self.launch_preparation.lock().await;
        if self.shutdown.load(Ordering::Acquire) {
            return Err(EngineError::backend(
                "managed vLLM was shut down before exact KV preparation".to_string(),
            ));
        }
        let coordinator = deployment.coordinator().map_err(EngineError::backend)?;
        let cached_template = { self.exact_plan.lock().clone() };
        let template = if let Some(template) = cached_template {
            template
        } else {
            let python = self.spec.python.clone();
            let invocation_for_task = invocation.clone();
            let planned = tokio::task::spawn_blocking(move || {
                run_managed_vllm_planner(&python, &invocation_for_task)
            })
            .await
            .map_err(|error| {
                EngineError::backend(format!(
                    "managed vLLM planner task failed before completion: {error}"
                ))
            })
            .and_then(|result| result.map_err(EngineError::backend));
            let mut template = match planned {
                Ok(template) => template,
                Err(error) => {
                    if let Some(telemetry) = &self.telemetry {
                        telemetry.planning_rejected("planner");
                    }
                    return Err(error);
                }
            };
            if template.live_resize {
                let alignment_coordinator = coordinator.clone();
                let domains = template.grant_request.memory_domains.clone();
                let bytes_per_block = template
                    .grant_request
                    .candidates
                    .first()
                    .map(|candidate| candidate.bytes_per_block)
                    .ok_or_else(|| {
                        EngineError::backend(
                            "managed vLLM elastic planner returned no grant candidates",
                        )
                    })?;
                let alignment = tokio::task::spawn_blocking(move || {
                    alignment_coordinator
                        .managed_vllm_live_resize_alignment_blocks(&domains, bytes_per_block)
                })
                .await
                .map_err(|error| {
                    EngineError::backend(format!(
                        "managed vLLM CUDA VMM alignment task failed: {error}"
                    ))
                })?
                .map_err(EngineError::backend)?;
                align_managed_vllm_elastic_plan(&mut template, alignment)
                    .map_err(EngineError::backend)?;
            }
            *self.exact_plan.lock() = Some(template.clone());
            template
        };
        let grant_request = template.grant_request.clone();
        let grant_coordinator = coordinator.clone();
        let grant = tokio::task::spawn_blocking(move || {
            grant_coordinator.reserve_provisional_kv_grant(&grant_request)
        })
        .await
        .map_err(|error| {
            EngineError::backend(format!(
                "managed vLLM KV grant task failed before completion: {error}"
            ))
        })
        .and_then(|result| {
            result.map_err(|error| EngineError::backend(format!("managed vLLM KV grant: {error}")))
        });
        let grant = match grant {
            Ok(grant) => grant,
            Err(error) => {
                if let Some(telemetry) = &self.telemetry {
                    telemetry.planning_rejected("authority");
                }
                return Err(error);
            }
        };
        if let Some(telemetry) = &self.telemetry {
            telemetry.observe_grant(&template, &grant);
        }
        let granted_bytes = match grant
            .selected_candidate
            .block_count
            .checked_mul(grant.selected_candidate.bytes_per_block)
        {
            Some(bytes) => bytes,
            None => {
                coordinator.cancel_provisional_kv_grants(participant_base);
                return Err(EngineError::backend(
                    "managed vLLM KV grant bytes overflowed",
                ));
            }
        };
        let config = match build_kv_transfer_config(
            &deployment.control_endpoint,
            participant_base,
            &invocation.model_fingerprint,
            &invocation.device_ids,
            deployment.lease_ttl_ms,
            Some(&grant.proof),
            template.live_resize,
        ) {
            Ok(config) => config,
            Err(error) => {
                coordinator.cancel_provisional_kv_grants(participant_base);
                return Err(EngineError::backend(error));
            }
        };
        let launch_bytes = if template.live_resize {
            template.maximum_bytes_per_rank
        } else {
            granted_bytes
        };
        self.install_exact_launch(config, launch_bytes, grant.selected_candidate.block_count);
        let available = grant
            .authority_snapshot
            .domains
            .iter()
            .filter_map(|domain| {
                invocation.device_ids.iter().find_map(|device_id| {
                    (domain.domain
                        == super::memory::MemoryDomain::Cuda {
                            device_id: *device_id,
                        })
                    .then_some(format!("{}:{}", device_id, domain.available_bytes))
                })
            })
            .collect::<Vec<_>>()
            .join(",");
        log::info!(
            "[vllm-memory] grant model={} desired_per_rank={} granted_per_rank={} virtual_maximum_per_rank={} minimum_per_rank={} target_concurrency={} effective_concurrency={} candidate={} generation={} live_resize={} authority_available=[{}]",
            invocation.model_fingerprint,
            template.requested_bytes_per_rank,
            granted_bytes,
            template.maximum_bytes_per_rank,
            template.minimum_bytes_per_rank,
            template.target_concurrency,
            grant.selected_candidate.effective_target_concurrency,
            grant.selected_candidate_index,
            grant.proof.authority_generation,
            template.live_resize,
            available,
        );
        Ok(())
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
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        if let Some(telemetry) = &self.telemetry {
            telemetry.set_generation(generation);
        }
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
                    if let Some(telemetry) = &self.telemetry {
                        telemetry.mark_active();
                    }
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
    if result != 0 && std::io::Error::last_os_error().raw_os_error() != Some(libc::EPERM) {
        return false;
    }

    // kill(2) reports a process group as present while its only remaining
    // members are zombies. Those processes have no address space, file
    // descriptors, or CUDA mappings, so they cannot retain an imported pool.
    // This matters in containers whose PID 1 does not promptly reap orphaned
    // vLLM multiprocessing children. Fail closed if procfs cannot prove the
    // group contains only dead members.
    #[cfg(target_os = "linux")]
    {
        linux_process_group_has_live_members(process_group).unwrap_or(true)
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

#[cfg(target_os = "linux")]
fn parse_linux_proc_stat(stat: &str) -> std::io::Result<(i32, u8)> {
    let invalid = || {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "malformed Linux /proc process stat",
        )
    };
    // The command field is parenthesized and may itself contain spaces or
    // parentheses, so fields after it must be located from the final ')'.
    let command_end = stat.rfind(')').ok_or_else(invalid)?;
    let mut fields = stat
        .get(command_end + 1..)
        .ok_or_else(invalid)?
        .split_whitespace();
    let state = fields
        .next()
        .and_then(|value| value.as_bytes().first().copied())
        .ok_or_else(invalid)?;
    let _parent_pid = fields.next().ok_or_else(invalid)?;
    let process_group = fields
        .next()
        .ok_or_else(invalid)?
        .parse::<i32>()
        .map_err(|_| invalid())?;
    Ok((process_group, state))
}

#[cfg(target_os = "linux")]
fn linux_proc_stat_is_live_group_member(stat: &str, process_group: i32) -> std::io::Result<bool> {
    let (member_group, state) = parse_linux_proc_stat(stat)?;
    Ok(member_group == process_group && !matches!(state, b'Z' | b'X' | b'x'))
}

#[cfg(target_os = "linux")]
fn linux_process_group_has_live_members(process_group: i32) -> std::io::Result<bool> {
    for entry in std::fs::read_dir("/proc")? {
        let entry = entry?;
        if entry.file_name().to_string_lossy().parse::<u32>().is_err() {
            continue;
        }
        let stat_path = entry.path().join("stat");
        let stat = match std::fs::read_to_string(&stat_path) {
            Ok(stat) => stat,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        if linux_proc_stat_is_live_group_member(&stat, process_group)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn start_managed_vllm_supervisor(
    process: Arc<ManagedVllmProcess>,
    deployment: Arc<ManagedVllmDeployment>,
    participant_id: String,
) {
    tokio::spawn(async move {
        let mut consecutive_health_failures = 0u8;
        let mut low_kv_utilization_since = None;
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
                    match supervise_managed_vllm_live_resize(
                        &process,
                        &deployment,
                        &participant_id,
                        &mut low_kv_utilization_since,
                    )
                    .await
                    {
                        Ok(()) => {
                            consecutive_health_failures = 0;
                            process.mark_routable(generation, readiness_epoch, kv_readiness_epoch);
                            continue;
                        }
                        Err(error) => {
                            log::error!(
                                "[vllm-supervisor] live KV resize policy failed for model {}: {}",
                                process.spec.served_model_name,
                                error,
                            );
                            process.mark_suspect();
                            consecutive_health_failures = MAX_CONSECUTIVE_HEALTH_FAILURES;
                        }
                    }
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
            if let Err(error) = process
                .prepare_exact_launch(&deployment, &participant_id)
                .await
            {
                log::error!(
                    "[vllm-supervisor] model {} could not obtain a fresh exact KV grant: {}",
                    process.spec.served_model_name,
                    error,
                );
                continue;
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
                Err(error) => {
                    log::error!(
                        "[vllm-supervisor] failed to restart model {}: {}",
                        process.spec.served_model_name,
                        error
                    );
                    if let Err(retire_error) =
                        deployment.retire_participants_after_backend_exit(&participant_id)
                    {
                        log::error!(
                            "[vllm-supervisor] failed to cancel the unstarted generation for model {}: {}",
                            process.spec.served_model_name,
                            retire_error,
                        );
                        break;
                    }
                }
            }
        }
    });
}

async fn supervise_managed_vllm_live_resize(
    process: &Arc<ManagedVllmProcess>,
    deployment: &Arc<ManagedVllmDeployment>,
    participant_base: &str,
    low_utilization_since: &mut Option<Instant>,
) -> Result<(), String> {
    let Some(policy) = process.spec.settings.live_resize.as_ref() else {
        return Ok(());
    };
    let coordinator = deployment.coordinator()?;
    let Some(resize) = coordinator.managed_vllm_resize_snapshot(participant_base)? else {
        return Ok(());
    };
    if let Some(failure) = resize.failure {
        return Err(format!("coordinator fenced the pending resize: {failure}"));
    }
    if resize.pending_generation.is_some() {
        *low_utilization_since = None;
        return Ok(());
    }
    let devices = coordinator.managed_vllm_kv_snapshot(participant_base)?;
    if devices.is_empty() || devices.iter().any(|device| device.allocated_blocks == 0) {
        return Ok(());
    }
    let utilization_percent = devices
        .iter()
        .map(|device| {
            device
                .active_blocks
                .saturating_mul(100)
                .div_ceil(device.allocated_blocks)
        })
        .max()
        .unwrap_or_default();
    let template = process
        .exact_plan
        .lock()
        .clone()
        .ok_or_else(|| "live KV resize has no certified exact plan".to_string())?;
    let step = template
        .sequence_blocks
        .checked_mul(template.target_concurrency)
        .ok_or_else(|| "live KV resize step overflowed".to_string())?
        .max(resize.resize_alignment_blocks);
    let aligned_up = |value: u64| {
        value
            .checked_add(resize.resize_alignment_blocks - 1)
            .map(|value| value / resize.resize_alignment_blocks * resize.resize_alignment_blocks)
            .ok_or_else(|| "live KV resize target overflowed".to_string())
    };

    let target = if utilization_percent >= policy.grow_utilization_percent as u64
        && resize.current_block_count < resize.maximum_block_count
    {
        *low_utilization_since = None;
        Some(
            aligned_up(resize.current_block_count.saturating_add(step))?
                .min(resize.maximum_block_count),
        )
    } else if utilization_percent <= policy.shrink_utilization_percent as u64 {
        let since = low_utilization_since.get_or_insert_with(Instant::now);
        let floor = resize.minimum_block_count;
        if since.elapsed() >= policy.shrink_idle && resize.current_block_count > floor {
            let raw = resize.current_block_count.saturating_sub(step).max(floor);
            let aligned = raw / resize.resize_alignment_blocks * resize.resize_alignment_blocks;
            *low_utilization_since = None;
            Some(aligned.max(floor))
        } else {
            None
        }
    } else {
        *low_utilization_since = None;
        None
    };
    let Some(target) = target.filter(|target| *target != resize.current_block_count) else {
        return Ok(());
    };
    let resize_coordinator = coordinator.clone();
    let participant_base = participant_base.to_string();
    let requested = tokio::task::spawn_blocking(move || {
        resize_coordinator.request_managed_vllm_resize(&participant_base, target)
    })
    .await
    .map_err(|error| format!("live KV resize task failed: {error}"))?;
    let generation = match requested {
        Ok(generation) => generation,
        Err(error) if error.contains("resize was not admitted") => {
            log::debug!(
                "[vllm-supervisor] deferred live KV resize model={} target_blocks={}: {}",
                process.spec.served_model_name,
                target,
                error,
            );
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    log::info!(
        "[vllm-supervisor] requested live KV resize model={} generation={} target_blocks={} utilization_percent={}",
        process.spec.served_model_name,
        generation,
        target,
        utilization_percent,
    );
    Ok(())
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
                if let Some(telemetry) = &self.process.telemetry {
                    telemetry.mark_released();
                }
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
        resolved_target_concurrency: usize,
        metrics: kapsl_monitor::metrics::KapslMetrics,
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
        if matches!(
            settings.kv_cache_policy,
            ManagedVllmKvCachePolicy::LegacyFraction { .. }
        ) {
            log::warn!(
                "managed vLLM legacy_fraction memory sizing is deprecated; use kv_cache.mode: auto. Legacy fractions remain available only for compatibility and rollback"
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
        let cuda_visible_devices = child_cuda_visibility(&device_ids)?;
        let (launch, planner_invocation) = match &settings.kv_cache_policy {
            ManagedVllmKvCachePolicy::LegacyFraction {
                gpu_memory_utilization,
            } => (
                ManagedVllmLaunchSpec {
                    kv_transfer_config: build_kv_transfer_config(
                        &deployment.control_endpoint,
                        &participant_id,
                        &fingerprint,
                        &device_ids,
                        deployment.lease_ttl_ms,
                        None,
                        false,
                    )?,
                    memory_argument: ManagedVllmMemoryArgument::LegacyFraction(
                        *gpu_memory_utilization,
                    ),
                },
                None,
            ),
            ManagedVllmKvCachePolicy::Auto { .. } | ManagedVllmKvCachePolicy::Fixed { .. } => (
                ManagedVllmLaunchSpec {
                    kv_transfer_config: String::new(),
                    memory_argument: ManagedVllmMemoryArgument::ExactBytes(0),
                },
                Some(ManagedVllmPlannerInvocation {
                    model_root: model_root.clone(),
                    model_fingerprint: fingerprint.clone(),
                    participant_base: participant_id.clone(),
                    device_ids: device_ids.clone(),
                    tensor_parallel_size,
                    max_model_len: settings.max_model_len,
                    resolved_target_concurrency,
                    policy: settings.kv_cache_policy.clone(),
                    live_resize: settings.live_resize.clone(),
                    timeout: settings.startup_timeout,
                    cuda_visible_devices: cuda_visible_devices.clone(),
                    output_path: model_root_log.join("kv-plan.json"),
                    log_path: model_root_log.join("kv-planner.log"),
                }),
            ),
        };
        let telemetry = ManagedVllmTelemetry::new(
            metrics,
            manifest.project_name.clone(),
            replica_id,
            &device_ids,
            resolved_target_concurrency,
        );
        let process = Arc::new(ManagedVllmProcess::new(
            ManagedVllmProcessSpec {
                python: deployment.python.clone(),
                model_root: model_root.clone(),
                served_model_name: manifest.project_name.clone(),
                endpoint,
                port,
                log_path,
                settings,
                tensor_parallel_size,
                cuda_visible_devices,
                planner_invocation,
            },
            launch,
            Some(telemetry),
        ));
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
        self.process
            .prepare_exact_launch(&self.deployment, &self.runtime.participant_id)
            .await?;
        if let Err(error) = self.process.spawn_child() {
            self.shutdown_managed_backend();
            return Err(error);
        }
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
        let mut policy = BatchingPolicy::delegated();
        policy.max_requests = self
            .process
            .spec
            .settings
            .live_resize
            .as_ref()
            .map(|settings| settings.maximum_concurrency)
            .or_else(|| {
                self.process
                    .telemetry
                    .as_ref()
                    .map(|telemetry| telemetry.target_concurrency())
            })
            .unwrap_or(1);
        policy.with_priority_support()
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
        let kv_snapshots = self
            .deployment
            .coordinator()
            .and_then(|coordinator| {
                coordinator.managed_vllm_kv_snapshot(&self.runtime.participant_id)
            })
            .unwrap_or_default();
        if let Some(telemetry) = &self.process.telemetry {
            telemetry.refresh_live(&kv_snapshots);
        }
        let kv_cache_bytes_capacity = kv_snapshots
            .iter()
            .fold(0_u64, |total, row| total.saturating_add(row.backing_bytes));
        let kv_cache_bytes_used = kv_snapshots.iter().fold(0_u64, |total, row| {
            total.saturating_add(row.logical_leased_bytes)
        });
        // Tensor-parallel ranks share one logical native block table, so block
        // occupancy is the most constrained rank rather than a sum across
        // physical devices. Physical byte metrics above remain device-summed.
        let kv_cache_blocks_total = kv_snapshots
            .iter()
            .map(|row| row.total_blocks)
            .min()
            .unwrap_or(0);
        let kv_cache_blocks_free = kv_snapshots
            .iter()
            .map(|row| row.idle_blocks)
            .min()
            .unwrap_or(0);
        let kv_cache_sequences = kv_snapshots
            .iter()
            .map(|row| row.active_sequences)
            .max()
            .unwrap_or(0);
        let planned_memory = self
            .actual_memory()
            .allocations
            .iter()
            .map(|allocation| allocation.bytes)
            .fold(0_usize, usize::saturating_add);
        EngineMetrics {
            memory_usage: planned_memory.saturating_add(metric_usize(kv_cache_bytes_capacity)),
            batch_size: self
                .process
                .spec
                .settings
                .live_resize
                .as_ref()
                .map(|settings| settings.maximum_concurrency)
                .or_else(|| {
                    self.process
                        .telemetry
                        .as_ref()
                        .map(|telemetry| telemetry.target_concurrency())
                })
                .unwrap_or(1),
            error_rate: if requests == 0 {
                0.0
            } else {
                errors as f64 / requests as f64
            },
            kv_cache_bytes_used: metric_usize(kv_cache_bytes_used),
            kv_cache_bytes_capacity: metric_usize(kv_cache_bytes_capacity),
            kv_cache_blocks_total: metric_usize(kv_cache_blocks_total),
            kv_cache_blocks_free: metric_usize(kv_cache_blocks_free),
            kv_cache_sequences: metric_usize(kv_cache_sequences),
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
    provisioning_grant: Option<&kapsl_kv_abi::KvProvisioningGrant>,
    live_resize: bool,
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
    let mut extra = serde_json::json!({
        "kapsl_control_endpoint": control_endpoint,
        "kapsl_participant_id": participant_id,
        "kapsl_model_fingerprint": model_fingerprint,
        "kapsl_kv_mode": "shared_pool",
        "kapsl_memory_domains": memory_domains,
        "kapsl_rank_device_map": rank_device_map,
        "kapsl_lease_ttl_ms": lease_ttl_ms,
        "kapsl_live_resize": live_resize,
    });
    if let Some(grant) = provisioning_grant {
        extra["kapsl_provisioning_grant"] = serde_json::to_value(grant)
            .map_err(|error| format!("serialize managed vLLM provisioning grant: {error}"))?;
    }
    serde_json::to_string(&serde_json::json!({
        "kv_connector": "KapslConnectorV1",
        "kv_role": "kv_both",
        "kv_connector_module_path": "kapsl_vllm_connector",
        "kv_connector_extra_config": extra,
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

    // Frozen output emitted by kapsl_vllm_connector.planning 0.7.0. Keeping
    // one exact cross-language fixture catches JSON/digest/arithmetic drift
    // before the executor-backed planner is exercised on a GPU host.
    const SYNTHETIC_PLANNER_OUTPUT: &str = r#"{"geometry":{"attention_backend":"FLASH_ATTN","identity":{"adapter_id":"kapsl-vllm-connector","adapter_version":"0.7.0","backend_version":"0.26.1rc1.dev1130+g2ec6f0d71","layout_version":1,"profile_id":"vllm-v1-packed-cuda-ipc/flash-attn"},"layout_id":"vllm-v1-packed","max_model_len":1024,"model_fingerprint":"sha256:model","ranks":[{"cache_groups":[{"block_size_tokens":16,"bytes_per_group_block":1024,"element_type":{"bits":16,"bytes":2,"name":"float16"},"group_id":"vllm.group.0","key_head_dim":4,"kv_heads":4,"layers":["layer.0"],"policy":{"kind":"full_attention"},"required_blocks_per_sequence":64,"value_head_dim":4}],"device_id":0,"fixed_overhead_blocks":1,"pool_bytes_per_block":1024,"rank":0,"required_blocks_per_sequence":64}],"tensor_parallel_size":1,"total_pool_bytes_per_block":1024},"geometry_digest":"sha256:0221ae33b2c0585d8b3d97c875bbd9e8ceb6a8b7df5f4accd4ff435b6708b697","policy":{"alignment_blocks":1,"headroom_percent":20,"prefix_blocks":0,"strict_concurrency":false,"target_concurrency":4},"schema_version":1,"sizing":{"ranks":[{"base_blocks":257,"bytes_per_block":1024,"concurrency_reduced":false,"desired_blocks":309,"desired_bytes":316416,"device_id":0,"effective_target_concurrency":4,"headroom_blocks":52,"maximum_blocks":309,"maximum_bytes":316416,"minimum_blocks":65,"minimum_bytes":66560,"rank":0,"sequence_blocks":64}],"total_desired_bytes":316416,"total_maximum_bytes":316416},"status":"planned","supported":true}"#;

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_proc_stat_parser_distinguishes_live_and_zombie_group_members() {
        let live = "101 (worker name (rank 0)) S 10 77 77 0 -1";
        let zombie = "102 (worker name) Z 10 77 77 0 -1";
        let other_group = "103 (worker name) S 10 88 88 0 -1";

        assert_eq!(parse_linux_proc_stat(live).unwrap(), (77, b'S'));
        assert_eq!(parse_linux_proc_stat(zombie).unwrap(), (77, b'Z'));
        assert_ne!(parse_linux_proc_stat(other_group).unwrap().0, 77);
        assert!(linux_proc_stat_is_live_group_member(live, 77).unwrap());
        assert!(!linux_proc_stat_is_live_group_member(zombie, 77).unwrap());
        assert!(!linux_proc_stat_is_live_group_member(other_group, 77).unwrap());
        assert!(parse_linux_proc_stat("malformed").is_err());
    }

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

    fn synthetic_planner_invocation(root: &Path) -> ManagedVllmPlannerInvocation {
        ManagedVllmPlannerInvocation {
            model_root: root.to_path_buf(),
            model_fingerprint: "sha256:model".to_string(),
            participant_base: "kapsl-test-1-0".to_string(),
            device_ids: vec![0],
            tensor_parallel_size: 1,
            max_model_len: 1024,
            resolved_target_concurrency: 4,
            policy: ManagedVllmKvCachePolicy::Auto {
                target_concurrency: Some(4),
                headroom_percent: 20,
                min_bytes: None,
                max_bytes: None,
                strict: false,
            },
            live_resize: None,
            timeout: Duration::from_secs(300),
            cuda_visible_devices: "0".to_string(),
            output_path: root.join("kv-plan.json"),
            log_path: root.join("kv-planner.log"),
        }
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
        Arc::new(ManagedVllmProcess::new(
            ManagedVllmProcessSpec {
                python: PathBuf::from("python"),
                model_root: PathBuf::from("model"),
                served_model_name: "test-model".to_string(),
                endpoint: "http://127.0.0.1:12345".to_string(),
                port: 12345,
                log_path: PathBuf::from("vllm-test.log"),
                settings: ManagedVllmSettings {
                    gpu_memory_utilization: 0.25,
                    kv_cache_policy: ManagedVllmKvCachePolicy::LegacyFraction {
                        gpu_memory_utilization: 0.25,
                    },
                    live_resize: None,
                    legacy_top_level_fraction_authored: false,
                    max_model_len: 512,
                    startup_timeout: Duration::from_secs(30),
                },
                tensor_parallel_size: 1,
                cuda_visible_devices: "0".to_string(),
                planner_invocation: None,
            },
            ManagedVllmLaunchSpec {
                kv_transfer_config: "{}".to_string(),
                memory_argument: ManagedVllmMemoryArgument::LegacyFraction(0.25),
            },
            None,
        ))
    }

    #[test]
    fn certified_profile_matches_certified_tuple() {
        assert_eq!(
            certified_vllm_profile(),
            "kapsl-vllm-connector,0.7.0,0.26.1rc1.dev1130+g2ec6f0d71,vllm-v1-packed-cuda-ipc/flash-attn"
        );
        assert_eq!(
            certified_vllm_elastic_profile(),
            "kapsl-vllm-connector,0.7.0,0.26.1rc1.dev1130+g2ec6f0d71,vllm-v1-packed-cuda-vmm/flash-attn-blnhc"
        );
    }

    #[test]
    fn kv_policy_defaults_to_exact_auto_sizing() {
        let settings = settings_from_metadata("{}\n").unwrap();

        assert_eq!(
            settings.kv_cache_policy,
            ManagedVllmKvCachePolicy::Auto {
                target_concurrency: None,
                headroom_percent: 20,
                min_bytes: None,
                max_bytes: None,
                strict: false,
            }
        );
        assert!(settings.validate_launch_policy().is_ok());
    }

    #[test]
    fn memory_mode_override_is_explicit_and_never_reinterprets_authored_legacy() {
        let mut settings = ManagedVllmSettings {
            gpu_memory_utilization: 0.5,
            kv_cache_policy: ManagedVllmKvCachePolicy::Auto {
                target_concurrency: Some(8),
                headroom_percent: 20,
                min_bytes: None,
                max_bytes: None,
                strict: false,
            },
            live_resize: None,
            legacy_top_level_fraction_authored: false,
            max_model_len: 1024,
            startup_timeout: Duration::from_secs(300),
        };
        settings
            .apply_memory_mode_override(Some("legacy-fraction"))
            .unwrap();
        assert_eq!(
            settings.kv_cache_policy,
            ManagedVllmKvCachePolicy::LegacyFraction {
                gpu_memory_utilization: 0.5
            }
        );
        assert!(settings.apply_memory_mode_override(Some("exact")).is_err());
        assert!(settings
            .apply_memory_mode_override(Some("unexpected"))
            .is_err());
    }

    #[test]
    fn python_planner_fixture_has_matching_digest_arithmetic_and_grant_ladder() {
        let directory = tempfile::tempdir().unwrap();
        let invocation = synthetic_planner_invocation(directory.path());

        let template =
            parse_managed_vllm_planner_output(SYNTHETIC_PLANNER_OUTPUT, &invocation).unwrap();

        assert_eq!(template.requested_bytes_per_rank, 316_416);
        assert_eq!(template.minimum_bytes_per_rank, 66_560);
        assert_eq!(template.target_concurrency, 4);
        assert_eq!(
            template
                .grant_request
                .candidates
                .iter()
                .map(|candidate| (
                    candidate.block_count,
                    candidate.effective_target_concurrency,
                ))
                .collect::<Vec<_>>(),
            vec![
                (309, 4),
                (257, 4),
                (232, 3),
                (193, 3),
                (155, 2),
                (129, 2),
                (78, 1),
                (65, 1),
            ]
        );
        assert_eq!(template.grant_request.candidates[0].bytes_per_block, 1024);
        assert_eq!(
            template.grant_request.geometry_digest,
            "sha256:0221ae33b2c0585d8b3d97c875bbd9e8ceb6a8b7df5f4accd4ff435b6708b697"
        );
    }

    #[test]
    fn elastic_planner_fixture_separates_initial_physical_and_virtual_maximum() {
        let directory = tempfile::tempdir().unwrap();
        let mut invocation = synthetic_planner_invocation(directory.path());
        invocation.live_resize = Some(ManagedVllmLiveResizeSettings {
            maximum_concurrency: 8,
            grow_utilization_percent: 80,
            shrink_utilization_percent: 25,
            shrink_idle: Duration::from_secs(60),
        });
        let mut value: serde_json::Value = serde_json::from_str(SYNTHETIC_PLANNER_OUTPUT).unwrap();
        value["geometry"]["identity"]["profile_id"] =
            serde_json::json!(MANAGED_VLLM_ELASTIC_PROFILE_ID);
        value["geometry"]["layout_id"] = serde_json::json!("BLNHC");
        value["policy"]["maximum_concurrency"] = serde_json::json!(8);
        value["sizing"]["ranks"][0]["maximum_blocks"] = serde_json::json!(616);
        value["sizing"]["ranks"][0]["maximum_bytes"] = serde_json::json!(630_784);
        value["sizing"]["total_maximum_bytes"] = serde_json::json!(630_784);
        value["geometry_digest"] =
            serde_json::json!(managed_vllm_geometry_digest(&value["geometry"]).unwrap());

        let mut template =
            parse_managed_vllm_planner_output(&value.to_string(), &invocation).unwrap();

        assert!(template.live_resize);
        assert_eq!(template.requested_bytes_per_rank, 316_416);
        assert_eq!(template.maximum_bytes_per_rank, 630_784);
        assert_eq!(template.grant_request.maximum_block_count, Some(616));
        align_managed_vllm_elastic_plan(&mut template, 64).unwrap();
        assert_eq!(template.minimum_bytes_per_rank, 128 * 1024);
        assert_eq!(template.requested_bytes_per_rank, 320 * 1024);
        assert_eq!(template.maximum_block_count, 640);
        assert_eq!(template.maximum_bytes_per_rank, 640 * 1024);
        assert_eq!(template.grant_request.maximum_block_count, Some(640));
        assert!(template
            .grant_request
            .candidates
            .iter()
            .all(|candidate| candidate.block_count.is_multiple_of(64)));
    }

    #[test]
    fn planner_output_rejects_digest_identity_arithmetic_and_packed_stride_drift() {
        let directory = tempfile::tempdir().unwrap();
        let invocation = synthetic_planner_invocation(directory.path());
        let baseline: serde_json::Value = serde_json::from_str(SYNTHETIC_PLANNER_OUTPUT).unwrap();

        let mut bad_digest = baseline.clone();
        bad_digest["geometry_digest"] = serde_json::json!(format!("sha256:{}", "0".repeat(64)));
        assert!(
            parse_managed_vllm_planner_output(&bad_digest.to_string(), &invocation)
                .unwrap_err()
                .contains("digest")
        );

        let mut bad_identity = baseline.clone();
        bad_identity["geometry"]["identity"]["backend_version"] = serde_json::json!("other");
        let digest = managed_vllm_geometry_digest(&bad_identity["geometry"]).unwrap();
        bad_identity["geometry_digest"] = serde_json::json!(digest);
        assert!(
            parse_managed_vllm_planner_output(&bad_identity.to_string(), &invocation)
                .unwrap_err()
                .contains("identity")
        );

        let mut bad_arithmetic = baseline.clone();
        bad_arithmetic["sizing"]["ranks"][0]["desired_blocks"] = serde_json::json!(308);
        assert!(
            parse_managed_vllm_planner_output(&bad_arithmetic.to_string(), &invocation)
                .unwrap_err()
                .contains("sizing")
        );

        let mut bad_stride = baseline;
        bad_stride["geometry"]["ranks"][0]["pool_bytes_per_block"] = serde_json::json!(2048);
        bad_stride["geometry"]["total_pool_bytes_per_block"] = serde_json::json!(2048);
        bad_stride["sizing"]["ranks"][0]["bytes_per_block"] = serde_json::json!(2048);
        bad_stride["sizing"]["ranks"][0]["desired_bytes"] = serde_json::json!(632832);
        bad_stride["sizing"]["ranks"][0]["minimum_bytes"] = serde_json::json!(133120);
        bad_stride["sizing"]["ranks"][0]["maximum_bytes"] = serde_json::json!(632832);
        bad_stride["sizing"]["total_desired_bytes"] = serde_json::json!(632832);
        bad_stride["sizing"]["total_maximum_bytes"] = serde_json::json!(632832);
        let digest = managed_vllm_geometry_digest(&bad_stride["geometry"]).unwrap();
        bad_stride["geometry_digest"] = serde_json::json!(digest);
        assert!(
            parse_managed_vllm_planner_output(&bad_stride.to_string(), &invocation)
                .unwrap_err()
                .contains("packed pool")
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
        assert!(auto.validate_launch_policy().is_ok());

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
        assert!(fixed.validate_launch_policy().is_ok());

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
    fn live_resize_policy_parses_bounded_thresholds_and_requires_auto_mode() {
        let settings = settings_from_metadata(
            r#"
serving:
  vllm:
    kv_cache:
      mode: auto
      target_concurrency: 4
      live_resize:
        maximum_concurrency: 16
        grow_utilization_percent: 75
        shrink_utilization_percent: 20
        shrink_idle_seconds: 45
"#,
        )
        .unwrap();
        assert_eq!(
            settings.live_resize,
            Some(ManagedVllmLiveResizeSettings {
                maximum_concurrency: 16,
                grow_utilization_percent: 75,
                shrink_utilization_percent: 20,
                shrink_idle: Duration::from_secs(45),
            })
        );
        settings.validate_launch_policy().unwrap();

        let disabled = settings_from_metadata(
            r#"
serving:
  vllm:
    kv_cache:
      mode: auto
      live_resize:
        enabled: false
"#,
        )
        .unwrap();
        assert!(disabled.live_resize.is_none());

        for metadata in [
            r#"
serving:
  vllm:
    kv_cache:
      mode: auto
      live_resize: {}
"#,
            r#"
serving:
  vllm:
    kv_cache:
      mode: auto
      target_concurrency: 8
      live_resize:
        maximum_concurrency: 4
"#,
            r#"
serving:
  vllm:
    kv_cache:
      mode: auto
      live_resize:
        maximum_concurrency: 8
        grow_utilization_percent: 25
        shrink_utilization_percent: 25
"#,
            r#"
serving:
  vllm:
    kv_cache:
      mode: auto
      live_resize:
        enabled: false
        maximum_concurrency: 8
"#,
        ] {
            let result = settings_from_metadata(metadata)
                .and_then(|settings| settings.validate_launch_policy().map(|_| settings));
            assert!(
                result.is_err(),
                "live resize metadata should fail: {metadata}"
            );
        }
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
            None,
            false,
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
    fn exact_kv_config_carries_the_single_use_authority_proof() {
        let proof = kapsl_kv_abi::KvProvisioningGrant {
            token: "opaque-token".to_string(),
            geometry_digest: format!("sha256:{}", "a".repeat(64)),
            authority_generation: 7,
            expires_at_unix_ms: 42,
        };
        let encoded = build_kv_transfer_config(
            "unix:///tmp/kapsl.sock",
            "worker",
            "sha256:model",
            &[0],
            30_000,
            Some(&proof),
            false,
        )
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&encoded).unwrap();

        assert_eq!(
            value["kv_connector_extra_config"]["kapsl_provisioning_grant"],
            serde_json::to_value(proof).unwrap()
        );

        let live_encoded = build_kv_transfer_config(
            "unix:///tmp/kapsl.sock",
            "worker",
            "sha256:model",
            &[0],
            30_000,
            None,
            true,
        )
        .unwrap();
        let live_value: serde_json::Value = serde_json::from_str(&live_encoded).unwrap();
        assert_eq!(
            live_value["kv_connector_extra_config"]["kapsl_live_resize"],
            true
        );
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
        let process = ManagedVllmProcess::new(
            ManagedVllmProcessSpec {
                python: PathBuf::from("python"),
                model_root: model_root.clone(),
                served_model_name: "test-model".to_string(),
                endpoint: "http://127.0.0.1:12345".to_string(),
                port: 12345,
                log_path: directory.path().join("vllm.log"),
                settings: ManagedVllmSettings {
                    gpu_memory_utilization: 0.25,
                    kv_cache_policy: ManagedVllmKvCachePolicy::LegacyFraction {
                        gpu_memory_utilization: 0.25,
                    },
                    live_resize: None,
                    legacy_top_level_fraction_authored: false,
                    max_model_len: 512,
                    startup_timeout: Duration::from_secs(30),
                },
                tensor_parallel_size: 1,
                cuda_visible_devices: "0".to_string(),
                planner_invocation: None,
            },
            ManagedVllmLaunchSpec {
                kv_transfer_config: "{}".to_string(),
                memory_argument: ManagedVllmMemoryArgument::LegacyFraction(0.25),
            },
            None,
        );

        let command = process.build_command().unwrap();
        let arguments = command
            .get_args()
            .map(std::ffi::OsStr::to_os_string)
            .collect::<Vec<_>>();
        assert!(arguments
            .windows(2)
            .any(|pair| { pair[0] == "--model" && pair[1] == model_root.as_os_str() }));
    }

    #[test]
    fn exact_process_uses_the_pinned_byte_flag_and_never_the_fraction() {
        let directory = tempfile::tempdir().unwrap();
        let model_root = directory.path().join("model");
        std::fs::create_dir(&model_root).unwrap();
        let mut process = test_managed_process();
        let process = Arc::get_mut(&mut process).expect("test owns the process");
        process.spec.model_root = model_root;
        process.spec.log_path = directory.path().join("vllm.log");
        process.install_exact_launch("{\"exact\":true}".to_string(), 316_416, 309);

        let command = process.build_command().unwrap();
        let arguments = command
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--kv-cache-memory-bytes", "316416"]));
        assert!(!arguments
            .iter()
            .any(|argument| argument == "--gpu-memory-utilization"));
        assert!(arguments
            .iter()
            .any(|argument| argument == "--enable-prefix-caching"));
        assert!(!arguments
            .iter()
            .any(|argument| argument == "--no-async-scheduling"));
    }

    #[test]
    fn elastic_process_disables_overlapping_vllm_batches() {
        let directory = tempfile::tempdir().unwrap();
        let model_root = directory.path().join("model");
        std::fs::create_dir(&model_root).unwrap();
        let mut process = test_managed_process();
        let process = Arc::get_mut(&mut process).expect("test owns the process");
        process.spec.model_root = model_root;
        process.spec.log_path = directory.path().join("vllm.log");
        process.spec.settings.live_resize = Some(ManagedVllmLiveResizeSettings {
            maximum_concurrency: 8,
            grow_utilization_percent: 80,
            shrink_utilization_percent: 25,
            shrink_idle: Duration::from_secs(60),
        });
        process.install_exact_launch("{\"exact\":true}".to_string(), 316_416, 309);

        let command = process.build_command().unwrap();
        let arguments = command
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert!(arguments
            .iter()
            .any(|argument| argument == "--no-async-scheduling"));
    }

    #[cfg(all(unix, feature = "gpu-device-pool"))]
    #[tokio::test]
    async fn failed_child_spawn_releases_exact_grant_and_participant_namespace() {
        use kapsl_hal::device::{Device, DeviceBackend, DeviceInfo};

        let directory = tempfile::tempdir().unwrap();
        let model_root = directory.path().join("model");
        std::fs::create_dir(&model_root).unwrap();
        let invocation = synthetic_planner_invocation(directory.path());
        let mut template =
            parse_managed_vllm_planner_output(SYNTHETIC_PLANNER_OUTPUT, &invocation).unwrap();
        // The host test exercises lifecycle cleanup without opening CUDA. The
        // production planner binds this same claim to CUDA domains.
        template.grant_request.memory_domains =
            BTreeSet::from([kapsl_kv_abi::KvMemoryDomain::Host]);
        let authority = MemoryAuthority::new_accounting_only_for_test(&DeviceInfo {
            cpu_cores: 1,
            total_memory: 1024 * 1024 * 1024,
            os_type: "test".to_string(),
            os_release: "test".to_string(),
            has_cuda: true,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: vec![Device {
                id: 0,
                name: "synthetic-cuda".to_string(),
                backend: DeviceBackend::Cuda,
                memory_mb: 1024,
                compute_units: 1,
                pci_bus_id: None,
                partition_id: None,
                driver_version: None,
                cuda_version: None,
                compute_capability: None,
                utilization_gpu_pct: None,
                temperature_c: None,
                supports_fp16: true,
                supports_int8: true,
            }],
        })
        .unwrap();
        let coordinator =
            ExternalKvCoordinator::new(authority.clone(), Duration::from_secs(30)).unwrap();
        let port = reserve_loopback_port().unwrap();
        let deployment = Arc::new(ManagedVllmDeployment {
            python: directory.path().join("missing-python"),
            control_endpoint: format!("unix://{}", directory.path().join("kv.sock").display()),
            runtime_root: directory.path().to_path_buf(),
            lease_ttl_ms: 30_000,
            runtimes: parking_lot::Mutex::new(HashMap::new()),
            coordinator: parking_lot::RwLock::new(Some(coordinator.clone())),
        });
        let process = Arc::new(ManagedVllmProcess::new(
            ManagedVllmProcessSpec {
                python: directory.path().join("missing-python"),
                model_root,
                served_model_name: "test-model".to_string(),
                endpoint: format!("http://127.0.0.1:{port}"),
                port,
                log_path: directory.path().join("vllm.log"),
                settings: ManagedVllmSettings {
                    gpu_memory_utilization: 0.5,
                    kv_cache_policy: invocation.policy.clone(),
                    live_resize: None,
                    legacy_top_level_fraction_authored: false,
                    max_model_len: invocation.max_model_len,
                    startup_timeout: invocation.timeout,
                },
                tensor_parallel_size: 1,
                cuda_visible_devices: "0".to_string(),
                planner_invocation: Some(invocation.clone()),
            },
            ManagedVllmLaunchSpec {
                kv_transfer_config: String::new(),
                memory_argument: ManagedVllmMemoryArgument::ExactBytes(0),
            },
            None,
        ));
        *process.exact_plan.lock() = Some(template);

        process
            .prepare_exact_launch(&deployment, &invocation.participant_base)
            .await
            .unwrap();
        assert!(!authority.snapshot().rows.is_empty());
        assert!(process.spawn_child().is_err());
        assert!(process.child.lock().is_none());

        let runtime = ManagedVllmRuntime {
            process: process.clone(),
            participant_id: invocation.participant_base.clone(),
            participant_retired: parking_lot::Mutex::new(false),
        };
        assert!(runtime.shutdown(&deployment));
        assert!(authority.snapshot().rows.is_empty());
        assert_eq!(
            coordinator.cancel_provisional_kv_grants(&invocation.participant_base),
            0
        );
        assert!(!coordinator
            .managed_participant_is_active(&invocation.participant_base)
            .unwrap());
        assert!(std::net::TcpListener::bind(("127.0.0.1", port)).is_ok());
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
