use super::*;
use kapsl_engine_api::{
    EngineStream, MemoryAllocation, MemoryAllocationClass, MemoryAllocationSource, MemoryDomain,
    MemoryReport,
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
pub(crate) const MANAGED_VLLM_ADAPTER_VERSION: &str = "0.5.0";
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
import importlib.metadata as md
import json
import platform
import torch
from kapsl_vllm_connector import ADAPTER_PROFILE_ID, ADAPTER_VERSION
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
    let actual: ManagedVllmEnvironment = serde_json::from_str(stdout.trim())
        .map_err(|error| format!("environment probe emitted invalid JSON: {error}"))?;
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
    };
    if actual != expected {
        return Err(format!(
            "binary environment mismatch: {actual:?} != {expected:?}"
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ManagedVllmSettings {
    gpu_memory_utilization: f64,
    max_model_len: usize,
    startup_timeout: Duration,
}

impl ManagedVllmSettings {
    fn from_manifest(manifest: &Manifest) -> Result<Self, String> {
        let mut settings = Self {
            gpu_memory_utilization: 0.5,
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

        if let Some(raw) = value("gpu_memory_utilization") {
            settings.gpu_memory_utilization = raw.as_f64().ok_or_else(|| {
                "metadata.serving.vllm.gpu_memory_utilization must be a number".to_string()
            })?;
        }
        if !(0.1..=0.9).contains(&settings.gpu_memory_utilization) {
            return Err(
                "metadata.serving.vllm.gpu_memory_utilization must be between 0.1 and 0.9"
                    .to_string(),
            );
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
    child: Mutex<Option<Child>>,
    shutdown: AtomicBool,
    restarts: AtomicU32,
}

impl ManagedVllmProcess {
    fn new(spec: ManagedVllmProcessSpec) -> Self {
        Self {
            spec,
            child: Mutex::new(None),
            shutdown: AtomicBool::new(false),
            restarts: AtomicU32::new(0),
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
        if self.shutdown.load(Ordering::Acquire) {
            return Err(EngineError::backend(
                "managed vLLM process has been shut down".to_string(),
            ));
        }
        let child = self.build_command()?.spawn().map_err(|error| {
            EngineError::backend(format!(
                "launch managed vLLM with {}: {error}",
                self.spec.python.display()
            ))
        })?;
        log::info!(
            "Managed vLLM process started: pid={} endpoint={} log={}",
            child.id(),
            self.spec.endpoint,
            self.spec.log_path.display()
        );
        *self.child.lock() = Some(child);
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

    fn call_health(&self) -> Result<(), EngineError> {
        if let Some(status) = self.try_wait()? {
            return Err(EngineError::backend(format!(
                "managed vLLM process exited with {status}; log: {}",
                self.spec.log_path.display()
            )));
        }
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(HEALTH_TIMEOUT))
            .timeout_per_call(Some(HEALTH_TIMEOUT))
            .build()
            .into();
        agent
            .get(&format!("{}/health", self.spec.endpoint))
            .call()
            .map(|_| ())
            .map_err(|error| EngineError::backend(format!("managed vLLM health check: {error}")))
    }

    async fn wait_ready(&self) -> Result<(), EngineError> {
        let deadline = Instant::now() + self.spec.settings.startup_timeout;
        loop {
            if let Some(status) = self.try_wait()? {
                return Err(EngineError::backend(format!(
                    "managed vLLM exited before readiness with {status}; log: {}",
                    self.spec.log_path.display()
                )));
            }
            if self.call_health().is_ok() {
                return Ok(());
            }
            if Instant::now() >= deadline {
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
        self.shutdown.store(true, Ordering::Release);
        self.stop_child()
    }
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
                if process.call_health().is_ok() {
                    consecutive_health_failures = 0;
                    continue;
                }
                consecutive_health_failures = consecutive_health_failures.saturating_add(1);
                if consecutive_health_failures < MAX_CONSECUTIVE_HEALTH_FAILURES {
                    continue;
                }
                log::error!(
                    "[vllm-supervisor] endpoint {} failed {} consecutive health checks",
                    process.spec.endpoint,
                    consecutive_health_failures
                );
            }

            // A restart is a lifecycle boundary: reap the complete old process
            // group, then retire its connector registration and isolated IPC
            // pool before a new vLLM engine UUID can register. If either fence
            // fails, retaining authority is safer than starting a second
            // backend against memory that may still be imported.
            if !process.stop_child() {
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
                Ok(()) => match process.wait_ready().await {
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

    fn agent_for_request(request: &InferenceRequest) -> ureq::Agent {
        let timeout = request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.timeout_ms)
            .map(Duration::from_millis)
            .unwrap_or(DEFAULT_REQUEST_TIMEOUT);
        ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .timeout_per_call(Some(timeout))
            .build()
            .into()
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
        self.process.call_health()?;
        let (path, payload) = self.request_payload(request, false)?;
        let chat = path.contains("chat");
        let url = format!("{}{}", self.process.spec.endpoint, path);
        let mut response = Self::agent_for_request(request)
            .post(&url)
            .header("Content-Type", "application/json")
            .send(payload)
            .map_err(|error| EngineError::backend(format!("vLLM request to {url}: {error}")))?;
        let body = response
            .body_mut()
            .read_to_string()
            .map_err(|error| EngineError::backend(format!("read vLLM response: {error}")))?;
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
        if let Err(error) = self.process.wait_ready().await {
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

    fn self_batches(&self) -> bool {
        true
    }

    fn batching_policy(&self) -> BatchingPolicy {
        BatchingPolicy::delegated().with_priority_support()
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        self.requests.fetch_add(1, Ordering::Relaxed);
        if let Err(error) = self.process.call_health() {
            self.errors.fetch_add(1, Ordering::Relaxed);
            return Box::pin(stream::once(async move { Err(error) }));
        }
        let (path, payload) = match self.request_payload(request, true) {
            Ok(request) => request,
            Err(error) => {
                self.errors.fetch_add(1, Ordering::Relaxed);
                return Box::pin(stream::once(async move { Err(error) }));
            }
        };
        let url = format!("{}{}", self.process.spec.endpoint, path);
        let chat = path.contains("chat");
        let request = request.clone();
        let errors_for_thread = self.errors.clone();
        let generated_for_thread = self.generated_tokens.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        std::thread::spawn(move || {
            let mut response = match Self::agent_for_request(&request)
                .post(&url)
                .header("Content-Type", "application/json")
                .send(payload)
            {
                Ok(response) => response,
                Err(error) => {
                    errors_for_thread.fetch_add(1, Ordering::Relaxed);
                    let _ = tx.send(Err(EngineError::backend(format!(
                        "streaming vLLM request to {url}: {error}"
                    ))));
                    return;
                }
            };
            let reader = std::io::BufReader::new(response.body_mut().as_reader());
            for line in reader.lines() {
                if request
                    .cancellation
                    .as_ref()
                    .is_some_and(|token| token.is_cancelled())
                {
                    break;
                }
                let line = match line {
                    Ok(line) => line,
                    Err(error) => {
                        errors_for_thread.fetch_add(1, Ordering::Relaxed);
                        let _ = tx.send(Err(EngineError::backend(format!(
                            "read vLLM stream: {error}"
                        ))));
                        return;
                    }
                };
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" {
                    break;
                }
                let value: serde_json::Value = match serde_json::from_str(data) {
                    Ok(value) => value,
                    Err(error) => {
                        errors_for_thread.fetch_add(1, Ordering::Relaxed);
                        let _ = tx.send(Err(EngineError::backend(format!(
                            "decode vLLM stream event: {error}"
                        ))));
                        return;
                    }
                };
                let choice = value
                    .get("choices")
                    .and_then(serde_json::Value::as_array)
                    .and_then(|choices| choices.first());
                let text = if chat {
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
                generated_for_thread.fetch_add(estimate_vllm_tokens(text), Ordering::Relaxed);
                match Self::output_packet(text.to_string()) {
                    Ok(packet) => {
                        if tx.send(Ok(packet)).is_err() {
                            return;
                        }
                    }
                    Err(error) => {
                        errors_for_thread.fetch_add(1, Ordering::Relaxed);
                        let _ = tx.send(Err(error));
                        return;
                    }
                }
            }
        });

        let stream = stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });
        Box::pin(stream)
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
        self.process.call_health()
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
    let mut assets = std::fs::read_dir(model_root)
        .map_err(|error| format!("read model directory {}: {error}", model_root.display()))?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let metadata = entry.metadata().ok()?;
            metadata.is_file().then(|| {
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    metadata.len(),
                )
            })
        })
        .collect::<Vec<_>>();
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
    // PackageLoader exposes the manifest's primary model file, while vLLM
    // consumes the complete Hugging Face directory beside that file. Accept
    // either representation so preliminary admission and the real backend
    // load estimate the same weights.
    let model_root = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent().unwrap_or(model_path)
    };
    let weight_bytes = std::fs::read_dir(model_root)
        .map_err(|error| format!("read model directory {}: {error}", model_root.display()))?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.path().extension().and_then(|value| value.to_str()) == Some("safetensors")
        })
        .try_fold(0usize, |total, entry| {
            entry
                .metadata()
                .map(|metadata| total.saturating_add(metadata.len() as usize))
                .map_err(|error| format!("stat model shard {}: {error}", entry.path().display()))
        })?;
    if weight_bytes == 0 {
        return Err(format!(
            "managed vLLM model directory {} contains no .safetensors weights",
            model_root.display()
        ));
    }
    let per_device_weights = weight_bytes
        .saturating_add(device_ids.len() - 1)
        .saturating_div(device_ids.len());
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

    #[test]
    fn certified_profile_matches_certified_tuple() {
        assert_eq!(
            certified_vllm_profile(),
            "kapsl-vllm-connector,0.5.0,0.26.1rc1.dev1130+g2ec6f0d71,vllm-v1-packed-cuda-ipc/flash-attn"
        );
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
