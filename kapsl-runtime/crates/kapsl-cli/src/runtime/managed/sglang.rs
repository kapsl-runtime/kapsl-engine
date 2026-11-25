//! Managed SGLang: an opaque KV participant launched through the Kapsl adapter.
//!
//! SGLang keeps its own KV pool and allocates all of it at startup. The
//! runtime therefore bounds the pool with `--max-total-tokens`, admits the
//! server's whole static footprint (weights, workspace, and that pool) at
//! model load, and lets the adapter take a capacity lease per request. The
//! server becomes routable once it answers its health endpoint and its
//! participant has registered a pool no larger than the one admitted.

use super::server::{
    ManagedKvStatus, ManagedServerDriver, ManagedServerEngine, ManagedServerProcess,
    ManagedServerSpec,
};
use super::*;

pub(crate) const MANAGED_SGLANG_ADAPTER_ID: &str = "kapsl-sglang-adapter";
pub(crate) const MANAGED_SGLANG_ADAPTER_VERSION: &str = "0.2.0";
pub(crate) const MANAGED_SGLANG_BACKEND_VERSION: &str = "0.5.23";

/// Python executable of an operator-provided SGLang environment. There is no
/// signed SGLang backend pack, so this is the only way to select one.
const MANAGED_SGLANG_PYTHON_ENV: &str = "KAPSL_SGLANG_PYTHON";
/// Module behind the adapter's `kapsl-sglang-serve` launcher, which refuses to
/// start SGLang unless the adapter will govern it.
const MANAGED_SGLANG_LAUNCH_MODULE: &str = "kapsl_sglang_adapter.launch";
const DEFAULT_TOKENS_PER_REQUEST: u64 = 4096;
/// SGLang stores K and V in the model's 16-bit float type unless told
/// otherwise, and this launcher never tells it otherwise.
const KV_ELEMENT_BYTES: u64 = 2;
/// SGLang allocates one padding page beyond the schedulable pool; its default
/// page is one token.
const KV_PADDING_TOKENS: u64 = 1;

/// Validate the operator's SGLang environment and prepare the deployment.
pub(crate) fn prepare_managed_sglang(
    state_dir: Option<&Path>,
    requested_control_socket: Option<&Path>,
    lease_ttl_ms: u64,
) -> Result<PreparedManagedServerDeployment, String> {
    if !cfg!(all(feature = "gpu-device-pool", target_os = "linux")) {
        return Err(
            "managed SGLang requires a Linux Kapsl build with `gpu-device-pool` enabled"
                .to_string(),
        );
    }
    let python = discover_sglang_python()?;
    log::info!(
        "Managed SGLang environment: {} (SGLang {}, {} {})",
        python.display(),
        MANAGED_SGLANG_BACKEND_VERSION,
        MANAGED_SGLANG_ADAPTER_ID,
        MANAGED_SGLANG_ADAPTER_VERSION,
    );
    ManagedServerDeployment::prepare(
        ResolvedServingBackend::Sglang,
        "SGLang",
        python,
        state_dir,
        requested_control_socket,
        lease_ttl_ms,
    )
}

fn discover_sglang_python() -> Result<PathBuf, String> {
    let python = std::env::var_os(MANAGED_SGLANG_PYTHON_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            format!(
                "managed SGLang needs {MANAGED_SGLANG_PYTHON_ENV} to name the Python executable of an environment with SGLang {MANAGED_SGLANG_BACKEND_VERSION} and {MANAGED_SGLANG_ADAPTER_ID} {MANAGED_SGLANG_ADAPTER_VERSION} installed; there is no signed SGLang backend pack"
            )
        })?;
    probe_sglang_python(&python)
        .and_then(|environment| check_sglang_environment(&environment))
        .map_err(|error| {
            format!(
                "{MANAGED_SGLANG_PYTHON_ENV}={} is not a usable managed-SGLang environment: {error}",
                python.display()
            )
        })?;
    Ok(python)
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct ManagedSglangEnvironment {
    sglang: String,
    adapter_distribution: String,
    adapter: String,
    pinned_sglang: Vec<String>,
    kv_abi_major: u64,
    kv_abi_minor: u64,
    participant_config_env: String,
}

fn probe_sglang_python(python: &Path) -> Result<ManagedSglangEnvironment, String> {
    // Importing the adapter does not import SGLang or torch, so this stays a
    // metadata probe; the launcher and the scheduler hook repeat the checks
    // that need the real modules.
    const PROBE: &str = r#"
import importlib.metadata as md
import json
import kapsl_sglang_adapter as adapter
from kapsl_sglang_adapter.config import PARTICIPANT_CONFIG_ENV
print(json.dumps({
    "sglang": md.version("sglang"),
    "adapter_distribution": md.version("kapsl-sglang-adapter"),
    "adapter": adapter.ADAPTER_VERSION,
    "pinned_sglang": list(adapter.PINNED_SGLANG_VERSIONS),
    "kv_abi_major": adapter.ABI_VERSION["major"],
    "kv_abi_minor": adapter.ABI_VERSION["minor"],
    "participant_config_env": PARTICIPANT_CONFIG_ENV,
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
    serde_json::from_str(stdout.trim())
        .map_err(|error| format!("environment probe emitted invalid fields: {error}"))
}

fn check_sglang_environment(actual: &ManagedSglangEnvironment) -> Result<(), String> {
    let mut problems = Vec::new();
    if actual.sglang != MANAGED_SGLANG_BACKEND_VERSION {
        problems.push(format!(
            "SGLang {} is installed, but exactly {MANAGED_SGLANG_BACKEND_VERSION} is supported",
            actual.sglang
        ));
    }
    if actual.adapter != MANAGED_SGLANG_ADAPTER_VERSION
        || actual.adapter_distribution != MANAGED_SGLANG_ADAPTER_VERSION
    {
        problems.push(format!(
            "{MANAGED_SGLANG_ADAPTER_ID} {} (distribution {}) is installed, but exactly {MANAGED_SGLANG_ADAPTER_VERSION} is supported",
            actual.adapter, actual.adapter_distribution
        ));
    }
    if !actual
        .pinned_sglang
        .iter()
        .any(|version| version == MANAGED_SGLANG_BACKEND_VERSION)
    {
        problems.push(format!(
            "the adapter does not pin SGLang {MANAGED_SGLANG_BACKEND_VERSION}"
        ));
    }
    if (actual.kv_abi_major, actual.kv_abi_minor)
        != (MANAGED_VLLM_KV_ABI_MAJOR, MANAGED_VLLM_KV_ABI_MINOR)
    {
        problems.push(format!(
            "the adapter speaks KV ABI {}.{}, but this runtime speaks {MANAGED_VLLM_KV_ABI_MAJOR}.{MANAGED_VLLM_KV_ABI_MINOR}",
            actual.kv_abi_major, actual.kv_abi_minor
        ));
    }
    if actual.participant_config_env != kapsl_kv_abi::KV_PARTICIPANT_CONFIG_ENV {
        problems.push(format!(
            "the adapter reads its configuration from {}, but this runtime writes {}",
            actual.participant_config_env,
            kapsl_kv_abi::KV_PARTICIPANT_CONFIG_ENV
        ));
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; "))
    }
}

/// Launch settings read from `metadata.serving.sglang`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ManagedSglangSettings {
    /// Size of SGLang's KV pool in tokens (`--max-total-tokens`).
    max_total_tokens: Option<u64>,
    /// Longest context a request may use (`--context-length`).
    context_length: Option<u64>,
    /// Bytes to admit for the KV pool instead of the config.json estimate.
    kv_pool_bytes: Option<u64>,
    startup_timeout: Duration,
}

impl ManagedSglangSettings {
    fn from_manifest(manifest: &Manifest) -> Result<Self, String> {
        let mut settings = Self {
            max_total_tokens: None,
            context_length: None,
            kv_pool_bytes: None,
            startup_timeout: DEFAULT_STARTUP_TIMEOUT,
        };
        let Some(sglang) = manifest
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("serving"))
            .and_then(|serving| serving.get("sglang"))
        else {
            return Ok(settings);
        };
        let object = sglang
            .as_mapping()
            .ok_or_else(|| "metadata.serving.sglang must be an object when present".to_string())?;
        for (key, value) in object {
            let key = key
                .as_str()
                .ok_or_else(|| "metadata.serving.sglang keys must be strings".to_string())?;
            let path = format!("metadata.serving.sglang.{key}");
            match key {
                "max_total_tokens" => {
                    let tokens = positive_integer(value, &path)?;
                    if tokens > u64::from(u32::MAX) {
                        return Err(format!("{path} must not exceed {}", u32::MAX));
                    }
                    settings.max_total_tokens = Some(tokens);
                }
                "context_length" => {
                    settings.context_length = Some(positive_integer(value, &path)?);
                }
                "kv_pool_bytes" => {
                    settings.kv_pool_bytes = Some(positive_integer(value, &path)?);
                }
                "startup_timeout_seconds" => {
                    let seconds = positive_integer(value, &path)?;
                    if !(30..=1800).contains(&seconds) {
                        return Err(format!("{path} must be between 30 and 1800"));
                    }
                    settings.startup_timeout = Duration::from_secs(seconds);
                }
                // A misspelled limit must not silently launch with a default.
                _ => {
                    return Err(format!(
                        "{path} is not a managed SGLang setting; expected max_total_tokens, context_length, kv_pool_bytes, or startup_timeout_seconds"
                    ));
                }
            }
        }
        Ok(settings)
    }

    fn pool_tokens(&self, target_concurrency: usize) -> Result<u64, String> {
        match self.max_total_tokens {
            Some(tokens) => Ok(tokens),
            None => u64::try_from(target_concurrency.max(1))
                .ok()
                .and_then(|concurrency| concurrency.checked_mul(DEFAULT_TOKENS_PER_REQUEST))
                .filter(|tokens| *tokens <= u64::from(u32::MAX))
                .ok_or_else(|| {
                    "managed SGLang default KV pool size overflowed; set metadata.serving.sglang.max_total_tokens"
                        .to_string()
                }),
        }
    }
}

pub(crate) fn validate_managed_sglang_launch_policy(manifest: &Manifest) -> Result<(), String> {
    ManagedSglangSettings::from_manifest(manifest).map(|_| ())
}

fn positive_integer(raw: &serde_yaml::Value, path: &str) -> Result<u64, String> {
    raw.as_u64()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{path} must be a positive integer"))
}

#[derive(Debug, Deserialize)]
struct SglangModelConfig {
    num_hidden_layers: Option<u64>,
    num_attention_heads: Option<u64>,
    num_key_value_heads: Option<u64>,
    hidden_size: Option<u64>,
    head_dim: Option<u64>,
}

/// Device bytes one token occupies in SGLang's multi-head-attention KV pool:
/// a K and a V vector per layer and KV head.
fn kv_bytes_per_token(model_root: &Path) -> Result<u64, String> {
    let path = model_root.join("config.json");
    let raw = std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let config: SglangModelConfig = serde_json::from_slice(&raw)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let unsupported = |field: &str| {
        format!(
            "{} has no usable {field}, so the KV pool cannot be sized from it; set metadata.serving.sglang.kv_pool_bytes",
            path.display()
        )
    };
    let layers = config
        .num_hidden_layers
        .filter(|value| *value > 0)
        .ok_or_else(|| unsupported("num_hidden_layers"))?;
    let attention_heads = config.num_attention_heads.filter(|value| *value > 0);
    let kv_heads = config
        .num_key_value_heads
        .filter(|value| *value > 0)
        .or(attention_heads)
        .ok_or_else(|| unsupported("num_key_value_heads"))?;
    let head_dim = config
        .head_dim
        .filter(|value| *value > 0)
        .or_else(|| {
            let heads = attention_heads?;
            config
                .hidden_size
                .filter(|hidden| *hidden > 0 && hidden % heads == 0)
                .map(|hidden| hidden / heads)
        })
        .ok_or_else(|| unsupported("head_dim"))?;
    [layers, kv_heads, head_dim, KV_ELEMENT_BYTES]
        .into_iter()
        .try_fold(2u64, u64::checked_mul)
        .ok_or_else(|| {
            format!(
                "{} describes an implausibly large KV geometry",
                path.display()
            )
        })
}

/// Bytes to admit for a pool of `pool_tokens`. The adapter charges a fully
/// leased pool at its physical size rounded up to whole bytes per page, which
/// the last term covers.
fn kv_pool_claim_bytes(pool_tokens: u64, bytes_per_token: u64) -> Result<u64, String> {
    pool_tokens
        .checked_add(KV_PADDING_TOKENS)
        .and_then(|tokens| tokens.checked_mul(bytes_per_token))
        .and_then(|bytes| bytes.checked_add(pool_tokens))
        .ok_or_else(|| "managed SGLang KV pool size overflowed".to_string())
}

fn safetensors_weight_bytes(model_root: &Path) -> Result<usize, String> {
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
        weight_bytes = usize::try_from(metadata.len())
            .ok()
            .and_then(|bytes| weight_bytes.checked_add(bytes))
            .ok_or_else(|| {
                format!(
                    "SafeTensors weight bytes overflowed for model directory {}",
                    model_root.display()
                )
            })?;
    }
    if weight_bytes == 0 {
        return Err(format!(
            "managed SGLang model directory {} contains no .safetensors weights",
            model_root.display()
        ));
    }
    Ok(weight_bytes)
}

/// Static device footprint of one SGLang server: its weights, a workspace
/// estimate, and the whole KV pool it allocates at startup.
fn managed_sglang_memory_report(
    model_root: &Path,
    device_id: usize,
    model_id: u32,
    replica_id: u32,
    kv_pool_bytes: u64,
) -> Result<MemoryReport, String> {
    let weight_bytes = safetensors_weight_bytes(model_root)?;
    let workspace_bytes = (weight_bytes / 8).max(256 * 1024 * 1024);
    let kv_pool_bytes = usize::try_from(kv_pool_bytes)
        .map_err(|_| "managed SGLang KV pool is too large for this host".to_string())?;
    let allocation = |part: &str, class, bytes| MemoryAllocation {
        allocation_id: format!("managed-sglang:{model_id}:{replica_id}:{part}:{device_id}"),
        domain: MemoryDomain::Cuda { device_id },
        class,
        source: MemoryAllocationSource::BackendManaged,
        bytes,
    };
    Ok(MemoryReport {
        allocations: vec![
            allocation(
                "weights",
                MemoryAllocationClass::PersistentWeights,
                weight_bytes,
            ),
            allocation(
                "workspace",
                MemoryAllocationClass::TransientWorkspace,
                workspace_bytes,
            ),
            allocation("kv-pool", MemoryAllocationClass::KvCache, kv_pool_bytes),
        ],
    })
}

struct SglangDriver {
    python: PathBuf,
    model_root: PathBuf,
    served_model_name: String,
    port: u16,
    pool_tokens: u64,
    context_length: Option<u64>,
    cuda_visible_devices: String,
    participant_config: String,
    /// Bytes admitted for the KV pool at model load.
    admitted_kv_bytes: u64,
}

impl ManagedServerDriver for SglangDriver {
    fn name(&self) -> &'static str {
        "SGLang"
    }

    fn log_label(&self) -> &'static str {
        "sglang"
    }

    fn command(&self) -> Result<Command, EngineError> {
        let mut command = Command::new(&self.python);
        command
            .arg("-m")
            .arg(MANAGED_SGLANG_LAUNCH_MODULE)
            .arg("--model-path")
            .arg(&self.model_root)
            .args(["--served-model-name", &self.served_model_name])
            .args(["--host", "127.0.0.1"])
            .args(["--port", &self.port.to_string()])
            .args(["--max-total-tokens", &self.pool_tokens.to_string()])
            .env(
                kapsl_kv_abi::KV_PARTICIPANT_CONFIG_ENV,
                &self.participant_config,
            )
            .env("CUDA_VISIBLE_DEVICES", &self.cuda_visible_devices)
            // SGLang's default /health runs a one-token generation. That
            // request needs a KV lease like any other, so a device with no
            // lease capacity left would read as a dead server and be
            // restarted. Liveness is /health; governance is the participant.
            .env("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", "false");
        if let Some(context_length) = self.context_length {
            command.args(["--context-length", &context_length.to_string()]);
        }
        if let Some(bin_dir) = self
            .python
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            let mut paths = vec![bin_dir.to_path_buf()];
            if let Some(existing) = std::env::var_os("PATH") {
                paths.extend(std::env::split_paths(&existing));
            }
            let path = std::env::join_paths(paths).map_err(|error| {
                EngineError::backend(format!("build managed SGLang PATH: {error}"))
            })?;
            command.env("PATH", path);
        }
        Ok(command)
    }

    fn kv_status(
        &self,
        coordinator: &ExternalKvCoordinator,
        participant_base: &str,
    ) -> Result<Option<ManagedKvStatus>, String> {
        let Some(snapshot) = coordinator.managed_opaque_kv_snapshot(participant_base)? else {
            return Ok(None);
        };
        let capacity_bytes = snapshot.capacity_bytes.ok_or_else(|| {
            format!("participant under '{participant_base}' registered a KV pool with no ceiling")
        })?;
        if capacity_bytes > self.admitted_kv_bytes {
            return Err(format!(
                "participant under '{participant_base}' registered a {capacity_bytes}-byte KV pool, but {} bytes were admitted for it; set metadata.serving.sglang.kv_pool_bytes to the pool's real size",
                self.admitted_kv_bytes
            ));
        }
        Ok(Some(ManagedKvStatus {
            revision: snapshot.participant_epoch,
            capacity_bytes,
            leased_bytes: snapshot.leased_bytes,
            active_leases: snapshot.active_leases,
        }))
    }
}

/// Build the engine for one SGLang replica on one CUDA device. Nothing is
/// launched until the engine is loaded.
pub(crate) fn create_managed_sglang_engine(
    deployment: Arc<ManagedServerDeployment>,
    manifest: &Manifest,
    model_path: &Path,
    model_id: u32,
    replica_id: u32,
    device_id: usize,
    target_concurrency: usize,
) -> Result<Box<dyn Engine>, String> {
    let model_root = if model_path.is_dir() {
        model_path.to_path_buf()
    } else {
        model_path.parent().unwrap_or(model_path).to_path_buf()
    };
    if !model_root.join("config.json").is_file() {
        return Err(format!(
            "managed SGLang requires a Hugging Face model directory containing config.json; package cache {} does not contain it",
            model_root.display()
        ));
    }

    let settings = ManagedSglangSettings::from_manifest(manifest)?;
    let pool_tokens = settings.pool_tokens(target_concurrency)?;
    let admitted_kv_bytes = match settings.kv_pool_bytes {
        Some(bytes) => bytes,
        None => kv_pool_claim_bytes(pool_tokens, kv_bytes_per_token(&model_root)?)?,
    };
    let memory_report = managed_sglang_memory_report(
        &model_root,
        device_id,
        model_id,
        replica_id,
        admitted_kv_bytes,
    )?;

    let port = reserve_loopback_port()?;
    let participant_base = format!(
        "kapsl-{}-{}-{}",
        sanitize_identifier(&manifest.project_name),
        model_id,
        replica_id
    );
    let replica_dir = deployment
        .runtime_root
        .join(format!("model-{model_id}-replica-{replica_id}"));
    std::fs::create_dir_all(&replica_dir).map_err(|error| {
        format!(
            "create managed SGLang model directory {}: {error}",
            replica_dir.display()
        )
    })?;
    let participant_config = managed_kv_participant_config(
        &deployment.control_endpoint,
        &participant_base,
        &model_fingerprint(manifest, &model_root)?,
        kapsl_kv_abi::KvParticipantMode::Opaque,
        &[device_id],
        deployment.lease_ttl_ms,
    )?;
    participant_config
        .validate()
        .map_err(|error| format!("managed SGLang participant config: {error}"))?;
    let participant_config = serde_json::to_string(&participant_config)
        .map_err(|error| format!("serialize managed SGLang participant config: {error}"))?;

    log::info!(
        "[sglang-memory] model={} replica={} device={} max_total_tokens={} admitted_kv_pool_bytes={} admitted_static_bytes={}",
        manifest.project_name,
        replica_id,
        device_id,
        pool_tokens,
        admitted_kv_bytes,
        memory_report
            .allocations
            .iter()
            .fold(0usize, |total, allocation| total
                .saturating_add(allocation.bytes)),
    );

    let driver = SglangDriver {
        python: deployment.python.clone(),
        model_root,
        served_model_name: manifest.project_name.clone(),
        port,
        pool_tokens,
        context_length: settings.context_length,
        cuda_visible_devices: child_cuda_visibility(&[device_id])?,
        participant_config,
        admitted_kv_bytes,
    };
    let process = ManagedServerProcess::new(
        Box::new(driver),
        ManagedServerSpec {
            served_model_name: manifest.project_name.clone(),
            participant_base,
            endpoint: format!("http://127.0.0.1:{port}"),
            log_path: replica_dir.join("sglang.log"),
            startup_timeout: settings.startup_timeout,
            target_concurrency: target_concurrency.max(1),
        },
    )?;
    ManagedServerEngine::create(deployment, process, model_id, memory_report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_kv_abi::KvCoordinator as _;

    fn manifest(sglang: Option<&str>) -> Manifest {
        Manifest {
            project_name: "test-model".to_string(),
            framework: "safetensors".to_string(),
            version: "1.0.0".to_string(),
            created_at: String::new(),
            model_file: "model.safetensors".to_string(),
            format: Some("safetensors".to_string()),
            model_type: Some("causal-lm".to_string()),
            task: Some("generate".to_string()),
            metadata: sglang.map(|settings| {
                serde_yaml::from_str(&format!(
                    "serving:\n  backend: sglang\n  sglang: {settings}\n"
                ))
                .expect("serving metadata")
            }),
            hardware_requirements: kapsl_core::HardwareRequirements::default(),
            cron_jobs: Vec::new(),
        }
    }

    fn driver(admitted_kv_bytes: u64) -> SglangDriver {
        SglangDriver {
            python: PathBuf::from("/opt/sglang/bin/python"),
            model_root: PathBuf::from("/models/qwen"),
            served_model_name: "qwen".to_string(),
            port: 30000,
            pool_tokens: 8192,
            context_length: None,
            cuda_visible_devices: "1".to_string(),
            participant_config: "{\"kapsl_kv_mode\":\"opaque\"}".to_string(),
            admitted_kv_bytes,
        }
    }

    #[test]
    fn settings_default_and_parse_every_field() {
        let defaults = ManagedSglangSettings::from_manifest(&manifest(None)).unwrap();
        assert_eq!(defaults.max_total_tokens, None);
        assert_eq!(defaults.startup_timeout, DEFAULT_STARTUP_TIMEOUT);
        assert_eq!(
            defaults.pool_tokens(4).unwrap(),
            4 * DEFAULT_TOKENS_PER_REQUEST
        );
        assert_eq!(defaults.pool_tokens(0).unwrap(), DEFAULT_TOKENS_PER_REQUEST);

        let settings = ManagedSglangSettings::from_manifest(&manifest(Some(
            "{max_total_tokens: 2048, context_length: 1024, kv_pool_bytes: 4096, startup_timeout_seconds: 60}",
        )))
        .unwrap();
        assert_eq!(
            settings,
            ManagedSglangSettings {
                max_total_tokens: Some(2048),
                context_length: Some(1024),
                kv_pool_bytes: Some(4096),
                startup_timeout: Duration::from_secs(60),
            }
        );
        assert_eq!(settings.pool_tokens(4).unwrap(), 2048);
    }

    #[test]
    fn settings_reject_unknown_keys_and_unusable_values() {
        for (settings, expected) in [
            ("{max_total_token: 2048}", "is not a managed SGLang setting"),
            ("{max_total_tokens: 0}", "must be a positive integer"),
            ("{max_total_tokens: -1}", "must be a positive integer"),
            ("{max_total_tokens: 4294967296}", "must not exceed"),
            ("{context_length: many}", "must be a positive integer"),
            ("{kv_pool_bytes: 0}", "must be a positive integer"),
            ("{startup_timeout_seconds: 5}", "between 30 and 1800"),
            ("[1, 2]", "must be an object"),
        ] {
            let error = validate_managed_sglang_launch_policy(&manifest(Some(settings)))
                .expect_err(settings);
            assert!(error.contains(expected), "{settings}: {error}");
        }
    }

    #[test]
    fn kv_geometry_follows_the_model_config() {
        let root = tempfile::tempdir().unwrap();
        let write = |config: serde_json::Value| {
            std::fs::write(root.path().join("config.json"), config.to_string()).unwrap();
        };

        // Qwen2.5-0.5B: 24 layers, 2 KV heads of 64 dimensions, 16-bit floats.
        write(serde_json::json!({
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "hidden_size": 896,
            "torch_dtype": "bfloat16",
        }));
        assert_eq!(
            kv_bytes_per_token(root.path()).unwrap(),
            2 * 24 * 2 * 64 * 2
        );

        // Without grouped-query attention every attention head holds KV.
        write(serde_json::json!({
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "hidden_size": 768,
        }));
        assert_eq!(
            kv_bytes_per_token(root.path()).unwrap(),
            2 * 12 * 12 * 64 * 2
        );

        // An explicit head dimension wins over hidden_size / heads.
        write(serde_json::json!({
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "hidden_size": 1024,
            "head_dim": 128,
        }));
        assert_eq!(kv_bytes_per_token(root.path()).unwrap(), 2 * 2 * 128 * 2);

        write(serde_json::json!({"num_attention_heads": 4, "hidden_size": 1024}));
        let error = kv_bytes_per_token(root.path()).unwrap_err();
        assert!(error.contains("num_hidden_layers"), "{error}");
        assert!(error.contains("kv_pool_bytes"), "{error}");

        write(
            serde_json::json!({"num_hidden_layers": 2, "num_attention_heads": 3, "hidden_size": 1024}),
        );
        assert!(kv_bytes_per_token(root.path())
            .unwrap_err()
            .contains("head_dim"));
    }

    #[test]
    fn kv_pool_claim_covers_the_padding_page_and_per_page_rounding() {
        assert_eq!(
            kv_pool_claim_bytes(8192, 12_288).unwrap(),
            8193 * 12_288 + 8192
        );
        assert!(kv_pool_claim_bytes(u64::MAX, 2).is_err());
        assert!(kv_pool_claim_bytes(u64::from(u32::MAX), u64::MAX / 2).is_err());
    }

    #[test]
    fn memory_report_admits_weights_workspace_and_the_whole_pool() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("model-00001.safetensors"), [0u8; 4096]).unwrap();
        std::fs::write(root.path().join("model-00002.safetensors"), [0u8; 1024]).unwrap();
        std::fs::write(root.path().join("tokenizer.json"), [0u8; 512]).unwrap();

        let report = managed_sglang_memory_report(root.path(), 1, 7, 0, 1_000_000).unwrap();
        let rows = report
            .allocations
            .iter()
            .map(|allocation| {
                assert_eq!(allocation.domain, MemoryDomain::Cuda { device_id: 1 });
                assert_eq!(allocation.source, MemoryAllocationSource::BackendManaged);
                (
                    allocation.allocation_id.as_str(),
                    allocation.class,
                    allocation.bytes,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            rows,
            vec![
                (
                    "managed-sglang:7:0:weights:1",
                    MemoryAllocationClass::PersistentWeights,
                    5120
                ),
                (
                    "managed-sglang:7:0:workspace:1",
                    MemoryAllocationClass::TransientWorkspace,
                    256 * 1024 * 1024
                ),
                (
                    "managed-sglang:7:0:kv-pool:1",
                    MemoryAllocationClass::KvCache,
                    1_000_000
                ),
            ]
        );

        let empty = tempfile::tempdir().unwrap();
        assert!(managed_sglang_memory_report(empty.path(), 0, 1, 0, 1)
            .unwrap_err()
            .contains("no .safetensors weights"));
    }

    #[test]
    fn command_runs_the_adapter_launcher_with_one_config_document() {
        let mut driver = driver(1);
        driver.context_length = Some(4096);
        let command = driver.command().unwrap();
        assert_eq!(command.get_program(), "/opt/sglang/bin/python");
        let args = command
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            args,
            [
                "-m",
                "kapsl_sglang_adapter.launch",
                "--model-path",
                "/models/qwen",
                "--served-model-name",
                "qwen",
                "--host",
                "127.0.0.1",
                "--port",
                "30000",
                "--max-total-tokens",
                "8192",
                "--context-length",
                "4096",
            ]
        );
        let environment = command
            .get_envs()
            .filter_map(|(name, value)| {
                Some((
                    name.to_string_lossy().into_owned(),
                    value?.to_string_lossy().into_owned(),
                ))
            })
            .collect::<HashMap<_, _>>();
        assert_eq!(
            environment["KAPSL_KV_PARTICIPANT_CONFIG"],
            "{\"kapsl_kv_mode\":\"opaque\"}"
        );
        assert_eq!(environment["CUDA_VISIBLE_DEVICES"], "1");
        assert_eq!(
            environment["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"],
            "false"
        );
        assert!(!environment
            .keys()
            .any(|name| name.starts_with("KAPSL_KV_") && name != "KAPSL_KV_PARTICIPANT_CONFIG"));
    }

    #[test]
    fn participant_config_is_the_opaque_contract_document() {
        let config = managed_kv_participant_config(
            "unix:///run/kapsl/kv-control.sock",
            "kapsl-model-1-0",
            "sha256:model",
            kapsl_kv_abi::KvParticipantMode::Opaque,
            &[3],
            30_000,
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(
            serde_json::to_value(&config).unwrap(),
            serde_json::json!({
                "kapsl_control_endpoint": "unix:///run/kapsl/kv-control.sock",
                "kapsl_participant_id": "kapsl-model-1-0",
                "kapsl_model_fingerprint": "sha256:model",
                "kapsl_kv_mode": "opaque",
                "kapsl_memory_domains": [{"kind": "cuda", "device_id": 3}],
                "kapsl_rank_device_map": {"0": 3},
                "kapsl_lease_ttl_ms": 30_000,
                "kapsl_live_resize": false,
                "kapsl_vmm_conformance": false,
            })
        );
    }

    #[test]
    fn environment_check_names_every_mismatch() {
        let supported = || ManagedSglangEnvironment {
            sglang: MANAGED_SGLANG_BACKEND_VERSION.to_string(),
            adapter_distribution: MANAGED_SGLANG_ADAPTER_VERSION.to_string(),
            adapter: MANAGED_SGLANG_ADAPTER_VERSION.to_string(),
            pinned_sglang: vec![MANAGED_SGLANG_BACKEND_VERSION.to_string()],
            kv_abi_major: MANAGED_VLLM_KV_ABI_MAJOR,
            kv_abi_minor: MANAGED_VLLM_KV_ABI_MINOR,
            participant_config_env: "KAPSL_KV_PARTICIPANT_CONFIG".to_string(),
        };
        check_sglang_environment(&supported()).unwrap();

        let mut drifted = supported();
        drifted.sglang = "0.5.24".to_string();
        drifted.adapter = "0.1.0".to_string();
        drifted.kv_abi_minor += 1;
        drifted.participant_config_env = "KAPSL_KV_CONTROL_ENDPOINT".to_string();
        let error = check_sglang_environment(&drifted).unwrap_err();
        for expected in [
            "SGLang 0.5.24 is installed",
            "kapsl-sglang-adapter 0.1.0",
            "speaks KV ABI",
            "reads its configuration from KAPSL_KV_CONTROL_ENDPOINT",
        ] {
            assert!(error.contains(expected), "{expected}: {error}");
        }

        let mut unpinned = supported();
        unpinned.pinned_sglang = vec!["0.5.22".to_string()];
        assert!(check_sglang_environment(&unpinned)
            .unwrap_err()
            .contains("does not pin SGLang"));
    }

    fn coordinator() -> Arc<ExternalKvCoordinator> {
        use kapsl_hal::device::{Device, DeviceBackend, DeviceInfo};
        let info = DeviceInfo {
            cpu_cores: 1,
            total_memory: 1024 * 1024,
            os_type: "test".to_string(),
            os_release: "test".to_string(),
            has_cuda: false,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: vec![Device {
                id: 0,
                name: "test-cpu".to_string(),
                backend: DeviceBackend::Cpu,
                memory_mb: 0,
                compute_units: 1,
                pci_bus_id: None,
                partition_id: None,
                driver_version: None,
                cuda_version: None,
                compute_capability: None,
                utilization_gpu_pct: None,
                temperature_c: None,
                supports_fp16: false,
                supports_int8: true,
            }],
        };
        let memory = crate::runtime::memory::MemoryAuthority::new(&info).expect("authority");
        ExternalKvCoordinator::new(memory, Duration::from_secs(30)).expect("coordinator")
    }

    fn registration(max_allocations: Option<u64>) -> kapsl_kv_abi::KvParticipantRegistration {
        kapsl_kv_abi::KvParticipantRegistration {
            participant_id: "kapsl-model-1-0:scheduler".to_string(),
            backend: "sglang".to_string(),
            model_fingerprint: "sha256:test".to_string(),
            capabilities: kapsl_kv_abi::KvBackendCapabilities::opaque_connected(),
            capacity_model: kapsl_kv_abi::KvCapacityModel {
                groups: vec![kapsl_kv_abi::KvCapacityGroup {
                    group_id: "sglang.kv.0".to_string(),
                    pool_id: "sglang.pool.0".to_string(),
                    allocation_granularity_tokens: 1,
                    bytes_per_allocation: 64,
                    memory_domains: vec![kapsl_kv_abi::KvMemoryDomain::Host],
                    max_allocations,
                }],
            },
            adapter_profile: None,
            topology: None,
            provisioning_grant: None,
        }
    }

    #[test]
    fn readiness_needs_a_registered_pool_within_the_admitted_bytes() {
        let coordinator = coordinator();
        let base = "kapsl-model-1-0";
        assert_eq!(driver(6400).kv_status(&coordinator, base).unwrap(), None);

        let receipt = coordinator.register(&registration(Some(100))).unwrap();
        assert_eq!(
            driver(6400).kv_status(&coordinator, base).unwrap(),
            Some(ManagedKvStatus {
                revision: receipt.participant_epoch,
                capacity_bytes: 6400,
                leased_bytes: 0,
                active_leases: 0,
            })
        );

        // A server whose pool is larger than what was admitted stays unroutable.
        let error = driver(6399).kv_status(&coordinator, base).unwrap_err();
        assert!(error.contains("registered a 6400-byte KV pool"), "{error}");
        assert!(error.contains("6399 bytes were admitted"), "{error}");
    }

    #[test]
    fn readiness_refuses_a_pool_without_a_ceiling() {
        let coordinator = coordinator();
        coordinator.register(&registration(None)).unwrap();
        let error = driver(u64::MAX)
            .kv_status(&coordinator, "kapsl-model-1-0")
            .unwrap_err();
        assert!(error.contains("no ceiling"), "{error}");
    }
}
