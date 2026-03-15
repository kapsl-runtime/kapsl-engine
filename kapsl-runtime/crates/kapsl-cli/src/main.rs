use base64::engine::general_purpose::{
    STANDARD as BASE64, URL_SAFE_NO_PAD as BASE64_URL_SAFE_NO_PAD,
};
use base64::Engine as _;
use clap::{
    parser::ValueSource, ArgGroup, ArgMatches, FromArgMatches, Parser, Subcommand, ValueEnum,
};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use futures::stream;
use infer_adapter::{default_request_adapter_registry, parse_inference_request_with_registry};
use kapsl_backends::{BackendFactory, OnnxRuntimeTuning};
use kapsl_core::loader::Manifest;
use kapsl_core::{AutoScaler, ModelInfo, ModelRegistry, ModelStatus, PackageLoader, ScalingPolicy};
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, EngineHandle, EngineMetrics, EngineModelInfo,
    InferenceRequest, TensorDtype,
};
use kapsl_hal::device::DeviceInfo;
use kapsl_ipc::{
    IpcServer, RequestHeader, ResponseHeader, TcpServer, OP_INFER, OP_INFER_STREAM, STATUS_ERR,
    STATUS_OK, STATUS_STREAM_CHUNK, STATUS_STREAM_END,
};
use kapsl_llm::llm_backend::LLMBackend;
use kapsl_llm::rag::{
    build_rag_prompt, CitationStyle, RagChunk, RagPromptConfig, WhitespaceTokenCounter,
};
use kapsl_monitor::middleware::MonitoringMiddleware;
use kapsl_rag::extension::{
    ConnectorRuntimeHandle, ExtensionManager, ExtensionRegistry, InstalledExtension,
};
use kapsl_rag::vector::SqliteVectorStore;
use kapsl_rag::{
    AccessControl, ConnectorClient, DocStore, EmbeddedChunk, FsDocStore, VectorQuery, VectorStore,
};
use kapsl_rag_sdk::protocol::{ConnectorRequestKind, ConnectorResponseKind, ConnectorResult};
use kapsl_rag_sdk::types::{DeltaOp, DocumentDelta, DocumentPayload, SourceDescriptor};
use kapsl_scheduler::{
    determine_priority, PoolStrategy, ReplicaPool, ReplicaScheduler,
    RequestMetadata as SchedulerRequestMetadata, Scheduler,
};
use kapsl_shm::memory::ShmManager;
use kapsl_shm::ShmServer;
use kapsl_transport::TransportServer;
use mime_guess;
use parking_lot::{Mutex, RwLock};
use prometheus::{Encoder, Registry, TextEncoder};
use rand::rngs::OsRng;
use rand::RngCore;
use rust_embed::RustEmbed;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::future::Future;
use std::io::{Cursor, Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{Pid, System};
use tar::{Archive, Builder};
use tokio::sync::Mutex as AsyncMutex;
use warp::http::StatusCode;
use warp::{Filter, Reply};

#[cfg(unix)]
use std::os::unix::net::UnixStream;

mod infer_adapter;

#[derive(RustEmbed)]
#[folder = "../../ui"]
struct UiAssets;

type DynError = Box<dyn std::error::Error + Send + Sync>;
type ReplicaPools = Arc<RwLock<HashMap<u32, Arc<ReplicaPool<Scheduler>>>>>;

const CLI_AFTER_HELP: &str = "\
Examples:
  kapsl run --model models/mnist/mnist.aimod
  kapsl control --runtime gpu0=http://127.0.0.1:9095 --runtime gpu1=http://127.0.0.1:9096
  kapsl build ./models/gpt-llm
  kapsl build --model ./model.onnx --output ./model.aimod
  kapsl build ./model.onnx --output ./model.aimod
  kapsl build   (from inside a model context directory)
  kapsl push acme/mnist:prod ./model.aimod
  kapsl push acme/mnist:prod   (from inside a directory with a single .aimod)
  kapsl push acme/mnist:prod ./model.aimod --remote-url oci://ghcr.io
  kapsl pull acme/mnist:prod --destination-dir ./models
  kapsl login

Compatibility:
  kapsl --model models/mnist/mnist.aimod
    (equivalent to `kapsl run --model models/mnist/mnist.aimod`)";

#[derive(Parser, Debug)]
#[command(
    name = "kapsl",
    author,
    version,
    about = "Kapsl runtime and packaging CLI",
    long_about = "Run model packages, build new packages, and sync packages with a remote backend.",
    after_help = CLI_AFTER_HELP
)]
struct Cli {
    #[command(subcommand)]
    command: Option<KapslCommand>,

    #[command(flatten)]
    run: Args,
}

#[derive(Subcommand, Debug)]
enum KapslCommand {
    /// Run the Kapsl runtime server
    Run(Args),
    /// Run multi-runtime control loop (cross-port weights + scaling policy orchestration)
    Control(ControlCommandArgs),
    /// Build a .aimod model package from a source model file
    Build(BuildCommandArgs),
    /// Push a .aimod package to the configured remote backend
    Push(PushCommandArgs),
    /// Pull a .aimod package from the configured remote backend
    Pull(PullCommandArgs),
    /// Authenticate with a remote backend and save token locally
    Login(LoginCommandArgs),
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Run Options")]
struct Args {
    /// Path to .aimod package(s)
    #[arg(short, long)]
    model: Vec<PathBuf>,

    /// Transport mode: socket, tcp, shm, hybrid, or auto
    #[arg(long, default_value = "socket")]
    transport: String,

    /// Path to unix socket (for socket mode)
    #[cfg_attr(unix, arg(short, long, default_value = "/tmp/kapsl.sock"))]
    #[cfg_attr(windows, arg(short, long, default_value = r"\\.\pipe\kapsl"))]
    socket: String,

    /// Bind address for TCP server
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,

    /// TCP port for network transport
    #[arg(long, default_value_t = 9096)]
    port: u16,

    /// Max batch size
    #[arg(long, default_value_t = 4)]
    batch_size: usize,

    /// Max queue depth per scheduler priority queue / worker
    #[arg(long, default_value_t = 256)]
    scheduler_queue_size: usize,

    /// Max requests to aggregate into a throughput micro-batch
    #[arg(long, default_value_t = 4)]
    scheduler_max_micro_batch: usize,

    /// Target queue delay in ms before dispatching throughput micro-batches
    #[arg(long, default_value_t = 2)]
    scheduler_queue_delay_ms: u64,

    /// Runtime performance profile that tunes defaults when related flags are not explicitly set.
    /// Defaults to "auto" which selects parameters based on model size and system resources.
    #[arg(long, value_enum, default_value_t = PerformanceProfile::Auto)]
    performance_profile: PerformanceProfile,

    /// Metrics server port
    #[arg(long, default_value_t = 9095)]
    metrics_port: u16,

    /// Bind address for HTTP API/UI/metrics server
    #[arg(long, default_value = "127.0.0.1")]
    http_bind: String,

    /// Root directory for runtime state (rag-data, extensions, extensions-config, auth-store.json).
    /// When set, overrides KAPSL_RAG_STORAGE_ROOT, KAPSL_EXTENSIONS_ROOT, KAPSL_EXT_CONFIG_ROOT,
    /// and KAPSL_AUTH_STORE_PATH.
    #[arg(long, value_name = "DIR")]
    state_dir: Option<PathBuf>,

    /// Mesh topology (data-parallel, tensor-parallel, pipeline-parallel, mixed)
    #[arg(long, default_value = "data-parallel")]
    topology: String,

    /// Tensor parallelism degree (number of devices per TP group)
    #[arg(long, default_value_t = 1)]
    tp_degree: usize,

    /// Run as isolated worker process (internal)
    #[arg(long, hide = true)]
    worker: bool,

    /// Model ID for isolated worker process (internal)
    #[arg(long, hide = true)]
    worker_model_id: Option<u32>,

    /// Global ONNX Runtime memory-pattern setting for all ONNX models (true/false).
    #[arg(long, value_name = "BOOL")]
    onnx_memory_pattern: Option<bool>,

    /// Global ONNX Runtime CPU arena toggle for all ONNX models (true/false).
    #[arg(long, value_name = "BOOL")]
    onnx_disable_cpu_mem_arena: Option<bool>,

    /// Global ONNX Runtime session bucket count for shape-bucketed session reuse.
    #[arg(long, value_name = "N")]
    onnx_session_buckets: Option<usize>,

    /// Global ONNX Runtime non-batch dimension bucket granularity.
    #[arg(long, value_name = "N")]
    onnx_bucket_dim_granularity: Option<usize>,

    /// Global ONNX Runtime number of leading dims used for bucket keys.
    #[arg(long, value_name = "N")]
    onnx_bucket_max_dims: Option<usize>,

    /// Global peak-concurrency hint exported in ONNX model metadata.
    #[arg(long, value_name = "N")]
    onnx_peak_concurrency_hint: Option<u32>,

    /// Shared memory pool size in MiB for shm/hybrid/auto transports.
    /// Also read from KAPSL_SHM_SIZE_MB. Default: 256 MiB.
    #[arg(long, value_name = "MIB")]
    shm_size_mb: Option<usize>,

    /// Per-model ONNX tuning override.
    /// Format: `<model_id|*>:k=v[,k=v...]`
    /// Keys: memory_pattern, disable_cpu_mem_arena, session_buckets,
    /// bucket_dim_granularity, bucket_max_dims, peak_concurrency.
    #[arg(long, value_name = "SPEC")]
    onnx_model_tuning: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct OnnxTuningProfile {
    global: OnnxRuntimeTuning,
    per_model: HashMap<u32, OnnxRuntimeTuning>,
}

fn merge_onnx_runtime_tuning(
    base: &OnnxRuntimeTuning,
    overrides: &OnnxRuntimeTuning,
) -> OnnxRuntimeTuning {
    OnnxRuntimeTuning {
        memory_pattern: overrides.memory_pattern.or(base.memory_pattern),
        disable_cpu_mem_arena: overrides
            .disable_cpu_mem_arena
            .or(base.disable_cpu_mem_arena),
        session_buckets: overrides.session_buckets.or(base.session_buckets),
        bucket_dim_granularity: overrides
            .bucket_dim_granularity
            .or(base.bucket_dim_granularity),
        bucket_max_dims: overrides.bucket_max_dims.or(base.bucket_max_dims),
        peak_concurrency_hint: overrides
            .peak_concurrency_hint
            .or(base.peak_concurrency_hint),
    }
}

impl OnnxTuningProfile {
    fn resolve(&self, model_id: u32) -> OnnxRuntimeTuning {
        if let Some(model_overrides) = self.per_model.get(&model_id) {
            merge_onnx_runtime_tuning(&self.global, model_overrides)
        } else {
            self.global.clone()
        }
    }
}

fn parse_bool_literal(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid boolean '{}'", value)),
    }
}

fn apply_onnx_tuning_pair(
    target: &mut OnnxRuntimeTuning,
    key: &str,
    value: &str,
) -> Result<(), String> {
    let normalized = key.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "memory_pattern" | "mem_pattern" => {
            target.memory_pattern = Some(parse_bool_literal(value)?);
        }
        "disable_cpu_mem_arena" | "cpu_mem_arena_disabled" => {
            target.disable_cpu_mem_arena = Some(parse_bool_literal(value)?);
        }
        "session_buckets" => {
            let parsed = value
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("invalid session_buckets '{}': {}", value, e))?;
            target.session_buckets = Some(parsed.max(1));
        }
        "bucket_dim_granularity" => {
            let parsed = value
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("invalid bucket_dim_granularity '{}': {}", value, e))?;
            target.bucket_dim_granularity = Some(parsed.max(1));
        }
        "bucket_max_dims" => {
            let parsed = value
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("invalid bucket_max_dims '{}': {}", value, e))?;
            target.bucket_max_dims = Some(parsed.max(1));
        }
        "peak_concurrency" | "peak_concurrency_hint" => {
            let parsed = value
                .trim()
                .parse::<u32>()
                .map_err(|e| format!("invalid peak_concurrency '{}': {}", value, e))?;
            target.peak_concurrency_hint = Some(parsed.max(1));
        }
        other => {
            return Err(format!(
                "unknown ONNX tuning key '{}'; expected one of memory_pattern, disable_cpu_mem_arena, session_buckets, bucket_dim_granularity, bucket_max_dims, peak_concurrency",
                other
            ));
        }
    }
    Ok(())
}

fn parse_onnx_model_tuning_spec(spec: &str) -> Result<(Option<u32>, OnnxRuntimeTuning), String> {
    let (selector_raw, config_raw) = spec.split_once(':').ok_or_else(|| {
        format!(
            "invalid --onnx-model-tuning '{}': expected '<model_id|*>:k=v[,k=v...]'",
            spec
        )
    })?;
    let selector = selector_raw.trim();
    let model_id = if selector == "*" {
        None
    } else {
        Some(
            selector
                .parse::<u32>()
                .map_err(|e| format!("invalid model selector '{}': {}", selector, e))?,
        )
    };

    let mut tuning = OnnxRuntimeTuning::default();
    for pair in config_raw.split(',') {
        let trimmed = pair.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (key, value) = trimmed
            .split_once('=')
            .ok_or_else(|| format!("invalid tuning pair '{}': expected k=v", trimmed))?;
        apply_onnx_tuning_pair(&mut tuning, key, value)?;
    }

    Ok((model_id, tuning))
}

fn build_onnx_tuning_profile(args: &Args) -> Result<OnnxTuningProfile, String> {
    let mut profile = OnnxTuningProfile {
        global: OnnxRuntimeTuning {
            memory_pattern: args.onnx_memory_pattern,
            disable_cpu_mem_arena: args.onnx_disable_cpu_mem_arena,
            session_buckets: args.onnx_session_buckets,
            bucket_dim_granularity: args.onnx_bucket_dim_granularity,
            bucket_max_dims: args.onnx_bucket_max_dims,
            peak_concurrency_hint: args.onnx_peak_concurrency_hint,
        },
        per_model: HashMap::new(),
    };

    for spec in &args.onnx_model_tuning {
        let (model_id, tuning) = parse_onnx_model_tuning_spec(spec)?;
        if let Some(model_id) = model_id {
            let merged = profile
                .per_model
                .get(&model_id)
                .map(|existing| merge_onnx_runtime_tuning(existing, &tuning))
                .unwrap_or(tuning);
            profile.per_model.insert(model_id, merged);
        } else {
            profile.global = merge_onnx_runtime_tuning(&profile.global, &tuning);
        }
    }

    Ok(profile)
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Build Options")]
struct BuildCommandArgs {
    /// Build context directory (Docker-style) or a model file path (for example ./models/gpt-llm or ./model.onnx)
    #[arg(value_name = "CONTEXT")]
    context: Option<PathBuf>,

    /// Source model file path (for example, model.onnx or model.gguf)
    #[arg(long, value_name = "PATH")]
    model: Option<PathBuf>,

    /// Output .aimod package path
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Optional project name override
    #[arg(long)]
    project_name: Option<String>,

    /// Optional framework override
    #[arg(long)]
    framework: Option<String>,

    /// Optional version override
    #[arg(long)]
    version: Option<String>,

    /// Optional JSON object string for metadata
    #[arg(long, value_name = "JSON")]
    metadata_json: Option<String>,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Push Options")]
struct PushCommandArgs {
    /// Push target in the format <repo_name>/<model>:<label>
    #[arg(value_name = "TARGET")]
    target: String,

    /// Package path to push (defaults to the only `.aimod` in the current directory)
    #[arg(value_name = "KAPSL")]
    kapsl: Option<PathBuf>,

    /// Package path to push
    #[arg(long, alias = "kapsl-path", value_name = "PATH")]
    model: Option<PathBuf>,

    /// Override remote URL for this upload
    #[arg(long, value_name = "URL")]
    remote_url: Option<String>,

    /// Bearer token for authenticated remote backends (also read from KAPSL_REMOTE_TOKEN)
    #[arg(long, value_name = "TOKEN")]
    remote_token: Option<String>,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Pull Options")]
#[command(
    group(
        ArgGroup::new("pull_target")
            .required(true)
            .args(["target", "model"])
    )
)]
struct PullCommandArgs {
    /// Pull target in the format <repo_name>/<model>:<label>
    #[arg(value_name = "TARGET")]
    target: Option<String>,

    /// Pull target in the format <repo_name>/<model>:<label>
    #[arg(long, alias = "target-ref", value_name = "TARGET")]
    model: Option<String>,

    /// Optional OCI digest reference when using an `oci://` remote (e.g. `sha256:<digest>` or `@sha256:<digest>`)
    #[arg(long = "ref", value_name = "REF")]
    reference: Option<String>,

    /// Destination directory for the downloaded package
    #[arg(long, value_name = "DIR")]
    destination_dir: Option<PathBuf>,

    /// Override remote URL for this download
    #[arg(long, value_name = "URL")]
    remote_url: Option<String>,

    /// Bearer token for authenticated remote backends (also read from KAPSL_REMOTE_TOKEN)
    #[arg(long, value_name = "TOKEN")]
    remote_token: Option<String>,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Login Options")]
struct LoginCommandArgs {
    /// Backend base URL (defaults to KAPSL_REMOTE_URL or https://api.kapsl.net/v1)
    #[arg(long, value_name = "URL")]
    remote_url: Option<String>,

    /// OAuth provider to use
    #[arg(long, value_enum, default_value_t = OAuthProvider::GitHub)]
    provider: OAuthProvider,

    /// Local callback host for browser redirect
    #[arg(long, value_name = "HOST", default_value = "127.0.0.1")]
    callback_host: String,

    /// Local callback port (0 picks an ephemeral free port)
    #[arg(long, value_name = "PORT", default_value_t = 0)]
    callback_port: u16,

    /// Max time to wait for browser login callback
    #[arg(long, value_name = "SECONDS", default_value_t = 180)]
    timeout_seconds: u64,

    /// Print login URL instead of opening a browser automatically
    #[arg(long, default_value_t = false)]
    no_browser: bool,

    /// Use OAuth Device Code flow for headless/SSH environments (GitHub provider only)
    #[arg(
        long = "device-code",
        visible_alias = "headless",
        default_value_t = false
    )]
    device_code: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OAuthProvider {
    #[value(name = "github", alias = "git-hub")]
    GitHub,
    #[value(name = "google")]
    Google,
}

impl OAuthProvider {
    fn route_segment(self) -> &'static str {
        match self {
            Self::GitHub => "github",
            Self::Google => "google",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum RuntimeGroupProfile {
    Latency,
    Balanced,
    Throughput,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Control Options")]
struct ControlCommandArgs {
    /// Runtime endpoint definition as NAME=URL (repeat for each runtime)
    #[arg(long = "runtime", value_name = "NAME=URL", required = true)]
    runtimes: Vec<String>,

    /// Runtime profile override as NAME=latency|balanced|throughput
    #[arg(long = "runtime-profile", value_name = "NAME=PROFILE")]
    runtime_profiles: Vec<String>,

    /// Runtime token override as NAME=TOKEN
    #[arg(long = "runtime-token", value_name = "NAME=TOKEN")]
    runtime_tokens: Vec<String>,

    /// Shared bearer token for all runtimes that don't define --runtime-token
    #[arg(long = "auth-token", value_name = "TOKEN")]
    auth_token: Option<String>,

    /// Memory budget override per runtime as NAME=BYTES
    #[arg(long = "memory-budget-bytes", value_name = "NAME=BYTES")]
    memory_budget_bytes: Vec<String>,

    /// Poll interval in seconds
    #[arg(long, default_value_t = 5)]
    interval_seconds: u64,

    /// Per-request timeout in milliseconds for runtime API polling/update calls
    #[arg(long, default_value_t = 1500)]
    timeout_ms: u64,

    /// Queue depth target for score normalization
    #[arg(long, default_value_t = 10)]
    queue_target: usize,

    /// Score threshold above which runtime weight decreases
    #[arg(long, default_value_t = 0.85)]
    high_pressure_score: f64,

    /// Score threshold below which runtime weight increases
    #[arg(long, default_value_t = 0.45)]
    low_pressure_score: f64,

    /// GPU hot threshold (0.0-1.0)
    #[arg(long, default_value_t = 0.92)]
    hot_gpu_utilization: f64,

    /// Memory hot threshold (0.0-1.0; requires --memory-budget-bytes)
    #[arg(long, default_value_t = 0.90)]
    hot_memory_utilization: f64,

    /// Sustained overload window before extra traffic shedding
    #[arg(long, default_value_t = 30)]
    overload_window_seconds: u64,

    /// Sustained hot window before extra traffic shedding
    #[arg(long, default_value_t = 20)]
    hot_window_seconds: u64,

    /// Hold runtime at zero weight after health/API failure
    #[arg(long, default_value_t = 30)]
    unhealthy_hold_seconds: u64,

    /// Max proportional weight move per cycle
    #[arg(long, default_value_t = 0.10)]
    weight_step: f64,

    /// Minimum non-zero weight for eligible runtimes
    #[arg(long, default_value_t = 0.05)]
    weight_floor: f64,

    /// Extra weight cut when overload/hot windows are exceeded
    #[arg(long, default_value_t = 0.20)]
    overload_shift_fraction: f64,

    /// Don't POST scaling policy updates; only compute scores/weights
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Output file for computed weights and telemetry snapshot
    #[arg(
        long,
        value_name = "PATH",
        default_value = "runtime-control-weights.json"
    )]
    weights_file: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PerformanceProfile {
    /// Automatically select parameters based on model size and system resources (default)
    Auto,
    Standard,
    Balanced,
    Throughput,
    Latency,
}

impl PerformanceProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Standard => "standard",
            Self::Balanced => "balanced",
            Self::Throughput => "throughput",
            Self::Latency => "latency",
        }
    }
}

#[derive(Debug, Default)]
struct AppliedPerformanceTuning {
    batch_size: Option<usize>,
    transport: Option<String>,
    scheduler_queue_size: Option<usize>,
    scheduler_max_micro_batch: Option<usize>,
    scheduler_queue_delay_ms: Option<u64>,
    media_preprocess: Option<String>,
    rust_log: Option<String>,
    /// Populated when Auto profile is used; emitted after env_logger::init().
    auto_tune_rationale: Option<String>,
}

const DEFAULT_REMOTE_URL: &str = "https://api.kapsl.net/v1";
const REMOTE_PLACEHOLDER_URL: &str = "https://placeholder-kapsl-registry.example.com/v1";
const REMOTE_PLACEHOLDER_DIR: &str = ".kapsl-remote-placeholder";
const EXTENSION_MARKETPLACE_URL: &str = "https://api.kapsl.net/api/v1/extensions/marketplace";
const API_TOKEN_ENV: &str = "KAPSL_API_TOKEN";
const API_READER_TOKEN_ENV: &str = "KAPSL_API_TOKEN_READER";
const API_WRITER_TOKEN_ENV: &str = "KAPSL_API_TOKEN_WRITER";
const API_ADMIN_TOKEN_ENV: &str = "KAPSL_API_TOKEN_ADMIN";
const AUTH_STORE_PATH_ENV: &str = "KAPSL_AUTH_STORE_PATH";
const DEFAULT_AUTH_STORE_FILENAME: &str = "auth-store.json";
const LOG_SENSITIVE_IDS_ENV: &str = "KAPSL_LOG_SENSITIVE_IDS";
const RAG_STORAGE_ROOT_ENV: &str = "KAPSL_RAG_STORAGE_ROOT";
const REMOTE_URL_ENV: &str = "KAPSL_REMOTE_URL";
const REMOTE_TOKEN_ENV: &str = "KAPSL_REMOTE_TOKEN";
const REMOTE_TOKEN_STORE_PATH_ENV: &str = "KAPSL_REMOTE_TOKEN_STORE_PATH";
const REMOTE_PLACEHOLDER_URL_ENV: &str = "KAPSL_REMOTE_PLACEHOLDER_URL";
const REMOTE_PLACEHOLDER_DIR_ENV: &str = "KAPSL_REMOTE_PLACEHOLDER_DIR";
const EXTENSION_MARKETPLACE_URL_ENV: &str = "KAPSL_EXTENSION_MARKETPLACE_URL";
const ALLOW_INSECURE_HTTP_ENV: &str = "KAPSL_ALLOW_INSECURE_HTTP";
const OCI_REMOTE_PREFIX: &str = "oci://";
const KAPSL_OCI_ARTIFACT_TYPE: &str = "application/vnd.kapsl.aimod.v1";
const KAPSL_OCI_LAYER_TYPE: &str = "application/vnd.kapsl.aimod.v1";
const KAPSL_OCI_CONFIG_TYPE: &str = "application/vnd.kapsl.aimod.config.v1+json";
const ORAS_BIN_ENV: &str = "KAPSL_ORAS_BIN";
const OCI_USERNAME_ENV: &str = "KAPSL_OCI_USERNAME";
const OCI_PASSWORD_ENV: &str = "KAPSL_OCI_PASSWORD";
const LLM_ISOLATE_PROCESS_ENV: &str = "KAPSL_LLM_ISOLATE_PROCESS";
const LLM_ALLOW_SCHEDULER_MICROBATCH_ENV: &str = "KAPSL_LLM_ALLOW_SCHEDULER_MICROBATCH";
const PROVIDER_POLICY_ENV: &str = "KAPSL_PROVIDER_POLICY";
const EXTENSIONS_ROOT_ENV: &str = "KAPSL_EXTENSIONS_ROOT";
const EXT_CONFIG_ROOT_ENV: &str = "KAPSL_EXT_CONFIG_ROOT";
const SCHEDULER_QUEUE_OVERFLOW_POLICY_ENV: &str = "KAPSL_SCHEDULER_QUEUE_OVERFLOW_POLICY";
const LEGACY_SCHEDULER_QUEUE_OVERFLOW_POLICY_ENV: &str = "KAPSL_LITE_INGRESS_BACKPRESSURE";
const INTER_MODEL_ROUTES_ENV: &str = "KAPSL_INTER_MODEL_ROUTES";
const LEGACY_INTER_MODEL_ROUTES_ENV: &str = "KAPSL_LITE_INTER_MODEL_ROUTES";
const INTER_MODEL_RELAY_MIN_INTERVAL_MS_ENV: &str = "KAPSL_INTER_MODEL_RELAY_MIN_INTERVAL_MS";
const LEGACY_INTER_MODEL_RELAY_MIN_INTERVAL_MS_ENV: &str =
    "KAPSL_LITE_INTER_MODEL_RELAY_MIN_INTERVAL_MS";
const INTER_MODEL_RELAY_SESSION_PREFIX: &str = "relay/";
const DEFAULT_INTER_MODEL_RELAY_MIN_INTERVAL_MS: u64 = 2000;
const PRESSURE_MEMORY_CONSERVE_PCT_ENV: &str = "KAPSL_SERVER_PRESSURE_MEMORY_CONSERVE_PCT";
const PRESSURE_MEMORY_EMERGENCY_PCT_ENV: &str = "KAPSL_SERVER_PRESSURE_MEMORY_EMERGENCY_PCT";
const PRESSURE_GPU_UTIL_CONSERVE_PCT_ENV: &str = "KAPSL_SERVER_PRESSURE_GPU_UTIL_CONSERVE_PCT";
const PRESSURE_GPU_UTIL_EMERGENCY_PCT_ENV: &str = "KAPSL_SERVER_PRESSURE_GPU_UTIL_EMERGENCY_PCT";
const PRESSURE_GPU_MEM_CONSERVE_PCT_ENV: &str = "KAPSL_SERVER_PRESSURE_GPU_MEM_CONSERVE_PCT";
const PRESSURE_GPU_MEM_EMERGENCY_PCT_ENV: &str = "KAPSL_SERVER_PRESSURE_GPU_MEM_EMERGENCY_PCT";
const PRESSURE_CONSERVE_MAX_TOKENS_ENV: &str = "KAPSL_SERVER_PRESSURE_CONSERVE_MAX_NEW_TOKENS";
const PRESSURE_EMERGENCY_MAX_TOKENS_ENV: &str = "KAPSL_SERVER_PRESSURE_EMERGENCY_MAX_NEW_TOKENS";
const RAG_DEFAULT_TOP_K: usize = 4;
const RAG_MAX_TOP_K: usize = 32;
const RAG_EMBEDDING_DIM: usize = 256;
const RAG_CHUNK_SIZE: usize = 200;
const RAG_CHUNK_OVERLAP: usize = 40;
const RAG_CONTEXT_MAX_TOKENS: usize = 768;

#[derive(Debug)]
struct ApiUnauthorized;

impl warp::reject::Reject for ApiUnauthorized {}

#[derive(Debug)]
struct ApiForbidden;

impl warp::reject::Reject for ApiForbidden {}

#[derive(Debug)]
struct ApiLocalOnly;

impl warp::reject::Reject for ApiLocalOnly {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ApiRole {
    Reader,
    Writer,
    Admin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApiScope {
    Read,
    Write,
    Admin,
}

impl ApiRole {
    fn allows(self, required: ApiRole) -> bool {
        use ApiRole::{Admin, Reader, Writer};
        matches!(
            (self, required),
            (Admin, _) | (Writer, Reader) | (Writer, Writer) | (Reader, Reader)
        )
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ApiRoleTokenConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    reader_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    writer_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    admin_token: Option<String>,
}

impl ApiRoleTokenConfig {
    fn normalize_token(value: Option<String>) -> Option<String> {
        value.and_then(|raw| {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
    }

    fn from_env() -> Self {
        let shared_token = optional_env_var(API_TOKEN_ENV);
        Self {
            reader_token: optional_env_var(API_READER_TOKEN_ENV).or(shared_token.clone()),
            writer_token: optional_env_var(API_WRITER_TOKEN_ENV).or(shared_token.clone()),
            admin_token: optional_env_var(API_ADMIN_TOKEN_ENV).or(shared_token),
        }
    }

    fn auth_enabled(&self) -> bool {
        self.reader_token.is_some() || self.writer_token.is_some() || self.admin_token.is_some()
    }

    fn role_for_token(&self, presented_token: &str) -> Option<ApiRole> {
        if self
            .admin_token
            .as_deref()
            .is_some_and(|token| authorization_matches_token(Some(presented_token), token))
        {
            return Some(ApiRole::Admin);
        }
        if self
            .writer_token
            .as_deref()
            .is_some_and(|token| authorization_matches_token(Some(presented_token), token))
        {
            return Some(ApiRole::Writer);
        }
        if self
            .reader_token
            .as_deref()
            .is_some_and(|token| authorization_matches_token(Some(presented_token), token))
        {
            return Some(ApiRole::Reader);
        }
        None
    }

    fn role_from_authorization_header(&self, authorization: Option<&str>) -> Option<ApiRole> {
        let raw_header = authorization?;
        let trimmed = raw_header.trim();
        if trimmed.is_empty() {
            return None;
        }
        if let Some((scheme, token)) = trimmed.split_once(' ') {
            if scheme.eq_ignore_ascii_case("bearer") {
                return self.role_for_token(token.trim());
            }
        }
        self.role_for_token(trimmed)
    }

    fn update_from_payload(&mut self, payload: ApiRoleTokenConfig) -> Result<(), String> {
        self.reader_token = Self::normalize_token(payload.reader_token);
        self.writer_token = Self::normalize_token(payload.writer_token);
        self.admin_token = Self::normalize_token(payload.admin_token);
        if self.auth_enabled() && self.admin_token.is_none() {
            return Err("admin_token is required when role auth is enabled".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum ApiUserStatus {
    #[default]
    Active,
    Suspended,
}

impl ApiUserStatus {
    fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiAuthUser {
    id: String,
    username: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    display_name: Option<String>,
    role: ApiRole,
    #[serde(default)]
    status: ApiUserStatus,
    created_at: u64,
    updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiAuthKey {
    id: String,
    user_id: String,
    name: String,
    key_prefix: String,
    key_hash: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    scopes: Vec<String>,
    created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_used_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    revoked_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ApiAuthStoreFile {
    #[serde(default)]
    users: Vec<ApiAuthUser>,
    #[serde(default)]
    api_keys: Vec<ApiAuthKey>,
}

impl ApiAuthStoreFile {
    fn load(path: &Path) -> Self {
        let Ok(raw) = fs::read_to_string(path) else {
            return Self::default();
        };
        match serde_json::from_str::<Self>(&raw) {
            Ok(parsed) => parsed,
            Err(error) => {
                log::warn!(
                    "Failed to parse auth store file {}: {}. Starting with empty store.",
                    path.display(),
                    error
                );
                Self::default()
            }
        }
    }
}

#[derive(Debug, Serialize)]
struct ApiAuthStatusResponse {
    auth_enabled: bool,
    legacy_auth_enabled: bool,
    store_path: String,
    user_count: usize,
    key_count: usize,
    active_key_count: usize,
    active_admin_key_count: usize,
}

#[derive(Debug, Serialize)]
struct ApiRoleSummary {
    role: ApiRole,
    description: String,
    user_count: usize,
    active_key_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ApiAuthUserView {
    id: String,
    username: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    display_name: Option<String>,
    role: ApiRole,
    status: ApiUserStatus,
    created_at: u64,
    updated_at: u64,
    key_count: usize,
    active_key_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ApiAuthKeyView {
    id: String,
    user_id: String,
    username: String,
    user_role: ApiRole,
    name: String,
    key_prefix: String,
    scopes: Vec<String>,
    created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_used_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    revoked_at: Option<u64>,
    active: bool,
}

#[derive(Debug, Deserialize)]
struct CreateAuthUserRequest {
    username: String,
    #[serde(default)]
    display_name: Option<String>,
    role: ApiRole,
    #[serde(default)]
    status: Option<ApiUserStatus>,
}

#[derive(Debug, Deserialize)]
struct UpdateAuthUserRequest {
    #[serde(default)]
    display_name: Option<Option<String>>,
    #[serde(default)]
    role: Option<ApiRole>,
    #[serde(default)]
    status: Option<ApiUserStatus>,
}

#[derive(Debug, Deserialize)]
struct CreateApiKeyRequest {
    name: String,
    #[serde(default)]
    scopes: Option<Vec<String>>,
    #[serde(default)]
    expires_in_days: Option<u32>,
}

#[derive(Debug, Serialize)]
struct CreateApiKeyResponse {
    api_key: ApiAuthKeyView,
    raw_key: String,
}

#[derive(Debug, Deserialize, Default)]
struct ApiAuthLoginRequest {
    #[serde(default)]
    token: Option<String>,
}

#[derive(Debug, Serialize)]
struct ApiAuthLoginAccess {
    read: bool,
    write: bool,
    admin: bool,
}

#[derive(Debug, Serialize)]
struct ApiAuthLoginResponse {
    authenticated: bool,
    auth_enabled: bool,
    legacy_auth_enabled: bool,
    role: ApiRole,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    scopes: Vec<String>,
    mode: String,
    access: ApiAuthLoginAccess,
}

#[derive(Debug)]
struct ApiAuthState {
    legacy_tokens: ApiRoleTokenConfig,
    store_path: PathBuf,
    store: ApiAuthStoreFile,
    key_hash_index: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
struct ApiAuthGrant {
    role: ApiRole,
    scopes: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
struct ApiAuthGrantMatch {
    grant: ApiAuthGrant,
    matched_key_index: Option<usize>,
}

impl ApiAuthState {
    fn from_store_path(store_path: PathBuf) -> Self {
        let legacy_tokens = ApiRoleTokenConfig::from_env();
        let store = ApiAuthStoreFile::load(&store_path);
        let mut state = Self {
            legacy_tokens,
            key_hash_index: Self::build_key_hash_index(&store),
            store,
            store_path,
        };
        if state.store.users.is_empty() {
            state.seed_default_users();
            if let Err(error) = state.save_store() {
                log::warn!("Failed to persist default auth users: {}", error);
            }
        }
        state
    }

    fn build_key_hash_index(store: &ApiAuthStoreFile) -> HashMap<String, usize> {
        let mut index = HashMap::with_capacity(store.api_keys.len());
        for (position, key) in store.api_keys.iter().enumerate() {
            if index.insert(key.key_hash.clone(), position).is_some() {
                log::warn!(
                    "Duplicate key hash detected in auth store; latest entry will be used for lookup"
                );
            }
        }
        index
    }

    fn save_store(&self) -> Result<(), String> {
        if let Some(parent) = self.store_path.parent() {
            fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "failed to create auth store directory {}: {}",
                    parent.display(),
                    error
                )
            })?;
        }
        let serialized = serde_json::to_string_pretty(&self.store)
            .map_err(|error| format!("failed to serialize auth store: {}", error))?;
        let tmp_path = self.store_path.with_extension("tmp");
        fs::write(&tmp_path, serialized).map_err(|error| {
            format!(
                "failed to write auth store temp file {}: {}",
                tmp_path.display(),
                error
            )
        })?;
        if self.store_path.exists() {
            fs::remove_file(&self.store_path).map_err(|error| {
                format!(
                    "failed to replace existing auth store file {}: {}",
                    self.store_path.display(),
                    error
                )
            })?;
        }
        fs::rename(&tmp_path, &self.store_path).map_err(|error| {
            format!(
                "failed to replace auth store file {}: {}",
                self.store_path.display(),
                error
            )
        })?;
        Ok(())
    }

    fn seed_default_users(&mut self) {
        let now = now_unix_seconds();
        self.store.users = vec![
            ApiAuthUser {
                id: generate_random_id("usr"),
                username: "admin".to_string(),
                display_name: Some("Default Admin".to_string()),
                role: ApiRole::Admin,
                status: ApiUserStatus::Active,
                created_at: now,
                updated_at: now,
            },
            ApiAuthUser {
                id: generate_random_id("usr"),
                username: "operator".to_string(),
                display_name: Some("Runtime Operator".to_string()),
                role: ApiRole::Writer,
                status: ApiUserStatus::Active,
                created_at: now,
                updated_at: now,
            },
            ApiAuthUser {
                id: generate_random_id("usr"),
                username: "viewer".to_string(),
                display_name: Some("Read-Only Viewer".to_string()),
                role: ApiRole::Reader,
                status: ApiUserStatus::Active,
                created_at: now,
                updated_at: now,
            },
        ];
    }

    fn auth_enabled(&self) -> bool {
        self.legacy_tokens.auth_enabled() || self.active_key_count() > 0
    }

    fn active_key_count(&self) -> usize {
        let now = now_unix_seconds();
        self.store
            .api_keys
            .iter()
            .filter(|key| {
                self.user_by_id(&key.user_id)
                    .is_some_and(|user| Self::is_key_active_for_user(key, user, now))
            })
            .count()
    }

    fn active_admin_key_count(&self) -> usize {
        let now = now_unix_seconds();
        self.store
            .api_keys
            .iter()
            .filter(|key| {
                self.user_by_id(&key.user_id).is_some_and(|user| {
                    user.role == ApiRole::Admin && Self::is_key_active_for_user(key, user, now)
                })
            })
            .count()
    }

    fn active_key_count_for_user(&self, user_id: &str) -> usize {
        let now = now_unix_seconds();
        self.store
            .api_keys
            .iter()
            .filter(|key| {
                if key.user_id != user_id {
                    return false;
                }
                self.user_by_id(&key.user_id)
                    .is_some_and(|user| Self::is_key_active_for_user(key, user, now))
            })
            .count()
    }

    fn is_key_active_for_user(key: &ApiAuthKey, user: &ApiAuthUser, now: u64) -> bool {
        if !user.status.is_active() || key.revoked_at.is_some() {
            return false;
        }
        key.expires_at.is_none_or(|expiry| expiry > now)
    }

    fn user_by_id(&self, user_id: &str) -> Option<&ApiAuthUser> {
        self.store.users.iter().find(|user| user.id == user_id)
    }

    fn grant_from_authorization_header_read(
        &self,
        authorization: Option<&str>,
    ) -> Option<ApiAuthGrantMatch> {
        let presented = parse_authorization_token(authorization)?;
        if let Some((role, scopes, key_index)) = self.grant_for_api_key_token_read(presented) {
            return Some(ApiAuthGrantMatch {
                grant: ApiAuthGrant {
                    role,
                    scopes: Some(scopes),
                },
                matched_key_index: Some(key_index),
            });
        }
        self.legacy_tokens
            .role_from_authorization_header(authorization)
            .map(|role| ApiAuthGrantMatch {
                grant: ApiAuthGrant { role, scopes: None },
                matched_key_index: None,
            })
    }

    #[cfg(test)]
    fn role_from_authorization_header(&mut self, authorization: Option<&str>) -> Option<ApiRole> {
        self.grant_from_authorization_header_read(authorization)
            .map(|matched| matched.grant.role)
    }

    fn grant_for_api_key_token_read(
        &self,
        presented_token: &str,
    ) -> Option<(ApiRole, Vec<String>, usize)> {
        let token_hash = sha256_hex(presented_token);
        let key_index = self.key_hash_index.get(&token_hash).copied()?;
        let key = self.store.api_keys.get(key_index)?;
        if !constant_time_eq(&key.key_hash, &token_hash) {
            return None;
        }

        let now = now_unix_seconds();
        let user = self.user_by_id(&key.user_id)?;
        let role = user.role;
        let scopes = key.scopes.clone();
        let is_active = Self::is_key_active_for_user(key, user, now);
        if !is_active {
            return None;
        }
        Some((role, scopes, key_index))
    }

    fn touch_key_last_used_by_index(&mut self, key_index: usize, now: u64) {
        if let Some(key) = self.store.api_keys.get_mut(key_index) {
            if key.last_used_at != Some(now) {
                key.last_used_at = Some(now);
            }
        }
    }

    fn legacy_token_config(&self) -> ApiRoleTokenConfig {
        self.legacy_tokens.clone()
    }

    fn update_legacy_token_config(
        &mut self,
        payload: ApiRoleTokenConfig,
    ) -> Result<ApiRoleTokenConfig, String> {
        self.legacy_tokens.update_from_payload(payload)?;
        Ok(self.legacy_tokens.clone())
    }

    fn status_response(&self) -> ApiAuthStatusResponse {
        ApiAuthStatusResponse {
            auth_enabled: self.auth_enabled(),
            legacy_auth_enabled: self.legacy_tokens.auth_enabled(),
            store_path: self.store_path.to_string_lossy().to_string(),
            user_count: self.store.users.len(),
            key_count: self.store.api_keys.len(),
            active_key_count: self.active_key_count(),
            active_admin_key_count: self.active_admin_key_count(),
        }
    }

    fn role_summaries(&self) -> Vec<ApiRoleSummary> {
        [ApiRole::Admin, ApiRole::Writer, ApiRole::Reader]
            .iter()
            .copied()
            .map(|role| {
                let user_count = self
                    .store
                    .users
                    .iter()
                    .filter(|user| user.role == role)
                    .count();
                let now = now_unix_seconds();
                let active_key_count = self
                    .store
                    .api_keys
                    .iter()
                    .filter(|key| {
                        self.user_by_id(&key.user_id).is_some_and(|user| {
                            user.role == role && Self::is_key_active_for_user(key, user, now)
                        })
                    })
                    .count();
                ApiRoleSummary {
                    role,
                    description: role_description(role).to_string(),
                    user_count,
                    active_key_count,
                }
            })
            .collect()
    }

    fn list_users(&self) -> Vec<ApiAuthUserView> {
        self.store
            .users
            .iter()
            .map(|user| self.user_view(user))
            .collect()
    }

    fn user_view(&self, user: &ApiAuthUser) -> ApiAuthUserView {
        let key_count = self
            .store
            .api_keys
            .iter()
            .filter(|key| key.user_id == user.id)
            .count();
        let active_key_count = self.active_key_count_for_user(&user.id);
        ApiAuthUserView {
            id: user.id.clone(),
            username: user.username.clone(),
            display_name: user.display_name.clone(),
            role: user.role,
            status: user.status,
            created_at: user.created_at,
            updated_at: user.updated_at,
            key_count,
            active_key_count,
        }
    }

    fn list_keys(&self, user_id: Option<&str>) -> Vec<ApiAuthKeyView> {
        let now = now_unix_seconds();
        let mut keys = self
            .store
            .api_keys
            .iter()
            .filter(|key| user_id.is_none_or(|expected| expected == key.user_id))
            .filter_map(|key| {
                let user = self.user_by_id(&key.user_id)?;
                Some(self.key_view(key, user, now))
            })
            .collect::<Vec<_>>();
        keys.sort_by(|left, right| right.created_at.cmp(&left.created_at));
        keys
    }

    fn key_view(&self, key: &ApiAuthKey, user: &ApiAuthUser, now: u64) -> ApiAuthKeyView {
        ApiAuthKeyView {
            id: key.id.clone(),
            user_id: key.user_id.clone(),
            username: user.username.clone(),
            user_role: user.role,
            name: key.name.clone(),
            key_prefix: key.key_prefix.clone(),
            scopes: key.scopes.clone(),
            created_at: key.created_at,
            created_by: key.created_by.clone(),
            last_used_at: key.last_used_at,
            expires_at: key.expires_at,
            revoked_at: key.revoked_at,
            active: Self::is_key_active_for_user(key, user, now),
        }
    }

    fn create_user(&mut self, request: CreateAuthUserRequest) -> Result<ApiAuthUserView, String> {
        let username = normalize_username(&request.username)?;
        if self
            .store
            .users
            .iter()
            .any(|user| user.username.eq_ignore_ascii_case(&username))
        {
            return Err(format!("user `{}` already exists", username));
        }

        let now = now_unix_seconds();
        let user = ApiAuthUser {
            id: generate_random_id("usr"),
            username,
            display_name: normalize_optional_text(request.display_name),
            role: request.role,
            status: request.status.unwrap_or(ApiUserStatus::Active),
            created_at: now,
            updated_at: now,
        };
        let user_id = user.id.clone();
        self.store.users.push(user);
        self.save_store()?;
        let created = self
            .store
            .users
            .iter()
            .find(|user| user.id == user_id)
            .ok_or_else(|| "failed to load created user".to_string())?;
        Ok(self.user_view(created))
    }

    fn update_user(
        &mut self,
        user_id: &str,
        request: UpdateAuthUserRequest,
    ) -> Result<ApiAuthUserView, String> {
        let user_index = self
            .store
            .users
            .iter()
            .position(|user| user.id == user_id)
            .ok_or_else(|| format!("user `{}` not found", user_id))?;

        let mut updated_user = self.store.users[user_index].clone();
        if let Some(display_name) = request.display_name {
            updated_user.display_name = normalize_optional_text(display_name);
        }
        if let Some(new_role) = request.role {
            if updated_user.role == ApiRole::Admin
                && new_role != ApiRole::Admin
                && self.active_key_count_for_user(&updated_user.id) > 0
                && self.active_admin_key_count() <= self.active_key_count_for_user(&updated_user.id)
                && self.active_key_count() > self.active_key_count_for_user(&updated_user.id)
            {
                return Err(
                    "cannot remove admin role from the last admin with active API keys".to_string(),
                );
            }
            updated_user.role = new_role;
        }
        if let Some(new_status) = request.status {
            updated_user.status = new_status;
        }
        updated_user.updated_at = now_unix_seconds();
        self.store.users[user_index] = updated_user.clone();
        self.save_store()?;
        Ok(self.user_view(&updated_user))
    }

    fn create_api_key(
        &mut self,
        user_id: &str,
        request: CreateApiKeyRequest,
    ) -> Result<CreateApiKeyResponse, String> {
        let user = self
            .user_by_id(user_id)
            .cloned()
            .ok_or_else(|| format!("user `{}` not found", user_id))?;
        if !user.status.is_active() {
            return Err("cannot create API key for a suspended user".to_string());
        }
        if self.active_key_count() == 0 && user.role != ApiRole::Admin {
            return Err("first API key must belong to an admin user".to_string());
        }

        let name = normalize_required_text(&request.name, "name")?;
        let expires_at = match request.expires_in_days {
            Some(0) => return Err("expires_in_days must be greater than 0".to_string()),
            Some(days) => Some(now_unix_seconds() + (days as u64 * 86_400)),
            None => None,
        };
        let scopes = normalize_scopes(request.scopes);
        let raw_key = generate_api_key();
        let key_hash = sha256_hex(&raw_key);
        let key_prefix: String = raw_key.chars().take(12).collect();
        if self.key_hash_index.contains_key(&key_hash) {
            return Err("generated API key collided, retry".to_string());
        }

        let now = now_unix_seconds();
        let key_hash_for_index = key_hash.clone();
        let key = ApiAuthKey {
            id: generate_random_id("key"),
            user_id: user.id.clone(),
            name,
            key_prefix,
            key_hash,
            scopes,
            created_at: now,
            created_by: None,
            last_used_at: None,
            expires_at,
            revoked_at: None,
        };
        let key_id = key.id.clone();
        let key_index = self.store.api_keys.len();
        self.store.api_keys.push(key);
        self.key_hash_index.insert(key_hash_for_index, key_index);
        self.save_store()?;
        let created_key = self
            .store
            .api_keys
            .iter()
            .find(|existing| existing.id == key_id)
            .ok_or_else(|| "failed to load created API key".to_string())?;
        let view = self.key_view(created_key, &user, now);
        Ok(CreateApiKeyResponse {
            api_key: view,
            raw_key,
        })
    }

    fn revoke_api_key(&mut self, key_id: &str) -> Result<ApiAuthKeyView, String> {
        let key_index = self
            .store
            .api_keys
            .iter()
            .position(|key| key.id == key_id)
            .ok_or_else(|| format!("api key `{}` not found", key_id))?;
        let now = now_unix_seconds();
        let key = self.store.api_keys[key_index].clone();
        let user = self
            .user_by_id(&key.user_id)
            .cloned()
            .ok_or_else(|| format!("user `{}` not found for key", key.user_id))?;
        let key_active = Self::is_key_active_for_user(&key, &user, now);
        if key_active
            && user.role == ApiRole::Admin
            && self.active_admin_key_count() <= 1
            && self.active_key_count() > 1
        {
            return Err(
                "cannot revoke the last active admin key while other keys remain active"
                    .to_string(),
            );
        }

        if self.store.api_keys[key_index].revoked_at.is_none() {
            self.store.api_keys[key_index].revoked_at = Some(now);
            self.save_store()?;
        }
        let updated = self.store.api_keys[key_index].clone();
        Ok(self.key_view(&updated, &user, now))
    }
}

fn role_description(role: ApiRole) -> &'static str {
    match role {
        ApiRole::Admin => "Full control of runtime and access management",
        ApiRole::Writer => "Can modify runtime state and extensions",
        ApiRole::Reader => "Read-only runtime access",
    }
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn resolve_auth_store_path() -> PathBuf {
    if let Some(path) = optional_env_var(AUTH_STORE_PATH_ENV) {
        return PathBuf::from(path);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".kapsl")
            .join(DEFAULT_AUTH_STORE_FILENAME);
    }
    if let Some(profile) = std::env::var_os("USERPROFILE") {
        return PathBuf::from(profile)
            .join(".kapsl")
            .join(DEFAULT_AUTH_STORE_FILENAME);
    }
    PathBuf::from(format!(".{}", DEFAULT_AUTH_STORE_FILENAME))
}

fn parse_authorization_token(header_value: Option<&str>) -> Option<&str> {
    let raw_header = header_value?;
    let trimmed = raw_header.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some((scheme, token)) = trimmed.split_once(' ') {
        if scheme.eq_ignore_ascii_case("bearer") {
            let parsed = token.trim();
            if parsed.is_empty() {
                return None;
            }
            return Some(parsed);
        }
    }
    Some(trimmed)
}

fn normalize_required_text(value: &str, field: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(format!("{} is required", field))
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalize_optional_text(value: Option<String>) -> Option<String> {
    value.and_then(|raw| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn normalize_username(username: &str) -> Result<String, String> {
    let normalized = normalize_required_text(username, "username")?;
    Ok(normalized.to_ascii_lowercase())
}

fn normalize_scopes(scopes: Option<Vec<String>>) -> Vec<String> {
    let mut seen = HashSet::new();
    scopes
        .unwrap_or_default()
        .into_iter()
        .filter_map(|scope| {
            let trimmed = scope.trim();
            if trimmed.is_empty() {
                None
            } else {
                let normalized = trimmed.to_string();
                if seen.insert(normalized.clone()) {
                    Some(normalized)
                } else {
                    None
                }
            }
        })
        .collect()
}

fn scope_token_allows(scope: &str, required: ApiScope) -> bool {
    let normalized = scope.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return false;
    }
    if normalized == "*" || normalized == "*:*" {
        return true;
    }

    match required {
        ApiScope::Read => matches!(
            normalized.as_str(),
            "api:read" | "read" | "api:write" | "write" | "api:admin" | "admin" | "api:*"
        ),
        ApiScope::Write => {
            matches!(
                normalized.as_str(),
                "api:write" | "write" | "api:admin" | "admin" | "api:*"
            )
        }
        ApiScope::Admin => matches!(normalized.as_str(), "api:admin" | "admin" | "api:*"),
    }
}

fn key_scopes_allow(scopes: &[String], required: ApiScope) -> bool {
    // Backward compatibility: empty scopes behave like unrestricted role-based keys.
    scopes.is_empty()
        || scopes
            .iter()
            .any(|scope| scope_token_allows(scope, required))
}

fn is_loopback_remote(remote: Option<std::net::SocketAddr>) -> bool {
    remote.is_some_and(|addr| addr.ip().is_loopback())
}

fn generate_random_id(prefix: &str) -> String {
    let mut bytes = [0u8; 8];
    OsRng.fill_bytes(&mut bytes);
    let mut suffix = String::with_capacity(16);
    for byte in bytes {
        suffix.push_str(&format!("{:02x}", byte));
    }
    format!("{}_{}", prefix, suffix)
}

fn generate_api_key() -> String {
    let mut bytes = [0u8; 24];
    OsRng.fill_bytes(&mut bytes);
    let secret = BASE64_URL_SAFE_NO_PAD.encode(bytes);
    format!("kpsl_{}", secret)
}

fn sha256_hex(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push_str(&format!("{:02x}", byte));
    }
    output
}

#[derive(Debug, Deserialize)]
struct PackageKapslRequest {
    model_path: String,
    output_path: Option<String>,
    project_name: Option<String>,
    framework: Option<String>,
    version: Option<String>,
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct PackageKapslResponse {
    status: String,
    kapsl_path: String,
    project_name: String,
    framework: String,
    version: String,
}

#[derive(Debug, Deserialize)]
struct PushKapslRequest {
    kapsl_path: String,
    target: String,
    remote_url: Option<String>,
    remote_token: Option<String>,
    #[serde(default)]
    interactive_login: bool,
}

#[derive(Debug, Serialize)]
struct PushKapslResponse {
    status: String,
    remote_url: String,
    artifact_url: String,
    mirrored_path: String,
    bytes_uploaded: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    manifest_digest: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PullKapslRequest {
    target: String,
    reference: Option<String>,
    destination_dir: Option<String>,
    remote_url: Option<String>,
    remote_token: Option<String>,
    #[serde(default)]
    interactive_login: bool,
}

#[derive(Debug, Serialize)]
struct PullKapslResponse {
    status: String,
    remote_url: String,
    artifact_url: String,
    kapsl_path: String,
    bytes_downloaded: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct RemoteTokenStoreFile {
    #[serde(default)]
    tokens_by_remote: HashMap<String, String>,
    #[serde(default)]
    last_remote_url: Option<String>,
}

#[derive(Debug, Serialize)]
struct LoginResponse {
    status: String,
    remote_url: String,
    auth_base_url: String,
    provider: String,
    login_method: String,
    callback_url: String,
    token_store_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    verification_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_code: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeviceCodeStartResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    verification_uri_complete: Option<String>,
    expires_in: Option<u64>,
    interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct DeviceCodePollResponse {
    status: String,
    token: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
    interval: Option<u64>,
}

#[derive(Clone)]
struct RagRuntimeState {
    vector_store: Arc<SqliteVectorStore>,
    doc_store: FsDocStore,
}

#[derive(Debug, Deserialize)]
struct SyncExtensionRequest {
    workspace_id: String,
    source_id: Option<String>,
    cursor: Option<String>,
    tenant_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RagQueryRequest {
    workspace_id: String,
    query: String,
    source_id: Option<String>,
    source_ids: Option<Vec<String>>,
    top_k: Option<usize>,
    min_score: Option<f32>,
    tenant_id: Option<String>,
    #[serde(default)]
    allowed_users: Vec<String>,
    #[serde(default)]
    allowed_groups: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct InferRagOptions {
    enabled: Option<bool>,
    workspace_id: String,
    source_id: Option<String>,
    source_ids: Option<Vec<String>>,
    top_k: Option<usize>,
    min_score: Option<f32>,
    tenant_id: Option<String>,
    max_context_tokens: Option<usize>,
    max_chunks: Option<usize>,
    max_per_source: Option<usize>,
}

#[derive(Debug)]
enum RagAugmentError {
    BadRequest(String),
    Internal(String),
}

impl RagAugmentError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}

fn optional_env_var(name: &str) -> Option<String> {
    let value = std::env::var(name).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn optional_env_var_alias(primary: &str, legacy: &str) -> Option<String> {
    optional_env_var(primary).or_else(|| optional_env_var(legacy))
}

fn arg_user_supplied(matches: &ArgMatches, arg: &str) -> bool {
    !matches!(
        matches.value_source(arg),
        None | Some(ValueSource::DefaultValue)
    )
}

struct AutoTunedPolicy {
    batch_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    scheduler_queue_size: usize,
    rationale: String,
}

/// Derive scheduler parameters from model file size and available system resources.
///
/// When multiple models are loaded, sizes based on the largest (conservative).
/// Falls back to safe defaults if model paths can't be stat'd.
fn auto_tune_policy(model_paths: &[PathBuf]) -> AutoTunedPolicy {
    // Largest model file size in MB — proxy for parameter count / memory footprint
    let model_size_mb: u64 = model_paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .max()
        .map(|b| b / (1024 * 1024))
        .unwrap_or(0);

    // Available (not total) system RAM in GB
    let available_ram_gb: u64 = {
        let mut sys = System::new();
        sys.refresh_memory();
        sys.available_memory() / (1024 * 1024 * 1024)
    };

    // Logical CPU cores
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // Base policy from model file size
    let (mut batch_size, mut micro_batch, delay_ms, mut queue_size, size_tier) =
        if model_size_mb == 0 {
            // Unknown model size should remain deterministic across environments.
            // Keep the conservative Standard-equivalent defaults unchanged rather
            // than layering in host-dependent RAM/CPU reductions.
            return AutoTunedPolicy {
                batch_size: 4,
                scheduler_max_micro_batch: 4,
                scheduler_queue_delay_ms: 2,
                scheduler_queue_size: 256,
                rationale: format!(
                    "model={}MB (unknown), ram_avail={}GB, cpu_cores={}, conservative-defaults",
                    model_size_mb, available_ram_gb, cpu_cores
                ),
            };
        } else if model_size_mb < 500 {
            (16, 16, 6, 2048, "tiny (<500 MB)")
        } else if model_size_mb < 2_000 {
            (8, 8, 3, 512, "small (500 MB-2 GB)")
        } else if model_size_mb < 8_000 {
            (4, 4, 2, 256, "medium (2-8 GB)")
        } else {
            (2, 2, 1, 128, "large (>=8 GB)")
        };

    let mut notes = String::new();

    // Memory pressure: halve batch and queue if available RAM is tight
    if available_ram_gb > 0 && available_ram_gb < 4 {
        batch_size = (batch_size / 2).max(1);
        queue_size = (queue_size / 2).max(64);
        notes.push_str(&format!(", low-ram (avail={}GB)", available_ram_gb));
    }

    // CPU constraint: halve micro_batch on very low-core systems
    if cpu_cores <= 2 {
        micro_batch = (micro_batch / 2).max(1);
        notes.push_str(&format!(", low-cpu (cores={})", cpu_cores));
    }

    AutoTunedPolicy {
        batch_size,
        scheduler_max_micro_batch: micro_batch,
        scheduler_queue_delay_ms: delay_ms,
        scheduler_queue_size: queue_size,
        rationale: format!(
            "model={}MB ({}), ram_avail={}GB, cpu_cores={}{}",
            model_size_mb, size_tier, available_ram_gb, cpu_cores, notes
        ),
    }
}

fn apply_performance_profile(args: &mut Args, matches: &ArgMatches) -> AppliedPerformanceTuning {
    let mut tuning = AppliedPerformanceTuning::default();
    if args.worker || matches!(args.performance_profile, PerformanceProfile::Standard) {
        return tuning;
    }

    let batch_explicit = arg_user_supplied(matches, "batch_size");
    let transport_explicit = arg_user_supplied(matches, "transport");
    let scheduler_queue_size_explicit = arg_user_supplied(matches, "scheduler_queue_size");
    let scheduler_max_micro_batch_explicit =
        arg_user_supplied(matches, "scheduler_max_micro_batch");
    let scheduler_queue_delay_ms_explicit = arg_user_supplied(matches, "scheduler_queue_delay_ms");

    match args.performance_profile {
        PerformanceProfile::Standard => {}
        PerformanceProfile::Auto => {
            let policy = auto_tune_policy(&args.model);
            if !batch_explicit {
                args.batch_size = policy.batch_size;
                tuning.batch_size = Some(args.batch_size);
            }
            if !scheduler_queue_size_explicit {
                args.scheduler_queue_size = policy.scheduler_queue_size;
                tuning.scheduler_queue_size = Some(args.scheduler_queue_size);
            }
            if !scheduler_max_micro_batch_explicit {
                args.scheduler_max_micro_batch = policy.scheduler_max_micro_batch;
                tuning.scheduler_max_micro_batch = Some(args.scheduler_max_micro_batch);
            }
            if !scheduler_queue_delay_ms_explicit {
                args.scheduler_queue_delay_ms = policy.scheduler_queue_delay_ms;
                tuning.scheduler_queue_delay_ms = Some(args.scheduler_queue_delay_ms);
            }
            // Defer the log: env_logger is not yet initialized at this call site.
            tuning.auto_tune_rationale = Some(format!(
                "batch={}, micro_batch={}, delay={}ms, queue_size={} | {}",
                args.batch_size,
                args.scheduler_max_micro_batch,
                args.scheduler_queue_delay_ms,
                args.scheduler_queue_size,
                policy.rationale,
            ));
        }
        PerformanceProfile::Balanced => {
            if !batch_explicit {
                args.batch_size = 8;
                tuning.batch_size = Some(args.batch_size);
            }
            if !transport_explicit {
                args.transport = "hybrid".to_string();
                tuning.transport = Some(args.transport.clone());
            }
            if !scheduler_queue_size_explicit {
                args.scheduler_queue_size = 512;
                tuning.scheduler_queue_size = Some(args.scheduler_queue_size);
            }
            if !scheduler_max_micro_batch_explicit {
                args.scheduler_max_micro_batch = args.batch_size.max(1);
                tuning.scheduler_max_micro_batch = Some(args.scheduler_max_micro_batch);
            }
            if !scheduler_queue_delay_ms_explicit {
                args.scheduler_queue_delay_ms = 3;
                tuning.scheduler_queue_delay_ms = Some(args.scheduler_queue_delay_ms);
            }
        }
        PerformanceProfile::Throughput => {
            if !batch_explicit {
                args.batch_size = 16;
                tuning.batch_size = Some(args.batch_size);
            }
            if !transport_explicit {
                args.transport = "hybrid".to_string();
                tuning.transport = Some(args.transport.clone());
            }
            if !scheduler_queue_size_explicit {
                args.scheduler_queue_size = 2048;
                tuning.scheduler_queue_size = Some(args.scheduler_queue_size);
            }
            if !scheduler_max_micro_batch_explicit {
                args.scheduler_max_micro_batch = args.batch_size.max(1);
                tuning.scheduler_max_micro_batch = Some(args.scheduler_max_micro_batch);
            }
            if !scheduler_queue_delay_ms_explicit {
                args.scheduler_queue_delay_ms = 6;
                tuning.scheduler_queue_delay_ms = Some(args.scheduler_queue_delay_ms);
            }
            if std::env::var_os("RUST_LOG").is_none() {
                std::env::set_var("RUST_LOG", "warn");
                tuning.rust_log = Some("warn".to_string());
            }
        }
        PerformanceProfile::Latency => {
            if !batch_explicit {
                args.batch_size = 1;
                tuning.batch_size = Some(args.batch_size);
            }
            if !transport_explicit {
                args.transport = "socket".to_string();
                tuning.transport = Some(args.transport.clone());
            }
            if !scheduler_queue_size_explicit {
                args.scheduler_queue_size = 128;
                tuning.scheduler_queue_size = Some(args.scheduler_queue_size);
            }
            if !scheduler_max_micro_batch_explicit {
                args.scheduler_max_micro_batch = 1;
                tuning.scheduler_max_micro_batch = Some(args.scheduler_max_micro_batch);
            }
            if !scheduler_queue_delay_ms_explicit {
                args.scheduler_queue_delay_ms = 0;
                tuning.scheduler_queue_delay_ms = Some(args.scheduler_queue_delay_ms);
            }
        }
    }

    tuning
}

fn dyn_error_from_message(message: impl Into<String>) -> DynError {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.into(),
    ))
}

fn parse_runtime_args_and_matches(argv: &[String]) -> Result<(Args, ArgMatches), DynError> {
    let cmd = <Args as clap::Args>::augment_args(clap::Command::new("kapsl"));
    let matches = cmd
        .try_get_matches_from(argv)
        .map_err(|e| dyn_error_from_message(e.to_string()))?;
    let args = Args::from_arg_matches(&matches)?;
    Ok((args, matches))
}

fn run_with_loading<T, F>(label: &str, action: F) -> Result<T, DynError>
where
    F: FnOnce() -> Result<T, DynError>,
{
    let running = Arc::new(AtomicBool::new(true));
    let spinner_running = Arc::clone(&running);
    let spinner_label = label.to_string();

    let spinner_handle = std::thread::spawn(move || {
        let frames = ['|', '/', '-', '\\'];
        let mut idx = 0usize;
        while spinner_running.load(Ordering::Relaxed) {
            eprint!("\r{} {}", spinner_label, frames[idx % frames.len()]);
            let _ = std::io::stderr().flush();
            std::thread::sleep(Duration::from_millis(120));
            idx = idx.wrapping_add(1);
        }
    });

    let result = action();

    running.store(false, Ordering::Relaxed);
    let _ = spinner_handle.join();

    if result.is_ok() {
        eprintln!("\r{} done", label);
    } else {
        eprintln!("\r{} failed", label);
    }

    result
}

async fn run_with_loading_async<T, E, Fut>(label: &str, future: Fut) -> Result<T, E>
where
    Fut: Future<Output = Result<T, E>>,
{
    let running = Arc::new(AtomicBool::new(true));
    let spinner_running = Arc::clone(&running);
    let spinner_label = label.to_string();

    let spinner_handle = std::thread::spawn(move || {
        let frames = ['|', '/', '-', '\\'];
        let mut idx = 0usize;
        while spinner_running.load(Ordering::Relaxed) {
            eprint!("\r{} {}", spinner_label, frames[idx % frames.len()]);
            let _ = std::io::stderr().flush();
            std::thread::sleep(Duration::from_millis(120));
            idx = idx.wrapping_add(1);
        }
    });

    let result = future.await;

    running.store(false, Ordering::Relaxed);
    let _ = spinner_handle.join();

    if result.is_ok() {
        eprintln!("\r{} done", label);
    } else {
        eprintln!("\r{} failed", label);
    }

    result
}

fn format_human_bytes(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit_idx = 0usize;
    while value >= 1024.0 && unit_idx + 1 < units.len() {
        value /= 1024.0;
        unit_idx += 1;
    }
    if unit_idx == 0 {
        format!("{}{}", bytes, units[unit_idx])
    } else if (value - value.round()).abs() < 0.05 {
        format!("{:.0}{}", value, units[unit_idx])
    } else {
        format!("{:.1}{}", value, units[unit_idx])
    }
}

fn print_build_summary(kapsl_path: &str) {
    let display_name = Path::new(kapsl_path)
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or(kapsl_path);
    match fs::metadata(kapsl_path) {
        Ok(metadata) => println!(
            "✓ Built {} ({})",
            display_name,
            format_human_bytes(metadata.len())
        ),
        Err(_) => println!("✓ Built {}", display_name),
    }
}

fn discover_kapsl_in_current_dir() -> Result<PathBuf, String> {
    let cwd =
        std::env::current_dir().map_err(|e| format!("Failed to read current directory: {}", e))?;
    let entries = fs::read_dir(&cwd)
        .map_err(|e| format!("Failed to read current directory {}: {}", cwd.display(), e))?;

    let mut candidates = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("aimod"))
            .unwrap_or(false)
        {
            candidates.push(path);
        }
    }

    candidates.sort();
    if candidates.is_empty() {
        return Err("No .aimod files found in the current directory. Pass an explicit package path via `kapsl push <repo>/<model>:<label> <PATH>` or `--model <PATH>`.".to_string());
    }
    if candidates.len() == 1 {
        return Ok(candidates.remove(0));
    }

    if let Some(dir_name) = cwd.file_name().and_then(|v| v.to_str()) {
        let expected = cwd.join(format!("{}.aimod", dir_name));
        if expected.exists() && expected.is_file() {
            return Ok(expected);
        }
    }

    Err(format!(
        "Multiple .aimod files found in the current directory. Pass an explicit path.\nFound: {}",
        candidates
            .iter()
            .map(|p| p.file_name().unwrap_or_default().to_string_lossy())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

fn execute_build_command(args: BuildCommandArgs) -> Result<(), DynError> {
    let metadata = match args.metadata_json.as_deref() {
        Some(raw) => Some(
            serde_json::from_str::<serde_json::Value>(raw)
                .map_err(|e| dyn_error_from_message(format!("Invalid --metadata-json: {}", e)))?,
        ),
        None => None,
    };

    let response = match args.context.as_ref() {
        Some(context_or_model_path) => {
            if context_or_model_path.is_dir() {
                run_with_loading("Building package", || {
                    create_kapsl_package_from_context(
                        context_or_model_path,
                        args.model.as_deref(),
                        args.output.as_deref(),
                        args.project_name.as_deref(),
                        args.framework.as_deref(),
                        args.version.as_deref(),
                        metadata.as_ref(),
                    )
                    .map_err(dyn_error_from_message)
                })?
            } else if looks_like_model_file_path(context_or_model_path)
                || context_or_model_path.is_file()
            {
                if args.model.is_some() {
                    return Err(dyn_error_from_message(
                        "When CONTEXT is a model file, do not also pass --model.",
                    ));
                }
                let request = PackageKapslRequest {
                    model_path: context_or_model_path.to_string_lossy().to_string(),
                    output_path: args.output.map(|p| p.to_string_lossy().to_string()),
                    project_name: args.project_name.clone(),
                    framework: args.framework.clone(),
                    version: args.version.clone(),
                    metadata: metadata.clone(),
                };
                run_with_loading("Building package", || {
                    create_kapsl_package(&request).map_err(dyn_error_from_message)
                })?
            } else {
                run_with_loading("Building package", || {
                    create_kapsl_package_from_context(
                        context_or_model_path,
                        args.model.as_deref(),
                        args.output.as_deref(),
                        args.project_name.as_deref(),
                        args.framework.as_deref(),
                        args.version.as_deref(),
                        metadata.as_ref(),
                    )
                    .map_err(dyn_error_from_message)
                })?
            }
        }
        None => {
            if let Some(model_path) = args.model.as_ref() {
                let request = PackageKapslRequest {
                    model_path: model_path.to_string_lossy().to_string(),
                    output_path: args.output.map(|p| p.to_string_lossy().to_string()),
                    project_name: args.project_name,
                    framework: args.framework,
                    version: args.version,
                    metadata,
                };
                run_with_loading("Building package", || {
                    create_kapsl_package(&request).map_err(dyn_error_from_message)
                })?
            } else {
                // Docker-style default: `kapsl build` means "build from the current directory".
                let context_dir = PathBuf::from(".");
                run_with_loading("Building package", || {
                    create_kapsl_package_from_context(
                        &context_dir,
                        None,
                        args.output.as_deref(),
                        args.project_name.as_deref(),
                        args.framework.as_deref(),
                        args.version.as_deref(),
                        metadata.as_ref(),
                    )
                    .map_err(dyn_error_from_message)
                })?
            }
        }
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .map_err(|e| dyn_error_from_message(format!("Failed to encode response: {}", e)))?
    );
    print_build_summary(&response.kapsl_path);
    Ok(())
}

fn execute_push_command(args: PushCommandArgs) -> Result<(), DynError> {
    if args.kapsl.is_some() && args.model.is_some() {
        return Err(dyn_error_from_message(
            "Push expects a single `.aimod` argument. Use either `kapsl push <repo>/<model>:<label> <KAPSL>` or `kapsl push <repo>/<model>:<label> --model <PATH>`.",
        ));
    }
    let target = parse_model_target(&args.target).map_err(dyn_error_from_message)?;

    let kapsl_path = match args.kapsl.as_ref().or(args.model.as_ref()) {
        Some(path) => path.clone(),
        None => discover_kapsl_in_current_dir().map_err(dyn_error_from_message)?,
    };

    let request = PushKapslRequest {
        kapsl_path: kapsl_path.to_string_lossy().to_string(),
        target: target.as_string(),
        remote_url: args.remote_url,
        remote_token: args.remote_token,
        interactive_login: true,
    };

    let response = run_with_loading("Uploading package", || {
        push_kapsl_to_placeholder_remote(&request).map_err(dyn_error_from_message)
    })?;
    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .map_err(|e| dyn_error_from_message(format!("Failed to encode response: {}", e)))?
    );
    Ok(())
}

fn execute_pull_command(args: PullCommandArgs) -> Result<(), DynError> {
    if args.target.is_some() && args.model.is_some() {
        return Err(dyn_error_from_message(
            "Pull expects a single target argument. Use either `kapsl pull <repo>/<model>:<label>` or `kapsl pull --model <repo>/<model>:<label>`.",
        ));
    }
    let target = args.target.or(args.model).ok_or_else(|| {
        dyn_error_from_message("Target is required. Usage: `kapsl pull <repo>/<model>:<label>`.")
    })?;
    let target = parse_model_target(&target).map_err(dyn_error_from_message)?;

    let request = PullKapslRequest {
        target: target.as_string(),
        reference: args.reference,
        destination_dir: args
            .destination_dir
            .map(|p| p.to_string_lossy().to_string()),
        remote_url: args.remote_url,
        remote_token: args.remote_token,
        interactive_login: true,
    };

    let response = run_with_loading("Downloading package", || {
        pull_kapsl_from_placeholder_remote(&request).map_err(dyn_error_from_message)
    })?;
    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .map_err(|e| dyn_error_from_message(format!("Failed to encode response: {}", e)))?
    );
    Ok(())
}

fn execute_login_command(args: LoginCommandArgs) -> Result<(), DynError> {
    let remote_url = resolved_login_remote_url(args.remote_url.as_deref());
    let auto_headless = args.no_browser || is_likely_headless_session();
    let use_device_code =
        args.device_code || (auto_headless && args.provider == OAuthProvider::GitHub);

    let response = if use_device_code {
        perform_device_code_login_flow(
            &remote_url,
            args.provider,
            args.timeout_seconds,
            args.no_browser,
        )
    } else {
        perform_browser_login_flow(
            &remote_url,
            args.provider,
            args.callback_host.trim(),
            args.callback_port,
            args.timeout_seconds,
            args.no_browser,
        )
    }
    .map_err(dyn_error_from_message)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .map_err(|e| dyn_error_from_message(format!("Failed to encode response: {}", e)))?
    );

    Ok(())
}

#[derive(Debug, Clone)]
struct ManagedRuntimeSpec {
    name: String,
    base_url: String,
    profile: RuntimeGroupProfile,
    auth_token: Option<String>,
    memory_budget_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct ControlScalingPolicy {
    min_replicas: u32,
    max_replicas: u32,
    target_queue_depth: usize,
    scale_down_threshold: usize,
    cooldown_seconds: u64,
}

impl RuntimeGroupProfile {
    fn default_scaling_policy(self) -> ControlScalingPolicy {
        match self {
            Self::Latency => ControlScalingPolicy {
                min_replicas: 1,
                max_replicas: 2,
                target_queue_depth: 2,
                scale_down_threshold: 1,
                cooldown_seconds: 180,
            },
            Self::Balanced => ControlScalingPolicy {
                min_replicas: 1,
                max_replicas: 4,
                target_queue_depth: 5,
                scale_down_threshold: 2,
                cooldown_seconds: 300,
            },
            Self::Throughput => ControlScalingPolicy {
                min_replicas: 1,
                max_replicas: 6,
                target_queue_depth: 10,
                scale_down_threshold: 3,
                cooldown_seconds: 600,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
struct ControlHealthResponse {
    status: String,
}

#[derive(Debug, Deserialize)]
struct ControlModelSummary {
    id: u32,
    #[serde(default)]
    base_model_id: u32,
    #[serde(default)]
    queue_depth: (usize, usize),
}

#[derive(Debug, Deserialize)]
struct ControlSystemStatsResponse {
    process_memory_bytes: usize,
    #[serde(default)]
    gpu_utilization: f64,
}

#[derive(Debug, Clone)]
struct RuntimeObservation {
    healthy: bool,
    health_status: String,
    total_queue_depth: usize,
    model_count: usize,
    avg_queue_depth: f64,
    gpu_utilization: f64,
    process_memory_bytes: u64,
    memory_utilization: Option<f64>,
    pressure_score: f64,
    overload: bool,
    hot: bool,
    model_ids: Vec<u32>,
}

#[derive(Debug, Default, Clone)]
struct RuntimeControlState {
    weight: f64,
    unhealthy_until: Option<Instant>,
    overload_duration: Duration,
    hot_duration: Duration,
    last_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ControlRuntimeSnapshot {
    name: String,
    base_url: String,
    profile: RuntimeGroupProfile,
    weight: f64,
    eligible: bool,
    healthy: bool,
    cooling_down: bool,
    pressure_score: Option<f64>,
    avg_queue_depth: Option<f64>,
    total_queue_depth: Option<usize>,
    model_count: Option<usize>,
    gpu_utilization: Option<f64>,
    memory_utilization: Option<f64>,
    process_memory_bytes: Option<u64>,
    message: Option<String>,
}

#[derive(Debug, Serialize)]
struct ControlWeightsFile {
    generated_at_ms: u64,
    interval_seconds: u64,
    runtimes: Vec<ControlRuntimeSnapshot>,
}

fn parse_named_pair(raw: &str, flag_name: &str) -> Result<(String, String), DynError> {
    let Some((name, value)) = raw.split_once('=') else {
        return Err(dyn_error_from_message(format!(
            "{} expects NAME=VALUE, received '{}'",
            flag_name, raw
        )));
    };
    let name = name.trim();
    let value = value.trim();
    if name.is_empty() || value.is_empty() {
        return Err(dyn_error_from_message(format!(
            "{} expects non-empty NAME and VALUE, received '{}'",
            flag_name, raw
        )));
    }
    Ok((name.to_string(), value.to_string()))
}

fn parse_named_overrides(
    values: &[String],
    flag_name: &str,
) -> Result<HashMap<String, String>, DynError> {
    let mut map = HashMap::with_capacity(values.len());
    for raw in values {
        let (name, value) = parse_named_pair(raw, flag_name)?;
        map.insert(name, value);
    }
    Ok(map)
}

fn parse_runtime_group_profile(raw: &str) -> Result<RuntimeGroupProfile, DynError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "latency" => Ok(RuntimeGroupProfile::Latency),
        "balanced" => Ok(RuntimeGroupProfile::Balanced),
        "throughput" => Ok(RuntimeGroupProfile::Throughput),
        other => Err(dyn_error_from_message(format!(
            "Invalid runtime profile '{}'. Expected one of: latency, balanced, throughput",
            other
        ))),
    }
}

fn format_authorization_header(token: Option<&str>) -> Option<String> {
    let raw = token?.trim();
    if raw.is_empty() {
        return None;
    }
    if let Some((scheme, _)) = raw.split_once(' ') {
        if scheme.eq_ignore_ascii_case("bearer") {
            return Some(raw.to_string());
        }
    }
    Some(format!("Bearer {}", raw))
}

fn parse_control_runtime_specs(
    args: &ControlCommandArgs,
) -> Result<Vec<ManagedRuntimeSpec>, DynError> {
    let profile_overrides = parse_named_overrides(&args.runtime_profiles, "--runtime-profile")?;
    let token_overrides = parse_named_overrides(&args.runtime_tokens, "--runtime-token")?;
    let memory_overrides =
        parse_named_overrides(&args.memory_budget_bytes, "--memory-budget-bytes")?;
    let shared_token = format_authorization_header(args.auth_token.as_deref());

    let mut seen = HashSet::new();
    let mut specs = Vec::with_capacity(args.runtimes.len());

    for raw in &args.runtimes {
        let (name, base_url_raw) = parse_named_pair(raw, "--runtime")?;
        if !seen.insert(name.clone()) {
            return Err(dyn_error_from_message(format!(
                "Duplicate runtime name '{}' in --runtime list",
                name
            )));
        }

        if !base_url_raw.starts_with("http://") && !base_url_raw.starts_with("https://") {
            return Err(dyn_error_from_message(format!(
                "Runtime '{}' URL must start with http:// or https:// (got '{}')",
                name, base_url_raw
            )));
        }
        let base_url = base_url_raw.trim_end_matches('/').to_string();

        let profile = match profile_overrides.get(&name) {
            Some(raw_profile) => parse_runtime_group_profile(raw_profile)?,
            None => RuntimeGroupProfile::Balanced,
        };

        let auth_token = match token_overrides.get(&name) {
            Some(raw_token) => format_authorization_header(Some(raw_token)),
            None => shared_token.clone(),
        };

        let memory_budget_bytes = match memory_overrides.get(&name) {
            Some(raw_budget) => Some(raw_budget.parse::<u64>().map_err(|error| {
                dyn_error_from_message(format!(
                    "Invalid --memory-budget-bytes for '{}': {}",
                    name, error
                ))
            })?),
            None => None,
        };

        specs.push(ManagedRuntimeSpec {
            name,
            base_url,
            profile,
            auth_token,
            memory_budget_bytes,
        });
    }

    Ok(specs)
}

fn control_http_get_json<T: DeserializeOwned>(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
) -> Result<T, String> {
    let mut request = agent.get(url);
    if let Some(auth_header) = format_authorization_header(token) {
        request = request.header("Authorization", &auth_header);
    }

    let mut response = request
        .call()
        .map_err(|error| format!("GET {} failed: {}", url, format_remote_http_error(error)))?;
    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|error| format!("Failed to read response body from {}: {}", url, error))?;
    serde_json::from_str::<T>(&body)
        .map_err(|error| format!("Failed to decode JSON response from {}: {}", url, error))
}

fn control_http_post_json(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    payload: &serde_json::Value,
) -> Result<(), String> {
    let mut request = agent.post(url).header("Content-Type", "application/json");
    if let Some(auth_header) = format_authorization_header(token) {
        request = request.header("Authorization", &auth_header);
    }

    let body = serde_json::to_string(payload)
        .map_err(|error| format!("Failed to serialize control payload for {}: {}", url, error))?;
    request
        .send(body)
        .map_err(|error| format!("POST {} failed: {}", url, format_remote_http_error(error)))?;
    Ok(())
}

fn collect_runtime_observation(
    agent: &ureq::Agent,
    runtime: &ManagedRuntimeSpec,
    queue_target: usize,
    hot_gpu_threshold: f64,
    hot_memory_threshold: f64,
) -> Result<RuntimeObservation, String> {
    let health_url = format!("{}/api/health", runtime.base_url);
    let models_url = format!("{}/api/models", runtime.base_url);
    let stats_url = format!("{}/api/system/stats", runtime.base_url);

    let health: ControlHealthResponse =
        control_http_get_json(agent, &health_url, runtime.auth_token.as_deref())?;
    let models: Vec<ControlModelSummary> =
        control_http_get_json(agent, &models_url, runtime.auth_token.as_deref())?;
    let stats: ControlSystemStatsResponse =
        control_http_get_json(agent, &stats_url, runtime.auth_token.as_deref())?;

    let total_queue_depth = models
        .iter()
        .map(|model| model.queue_depth.0.saturating_add(model.queue_depth.1))
        .sum::<usize>();
    let model_count = models.len();
    let avg_queue_depth = if model_count > 0 {
        total_queue_depth as f64 / model_count as f64
    } else {
        0.0
    };
    let queue_norm = if queue_target == 0 {
        0.0
    } else {
        (avg_queue_depth / queue_target as f64).clamp(0.0, 1.0)
    };
    let gpu_utilization = stats.gpu_utilization.clamp(0.0, 1.0);
    let memory_utilization = runtime.memory_budget_bytes.and_then(|budget| {
        if budget == 0 {
            None
        } else {
            Some((stats.process_memory_bytes as f64 / budget as f64).clamp(0.0, 1.0))
        }
    });
    let pressure_score =
        (0.5 * queue_norm) + (0.3 * gpu_utilization) + (0.2 * memory_utilization.unwrap_or(0.0));

    let overload = avg_queue_depth > (queue_target as f64 * 2.0);
    let hot = gpu_utilization >= hot_gpu_threshold
        || memory_utilization.is_some_and(|ratio| ratio >= hot_memory_threshold);

    let mut model_ids: Vec<u32> = models
        .iter()
        .map(|model| {
            if model.base_model_id > 0 {
                model.base_model_id
            } else {
                model.id
            }
        })
        .collect();
    model_ids.sort_unstable();
    model_ids.dedup();

    Ok(RuntimeObservation {
        healthy: health.status.eq_ignore_ascii_case("healthy"),
        health_status: health.status,
        total_queue_depth,
        model_count,
        avg_queue_depth,
        gpu_utilization,
        process_memory_bytes: stats.process_memory_bytes as u64,
        memory_utilization,
        pressure_score,
        overload,
        hot,
        model_ids,
    })
}

fn normalize_control_weights(
    runtimes: &[ManagedRuntimeSpec],
    states: &mut HashMap<String, RuntimeControlState>,
    observations: &HashMap<String, RuntimeObservation>,
    now: Instant,
    weight_floor: f64,
) {
    let mut eligible_names = Vec::new();
    for runtime in runtimes {
        let state = states
            .entry(runtime.name.clone())
            .or_insert_with(RuntimeControlState::default);
        let cooling_down = state.unhealthy_until.is_some_and(|until| until > now);
        let eligible = observations
            .get(&runtime.name)
            .is_some_and(|observation| observation.healthy)
            && !cooling_down;
        if eligible {
            state.weight = state.weight.max(weight_floor);
            eligible_names.push(runtime.name.clone());
        } else {
            state.weight = 0.0;
        }
    }

    if eligible_names.is_empty() {
        return;
    }

    let total_weight = eligible_names
        .iter()
        .filter_map(|name| states.get(name).map(|state| state.weight))
        .sum::<f64>();

    if total_weight <= f64::EPSILON {
        let equal_weight = 1.0 / eligible_names.len() as f64;
        for name in eligible_names {
            if let Some(state) = states.get_mut(&name) {
                state.weight = equal_weight;
            }
        }
        return;
    }

    for name in eligible_names {
        if let Some(state) = states.get_mut(&name) {
            state.weight /= total_weight;
        }
    }
}

fn apply_scaling_policy_to_runtime_models(
    agent: &ureq::Agent,
    runtime: &ManagedRuntimeSpec,
    model_ids: &[u32],
    policy: ControlScalingPolicy,
    applied: &mut HashMap<(String, u32), ControlScalingPolicy>,
) -> Result<usize, String> {
    let payload = serde_json::to_value(policy)
        .map_err(|error| format!("Failed to encode scaling payload: {}", error))?;
    let mut updated = 0usize;

    for model_id in model_ids {
        let key = (runtime.name.clone(), *model_id);
        if applied
            .get(&key)
            .is_some_and(|previous| *previous == policy)
        {
            continue;
        }

        let url = format!("{}/api/models/{}/scaling", runtime.base_url, model_id);
        control_http_post_json(agent, &url, runtime.auth_token.as_deref(), &payload)?;
        applied.insert(key, policy);
        updated = updated.saturating_add(1);
    }

    Ok(updated)
}

fn persist_control_weights(path: &Path, payload: &ControlWeightsFile) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "Failed to create control weights directory {}: {}",
                    parent.display(),
                    error
                )
            })?;
        }
    }
    let serialized = serde_json::to_string_pretty(payload)
        .map_err(|error| format!("Failed to serialize control weights snapshot: {}", error))?;
    fs::write(path, serialized).map_err(|error| {
        format!(
            "Failed to write control weights snapshot {}: {}",
            path.display(),
            error
        )
    })
}

fn execute_control_command(args: ControlCommandArgs) -> Result<(), DynError> {
    if args.queue_target == 0 {
        return Err(dyn_error_from_message(
            "--queue-target must be greater than 0",
        ));
    }

    let runtimes = parse_control_runtime_specs(&args)?;
    if runtimes.is_empty() {
        return Err(dyn_error_from_message(
            "At least one --runtime NAME=URL must be provided",
        ));
    }

    let _ = env_logger::try_init();

    let interval = Duration::from_secs(args.interval_seconds.max(1));
    let timeout = Duration::from_millis(args.timeout_ms.max(1));
    let unhealthy_hold = Duration::from_secs(args.unhealthy_hold_seconds.max(1));
    let overload_window = Duration::from_secs(args.overload_window_seconds.max(1));
    let hot_window = Duration::from_secs(args.hot_window_seconds.max(1));
    let high_pressure = args.high_pressure_score.clamp(0.0, 1.0);
    let low_pressure = args.low_pressure_score.clamp(0.0, high_pressure);
    let hot_gpu = args.hot_gpu_utilization.clamp(0.0, 1.0);
    let hot_memory = args.hot_memory_utilization.clamp(0.0, 1.0);
    let weight_step = args.weight_step.clamp(0.0, 1.0);
    let weight_floor = args.weight_floor.clamp(0.0, 1.0);
    let overload_shift = args.overload_shift_fraction.clamp(0.0, 1.0);
    let control_agent_config = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .timeout_per_call(Some(timeout))
        .build();
    let control_agent: ureq::Agent = control_agent_config.into();

    let mut states = HashMap::new();
    let initial_weight = 1.0 / runtimes.len() as f64;
    for runtime in &runtimes {
        states.insert(
            runtime.name.clone(),
            RuntimeControlState {
                weight: initial_weight,
                ..RuntimeControlState::default()
            },
        );
    }
    let mut applied_scaling_policies: HashMap<(String, u32), ControlScalingPolicy> = HashMap::new();

    log::info!(
        "Control loop started (runtimes={}, interval={}s, dry_run={}, weights_file={})",
        runtimes.len(),
        interval.as_secs(),
        args.dry_run,
        args.weights_file.display()
    );
    for runtime in &runtimes {
        log::info!(
            "  - {} => {} profile={} memory_budget_bytes={:?}",
            runtime.name,
            runtime.base_url,
            match runtime.profile {
                RuntimeGroupProfile::Latency => "latency",
                RuntimeGroupProfile::Balanced => "balanced",
                RuntimeGroupProfile::Throughput => "throughput",
            },
            runtime.memory_budget_bytes
        );
    }

    loop {
        let cycle_started = Instant::now();
        let now = Instant::now();
        let mut observations: HashMap<String, RuntimeObservation> = HashMap::new();
        let mut poll_errors: HashMap<String, String> = HashMap::new();

        for runtime in &runtimes {
            match collect_runtime_observation(
                &control_agent,
                runtime,
                args.queue_target,
                hot_gpu,
                hot_memory,
            ) {
                Ok(observation) => {
                    observations.insert(runtime.name.clone(), observation);
                }
                Err(error) => {
                    poll_errors.insert(runtime.name.clone(), error);
                }
            }
        }

        for runtime in &runtimes {
            let state = states
                .entry(runtime.name.clone())
                .or_insert_with(RuntimeControlState::default);

            if let Some(error) = poll_errors.get(&runtime.name) {
                state.unhealthy_until = Some(now + unhealthy_hold);
                state.overload_duration = Duration::from_secs(0);
                state.hot_duration = Duration::from_secs(0);
                state.last_error = Some(error.clone());
                state.weight = 0.0;
                continue;
            }

            let Some(observation) = observations.get(&runtime.name) else {
                state.unhealthy_until = Some(now + unhealthy_hold);
                state.overload_duration = Duration::from_secs(0);
                state.hot_duration = Duration::from_secs(0);
                state.last_error = Some("Observation missing after poll".to_string());
                state.weight = 0.0;
                continue;
            };

            if !observation.healthy {
                state.unhealthy_until = Some(now + unhealthy_hold);
                state.overload_duration = Duration::from_secs(0);
                state.hot_duration = Duration::from_secs(0);
                state.last_error =
                    Some(format!("Health status is '{}'", observation.health_status));
                state.weight = 0.0;
                continue;
            }

            if state.unhealthy_until.is_some_and(|until| until > now) {
                state.weight = 0.0;
                continue;
            }

            state.unhealthy_until = None;
            state.last_error = None;

            if observation.overload {
                state.overload_duration = state.overload_duration.saturating_add(interval);
            } else {
                state.overload_duration = Duration::from_secs(0);
            }
            if observation.hot {
                state.hot_duration = state.hot_duration.saturating_add(interval);
            } else {
                state.hot_duration = Duration::from_secs(0);
            }

            if observation.pressure_score > high_pressure {
                state.weight = (state.weight - weight_step).max(weight_floor);
            } else if observation.pressure_score < low_pressure {
                state.weight += weight_step;
            }

            if state.overload_duration >= overload_window || state.hot_duration >= hot_window {
                state.weight = (state.weight - overload_shift).max(weight_floor);
            }

            if !args.dry_run {
                let desired_policy = runtime.profile.default_scaling_policy();
                if let Err(error) = apply_scaling_policy_to_runtime_models(
                    &control_agent,
                    runtime,
                    &observation.model_ids,
                    desired_policy,
                    &mut applied_scaling_policies,
                ) {
                    state.last_error = Some(error);
                }
            }
        }

        normalize_control_weights(&runtimes, &mut states, &observations, now, weight_floor);

        let snapshots: Vec<ControlRuntimeSnapshot> = runtimes
            .iter()
            .map(|runtime| {
                let state = states
                    .get(&runtime.name)
                    .cloned()
                    .unwrap_or_else(RuntimeControlState::default);
                let observation = observations.get(&runtime.name);
                let cooling_down = state.unhealthy_until.is_some_and(|until| until > now);
                let eligible = observation.is_some_and(|obs| obs.healthy) && !cooling_down;
                ControlRuntimeSnapshot {
                    name: runtime.name.clone(),
                    base_url: runtime.base_url.clone(),
                    profile: runtime.profile,
                    weight: state.weight,
                    eligible,
                    healthy: observation.is_some_and(|obs| obs.healthy),
                    cooling_down,
                    pressure_score: observation.map(|obs| obs.pressure_score),
                    avg_queue_depth: observation.map(|obs| obs.avg_queue_depth),
                    total_queue_depth: observation.map(|obs| obs.total_queue_depth),
                    model_count: observation.map(|obs| obs.model_count),
                    gpu_utilization: observation.map(|obs| obs.gpu_utilization),
                    memory_utilization: observation.and_then(|obs| obs.memory_utilization),
                    process_memory_bytes: observation.map(|obs| obs.process_memory_bytes),
                    message: state.last_error.clone(),
                }
            })
            .collect();

        let output = ControlWeightsFile {
            generated_at_ms: now_unix_seconds().saturating_mul(1000),
            interval_seconds: interval.as_secs(),
            runtimes: snapshots,
        };
        if let Err(error) = persist_control_weights(&args.weights_file, &output) {
            log::warn!("{}", error);
        }

        let summary = output
            .runtimes
            .iter()
            .map(|runtime| {
                let score = runtime
                    .pressure_score
                    .map(|value| format!("{:.2}", value))
                    .unwrap_or_else(|| "-".to_string());
                format!(
                    "{}:w={:.2},score={},healthy={},cooldown={}",
                    runtime.name, runtime.weight, score, runtime.healthy, runtime.cooling_down
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        log::info!("Control tick {}", summary);

        let elapsed = cycle_started.elapsed();
        if elapsed < interval {
            std::thread::sleep(interval - elapsed);
        }
    }
}

fn runtime_argv_from_invocation(raw_argv: &[String]) -> Vec<String> {
    if matches!(raw_argv.get(1).map(|s| s.as_str()), Some("run")) {
        let mut runtime_argv = Vec::with_capacity(raw_argv.len().saturating_sub(1));
        runtime_argv.push(raw_argv[0].clone());
        runtime_argv.extend(raw_argv.iter().skip(2).cloned());
        runtime_argv
    } else {
        raw_argv.to_vec()
    }
}

fn env_flag(name: &str) -> bool {
    optional_env_var(name)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn provider_policy() -> String {
    optional_env_var(PROVIDER_POLICY_ENV)
        .unwrap_or_else(|| "fastest".to_string())
        .trim()
        .to_ascii_lowercase()
}

fn parse_bind_ip(raw: &str, fallback: IpAddr, field_name: &str) -> IpAddr {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return fallback;
    }
    match trimmed.parse::<IpAddr>() {
        Ok(addr) => addr,
        Err(error) => {
            log::warn!(
                "Invalid {} value `{}`: {}. Falling back to {}",
                field_name,
                trimmed,
                error,
                fallback
            );
            fallback
        }
    }
}

fn preflight_http_bind(http_bind: IpAddr, port: u16) -> Result<(), DynError> {
    use std::net::{SocketAddr, TcpListener};

    let addr = SocketAddr::new(http_bind, port);
    match TcpListener::bind(addr) {
        Ok(listener) => {
            drop(listener);
            Ok(())
        }
        Err(error) => {
            let mut message = format!("Failed to bind HTTP API on {}: {}", addr, error);
            if matches!(error.kind(), std::io::ErrorKind::AddrInUse) {
                message.push_str(
                    ". Another process is already using this port. Stop the other runtime or pick a different port with --metrics-port.",
                );
            }
            Err(message.into())
        }
    }
}

#[cfg(unix)]
fn preflight_ipc_socket(socket_path: &str) -> Result<(), DynError> {
    use std::os::unix::net::UnixStream;
    use std::path::Path;

    if !Path::new(socket_path).exists() {
        return Ok(());
    }

    if UnixStream::connect(socket_path).is_ok() {
        return Err(format!(
            "IPC socket path {} is already in use. Stop the other runtime or choose a different path with --socket.",
            socket_path
        )
        .into());
    }

    Ok(())
}

#[cfg(not(unix))]
fn preflight_ipc_socket(_socket_path: &str) -> Result<(), DynError> {
    Ok(())
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut diff = 0u8;
    for (lhs, rhs) in left.as_bytes().iter().zip(right.as_bytes()) {
        diff |= lhs ^ rhs;
    }
    diff == 0
}

fn authorization_matches_token(header_value: Option<&str>, expected_token: &str) -> bool {
    let Some(raw_header) = header_value else {
        return false;
    };
    let trimmed = raw_header.trim();
    if trimmed.is_empty() {
        return false;
    }
    if let Some((scheme, token)) = trimmed.split_once(' ') {
        if scheme.eq_ignore_ascii_case("bearer") {
            return constant_time_eq(token.trim(), expected_token);
        }
    }
    constant_time_eq(trimmed, expected_token)
}

fn api_auth_filter(
    required_role: ApiRole,
    required_scope: ApiScope,
    auth_state: Arc<RwLock<ApiAuthState>>,
) -> impl Filter<Extract = (), Error = warp::Rejection> + Clone {
    warp::header::optional::<String>("authorization")
        .and(warp::addr::remote())
        .and_then(
            move |authorization: Option<String>, remote: Option<std::net::SocketAddr>| {
                let auth_state = auth_state.clone();
                async move {
                    let grant_match = {
                        let state = auth_state.read();
                        if !state.auth_enabled() {
                            if is_loopback_remote(remote) {
                                return Ok::<(), warp::Rejection>(());
                            }
                            return Err(warp::reject::custom(ApiLocalOnly));
                        }

                        state.grant_from_authorization_header_read(authorization.as_deref())
                    };

                    let Some(grant_match) = grant_match else {
                        return Err(warp::reject::custom(ApiUnauthorized));
                    };

                    if !grant_match.grant.role.allows(required_role) {
                        return Err(warp::reject::custom(ApiForbidden));
                    }

                    if let Some(scopes) = grant_match.grant.scopes.as_ref() {
                        if !key_scopes_allow(scopes, required_scope) {
                            return Err(warp::reject::custom(ApiForbidden));
                        }
                    }

                    if let Some(key_index) = grant_match.matched_key_index {
                        if let Some(mut state) = auth_state.try_write() {
                            state.touch_key_last_used_by_index(key_index, now_unix_seconds());
                        }
                    }

                    Ok(())
                }
            },
        )
        .untuple_one()
}

async fn map_api_auth_rejection(
    rejection: warp::Rejection,
) -> Result<(warp::reply::Response,), warp::Rejection> {
    if rejection.find::<ApiForbidden>().is_some() {
        return Ok((warp::reply::with_status(
            warp::reply::json(&json!({
                "error": "Forbidden"
            })),
            warp::http::StatusCode::FORBIDDEN,
        )
        .into_response(),));
    }
    if rejection.find::<ApiUnauthorized>().is_some() {
        return Ok((warp::reply::with_status(
            warp::reply::json(&json!({
                "error": "Unauthorized"
            })),
            warp::http::StatusCode::UNAUTHORIZED,
        )
        .into_response(),));
    }
    if rejection.find::<ApiLocalOnly>().is_some() {
        return Ok((warp::reply::with_status(
            warp::reply::json(&json!({
                "error": "Unauthorized",
                "detail": "Authentication is disabled; this endpoint is restricted to loopback clients only."
            })),
            warp::http::StatusCode::FORBIDDEN,
        )
        .into_response(),));
    }
    Err(rejection)
}

fn redact_identifier_for_logs(raw: &str, expose_sensitive: bool) -> String {
    if expose_sensitive || raw == "-" || raw.is_empty() {
        return raw.to_string();
    }
    let prefix: String = raw.chars().take(4).collect();
    format!("{}...[redacted]", prefix)
}

fn reply_into_response<R: Reply>(reply: R) -> warp::reply::Response {
    reply.into_response()
}

fn status_code_for_engine_error(error: &EngineError) -> warp::http::StatusCode {
    use warp::http::StatusCode;

    match error {
        EngineError::InvalidInput { .. } => StatusCode::BAD_REQUEST,
        EngineError::ModelNotLoaded => StatusCode::SERVICE_UNAVAILABLE,
        EngineError::Overloaded { .. } | EngineError::ResourceExhausted { .. } => {
            StatusCode::TOO_MANY_REQUESTS
        }
        EngineError::TimeoutError { .. } => StatusCode::GATEWAY_TIMEOUT,
        EngineError::Cancelled { .. } => StatusCode::REQUEST_TIMEOUT,
        EngineError::Backend { .. }
        | EngineError::ModelLoadError { .. }
        | EngineError::InferenceError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn inferred_batch_size(shape: &[i64]) -> usize {
    shape
        .first()
        .copied()
        .filter(|dim| *dim > 0)
        .map(|dim| dim as usize)
        .unwrap_or(1)
}

fn scheduler_priority_for_request(request: &InferenceRequest) -> kapsl_scheduler::Priority {
    let scheduler_metadata = SchedulerRequestMetadata {
        priority: request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.priority)
            .unwrap_or(1),
        sla_deadline: request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.timeout_ms),
        batch_size: inferred_batch_size(&request.input.shape),
        input_size_bytes: Some(request.input.data.len()),
        estimated_flops: None,
    };

    determine_priority(&scheduler_metadata)
}

#[cfg(test)]
mod security_tests {
    use super::*;

    #[test]
    fn test_authorization_matches_token_plain_and_bearer() {
        assert!(authorization_matches_token(Some("token123"), "token123"));
        assert!(authorization_matches_token(
            Some("Bearer token123"),
            "token123"
        ));
        assert!(authorization_matches_token(
            Some("bearer token123"),
            "token123"
        ));
    }

    #[test]
    fn test_format_authorization_header_from_remote_token() {
        assert_eq!(
            format_authorization_header(Some("abc123")),
            Some("Bearer abc123".to_string())
        );
        assert_eq!(
            format_authorization_header(Some("Bearer abc123")),
            Some("Bearer abc123".to_string())
        );
    }

    #[test]
    fn test_resolved_remote_token_prefers_explicit_token() {
        assert_eq!(
            resolved_remote_token(DEFAULT_REMOTE_URL, Some("abc123")),
            Some("Bearer abc123".to_string())
        );
        assert_eq!(
            resolved_remote_token(DEFAULT_REMOTE_URL, Some("Bearer abc123")),
            Some("Bearer abc123".to_string())
        );
    }

    #[test]
    fn test_auth_base_url_from_remote_url() {
        assert_eq!(
            auth_base_url_from_remote_url("https://api.kapsl.net/v1").expect("valid"),
            "https://api.kapsl.net"
        );
        assert_eq!(
            auth_base_url_from_remote_url("https://idx.example.com/api/v1").expect("valid"),
            "https://idx.example.com"
        );
        assert!(auth_base_url_from_remote_url("oci://ghcr.io").is_err());
    }

    #[test]
    fn test_resolved_login_remote_url_uses_store_fallback() {
        let old_home = std::env::var_os("HOME");
        let old_remote_url = std::env::var_os(REMOTE_URL_ENV);
        let old_placeholder_url = std::env::var_os(REMOTE_PLACEHOLDER_URL_ENV);
        let unique = format!("test-home-{}", std::process::id());
        let temp_home = std::env::temp_dir().join(unique);
        fs::create_dir_all(&temp_home).expect("create temp home");
        std::env::set_var("HOME", &temp_home);
        std::env::remove_var(REMOTE_URL_ENV);
        std::env::remove_var(REMOTE_PLACEHOLDER_URL_ENV);

        let expected = "https://idx.example.com/v1";
        store_remote_token_for_remote(expected, "Bearer abc").expect("store token");

        assert_eq!(resolved_login_remote_url(None), expected);

        if let Some(value) = old_home {
            std::env::set_var("HOME", value);
        } else {
            std::env::remove_var("HOME");
        }
        if let Some(value) = old_remote_url {
            std::env::set_var(REMOTE_URL_ENV, value);
        } else {
            std::env::remove_var(REMOTE_URL_ENV);
        }
        if let Some(value) = old_placeholder_url {
            std::env::set_var(REMOTE_PLACEHOLDER_URL_ENV, value);
        } else {
            std::env::remove_var(REMOTE_PLACEHOLDER_URL_ENV);
        }
    }

    #[test]
    fn test_is_likely_headless_session_detects_ssh_env() {
        let old_ssh_connection = std::env::var_os("SSH_CONNECTION");
        let old_ssh_client = std::env::var_os("SSH_CLIENT");
        let old_ssh_tty = std::env::var_os("SSH_TTY");
        std::env::remove_var("SSH_CONNECTION");
        std::env::remove_var("SSH_CLIENT");
        std::env::remove_var("SSH_TTY");
        assert!(!is_likely_headless_session());

        std::env::set_var("SSH_CONNECTION", "1");
        assert!(is_likely_headless_session());

        if let Some(value) = old_ssh_connection {
            std::env::set_var("SSH_CONNECTION", value);
        } else {
            std::env::remove_var("SSH_CONNECTION");
        }
        if let Some(value) = old_ssh_client {
            std::env::set_var("SSH_CLIENT", value);
        } else {
            std::env::remove_var("SSH_CLIENT");
        }
        if let Some(value) = old_ssh_tty {
            std::env::set_var("SSH_TTY", value);
        } else {
            std::env::remove_var("SSH_TTY");
        }
    }

    #[test]
    fn test_device_code_login_rejects_non_github_provider() {
        let error = perform_device_code_login_flow(
            "https://idx.example.com/v1",
            OAuthProvider::Google,
            60,
            true,
        )
        .expect_err("expected provider restriction error");
        assert!(error.contains("supports only --provider github"));
    }

    #[test]
    fn test_authorization_mismatch_rejected() {
        assert!(!authorization_matches_token(
            Some("Bearer wrong"),
            "token123"
        ));
        assert!(!authorization_matches_token(Some("wrong"), "token123"));
        assert!(!authorization_matches_token(None, "token123"));
    }

    #[test]
    fn test_api_role_resolution_and_hierarchy() {
        let config = ApiRoleTokenConfig {
            reader_token: Some("reader-token".to_string()),
            writer_token: Some("writer-token".to_string()),
            admin_token: Some("admin-token".to_string()),
        };

        assert_eq!(
            config.role_from_authorization_header(Some("Bearer reader-token")),
            Some(ApiRole::Reader)
        );
        assert_eq!(
            config.role_from_authorization_header(Some("writer-token")),
            Some(ApiRole::Writer)
        );
        assert_eq!(
            config.role_from_authorization_header(Some("Bearer admin-token")),
            Some(ApiRole::Admin)
        );
        assert_eq!(config.role_from_authorization_header(Some("invalid")), None);

        assert!(ApiRole::Admin.allows(ApiRole::Admin));
        assert!(ApiRole::Admin.allows(ApiRole::Writer));
        assert!(ApiRole::Admin.allows(ApiRole::Reader));
        assert!(ApiRole::Writer.allows(ApiRole::Writer));
        assert!(ApiRole::Writer.allows(ApiRole::Reader));
        assert!(!ApiRole::Writer.allows(ApiRole::Admin));
        assert!(!ApiRole::Reader.allows(ApiRole::Writer));
    }

    #[test]
    fn test_api_key_scope_hierarchy_and_wildcards() {
        assert!(key_scopes_allow(&["*:*".to_string()], ApiScope::Admin));
        assert!(key_scopes_allow(&["api:*".to_string()], ApiScope::Write));
        assert!(key_scopes_allow(&["api:admin".to_string()], ApiScope::Read));
        assert!(key_scopes_allow(&["write".to_string()], ApiScope::Read));
        assert!(!key_scopes_allow(
            &["api:read".to_string()],
            ApiScope::Write
        ));
        assert!(!key_scopes_allow(&["read".to_string()], ApiScope::Admin));
        assert!(key_scopes_allow(&[], ApiScope::Admin));
    }

    #[test]
    fn test_loopback_remote_detection() {
        let loopback: std::net::SocketAddr = "127.0.0.1:9095".parse().expect("valid socket");
        let non_loopback: std::net::SocketAddr = "10.1.2.3:9095".parse().expect("valid socket");
        assert!(is_loopback_remote(Some(loopback)));
        assert!(!is_loopback_remote(Some(non_loopback)));
        assert!(!is_loopback_remote(None));
    }

    #[test]
    fn test_api_role_update_requires_admin_token_when_enabled() {
        let mut config = ApiRoleTokenConfig::default();

        let result = config.update_from_payload(ApiRoleTokenConfig {
            reader_token: Some("reader-token".to_string()),
            writer_token: None,
            admin_token: None,
        });
        assert!(result.is_err());

        let result = config.update_from_payload(ApiRoleTokenConfig {
            reader_token: Some("reader-token".to_string()),
            writer_token: Some("writer-token".to_string()),
            admin_token: Some("admin-token".to_string()),
        });
        assert!(result.is_ok());
        assert!(config.auth_enabled());
    }

    #[test]
    fn test_redact_identifier_for_logs() {
        assert_eq!(
            redact_identifier_for_logs("request-abcde", false),
            "requ...[redacted]"
        );
        assert_eq!(
            redact_identifier_for_logs("request-abcde", true),
            "request-abcde"
        );
        assert_eq!(redact_identifier_for_logs("-", false), "-");
    }

    fn make_test_auth_state() -> ApiAuthState {
        let now = now_unix_seconds();
        let mut bytes = [0u8; 4];
        OsRng.fill_bytes(&mut bytes);
        let unique_suffix = bytes
            .iter()
            .map(|byte| format!("{:02x}", byte))
            .collect::<String>();
        let store_path =
            std::env::temp_dir().join(format!("kapsl-auth-state-{}.json", unique_suffix));
        ApiAuthState {
            legacy_tokens: ApiRoleTokenConfig::default(),
            store_path,
            store: ApiAuthStoreFile {
                users: vec![
                    ApiAuthUser {
                        id: "user-admin".to_string(),
                        username: "admin".to_string(),
                        display_name: Some("Admin".to_string()),
                        role: ApiRole::Admin,
                        status: ApiUserStatus::Active,
                        created_at: now,
                        updated_at: now,
                    },
                    ApiAuthUser {
                        id: "user-reader".to_string(),
                        username: "reader".to_string(),
                        display_name: Some("Reader".to_string()),
                        role: ApiRole::Reader,
                        status: ApiUserStatus::Active,
                        created_at: now,
                        updated_at: now,
                    },
                ],
                api_keys: Vec::new(),
            },
            key_hash_index: HashMap::new(),
        }
    }

    #[test]
    fn test_first_api_key_must_be_admin() {
        let mut state = make_test_auth_state();
        let result = state.create_api_key(
            "user-reader",
            CreateApiKeyRequest {
                name: "reader-key".to_string(),
                scopes: None,
                expires_in_days: None,
            },
        );
        assert!(result.is_err());
        let _ = fs::remove_file(&state.store_path);
    }

    #[test]
    fn test_generated_api_key_authenticates_user_role() {
        let mut state = make_test_auth_state();
        let created = state
            .create_api_key(
                "user-admin",
                CreateApiKeyRequest {
                    name: "admin-key".to_string(),
                    scopes: Some(vec!["*:*".to_string()]),
                    expires_in_days: Some(30),
                },
            )
            .expect("create key");
        assert_eq!(
            state.role_from_authorization_header(Some(&format!("Bearer {}", created.raw_key))),
            Some(ApiRole::Admin)
        );
        let _ = fs::remove_file(&state.store_path);
    }

    #[test]
    fn test_generated_api_key_preserves_scopes_for_authorization() {
        let mut state = make_test_auth_state();
        let created = state
            .create_api_key(
                "user-admin",
                CreateApiKeyRequest {
                    name: "scoped-admin-key".to_string(),
                    scopes: Some(vec!["api:read".to_string()]),
                    expires_in_days: Some(30),
                },
            )
            .expect("create key");

        let grant = state
            .grant_from_authorization_header_read(Some(&format!("Bearer {}", created.raw_key)))
            .expect("grant");
        let ApiAuthGrantMatch {
            grant: ApiAuthGrant { role, scopes },
            matched_key_index,
        } = grant;
        assert!(matched_key_index.is_some());
        let scopes = scopes.expect("scopes");
        assert_eq!(role, ApiRole::Admin);
        assert!(key_scopes_allow(&scopes, ApiScope::Read));
        assert!(!key_scopes_allow(&scopes, ApiScope::Write));
        assert!(!key_scopes_allow(&scopes, ApiScope::Admin));
        let _ = fs::remove_file(&state.store_path);
    }

    fn parse_and_tune(argv: &[&str]) -> (Args, AppliedPerformanceTuning) {
        let argv: Vec<String> = argv.iter().map(|arg| (*arg).to_string()).collect();
        let (mut args, matches) = parse_runtime_args_and_matches(&argv).expect("valid args");
        let tuning = apply_performance_profile(&mut args, &matches);
        (args, tuning)
    }

    #[test]
    fn test_throughput_profile_tunes_defaults() {
        let (args, tuning) = parse_and_tune(&["kapsl", "--performance-profile", "throughput"]);
        assert_eq!(args.batch_size, 16);
        assert_eq!(args.transport, "hybrid");
        assert_eq!(args.scheduler_queue_size, 2048);
        assert_eq!(args.scheduler_max_micro_batch, 16);
        assert_eq!(args.scheduler_queue_delay_ms, 6);
        assert_eq!(tuning.batch_size, Some(16));
        assert_eq!(tuning.transport.as_deref(), Some("hybrid"));
        assert_eq!(tuning.scheduler_queue_size, Some(2048));
        assert_eq!(tuning.scheduler_max_micro_batch, Some(16));
        assert_eq!(tuning.scheduler_queue_delay_ms, Some(6));
        assert_eq!(tuning.media_preprocess, None);
    }

    #[test]
    fn test_profile_does_not_override_explicit_batch_or_transport() {
        let (args, tuning) = parse_and_tune(&[
            "kapsl",
            "--performance-profile",
            "throughput",
            "--batch-size",
            "2",
            "--transport",
            "tcp",
        ]);
        assert_eq!(args.batch_size, 2);
        assert_eq!(args.transport, "tcp");
        assert_eq!(tuning.batch_size, None);
        assert_eq!(tuning.transport, None);
    }

    #[test]
    fn test_standard_profile_keeps_existing_defaults() {
        // Explicitly pass --performance-profile standard to verify Standard is a no-op.
        let (args, tuning) = parse_and_tune(&["kapsl", "--performance-profile", "standard"]);
        assert_eq!(args.batch_size, 4);
        assert_eq!(args.transport, "socket");
        assert_eq!(args.scheduler_queue_size, 256);
        assert_eq!(args.scheduler_max_micro_batch, 4);
        assert_eq!(args.scheduler_queue_delay_ms, 2);
        assert_eq!(args.performance_profile, PerformanceProfile::Standard);
        assert_eq!(tuning.batch_size, None);
        assert_eq!(tuning.transport, None);
        assert_eq!(tuning.scheduler_queue_size, None);
        assert_eq!(tuning.scheduler_max_micro_batch, None);
        assert_eq!(tuning.scheduler_queue_delay_ms, None);
    }

    #[test]
    fn test_auto_profile_is_default_and_uses_conservative_defaults_with_no_model() {
        // No --model → model_size_mb=0 → conservative defaults (same values as Standard).
        // tuning fields are Some because Auto applied them.
        let (args, tuning) = parse_and_tune(&["kapsl"]);
        assert_eq!(args.performance_profile, PerformanceProfile::Auto);
        assert_eq!(args.batch_size, 4);
        assert_eq!(args.scheduler_queue_size, 256);
        assert_eq!(args.scheduler_max_micro_batch, 4);
        assert_eq!(args.scheduler_queue_delay_ms, 2);
        assert!(tuning.batch_size.is_some());
        assert!(tuning.auto_tune_rationale.is_some());
        let rationale = tuning.auto_tune_rationale.unwrap();
        assert!(
            rationale.contains("unknown"),
            "rationale should say unknown: {}",
            rationale
        );
    }

    #[test]
    fn test_auto_profile_respects_explicit_batch_size() {
        let (args, tuning) = parse_and_tune(&["kapsl", "--batch-size", "1"]);
        assert_eq!(args.performance_profile, PerformanceProfile::Auto);
        assert_eq!(args.batch_size, 1); // user explicit value preserved
        assert_eq!(tuning.batch_size, None); // not overridden by auto-tune
    }

    #[test]
    fn test_onnx_tuning_profile_resolves_global_and_per_model_overrides() {
        let (args, _) = parse_and_tune(&[
            "kapsl",
            "--onnx-memory-pattern",
            "false",
            "--onnx-model-tuning",
            "*:session_buckets=2",
            "--onnx-model-tuning",
            "7:disable_cpu_mem_arena=true,peak_concurrency=8",
        ]);

        let profile = build_onnx_tuning_profile(&args).expect("valid ONNX tuning profile");
        let model_7 = profile.resolve(7);
        assert_eq!(model_7.memory_pattern, Some(false));
        assert_eq!(model_7.session_buckets, Some(2));
        assert_eq!(model_7.disable_cpu_mem_arena, Some(true));
        assert_eq!(model_7.peak_concurrency_hint, Some(8));

        let model_9 = profile.resolve(9);
        assert_eq!(model_9.memory_pattern, Some(false));
        assert_eq!(model_9.session_buckets, Some(2));
        assert_eq!(model_9.disable_cpu_mem_arena, None);
        assert_eq!(model_9.peak_concurrency_hint, None);
    }

    #[test]
    fn test_onnx_tuning_profile_rejects_unknown_keys() {
        let (args, _) = parse_and_tune(&["kapsl", "--onnx-model-tuning", "3:not_a_real_key=1"]);
        let err = build_onnx_tuning_profile(&args).expect_err("unknown key should fail");
        assert!(err.contains("unknown ONNX tuning key"));
    }

    #[test]
    fn test_media_validation_rejects_shape_mismatch() {
        let request = InferenceRequest::new(BinaryTensorPacket {
            shape: vec![1, 3, 128, 128],
            dtype: TensorDtype::Float32,
            data: vec![0; 3 * 128 * 128 * 4],
        });
        let model_info = EngineModelInfo {
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![]],
            input_dtypes: vec!["float32".to_string()],
            output_dtypes: vec![],
            framework: Some("onnx".to_string()),
            model_version: None,
            peak_concurrency: None,
        };
        let result = validate_inference_request_against_model_info(&request, &model_info);
        assert!(result.is_err());
        assert!(result
            .expect_err("shape mismatch should fail")
            .contains("shape mismatch"));
    }

    #[test]
    fn test_media_validation_accepts_dynamic_dims_and_dtype_match() {
        let request = InferenceRequest::new(BinaryTensorPacket {
            shape: vec![1, 3, 320, 320],
            dtype: TensorDtype::Float32,
            data: vec![0; 3 * 320 * 320 * 4],
        });
        let model_info = EngineModelInfo {
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            input_shapes: vec![vec![1, 3, -1, -1]],
            output_shapes: vec![vec![]],
            input_dtypes: vec!["float32".to_string()],
            output_dtypes: vec![],
            framework: Some("onnx".to_string()),
            model_version: None,
            peak_concurrency: None,
        };
        let result = validate_inference_request_against_model_info(&request, &model_info);
        assert!(result.is_ok());
    }

    #[test]
    fn test_status_code_for_engine_error_overloaded_and_timeout() {
        let overloaded = EngineError::overloaded("queue full");
        assert_eq!(
            status_code_for_engine_error(&overloaded),
            warp::http::StatusCode::TOO_MANY_REQUESTS
        );

        let timeout = EngineError::timeout("deadline exceeded");
        assert_eq!(
            status_code_for_engine_error(&timeout),
            warp::http::StatusCode::GATEWAY_TIMEOUT
        );
    }

    #[test]
    fn test_status_code_for_engine_error_backend_and_input() {
        let backend = EngineError::backend("runtime panic");
        assert_eq!(
            status_code_for_engine_error(&backend),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR
        );

        let invalid = EngineError::invalid_input("bad tensor shape");
        assert_eq!(
            status_code_for_engine_error(&invalid),
            warp::http::StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn test_parse_queue_overflow_policy_literal() {
        assert_eq!(
            parse_queue_overflow_policy_literal("block"),
            Some(kapsl_scheduler::QueueOverflowPolicy::Block)
        );
        assert_eq!(
            parse_queue_overflow_policy_literal("latest_only"),
            Some(kapsl_scheduler::QueueOverflowPolicy::DropNewest)
        );
        assert_eq!(
            parse_queue_overflow_policy_literal("drop_oldest"),
            Some(kapsl_scheduler::QueueOverflowPolicy::DropOldest)
        );
        assert_eq!(parse_queue_overflow_policy_literal("unknown"), None);
    }

    #[test]
    fn test_manifest_queue_overflow_policy_prefers_runtime_server_path() {
        let metadata: serde_yaml::Value = serde_yaml::from_str(
            r#"
runtime:
  server:
    queue_overflow_policy: drop_oldest
"#,
        )
        .expect("parse yaml");
        let manifest = Manifest {
            project_name: "demo".to_string(),
            framework: "onnx".to_string(),
            version: "1.0.0".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            model_file: "model.onnx".to_string(),
            metadata: Some(metadata),
            hardware_requirements: kapsl_core::HardwareRequirements::default(),
        };

        let policy = manifest_queue_overflow_policy(&manifest);
        assert_eq!(
            policy,
            Some(kapsl_scheduler::QueueOverflowPolicy::DropOldest)
        );
    }

    #[test]
    fn test_effective_topology_choice_falls_back_tensor_parallel_to_data_parallel() {
        let manifest = Manifest {
            project_name: "demo".to_string(),
            framework: "onnx".to_string(),
            version: "1.0.0".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            model_file: "model.onnx".to_string(),
            metadata: None,
            hardware_requirements: kapsl_core::HardwareRequirements::default(),
        };

        let choice = resolve_effective_topology_choice(&manifest, "tensor-parallel", 4, None);
        assert!(!choice.use_pipeline_backend);
        assert_eq!(choice.worker_topology, "data-parallel");
        assert_eq!(choice.worker_tp_degree, 1);
        assert!(matches!(
            choice.mesh_topology,
            kapsl_hal::device_mesh::MeshTopology::DataParallel
        ));
    }

    #[test]
    fn test_effective_topology_choice_uses_pipeline_metadata_stage_count() {
        let metadata: serde_yaml::Value = serde_yaml::from_str(
            r#"
llm:
  pipeline:
    stages:
      - stage0.onnx
      - stage1.onnx
"#,
        )
        .expect("parse yaml");
        let manifest = Manifest {
            project_name: "demo".to_string(),
            framework: "llm".to_string(),
            version: "1.0.0".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            model_file: "model.onnx".to_string(),
            metadata: Some(metadata),
            hardware_requirements: kapsl_core::HardwareRequirements::default(),
        };
        let stages = manifest_llm_pipeline_stages(&manifest);

        let choice =
            resolve_effective_topology_choice(&manifest, "pipeline-parallel", 8, stages.as_deref());
        assert!(choice.use_pipeline_backend);
        assert_eq!(choice.worker_topology, "pipeline-parallel");
        assert_eq!(choice.worker_tp_degree, 2);
        assert!(matches!(
            choice.mesh_topology,
            kapsl_hal::device_mesh::MeshTopology::PipelineParallel { stages: 2 }
        ));
    }

    #[test]
    fn test_effective_topology_choice_falls_back_without_pipeline_metadata() {
        let manifest = Manifest {
            project_name: "demo".to_string(),
            framework: "llm".to_string(),
            version: "1.0.0".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            model_file: "model.onnx".to_string(),
            metadata: None,
            hardware_requirements: kapsl_core::HardwareRequirements::default(),
        };

        let choice = resolve_effective_topology_choice(&manifest, "pipeline-parallel", 4, None);
        assert!(!choice.use_pipeline_backend);
        assert_eq!(choice.worker_topology, "data-parallel");
        assert_eq!(choice.worker_tp_degree, 1);
        assert!(matches!(
            choice.mesh_topology,
            kapsl_hal::device_mesh::MeshTopology::DataParallel
        ));
    }

    #[test]
    fn test_evaluate_runtime_pressure_state_transitions() {
        let config = RuntimePressureConfig {
            memory_conserve_ratio: 0.7,
            memory_emergency_ratio: 0.9,
            gpu_util_conserve_ratio: 0.8,
            gpu_util_emergency_ratio: 0.95,
            gpu_mem_conserve_ratio: 0.8,
            gpu_mem_emergency_ratio: 0.95,
            conserve_max_new_tokens: Some(256),
            emergency_max_new_tokens: Some(128),
        };

        let normal = RuntimeSamples {
            process_memory_bytes: 2 * 1024,
            total_system_memory_bytes: Some(10 * 1024),
            gpu_utilization: 0.2,
            gpu_memory_bytes: Some(100),
            gpu_memory_total_bytes: Some(1000),
            collected_at_ms: 0,
        };
        assert_eq!(
            evaluate_runtime_pressure_state(&normal, &config),
            RuntimePressureState::Normal
        );

        let conserve = RuntimeSamples {
            process_memory_bytes: 8 * 1024,
            total_system_memory_bytes: Some(10 * 1024),
            ..normal.clone()
        };
        assert_eq!(
            evaluate_runtime_pressure_state(&conserve, &config),
            RuntimePressureState::Conserve
        );

        let emergency = RuntimeSamples {
            gpu_utilization: 0.97,
            ..normal
        };
        assert_eq!(
            evaluate_runtime_pressure_state(&emergency, &config),
            RuntimePressureState::Emergency
        );
    }
}

#[cfg(test)]
mod inter_model_relay_tests {
    use super::*;

    #[test]
    fn test_parse_inter_model_routes_supports_multiple_separators() {
        let routes = parse_inter_model_routes(
            "vision=reasoner;audio=reasoner,transcriber\nmonitor->alerter;vision=reasoner",
        );

        assert_eq!(
            routes.get("vision"),
            Some(&vec!["reasoner".to_string()]),
            "duplicate targets should be deduplicated"
        );
        assert_eq!(
            routes.get("audio"),
            Some(&vec!["reasoner".to_string(), "transcriber".to_string()])
        );
        assert_eq!(routes.get("monitor"), Some(&vec!["alerter".to_string()]));
    }

    #[test]
    fn test_relay_prompt_from_output_only_accepts_utf8_payloads() {
        let utf8_packet = BinaryTensorPacket {
            shape: vec![1, 5],
            dtype: TensorDtype::Utf8,
            data: b"hello".to_vec(),
        };
        let prompt = relay_prompt_from_output("vision", &utf8_packet).expect("utf8 prompt");
        assert!(prompt.contains("Report from vision"));
        assert!(prompt.contains("hello"));

        let non_utf8_packet = BinaryTensorPacket {
            shape: vec![1, 1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 0, 0],
        };
        assert!(relay_prompt_from_output("vision", &non_utf8_packet).is_none());
    }

    #[test]
    fn test_relay_state_rate_limits_per_source_model_id() {
        let state = InterModelRelayState {
            routes: parse_inter_model_routes("a=b"),
            min_interval: Duration::from_secs(60),
            last_relay_at: Arc::new(Mutex::new(HashMap::new())),
        };

        assert!(state.should_emit(7));
        assert!(!state.should_emit(7), "second emit should be rate limited");
        assert!(state.should_emit(8), "different source id should emit");
    }
}

#[cfg(test)]
mod oci_remote_tests {
    use super::*;

    #[test]
    fn test_parse_model_target_valid_and_invalid() {
        assert_eq!(
            parse_model_target("alice/mnist:prod")
                .expect("valid target")
                .as_string(),
            "alice/mnist:prod"
        );
        assert!(parse_model_target("alice/mnist").is_err());
        assert!(parse_model_target("alice/mnist:").is_err());
        assert!(parse_model_target("/mnist:prod").is_err());
        assert!(parse_model_target("alice/mnist:pro:d").is_err());
    }

    #[test]
    fn test_parse_oci_remote_prefix_basic_and_trailing_slash() {
        assert_eq!(
            parse_oci_remote_prefix("oci://ghcr.io").expect("valid prefix"),
            "ghcr.io"
        );
        assert_eq!(
            parse_oci_remote_prefix("oci://ghcr.io/acme/").expect("valid prefix"),
            "ghcr.io/acme"
        );
    }

    #[test]
    fn test_parse_oci_remote_prefix_rejects_tag_and_digest() {
        assert!(parse_oci_remote_prefix("oci://ghcr.io/acme:latest").is_err());
        assert!(parse_oci_remote_prefix("oci://ghcr.io/acme@sha256:0123").is_err());
    }

    #[test]
    fn test_build_oci_repo_for_target() {
        let target = parse_model_target("alice/mnist:prod").expect("target");
        assert_eq!(
            build_oci_repo_for_target("oci://ghcr.io", &target).expect("repo"),
            "ghcr.io/alice/mnist"
        );
        assert_eq!(
            build_oci_repo_for_target("oci://ghcr.io/team", &target).expect("repo"),
            "ghcr.io/team/alice/mnist"
        );
    }

    #[test]
    fn test_build_oci_reference_tag_and_digest_override() {
        let repo = "ghcr.io/acme/kapsl";
        let tag = "prod";
        assert_eq!(
            build_oci_reference(repo, tag, None).expect("tag ref"),
            "ghcr.io/acme/kapsl:prod"
        );

        let digest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert_eq!(
            build_oci_reference(repo, tag, Some(digest)).expect("digest ref"),
            format!("ghcr.io/acme/kapsl@{}", digest)
        );
        assert_eq!(
            build_oci_reference(repo, tag, Some(&format!("@{}", digest))).expect("digest ref"),
            format!("ghcr.io/acme/kapsl@{}", digest)
        );
        assert_eq!(
            build_oci_reference(repo, tag, Some(&format!("{}@{}", repo, digest)))
                .expect("full ref"),
            format!("{}@{}", repo, digest)
        );
    }

    #[test]
    fn test_parse_oras_manifest_digest_prefers_digest_line() {
        let digest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let output = format!(
            "Uploading...\nUploaded layer: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nDigest: {}\nDone\n",
            digest
        );
        assert_eq!(
            parse_oras_manifest_digest(&output),
            Some(digest.to_string())
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelTargetRef {
    repo: String,
    model: String,
    label: String,
}

impl ModelTargetRef {
    fn as_string(&self) -> String {
        format!("{}/{}:{}", self.repo, self.model, self.label)
    }
}

fn is_valid_target_part(part: &str) -> bool {
    !part.is_empty()
        && part
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-')
}

fn parse_model_target(raw: &str) -> Result<ModelTargetRef, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(
            "Target cannot be empty. Expected format: <repo_name>/<model>:<label>.".to_string(),
        );
    }

    let (repo, rest) = trimmed.split_once('/').ok_or_else(|| {
        format!(
            "Invalid target `{}`. Expected format: <repo_name>/<model>:<label>.",
            trimmed
        )
    })?;

    if rest.contains('/') {
        return Err(format!(
            "Invalid target `{}`. Only one `/` is allowed (between repo and model).",
            trimmed
        ));
    }

    let (model, label) = rest.split_once(':').ok_or_else(|| {
        format!(
            "Invalid target `{}`. Expected format: <repo_name>/<model>:<label>.",
            trimmed
        )
    })?;

    if label.contains(':') {
        return Err(format!(
            "Invalid target `{}`. Label must not contain `:`.",
            trimmed
        ));
    }

    if !is_valid_target_part(repo) {
        return Err(format!(
            "Invalid repo `{}` in target `{}`. Allowed characters: [A-Za-z0-9._-].",
            repo, trimmed
        ));
    }
    if !is_valid_target_part(model) {
        return Err(format!(
            "Invalid model `{}` in target `{}`. Allowed characters: [A-Za-z0-9._-].",
            model, trimmed
        ));
    }
    if !is_valid_target_part(label) {
        return Err(format!(
            "Invalid label `{}` in target `{}`. Allowed characters: [A-Za-z0-9._-].",
            label, trimmed
        ));
    }

    Ok(ModelTargetRef {
        repo: repo.to_string(),
        model: model.to_string(),
        label: label.to_string(),
    })
}

fn resolved_remote_url(custom_url: Option<&str>) -> String {
    if let Some(url) = custom_url {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    if let Some(url) = optional_env_var(REMOTE_URL_ENV) {
        return url;
    }

    if let Some(url) = optional_env_var(REMOTE_PLACEHOLDER_URL_ENV) {
        return url;
    }

    DEFAULT_REMOTE_URL.to_string()
}

fn resolved_login_remote_url(custom_url: Option<&str>) -> String {
    if let Some(url) = custom_url {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    if let Some(url) = optional_env_var(REMOTE_URL_ENV) {
        return url;
    }

    if let Some(url) = optional_env_var(REMOTE_PLACEHOLDER_URL_ENV) {
        return url;
    }

    if let Some(url) = read_last_remote_url_from_store() {
        return url;
    }

    DEFAULT_REMOTE_URL.to_string()
}

fn auth_base_url_from_remote_url(remote_url: &str) -> Result<String, String> {
    let trimmed = remote_url.trim().trim_end_matches('/');
    if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
        return Err(format!(
            "Remote URL must start with http:// or https:// for login flow (got '{}')",
            remote_url
        ));
    }

    if let Some(stripped) = trimmed.strip_suffix("/api/v1") {
        if stripped.is_empty() {
            return Err(format!("Invalid remote URL '{}'", remote_url));
        }
        return Ok(stripped.to_string());
    }
    if let Some(stripped) = trimmed.strip_suffix("/v1") {
        if stripped.is_empty() {
            return Err(format!("Invalid remote URL '{}'", remote_url));
        }
        return Ok(stripped.to_string());
    }
    Ok(trimmed.to_string())
}

fn perform_browser_login_flow(
    remote_url: &str,
    provider: OAuthProvider,
    callback_host: &str,
    callback_port: u16,
    timeout_seconds: u64,
    no_browser: bool,
) -> Result<LoginResponse, String> {
    if is_oci_remote_url(remote_url) {
        return Err(
            "Login is only supported for HTTP(S) remote backends, not oci:// remotes.".to_string(),
        );
    }

    let auth_base_url = auth_base_url_from_remote_url(remote_url)?;
    let callback_addr = format!("{}:{}", callback_host.trim(), callback_port);
    let listener = TcpListener::bind(&callback_addr).map_err(|e| {
        format!(
            "Failed to bind local login callback listener at {}: {}",
            callback_addr, e
        )
    })?;
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("Failed to configure callback listener: {}", e))?;

    let local_addr = listener
        .local_addr()
        .map_err(|e| format!("Failed to read callback address: {}", e))?;
    let callback_url = format!("http://{}/callback", local_addr);
    let login_url = format!(
        "{}/auth/{}/login?redirect_uri={}",
        auth_base_url,
        provider.route_segment(),
        percent_encode_query_component(&callback_url)
    );

    if no_browser {
        println!("Open this URL to sign in:\n{}", login_url);
    } else if !open_browser(&login_url) {
        println!(
            "Could not open a browser automatically. Open this URL to sign in:\n{}",
            login_url
        );
    }

    let timeout = Duration::from_secs(timeout_seconds.max(1));
    let token = wait_for_login_callback_token(listener, timeout)
        .map_err(|e| format!("Login callback failed: {}", e))?;

    let token_store_path = store_remote_token_for_remote(remote_url, &token)?;

    Ok(LoginResponse {
        status: "ok".to_string(),
        remote_url: remote_url.to_string(),
        auth_base_url,
        provider: provider.route_segment().to_string(),
        login_method: "browser-callback".to_string(),
        callback_url,
        token_store_path: token_store_path.to_string_lossy().to_string(),
        verification_uri: None,
        user_code: None,
    })
}

fn perform_device_code_login_flow(
    remote_url: &str,
    provider: OAuthProvider,
    timeout_seconds: u64,
    no_browser: bool,
) -> Result<LoginResponse, String> {
    if is_oci_remote_url(remote_url) {
        return Err(
            "Login is only supported for HTTP(S) remote backends, not oci:// remotes.".to_string(),
        );
    }
    if provider != OAuthProvider::GitHub {
        return Err("Device code flow currently supports only --provider github.".to_string());
    }

    let auth_base_url = auth_base_url_from_remote_url(remote_url)?;
    let start_url = format!(
        "{}/auth/{}/device/start",
        auth_base_url,
        provider.route_segment()
    );
    let poll_url = format!(
        "{}/auth/{}/device/poll",
        auth_base_url,
        provider.route_segment()
    );

    let mut start_response = ureq::post(&start_url)
        .header("Accept", "application/json")
        .header("Content-Type", "application/json")
        .send("{}")
        .map_err(|error| match error {
            ureq::Error::StatusCode(404) => format!(
                "Remote backend does not support device code login at {} (missing endpoint /auth/{}/device/start).",
                auth_base_url,
                provider.route_segment()
            ),
            other => format!(
                "Failed to start device code login at {}: {}",
                start_url,
                format_remote_http_error(other)
            ),
        })?;
    let start_body = start_response
        .body_mut()
        .read_to_string()
        .map_err(|error| {
            format!(
                "Failed to read device code start response from {}: {}",
                start_url, error
            )
        })?;
    let start: DeviceCodeStartResponse = serde_json::from_str(&start_body).map_err(|error| {
        format!(
            "Failed to decode device code start response from {}: {}",
            start_url, error
        )
    })?;

    let device_code = start.device_code.trim();
    if device_code.is_empty() {
        return Err("Remote backend returned an empty device_code.".to_string());
    }
    let verification_uri = start.verification_uri.trim();
    if verification_uri.is_empty() {
        return Err("Remote backend returned an empty verification_uri.".to_string());
    }
    let user_code = start.user_code.trim();
    if user_code.is_empty() {
        return Err("Remote backend returned an empty user_code.".to_string());
    }

    if let Some(complete_url) = start.verification_uri_complete.as_deref() {
        let trimmed = complete_url.trim();
        if !trimmed.is_empty() {
            if no_browser {
                println!("Open this URL to authorize this login:\n{}", trimmed);
            } else if !open_browser(trimmed) {
                println!(
                    "Could not open a browser automatically. Open this URL to authorize this login:\n{}",
                    trimmed
                );
            }
        }
    }
    println!(
        "If prompted, open {} and enter code: {}",
        verification_uri, user_code
    );
    println!("Waiting for authorization approval...");

    let started_at = Instant::now();
    let timeout = Duration::from_secs(timeout_seconds.max(1));
    let mut interval_secs = start.interval.unwrap_or(5).max(1);
    let expires_in = start.expires_in.unwrap_or(timeout.as_secs()).max(1);
    let flow_deadline = started_at + Duration::from_secs(expires_in);
    let timeout_deadline = started_at + timeout;

    loop {
        let now = Instant::now();
        if now >= timeout_deadline {
            return Err("Timed out waiting for device authorization approval.".to_string());
        }
        if now >= flow_deadline {
            return Err("Device authorization code expired. Start login again.".to_string());
        }
        std::thread::sleep(Duration::from_secs(interval_secs));

        let poll_payload = serde_json::json!({
            "device_code": device_code
        });
        let mut poll_response = ureq::post(&poll_url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .send(
                serde_json::to_string(&poll_payload)
                    .map_err(|error| format!("Failed to encode device poll payload: {}", error))?,
            )
            .map_err(|error| match error {
                ureq::Error::StatusCode(404) => format!(
                    "Remote backend does not support device code polling at {} (missing endpoint /auth/{}/device/poll).",
                    auth_base_url,
                    provider.route_segment()
                ),
                other => format!(
                    "Failed to poll device code login at {}: {}",
                    poll_url,
                    format_remote_http_error(other)
                ),
            })?;
        let poll_body = poll_response.body_mut().read_to_string().map_err(|error| {
            format!(
                "Failed to read device poll response from {}: {}",
                poll_url, error
            )
        })?;
        let poll: DeviceCodePollResponse = serde_json::from_str(&poll_body).map_err(|error| {
            format!(
                "Failed to decode device poll response from {}: {}",
                poll_url, error
            )
        })?;

        match poll.status.trim() {
            "approved" => {
                let token = poll.token.unwrap_or_default();
                let trimmed = token.trim();
                if trimmed.is_empty() {
                    return Err("Device authorization completed without token.".to_string());
                }
                let token_store_path = store_remote_token_for_remote(remote_url, trimmed)?;
                return Ok(LoginResponse {
                    status: "ok".to_string(),
                    remote_url: remote_url.to_string(),
                    auth_base_url,
                    provider: provider.route_segment().to_string(),
                    login_method: "device-code".to_string(),
                    callback_url: String::new(),
                    token_store_path: token_store_path.to_string_lossy().to_string(),
                    verification_uri: Some(verification_uri.to_string()),
                    user_code: Some(user_code.to_string()),
                });
            }
            "pending" => {
                interval_secs = poll.interval.unwrap_or(interval_secs).max(1);
                continue;
            }
            "denied" => {
                return Err("Device authorization was denied by the user.".to_string());
            }
            "expired" => {
                return Err("Device authorization code expired. Start login again.".to_string());
            }
            "error" => {
                let err = poll
                    .error
                    .unwrap_or_else(|| "device_code_error".to_string());
                let description = poll.error_description.unwrap_or_default();
                if description.trim().is_empty() {
                    return Err(format!("Device authorization failed: {}", err));
                }
                return Err(format!(
                    "Device authorization failed: {} ({})",
                    err, description
                ));
            }
            other => {
                return Err(format!("Unexpected device authorization status: {}", other));
            }
        }
    }
}

#[derive(Debug)]
struct RemoteHttpRequestError {
    status_code: Option<u16>,
    message: String,
}

impl RemoteHttpRequestError {}

fn maybe_auto_login_for_remote(
    remote_url: &str,
    request_has_explicit_token: bool,
    interactive_login: bool,
    remote_token: &mut Option<String>,
    http_error: &RemoteHttpRequestError,
) -> Result<bool, String> {
    if !interactive_login || request_has_explicit_token || remote_token.is_some() {
        return Ok(false);
    }
    if http_error.status_code != Some(401) {
        return Ok(false);
    }

    println!(
        "Remote backend requires authentication. Starting browser login for {} ...",
        remote_url
    );
    let browser_login = perform_browser_login_flow(
        remote_url,
        OAuthProvider::GitHub,
        "127.0.0.1",
        0,
        180,
        false,
    );
    if let Err(error) = browser_login {
        println!(
            "Browser login flow failed ({}). Falling back to device code flow...",
            error
        );
        let _ = perform_device_code_login_flow(remote_url, OAuthProvider::GitHub, 600, true)?;
    }
    *remote_token = resolved_remote_token(remote_url, None);
    Ok(remote_token.is_some())
}

fn is_likely_headless_session() -> bool {
    std::env::var_os("SSH_CONNECTION").is_some()
        || std::env::var_os("SSH_CLIENT").is_some()
        || std::env::var_os("SSH_TTY").is_some()
}

fn remote_token_store_key(remote_url: &str) -> String {
    auth_base_url_from_remote_url(remote_url).unwrap_or_else(|_| remote_url.trim().to_string())
}

fn resolve_remote_token_store_path() -> PathBuf {
    if let Some(path) = optional_env_var(REMOTE_TOKEN_STORE_PATH_ENV) {
        return PathBuf::from(path);
    }

    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".kapsl").join("remote-token-store.json")
}

fn load_remote_token_store(path: &Path) -> RemoteTokenStoreFile {
    let Ok(raw) = fs::read_to_string(path) else {
        return RemoteTokenStoreFile::default();
    };

    serde_json::from_str::<RemoteTokenStoreFile>(&raw).unwrap_or_default()
}

fn save_remote_token_store(path: &Path, store: &RemoteTokenStoreFile) -> Result<(), String> {
    let parent = path.parent().ok_or_else(|| {
        format!(
            "Invalid token store path (missing parent directory): {}",
            path.display()
        )
    })?;
    fs::create_dir_all(parent).map_err(|e| {
        format!(
            "Failed to create token store directory {}: {}",
            parent.display(),
            e
        )
    })?;

    let raw = serde_json::to_string_pretty(store)
        .map_err(|e| format!("Failed to serialize token store: {}", e))?;
    fs::write(path, raw)
        .map_err(|e| format!("Failed to write token store {}: {}", path.display(), e))
}

fn read_stored_remote_token_for_remote(remote_url: &str) -> Option<String> {
    let path = resolve_remote_token_store_path();
    let store = load_remote_token_store(&path);
    let key = remote_token_store_key(remote_url);
    store.tokens_by_remote.get(&key).cloned()
}

fn read_last_remote_url_from_store() -> Option<String> {
    let path = resolve_remote_token_store_path();
    let store = load_remote_token_store(&path);
    store
        .last_remote_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn store_remote_token_for_remote(remote_url: &str, token: &str) -> Result<PathBuf, String> {
    let path = resolve_remote_token_store_path();
    let mut store = load_remote_token_store(&path);
    let trimmed_remote_url = remote_url.trim();
    if !trimmed_remote_url.is_empty() {
        store.last_remote_url = Some(trimmed_remote_url.to_string());
    }
    store
        .tokens_by_remote
        .insert(remote_token_store_key(remote_url), token.trim().to_string());
    save_remote_token_store(&path, &store)?;
    Ok(path)
}

fn resolved_remote_token(remote_url: &str, custom_token: Option<&str>) -> Option<String> {
    if let Some(explicit) = format_authorization_header(custom_token) {
        return Some(explicit);
    }

    let env_token = optional_env_var(REMOTE_TOKEN_ENV);
    if let Some(env_header) = format_authorization_header(env_token.as_deref()) {
        return Some(env_header);
    }

    format_authorization_header(read_stored_remote_token_for_remote(remote_url).as_deref())
}

fn percent_encode_query_component(input: &str) -> String {
    let mut encoded = String::with_capacity(input.len());
    for byte in input.bytes() {
        let ch = byte as char;
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '~') {
            encoded.push(ch);
        } else {
            encoded.push('%');
            encoded.push_str(&format!("{:02X}", byte));
        }
    }
    encoded
}

fn open_browser(url: &str) -> bool {
    #[cfg(target_os = "macos")]
    {
        return Command::new("open").arg(url).status().is_ok();
    }
    #[cfg(target_os = "windows")]
    {
        return Command::new("cmd")
            .args(["/C", "start", "", url])
            .status()
            .is_ok();
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        Command::new("xdg-open").arg(url).status().is_ok()
    }
}

fn wait_for_login_callback_token(
    listener: TcpListener,
    timeout: Duration,
) -> Result<String, String> {
    let deadline = Instant::now() + timeout;
    loop {
        match listener.accept() {
            Ok((mut stream, _peer)) => {
                return handle_login_callback_stream(&mut stream);
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err("timed out waiting for login callback".to_string());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(err) => {
                return Err(format!("failed to accept callback connection: {}", err));
            }
        }
    }
}

fn handle_login_callback_stream(stream: &mut TcpStream) -> Result<String, String> {
    let mut buffer = [0u8; 8192];
    let bytes_read = stream
        .read(&mut buffer)
        .map_err(|e| format!("failed to read callback request: {}", e))?;
    if bytes_read == 0 {
        return Err("empty callback request".to_string());
    }

    let request = String::from_utf8_lossy(&buffer[..bytes_read]);
    let request_line = request
        .lines()
        .next()
        .ok_or_else(|| "missing callback request line".to_string())?;
    let path = request_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| "malformed callback request line".to_string())?;

    let token = extract_query_value_from_path(path, "token");
    if let Some(raw_token) = token {
        let trimmed = raw_token.trim();
        if !trimmed.is_empty() {
            let body =
                "<html><body><h3>Login complete</h3><p>You can close this tab.</p></body></html>";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes());
            return Ok(trimmed.to_string());
        }
    }

    let body = "<html><body><h3>Login failed</h3><p>Token not found in callback.</p></body></html>";
    let response = format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
    Err("callback did not include token".to_string())
}

fn extract_query_value_from_path(path: &str, key: &str) -> Option<String> {
    let (_, query) = path.split_once('?')?;
    for pair in query.split('&') {
        let (raw_key, raw_value) = pair.split_once('=').unwrap_or((pair, ""));
        if raw_key == key {
            return Some(percent_decode(raw_value));
        }
    }
    None
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = &value[i + 1..i + 3];
                if let Ok(decoded) = u8::from_str_radix(hex, 16) {
                    out.push(decoded);
                    i += 3;
                    continue;
                }
                out.push(bytes[i]);
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            ch => {
                out.push(ch);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn placeholder_remote_storage_dir() -> PathBuf {
    if let Some(path) = optional_env_var(REMOTE_PLACEHOLDER_DIR_ENV) {
        return PathBuf::from(path);
    }

    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(REMOTE_PLACEHOLDER_DIR)
}

fn is_default_placeholder_remote(remote_url: &str) -> bool {
    remote_url.trim_end_matches('/') == REMOTE_PLACEHOLDER_URL.trim_end_matches('/')
}

fn artifact_url_for_remote(remote_url: &str, target: &ModelTargetRef) -> String {
    format!(
        "{}/kapsl/{}/{}:{}",
        remote_url.trim_end_matches('/'),
        target.repo,
        target.model,
        target.label
    )
}

fn placeholder_remote_artifact_path(target: &ModelTargetRef) -> PathBuf {
    placeholder_remote_storage_dir()
        .join(&target.repo)
        .join(&target.model)
        .join(format!("{}.aimod", target.label))
}

fn format_remote_http_error(error: ureq::Error) -> String {
    match error {
        ureq::Error::StatusCode(status) => format!("Remote backend returned HTTP {}", status),
        other => other.to_string(),
    }
}

fn push_kapsl_to_http_remote(
    artifact_url: &str,
    source_path: &Path,
    authorization_header: Option<&str>,
) -> Result<u64, RemoteHttpRequestError> {
    let file_size = fs::metadata(source_path).map_err(|e| RemoteHttpRequestError {
        status_code: None,
        message: format!(
            "Failed to read .aimod metadata for upload {}: {}",
            source_path.display(),
            e
        ),
    })?;
    let file = File::open(source_path).map_err(|e| RemoteHttpRequestError {
        status_code: None,
        message: format!(
            "Failed to open .aimod for upload {}: {}",
            source_path.display(),
            e
        ),
    })?;

    let mut request = ureq::put(artifact_url).header("Content-Type", "application/octet-stream");
    if let Some(header) = authorization_header {
        request = request.header("Authorization", header);
    }
    request.send(file).map_err(|e| {
        let status = match e {
            ureq::Error::StatusCode(code) => Some(code),
            _ => None,
        };
        RemoteHttpRequestError {
            status_code: status,
            message: format!(
                "Failed to upload .aimod to remote backend {}: {}",
                artifact_url,
                format_remote_http_error(e)
            ),
        }
    })?;

    Ok(file_size.len())
}

fn pull_kapsl_from_http_remote(
    artifact_url: &str,
    authorization_header: Option<&str>,
) -> Result<Vec<u8>, RemoteHttpRequestError> {
    let mut request = ureq::get(artifact_url);
    if let Some(header) = authorization_header {
        request = request.header("Authorization", header);
    }
    let mut response = request.call().map_err(|e| {
        let status = match e {
            ureq::Error::StatusCode(code) => Some(code),
            _ => None,
        };
        RemoteHttpRequestError {
            status_code: status,
            message: format!(
                "Failed to download .aimod from remote backend {}: {}",
                artifact_url,
                format_remote_http_error(e)
            ),
        }
    })?;

    response
        .body_mut()
        .read_to_vec()
        .map_err(|e| RemoteHttpRequestError {
            status_code: None,
            message: format!(
                "Failed to read .aimod response body from {}: {}",
                artifact_url, e
            ),
        })
}

#[derive(Debug, Clone)]
struct OrasAuth {
    username: String,
    password: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct KapslOciConfig {
    artifact_type: String,
    filename: String,
    sha256: String,
    size: u64,
}

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Result<Self, String> {
        let mut nonce_bytes = [0u8; 8];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = hex_encode(&nonce_bytes);
        let dir = std::env::temp_dir().join(format!("{}-{}-{}", prefix, std::process::id(), nonce));
        fs::create_dir_all(&dir).map_err(|e| {
            format!(
                "Failed to create temporary directory {}: {}",
                dir.display(),
                e
            )
        })?;
        Ok(Self { path: dir })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn is_oci_remote_url(remote_url: &str) -> bool {
    remote_url.trim().starts_with(OCI_REMOTE_PREFIX)
}

fn parse_oci_remote_prefix(remote_url: &str) -> Result<String, String> {
    let trimmed = remote_url.trim();
    if !trimmed.starts_with(OCI_REMOTE_PREFIX) {
        return Err(format!(
            "Invalid OCI remote URL (expected {}<registry>[/prefix]): {}",
            OCI_REMOTE_PREFIX, trimmed
        ));
    }
    let prefix = trimmed
        .trim_start_matches(OCI_REMOTE_PREFIX)
        .trim()
        .trim_end_matches('/')
        .to_string();
    if prefix.is_empty() {
        return Err(format!(
            "OCI remote URL is missing registry (expected {}<registry>[/prefix])",
            OCI_REMOTE_PREFIX
        ));
    }
    if prefix.contains("://") {
        return Err(format!(
            "OCI remote URL must be an OCI reference, not a URL: {}",
            prefix
        ));
    }
    if prefix.contains('@') {
        return Err(format!(
            "OCI remote URL must be a registry/prefix without digest, got: {}",
            prefix
        ));
    }

    let segments: Vec<&str> = prefix.split('/').collect();
    if segments.iter().any(|segment| segment.is_empty()) {
        return Err(format!(
            "OCI remote URL contains empty path segments, got: {}",
            prefix
        ));
    }

    for segment in segments.iter().skip(1) {
        if segment.contains(':') {
            return Err(format!(
                "OCI remote URL prefix path must not contain tags, got: {}",
                prefix
            ));
        }
    }

    Ok(prefix)
}

fn build_oci_repo_for_target(remote_url: &str, target: &ModelTargetRef) -> Result<String, String> {
    let prefix = parse_oci_remote_prefix(remote_url)?;
    Ok(format!("{}/{}/{}", prefix, target.repo, target.model))
}

fn oci_registry_for_repo(repo: &str) -> Result<String, String> {
    let registry = repo
        .split('/')
        .next()
        .ok_or_else(|| format!("Invalid OCI repository: {}", repo))?;
    if registry.trim().is_empty() {
        return Err(format!("Invalid OCI repository: {}", repo));
    }
    Ok(registry.to_string())
}

fn oras_bin() -> String {
    optional_env_var(ORAS_BIN_ENV).unwrap_or_else(|| "oras".to_string())
}

fn load_oras_auth_from_env() -> Result<Option<OrasAuth>, String> {
    let username = optional_env_var(OCI_USERNAME_ENV);
    let password = optional_env_var(OCI_PASSWORD_ENV);
    match (username, password) {
        (None, None) => Ok(None),
        (Some(username), Some(password)) => Ok(Some(OrasAuth { username, password })),
        _ => Err(format!(
            "OCI auth requires both {} and {} to be set (or neither).",
            OCI_USERNAME_ENV, OCI_PASSWORD_ENV
        )),
    }
}

fn ensure_oras_support(oras_bin: &str) -> Result<(), String> {
    let version = Command::new(oras_bin)
        .arg("version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| {
            format!(
                "Failed to execute `{}`: {}. Install ORAS and ensure it is on PATH (or set {}).",
                oras_bin, e, ORAS_BIN_ENV
            )
        })?;
    if !version.status.success() {
        let stderr = String::from_utf8_lossy(&version.stderr);
        return Err(format!(
            "`{} version` failed (exit {}): {}",
            oras_bin,
            version.status.code().unwrap_or(-1),
            stderr.trim()
        ));
    }

    let push_help = Command::new(oras_bin)
        .args(["push", "--help"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to execute `{}`: {}", oras_bin, e))?;
    let push_help_text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&push_help.stdout),
        String::from_utf8_lossy(&push_help.stderr)
    );
    for required in ["--artifact-type", "--annotation", "--config"] {
        if !push_help_text.contains(required) {
            return Err(format!(
                "`oras push` does not support required flag {}. Please upgrade ORAS.",
                required
            ));
        }
    }

    let pull_help = Command::new(oras_bin)
        .args(["pull", "--help"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to execute `{}`: {}", oras_bin, e))?;
    let pull_help_text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&pull_help.stdout),
        String::from_utf8_lossy(&pull_help.stderr)
    );
    if !pull_help_text.contains("--output") {
        return Err(
            "`oras pull` does not appear to support `--output`. Please upgrade ORAS.".to_string(),
        );
    }

    Ok(())
}

fn sha256_file_hex(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|e| format!("Failed to open file for sha256 {}: {}", path.display(), e))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("Failed to read file for sha256 {}: {}", path.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(hex_encode(digest.as_slice()))
}

fn read_manifest_from_kapsl_archive(package_path: &Path) -> Result<Manifest, String> {
    let file = File::open(package_path).map_err(|e| {
        format!(
            "Failed to open .aimod to read metadata.json {}: {}",
            package_path.display(),
            e
        )
    })?;
    let tar = GzDecoder::new(file);
    let mut archive = Archive::new(tar);
    let entries = archive.entries().map_err(|e| {
        format!(
            "Failed to list .aimod archive entries for {}: {}",
            package_path.display(),
            e
        )
    })?;

    for entry in entries {
        let mut entry = entry.map_err(|e| {
            format!(
                "Failed to read .aimod archive entry for {}: {}",
                package_path.display(),
                e
            )
        })?;
        let path = entry.path().map_err(|e| {
            format!(
                "Failed to read .aimod entry path for {}: {}",
                package_path.display(),
                e
            )
        })?;
        if path.as_ref() != Path::new("metadata.json") {
            continue;
        }

        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).map_err(|e| {
            format!(
                "Failed to read metadata.json from {}: {}",
                package_path.display(),
                e
            )
        })?;
        let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|e| {
            format!(
                "Failed to parse metadata.json in {}: {}",
                package_path.display(),
                e
            )
        })?;
        return Ok(manifest);
    }

    Err(format!(
        "Manifest not found in package (metadata.json missing): {}",
        package_path.display()
    ))
}

fn parse_oras_manifest_digest(output: &str) -> Option<String> {
    let mut last_digest = None;
    for line in output.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("digest:") {
            if let Some(idx) = lower.find("sha256:") {
                let candidate = &trimmed[idx..];
                if candidate.len() >= "sha256:".len() + 64
                    && candidate["sha256:".len()..]
                        .chars()
                        .take(64)
                        .all(|ch| ch.is_ascii_hexdigit())
                {
                    return Some(candidate[..("sha256:".len() + 64)].to_string());
                }
            }
        }
        if let Some(idx) = lower.find("sha256:") {
            let candidate = &trimmed[idx..];
            if candidate.len() >= "sha256:".len() + 64
                && candidate["sha256:".len()..]
                    .chars()
                    .take(64)
                    .all(|ch| ch.is_ascii_hexdigit())
            {
                last_digest = Some(candidate[..("sha256:".len() + 64)].to_string());
            }
        }
    }
    last_digest
}

fn oras_login(
    oras_bin: &str,
    registry: &str,
    auth: &OrasAuth,
    docker_config_dir: Option<&Path>,
) -> Result<(), String> {
    let mut cmd = Command::new(oras_bin);
    cmd.arg("login")
        .arg(registry)
        .arg("--username")
        .arg(&auth.username)
        .arg("--password-stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = docker_config_dir {
        cmd.env("DOCKER_CONFIG", dir);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start `oras login`: {}", e))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(auth.password.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .map_err(|e| format!("Failed to write `oras login` password: {}", e))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to wait for `oras login`: {}", e))?;
    if !output.status.success() {
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return Err(format!(
            "`oras login` failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            combined.trim()
        ));
    }

    Ok(())
}

fn oras_push_kapsl(
    oras_bin: &str,
    reference: &str,
    kapsl_path: &Path,
    config_path: &Path,
    annotations: &[(String, String)],
    docker_config_dir: Option<&Path>,
) -> Result<Option<String>, String> {
    let config_spec = format!("{}:{}", config_path.display(), KAPSL_OCI_CONFIG_TYPE);
    let layer_spec = format!("{}:{}", kapsl_path.display(), KAPSL_OCI_LAYER_TYPE);

    let mut cmd = Command::new(oras_bin);
    cmd.arg("push")
        .arg("--artifact-type")
        .arg(KAPSL_OCI_ARTIFACT_TYPE)
        .arg("--config")
        .arg(&config_spec);
    for (key, value) in annotations {
        cmd.arg("--annotation").arg(format!("{}={}", key, value));
    }
    cmd.arg(reference)
        .arg(&layer_spec)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = docker_config_dir {
        cmd.env("DOCKER_CONFIG", dir);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute `oras push`: {}", e))?;
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        return Err(format!(
            "`oras push` failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            combined.trim()
        ));
    }
    Ok(parse_oras_manifest_digest(&combined))
}

fn oras_pull(
    oras_bin: &str,
    reference: &str,
    output_dir: &Path,
    docker_config_dir: Option<&Path>,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|e| {
        format!(
            "Failed to create OCI pull output dir {}: {}",
            output_dir.display(),
            e
        )
    })?;
    let mut cmd = Command::new(oras_bin);
    cmd.arg("pull")
        .arg(reference)
        .arg("--output")
        .arg(output_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = docker_config_dir {
        cmd.env("DOCKER_CONFIG", dir);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute `oras pull`: {}", e))?;
    if !output.status.success() {
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return Err(format!(
            "`oras pull` failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            combined.trim()
        ));
    }
    Ok(())
}

fn build_oci_reference(
    repo: &str,
    tag: &str,
    reference_override: Option<&str>,
) -> Result<String, String> {
    if let Some(reference_override) = reference_override {
        let trimmed = reference_override.trim();
        if trimmed.is_empty() {
            return Err("OCI reference override cannot be empty".to_string());
        }

        if trimmed.starts_with("sha256:") {
            return Ok(format!("{}@{}", repo, trimmed));
        }
        if trimmed.starts_with("@sha256:") {
            return Ok(format!("{}{}", repo, trimmed));
        }
        if trimmed.contains("@sha256:") {
            return Ok(trimmed.to_string());
        }

        return Err(
            "Invalid OCI reference override. Expected `sha256:<digest>`, `@sha256:<digest>`, or `<repo>@sha256:<digest>`."
                .to_string(),
        );
    }

    Ok(format!("{}:{}", repo, tag))
}

fn push_kapsl_to_oci_remote(
    remote_url: &str,
    absolute_path: &Path,
    target: &ModelTargetRef,
    filename: &str,
) -> Result<PushKapslResponse, String> {
    let oras = oras_bin();
    ensure_oras_support(&oras)?;

    let remote_url = remote_url.trim().trim_end_matches('/').to_string();
    let repo = build_oci_repo_for_target(&remote_url, target)?;
    let reference = format!("{}:{}", repo, target.label);

    let bytes_uploaded = fs::metadata(absolute_path)
        .map_err(|e| format!("Failed to stat .aimod {}: {}", absolute_path.display(), e))?
        .len();
    let sha256 = sha256_file_hex(absolute_path)?;

    let manifest = read_manifest_from_kapsl_archive(absolute_path).ok();

    let mut annotations = Vec::new();
    annotations.push(("io.kapsl.aimod.target".to_string(), target.as_string()));
    annotations.push(("io.kapsl.aimod.repo".to_string(), target.repo.clone()));
    annotations.push(("io.kapsl.aimod.model".to_string(), target.model.clone()));
    annotations.push(("io.kapsl.aimod.label".to_string(), target.label.clone()));
    annotations.push(("io.kapsl.aimod.filename".to_string(), filename.to_string()));
    annotations.push(("io.kapsl.aimod.sha256".to_string(), sha256.clone()));
    annotations.push((
        "io.kapsl.aimod.size".to_string(),
        bytes_uploaded.to_string(),
    ));
    if let Some(manifest) = manifest.as_ref() {
        annotations.push((
            "io.kapsl.aimod.project_name".to_string(),
            manifest.project_name.clone(),
        ));
        annotations.push((
            "io.kapsl.aimod.framework".to_string(),
            manifest.framework.clone(),
        ));
        annotations.push((
            "io.kapsl.aimod.version".to_string(),
            manifest.version.clone(),
        ));
        annotations.push((
            "io.kapsl.aimod.created_at".to_string(),
            manifest.created_at.clone(),
        ));
        annotations.push((
            "org.opencontainers.image.title".to_string(),
            manifest.project_name.clone(),
        ));
        annotations.push((
            "org.opencontainers.image.version".to_string(),
            manifest.version.clone(),
        ));
        annotations.push((
            "org.opencontainers.image.created".to_string(),
            manifest.created_at.clone(),
        ));
    } else {
        annotations.push((
            "org.opencontainers.image.title".to_string(),
            filename.to_string(),
        ));
    }

    let config = KapslOciConfig {
        artifact_type: KAPSL_OCI_ARTIFACT_TYPE.to_string(),
        filename: filename.to_string(),
        sha256,
        size: bytes_uploaded,
    };

    let temp_dir = TempDirGuard::new("kapsl-oci")?;
    let config_path = temp_dir.path().join("kapsl-config.json");
    let config_bytes = serde_json::to_vec(&config)
        .map_err(|e| format!("Failed to encode OCI config JSON: {}", e))?;
    fs::write(&config_path, &config_bytes).map_err(|e| {
        format!(
            "Failed to write OCI config JSON to {}: {}",
            config_path.display(),
            e
        )
    })?;

    let auth = load_oras_auth_from_env()?;
    let docker_config_dir = if auth.is_some() {
        let docker_dir = temp_dir.path().join("docker-config");
        fs::create_dir_all(&docker_dir).map_err(|e| {
            format!(
                "Failed to create docker config directory {}: {}",
                docker_dir.display(),
                e
            )
        })?;
        Some(docker_dir)
    } else {
        None
    };

    if let Some(auth) = auth.as_ref() {
        let registry = oci_registry_for_repo(&repo)?;
        oras_login(&oras, &registry, auth, docker_config_dir.as_deref())?;
    }

    let manifest_digest = oras_push_kapsl(
        &oras,
        &reference,
        absolute_path,
        &config_path,
        &annotations,
        docker_config_dir.as_deref(),
    )?;

    let artifact_url = format!("{}{}", OCI_REMOTE_PREFIX, reference);

    Ok(PushKapslResponse {
        status: "ok".to_string(),
        remote_url,
        artifact_url: artifact_url.clone(),
        mirrored_path: artifact_url,
        bytes_uploaded,
        manifest_digest,
    })
}

fn pull_kapsl_from_oci_remote(
    remote_url: &str,
    target: &ModelTargetRef,
    reference_override: Option<&str>,
    destination_dir: &Path,
) -> Result<PullKapslResponse, String> {
    let oras = oras_bin();
    ensure_oras_support(&oras)?;

    let remote_url = remote_url.trim().trim_end_matches('/').to_string();
    let repo = build_oci_repo_for_target(&remote_url, target)?;
    let reference = build_oci_reference(&repo, &target.label, reference_override)?;
    let filename = format!("{}.aimod", target.model);

    let temp_dir = TempDirGuard::new("kapsl-oci-pull")?;
    let auth = load_oras_auth_from_env()?;
    let docker_config_dir = if auth.is_some() {
        let docker_dir = temp_dir.path().join("docker-config");
        fs::create_dir_all(&docker_dir).map_err(|e| {
            format!(
                "Failed to create docker config directory {}: {}",
                docker_dir.display(),
                e
            )
        })?;
        Some(docker_dir)
    } else {
        None
    };

    if let Some(auth) = auth.as_ref() {
        let registry = oci_registry_for_repo(&repo)?;
        oras_login(&oras, &registry, auth, docker_config_dir.as_deref())?;
    }

    oras_pull(
        &oras,
        &reference,
        temp_dir.path(),
        docker_config_dir.as_deref(),
    )?;

    let expected = temp_dir.path().join(&filename);
    let pulled_path = if expected.exists() {
        expected
    } else {
        let mut kapsls = Vec::new();
        let entries = fs::read_dir(temp_dir.path()).map_err(|e| {
            format!(
                "Failed to read OCI pull output dir {}: {}",
                temp_dir.path().display(),
                e
            )
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read OCI pull dir entry: {}", e))?;
            let path = entry.path();
            if path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("aimod"))
                .unwrap_or(false)
            {
                kapsls.push(path);
            }
        }

        kapsls.sort();
        if kapsls.len() == 1 {
            kapsls.remove(0)
        } else if kapsls.is_empty() {
            return Err(
                "OCI pull succeeded but no .aimod file was found in the pulled artifact."
                    .to_string(),
            );
        } else {
            return Err(format!(
                "OCI pull produced multiple .aimod files; expected one. Files: {}",
                kapsls
                    .iter()
                    .map(|p| p.file_name().unwrap_or_default().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    };

    fs::create_dir_all(destination_dir).map_err(|e| {
        format!(
            "Failed to create destination directory {}: {}",
            destination_dir.display(),
            e
        )
    })?;
    let output_path = destination_dir.join(&filename);
    fs::copy(&pulled_path, &output_path).map_err(|e| {
        format!(
            "Failed to write pulled .aimod to {}: {}",
            output_path.display(),
            e
        )
    })?;

    let bytes_downloaded = fs::metadata(&output_path)
        .map_err(|e| {
            format!(
                "Failed to stat pulled .aimod {}: {}",
                output_path.display(),
                e
            )
        })?
        .len();

    let absolute_output_path = output_path.canonicalize().unwrap_or(output_path);
    let artifact_url = format!("{}{}", OCI_REMOTE_PREFIX, reference);

    Ok(PullKapslResponse {
        status: "ok".to_string(),
        remote_url,
        artifact_url,
        kapsl_path: absolute_output_path.to_string_lossy().to_string(),
        bytes_downloaded,
    })
}

fn extension_marketplace_url(custom_url: Option<&str>) -> String {
    if let Some(url) = custom_url {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    if let Some(url) = optional_env_var(EXTENSION_MARKETPLACE_URL_ENV) {
        return url;
    }

    EXTENSION_MARKETPLACE_URL.to_string()
}

fn is_valid_extension_id(extension_id: &str) -> bool {
    !extension_id.trim().is_empty()
        && extension_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
}

fn fetch_extension_marketplace(
    query: Option<&str>,
    marketplace_url: Option<&str>,
) -> Result<serde_json::Value, String> {
    let marketplace_url = extension_marketplace_url(marketplace_url);
    let mut request = ureq::get(&marketplace_url);

    if let Some(q) = query {
        let trimmed = q.trim();
        if !trimmed.is_empty() {
            request = request.query("q", trimmed);
        }
    }

    let mut response = request.call().map_err(|e| {
        format!(
            "Failed to query extension marketplace {}: {}",
            marketplace_url,
            format_remote_http_error(e)
        )
    })?;

    let raw = response.body_mut().read_to_string().map_err(|e| {
        format!(
            "Failed to read extension marketplace response from {}: {}",
            marketplace_url, e
        )
    })?;

    serde_json::from_str::<serde_json::Value>(&raw).map_err(|e| {
        format!(
            "Failed to parse extension marketplace response as JSON from {}: {}",
            marketplace_url, e
        )
    })
}

fn collect_extension_manifest_dirs(dir: &Path, matches: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| {
        format!(
            "Failed to inspect extracted extension archive directory {}: {}",
            dir.display(),
            e
        )
    })? {
        let entry = entry.map_err(|e| format!("Failed to read archive directory entry: {}", e))?;
        let path = entry.path();
        if path.is_dir() {
            collect_extension_manifest_dirs(&path, matches)?;
            continue;
        }

        if path.file_name().and_then(|n| n.to_str()) == Some("rag-extension.toml") {
            if let Some(parent) = path.parent() {
                matches.push(parent.to_path_buf());
            }
        }
    }

    Ok(())
}

fn find_extension_manifest_root(extract_dir: &Path) -> Result<PathBuf, String> {
    let mut matches = Vec::new();
    collect_extension_manifest_dirs(extract_dir, &mut matches)?;

    if matches.is_empty() {
        return Err(format!(
            "Marketplace archive did not contain rag-extension.toml under {}",
            extract_dir.display()
        ));
    }

    if matches.len() > 1 {
        return Err(format!(
            "Marketplace archive contained multiple extension manifests under {}",
            extract_dir.display()
        ));
    }

    Ok(matches.remove(0))
}

fn unpack_marketplace_archive(archive_bytes: &[u8], target_dir: &Path) -> Result<(), String> {
    let decoder = GzDecoder::new(Cursor::new(archive_bytes));
    let mut archive = Archive::new(decoder);
    let entries = archive
        .entries()
        .map_err(|e| format!("Failed to read extension marketplace archive: {}", e))?;

    for entry in entries {
        let mut entry =
            entry.map_err(|e| format!("Failed to read extension archive entry: {}", e))?;
        let unpacked = entry.unpack_in(target_dir).map_err(|e| {
            format!(
                "Failed to unpack extension archive into {}: {}",
                target_dir.display(),
                e
            )
        })?;
        if !unpacked {
            return Err("Extension archive contains invalid paths".to_string());
        }
    }

    Ok(())
}

fn install_extension_from_marketplace(
    registry: &ExtensionRegistry,
    extension_id: &str,
    marketplace_url: Option<&str>,
) -> Result<InstalledExtension, String> {
    let extension_id = extension_id.trim();
    if !is_valid_extension_id(extension_id) {
        return Err(format!("Invalid extension_id `{}`", extension_id));
    }

    let marketplace_url = extension_marketplace_url(marketplace_url);
    let download_url = format!(
        "{}/{}/download",
        marketplace_url.trim_end_matches('/'),
        extension_id
    );

    let mut response = ureq::get(&download_url).call().map_err(|e| {
        format!(
            "Failed to download extension `{}` from marketplace {}: {}",
            extension_id,
            marketplace_url,
            format_remote_http_error(e)
        )
    })?;

    let archive_bytes = response.body_mut().read_to_vec().map_err(|e| {
        format!(
            "Failed to read downloaded extension `{}` archive from {}: {}",
            extension_id, download_url, e
        )
    })?;

    let timestamp = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let temp_dir = std::env::temp_dir().join(format!(
        "kapsl-extension-marketplace-{}-{}",
        std::process::id(),
        timestamp
    ));
    fs::create_dir_all(&temp_dir).map_err(|e| {
        format!(
            "Failed to prepare temporary extension directory {}: {}",
            temp_dir.display(),
            e
        )
    })?;

    let install_result = (|| {
        unpack_marketplace_archive(&archive_bytes, &temp_dir)?;
        let extracted_root = find_extension_manifest_root(&temp_dir)?;
        registry
            .install_from_dir(&extracted_root)
            .map_err(|e| e.to_string())
    })();

    let _ = fs::remove_dir_all(&temp_dir);
    install_result
}

fn infer_framework_from_model_path(model_path: &Path) -> String {
    let ext = model_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "onnx" => "onnx".to_string(),
        "gguf" => "gguf".to_string(),
        "safetensors" => "pytorch".to_string(),
        "pt" | "pth" => "pytorch".to_string(),
        "pb" => "tensorflow".to_string(),
        _ => "onnx".to_string(),
    }
}

fn looks_like_model_file_path(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "onnx" | "gguf" | "safetensors" | "pt" | "pth" | "pb"
    )
}

fn append_tar_bytes_entry(
    builder: &mut Builder<GzEncoder<File>>,
    entry_path: &str,
    bytes: &[u8],
) -> Result<(), String> {
    let mut header = tar::Header::new_gnu();
    header
        .set_path(entry_path)
        .map_err(|e| format!("Failed to set tar path {}: {}", entry_path, e))?;
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append(&header, bytes)
        .map_err(|e| format!("Failed to append {} to archive: {}", entry_path, e))
}

fn create_kapsl_package(request: &PackageKapslRequest) -> Result<PackageKapslResponse, String> {
    let input_model_path = PathBuf::from(request.model_path.trim());
    if !input_model_path.exists() {
        return Err(format!(
            "Model file does not exist: {}",
            input_model_path.display()
        ));
    }

    if !input_model_path.is_file() {
        return Err(format!(
            "Model path must be a file: {}",
            input_model_path.display()
        ));
    }

    let model_path = input_model_path.canonicalize().map_err(|e| {
        format!(
            "Failed to resolve model file path {}: {}",
            input_model_path.display(),
            e
        )
    })?;

    let model_file_name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("Invalid model filename: {}", model_path.display()))?
        .to_string();

    let project_name = request
        .project_name
        .as_ref()
        .map(|name| name.trim())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .or_else(|| {
            model_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "kapsl-model".to_string());

    let framework = request
        .framework
        .as_ref()
        .map(|framework| framework.trim())
        .filter(|framework| !framework.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| infer_framework_from_model_path(&model_path));

    let version = request
        .version
        .as_ref()
        .map(|version| version.trim())
        .filter(|version| !version.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| "1.0.0".to_string());

    let mut output_path = request
        .output_path
        .as_ref()
        .map(|path| PathBuf::from(path.trim()))
        .unwrap_or_else(|| PathBuf::from(format!("{}.aimod", project_name)));

    if output_path.extension().and_then(|ext| ext.to_str()) != Some("aimod") {
        output_path.set_extension("aimod");
    }

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create parent directory {}: {}",
                    parent.display(),
                    e
                )
            })?;
        }
    }

    let created_at = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("System clock error: {}", e))?
        .as_secs();

    let metadata = match request.metadata.as_ref() {
        Some(metadata) => Some(
            serde_yaml::to_value(metadata)
                .map_err(|e| format!("Failed to convert metadata payload: {}", e))?,
        ),
        None => None,
    };

    let manifest = Manifest {
        project_name: project_name.clone(),
        framework: framework.clone(),
        version: version.clone(),
        created_at: created_at.to_string(),
        model_file: model_file_name.clone(),
        metadata,
        hardware_requirements: kapsl_core::HardwareRequirements::default(),
    };

    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| format!("Failed to encode metadata.json: {}", e))?;
    let output_file = File::create(&output_path).map_err(|e| {
        format!(
            "Failed to create output package {}: {}",
            output_path.display(),
            e
        )
    })?;
    let encoder = GzEncoder::new(output_file, Compression::default());
    let mut archive = Builder::new(encoder);

    append_tar_bytes_entry(&mut archive, "metadata.json", &manifest_bytes)?;

    archive
        .append_path_with_name(&model_path, &model_file_name)
        .map_err(|e| format!("Failed to add model file to archive: {}", e))?;

    let encoder = archive
        .into_inner()
        .map_err(|e| format!("Failed to finalize tar archive: {}", e))?;
    encoder
        .finish()
        .map_err(|e| format!("Failed to finalize gzip stream: {}", e))?;

    let absolute_output_path = output_path.canonicalize().unwrap_or(output_path);

    Ok(PackageKapslResponse {
        status: "ok".to_string(),
        kapsl_path: absolute_output_path.to_string_lossy().to_string(),
        project_name,
        framework,
        version,
    })
}

fn find_model_file_in_context(context_dir: &Path) -> Result<PathBuf, String> {
    let mut stack = vec![context_dir.to_path_buf()];
    let mut matches = Vec::new();
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|e| format!("Failed to read context directory {}: {}", dir.display(), e))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|e| format!("Failed to inspect {}: {}", path.display(), e))?;
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|v| v.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if matches!(
                ext.as_str(),
                "onnx" | "gguf" | "safetensors" | "pt" | "pth" | "pb"
            ) {
                matches.push(path);
            }
        }
    }

    matches.sort();
    if matches.is_empty() {
        return Err(format!(
            "No model file found in context {}. Pass --model explicitly.",
            context_dir.display()
        ));
    }
    if matches.len() > 1 {
        return Err(format!(
            "Multiple model files found in context {}. Pass --model explicitly.",
            context_dir.display()
        ));
    }
    Ok(matches.remove(0))
}

type ContextManifest = (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<kapsl_core::HardwareRequirements>,
    Option<serde_json::Value>,
);

fn parse_context_manifest(context_dir: &Path) -> Result<ContextManifest, String> {
    let metadata_path = context_dir.join("metadata.json");
    if !metadata_path.exists() {
        return Ok((None, None, None, None, None, None));
    }

    let raw = fs::read_to_string(&metadata_path)
        .map_err(|e| format!("Failed to read {}: {}", metadata_path.display(), e))?;
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        format!(
            "Invalid metadata.json in {}: {}",
            metadata_path.display(),
            e
        )
    })?;
    let Some(obj) = value.as_object() else {
        return Err(format!(
            "metadata.json in {} must be a JSON object",
            metadata_path.display()
        ));
    };

    let project_name = obj
        .get("project_name")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let framework = obj
        .get("framework")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let version = obj
        .get("version")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let model_file = obj
        .get("model_file")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let hardware_requirements = obj
        .get("hardware_requirements")
        .cloned()
        .map(serde_json::from_value::<kapsl_core::HardwareRequirements>)
        .transpose()
        .map_err(|e| format!("Invalid hardware_requirements in metadata.json: {}", e))?;

    let metadata = obj.get("metadata").cloned();

    Ok((
        project_name,
        framework,
        version,
        model_file,
        hardware_requirements,
        metadata,
    ))
}

fn normalize_output_path_for_context(
    context_dir: &Path,
    output: Option<&Path>,
    project_name: &str,
) -> PathBuf {
    let mut output_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| context_dir.join(format!("{}.aimod", project_name)));
    if output_path.extension().and_then(|v| v.to_str()) != Some("aimod") {
        output_path.set_extension("aimod");
    }
    output_path
}

fn collect_existing_file_references_from_metadata(
    context_dir: &Path,
    value: &serde_yaml::Value,
    out: &mut HashSet<PathBuf>,
) {
    match value {
        serde_yaml::Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return;
            }
            let candidate = context_dir.join(trimmed);
            if !candidate.exists() || !candidate.is_file() {
                return;
            }
            if let Ok(relative) = candidate.strip_prefix(context_dir) {
                out.insert(relative.to_path_buf());
            }
        }
        serde_yaml::Value::Sequence(seq) => {
            for item in seq {
                collect_existing_file_references_from_metadata(context_dir, item, out);
            }
        }
        serde_yaml::Value::Mapping(map) => {
            for (_, item) in map {
                collect_existing_file_references_from_metadata(context_dir, item, out);
            }
        }
        _ => {}
    }
}

fn collect_context_files(
    context_dir: &Path,
    output_path: &Path,
    primary_model_ext: &str,
    keep_primary_model_files: &HashSet<PathBuf>,
) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let output_canonical = output_path.canonicalize().ok();
    let output_in_context_relative = output_path
        .strip_prefix(context_dir)
        .ok()
        .map(PathBuf::from);

    let primary_model_ext = primary_model_ext.trim().to_ascii_lowercase();
    let mut keep_onnx_stems_by_dir: HashMap<PathBuf, HashSet<String>> = HashMap::new();
    if primary_model_ext == "onnx" {
        for rel in keep_primary_model_files {
            if rel
                .extension()
                .and_then(|v| v.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("onnx"))
                .unwrap_or(false)
            {
                let dir = rel.parent().unwrap_or(Path::new("")).to_path_buf();
                let stem = rel
                    .file_stem()
                    .and_then(|v| v.to_str())
                    .unwrap_or("")
                    .to_string();
                if !stem.is_empty() {
                    keep_onnx_stems_by_dir.entry(dir).or_default().insert(stem);
                }
            }
        }
    }

    // First collect all file paths and (when applicable) the ONNX stems per directory, so we can
    // reliably decide which *.onnx_data* files belong to which model file.
    let mut all_files = Vec::new();
    let mut onnx_stems_by_dir: HashMap<PathBuf, Vec<String>> = HashMap::new();
    let mut primary_model_count_by_dir: HashMap<PathBuf, usize> = HashMap::new();
    let mut stack = vec![context_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|e| format!("Failed to read context directory {}: {}", dir.display(), e))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|e| format!("Failed to inspect {}: {}", path.display(), e))?;
            if file_type.is_dir() {
                let rel = path
                    .strip_prefix(context_dir)
                    .map_err(|e| format!("Failed to resolve dir path {}: {}", path.display(), e))?
                    .to_path_buf();
                if rel.components().any(|c| {
                    matches!(
                        c.as_os_str().to_str(),
                        Some(".git" | "__pycache__" | ".pytest_cache")
                    )
                }) {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }

            let relative = path
                .strip_prefix(context_dir)
                .map_err(|e| format!("Failed to resolve file path {}: {}", path.display(), e))?
                .to_path_buf();

            let ext = relative
                .extension()
                .and_then(|v| v.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();

            if !primary_model_ext.is_empty() && ext == primary_model_ext {
                let dir_rel = relative.parent().unwrap_or(Path::new("")).to_path_buf();
                *primary_model_count_by_dir.entry(dir_rel).or_insert(0) += 1;
            }

            if primary_model_ext == "onnx" && ext == "onnx" {
                let dir_rel = relative.parent().unwrap_or(Path::new("")).to_path_buf();
                let stem = relative
                    .file_stem()
                    .and_then(|v| v.to_str())
                    .unwrap_or("")
                    .to_string();
                if !stem.is_empty() {
                    onnx_stems_by_dir.entry(dir_rel).or_default().push(stem);
                }
            }

            all_files.push((path, relative));
        }
    }

    let mut files = Vec::new();
    for (path, relative) in all_files {
        // Skip common local/system noise.
        if relative
            .file_name()
            .and_then(|v| v.to_str())
            .map(|name| name == ".DS_Store")
            .unwrap_or(false)
        {
            continue;
        }
        if relative.components().any(|c| {
            matches!(
                c.as_os_str().to_str(),
                Some(".git" | "__pycache__" | ".pytest_cache")
            )
        }) {
            continue;
        }

        // metadata.json is generated for each build.
        if relative == Path::new("metadata.json") {
            continue;
        }
        // Avoid recursively embedding existing packages.
        if relative.extension().and_then(|ext| ext.to_str()) == Some("aimod") {
            continue;
        }
        if output_in_context_relative
            .as_ref()
            .map(|p| p == &relative)
            .unwrap_or(false)
        {
            continue;
        }
        if output_canonical
            .as_ref()
            .map(|out| out == &path)
            .unwrap_or(false)
        {
            continue;
        }

        let ext = relative
            .extension()
            .and_then(|v| v.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if !primary_model_ext.is_empty()
            && ext == primary_model_ext
            && !keep_primary_model_files.is_empty()
            && primary_model_count_by_dir
                .get(relative.parent().unwrap_or(Path::new("")))
                .copied()
                .unwrap_or(0)
                > 1
            && !keep_primary_model_files.contains(&relative)
        {
            continue;
        }

        if primary_model_ext == "onnx" && !keep_primary_model_files.is_empty() {
            if let Some(name) = relative.file_name().and_then(|v| v.to_str()) {
                if name.contains(".onnx_data") {
                    let dir_rel = relative.parent().unwrap_or(Path::new("")).to_path_buf();
                    if primary_model_count_by_dir
                        .get(&dir_rel)
                        .copied()
                        .unwrap_or(0)
                        <= 1
                    {
                        files.push((path, relative));
                        continue;
                    }
                    let stems = onnx_stems_by_dir.get(&dir_rel);
                    let mut associated_stem: Option<&str> = None;
                    if let Some(stems) = stems {
                        for stem in stems {
                            let needle = format!("{}.onnx_data", stem);
                            if name.starts_with(&needle) {
                                associated_stem = Some(stem.as_str());
                                break;
                            }
                        }
                        if associated_stem.is_none() && stems.len() == 1 {
                            associated_stem = stems.first().map(|s| s.as_str());
                        }
                    }

                    if let Some(stem) = associated_stem {
                        if let Some(keep) = keep_onnx_stems_by_dir.get(&dir_rel) {
                            if !keep.contains(stem) {
                                continue;
                            }
                        } else {
                            // We can associate this data shard with a concrete ONNX model in this
                            // directory, but none of the ONNX models in this directory are kept.
                            continue;
                        }
                    }
                }
            }
        }

        files.push((path, relative));
    }

    files.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(files)
}

fn create_kapsl_package_from_context(
    context_path: &Path,
    model_override: Option<&Path>,
    output_override: Option<&Path>,
    project_name_override: Option<&str>,
    framework_override: Option<&str>,
    version_override: Option<&str>,
    metadata_override: Option<&serde_json::Value>,
) -> Result<PackageKapslResponse, String> {
    let context_input = PathBuf::from(context_path);
    if !context_input.exists() {
        return Err(format!(
            "Build context does not exist: {}",
            context_input.display()
        ));
    }
    if !context_input.is_dir() {
        return Err(format!(
            "Build context must be a directory: {}",
            context_input.display()
        ));
    }
    let context_dir = context_input.canonicalize().map_err(|e| {
        format!(
            "Failed to resolve build context {}: {}",
            context_input.display(),
            e
        )
    })?;

    let (
        project_name_from_manifest,
        framework_from_manifest,
        version_from_manifest,
        model_file_from_manifest,
        hardware_requirements_from_manifest,
        metadata_from_manifest,
    ) = parse_context_manifest(&context_dir)?;

    let model_path = if let Some(model_path) = model_override {
        let candidate = if model_path.is_absolute() {
            model_path.to_path_buf()
        } else {
            let in_context = context_dir.join(model_path);
            if in_context.exists() {
                in_context
            } else {
                model_path.to_path_buf()
            }
        };
        if !candidate.exists() || !candidate.is_file() {
            return Err(format!(
                "Model file does not exist: {}",
                candidate.display()
            ));
        }
        candidate.canonicalize().map_err(|e| {
            format!(
                "Failed to resolve model file path {}: {}",
                candidate.display(),
                e
            )
        })?
    } else if let Some(model_file) = model_file_from_manifest.as_deref() {
        let candidate = context_dir.join(model_file);
        if !candidate.exists() || !candidate.is_file() {
            return Err(format!(
                "metadata.json model_file does not exist in context: {}",
                candidate.display()
            ));
        }
        candidate
    } else {
        find_model_file_in_context(&context_dir)?
    };

    if !model_path.starts_with(&context_dir) {
        return Err(format!(
            "Model path {} must be inside build context {}",
            model_path.display(),
            context_dir.display()
        ));
    }

    let model_file = model_path
        .strip_prefix(&context_dir)
        .map_err(|e| {
            format!(
                "Failed to compute model path relative to context {}: {}",
                model_path.display(),
                e
            )
        })?
        .to_string_lossy()
        .to_string();

    let project_name = project_name_override
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or(project_name_from_manifest)
        .or_else(|| {
            context_dir
                .file_name()
                .and_then(|v| v.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "kapsl-model".to_string());

    let framework = framework_override
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or(framework_from_manifest)
        .unwrap_or_else(|| infer_framework_from_model_path(&model_path));

    let version = version_override
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or(version_from_manifest)
        .unwrap_or_else(|| "1.0.0".to_string());

    let output_path =
        normalize_output_path_for_context(&context_dir, output_override, &project_name);
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create parent directory {}: {}",
                    parent.display(),
                    e
                )
            })?;
        }
    }

    let created_at = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("System clock error: {}", e))?
        .as_secs()
        .to_string();

    let metadata_value = metadata_override
        .cloned()
        .or(metadata_from_manifest)
        .map(|value| {
            serde_yaml::to_value(value)
                .map_err(|e| format!("Failed to convert metadata payload: {}", e))
        })
        .transpose()?;

    let primary_model_ext = model_path
        .extension()
        .and_then(|v| v.to_str())
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();

    let mut referenced_files = HashSet::new();
    referenced_files.insert(PathBuf::from(&model_file));
    if let Some(metadata) = metadata_value.as_ref() {
        collect_existing_file_references_from_metadata(
            &context_dir,
            metadata,
            &mut referenced_files,
        );
    }

    let mut keep_primary_model_files: HashSet<PathBuf> = HashSet::new();
    if !primary_model_ext.is_empty() {
        for rel in &referenced_files {
            let ext = rel
                .extension()
                .and_then(|v| v.to_str())
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase();
            if !ext.is_empty() && ext == primary_model_ext {
                keep_primary_model_files.insert(rel.to_path_buf());
            }
        }
    }

    let manifest = Manifest {
        project_name: project_name.clone(),
        framework: framework.clone(),
        version: version.clone(),
        created_at,
        model_file,
        metadata: metadata_value,
        hardware_requirements: hardware_requirements_from_manifest.unwrap_or_default(),
    };

    let output_file = File::create(&output_path).map_err(|e| {
        format!(
            "Failed to create output package {}: {}",
            output_path.display(),
            e
        )
    })?;
    let encoder = GzEncoder::new(output_file, Compression::default());
    let mut archive = Builder::new(encoder);

    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| format!("Failed to encode metadata.json: {}", e))?;
    append_tar_bytes_entry(&mut archive, "metadata.json", &manifest_bytes)?;

    for (absolute, relative) in collect_context_files(
        &context_dir,
        &output_path,
        &primary_model_ext,
        &keep_primary_model_files,
    )? {
        archive
            .append_path_with_name(&absolute, &relative)
            .map_err(|e| format!("Failed to add {} to archive: {}", relative.display(), e))?;
    }

    let encoder = archive
        .into_inner()
        .map_err(|e| format!("Failed to finalize tar archive: {}", e))?;
    encoder
        .finish()
        .map_err(|e| format!("Failed to finalize gzip stream: {}", e))?;

    let absolute_output_path = output_path.canonicalize().unwrap_or(output_path);
    Ok(PackageKapslResponse {
        status: "ok".to_string(),
        kapsl_path: absolute_output_path.to_string_lossy().to_string(),
        project_name,
        framework,
        version,
    })
}

fn push_kapsl_to_placeholder_remote(
    request: &PushKapslRequest,
) -> Result<PushKapslResponse, String> {
    let target = parse_model_target(&request.target)?;
    let input_path = PathBuf::from(request.kapsl_path.trim());
    if !input_path.exists() {
        return Err(format!(
            ".aimod file does not exist: {}",
            input_path.display()
        ));
    }
    if !input_path.is_file() {
        return Err(format!(
            "Provided .aimod path is not a file: {}",
            input_path.display()
        ));
    }

    if input_path.extension().and_then(|ext| ext.to_str()) != Some("aimod") {
        return Err(format!(
            "Push expects a .aimod file, got: {}",
            input_path.display()
        ));
    }

    let absolute_path = input_path.canonicalize().map_err(|e| {
        format!(
            "Failed to resolve kapsl path {}: {}",
            input_path.display(),
            e
        )
    })?;
    let filename = absolute_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("Invalid kapsl filename: {}", absolute_path.display()))?
        .to_string();

    let remote_url = resolved_remote_url(request.remote_url.as_deref());
    if is_oci_remote_url(&remote_url) {
        return push_kapsl_to_oci_remote(&remote_url, &absolute_path, &target, &filename);
    }
    let artifact_url = artifact_url_for_remote(&remote_url, &target);
    let mut remote_token = resolved_remote_token(&remote_url, request.remote_token.as_deref());
    let request_has_explicit_token = request
        .remote_token
        .as_deref()
        .is_some_and(|v| !v.trim().is_empty());

    let (mirrored_path, bytes_uploaded) = if is_default_placeholder_remote(&remote_url) {
        let mirrored_path = placeholder_remote_artifact_path(&target);
        let parent_dir = mirrored_path.parent().ok_or_else(|| {
            format!(
                "Invalid placeholder storage path: {}",
                mirrored_path.display()
            )
        })?;
        fs::create_dir_all(parent_dir).map_err(|e| {
            format!(
                "Failed to prepare placeholder remote directory {}: {}",
                parent_dir.display(),
                e
            )
        })?;
        let bytes_uploaded = fs::copy(&absolute_path, &mirrored_path).map_err(|e| {
            format!(
                "Failed to mirror .aimod into placeholder remote {}: {}",
                mirrored_path.display(),
                e
            )
        })?;

        (mirrored_path.to_string_lossy().to_string(), bytes_uploaded)
    } else {
        let bytes_uploaded =
            match push_kapsl_to_http_remote(&artifact_url, &absolute_path, remote_token.as_deref())
            {
                Ok(bytes) => bytes,
                Err(http_error) => {
                    if maybe_auto_login_for_remote(
                        &remote_url,
                        request_has_explicit_token,
                        request.interactive_login,
                        &mut remote_token,
                        &http_error,
                    )? {
                        push_kapsl_to_http_remote(
                            &artifact_url,
                            &absolute_path,
                            remote_token.as_deref(),
                        )
                        .map_err(|e| e.message)?
                    } else {
                        return Err(http_error.message);
                    }
                }
            };
        // For real remote backends there is no local mirror path.
        (artifact_url.clone(), bytes_uploaded)
    };

    Ok(PushKapslResponse {
        status: "ok".to_string(),
        remote_url,
        artifact_url,
        mirrored_path,
        bytes_uploaded,
        manifest_digest: None,
    })
}

fn pull_kapsl_from_placeholder_remote(
    request: &PullKapslRequest,
) -> Result<PullKapslResponse, String> {
    let target = parse_model_target(&request.target)?;
    let filename = format!("{}.aimod", target.model);

    let destination_dir = request
        .destination_dir
        .as_ref()
        .map(|path| PathBuf::from(path.trim()))
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&destination_dir).map_err(|e| {
        format!(
            "Failed to create destination directory {}: {}",
            destination_dir.display(),
            e
        )
    })?;

    let remote_url = resolved_remote_url(request.remote_url.as_deref());
    if is_oci_remote_url(&remote_url) {
        return pull_kapsl_from_oci_remote(
            &remote_url,
            &target,
            request.reference.as_deref(),
            &destination_dir,
        );
    }
    let output_path = destination_dir.join(&filename);
    let artifact_url = artifact_url_for_remote(&remote_url, &target);
    let mut remote_token = resolved_remote_token(&remote_url, request.remote_token.as_deref());
    let request_has_explicit_token = request
        .remote_token
        .as_deref()
        .is_some_and(|v| !v.trim().is_empty());
    let bytes_downloaded = if is_default_placeholder_remote(&remote_url) {
        let mirrored_path = placeholder_remote_artifact_path(&target);
        if !mirrored_path.exists() {
            return Err(format!(
                "Placeholder remote artifact not found: {} for target {}. Push the package first or set KAPSL_REMOTE_PLACEHOLDER_DIR.",
                mirrored_path.display(),
                target.as_string()
            ));
        }
        fs::copy(&mirrored_path, &output_path).map_err(|e| {
            format!(
                "Failed to pull placeholder remote artifact to {}: {}",
                output_path.display(),
                e
            )
        })?
    } else {
        let bytes = match pull_kapsl_from_http_remote(&artifact_url, remote_token.as_deref()) {
            Ok(bytes) => bytes,
            Err(http_error) => {
                if maybe_auto_login_for_remote(
                    &remote_url,
                    request_has_explicit_token,
                    request.interactive_login,
                    &mut remote_token,
                    &http_error,
                )? {
                    pull_kapsl_from_http_remote(&artifact_url, remote_token.as_deref())
                        .map_err(|e| e.message)?
                } else {
                    return Err(http_error.message);
                }
            }
        };
        fs::write(&output_path, &bytes).map_err(|e| {
            format!(
                "Failed to write pulled .aimod to {}: {}",
                output_path.display(),
                e
            )
        })?;
        bytes.len() as u64
    };

    let absolute_output_path = output_path.canonicalize().unwrap_or(output_path);

    Ok(PullKapslResponse {
        status: "ok".to_string(),
        remote_url,
        artifact_url,
        kapsl_path: absolute_output_path.to_string_lossy().to_string(),
        bytes_downloaded,
    })
}

fn parse_env_bool(key: &str) -> Option<bool> {
    let value = optional_env_var(key)?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn yaml_lookup<'a>(value: &'a serde_yaml::Value, path: &[&str]) -> Option<&'a serde_yaml::Value> {
    let mut current = value;
    for key in path {
        let mapping = match current {
            serde_yaml::Value::Mapping(mapping) => mapping,
            _ => return None,
        };
        current = mapping.get(*key)?;
    }
    Some(current)
}

fn yaml_bool(value: &serde_yaml::Value) -> Option<bool> {
    match value {
        serde_yaml::Value::Bool(val) => Some(*val),
        serde_yaml::Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn extension_key(workspace_id: &str, extension_id: &str) -> String {
    format!("{workspace_id}:{extension_id}")
}

fn rag_storage_root() -> PathBuf {
    optional_env_var(RAG_STORAGE_ROOT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("rag-data"))
}

#[derive(Debug, Clone)]
struct RuntimeStateLayout {
    rag_root: PathBuf,
    extensions_root: PathBuf,
    extensions_config_root: PathBuf,
    auth_store_path: PathBuf,
}

fn resolve_runtime_state_layout(args: &Args) -> RuntimeStateLayout {
    if let Some(state_dir) = args.state_dir.as_ref() {
        RuntimeStateLayout {
            rag_root: state_dir.join("rag-data"),
            extensions_root: state_dir.join("extensions"),
            extensions_config_root: state_dir.join("extensions-config"),
            auth_store_path: state_dir.join(DEFAULT_AUTH_STORE_FILENAME),
        }
    } else {
        let extensions_root = PathBuf::from(
            optional_env_var(EXTENSIONS_ROOT_ENV).unwrap_or_else(|| "extensions".to_string()),
        );
        let extensions_config_root = PathBuf::from(
            optional_env_var(EXT_CONFIG_ROOT_ENV)
                .unwrap_or_else(|| "extensions-config".to_string()),
        );
        RuntimeStateLayout {
            rag_root: rag_storage_root(),
            extensions_root,
            extensions_config_root,
            auth_store_path: resolve_auth_store_path(),
        }
    }
}

#[cfg(test)]
mod state_layout_tests {
    use super::*;

    #[test]
    fn test_state_dir_namespaces_runtime_state_paths() {
        let state_dir = PathBuf::from("state");
        let args = Args {
            model: vec![],
            transport: "socket".to_string(),
            socket: "dummy.sock".to_string(),
            bind: "127.0.0.1".to_string(),
            port: 9096,
            batch_size: 4,
            scheduler_queue_size: 256,
            scheduler_max_micro_batch: 4,
            scheduler_queue_delay_ms: 2,
            performance_profile: PerformanceProfile::Standard,
            metrics_port: 9095,
            http_bind: "127.0.0.1".to_string(),
            state_dir: Some(state_dir.clone()),
            topology: "data-parallel".to_string(),
            tp_degree: 1,
            worker: false,
            worker_model_id: None,
            onnx_memory_pattern: None,
            onnx_disable_cpu_mem_arena: None,
            onnx_session_buckets: None,
            onnx_bucket_dim_granularity: None,
            onnx_bucket_max_dims: None,
            onnx_peak_concurrency_hint: None,
            onnx_model_tuning: Vec::new(),
            shm_size_mb: None,
        };

        let layout = resolve_runtime_state_layout(&args);
        assert_eq!(layout.rag_root, state_dir.join("rag-data"));
        assert_eq!(layout.extensions_root, state_dir.join("extensions"));
        assert_eq!(
            layout.extensions_config_root,
            state_dir.join("extensions-config")
        );
        assert_eq!(
            layout.auth_store_path,
            state_dir.join(DEFAULT_AUTH_STORE_FILENAME)
        );
    }
}

fn normalize_tenant_id(tenant_id: Option<&str>) -> String {
    tenant_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("default")
        .to_string()
}

fn normalize_source_ids(
    source_id: Option<String>,
    source_ids: Option<Vec<String>>,
) -> Option<Vec<String>> {
    let mut combined = Vec::new();

    if let Some(source_id) = source_id {
        let trimmed = source_id.trim();
        if !trimmed.is_empty() {
            combined.push(trimmed.to_string());
        }
    }

    if let Some(source_ids) = source_ids {
        for source_id in source_ids {
            let trimmed = source_id.trim();
            if !trimmed.is_empty() {
                combined.push(trimmed.to_string());
            }
        }
    }

    if combined.is_empty() {
        return None;
    }

    combined.sort();
    combined.dedup();
    Some(combined)
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn embed_text_for_rag_with_dim(text: &str, dimension: usize) -> Vec<f32> {
    if dimension == 0 {
        return Vec::new();
    }
    let mut embedding = vec![0.0f32; dimension];
    let mut token_count = 0usize;

    for token in text
        .split_whitespace()
        .map(|token| token.trim_matches(|ch: char| !ch.is_alphanumeric()))
        .filter(|token| !token.is_empty())
    {
        let normalized = token.to_ascii_lowercase();
        let hash = fnv1a_64(normalized.as_bytes());
        let index = (hash % dimension as u64) as usize;
        let sign = if (hash & 1) == 0 { 1.0 } else { -1.0 };
        embedding[index] += sign;
        token_count += 1;
    }

    if token_count == 0 {
        return embedding;
    }

    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }
    embedding
}

fn embed_text_for_rag(text: &str) -> Vec<f32> {
    embed_text_for_rag_with_dim(text, RAG_EMBEDDING_DIM)
}

fn chunk_document_text(text: &str) -> Vec<(i64, String)> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.is_empty() {
        return Vec::new();
    }

    let chunk_size = RAG_CHUNK_SIZE.max(1);
    let overlap = RAG_CHUNK_OVERLAP.min(chunk_size.saturating_sub(1));
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut index = 0i64;

    while start < tokens.len() {
        let end = (start + chunk_size).min(tokens.len());
        let chunk = tokens[start..end].join(" ");
        chunks.push((index, chunk));
        if end >= tokens.len() {
            break;
        }
        start = end.saturating_sub(overlap);
        index += 1;
    }

    chunks
}

fn is_textual_content_type(content_type: &str) -> bool {
    let lowered = content_type.trim().to_ascii_lowercase();
    lowered.starts_with("text/")
        || lowered.contains("json")
        || lowered.contains("xml")
        || lowered.contains("yaml")
        || lowered.contains("markdown")
        || lowered.contains("csv")
}

fn decode_text_document_payload(payload: &DocumentPayload) -> Result<(Vec<u8>, String), String> {
    let bytes = BASE64
        .decode(payload.bytes_b64.as_bytes())
        .map_err(|error| format!("invalid base64 document payload: {}", error))?;

    if bytes.is_empty() {
        return Err("document payload is empty".to_string());
    }

    match String::from_utf8(bytes.clone()) {
        Ok(text) => {
            if text.trim().is_empty() {
                Err("decoded document has no text content".to_string())
            } else {
                Ok((bytes, text))
            }
        }
        Err(_) if is_textual_content_type(&payload.content_type) => {
            let text = String::from_utf8_lossy(&bytes).to_string();
            if text.trim().is_empty() {
                Err("decoded document has no text content".to_string())
            } else {
                Ok((bytes, text))
            }
        }
        Err(_) => Err(format!(
            "unsupported non-text content type `{}`",
            payload.content_type
        )),
    }
}

fn merged_document_metadata(
    payload: &DocumentPayload,
    source_id: &str,
    doc_id: &str,
) -> HashMap<String, String> {
    let mut metadata = payload.metadata.clone();
    metadata.insert("source".to_string(), source_id.to_string());
    metadata.insert("doc_id".to_string(), doc_id.to_string());
    metadata.insert("document_id".to_string(), doc_id.to_string());
    metadata
}

async fn delete_document_from_rag(
    rag_state: &RagRuntimeState,
    tenant_id: &str,
    workspace_id: &str,
    source_id: &str,
    doc_id: &str,
) -> Result<(), String> {
    rag_state
        .vector_store
        .delete_by_doc(tenant_id, workspace_id, source_id, doc_id)
        .await
        .map_err(|error| format!("failed to delete document from vector store: {}", error))?;

    rag_state
        .doc_store
        .delete(&kapsl_rag::storage::DocKey {
            tenant_id: tenant_id.to_string(),
            workspace_id: workspace_id.to_string(),
            source_id: source_id.to_string(),
            doc_id: doc_id.to_string(),
        })
        .map_err(|error| format!("failed to delete document from doc store: {}", error))?;

    Ok(())
}

async fn ingest_document_payload_into_rag(
    rag_state: &RagRuntimeState,
    tenant_id: &str,
    workspace_id: &str,
    source_id: &str,
    payload: &DocumentPayload,
) -> Result<usize, String> {
    let doc_id = payload.id.trim();
    if doc_id.is_empty() {
        return Err("document id is empty".to_string());
    }

    let (bytes, text) = decode_text_document_payload(payload)?;

    delete_document_from_rag(rag_state, tenant_id, workspace_id, source_id, doc_id).await?;

    rag_state
        .doc_store
        .put(
            &kapsl_rag::storage::DocKey {
                tenant_id: tenant_id.to_string(),
                workspace_id: workspace_id.to_string(),
                source_id: source_id.to_string(),
                doc_id: doc_id.to_string(),
            },
            &bytes,
        )
        .map_err(|error| format!("failed to persist document bytes: {}", error))?;

    let chunks = chunk_document_text(&text);
    if chunks.is_empty() {
        return Ok(0);
    }
    let chunk_count = chunks.len();

    let base_metadata = merged_document_metadata(payload, source_id, doc_id);
    let acl = AccessControl {
        allow_users: payload.acl.allow_users.clone(),
        allow_groups: payload.acl.allow_groups.clone(),
        deny_users: payload.acl.deny_users.clone(),
        deny_groups: payload.acl.deny_groups.clone(),
    };

    let mut embedded_chunks = Vec::with_capacity(chunks.len());
    for (chunk_index, chunk_text) in chunks {
        let mut metadata = base_metadata.clone();
        metadata.insert("chunk_index".to_string(), chunk_index.to_string());
        let embedding = embed_text_for_rag(&chunk_text);
        embedded_chunks.push(EmbeddedChunk {
            id: format!("{doc_id}:{chunk_index}"),
            tenant_id: tenant_id.to_string(),
            workspace_id: workspace_id.to_string(),
            source_id: source_id.to_string(),
            doc_id: doc_id.to_string(),
            chunk_index,
            text: chunk_text,
            embedding,
            metadata,
            acl: acl.clone(),
        });
    }

    rag_state
        .vector_store
        .upsert(embedded_chunks)
        .await
        .map_err(|error| format!("failed to upsert vector chunks: {}", error))?;

    Ok(chunk_count)
}

fn select_sync_source_id(
    explicit_source_id: Option<String>,
    connector_config: serde_json::Value,
    client: &mut ConnectorClient<ConnectorRuntimeHandle>,
) -> Result<String, String> {
    if let Some(source_id) = explicit_source_id {
        let trimmed = source_id.trim();
        if trimmed.is_empty() {
            return Err("source_id cannot be empty".to_string());
        }
        return Ok(trimmed.to_string());
    }

    let sources_response = client
        .request(ConnectorRequestKind::ListSources {
            config: connector_config,
        })
        .map_err(|error| format!("failed to list connector sources: {}", error))?;

    match sources_response.kind {
        ConnectorResponseKind::Err(error) => Err(error.message),
        ConnectorResponseKind::Ok(ConnectorResult::Sources(sources)) => {
            pick_default_source_id(&sources)
        }
        _ => Err("connector returned unexpected response for ListSources".to_string()),
    }
}

fn pick_default_source_id(sources: &[SourceDescriptor]) -> Result<String, String> {
    let source = sources
        .first()
        .ok_or_else(|| "connector returned no sources".to_string())?;
    let source_id = source.id.trim();
    if source_id.is_empty() {
        return Err("connector returned an empty source id".to_string());
    }
    Ok(source_id.to_string())
}

fn parse_infer_rag_options(
    payload: &serde_json::Value,
) -> Result<Option<InferRagOptions>, RagAugmentError> {
    let Some(raw_rag) = payload.get("rag") else {
        return Ok(None);
    };

    let options: InferRagOptions = serde_json::from_value(raw_rag.clone()).map_err(|error| {
        RagAugmentError::bad_request(format!("Invalid `rag` infer options: {}", error))
    })?;

    validate_infer_rag_options(Some(options))
}

fn validate_infer_rag_options(
    options: Option<InferRagOptions>,
) -> Result<Option<InferRagOptions>, RagAugmentError> {
    let Some(options) = options else {
        return Ok(None);
    };

    if options.enabled == Some(false) {
        return Ok(None);
    }

    if options.workspace_id.trim().is_empty() {
        return Err(RagAugmentError::bad_request(
            "`rag.workspace_id` is required",
        ));
    }

    if matches!(options.top_k, Some(0)) {
        return Err(RagAugmentError::bad_request(
            "`rag.top_k` must be greater than 0",
        ));
    }

    if matches!(options.max_context_tokens, Some(0)) {
        return Err(RagAugmentError::bad_request(
            "`rag.max_context_tokens` must be greater than 0",
        ));
    }

    if matches!(options.max_chunks, Some(0)) {
        return Err(RagAugmentError::bad_request(
            "`rag.max_chunks` must be greater than 0",
        ));
    }

    if matches!(options.max_per_source, Some(0)) {
        return Err(RagAugmentError::bad_request(
            "`rag.max_per_source` must be greater than 0",
        ));
    }

    Ok(Some(options))
}

#[derive(Debug, Deserialize)]
struct InferPayloadEnvelope<T> {
    #[serde(default)]
    rag: Option<InferRagOptions>,
    #[serde(flatten)]
    request: T,
}

fn parse_expected_dtype(value: Option<&String>) -> Option<TensorDtype> {
    let raw = value?;
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("unknown") {
        return None;
    }
    trimmed.parse::<TensorDtype>().ok()
}

fn validate_tensor_against_model_spec(
    label: &str,
    tensor: &BinaryTensorPacket,
    expected_shape: &[i64],
    expected_dtype: Option<TensorDtype>,
) -> Result<(), String> {
    if let Some(expected_dtype) = expected_dtype {
        if tensor.dtype != expected_dtype {
            return Err(format!(
                "{} dtype mismatch: expected `{}`, got `{}`",
                label, expected_dtype, tensor.dtype
            ));
        }
    }

    if expected_shape.is_empty() {
        return Ok(());
    }

    if tensor.shape.len() != expected_shape.len() {
        return Err(format!(
            "{} rank mismatch: expected {} dims {:?}, got {} dims {:?}",
            label,
            expected_shape.len(),
            expected_shape,
            tensor.shape.len(),
            tensor.shape
        ));
    }

    for (index, (actual, expected)) in tensor.shape.iter().zip(expected_shape.iter()).enumerate() {
        // <= 0 is treated as dynamic/unknown in model metadata.
        if *expected <= 0 {
            continue;
        }
        if actual != expected {
            return Err(format!(
                "{} shape mismatch at dim {}: expected {}, got {} (expected shape {:?}, got {:?})",
                label, index, expected, actual, expected_shape, tensor.shape
            ));
        }
    }

    Ok(())
}

fn validate_inference_request_against_model_info(
    request: &InferenceRequest,
    model_info: &EngineModelInfo,
) -> Result<(), String> {
    if model_info.input_names.is_empty() {
        return Ok(());
    }

    let mut input_index: HashMap<&str, usize> = HashMap::new();
    for (index, name) in model_info.input_names.iter().enumerate() {
        input_index.insert(name.as_str(), index);
    }

    let primary_name = model_info.input_names[0].as_str();
    let primary_shape = model_info.input_shapes.first().cloned().unwrap_or_default();
    let primary_dtype = parse_expected_dtype(model_info.input_dtypes.first());
    validate_tensor_against_model_spec(
        &format!("primary input `{}`", primary_name),
        &request.input,
        &primary_shape,
        primary_dtype,
    )?;

    for additional in &request.additional_inputs {
        let Some(&index) = input_index.get(additional.name.as_str()) else {
            return Err(format!(
                "unknown additional input `{}`. Model inputs: {}",
                additional.name,
                model_info.input_names.join(", ")
            ));
        };
        let expected_shape = model_info
            .input_shapes
            .get(index)
            .cloned()
            .unwrap_or_default();
        let expected_dtype = parse_expected_dtype(model_info.input_dtypes.get(index));
        validate_tensor_against_model_spec(
            &format!("additional input `{}`", additional.name),
            &additional.tensor,
            &expected_shape,
            expected_dtype,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn query_rag_chunks(
    rag_state: &RagRuntimeState,
    workspace_id: &str,
    tenant_id: Option<&str>,
    query: &str,
    source_id: Option<String>,
    source_ids: Option<Vec<String>>,
    top_k: Option<usize>,
    min_score: Option<f32>,
    allowed_users: Vec<String>,
    allowed_groups: Vec<String>,
) -> Result<Vec<RagChunk>, RagAugmentError> {
    let query = query.trim();
    if query.is_empty() {
        return Err(RagAugmentError::bad_request("RAG query cannot be empty"));
    }

    let source_ids = normalize_source_ids(source_id, source_ids);
    let top_k = top_k.unwrap_or(RAG_DEFAULT_TOP_K).clamp(1, RAG_MAX_TOP_K);
    let min_score = min_score.unwrap_or(0.0);
    let query_embedding = embed_text_for_rag(query);
    if query_embedding.is_empty() {
        return Ok(Vec::new());
    }

    let query_request = VectorQuery {
        query_embedding,
        top_k,
        tenant_id: normalize_tenant_id(tenant_id),
        workspace_id: workspace_id.to_string(),
        source_ids,
        allowed_users,
        allowed_groups,
        min_score,
    };

    let results = rag_state
        .vector_store
        .query(query_request)
        .await
        .map_err(|error| {
            RagAugmentError::internal(format!("Failed to query vector store: {}", error))
        })?;

    Ok(results
        .into_iter()
        .map(|result| RagChunk {
            id: result.chunk.id,
            text: result.chunk.text,
            score: result.score,
            metadata: result.chunk.metadata,
        })
        .collect())
}

fn inject_rag_context_into_prompt(prompt: &str, context: &str) -> String {
    let user_marker = "<start_of_turn>user\n";
    let end_marker = "<end_of_turn>";
    if let Some(user_start) = prompt.rfind(user_marker) {
        let content_start = user_start + user_marker.len();
        if prompt[content_start..].contains(end_marker) {
            let mut output = String::with_capacity(prompt.len() + context.len() + 160);
            output.push_str(&prompt[..content_start]);
            output.push_str("Use the retrieved context below when relevant.\n\n");
            output.push_str("[Retrieved Context]\n");
            output.push_str(context);
            output.push_str("\n[/Retrieved Context]\n\n");
            output.push_str(&prompt[content_start..]);
            return output;
        }
    }

    format!(
        "Use the retrieved context below when relevant.\n\n[Retrieved Context]\n{}\n[/Retrieved Context]\n\n{}",
        context, prompt
    )
}

async fn augment_inference_request_with_rag(
    request: &mut InferenceRequest,
    rag_options: &InferRagOptions,
    rag_state: &RagRuntimeState,
) -> Result<usize, RagAugmentError> {
    if request.input.dtype != TensorDtype::Utf8 {
        return Err(RagAugmentError::bad_request(
            "`rag` is currently supported only for `string` infer inputs",
        ));
    }

    let prompt = String::from_utf8(request.input.data.clone()).map_err(|error| {
        RagAugmentError::bad_request(format!("failed to decode UTF-8 prompt: {}", error))
    })?;

    let retrieved_chunks = query_rag_chunks(
        rag_state,
        &rag_options.workspace_id,
        rag_options.tenant_id.as_deref(),
        &prompt,
        rag_options.source_id.clone(),
        rag_options.source_ids.clone(),
        rag_options.top_k,
        rag_options.min_score,
        Vec::new(),
        Vec::new(),
    )
    .await?;

    if retrieved_chunks.is_empty() {
        return Ok(0);
    }

    let mut prompt_config = RagPromptConfig {
        max_context_tokens: RAG_CONTEXT_MAX_TOKENS,
        citation_style: CitationStyle::BracketedNumber,
        ..RagPromptConfig::default()
    };
    if let Some(max_context_tokens) = rag_options.max_context_tokens {
        prompt_config.max_context_tokens = max_context_tokens;
    }
    if let Some(max_chunks) = rag_options.max_chunks {
        prompt_config.max_chunks = max_chunks;
    }
    if let Some(max_per_source) = rag_options.max_per_source {
        prompt_config.max_per_source = max_per_source;
    }
    if let Some(min_score) = rag_options.min_score {
        prompt_config.min_score = min_score;
    }

    let rag_prompt = build_rag_prompt(&retrieved_chunks, &prompt_config, &WhitespaceTokenCounter);
    if rag_prompt.context.trim().is_empty() {
        return Ok(0);
    }

    let augmented_prompt = inject_rag_context_into_prompt(&prompt, &rag_prompt.context);
    request.input.data = augmented_prompt.into_bytes();
    request.input.shape = vec![1, request.input.data.len() as i64];

    Ok(rag_prompt.used_chunks.len())
}

fn manifest_llm_flag(manifest: &Manifest, key: &str) -> Option<bool> {
    let meta = manifest.metadata.as_ref()?;
    let value = yaml_lookup(meta, &["llm", key])?;
    yaml_bool(value)
}

fn manifest_llm_pipeline_stages(manifest: &Manifest) -> Option<Vec<String>> {
    let meta = manifest.metadata.as_ref()?;
    let value = yaml_lookup(meta, &["llm", "pipeline", "stages"])?;
    match value {
        serde_yaml::Value::Sequence(items) => {
            let stages: Vec<String> = items
                .iter()
                .filter_map(|item| item.as_str().map(|s| s.to_string()))
                .collect();
            if stages.is_empty() {
                None
            } else {
                Some(stages)
            }
        }
        _ => None,
    }
}

fn resolve_isolate_process(manifest: &Manifest) -> bool {
    if let Some(env) = parse_env_bool(LLM_ISOLATE_PROCESS_ENV) {
        return env;
    }
    manifest_llm_flag(manifest, "isolate_process").unwrap_or(false)
}

fn resolve_scheduler_tuning_for_framework(
    manifest: &Manifest,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
) -> (usize, u64) {
    if !manifest.framework.eq_ignore_ascii_case("llm") {
        return (scheduler_max_micro_batch, scheduler_queue_delay_ms);
    }

    if env_flag(LLM_ALLOW_SCHEDULER_MICROBATCH_ENV) {
        return (scheduler_max_micro_batch, scheduler_queue_delay_ms);
    }

    let mut resolved_micro_batch = scheduler_max_micro_batch.max(1);
    let mut resolved_queue_delay_ms = scheduler_queue_delay_ms;

    if resolved_micro_batch > 1 {
        log::info!(
            "Framework=llm: overriding scheduler_max_micro_batch {} -> 1 to avoid outer micro-batch serialization. Set {}=1 to keep configured value.",
            resolved_micro_batch,
            LLM_ALLOW_SCHEDULER_MICROBATCH_ENV
        );
        resolved_micro_batch = 1;
    }
    if resolved_queue_delay_ms > 0 {
        log::info!(
            "Framework=llm: overriding scheduler_queue_delay_ms {} -> 0 to reduce queueing delay. Set {}=1 to keep configured value.",
            resolved_queue_delay_ms,
            LLM_ALLOW_SCHEDULER_MICROBATCH_ENV
        );
        resolved_queue_delay_ms = 0;
    }

    (resolved_micro_batch, resolved_queue_delay_ms)
}

fn parse_queue_overflow_policy_literal(
    value: &str,
) -> Option<kapsl_scheduler::QueueOverflowPolicy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "block" | "blocking" => Some(kapsl_scheduler::QueueOverflowPolicy::Block),
        "drop_newest" | "drop-newest" | "latest_only" | "latest-only" | "latest" => {
            Some(kapsl_scheduler::QueueOverflowPolicy::DropNewest)
        }
        "drop_oldest" | "drop-oldest" => Some(kapsl_scheduler::QueueOverflowPolicy::DropOldest),
        _ => None,
    }
}

fn manifest_queue_overflow_policy(
    manifest: &Manifest,
) -> Option<kapsl_scheduler::QueueOverflowPolicy> {
    let meta = manifest.metadata.as_ref()?;
    for path in [
        &["runtime", "server", "queue_overflow_policy"][..],
        &["runtime", "queue_overflow_policy"][..],
        &["scheduler", "queue_overflow_policy"][..],
        &["queue_overflow_policy"][..],
    ] {
        if let Some(value) = yaml_lookup(meta, path) {
            if let Some(raw) = value.as_str() {
                if let Some(policy) = parse_queue_overflow_policy_literal(raw) {
                    return Some(policy);
                }
            }
        }
    }
    None
}

fn resolve_queue_overflow_policy(manifest: &Manifest) -> kapsl_scheduler::QueueOverflowPolicy {
    if let Some(value) = optional_env_var_alias(
        SCHEDULER_QUEUE_OVERFLOW_POLICY_ENV,
        LEGACY_SCHEDULER_QUEUE_OVERFLOW_POLICY_ENV,
    ) {
        if let Some(policy) = parse_queue_overflow_policy_literal(&value) {
            return policy;
        }
        log::warn!(
            "Invalid {} value '{}'; expected block|drop_newest|drop_oldest. Falling back to manifest/default.",
            SCHEDULER_QUEUE_OVERFLOW_POLICY_ENV,
            value
        );
    }

    manifest_queue_overflow_policy(manifest).unwrap_or(kapsl_scheduler::QueueOverflowPolicy::Block)
}

fn log_queue_policy_caveat(policy: kapsl_scheduler::QueueOverflowPolicy) {
    if matches!(policy, kapsl_scheduler::QueueOverflowPolicy::DropOldest) {
        log::warn!(
            "Scheduler queue policy 'drop_oldest' evicts the oldest queued request when capacity is reached"
        );
    }
}

struct EffectiveTopologyChoice {
    mesh_topology: kapsl_hal::device_mesh::MeshTopology,
    worker_topology: &'static str,
    worker_tp_degree: usize,
    use_pipeline_backend: bool,
}

fn resolve_effective_topology_choice(
    manifest: &Manifest,
    requested_topology: &str,
    requested_tp_degree: usize,
    pipeline_stages: Option<&[String]>,
) -> EffectiveTopologyChoice {
    use kapsl_hal::device_mesh::MeshTopology;

    let requested = requested_topology.trim().to_ascii_lowercase();
    let requested_degree = requested_tp_degree.max(1);
    let pipeline_stage_count = pipeline_stages.map(|stages| stages.len()).unwrap_or(0);
    let pipeline_ready = manifest.framework == "llm" && pipeline_stage_count > 0;

    match requested.as_str() {
        "pipeline" | "pipeline-parallel" => {
            if pipeline_ready {
                if requested_degree != pipeline_stage_count {
                    log::warn!(
                        "Ignoring --tp-degree={} for pipeline mode; using metadata stage count={}",
                        requested_degree,
                        pipeline_stage_count
                    );
                }
                EffectiveTopologyChoice {
                    mesh_topology: MeshTopology::PipelineParallel {
                        stages: pipeline_stage_count,
                    },
                    worker_topology: "pipeline-parallel",
                    worker_tp_degree: pipeline_stage_count,
                    use_pipeline_backend: true,
                }
            } else {
                log::warn!(
                    "Pipeline topology requested but no usable LLM pipeline metadata found; falling back to data-parallel"
                );
                EffectiveTopologyChoice {
                    mesh_topology: MeshTopology::DataParallel,
                    worker_topology: "data-parallel",
                    worker_tp_degree: 1,
                    use_pipeline_backend: false,
                }
            }
        }
        "mixed" => {
            if pipeline_ready {
                log::warn!(
                    "Mixed topology is not fully implemented; using pipeline-parallel execution based on metadata stages"
                );
                EffectiveTopologyChoice {
                    mesh_topology: MeshTopology::PipelineParallel {
                        stages: pipeline_stage_count,
                    },
                    worker_topology: "pipeline-parallel",
                    worker_tp_degree: pipeline_stage_count,
                    use_pipeline_backend: true,
                }
            } else {
                log::warn!(
                    "Mixed topology requested but no usable LLM pipeline metadata found; falling back to data-parallel"
                );
                EffectiveTopologyChoice {
                    mesh_topology: MeshTopology::DataParallel,
                    worker_topology: "data-parallel",
                    worker_tp_degree: 1,
                    use_pipeline_backend: false,
                }
            }
        }
        "tensor-parallel" => {
            log::warn!(
                "Tensor-parallel topology is not fully implemented in backend execution; falling back to data-parallel"
            );
            EffectiveTopologyChoice {
                mesh_topology: MeshTopology::DataParallel,
                worker_topology: "data-parallel",
                worker_tp_degree: 1,
                use_pipeline_backend: false,
            }
        }
        _ => EffectiveTopologyChoice {
            mesh_topology: MeshTopology::DataParallel,
            worker_topology: "data-parallel",
            worker_tp_degree: 1,
            use_pipeline_backend: false,
        },
    }
}

fn select_mesh_devices(
    requirements: &kapsl_core::HardwareRequirements,
    device_info: &DeviceInfo,
) -> Result<Vec<kapsl_hal::device::Device>, String> {
    let strategy = requirements
        .strategy
        .as_deref()
        .unwrap_or("")
        .to_ascii_lowercase();
    let allow_multi = matches!(
        strategy.as_str(),
        "pool"
            | "round-robin"
            | "data-parallel"
            | "pipeline"
            | "pipeline-parallel"
            | "tensor-parallel"
            | "auto"
    );
    let mut preferred_device_id = requirements.device_id.map(|id| id as usize);
    if allow_multi {
        preferred_device_id = None;
    }
    let mut providers: Vec<String> = Vec::new();
    if let Some(pref) = &requirements.preferred_provider {
        providers.push(pref.clone());
    }
    providers.extend(requirements.fallback_providers.clone());
    let provider_policy = provider_policy();
    let cpu_only_or_empty = providers.is_empty()
        || providers
            .iter()
            .all(|provider| matches!(provider.trim().to_ascii_lowercase().as_str(), "" | "cpu"));
    if provider_policy != "manifest" && cpu_only_or_empty {
        let mut push_if_missing = |provider: &str| {
            if providers
                .iter()
                .all(|candidate| !candidate.eq_ignore_ascii_case(provider))
            {
                providers.push(provider.to_string());
            }
        };
        if device_info.has_cuda {
            push_if_missing("tensorrt");
            push_if_missing("cuda");
        }
        if device_info.has_metal {
            push_if_missing("coreml");
        }
        if device_info.has_rocm {
            push_if_missing("rocm");
        }
        if device_info.has_directml {
            push_if_missing("directml");
        }
        push_if_missing("cpu");
    }

    let mut selected: Vec<kapsl_hal::device::Device> = Vec::new();
    if !providers.is_empty() {
        for provider in &providers {
            let provider_lower = provider.to_lowercase();
            let backend_key = match provider_lower.as_str() {
                "tensorrt" => "cuda".to_string(),
                "coreml" => "metal".to_string(),
                other => other.to_string(),
            };
            let mut matches: Vec<kapsl_hal::device::Device> = device_info
                .devices
                .iter()
                .filter(|d| d.backend.to_string().to_lowercase() == backend_key)
                .cloned()
                .collect();
            if backend_key != "cpu" {
                if let Some(min_vram) = requirements.min_vram_mb {
                    matches.retain(|d| d.memory_mb >= min_vram);
                }
                if backend_key == "cuda" {
                    if let Some(min_ver) = &requirements.min_cuda_version {
                        matches.retain(|d| {
                            d.cuda_version
                                .as_ref()
                                .map(|ver| ver >= min_ver)
                                .unwrap_or(false)
                        });
                    }
                }
            }
            if let Some(dev_id) = preferred_device_id {
                if backend_key != "cpu" {
                    matches.retain(|d| d.id == dev_id);
                }
            }
            if !matches.is_empty() {
                selected = matches;
                break;
            }
        }
    } else {
        selected = device_info.devices.clone();
        if let Some(dev_id) = preferred_device_id {
            selected.retain(|d| d.id == dev_id);
        }
    }

    if selected.is_empty() {
        if let Some(dev_id) = preferred_device_id {
            if providers.is_empty() {
                return Err(format!(
                    "Device ID {} not found in available devices",
                    dev_id
                ));
            }
            return Err(format!(
                "Device ID {} not found for providers {:?}",
                dev_id, providers
            ));
        }
        selected = device_info.devices.clone();
    }

    Ok(selected)
}

#[derive(Debug, Clone, Default)]
struct RuntimeSamples {
    process_memory_bytes: usize,
    total_system_memory_bytes: Option<usize>,
    gpu_utilization: f64,
    gpu_memory_bytes: Option<usize>,
    gpu_memory_total_bytes: Option<usize>,
    collected_at_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimePressureState {
    Normal = 0,
    Conserve = 1,
    Emergency = 2,
}

impl RuntimePressureState {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Conserve,
            2 => Self::Emergency,
            _ => Self::Normal,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Conserve => "conserve",
            Self::Emergency => "emergency",
        }
    }
}

#[derive(Debug, Clone)]
struct RuntimePressureConfig {
    memory_conserve_ratio: f64,
    memory_emergency_ratio: f64,
    gpu_util_conserve_ratio: f64,
    gpu_util_emergency_ratio: f64,
    gpu_mem_conserve_ratio: f64,
    gpu_mem_emergency_ratio: f64,
    conserve_max_new_tokens: Option<u32>,
    emergency_max_new_tokens: Option<u32>,
}

impl RuntimePressureConfig {
    fn from_env() -> Self {
        let memory_conserve_pct = optional_env_var(PRESSURE_MEMORY_CONSERVE_PCT_ENV)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(80.0)
            .clamp(0.0, 100.0);
        let memory_emergency_pct = optional_env_var(PRESSURE_MEMORY_EMERGENCY_PCT_ENV)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(90.0)
            .clamp(memory_conserve_pct, 100.0);
        let gpu_util_conserve_pct = optional_env_var(PRESSURE_GPU_UTIL_CONSERVE_PCT_ENV)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(85.0)
            .clamp(0.0, 100.0);
        let gpu_util_emergency_pct = optional_env_var(PRESSURE_GPU_UTIL_EMERGENCY_PCT_ENV)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(95.0)
            .clamp(gpu_util_conserve_pct, 100.0);
        let gpu_mem_conserve_pct = optional_env_var(PRESSURE_GPU_MEM_CONSERVE_PCT_ENV)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(85.0)
            .clamp(0.0, 100.0);
        let gpu_mem_emergency_pct = optional_env_var(PRESSURE_GPU_MEM_EMERGENCY_PCT_ENV)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(95.0)
            .clamp(gpu_mem_conserve_pct, 100.0);
        let conserve_max_new_tokens = optional_env_var(PRESSURE_CONSERVE_MAX_TOKENS_ENV)
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v > 0);
        let emergency_max_new_tokens = optional_env_var(PRESSURE_EMERGENCY_MAX_TOKENS_ENV)
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v > 0)
            .or(Some(128));

        Self {
            memory_conserve_ratio: memory_conserve_pct / 100.0,
            memory_emergency_ratio: memory_emergency_pct / 100.0,
            gpu_util_conserve_ratio: gpu_util_conserve_pct / 100.0,
            gpu_util_emergency_ratio: gpu_util_emergency_pct / 100.0,
            gpu_mem_conserve_ratio: gpu_mem_conserve_pct / 100.0,
            gpu_mem_emergency_ratio: gpu_mem_emergency_pct / 100.0,
            conserve_max_new_tokens,
            emergency_max_new_tokens,
        }
    }

    fn max_new_tokens_cap(&self, state: RuntimePressureState) -> Option<u32> {
        match state {
            RuntimePressureState::Normal => None,
            RuntimePressureState::Conserve => self.conserve_max_new_tokens,
            RuntimePressureState::Emergency => self.emergency_max_new_tokens,
        }
    }
}

fn evaluate_runtime_pressure_state(
    samples: &RuntimeSamples,
    config: &RuntimePressureConfig,
) -> RuntimePressureState {
    let process_memory_ratio = samples
        .total_system_memory_bytes
        .filter(|total| *total > 0)
        .map(|total| (samples.process_memory_bytes as f64 / total as f64).clamp(0.0, 1.0));

    let gpu_util_ratio = samples.gpu_utilization.clamp(0.0, 1.0);
    let gpu_mem_ratio = match (samples.gpu_memory_bytes, samples.gpu_memory_total_bytes) {
        (Some(used), Some(total)) if total > 0 => {
            Some((used as f64 / total as f64).clamp(0.0, 1.0))
        }
        _ => None,
    };

    let emergency = process_memory_ratio
        .is_some_and(|ratio| ratio >= config.memory_emergency_ratio)
        || gpu_util_ratio >= config.gpu_util_emergency_ratio
        || gpu_mem_ratio.is_some_and(|ratio| ratio >= config.gpu_mem_emergency_ratio);
    if emergency {
        return RuntimePressureState::Emergency;
    }

    let conserve = process_memory_ratio.is_some_and(|ratio| ratio >= config.memory_conserve_ratio)
        || gpu_util_ratio >= config.gpu_util_conserve_ratio
        || gpu_mem_ratio.is_some_and(|ratio| ratio >= config.gpu_mem_conserve_ratio);
    if conserve {
        return RuntimePressureState::Conserve;
    }

    RuntimePressureState::Normal
}

#[derive(Debug, Clone)]
struct ThroughputSample {
    last_total: u64,
    last_timestamp: Instant,
    throughput: f64,
}

type InterModelRoutes = HashMap<String, Vec<String>>;

#[derive(Debug, Clone)]
struct InterModelRelayState {
    routes: InterModelRoutes,
    min_interval: Duration,
    last_relay_at: Arc<Mutex<HashMap<u32, Instant>>>,
}

impl InterModelRelayState {
    fn from_env() -> Self {
        let routes_raw =
            optional_env_var_alias(INTER_MODEL_ROUTES_ENV, LEGACY_INTER_MODEL_ROUTES_ENV)
                .unwrap_or_default();
        let routes = parse_inter_model_routes(&routes_raw);
        let min_interval_ms = optional_env_var_alias(
            INTER_MODEL_RELAY_MIN_INTERVAL_MS_ENV,
            LEGACY_INTER_MODEL_RELAY_MIN_INTERVAL_MS_ENV,
        )
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_INTER_MODEL_RELAY_MIN_INTERVAL_MS)
        .max(100);

        Self {
            routes,
            min_interval: Duration::from_millis(min_interval_ms),
            last_relay_at: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn has_routes(&self) -> bool {
        !self.routes.is_empty()
    }

    fn targets_for(&self, source_model_name: &str) -> Option<&[String]> {
        self.routes.get(source_model_name).map(Vec::as_slice)
    }

    fn should_emit(&self, source_model_id: u32) -> bool {
        let now = Instant::now();
        let mut last = self.last_relay_at.lock();
        if let Some(previous) = last.get(&source_model_id).copied() {
            if now.duration_since(previous) < self.min_interval {
                return false;
            }
        }
        last.insert(source_model_id, now);
        true
    }
}

fn parse_inter_model_routes(raw: &str) -> InterModelRoutes {
    let mut routes = HashMap::<String, Vec<String>>::new();
    for rule in raw.split([';', '\n']) {
        let trimmed = rule.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some((source_raw, target_raw)) =
            trimmed.split_once('=').or_else(|| trimmed.split_once("->"))
        else {
            continue;
        };
        let source = source_raw.trim();
        if source.is_empty() {
            continue;
        }
        for target in target_raw
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let entry = routes.entry(source.to_string()).or_default();
            if !entry.iter().any(|existing| existing == target) {
                entry.push(target.to_string());
            }
        }
    }
    routes
}

fn relay_prompt_from_output(
    source_model_name: &str,
    output: &BinaryTensorPacket,
) -> Option<String> {
    if output.dtype != TensorDtype::Utf8 {
        return None;
    }
    let text = std::str::from_utf8(&output.data).ok()?.trim();
    if text.is_empty() {
        return None;
    }
    Some(format!("Report from {}:\n{}", source_model_name, text))
}

fn resolve_target_base_model_id(
    model_registry: &ModelRegistry,
    target_model_name: &str,
) -> Option<u32> {
    let mut candidates = model_registry
        .list()
        .into_iter()
        .filter(|model| model.name == target_model_name && model.status == ModelStatus::Active)
        .collect::<Vec<_>>();
    candidates.sort_by_key(|model| model.id);
    candidates.first().map(|model| model.base_model_id)
}

fn maybe_publish_inter_model_relays(
    relay_state: &InterModelRelayState,
    source_model_id: u32,
    source_model_name: &str,
    request_is_relay: bool,
    output: &BinaryTensorPacket,
    replica_pools: &ReplicaPools,
    model_registry: &ModelRegistry,
) {
    if request_is_relay {
        return;
    }

    let Some(targets) = relay_state.targets_for(source_model_name) else {
        return;
    };
    if targets.is_empty() || !relay_state.should_emit(source_model_id) {
        return;
    }

    let Some(prompt) = relay_prompt_from_output(source_model_name, output) else {
        return;
    };

    for target_model_name in targets {
        if target_model_name == source_model_name {
            continue;
        }

        let Some(target_base_model_id) =
            resolve_target_base_model_id(model_registry, target_model_name)
        else {
            log::warn!(
                "Inter-model relay target not found: source={} target={}",
                source_model_name,
                target_model_name
            );
            continue;
        };

        let target_pool = {
            let pools = replica_pools.read();
            pools.get(&target_base_model_id).cloned()
        };
        let Some(target_pool) = target_pool else {
            log::warn!(
                "Inter-model relay pool missing: source={} target={} base_model_id={}",
                source_model_name,
                target_model_name,
                target_base_model_id
            );
            continue;
        };

        let data = prompt.clone().into_bytes();
        let relay_request = InferenceRequest {
            input: BinaryTensorPacket {
                shape: vec![1, data.len() as i64],
                dtype: TensorDtype::Utf8,
                data,
            },
            additional_inputs: Vec::new(),
            session_id: Some(format!(
                "{}{}->{}",
                INTER_MODEL_RELAY_SESSION_PREFIX, source_model_id, target_base_model_id
            )),
            metadata: Some(kapsl_engine_api::RequestMetadata {
                request_id: Some(format!(
                    "relay-{}-{}-{}",
                    source_model_id,
                    target_base_model_id,
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis()
                )),
                priority: Some(1),
                ..kapsl_engine_api::RequestMetadata::default()
            }),
            cancellation: None,
        };

        let source_model_name = source_model_name.to_string();
        let target_model_name = target_model_name.clone();
        tokio::spawn(async move {
            if let Err(error) = target_pool
                .infer(&relay_request, kapsl_scheduler::Priority::Throughput, false)
                .await
            {
                log::warn!(
                    "Inter-model relay failed: source={} target={} error={}",
                    source_model_name,
                    target_model_name,
                    error
                );
            }
        });
    }
}

fn update_throughput(
    samples: &mut HashMap<u32, ThroughputSample>,
    model_id: u32,
    total_inferences: u64,
    now: Instant,
) -> f64 {
    if let Some(entry) = samples.get_mut(&model_id) {
        let elapsed = now.duration_since(entry.last_timestamp).as_secs_f64();
        let delta = total_inferences.saturating_sub(entry.last_total);
        let throughput = if elapsed > 0.0 {
            delta as f64 / elapsed
        } else {
            entry.throughput
        };
        entry.last_total = total_inferences;
        entry.last_timestamp = now;
        entry.throughput = throughput;
        throughput
    } else {
        samples.insert(
            model_id,
            ThroughputSample {
                last_total: total_inferences,
                last_timestamp: now,
                throughput: 0.0,
            },
        );
        0.0
    }
}

fn sample_nvidia_smi() -> Option<(f64, usize, usize)> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut total_util = 0.0;
    let mut total_mem_mb = 0.0;
    let mut total_mem_capacity_mb = 0.0;
    let mut count = 0.0;

    for line in stdout.lines() {
        let mut parts = line.split(',');
        let util_str = parts.next().map(|s| s.trim());
        let mem_str = parts.next().map(|s| s.trim());
        let mem_total_str = parts.next().map(|s| s.trim());
        let (util_str, mem_str, mem_total_str) = match (util_str, mem_str, mem_total_str) {
            (Some(util_str), Some(mem_str), Some(mem_total_str)) => {
                (util_str, mem_str, mem_total_str)
            }
            _ => continue,
        };

        if let (Ok(util), Ok(mem_mb), Ok(mem_total_mb)) = (
            util_str.parse::<f64>(),
            mem_str.parse::<f64>(),
            mem_total_str.parse::<f64>(),
        ) {
            total_util += util;
            total_mem_mb += mem_mb;
            total_mem_capacity_mb += mem_total_mb;
            count += 1.0;
        }
    }

    if count == 0.0 {
        return None;
    }

    let avg_util = (total_util / count) / 100.0;
    let mem_bytes = (total_mem_mb * 1024.0 * 1024.0) as usize;
    let mem_capacity_bytes = (total_mem_capacity_mb * 1024.0 * 1024.0) as usize;
    Some((avg_util, mem_bytes, mem_capacity_bytes))
}

struct WorkerProcess {
    socket_path: String,
    child: Mutex<Child>,
}

impl WorkerProcess {
    fn try_wait(&self) -> Option<std::process::ExitStatus> {
        self.child.lock().try_wait().ok().flatten()
    }

    fn kill(&self) {
        let mut child = self.child.lock();
        if let Ok(None) = child.try_wait() {
            let _ = child.kill();
        }
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(unix)]
fn socket_ready(socket_path: &str) -> bool {
    if !Path::new(socket_path).exists() {
        return false;
    }
    UnixStream::connect(socket_path).is_ok()
}

#[cfg(not(unix))]
fn socket_ready(_socket_path: &str) -> bool {
    false
}

#[allow(clippy::too_many_arguments)]
fn spawn_worker_process(
    model_id: u32,
    model_path: &Path,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    topology: &str,
    tp_degree: usize,
    onnx_tuning: &OnnxRuntimeTuning,
) -> Result<WorkerProcess, Box<dyn std::error::Error + Send + Sync>> {
    #[cfg(not(unix))]
    {
        let _ = (
            model_id,
            model_path,
            batch_size,
            scheduler_queue_size,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
            topology,
            tp_degree,
            onnx_tuning,
        );
        return Err("Isolated workers are only supported on unix platforms".into());
    }

    #[cfg(unix)]
    {
        let socket_path = format!("/tmp/kapsl-worker-{}-{}.sock", model_id, std::process::id());
        if Path::new(&socket_path).exists() {
            std::fs::remove_file(&socket_path)?;
        }

        let exe = std::env::current_exe()?;
        let mut command = Command::new(exe);
        command
            .arg("--worker")
            .arg("--worker-model-id")
            .arg(model_id.to_string())
            .arg("--model")
            .arg(model_path)
            .arg("--socket")
            .arg(&socket_path)
            .arg("--transport")
            .arg("socket")
            .arg("--batch-size")
            .arg(batch_size.to_string())
            .arg("--scheduler-queue-size")
            .arg(scheduler_queue_size.to_string())
            .arg("--scheduler-max-micro-batch")
            .arg(scheduler_max_micro_batch.to_string())
            .arg("--scheduler-queue-delay-ms")
            .arg(scheduler_queue_delay_ms.to_string())
            .arg("--topology")
            .arg(topology)
            .arg("--tp-degree")
            .arg(tp_degree.to_string())
            .env(LLM_ISOLATE_PROCESS_ENV, "0");
        if let Some(value) = onnx_tuning.memory_pattern {
            command.arg("--onnx-memory-pattern").arg(value.to_string());
        }
        if let Some(value) = onnx_tuning.disable_cpu_mem_arena {
            command
                .arg("--onnx-disable-cpu-mem-arena")
                .arg(value.to_string());
        }
        if let Some(value) = onnx_tuning.session_buckets {
            command.arg("--onnx-session-buckets").arg(value.to_string());
        }
        if let Some(value) = onnx_tuning.bucket_dim_granularity {
            command
                .arg("--onnx-bucket-dim-granularity")
                .arg(value.to_string());
        }
        if let Some(value) = onnx_tuning.bucket_max_dims {
            command.arg("--onnx-bucket-max-dims").arg(value.to_string());
        }
        if let Some(value) = onnx_tuning.peak_concurrency_hint {
            command
                .arg("--onnx-peak-concurrency-hint")
                .arg(value.to_string());
        }
        let child = command.spawn()?;

        Ok(WorkerProcess {
            socket_path,
            child: Mutex::new(child),
        })
    }
}

fn wait_for_worker_ready(worker: &WorkerProcess, timeout: Duration) -> Result<(), EngineError> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = worker.try_wait() {
            return Err(EngineError::backend(format!(
                "Worker exited before ready: {}",
                status
            )));
        }
        if socket_ready(&worker.socket_path) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(EngineError::backend(
                "Timed out waiting for worker socket".to_string(),
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

async fn wait_for_worker_ready_async(
    worker: &WorkerProcess,
    timeout: Duration,
) -> Result<(), EngineError> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = worker.try_wait() {
            return Err(EngineError::backend(format!(
                "Worker exited before ready: {}",
                status
            )));
        }
        if socket_ready(&worker.socket_path) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(EngineError::backend(
                "Timed out waiting for worker socket".to_string(),
            ));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

struct RemoteEngine {
    model_id: u32,
    socket_path: String,
    worker: Arc<WorkerProcess>,
}

impl RemoteEngine {
    fn new(model_id: u32, worker: Arc<WorkerProcess>) -> Self {
        Self {
            model_id,
            socket_path: worker.socket_path.clone(),
            worker,
        }
    }

    #[cfg(unix)]
    fn connect(&self) -> Result<UnixStream, EngineError> {
        if let Some(status) = self.worker.try_wait() {
            return Err(EngineError::backend(format!(
                "Worker process exited: {}",
                status
            )));
        }
        UnixStream::connect(&self.socket_path)
            .map_err(|e| EngineError::backend(format!("IPC connect failed: {}", e)))
    }

    #[cfg(unix)]
    fn read_response_header(&self, conn: &mut UnixStream) -> Result<ResponseHeader, EngineError> {
        let mut header_buf = [0u8; 8];
        conn.read_exact(&mut header_buf)
            .map_err(|e| EngineError::backend(format!("IPC read header failed: {}", e)))?;
        let status = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
        let payload_size = u32::from_le_bytes(header_buf[4..8].try_into().unwrap());
        Ok(ResponseHeader {
            status,
            payload_size,
        })
    }
}

#[async_trait::async_trait]
impl Engine for RemoteEngine {
    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        #[cfg(not(unix))]
        {
            let _ = request;
            return Err(EngineError::backend(
                "IPC isolation is only supported on unix platforms".to_string(),
            ));
        }

        #[cfg(unix)]
        {
            let mut conn = self.connect()?;
            let payload = bincode::serialize(request)
                .map_err(|e| EngineError::backend(format!("IPC serialize failed: {}", e)))?;

            let header = RequestHeader {
                model_id: self.model_id,
                op_code: OP_INFER,
                payload_size: payload.len() as u32,
            };

            conn.write_all(&header.model_id.to_le_bytes())
                .map_err(|e| EngineError::backend(format!("IPC write failed: {}", e)))?;
            conn.write_all(&header.op_code.to_le_bytes())
                .map_err(|e| EngineError::backend(format!("IPC write failed: {}", e)))?;
            conn.write_all(&header.payload_size.to_le_bytes())
                .map_err(|e| EngineError::backend(format!("IPC write failed: {}", e)))?;
            conn.write_all(&payload)
                .map_err(|e| EngineError::backend(format!("IPC write failed: {}", e)))?;

            let resp = self.read_response_header(&mut conn)?;
            let mut payload = vec![0u8; resp.payload_size as usize];
            conn.read_exact(&mut payload)
                .map_err(|e| EngineError::backend(format!("IPC read failed: {}", e)))?;

            if resp.status == STATUS_OK {
                bincode::deserialize::<BinaryTensorPacket>(&payload)
                    .map_err(|e| EngineError::backend(format!("IPC decode failed: {}", e)))
            } else {
                let msg = String::from_utf8_lossy(&payload);
                Err(EngineError::backend(format!(
                    "Remote error (status {}): {}",
                    resp.status, msg
                )))
            }
        }
    }

    fn infer_stream(
        &self,
        request: &InferenceRequest,
    ) -> std::pin::Pin<
        Box<dyn futures::stream::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
    > {
        #[cfg(not(unix))]
        {
            let _ = request;
            let stream = stream::once(async {
                Err(EngineError::backend(
                    "IPC isolation is only supported on unix platforms".to_string(),
                ))
            });
            return Box::pin(stream);
        }

        #[cfg(unix)]
        {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let socket_path = self.socket_path.clone();
            let model_id = self.model_id;
            let request = request.clone();
            let worker = self.worker.clone();

            std::thread::spawn(move || {
                let mut conn = match UnixStream::connect(&socket_path) {
                    Ok(conn) => conn,
                    Err(e) => {
                        let _ = tx.send(Err(EngineError::backend(format!(
                            "IPC connect failed: {}",
                            e
                        ))));
                        return;
                    }
                };

                if let Some(status) = worker.try_wait() {
                    let _ = tx.send(Err(EngineError::backend(format!(
                        "Worker process exited: {}",
                        status
                    ))));
                    return;
                }

                let payload = match bincode::serialize(&request) {
                    Ok(payload) => payload,
                    Err(e) => {
                        let _ = tx.send(Err(EngineError::backend(format!(
                            "IPC serialize failed: {}",
                            e
                        ))));
                        return;
                    }
                };

                let header = RequestHeader {
                    model_id,
                    op_code: OP_INFER_STREAM,
                    payload_size: payload.len() as u32,
                };

                if conn.write_all(&header.model_id.to_le_bytes()).is_err()
                    || conn.write_all(&header.op_code.to_le_bytes()).is_err()
                    || conn.write_all(&header.payload_size.to_le_bytes()).is_err()
                    || conn.write_all(&payload).is_err()
                {
                    let _ = tx.send(Err(EngineError::backend("IPC write failed".to_string())));
                    return;
                }

                loop {
                    let mut header_buf = [0u8; 8];
                    if conn.read_exact(&mut header_buf).is_err() {
                        let _ = tx.send(Err(EngineError::backend("IPC read failed".to_string())));
                        return;
                    }
                    let status = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
                    let payload_size = u32::from_le_bytes(header_buf[4..8].try_into().unwrap());

                    if status == STATUS_STREAM_END {
                        break;
                    }

                    let mut payload = vec![0u8; payload_size as usize];
                    if conn.read_exact(&mut payload).is_err() {
                        let _ = tx.send(Err(EngineError::backend("IPC read failed".to_string())));
                        return;
                    }

                    if status == STATUS_STREAM_CHUNK {
                        match bincode::deserialize::<BinaryTensorPacket>(&payload) {
                            Ok(packet) => {
                                if tx.send(Ok(packet)).is_err() {
                                    return;
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(Err(EngineError::backend(format!(
                                    "IPC decode failed: {}",
                                    e
                                ))));
                                return;
                            }
                        }
                    } else if status == STATUS_ERR {
                        let msg = String::from_utf8_lossy(&payload);
                        let _ =
                            tx.send(Err(EngineError::backend(format!("Remote error: {}", msg))));
                        return;
                    } else {
                        let _ = tx.send(Err(EngineError::backend(format!(
                            "Unexpected IPC status: {}",
                            status
                        ))));
                        return;
                    }
                }
            });

            let stream = stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|item| (item, rx))
            });
            Box::pin(stream)
        }
    }

    fn unload(&mut self) {
        // Shared worker lifecycle is owned by Arc<WorkerProcess>.
    }

    fn metrics(&self) -> EngineMetrics {
        #[cfg(not(unix))]
        {
            EngineMetrics::default()
        }

        #[cfg(unix)]
        {
            let pid = self.worker.child.lock().id();
            let pid = Pid::from_u32(pid);

            let mut system = System::new();
            system.refresh_process(pid);
            let memory_usage = system
                .process(pid)
                .map(|p| p.memory() as usize)
                .unwrap_or(0);

            EngineMetrics {
                memory_usage,
                ..EngineMetrics::default()
            }
        }
    }

    fn health_check(&self) -> Result<(), EngineError> {
        #[cfg(not(unix))]
        {
            return Err(EngineError::backend(
                "IPC isolation is only supported on unix platforms".to_string(),
            ));
        }

        #[cfg(unix)]
        {
            if let Some(status) = self.worker.try_wait() {
                return Err(EngineError::backend(format!(
                    "Worker process exited: {}",
                    status
                )));
            }
            UnixStream::connect(&self.socket_path)
                .map(|_| ())
                .map_err(|e| EngineError::backend(format!("IPC health check failed: {}", e)))
        }
    }
}

fn allocate_model_id(counter: &AtomicU32, recycled_ids: &Mutex<Vec<u32>>) -> u32 {
    if let Some(id) = recycled_ids.lock().pop() {
        id
    } else {
        counter.fetch_add(1, Ordering::SeqCst)
    }
}

fn recycle_model_id(model_id: u32, recycled_ids: &Mutex<Vec<u32>>) {
    recycled_ids.lock().push(model_id);
}

async fn run_worker(
    args: &Args,
    device_info: &DeviceInfo,
    onnx_tuning_profile: &OnnxTuningProfile,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if args.model.len() != 1 {
        return Err("Worker mode expects exactly one --model".into());
    }

    let model_id = args.worker_model_id.unwrap_or(0);
    let model_path = &args.model[0];
    let onnx_tuning = onnx_tuning_profile.resolve(model_id);

    let registry = Arc::new(Registry::new());
    let model_registry = Arc::new(ModelRegistry::new());
    let shared_metrics = kapsl_monitor::metrics::KapslMetrics::new(&registry);

    let pool = load_model(
        model_id,
        model_path,
        device_info,
        args.batch_size,
        args.scheduler_queue_size,
        args.scheduler_max_micro_batch,
        args.scheduler_queue_delay_ms,
        &model_registry,
        &shared_metrics,
        &args.topology,
        args.tp_degree,
        onnx_tuning,
    )
    .await?;

    let mut schedulers = HashMap::new();
    schedulers.insert(model_id, pool as Arc<dyn ReplicaScheduler + Send + Sync>);

    let server = IpcServer::new(&args.socket, schedulers, None);
    log::info!(
        "Worker process serving model {} via IPC socket {}",
        model_id,
        args.socket
    );
    server.run().await?;
    Ok(())
}

/// Load a model
#[allow(clippy::too_many_arguments)]
fn load_model_blocking(
    model_id: u32,
    model_path: &PathBuf,
    device_info: &DeviceInfo,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    model_registry: &ModelRegistry,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    topology: &str,
    tp_degree: usize,
    onnx_tuning: OnnxRuntimeTuning,
) -> Result<Arc<ReplicaPool<Scheduler>>, Box<dyn std::error::Error + Send + Sync>> {
    log::info!(
        "Current directory: {:?}",
        std::env::current_dir().unwrap_or_default()
    );
    let absolute_path = match model_path.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            log::error!(
                "Failed to canonicalize model path {:?}: {} (CWD: {:?})",
                model_path,
                e,
                std::env::current_dir().unwrap_or_default()
            );
            return Err(format!("Invalid model path {:?}: {}", model_path, e).into());
        }
    };
    log::info!("Loading Model ID {}: {:?}", model_id, absolute_path);

    let loader = if looks_like_model_file_path(absolute_path.as_path()) {
        match PackageLoader::from_raw_file(absolute_path.as_path()) {
            Ok(loader) => {
                log::info!("Loading raw model file (no .aimod packaging)");
                loader
            }
            Err(e) => {
                return Err(format!("Failed to load raw model {}: {}", model_id, e).into());
            }
        }
    } else {
        match PackageLoader::load(absolute_path.as_path()) {
            Ok(loader) => loader,
            Err(e) => {
                log::error!("Failed to load model {}: {}", model_id, e);
                return Err(format!("Failed to load model {}: {}", model_id, e).into());
            }
        }
    };
    log::info!("✓ Package loaded");
    log::info!("  Project: {}", loader.manifest.project_name);
    log::info!("  Framework: {}", loader.manifest.framework);
    log::info!("  Version: {}", loader.manifest.version);
    let queue_overflow_policy = resolve_queue_overflow_policy(&loader.manifest);
    log_queue_policy_caveat(queue_overflow_policy);
    log::info!(
        "  Queue overflow policy: {}",
        queue_overflow_policy.as_str()
    );
    let (scheduler_max_micro_batch, scheduler_queue_delay_ms) =
        resolve_scheduler_tuning_for_framework(
            &loader.manifest,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
        );

    let model_file_path = loader.get_model_path();
    let isolate_process = resolve_isolate_process(&loader.manifest);
    if isolate_process {
        log::info!("✓ Process isolation enabled for Model ID {}", model_id);
    }

    BackendFactory::validate_requirements(&loader.manifest.hardware_requirements, device_info)
        .map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> = format!(
                "Requirements validation failed for model {}: {}",
                model_id, e
            )
            .into();
            err
        })?;

    // Initialize Device Mesh
    use kapsl_hal::device_mesh::DeviceMesh;
    let pipeline_stages = manifest_llm_pipeline_stages(&loader.manifest);
    let EffectiveTopologyChoice {
        mesh_topology,
        worker_topology,
        worker_tp_degree,
        use_pipeline_backend: use_pipeline,
    } = resolve_effective_topology_choice(
        &loader.manifest,
        topology,
        tp_degree,
        pipeline_stages.as_deref(),
    );

    let devices = select_mesh_devices(&loader.manifest.hardware_requirements, device_info)
        .map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> =
                format!("Failed to select devices for model {}: {}", model_id, e).into();
            err
        })?;

    let device_mesh = DeviceMesh::with_topology(devices, mesh_topology).map_err(|e| {
        let err: Box<dyn std::error::Error + Send + Sync> =
            format!("Failed to create device mesh: {}", e).into();
        err
    })?;

    log::info!(
        "✓ Device Mesh initialized: {} devices, topology: {:?}",
        device_mesh.world_size,
        device_mesh.topology
    );

    if use_pipeline {
        if let Some(stages) = &pipeline_stages {
            if stages.len() > device_mesh.world_size {
                return Err(format!(
                    "Pipeline stages ({}) exceed available devices ({})",
                    stages.len(),
                    device_mesh.world_size
                )
                .into());
            }
        }
    }

    // Create engines for each device in the mesh
    let mut engines: Vec<EngineHandle> = Vec::new();
    let worker = if isolate_process {
        match spawn_worker_process(
            model_id,
            &absolute_path,
            batch_size,
            scheduler_queue_size,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
            worker_topology,
            worker_tp_degree,
            &onnx_tuning,
        ) {
            Ok(worker) => match wait_for_worker_ready(&worker, Duration::from_secs(30)) {
                Ok(()) => Some(Arc::new(worker)),
                Err(e) => {
                    worker.kill();
                    log::warn!(
                        "Model {} requested process isolation, but worker was not ready; falling back to in-process load: {}",
                        model_id,
                        e
                    );
                    None
                }
            },
            Err(e) => {
                log::warn!(
                    "Model {} requested process isolation, but worker spawn failed; falling back to in-process load: {}",
                    model_id,
                    e
                );
                None
            }
        }
    } else {
        None
    };

    if use_pipeline {
        if let Some(worker) = &worker {
            let backend = RemoteEngine::new(model_id, worker.clone());
            let monitored_backend = MonitoringMiddleware::new_with_metrics(
                backend,
                model_id.to_string(),
                loader.manifest.version.clone(),
                shared_metrics.clone(),
            );
            let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
            engines.push(Arc::from(engine_box));
        } else {
            let device_ids: Vec<i32> = (0..device_mesh.world_size)
                .filter_map(|rank| device_mesh.get_device(rank))
                .map(|d| d.id as i32)
                .collect();
            let provider_policy = provider_policy();
            let mut backend = if provider_policy == "manifest" {
                let provider = device_mesh
                    .get_device(0)
                    .map(|d| d.backend.to_string())
                    .unwrap_or_else(|| "cpu".to_string());
                LLMBackend::with_devices(provider, device_ids)
            } else {
                LLMBackend::with_device_ids(device_ids)
            };
            tokio::runtime::Handle::current()
                .block_on(backend.load(&model_file_path))
                .map_err(|e| {
                    let err: Box<dyn std::error::Error + Send + Sync> =
                        format!("Failed to load pipeline model {}: {}", model_id, e).into();
                    err
                })?;

            let monitored_backend = MonitoringMiddleware::new_with_metrics(
                backend,
                model_id.to_string(),
                loader.manifest.version.clone(),
                shared_metrics.clone(),
            );
            let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
            engines.push(Arc::from(engine_box));
        }
    } else {
        // We need to iterate over ranks in the mesh
        for rank in 0..device_mesh.world_size {
            if let Some(device) = device_mesh.get_device(rank) {
                if let Some(worker) = &worker {
                    let backend = RemoteEngine::new(model_id, worker.clone());
                    let monitored_backend = MonitoringMiddleware::new_with_metrics(
                        backend,
                        model_id.to_string(),
                        loader.manifest.version.clone(),
                        shared_metrics.clone(),
                    );
                    let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
                    engines.push(Arc::from(engine_box));
                    continue;
                }
                let provider = device.backend.to_string();

                // Create backend for this specific device
                let mut backend = BackendFactory::create_backend_for_device_with_tuning(
                    &loader.manifest,
                    &provider,
                    device.id,
                    device_info,
                    &onnx_tuning,
                )?;

                tokio::runtime::Handle::current()
                    .block_on(backend.load(&model_file_path))
                    .map_err(|e| {
                        let err: Box<dyn std::error::Error + Send + Sync> = format!(
                            "Failed to load model {} on device {}: {}",
                            model_id, device.id, e
                        )
                        .into();
                        err
                    })?;

                let monitored_backend = MonitoringMiddleware::new_with_metrics(
                    backend,
                    model_id.to_string(),
                    loader.manifest.version.clone(),
                    shared_metrics.clone(),
                );

                let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
                engines.push(Arc::from(engine_box));
            }
        }
    }

    log::info!("✓ Loaded {} engine instances", engines.len());

    // Determine device/provider string for registry
    let device_str = device_info.get_best_provider().to_string();
    let optimization_level = loader
        .manifest
        .hardware_requirements
        .graph_optimization_level
        .clone()
        .unwrap_or_else(|| "basic".to_string());

    // Register model in the model registry
    let model_info = ModelInfo::new(
        model_id,
        loader.manifest.project_name.clone(),
        loader.manifest.version.clone(),
        loader.manifest.framework.clone(),
        device_str,
        optimization_level,
        absolute_path.to_string_lossy().to_string(), // Store absolute package path
    );
    model_registry.upsert(model_info);

    // Create Scheduler with the device mesh
    let scheduler = Arc::new(
        Scheduler::new(
            engines,
            batch_size,
            1, // workers per device
            scheduler_queue_size,
            true, // enable fallback
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
            Some(Arc::new(device_mesh)),
        )
        .with_queue_overflow_policy(queue_overflow_policy),
    );

    // Create ReplicaPool and add the primary scheduler
    let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
    pool.add_replica(0, scheduler);

    log::info!("✓ Scheduler started for Model ID {}\n", model_id);
    Ok(Arc::new(pool))
}

/// Load a model and create its scheduler
#[allow(clippy::too_many_arguments)]
async fn load_model(
    model_id: u32,
    model_path: &PathBuf,
    device_info: &DeviceInfo,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    model_registry: &ModelRegistry,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    topology: &str,
    tp_degree: usize,
    onnx_tuning: OnnxRuntimeTuning,
) -> Result<Arc<ReplicaPool<Scheduler>>, Box<dyn std::error::Error + Send + Sync>> {
    log::info!(
        "Current directory: {:?}",
        std::env::current_dir().unwrap_or_default()
    );
    let absolute_path = match model_path.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            log::error!(
                "Failed to canonicalize model path {:?}: {} (CWD: {:?})",
                model_path,
                e,
                std::env::current_dir().unwrap_or_default()
            );
            return Err(format!("Invalid model path {:?}: {}", model_path, e).into());
        }
    };
    log::info!("Loading Model ID {}: {:?}", model_id, absolute_path);

    let loader = if looks_like_model_file_path(&absolute_path) {
        match PackageLoader::from_raw_file(&absolute_path) {
            Ok(loader) => {
                log::info!("Loading raw model file (no .aimod packaging)");
                loader
            }
            Err(e) => {
                return Err(format!("Failed to load raw model {}: {}", model_id, e).into());
            }
        }
    } else {
        match PackageLoader::load(&absolute_path) {
            Ok(loader) => loader,
            Err(e) => {
                log::error!("Failed to load model {}: {}", model_id, e);
                return Err(format!("Failed to load model {}: {}", model_id, e).into());
            }
        }
    };
    log::info!("✓ Package loaded");
    log::info!("  Project: {}", loader.manifest.project_name);
    log::info!("  Framework: {}", loader.manifest.framework);
    log::info!("  Version: {}", loader.manifest.version);
    let queue_overflow_policy = resolve_queue_overflow_policy(&loader.manifest);
    log_queue_policy_caveat(queue_overflow_policy);
    log::info!(
        "  Queue overflow policy: {}",
        queue_overflow_policy.as_str()
    );
    let (scheduler_max_micro_batch, scheduler_queue_delay_ms) =
        resolve_scheduler_tuning_for_framework(
            &loader.manifest,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
        );

    let model_file_path = loader.get_model_path();
    let isolate_process = resolve_isolate_process(&loader.manifest);
    if isolate_process {
        log::info!("✓ Process isolation enabled for Model ID {}", model_id);
    }

    BackendFactory::validate_requirements(&loader.manifest.hardware_requirements, device_info)
        .map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> = format!(
                "Requirements validation failed for model {}: {}",
                model_id, e
            )
            .into();
            err
        })?;

    // Initialize Device Mesh
    use kapsl_hal::device_mesh::DeviceMesh;
    let pipeline_stages = manifest_llm_pipeline_stages(&loader.manifest);
    let EffectiveTopologyChoice {
        mesh_topology,
        worker_topology,
        worker_tp_degree,
        use_pipeline_backend: use_pipeline,
    } = resolve_effective_topology_choice(
        &loader.manifest,
        topology,
        tp_degree,
        pipeline_stages.as_deref(),
    );

    let devices = select_mesh_devices(&loader.manifest.hardware_requirements, device_info)
        .map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> =
                format!("Failed to select devices for model {}: {}", model_id, e).into();
            err
        })?;

    let device_mesh = DeviceMesh::with_topology(devices, mesh_topology).map_err(|e| {
        let err: Box<dyn std::error::Error + Send + Sync> =
            format!("Failed to create device mesh: {}", e).into();
        err
    })?;

    log::info!(
        "✓ Device Mesh initialized: {} devices, topology: {:?}",
        device_mesh.world_size,
        device_mesh.topology
    );

    if use_pipeline {
        if let Some(stages) = &pipeline_stages {
            if stages.len() > device_mesh.world_size {
                return Err(format!(
                    "Pipeline stages ({}) exceed available devices ({})",
                    stages.len(),
                    device_mesh.world_size
                )
                .into());
            }
        }
    }

    // Create engines for each device in the mesh
    let mut engines: Vec<EngineHandle> = Vec::new();
    let worker = if isolate_process {
        match spawn_worker_process(
            model_id,
            &absolute_path,
            batch_size,
            scheduler_queue_size,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
            worker_topology,
            worker_tp_degree,
            &onnx_tuning,
        ) {
            Ok(worker) => match wait_for_worker_ready_async(&worker, Duration::from_secs(30)).await
            {
                Ok(()) => Some(Arc::new(worker)),
                Err(e) => {
                    worker.kill();
                    log::warn!(
                        "Model {} requested process isolation, but worker was not ready; falling back to in-process load: {}",
                        model_id,
                        e
                    );
                    None
                }
            },
            Err(e) => {
                log::warn!(
                    "Model {} requested process isolation, but worker spawn failed; falling back to in-process load: {}",
                    model_id,
                    e
                );
                None
            }
        }
    } else {
        None
    };

    if use_pipeline {
        if let Some(worker) = &worker {
            let backend = RemoteEngine::new(model_id, worker.clone());
            let monitored_backend = MonitoringMiddleware::new_with_metrics(
                backend,
                model_id.to_string(),
                loader.manifest.version.clone(),
                shared_metrics.clone(),
            );
            let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
            engines.push(Arc::from(engine_box));
        } else {
            let device_ids: Vec<i32> = (0..device_mesh.world_size)
                .filter_map(|rank| device_mesh.get_device(rank))
                .map(|d| d.id as i32)
                .collect();
            let provider_policy = provider_policy();
            let mut backend = if provider_policy == "manifest" {
                let provider = device_mesh
                    .get_device(0)
                    .map(|d| d.backend.to_string())
                    .unwrap_or_else(|| "cpu".to_string());
                LLMBackend::with_devices(provider, device_ids)
            } else {
                LLMBackend::with_device_ids(device_ids)
            };
            backend.load(&model_file_path).await.map_err(|e| {
                let err: Box<dyn std::error::Error + Send + Sync> =
                    format!("Failed to load pipeline model {}: {}", model_id, e).into();
                err
            })?;

            let monitored_backend = MonitoringMiddleware::new_with_metrics(
                backend,
                model_id.to_string(),
                loader.manifest.version.clone(),
                shared_metrics.clone(),
            );
            let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
            engines.push(Arc::from(engine_box));
        }
    } else {
        // We need to iterate over ranks in the mesh
        for rank in 0..device_mesh.world_size {
            if let Some(device) = device_mesh.get_device(rank) {
                if let Some(worker) = &worker {
                    let backend = RemoteEngine::new(model_id, worker.clone());
                    let monitored_backend = MonitoringMiddleware::new_with_metrics(
                        backend,
                        model_id.to_string(),
                        loader.manifest.version.clone(),
                        shared_metrics.clone(),
                    );
                    let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
                    engines.push(Arc::from(engine_box));
                    continue;
                }
                let provider = device.backend.to_string();

                // Create backend for this specific device
                let mut backend = BackendFactory::create_backend_for_device_with_tuning(
                    &loader.manifest,
                    &provider,
                    device.id,
                    device_info,
                    &onnx_tuning,
                )?;

                backend.load(&model_file_path).await.map_err(|e| {
                    let err: Box<dyn std::error::Error + Send + Sync> = format!(
                        "Failed to load model {} on device {}: {}",
                        model_id, device.id, e
                    )
                    .into();
                    err
                })?;

                let monitored_backend = MonitoringMiddleware::new_with_metrics(
                    backend,
                    model_id.to_string(),
                    loader.manifest.version.clone(),
                    shared_metrics.clone(),
                );

                let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
                engines.push(Arc::from(engine_box));
            }
        }
    }

    log::info!("✓ Loaded {} engine instances", engines.len());

    // Determine device/provider string for registry
    let device_str = device_info.get_best_provider().to_string();
    let optimization_level = loader
        .manifest
        .hardware_requirements
        .graph_optimization_level
        .clone()
        .unwrap_or_else(|| "basic".to_string());

    // Register model in the model registry
    let model_info = ModelInfo::new(
        model_id,
        loader.manifest.project_name.clone(),
        loader.manifest.version.clone(),
        loader.manifest.framework.clone(),
        device_str,
        optimization_level,
        absolute_path.to_string_lossy().to_string(), // Store absolute package path
    );
    model_registry.upsert(model_info);

    // Create Scheduler with the device mesh
    let scheduler = Arc::new(
        Scheduler::new(
            engines,
            batch_size,
            1, // workers per device
            scheduler_queue_size,
            true, // enable fallback
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
            Some(Arc::new(device_mesh)),
        )
        .with_queue_overflow_policy(queue_overflow_policy),
    );

    // Create ReplicaPool and add the primary scheduler
    let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
    pool.add_replica(0, scheduler);

    log::info!("✓ Scheduler started for Model ID {}\n", model_id);

    Ok(Arc::new(pool))
}

/// Scale up a model by adding a new replica
#[allow(clippy::too_many_arguments)]
async fn scale_up_model(
    base_model_id: u32,
    replica_id: u32,
    unique_id: u32,
    model_path: &PathBuf,
    device_info: &DeviceInfo,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    topology: &str,
    tp_degree: usize,
    model_registry: &ModelRegistry,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    onnx_tuning: OnnxRuntimeTuning,
) -> Result<Arc<Scheduler>, Box<dyn std::error::Error + Send + Sync>> {
    log::info!(
        "Scaling up Model ID {} - Creating replica #{}",
        base_model_id,
        replica_id
    );

    let absolute_path = model_path
        .canonicalize()
        .map_err(|e| format!("Invalid model path {:?}: {}", model_path, e))?;

    let loader = if looks_like_model_file_path(&absolute_path) {
        PackageLoader::from_raw_file(&absolute_path)
            .map_err(|e| format!("Failed to load raw model: {e}"))?
    } else {
        PackageLoader::load(&absolute_path).map_err(|e| format!("Failed to load model: {e}"))?
    };
    let model_file_path = loader.get_model_path();
    let queue_overflow_policy = resolve_queue_overflow_policy(&loader.manifest);
    log_queue_policy_caveat(queue_overflow_policy);
    let (scheduler_max_micro_batch, scheduler_queue_delay_ms) =
        resolve_scheduler_tuning_for_framework(
            &loader.manifest,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
        );
    let pipeline_stages = manifest_llm_pipeline_stages(&loader.manifest);
    let EffectiveTopologyChoice {
        mesh_topology: _,
        worker_topology,
        worker_tp_degree,
        use_pipeline_backend,
    } = resolve_effective_topology_choice(
        &loader.manifest,
        topology,
        tp_degree,
        pipeline_stages.as_deref(),
    );

    BackendFactory::validate_requirements(&loader.manifest.hardware_requirements, device_info)
        .map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> = format!(
                "Requirements validation failed for replica {}: {}",
                replica_id, e
            )
            .into();
            err
        })?;

    let isolate_process = resolve_isolate_process(&loader.manifest);
    let worker = if isolate_process {
        match spawn_worker_process(
            unique_id,
            &absolute_path,
            batch_size,
            scheduler_queue_size,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
            worker_topology,
            worker_tp_degree,
            &onnx_tuning,
        ) {
            Ok(worker) => {
                let worker = Arc::new(worker);
                match wait_for_worker_ready_async(worker.as_ref(), Duration::from_secs(30)).await {
                    Ok(()) => Some(worker),
                    Err(e) => {
                        worker.kill();
                        log::warn!(
                            "Replica {} requested process isolation, but worker was not ready; falling back to in-process load: {}",
                            replica_id,
                            e
                        );
                        None
                    }
                }
            }
            Err(e) => {
                log::warn!(
                    "Replica {} requested process isolation, but worker spawn failed; falling back to in-process load: {}",
                    replica_id,
                    e
                );
                None
            }
        }
    } else {
        None
    };

    let device = device_info.get_best_provider();
    let optimization_level = loader
        .manifest
        .hardware_requirements
        .graph_optimization_level
        .clone()
        .unwrap_or_else(|| "basic".to_string());

    let model_info = ModelInfo::new_replica(
        unique_id,
        replica_id,
        base_model_id,
        loader.manifest.project_name.clone(),
        loader.manifest.version.clone(),
        loader.manifest.framework.clone(),
        device.to_string(),
        optimization_level,
        absolute_path.to_string_lossy().to_string(),
    );
    model_registry.upsert(model_info);

    let engine: Arc<dyn kapsl_engine_api::Engine> = if let Some(worker) = worker {
        let backend = RemoteEngine::new(unique_id, worker);
        let monitored_backend = MonitoringMiddleware::new_with_metrics(
            backend,
            base_model_id.to_string(),
            loader.manifest.version.clone(),
            shared_metrics.clone(),
        );
        Arc::new(monitored_backend)
    } else if use_pipeline_backend {
        let pipeline_devices =
            select_mesh_devices(&loader.manifest.hardware_requirements, device_info).map_err(
                |e| {
                    let err: Box<dyn std::error::Error + Send + Sync> = format!(
                        "Failed to select devices for pipeline replica {}: {}",
                        replica_id, e
                    )
                    .into();
                    err
                },
            )?;
        let device_ids: Vec<i32> = pipeline_devices.iter().map(|d| d.id as i32).collect();
        let provider_policy = provider_policy();
        let mut backend = if provider_policy == "manifest" {
            let provider = pipeline_devices
                .first()
                .map(|d| d.backend.to_string())
                .unwrap_or_else(|| "cpu".to_string());
            LLMBackend::with_devices(provider, device_ids)
        } else {
            LLMBackend::with_device_ids(device_ids)
        };
        backend.load(&model_file_path).await.map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> =
                format!("Failed to load pipeline replica {}: {}", replica_id, e).into();
            err
        })?;

        let monitored_backend = MonitoringMiddleware::new_with_metrics(
            backend,
            base_model_id.to_string(),
            loader.manifest.version.clone(),
            shared_metrics.clone(),
        );
        Arc::new(monitored_backend)
    } else {
        let mut backend = BackendFactory::create_best_backend_with_tuning(
            &loader.manifest,
            device_info,
            &onnx_tuning,
        )?;
        backend.load(&model_file_path).await.map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> =
                format!("Failed to load replica {}: {}", replica_id, e).into();
            err
        })?;

        let monitored_backend = MonitoringMiddleware::new_with_metrics(
            backend,
            base_model_id.to_string(),
            loader.manifest.version.clone(),
            shared_metrics.clone(),
        );
        Arc::new(monitored_backend)
    };
    let scheduler = Arc::new(
        Scheduler::new(
            vec![engine],
            batch_size,
            1,
            scheduler_queue_size,
            true,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
            None,
        )
        .with_queue_overflow_policy(queue_overflow_policy),
    );

    log::info!(
        "✓ Replica #{} started for Model ID {}",
        replica_id,
        base_model_id
    );

    Ok(scheduler)
}

/// Scale down a model by removing a replica
async fn scale_down_model(
    base_model_id: u32,
    replica_id: u32,
    unique_id: u32,
    model_registry: &ModelRegistry,
    replica_pools: &ReplicaPools,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    log::info!(
        "Scaling down Model ID {} - Removing replica #{}",
        base_model_id,
        replica_id
    );

    // Update status to Stopping
    if let Err(e) = model_registry.set_status(unique_id, ModelStatus::Stopping) {
        log::error!("Failed to set status to Stopping for {}: {}", unique_id, e);
    }

    // Remove replica from pool
    let pool = replica_pools.read().get(&base_model_id).cloned();
    if let Some(pool) = pool {
        let removed = pool.remove_replica(replica_id);
        if !removed {
            let _ = model_registry.set_status(unique_id, ModelStatus::Active);
            return Err(format!(
                "Replica #{} was not present in pool for model {}",
                replica_id, base_model_id
            )
            .into());
        }
    } else {
        let _ = model_registry.set_status(unique_id, ModelStatus::Active);
        return Err(format!("Replica pool not found for model {}", base_model_id).into());
    }

    // Update status to Inactive
    if let Err(e) = model_registry.set_status(unique_id, ModelStatus::Inactive) {
        log::error!("Failed to set status to Inactive for {}: {}", unique_id, e);
    }

    log::info!(
        "✓ Replica #{} stopped for Model ID {}",
        replica_id,
        base_model_id
    );

    Ok(())
}

const MEMORY_HEADROOM_FRACTION: f64 = 0.80;

fn cap_scale_up_target_by_memory_headroom(
    current_replicas: u32,
    proposed_target: u32,
    total_model_memory_bytes: usize,
    system_total_memory_kb: u64,
) -> u32 {
    if proposed_target <= current_replicas
        || current_replicas == 0
        || total_model_memory_bytes == 0
        || system_total_memory_kb == 0
    {
        return proposed_target;
    }

    let per_replica_bytes = total_model_memory_bytes as f64 / current_replicas as f64;
    if per_replica_bytes <= 0.0 {
        return proposed_target;
    }

    let budget_bytes = (system_total_memory_kb as f64 * 1024.0 * MEMORY_HEADROOM_FRACTION).max(1.0);
    let max_by_headroom = (budget_bytes / per_replica_bytes).floor() as u32;
    let capped_max = max_by_headroom.max(current_replicas).max(1);
    proposed_target.min(capped_max)
}

#[tokio::main]
async fn main() -> Result<(), DynError> {
    let raw_argv: Vec<String> = std::env::args().collect();
    let Cli {
        command,
        run: _legacy_run_args,
    } = Cli::parse_from(&raw_argv);
    match command {
        Some(KapslCommand::Build(args)) => return execute_build_command(args),
        Some(KapslCommand::Push(args)) => return execute_push_command(args),
        Some(KapslCommand::Pull(args)) => return execute_pull_command(args),
        Some(KapslCommand::Login(args)) => return execute_login_command(args),
        Some(KapslCommand::Control(args)) => return execute_control_command(args),
        Some(KapslCommand::Run(_)) | None => {}
    }

    let runtime_argv = runtime_argv_from_invocation(&raw_argv);
    let (mut args, matches) = parse_runtime_args_and_matches(&runtime_argv)?;
    let applied_tuning = apply_performance_profile(&mut args, &matches);
    let onnx_tuning_profile = Arc::new(
        build_onnx_tuning_profile(&args)
            .map_err(|e| format!("Invalid ONNX tuning configuration: {}", e))?,
    );
    env_logger::init();
    if let Some(rationale) = &applied_tuning.auto_tune_rationale {
        log::info!("[auto-tune] {}", rationale);
    }
    let startup_started_at = Instant::now();
    let state_layout = resolve_runtime_state_layout(&args);
    if let Some(state_dir) = args.state_dir.as_ref() {
        log::info!("Runtime state directory: {}", state_dir.display());
    }
    let api_auth_state = Arc::new(RwLock::new(ApiAuthState::from_store_path(
        state_layout.auth_store_path.clone(),
    )));
    let log_sensitive_ids = env_flag(LOG_SENSITIVE_IDS_ENV);
    let http_bind_addr = parse_bind_ip(&args.http_bind, IpAddr::from([127, 0, 0, 1]), "http_bind");
    let allow_insecure_http = env_flag(ALLOW_INSECURE_HTTP_ENV);
    if !http_bind_addr.is_loopback() && !allow_insecure_http {
        return Err(format!(
            "Refusing to bind HTTP API on non-loopback address {} without {}=1. Use a TLS-terminating reverse proxy if exposing runtime externally.",
            http_bind_addr, ALLOW_INSECURE_HTTP_ENV
        )
        .into());
    }
    if !http_bind_addr.is_loopback() {
        log::warn!(
            "HTTP API is bound to {}. Traffic is plaintext HTTP; place runtime behind TLS and network ACLs.",
            http_bind_addr
        );
    }

    log::info!("🚀 Starting kapsl-runtime...\n");
    log::info!(
        "Performance profile: {} (batch_size={}, transport={}, scheduler_queue_size={}, scheduler_max_micro_batch={}, scheduler_queue_delay_ms={})",
        args.performance_profile.as_str(),
        args.batch_size,
        args.transport,
        args.scheduler_queue_size,
        args.scheduler_max_micro_batch,
        args.scheduler_queue_delay_ms
    );
    if applied_tuning.batch_size.is_some()
        || applied_tuning.transport.is_some()
        || applied_tuning.scheduler_queue_size.is_some()
        || applied_tuning.scheduler_max_micro_batch.is_some()
        || applied_tuning.scheduler_queue_delay_ms.is_some()
        || applied_tuning.media_preprocess.is_some()
        || applied_tuning.rust_log.is_some()
    {
        log::info!(
            "Applied performance tuning overrides from profile: batch_size={:?}, transport={:?}, scheduler_queue_size={:?}, scheduler_max_micro_batch={:?}, scheduler_queue_delay_ms={:?}, media_preprocess={:?}, rust_log={:?}",
            applied_tuning.batch_size,
            applied_tuning.transport,
            applied_tuning.scheduler_queue_size,
            applied_tuning.scheduler_max_micro_batch,
            applied_tuning.scheduler_queue_delay_ms,
            applied_tuning.media_preprocess,
            applied_tuning.rust_log
        );
    }
    let auth_status = api_auth_state.read().status_response();
    if auth_status.auth_enabled {
        log::info!("API authentication is enabled for /api routes.");
        log::info!("   - Auth store: {}", auth_status.store_path);
        log::info!(
            "   - Users: {} (active keys={}, active admin keys={})",
            auth_status.user_count,
            auth_status.active_key_count,
            auth_status.active_admin_key_count
        );
        log::info!("   - Reader token env: {}", API_READER_TOKEN_ENV);
        log::info!("   - Writer token env: {}", API_WRITER_TOKEN_ENV);
        log::info!("   - Admin token env: {}", API_ADMIN_TOKEN_ENV);
        log::info!("   - Shared fallback token env: {}", API_TOKEN_ENV);
    } else {
        log::warn!(
            "API authentication is disabled. /api routes are restricted to loopback clients only. Create an API key via /api/auth/access/* or set {} / {} / {} (shared fallback {} is also supported).",
            API_READER_TOKEN_ENV,
            API_WRITER_TOKEN_ENV,
            API_ADMIN_TOKEN_ENV,
            API_TOKEN_ENV
        );
    }
    if !log_sensitive_ids {
        log::info!(
            "Sensitive request/session identifiers are redacted in logs (set {}=1 to disable redaction)",
            LOG_SENSITIVE_IDS_ENV
        );
    }

    // 1. Hardware Probe
    log::info!("=== Hardware Detection ===");
    let device_info = Arc::new(DeviceInfo::probe());
    log::info!("CPU: {} cores", device_info.cpu_cores);
    log::info!("Memory: {} MB", device_info.total_memory / 1024);
    log::info!("OS: {} ({})", device_info.os_type, device_info.os_release);
    log::info!(
        "CUDA: {}",
        if device_info.has_cuda {
            format!(
                "✓ Available ({})",
                device_info
                    .devices
                    .iter()
                    .find(|d| matches!(d.backend, kapsl_hal::device::DeviceBackend::Cuda))
                    .and_then(|d| d.cuda_version.as_ref())
                    .map(|s| s.as_str())
                    .unwrap_or("unknown")
            )
        } else {
            "✗ Not available".to_string()
        }
    );
    log::info!(
        "Metal: {}",
        if device_info.has_metal {
            "✓ Available"
        } else {
            "✗ Not available"
        }
    );
    log::info!("Best provider: {}\n", device_info.get_best_provider());

    if args.worker {
        return run_worker(&args, &device_info, onnx_tuning_profile.as_ref()).await;
    }

    // Fail fast on common collisions (avoid slow model load, and avoid panics in background tasks).
    preflight_http_bind(http_bind_addr, args.metrics_port)?;
    match args.transport.as_str() {
        "socket" | "hybrid" => preflight_ipc_socket(&args.socket)?,
        "auto" => {
            if !ShmServer::is_available() {
                preflight_ipc_socket(&args.socket)?;
            }
        }
        _ => {}
    }

    // Use Arc<RwLock<>> for thread-safe dynamic scheduler management
    let replica_pools: Arc<RwLock<HashMap<u32, Arc<ReplicaPool<Scheduler>>>>> =
        Arc::new(RwLock::new(HashMap::new()));
    // Keep replica_schedulers for auto-scaler to track individual instances if needed,
    // but actually we can just rely on ModelRegistry and ReplicaPool.
    // However, scale_up_model returns a Scheduler, which we need to add to the pool.
    // We also need to store it somewhere if we want to access it directly?
    // ReplicaPool stores it. So we don't need separate replica_schedulers map.

    let registry = Arc::new(Registry::new());
    let model_registry = Arc::new(ModelRegistry::new());
    let auto_scaler = Arc::new(RwLock::new(AutoScaler::new()));
    let unique_id_counter = Arc::new(AtomicU32::new(1000)); // Start replica IDs from 1000
    let model_id_counter = Arc::new(AtomicU32::new(0)); // ID counter for models
    let recycled_model_ids = Arc::new(Mutex::new(Vec::new()));
    let model_paths: Arc<RwLock<HashMap<u32, PathBuf>>> = Arc::new(RwLock::new(HashMap::new()));
    let runtime_samples = Arc::new(RwLock::new(RuntimeSamples::default()));
    let throughput_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let inter_model_relay_state = Arc::new(InterModelRelayState::from_env());
    if inter_model_relay_state.has_routes() {
        log::info!(
            "Inter-model relay enabled: routes={} min_interval_ms={} env={} (legacy {})",
            inter_model_relay_state.routes.len(),
            inter_model_relay_state.min_interval.as_millis(),
            INTER_MODEL_ROUTES_ENV,
            LEGACY_INTER_MODEL_ROUTES_ENV
        );
    }
    let runtime_pressure_config = Arc::new(RuntimePressureConfig::from_env());
    let runtime_pressure_state = Arc::new(AtomicU8::new(RuntimePressureState::Normal as u8));

    // Create shared metrics instance ONCE for all models
    let shared_metrics = kapsl_monitor::metrics::KapslMetrics::new(&registry);

    let runtime_samples_for_sampler = runtime_samples.clone();
    let has_cuda_for_sampler = device_info.has_cuda;
    let runtime_pressure_config_for_sampler = runtime_pressure_config.clone();
    let runtime_pressure_state_for_sampler = runtime_pressure_state.clone();
    tokio::spawn(async move {
        let pid = Pid::from_u32(std::process::id());
        let mut system = System::new();
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        let mut nvidia_smi_retry_after: Option<Instant> = None;
        system.refresh_memory();
        let total_system_memory_bytes = Some(system.total_memory() as usize * 1024);

        loop {
            interval.tick().await;

            system.refresh_process(pid);
            let process_memory_bytes = system
                .process(pid)
                .map(|p| p.memory() as usize)
                .unwrap_or(0);

            let now = Instant::now();
            let (gpu_utilization, gpu_memory_bytes, gpu_memory_total_bytes) =
                if has_cuda_for_sampler {
                    if nvidia_smi_retry_after.is_some_and(|retry_after| now < retry_after) {
                        (0.0, None, None)
                    } else {
                        match sample_nvidia_smi() {
                            Some((util, mem_bytes, mem_total_bytes)) => {
                                nvidia_smi_retry_after = None;
                                (util, Some(mem_bytes), Some(mem_total_bytes))
                            }
                            None => {
                                nvidia_smi_retry_after = Some(now + Duration::from_secs(30));
                                (0.0, None, None)
                            }
                        }
                    }
                } else {
                    (0.0, None, None)
                };

            let collected_at_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let snapshot = RuntimeSamples {
                process_memory_bytes,
                total_system_memory_bytes,
                gpu_utilization,
                gpu_memory_bytes,
                gpu_memory_total_bytes,
                collected_at_ms,
            };
            *runtime_samples_for_sampler.write() = snapshot.clone();

            let next_state =
                evaluate_runtime_pressure_state(&snapshot, &runtime_pressure_config_for_sampler);
            let previous_raw =
                runtime_pressure_state_for_sampler.swap(next_state as u8, Ordering::Relaxed);
            let previous = RuntimePressureState::from_u8(previous_raw);
            if previous != next_state {
                log::warn!(
                    "Runtime pressure state changed: {} -> {} (rss={}B total_mem={}B gpu_util={:.2} gpu_mem={:?}/{:?})",
                    previous.as_str(),
                    next_state.as_str(),
                    snapshot.process_memory_bytes,
                    snapshot.total_system_memory_bytes.unwrap_or(0),
                    snapshot.gpu_utilization,
                    snapshot.gpu_memory_bytes,
                    snapshot.gpu_memory_total_bytes
                );
            }
        }
    });

    // 2. Load Packagesƒ
    log::info!("=== Package Loading ===");
    for model_path in args.model.iter() {
        let model_id = allocate_model_id(&model_id_counter, &recycled_model_ids);
        let model_label = model_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| model_path.to_string_lossy().to_string());
        let load_label = format!("Loading {}", model_label);
        let pool = match run_with_loading_async(
            &load_label,
            load_model(
                model_id,
                model_path,
                &device_info,
                args.batch_size,
                args.scheduler_queue_size,
                args.scheduler_max_micro_batch,
                args.scheduler_queue_delay_ms,
                &model_registry,
                &shared_metrics,
                &args.topology,
                args.tp_degree,
                onnx_tuning_profile.resolve(model_id),
            ),
        )
        .await
        {
            Ok(pool) => pool,
            Err(e) => {
                recycle_model_id(model_id, &recycled_model_ids);
                return Err(e);
            }
        };
        replica_pools.write().insert(model_id, pool);
        model_paths.write().insert(model_id, model_path.clone());

        // Register default scaling policy for each model
        auto_scaler
            .write()
            .register_policy(model_id, ScalingPolicy::default());
    }

    log::info!("=== Starting Transport Server ===");
    log::info!("Transport mode: {}", args.transport);

    let replica_pools_clone = replica_pools.clone();

    // Convert concrete pools to trait objects for TransportServer
    let get_trait_schedulers = |pools: &HashMap<u32, Arc<ReplicaPool<Scheduler>>>| {
        let mut map = HashMap::new();
        for (k, v) in pools {
            map.insert(*k, v.clone() as Arc<dyn ReplicaScheduler + Send + Sync>);
        }
        map
    };
    let get_scheduler_lookup = || {
        let pools = replica_pools_clone.clone();
        Arc::new(move |model_id: u32| {
            pools
                .read()
                .get(&model_id)
                .map(|pool| pool.clone() as Arc<dyn ReplicaScheduler + Send + Sync>)
        })
            as Arc<dyn Fn(u32) -> Option<Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>
    };

    let shm_size: usize = args
        .shm_size_mb
        .or_else(|| {
            std::env::var("KAPSL_SHM_SIZE_MB")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
        })
        .unwrap_or(256)
        * 1024
        * 1024;

    let (server, serving_endpoint): (Box<dyn TransportServer>, String) =
        match args.transport.as_str() {
            "socket" => {
                log::info!("Socket: {}", args.socket);
                (
                    Box::new(IpcServer::new_with_lookup(
                        &args.socket,
                        get_scheduler_lookup(),
                        None,
                    )),
                    args.socket.clone(),
                )
            }
            "tcp" => {
                log::info!("TCP Address: {}:{}", args.bind, args.port);
                (
                    Box::new(TcpServer::new_with_lookup(
                        &args.bind,
                        args.port,
                        get_scheduler_lookup(),
                    )),
                    format!("{}:{}", args.bind, args.port),
                )
            }
            "shm" => {
                log::info!("Using shared memory transport");
                let shm_name = format!("/kapsl_shm_{}", std::process::id());
                log::info!("Shared memory: {}", shm_name);
                let pools = replica_pools.read();
                (
                    Box::new(ShmServer::new_with_registry(
                        &shm_name,
                        shm_size,
                        get_trait_schedulers(&pools),
                        Some(registry.clone()),
                    )),
                    shm_name,
                )
            }
            "hybrid" => {
                log::info!("Using hybrid transport (Socket + SHM)");
                log::info!("Socket: {}", args.socket);
                let shm_name = format!("/kapsl_shm_{}", std::process::id());
                log::info!("Shared memory: {}", shm_name);

                let shm_manager = Arc::new(
                    ShmManager::create(&shm_name, shm_size)
                        .map_err(|e| format!("Failed to create SHM manager: {}", e))?,
                );

                (
                    Box::new(IpcServer::new_with_lookup(
                        &args.socket,
                        get_scheduler_lookup(),
                        Some(shm_manager),
                    )),
                    format!("{} (shm: {})", args.socket, shm_name),
                )
            }
            "auto" => {
                if ShmServer::is_available() {
                    log::info!("Auto-selecting transport: shared memory");
                    let shm_name = format!("/kapsl_shm_{}", std::process::id());
                    log::info!("Shared memory: {}", shm_name);
                    let pools = replica_pools.read();
                    (
                        Box::new(ShmServer::new_with_registry(
                            &shm_name,
                            shm_size,
                            get_trait_schedulers(&pools),
                            Some(registry.clone()),
                        )),
                        shm_name,
                    )
                } else {
                    log::info!("Auto-selecting transport: socket");
                    log::info!("Socket: {}", args.socket);
                    (
                        Box::new(IpcServer::new_with_lookup(
                            &args.socket,
                            get_scheduler_lookup(),
                            None,
                        )),
                        args.socket.clone(),
                    )
                }
            }
            _ => {
                return Err(format!(
                    "Invalid transport mode: {}. Use 'socket', 'tcp', 'shm', 'hybrid', or 'auto'",
                    args.transport
                )
                .into());
            }
        };

    log::info!("✓ Server ready\n");
    log::info!("🎉 kapsl-runtime is running!");
    log::info!("════════════════════════════════════════\n");

    let registry_arc = registry.clone();
    let model_registry_clone = model_registry.clone();
    let shared_metrics_clone = shared_metrics.clone();
    let metrics_port = args.metrics_port;
    let http_bind_addr_for_api = http_bind_addr;
    let api_auth_state_for_api = api_auth_state.clone();
    let log_sensitive_ids_for_api = log_sensitive_ids;
    let device_info_for_api = device_info.clone(); // Clone Arc for API endpoints
    let model_paths_clone = model_paths.clone();
    let auto_scaler_api = auto_scaler.clone();
    let runtime_samples_clone = runtime_samples.clone();
    let throughput_samples_clone = throughput_samples.clone();
    let onnx_tuning_profile_for_api = onnx_tuning_profile.clone();

    let extensions_root = state_layout.extensions_root.clone();
    let extensions_config_root = state_layout.extensions_config_root.clone();
    fs::create_dir_all(&extensions_root)?;
    fs::create_dir_all(&extensions_config_root)?;
    let extension_registry = ExtensionRegistry::new(extensions_root);
    let extension_manager = Arc::new(ExtensionManager::new(
        extension_registry,
        extensions_config_root,
    ));
    let running_connectors: Arc<
        AsyncMutex<HashMap<String, ConnectorClient<ConnectorRuntimeHandle>>>,
    > = Arc::new(AsyncMutex::new(HashMap::new()));
    let rag_root = state_layout.rag_root.clone();
    let rag_docs_root = rag_root.join("docs");
    let rag_vector_path = rag_root.join("vectors.sqlite3");
    fs::create_dir_all(&rag_docs_root)?;
    let rag_state = RagRuntimeState {
        vector_store: Arc::new(SqliteVectorStore::open(&rag_vector_path)?),
        doc_store: FsDocStore::new(&rag_docs_root),
    };

    let (http_ready_tx, http_ready_rx) =
        tokio::sync::oneshot::channel::<Result<std::net::SocketAddr, String>>();

    tokio::spawn(async move {
        // Metrics endpoint (admin scope when auth is enabled; loopback only when disabled)
        let metrics_route = warp::path("metrics")
            .and(warp::get())
            .map(move || {
                let encoder = TextEncoder::new();
                let metric_families = registry_arc.gather();
                let mut buffer = vec![];
                encoder.encode(&metric_families, &mut buffer).unwrap();
                String::from_utf8(buffer).unwrap()
            })
            .map(reply_into_response);
        let metrics_route = api_auth_filter(
            ApiRole::Admin,
            ApiScope::Admin,
            api_auth_state_for_api.clone(),
        )
        .and(metrics_route)
        .map(|response: warp::reply::Response| response)
        .or_else(map_api_auth_rejection);

        // API routes
        let model_registry_for_list = model_registry_clone.clone();
        let replica_pools_for_list = replica_pools_clone.clone();
        let metrics_for_list = shared_metrics_clone.clone();
        let throughput_samples_for_list = throughput_samples_clone.clone();
        let extension_manager_for_api = extension_manager.clone();
        let running_connectors_for_api = running_connectors.clone();
        let rag_state_for_api = rag_state.clone();
        let list_models = warp::path!("api" / "models").and(warp::get()).map(move || {
            #[derive(Serialize)]
            struct ModelStatus {
                #[serde(flatten)]
                info: ModelInfo,
                active_inferences: i64,
                total_inferences: u64,
                queue_depth: (usize, usize),
                memory_usage: usize,
                gpu_utilization: f64,
                throughput: f64,
                healthy: bool,
            }

            let models = model_registry_for_list.list();
            let mut statuses = Vec::new();
            let now = Instant::now();
            let mut seen_ids = HashSet::new();
            let mut throughput_samples = throughput_samples_for_list.write();

            for model in models {
                seen_ids.insert(model.id);
                let model_id_str = model.id.to_string();
                let active = metrics_for_list
                    .active_inferences
                    .with_label_values(&[&model_id_str])
                    .get();

                let ok_label = "ok".to_string();
                let err_label = "err".to_string();
                let total = metrics_for_list
                    .inference_count
                    .with_label_values(&[&model_id_str, &ok_label])
                    .get()
                    + metrics_for_list
                        .inference_count
                        .with_label_values(&[&model_id_str, &err_label])
                        .get();

                let (queue_depth, healthy, engine_memory, engine_gpu_util) =
                    if let Some(pool) = replica_pools_for_list.read().get(&model.id) {
                        let metrics = pool.get_metrics();
                        (
                            pool.get_queue_depth(),
                            pool.is_healthy(),
                            metrics.memory_usage,
                            metrics.gpu_utilization,
                        )
                    } else {
                        ((0, 0), true, 0, 0.0)
                    };

                // `memory_usage` and `gpu_utilization` are engine-reported metrics only.
                // System-level RSS/GPU stats are exposed separately via GET /api/system/stats.
                let memory_usage = engine_memory;
                let gpu_utilization = engine_gpu_util;
                let throughput = update_throughput(&mut throughput_samples, model.id, total, now);

                statuses.push(ModelStatus {
                    info: model,
                    active_inferences: active,
                    total_inferences: total,
                    queue_depth,
                    memory_usage,
                    gpu_utilization,
                    throughput,
                    healthy,
                });
            }

            throughput_samples.retain(|id, _| seen_ids.contains(id));
            warp::reply::json(&statuses)
        });

        let model_registry_for_get = model_registry_clone.clone();
        let replica_pools_for_get = replica_pools_clone.clone();
        let metrics_for_get = shared_metrics_clone.clone();
        let throughput_samples_for_get = throughput_samples_clone.clone();
        let get_model =
            warp::path!("api" / "models" / u32)
                .and(warp::get())
                .map(move |model_id: u32| {
                    #[derive(Serialize)]
                    struct ModelDetailStatus {
                        #[serde(flatten)]
                        info: ModelInfo,
                        active_inferences: i64,
                        total_inferences: u64,
                        successful_inferences: u64,
                        failed_inferences: u64,
                        queue_depth: (usize, usize),
                        memory_usage: usize,
                        gpu_utilization: f64,
                        throughput: f64,
                        healthy: bool,
                    }

                    #[derive(Serialize)]
                    struct ErrorResponse {
                        error: String,
                    }

                    match model_registry_for_get.get(model_id) {
                        Some(model) => {
                            let model_id_str = model.id.to_string();
                            let active = metrics_for_get
                                .active_inferences
                                .with_label_values(&[&model_id_str])
                                .get();

                            let ok_label = "ok".to_string();
                            let err_label = "err".to_string();
                            let successful = metrics_for_get
                                .inference_count
                                .with_label_values(&[&model_id_str, &ok_label])
                                .get();

                            let failed = metrics_for_get
                                .inference_count
                                .with_label_values(&[&model_id_str, &err_label])
                                .get();

                            let (queue_depth, healthy, engine_memory, engine_gpu_util) =
                                if let Some(pool) = replica_pools_for_get.read().get(&model.id) {
                                    let metrics = pool.get_metrics();
                                    (
                                        pool.get_queue_depth(),
                                        pool.is_healthy(),
                                        metrics.memory_usage,
                                        metrics.gpu_utilization,
                                    )
                                } else {
                                    ((0, 0), true, 0, 0.0)
                                };

                            // `memory_usage` and `gpu_utilization` are engine-reported metrics only.
                            // System-level RSS/GPU stats are exposed separately via GET /api/system/stats.
                            let memory_usage = engine_memory;
                            let gpu_utilization = engine_gpu_util;
                            let throughput = {
                                let now = Instant::now();
                                let mut throughput_samples = throughput_samples_for_get.write();
                                update_throughput(
                                    &mut throughput_samples,
                                    model.id,
                                    successful + failed,
                                    now,
                                )
                            };

                            let status = ModelDetailStatus {
                                info: model,
                                active_inferences: active,
                                total_inferences: successful + failed,
                                successful_inferences: successful,
                                failed_inferences: failed,
                                queue_depth,
                                memory_usage,
                                gpu_utilization,
                                throughput,
                                healthy,
                            };

                            warp::reply::json(&status)
                        }
                        None => warp::reply::json(&ErrorResponse {
                            error: format!("Model {} not found", model_id),
                        }),
                    }
                });

        let model_registry_for_health = model_registry_clone.clone();
        let replica_pools_for_health = replica_pools_clone.clone();
        let health = warp::path!("api" / "health").and(warp::get()).map(move || {
            let total = model_registry_for_health.count();
            let mut healthy = 0;
            let mut unhealthy = 0;

            for model in model_registry_for_health.list() {
                if let Some(pool) = replica_pools_for_health.read().get(&model.id) {
                    if pool.is_healthy() {
                        healthy += 1;
                    } else {
                        unhealthy += 1;
                    }
                } else {
                    healthy += 1;
                }
            }

            let overall_status = if unhealthy == 0 {
                "healthy"
            } else {
                "degraded"
            };
            let response = json!({
                "status": overall_status.to_string(),
                "total_models": total,
                "healthy_models": healthy,
                "unhealthy_models": unhealthy,
            });
            warp::reply::json(&response)
        });

        // Hardware info endpoint
        let device_info_for_hw = device_info_for_api.clone();
        let hardware = warp::path!("api" / "hardware")
            .and(warp::get())
            .map(move || warp::reply::json(&*device_info_for_hw));

        // System-level runtime stats (process RSS, GPU utilization, etc).
        let runtime_samples_for_system_stats = runtime_samples_clone.clone();
        let runtime_pressure_state_for_system_stats = runtime_pressure_state.clone();
        let system_stats = warp::path!("api" / "system" / "stats")
            .and(warp::get())
            .map(move || {
                #[derive(Serialize)]
                struct SystemStatsResponse {
                    pid: u32,
                    process_memory_bytes: usize,
                    total_system_memory_bytes: Option<usize>,
                    gpu_utilization: f64,
                    gpu_memory_bytes: Option<usize>,
                    gpu_memory_total_bytes: Option<usize>,
                    pressure_state: String,
                    collected_at_ms: u64,
                }

                let samples = runtime_samples_for_system_stats.read().clone();
                let pressure_state = RuntimePressureState::from_u8(
                    runtime_pressure_state_for_system_stats.load(Ordering::Relaxed),
                );
                warp::reply::json(&SystemStatsResponse {
                    pid: std::process::id(),
                    process_memory_bytes: samples.process_memory_bytes,
                    total_system_memory_bytes: samples.total_system_memory_bytes,
                    gpu_utilization: samples.gpu_utilization,
                    gpu_memory_bytes: samples.gpu_memory_bytes,
                    gpu_memory_total_bytes: samples.gpu_memory_total_bytes,
                    pressure_state: pressure_state.as_str().to_string(),
                    collected_at_ms: samples.collected_at_ms,
                })
            });

        // Engine packaging + remote registry endpoints.
        let package_kapsl = warp::path!("api" / "engine" / "package")
            .and(warp::post())
            .and(warp::body::json())
            .map(|request: PackageKapslRequest| {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                match create_kapsl_package(&request) {
                    Ok(response) => {
                        warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                    }
                    Err(error) => warp::reply::with_status(
                        warp::reply::json(&ErrorResponse { error }),
                        StatusCode::BAD_REQUEST,
                    ),
                }
            });

        let push_kapsl = warp::path!("api" / "engine" / "push")
            .and(warp::post())
            .and(warp::body::json())
            .map(|request: PushKapslRequest| {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                match push_kapsl_to_placeholder_remote(&request) {
                    Ok(response) => {
                        warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                    }
                    Err(error) => warp::reply::with_status(
                        warp::reply::json(&ErrorResponse { error }),
                        StatusCode::BAD_REQUEST,
                    ),
                }
            });

        let pull_kapsl = warp::path!("api" / "engine" / "pull")
            .and(warp::post())
            .and(warp::body::json())
            .map(|request: PullKapslRequest| {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                match pull_kapsl_from_placeholder_remote(&request) {
                    Ok(response) => {
                        warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                    }
                    Err(error) => warp::reply::with_status(
                        warp::reply::json(&ErrorResponse { error }),
                        StatusCode::BAD_REQUEST,
                    ),
                }
            });

        // Extensions API
        #[derive(Deserialize)]
        struct InstallExtensionRequest {
            path: Option<String>,
            extension_id: Option<String>,
            marketplace_url: Option<String>,
        }

        #[derive(Deserialize)]
        struct ExtensionConfigRequest {
            workspace_id: String,
            config: serde_json::Value,
        }

        #[derive(Deserialize)]
        struct LaunchExtensionRequest {
            workspace_id: String,
        }

        let extension_manager_for_list = extension_manager_for_api.clone();
        let list_extensions = warp::path!("api" / "extensions")
            .and(warp::get())
            .map(move || {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                let extensions = match extension_manager_for_list.registry.discover() {
                    Ok(list) => list,
                    Err(err) => {
                        return warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: err.to_string(),
                            }),
                            StatusCode::INTERNAL_SERVER_ERROR,
                        );
                    }
                };

                let payload: Vec<_> = extensions
                    .into_iter()
                    .map(|ext| {
                        json!({
                            "manifest": ext.manifest,
                            "path": ext.path.to_string_lossy()
                        })
                    })
                    .collect();

                warp::reply::with_status(warp::reply::json(&payload), StatusCode::OK)
            });

        let list_marketplace_extensions = warp::path!("api" / "extensions" / "marketplace")
            .and(warp::get())
            .and(warp::query::<HashMap<String, String>>())
            .map(move |query: HashMap<String, String>| {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                let q = query.get("q").map(String::as_str);
                let marketplace_url = query.get("marketplace_url").map(String::as_str);

                match fetch_extension_marketplace(q, marketplace_url) {
                    Ok(payload) => {
                        warp::reply::with_status(warp::reply::json(&payload), StatusCode::OK)
                    }
                    Err(error) => warp::reply::with_status(
                        warp::reply::json(&ErrorResponse { error }),
                        StatusCode::BAD_GATEWAY,
                    ),
                }
            });

        let extension_manager_for_install = extension_manager_for_api.clone();
        let install_extension = warp::path!("api" / "extensions" / "install")
            .and(warp::post())
            .and(warp::body::json())
            .map(move |req: InstallExtensionRequest| {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                let path = req.path.as_deref().unwrap_or("").trim().to_string();
                let extension_id = req.extension_id.as_deref().unwrap_or("").trim().to_string();

                if path.is_empty() && extension_id.is_empty() {
                    return warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: "Provide either `path` or `extension_id`".to_string(),
                        }),
                        StatusCode::BAD_REQUEST,
                    );
                }

                let install_result: Result<InstalledExtension, String> = if !path.is_empty() {
                    let path = PathBuf::from(&path);
                    if !path.exists() {
                        return warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: format!("Extension path does not exist: {}", path.display()),
                            }),
                            StatusCode::BAD_REQUEST,
                        );
                    }
                    extension_manager_for_install
                        .registry
                        .install_from_dir(&path)
                        .map_err(|e| e.to_string())
                } else {
                    install_extension_from_marketplace(
                        &extension_manager_for_install.registry,
                        &extension_id,
                        req.marketplace_url.as_deref(),
                    )
                };

                match install_result {
                    Ok(ext) => warp::reply::with_status(
                        warp::reply::json(&json!({
                            "status": "ok",
                            "extension": {
                                "manifest": ext.manifest,
                                "path": ext.path.to_string_lossy()
                            }
                        })),
                        StatusCode::OK,
                    ),
                    Err(err) => warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: err.to_string(),
                        }),
                        StatusCode::BAD_REQUEST,
                    ),
                }
            });

        let extension_manager_for_uninstall = extension_manager_for_api.clone();
        let uninstall_extension = warp::path!("api" / "extensions" / String / "uninstall")
            .and(warp::post())
            .map(move |extension_id: String| {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                match extension_manager_for_uninstall
                    .registry
                    .uninstall(&extension_id)
                {
                    Ok(()) => warp::reply::with_status(
                        warp::reply::json(&json!({ "status": "ok" })),
                        StatusCode::OK,
                    ),
                    Err(err) => warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: err.to_string(),
                        }),
                        StatusCode::BAD_REQUEST,
                    ),
                }
            });

        let extension_manager_for_config = extension_manager_for_api.clone();
        let set_extension_config = warp::path!("api" / "extensions" / String / "config")
            .and(warp::post())
            .and(warp::body::json())
            .map(move |extension_id: String, req: ExtensionConfigRequest| {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                match extension_manager_for_config.set_workspace_config(
                    &req.workspace_id,
                    &extension_id,
                    &req.config,
                ) {
                    Ok(()) => warp::reply::with_status(
                        warp::reply::json(&json!({ "status": "ok" })),
                        StatusCode::OK,
                    ),
                    Err(err) => warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: err.to_string(),
                        }),
                        StatusCode::BAD_REQUEST,
                    ),
                }
            });

        let extension_manager_for_get_config = extension_manager_for_api.clone();
        let get_extension_config = warp::path!("api" / "extensions" / String / "config")
            .and(warp::get())
            .and(warp::query::<HashMap<String, String>>())
            .map(
                move |extension_id: String, query: HashMap<String, String>| {
                    use warp::http::StatusCode;

                    #[derive(Serialize)]
                    struct ErrorResponse {
                        error: String,
                    }

                    let workspace_id = match query.get("workspace_id") {
                        Some(id) if !id.trim().is_empty() => id.to_string(),
                        _ => {
                            return warp::reply::with_status(
                                warp::reply::json(&ErrorResponse {
                                    error: "workspace_id query parameter is required".to_string(),
                                }),
                                StatusCode::BAD_REQUEST,
                            );
                        }
                    };

                    match extension_manager_for_get_config
                        .get_workspace_config(&workspace_id, &extension_id)
                    {
                        Ok(config) => warp::reply::with_status(
                            warp::reply::json(&json!({ "config": config })),
                            StatusCode::OK,
                        ),
                        Err(err) => warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: err.to_string(),
                            }),
                            StatusCode::BAD_REQUEST,
                        ),
                    }
                },
            );

        let extension_manager_for_launch = extension_manager_for_api.clone();
        let running_connectors_for_launch = running_connectors_for_api.clone();
        let launch_extension = warp::path!("api" / "extensions" / String / "launch")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |extension_id: String, req: LaunchExtensionRequest| {
                let extension_manager = extension_manager_for_launch.clone();
                let running_connectors = running_connectors_for_launch.clone();
                async move {
                    use warp::http::StatusCode;

                    #[derive(Serialize)]
                    struct ErrorResponse {
                        error: String,
                    }

                    let key = extension_key(&req.workspace_id, &extension_id);
                    {
                        let running = running_connectors.lock().await;
                        if running.contains_key(&key) {
                            return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                warp::reply::json(&ErrorResponse {
                                    error: "connector already running".to_string(),
                                }),
                                StatusCode::CONFLICT,
                            ));
                        }
                    }

                    let extensions = match extension_manager.registry.discover() {
                        Ok(list) => list,
                        Err(err) => {
                            return Ok(warp::reply::with_status(
                                warp::reply::json(&ErrorResponse {
                                    error: err.to_string(),
                                }),
                                StatusCode::INTERNAL_SERVER_ERROR,
                            ));
                        }
                    };
                    let extension = match extensions
                        .into_iter()
                        .find(|ext| ext.manifest.id == extension_id)
                    {
                        Some(ext) => ext,
                        None => {
                            return Ok(warp::reply::with_status(
                                warp::reply::json(&ErrorResponse {
                                    error: format!("Extension {} not found", extension_id),
                                }),
                                StatusCode::NOT_FOUND,
                            ));
                        }
                    };

                    let mut client =
                        match extension_manager.launch_connector(&req.workspace_id, &extension) {
                            Ok(client) => client,
                            Err(err) => {
                                return Ok(warp::reply::with_status(
                                    warp::reply::json(&ErrorResponse {
                                        error: err.to_string(),
                                    }),
                                    StatusCode::BAD_REQUEST,
                                ));
                            }
                        };

                    let config = match extension_manager
                        .get_workspace_connector_config(&req.workspace_id, &extension_id)
                    {
                        Ok(config) => config.unwrap_or_else(|| json!({})),
                        Err(err) => {
                            let _ = client.shutdown();
                            return Ok(warp::reply::with_status(
                                warp::reply::json(&ErrorResponse {
                                    error: err.to_string(),
                                }),
                                StatusCode::BAD_REQUEST,
                            ));
                        }
                    };

                    let response =
                        match client.request(ConnectorRequestKind::ValidateConfig { config }) {
                            Ok(response) => response,
                            Err(err) => {
                                let _ = client.shutdown();
                                return Ok(warp::reply::with_status(
                                    warp::reply::json(&ErrorResponse {
                                        error: err.to_string(),
                                    }),
                                    StatusCode::BAD_REQUEST,
                                ));
                            }
                        };

                    if let ConnectorResponseKind::Err(err) = response.kind {
                        let _ = client.shutdown();
                        return Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse { error: err.message }),
                            StatusCode::BAD_REQUEST,
                        ));
                    }

                    let mut running = running_connectors.lock().await;
                    if running.contains_key(&key) {
                        let _ = client.shutdown();
                        return Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: "connector already running".to_string(),
                            }),
                            StatusCode::CONFLICT,
                        ));
                    }
                    running.insert(key, client);

                    Ok::<_, warp::Rejection>(warp::reply::with_status(
                        warp::reply::json(&json!({ "status": "ok" })),
                        StatusCode::OK,
                    ))
                }
            });

        let extension_manager_for_sync = extension_manager_for_api.clone();
        let running_connectors_for_sync = running_connectors_for_api.clone();
        let rag_state_for_sync = rag_state_for_api.clone();
        let sync_extension = warp::path!("api" / "extensions" / String / "sync")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |extension_id: String, req: SyncExtensionRequest| {
                let extension_manager = extension_manager_for_sync.clone();
                let running_connectors = running_connectors_for_sync.clone();
                let rag_state = rag_state_for_sync.clone();
                async move {
                    use warp::http::StatusCode;

                    #[derive(Serialize)]
                    struct ErrorResponse {
                        error: String,
                    }

                    let workspace_id = req.workspace_id.trim().to_string();
                    if workspace_id.is_empty() {
                        return Ok::<_, warp::Rejection>(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: "workspace_id is required".to_string(),
                            }),
                            StatusCode::BAD_REQUEST,
                        ));
                    }

                    let connector_config = match extension_manager
                        .get_workspace_connector_config(&workspace_id, &extension_id)
                    {
                        Ok(Some(config)) => config,
                        Ok(None) => {
                            return Ok(warp::reply::with_status(
                                warp::reply::json(&ErrorResponse {
                                    error: format!(
                                        "No connector config found for workspace `{}`",
                                        workspace_id
                                    ),
                                }),
                                StatusCode::BAD_REQUEST,
                            ));
                        }
                        Err(err) => {
                            return Ok(warp::reply::with_status(
                                warp::reply::json(&ErrorResponse {
                                    error: err.to_string(),
                                }),
                                StatusCode::BAD_REQUEST,
                            ));
                        }
                    };

                    let key = extension_key(&workspace_id, &extension_id);
                    let (source_id, deltas, upsert_payloads, delete_doc_ids, fetch_failures) = {
                        let mut running = running_connectors.lock().await;
                        let client = match running.get_mut(&key) {
                            Some(client) => client,
                            None => {
                                return Ok(warp::reply::with_status(
                                    warp::reply::json(&ErrorResponse {
                                        error: format!(
                                            "Connector `{}` is not running for workspace `{}`",
                                            extension_id, workspace_id
                                        ),
                                    }),
                                    StatusCode::CONFLICT,
                                ));
                            }
                        };

                        let source_id = match select_sync_source_id(
                            req.source_id.clone(),
                            connector_config.clone(),
                            client,
                        ) {
                            Ok(source_id) => source_id,
                            Err(error) => {
                                return Ok(warp::reply::with_status(
                                    warp::reply::json(&ErrorResponse { error }),
                                    StatusCode::BAD_REQUEST,
                                ));
                            }
                        };

                        let sync_response = match client.request(ConnectorRequestKind::Sync {
                            source_id: source_id.clone(),
                            cursor: req.cursor.clone(),
                        }) {
                            Ok(response) => response,
                            Err(error) => {
                                return Ok(warp::reply::with_status(
                                    warp::reply::json(&ErrorResponse {
                                        error: format!("Sync request failed: {}", error),
                                    }),
                                    StatusCode::BAD_REQUEST,
                                ));
                            }
                        };

                        let deltas: Vec<DocumentDelta> = match sync_response.kind {
                            ConnectorResponseKind::Err(err) => {
                                return Ok(warp::reply::with_status(
                                    warp::reply::json(&ErrorResponse { error: err.message }),
                                    StatusCode::BAD_REQUEST,
                                ));
                            }
                            ConnectorResponseKind::Ok(ConnectorResult::Deltas(deltas)) => deltas,
                            _ => {
                                return Ok(warp::reply::with_status(
                                    warp::reply::json(&ErrorResponse {
                                        error: "connector returned unexpected Sync response"
                                            .to_string(),
                                    }),
                                    StatusCode::BAD_REQUEST,
                                ));
                            }
                        };

                        let mut upsert_payloads = Vec::new();
                        let mut delete_doc_ids = Vec::new();
                        let mut fetch_failures = 0usize;
                        for delta in &deltas {
                            match delta.op {
                                DeltaOp::Delete => delete_doc_ids.push(delta.id.clone()),
                                DeltaOp::Upsert => {
                                    let response = match client.request(
                                        ConnectorRequestKind::FetchDocument {
                                            document_id: delta.id.clone(),
                                        },
                                    ) {
                                        Ok(response) => response,
                                        Err(error) => {
                                            fetch_failures += 1;
                                            log::warn!(
                                                "FetchDocument request failed: extension_id={} workspace_id={} source_id={} doc_id={} error={}",
                                                extension_id,
                                                workspace_id,
                                                source_id,
                                                delta.id,
                                                error
                                            );
                                            continue;
                                        }
                                    };

                                    match response.kind {
                                        ConnectorResponseKind::Err(err) => {
                                            fetch_failures += 1;
                                            log::warn!(
                                                "FetchDocument rejected by connector: extension_id={} workspace_id={} source_id={} doc_id={} error={}",
                                                extension_id,
                                                workspace_id,
                                                source_id,
                                                delta.id,
                                                err.message
                                            );
                                        }
                                        ConnectorResponseKind::Ok(ConnectorResult::Document(
                                            document,
                                        )) => {
                                            upsert_payloads.push(document);
                                        }
                                        _ => {
                                            fetch_failures += 1;
                                            log::warn!(
                                                "FetchDocument returned unexpected response: extension_id={} workspace_id={} source_id={} doc_id={}",
                                                extension_id,
                                                workspace_id,
                                                source_id,
                                                delta.id
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        (
                            source_id,
                            deltas,
                            upsert_payloads,
                            delete_doc_ids,
                            fetch_failures,
                        )
                    };

                    let tenant_id = normalize_tenant_id(req.tenant_id.as_deref());
                    let mut deleted = 0usize;
                    let mut upserted = 0usize;
                    let mut skipped_non_text = 0usize;
                    let mut failed = fetch_failures;
                    let mut chunk_count = 0usize;

                    for doc_id in delete_doc_ids {
                        match delete_document_from_rag(
                            &rag_state,
                            &tenant_id,
                            &workspace_id,
                            &source_id,
                            &doc_id,
                        )
                        .await
                        {
                            Ok(()) => deleted += 1,
                            Err(error) => {
                                failed += 1;
                                log::warn!(
                                    "Failed to delete document from RAG store: workspace_id={} source_id={} doc_id={} error={}",
                                    workspace_id,
                                    source_id,
                                    doc_id,
                                    error
                                );
                            }
                        }
                    }

                    for payload in upsert_payloads {
                        match ingest_document_payload_into_rag(
                            &rag_state,
                            &tenant_id,
                            &workspace_id,
                            &source_id,
                            &payload,
                        )
                        .await
                        {
                            Ok(chunks) if chunks > 0 => {
                                upserted += 1;
                                chunk_count += chunks;
                            }
                            Ok(_) => {
                                skipped_non_text += 1;
                            }
                            Err(error) => {
                                if error.contains("unsupported non-text content type")
                                    || error.contains("no text content")
                                {
                                    skipped_non_text += 1;
                                } else {
                                    failed += 1;
                                    log::warn!(
                                        "Failed to ingest document into RAG store: workspace_id={} source_id={} doc_id={} error={}",
                                        workspace_id,
                                        source_id,
                                        payload.id,
                                        error
                                    );
                                }
                            }
                        }
                    }

                    let next_cursor = deltas
                        .iter()
                        .filter_map(|delta| delta.modified_at.clone())
                        .max();

                    Ok::<_, warp::Rejection>(warp::reply::with_status(
                        warp::reply::json(&json!({
                            "status": "ok",
                            "workspace_id": workspace_id,
                            "extension_id": extension_id,
                            "source_id": source_id,
                            "tenant_id": tenant_id,
                            "deltas_total": deltas.len(),
                            "upserted_docs": upserted,
                            "deleted_docs": deleted,
                            "skipped_docs": skipped_non_text,
                            "failed_docs": failed,
                            "chunks_upserted": chunk_count,
                            "next_cursor": next_cursor,
                        })),
                        StatusCode::OK,
                    ))
                }
            });

        let rag_state_for_query = rag_state_for_api.clone();
        let query_rag = warp::path!("api" / "rag" / "query")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |req: RagQueryRequest| {
                let rag_state = rag_state_for_query.clone();
                async move {
                    use warp::http::StatusCode;

                    #[derive(Serialize)]
                    struct ErrorResponse {
                        error: String,
                    }

                    let workspace_id = req.workspace_id.trim().to_string();
                    if workspace_id.is_empty() {
                        return Ok::<_, warp::Rejection>(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: "workspace_id is required".to_string(),
                            }),
                            StatusCode::BAD_REQUEST,
                        ));
                    }

                    match query_rag_chunks(
                        &rag_state,
                        &workspace_id,
                        req.tenant_id.as_deref(),
                        &req.query,
                        req.source_id,
                        req.source_ids,
                        req.top_k,
                        req.min_score,
                        req.allowed_users,
                        req.allowed_groups,
                    )
                    .await
                    {
                        Ok(matches) => {
                            let count = matches.len();
                            Ok(warp::reply::with_status(
                                warp::reply::json(&json!({
                                    "status": "ok",
                                    "workspace_id": workspace_id,
                                    "matches": matches,
                                    "count": count
                                })),
                                StatusCode::OK,
                            ))
                        }
                        Err(RagAugmentError::BadRequest(error)) => Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse { error }),
                            StatusCode::BAD_REQUEST,
                        )),
                        Err(RagAugmentError::Internal(error)) => Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse { error }),
                            StatusCode::INTERNAL_SERVER_ERROR,
                        )),
                    }
                }
            });

        // POST /api/models/start - Start a new model
        #[derive(Deserialize)]
        struct StartModelRequest {
            model_path: String,
            model_id: Option<u32>,
            #[serde(default = "default_topology")]
            topology: String,
            #[serde(default = "default_tp_degree")]
            tp_degree: usize,
        }

        fn default_topology() -> String {
            "data-parallel".to_string()
        }

        fn default_tp_degree() -> usize {
            1
        }

        let model_registry_for_start = model_registry_clone.clone();
        let replica_pools_for_start = replica_pools_clone.clone();
        let device_info_for_start = device_info_for_api.clone();
        let batch_size_for_start = args.batch_size;
        let scheduler_queue_size_for_start = args.scheduler_queue_size;
        let scheduler_max_micro_batch_for_start = args.scheduler_max_micro_batch;
        let scheduler_queue_delay_ms_for_start = args.scheduler_queue_delay_ms;
        let shared_metrics_for_start = shared_metrics_clone.clone();
        let model_id_counter_for_start = model_id_counter.clone();
        let recycled_model_ids_for_start = recycled_model_ids.clone();
        let model_paths_for_start = model_paths_clone.clone();
        let onnx_tuning_profile_for_start = onnx_tuning_profile_for_api.clone();

        let start_model = warp::path!("api" / "models" / "start")
            .and(warp::post())
            .and(warp::body::json())
            .then(move |req: StartModelRequest| {
                let model_registry = model_registry_for_start.clone();
                let replica_pools = replica_pools_for_start.clone();
                let device_info = device_info_for_start.clone();
                let shared_metrics = shared_metrics_for_start.clone();
                let model_id_counter = model_id_counter_for_start.clone();
                let recycled_model_ids = recycled_model_ids_for_start.clone();
                let model_paths = model_paths_for_start.clone();
                let onnx_tuning_profile = onnx_tuning_profile_for_start.clone();

                async move {
                    use warp::http::StatusCode;

                    #[derive(Serialize)]
                    struct SuccessResponse {
                        message: String,
                        model_id: u32,
                    }

                    #[derive(Serialize)]
                    struct ErrorResponse {
                        error: String,
                    }

                    // Assign ID if missing
                    let (model_id, auto_assigned) = match req.model_id {
                        Some(id) => (id, false),
                        None => (
                            allocate_model_id(&model_id_counter, &recycled_model_ids),
                            true,
                        ),
                    };
                    let onnx_tuning = onnx_tuning_profile.resolve(model_id);

                    // Check if model ID already exists
                    if replica_pools.read().contains_key(&model_id) {
                        if auto_assigned {
                            recycle_model_id(model_id, &recycled_model_ids);
                        }
                        return warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: format!("Model ID {} already exists", model_id),
                            }),
                            StatusCode::BAD_REQUEST,
                        );
                    }
                    if let Some(info) = model_registry.get(model_id) {
                        match info.status {
                            ModelStatus::Inactive => {}
                            ModelStatus::Starting | ModelStatus::Loading => {
                                if auto_assigned {
                                    recycle_model_id(model_id, &recycled_model_ids);
                                }
                                return warp::reply::with_status(
                                    warp::reply::json(&ErrorResponse {
                                        error: format!("Model ID {} is already starting", model_id),
                                    }),
                                    StatusCode::BAD_REQUEST,
                                );
                            }
                            _ => {
                                if auto_assigned {
                                    recycle_model_id(model_id, &recycled_model_ids);
                                }
                                return warp::reply::with_status(
                                    warp::reply::json(&ErrorResponse {
                                        error: format!(
                                            "Model ID {} already exists (status: {:?})",
                                            model_id, info.status
                                        ),
                                    }),
                                    StatusCode::BAD_REQUEST,
                                );
                            }
                        }
                    }

                    // Load the model
                    let model_path = PathBuf::from(&req.model_path);
                    log::info!(
                        "Attempting to start model {} from path: {:?}",
                        model_id,
                        model_path
                    );

                    if !model_path.exists() {
                        log::error!("Model path does not exist: {:?}", model_path);
                        if auto_assigned {
                            recycle_model_id(model_id, &recycled_model_ids);
                        }
                        return warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: format!("Model path does not exist: {:?}", model_path),
                            }),
                            StatusCode::BAD_REQUEST,
                        );
                    }
                    let absolute_path = match model_path.canonicalize() {
                        Ok(p) => p,
                        Err(e) => {
                            log::error!(
                                "Failed to canonicalize model path {:?}: {}",
                                model_path,
                                e
                            );
                            if auto_assigned {
                                recycle_model_id(model_id, &recycled_model_ids);
                            }
                            return warp::reply::with_status(
                                warp::reply::json(&ErrorResponse {
                                    error: format!("Invalid model path {:?}: {}", model_path, e),
                                }),
                                StatusCode::BAD_REQUEST,
                            );
                        }
                    };

                    if let Some(info) = model_registry.get(model_id) {
                        if info.status == ModelStatus::Inactive {
                            let _ = model_registry.set_status(model_id, ModelStatus::Starting);
                        }
                    }

                    let model_name = absolute_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let device_str = device_info.get_best_provider().to_string();
                    let optimization_level = "basic".to_string();
                    let mut model_info = ModelInfo::new(
                        model_id,
                        model_name,
                        "unknown".to_string(),
                        "unknown".to_string(),
                        device_str,
                        optimization_level,
                        absolute_path.to_string_lossy().to_string(),
                    );
                    model_info.status = ModelStatus::Starting;
                    model_registry.upsert(model_info);
                    tokio::spawn({
                        let replica_pools = replica_pools.clone();
                        let model_paths = model_paths.clone();
                        let model_registry = model_registry.clone();
                        let device_info = device_info.clone();
                        let shared_metrics = shared_metrics.clone();
                        let model_path = model_path.clone();
                        let topology = req.topology.clone();
                        let recycled_model_ids = recycled_model_ids.clone();
                        let tp_degree = req.tp_degree;
                        let onnx_tuning = onnx_tuning.clone();
                        async move {
                            let model_registry_clone = model_registry.clone();
                            let device_info_clone = device_info.clone();
                            let shared_metrics_clone = shared_metrics.clone();
                            let model_path_thread = model_path.clone();
                            let topology_clone = topology.clone();

                            let res = tokio::task::spawn_blocking(move || {
                                load_model_blocking(
                                    model_id,
                                    &model_path_thread,
                                    &device_info_clone,
                                    batch_size_for_start,
                                    scheduler_queue_size_for_start,
                                    scheduler_max_micro_batch_for_start,
                                    scheduler_queue_delay_ms_for_start,
                                    &model_registry_clone,
                                    &shared_metrics_clone,
                                    &topology_clone,
                                    tp_degree,
                                    onnx_tuning.clone(),
                                )
                            })
                            .await;

                            match res {
                                Ok(Err(e)) => {
                                    log::error!("Failed to load model {}: {}", model_id, e);
                                    let _ =
                                        model_registry.set_status(model_id, ModelStatus::Inactive);
                                    if auto_assigned {
                                        recycle_model_id(model_id, &recycled_model_ids);
                                    }
                                }
                                Ok(Ok(pool)) => {
                                    replica_pools.write().insert(model_id, pool);
                                    model_paths.write().insert(model_id, model_path);
                                    let _ =
                                        model_registry.set_status(model_id, ModelStatus::Active);
                                }
                                Err(join_err) => {
                                    log::error!(
                                        "Loader task panicked/cancelled for {}: {}",
                                        model_id,
                                        join_err
                                    );
                                    let _ =
                                        model_registry.set_status(model_id, ModelStatus::Inactive);
                                    if auto_assigned {
                                        recycle_model_id(model_id, &recycled_model_ids);
                                    }
                                }
                            }
                        }
                    });
                    warp::reply::with_status(
                        warp::reply::json(&SuccessResponse {
                            message: "Model load started".to_string(),
                            model_id,
                        }),
                        StatusCode::ACCEPTED,
                    )
                }
            });

        // POST /api/models/:id/stop - Stop a model
        let model_registry_for_stop = model_registry_clone.clone();
        let replica_pools_for_stop = replica_pools_clone.clone();

        let stop_model = warp::path!("api" / "models" / u32 / "stop")
            .and(warp::post())
            .map(move |model_id: u32| {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct SuccessResponse {
                    message: String,
                }

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                // Check if model exists
                if !replica_pools_for_stop.read().contains_key(&model_id) {
                    return warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: format!("Model ID {} not found", model_id),
                        }),
                        StatusCode::NOT_FOUND,
                    );
                }

                // Update status to Stopping
                if let Err(e) = model_registry_for_stop.set_status(model_id, ModelStatus::Stopping)
                {
                    log::warn!(
                        "Failed to set status to Stopping during stop request for {}: {}",
                        model_id,
                        e
                    );
                    // Proceed anyway? Yes, we want to stop.
                }

                // Remove pool (this will drop it and clean up resources)
                replica_pools_for_stop.write().remove(&model_id);

                // Update status to Inactive
                if let Err(e) = model_registry_for_stop.set_status(model_id, ModelStatus::Inactive)
                {
                    log::warn!(
                        "Failed to set status to Inactive after stop for {}: {}",
                        model_id,
                        e
                    );
                }

                warp::reply::with_status(
                    warp::reply::json(&SuccessResponse {
                        message: format!("Model {} stopped successfully", model_id),
                    }),
                    StatusCode::OK,
                )
            });

        // POST /api/models/:id/remove - Remove a model and its replicas
        let model_registry_for_remove = model_registry_clone.clone();
        let replica_pools_for_remove = replica_pools_clone.clone();
        let model_paths_for_remove = model_paths_clone.clone();

        let remove_model = warp::path!("api" / "models" / u32 / "remove")
            .and(warp::post())
            .map(move |model_id: u32| {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct SuccessResponse {
                    message: String,
                }

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                let model_info = match model_registry_for_remove.get(model_id) {
                    Some(info) => info,
                    None => {
                        return warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: format!("Model ID {} not found", model_id),
                            }),
                            StatusCode::NOT_FOUND,
                        );
                    }
                };

                let base_model_id = model_info.base_model_id;
                let replicas = model_registry_for_remove.list_replicas(base_model_id);

                for replica in &replicas {
                    let _ = model_registry_for_remove.set_status(replica.id, ModelStatus::Stopping);
                }

                replica_pools_for_remove.write().remove(&base_model_id);
                model_paths_for_remove.write().remove(&base_model_id);

                for replica in replicas {
                    model_registry_for_remove.unregister(replica.id);
                }

                warp::reply::with_status(
                    warp::reply::json(&SuccessResponse {
                        message: format!("Model {} removed successfully", base_model_id),
                    }),
                    StatusCode::OK,
                )
            });

        // POST /api/models/:id/infer - Synchronous inference
        let replica_pools_for_infer = replica_pools_clone.clone();
        let model_registry_for_infer = model_registry_clone.clone();
        let request_adapters_for_infer = Arc::new(default_request_adapter_registry());
        let log_sensitive_ids_for_infer = log_sensitive_ids_for_api;
        let rag_state_for_infer = rag_state_for_api.clone();
        let inter_model_relay_for_infer = inter_model_relay_state.clone();
        let runtime_pressure_state_for_infer = runtime_pressure_state.clone();
        let runtime_pressure_config_for_infer = runtime_pressure_config.clone();
        let infer_route = warp::path!("api" / "models" / u32 / "infer")
            .and(warp::post())
            .and(warp::body::bytes())
            .and_then(move |model_id: u32, body: warp::hyper::body::Bytes| {
                let pools = replica_pools_for_infer.clone();
                let model_registry = model_registry_for_infer.clone();
                let request_adapters = request_adapters_for_infer.clone();
                let log_sensitive_ids = log_sensitive_ids_for_infer;
                let rag_state = rag_state_for_infer.clone();
                let inter_model_relay = inter_model_relay_for_infer.clone();
                let runtime_pressure_state = runtime_pressure_state_for_infer.clone();
                let runtime_pressure_config = runtime_pressure_config_for_infer.clone();
                async move {
                    use warp::http::StatusCode;
                    let scheduler = {
                        let p = pools.read();
                        p.get(&model_id).cloned()
                    };

                    match scheduler {
                        Some(pool) => {
                            if !pool.is_healthy() {
                                let overload =
                                    EngineError::overloaded("Model pool is overloaded".to_string());
                                let status = status_code_for_engine_error(&overload);
                                log::warn!(
                                    "Infer request rejected: model_id={} status={} reason={}",
                                    model_id,
                                    status.as_u16(),
                                    overload
                                );
                                return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                    warp::reply::json(
                                        &serde_json::json!({ "error": overload.to_string() }),
                                    ),
                                    status,
                                ));
                            }

                            let source_model_info = model_registry.get(model_id);
                            let model_framework = source_model_info
                                .as_ref()
                                .map(|model| model.framework.clone())
                                .unwrap_or_else(|| "unknown".to_string());
                            let source_model_name = source_model_info
                                .as_ref()
                                .map(|model| model.name.clone())
                                .unwrap_or_else(|| format!("model-{model_id}"));
                            // Fast-path tensor JSON inference: avoid an intermediate `serde_json::Value`
                            // for huge tensors (ex: `[1,3,224,224]` float32 represented as a JSON byte
                            // array). Falling back keeps support for other adapters.
                            let (payload_has_media, rag_options, mut request) =
                                if let Ok(envelope) =
                                    serde_json::from_slice::<InferPayloadEnvelope<InferenceRequest>>(
                                        body.as_ref(),
                                    )
                                {
                                    let rag_options = match validate_infer_rag_options(envelope.rag)
                                    {
                                        Ok(options) => options,
                                        Err(RagAugmentError::BadRequest(error)) => {
                                            return Ok::<_, warp::Rejection>(
                                                warp::reply::with_status(
                                                    warp::reply::json(
                                                        &serde_json::json!({ "error": error }),
                                                    ),
                                                    StatusCode::BAD_REQUEST,
                                                ),
                                            );
                                        }
                                        Err(RagAugmentError::Internal(error)) => {
                                            return Ok::<_, warp::Rejection>(
                                                warp::reply::with_status(
                                                    warp::reply::json(
                                                        &serde_json::json!({ "error": error }),
                                                    ),
                                                    StatusCode::INTERNAL_SERVER_ERROR,
                                                ),
                                            );
                                        }
                                    };
                                    (false, rag_options, envelope.request)
                                } else {
                                    let payload: serde_json::Value =
                                        match serde_json::from_slice(body.as_ref()) {
                                            Ok(payload) => payload,
                                            Err(error) => {
                                                log::warn!(
                                                    "Infer payload rejected (invalid JSON): model_id={} framework={} error={}",
                                                    model_id,
                                                    model_framework,
                                                    error
                                                );
                                                return Ok::<_, warp::Rejection>(
                                                    warp::reply::with_status(
                                                        warp::reply::json(&serde_json::json!({
                                                            "error": format!("Invalid infer payload: {}", error)
                                                        })),
                                                        StatusCode::BAD_REQUEST,
                                                    ),
                                                );
                                            }
                                        };

                                    let payload_has_media = payload.get("media").is_some();
                                    let rag_options = match parse_infer_rag_options(&payload) {
                                        Ok(options) => options,
                                        Err(RagAugmentError::BadRequest(error)) => {
                                            return Ok::<_, warp::Rejection>(
                                                warp::reply::with_status(
                                                    warp::reply::json(
                                                        &serde_json::json!({ "error": error }),
                                                    ),
                                                    StatusCode::BAD_REQUEST,
                                                ),
                                            );
                                        }
                                        Err(RagAugmentError::Internal(error)) => {
                                            return Ok::<_, warp::Rejection>(
                                                warp::reply::with_status(
                                                    warp::reply::json(
                                                        &serde_json::json!({ "error": error }),
                                                    ),
                                                    StatusCode::INTERNAL_SERVER_ERROR,
                                                ),
                                            );
                                        }
                                    };

                                    let request = match parse_inference_request_with_registry(
                                        payload,
                                        &model_framework,
                                        &request_adapters,
                                    ) {
                                        Ok(request) => request,
                                        Err(parse_error) => {
                                            let error_message = parse_error.to_string();
                                            let status = if parse_error.is_internal() {
                                                log::error!(
                                                    "Infer payload preprocessing failed: model_id={} framework={} has_media={} error={}",
                                                    model_id,
                                                    model_framework,
                                                    payload_has_media,
                                                    error_message
                                                );
                                                StatusCode::INTERNAL_SERVER_ERROR
                                            } else {
                                                log::warn!(
                                                    "Infer payload rejected: model_id={} framework={} has_media={} error={}",
                                                    model_id,
                                                    model_framework,
                                                    payload_has_media,
                                                    error_message
                                                );
                                                StatusCode::BAD_REQUEST
                                            };
                                            let client_error = if status
                                                == StatusCode::INTERNAL_SERVER_ERROR
                                            {
                                                "Failed to preprocess infer payload".to_string()
                                            } else {
                                                error_message
                                            };
                                            return Ok::<_, warp::Rejection>(
                                                warp::reply::with_status(
                                                    warp::reply::json(&serde_json::json!({
                                                        "error": client_error
                                                    })),
                                                    status,
                                                ),
                                            );
                                        }
                                    };

                                    (payload_has_media, rag_options, request)
                                };
                            if payload_has_media {
                                if let Some(model_info) = pool.model_info() {
                                    if let Err(error) = validate_inference_request_against_model_info(
                                        &request,
                                        &model_info,
                                    ) {
                                        return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                            warp::reply::json(
                                                &serde_json::json!({ "error": format!("Media tensor validation failed: {}", error) }),
                                            ),
                                            StatusCode::BAD_REQUEST,
                                        ));
                                    }
                                }
                            }
                            if let Some(rag_options) = rag_options {
                                match augment_inference_request_with_rag(
                                    &mut request,
                                    &rag_options,
                                    &rag_state,
                                )
                                .await
                                {
                                    Ok(chunks_used) => {
                                        if chunks_used > 0 {
                                            log::debug!(
                                                "Applied RAG context: model_id={} workspace_id={} chunks_used={}",
                                                model_id,
                                                rag_options.workspace_id,
                                                chunks_used
                                            );
                                        }
                                    }
                                    Err(RagAugmentError::BadRequest(error)) => {
                                        return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                            warp::reply::json(
                                                &serde_json::json!({ "error": error }),
                                            ),
                                            StatusCode::BAD_REQUEST,
                                        ));
                                    }
                                    Err(RagAugmentError::Internal(error)) => {
                                        return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                            warp::reply::json(
                                                &serde_json::json!({ "error": error }),
                                            ),
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                        ));
                                    }
                                }
                            }
                            let request_id = request
                                .metadata
                                .as_ref()
                                .and_then(|metadata| metadata.request_id.as_deref())
                                .unwrap_or("-");
                            let session_id = request.session_id.as_deref().unwrap_or("-");
                            let request_is_relay = session_id.starts_with(INTER_MODEL_RELAY_SESSION_PREFIX);
                            let request_id_for_log =
                                redact_identifier_for_logs(request_id, log_sensitive_ids);
                            let session_id_for_log =
                                redact_identifier_for_logs(session_id, log_sensitive_ids);
                            let scheduler_priority = scheduler_priority_for_request(&request);
                            let force_cpu = request
                                .metadata
                                .as_ref()
                                .and_then(|metadata| metadata.force_cpu)
                                .unwrap_or(false);
                            let pressure_state = RuntimePressureState::from_u8(
                                runtime_pressure_state.load(Ordering::Relaxed),
                            );
                            if pressure_state == RuntimePressureState::Emergency
                                && matches!(scheduler_priority, kapsl_scheduler::Priority::Throughput)
                            {
                                let error = EngineError::resource_exhausted(format!(
                                    "runtime pressure {}: throughput requests are temporarily rejected",
                                    pressure_state.as_str()
                                ));
                                let status = status_code_for_engine_error(&error);
                                log::warn!(
                                    "Infer execution rejected: model_id={} framework={} request_id={} session_id={} status={} error={}",
                                    model_id,
                                    model_framework,
                                    request_id_for_log,
                                    session_id_for_log,
                                    status.as_u16(),
                                    error
                                );
                                return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({ "error": error.to_string() })),
                                    status,
                                ));
                            }
                            if let Some(cap) = runtime_pressure_config.max_new_tokens_cap(pressure_state) {
                                let metadata = request
                                    .metadata
                                    .get_or_insert_with(kapsl_engine_api::RequestMetadata::default);
                                metadata.max_new_tokens =
                                    Some(metadata.max_new_tokens.map(|existing| existing.min(cap)).unwrap_or(cap));
                            }
                            // Attach a cancellation token so that if the HTTP request is dropped
                            // (client disconnects), we can stop any queued/in-flight work.
                            struct CancelOnDrop(kapsl_engine_api::CancellationToken);
                            impl Drop for CancelOnDrop {
                                fn drop(&mut self) {
                                    self.0.cancel();
                                }
                            }

                            let cancellation_token = request
                                .cancellation
                                .get_or_insert_with(kapsl_engine_api::CancellationToken::new)
                                .clone();
                            let _cancel_on_drop = CancelOnDrop(cancellation_token.clone());

                            let timeout_ms = request
                                .metadata
                                .as_ref()
                                .and_then(|metadata| metadata.timeout_ms)
                                .filter(|ms| *ms > 0);

                            let infer_fut = pool.infer(&request, scheduler_priority, force_cpu);
                            let infer_result = if let Some(timeout_ms) = timeout_ms {
                                match tokio::time::timeout(
                                    Duration::from_millis(timeout_ms),
                                    infer_fut,
                                )
                                .await
                                {
                                    Ok(result) => result,
                                    Err(_) => {
                                        cancellation_token.cancel();
                                        Err(EngineError::timeout(format!(
                                            "Inference timed out after {timeout_ms}ms"
                                        )))
                                    }
                                }
                            } else {
                                infer_fut.await
                            };

                            match infer_result {
                                Ok(output) => {
                                    maybe_publish_inter_model_relays(
                                        &inter_model_relay,
                                        model_id,
                                        &source_model_name,
                                        request_is_relay,
                                        &output,
                                        &pools,
                                        &model_registry,
                                    );
                                    Ok::<_, warp::Rejection>(warp::reply::with_status(
                                        warp::reply::json(&output),
                                        StatusCode::OK,
                                    ))
                                }
                                Err(e) => {
                                    let status = status_code_for_engine_error(&e);
                                    if status == StatusCode::INTERNAL_SERVER_ERROR {
                                        log::error!(
                                            "Infer execution failed: model_id={} framework={} request_id={} session_id={} status={} error={}",
                                            model_id,
                                            model_framework,
                                            request_id_for_log,
                                            session_id_for_log,
                                            status.as_u16(),
                                            e
                                        );
                                    } else {
                                        log::warn!(
                                            "Infer execution rejected: model_id={} framework={} request_id={} session_id={} status={} error={}",
                                            model_id,
                                            model_framework,
                                            request_id_for_log,
                                            session_id_for_log,
                                            status.as_u16(),
                                            e
                                        );
                                    }
                                    Ok(warp::reply::with_status(
                                        warp::reply::json(&serde_json::json!({ "error": e.to_string() })),
                                        status,
                                    ))
                                }
                            }
                        }
                        None => {
                            log::warn!("Infer request received for unknown model_id={}", model_id);
                            Ok(warp::reply::with_status(
                                warp::reply::json(
                                    &serde_json::json!({ "error": format!("Model {} not found", model_id) }),
                                ),
                                StatusCode::NOT_FOUND,
                            ))
                        }
                    }
                }
            });

        // GET /api/models/:id/scaling - Get scaling policy
        let auto_scaler_for_get = auto_scaler_api.clone();
        let get_scaling = warp::path!("api" / "models" / u32 / "scaling")
            .and(warp::get())
            .map(move |model_id: u32| {
                let scaler = auto_scaler_for_get.read();
                let policy = scaler.get_policy(model_id);
                warp::reply::json(&policy)
            });

        // POST /api/models/:id/scaling - Update scaling policy
        let auto_scaler_for_post = auto_scaler_api.clone();
        let update_scaling = warp::path!("api" / "models" / u32 / "scaling")
            .and(warp::post())
            .and(warp::body::json())
            .map(move |model_id: u32, policy: ScalingPolicy| {
                use warp::http::StatusCode;

                if let Err(error) = policy.validate() {
                    return warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({ "error": error })),
                        StatusCode::BAD_REQUEST,
                    );
                }

                let mut scaler = auto_scaler_for_post.write();
                scaler.register_policy(model_id, policy);
                warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "status": "ok" })),
                    StatusCode::OK,
                )
            });

        let api_auth_state_for_get_roles = api_auth_state_for_api.clone();
        let get_role_tokens =
            warp::path!("api" / "auth" / "roles")
                .and(warp::get())
                .map(move || {
                    use warp::http::StatusCode;
                    let config = api_auth_state_for_get_roles.read().legacy_token_config();
                    warp::reply::with_status(warp::reply::json(&config), StatusCode::OK)
                });

        let api_auth_state_for_set_roles = api_auth_state_for_api.clone();
        let set_role_tokens = warp::path!("api" / "auth" / "roles")
            .and(warp::post())
            .and(warp::body::json())
            .map(move |payload: ApiRoleTokenConfig| {
                use warp::http::StatusCode;

                let mut auth_state = api_auth_state_for_set_roles.write();
                match auth_state.update_legacy_token_config(payload) {
                    Ok(config) => {
                        warp::reply::with_status(warp::reply::json(&config), StatusCode::OK)
                    }
                    Err(error) => warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({ "error": error })),
                        StatusCode::BAD_REQUEST,
                    ),
                }
            });

        let api_auth_state_for_login = api_auth_state_for_api.clone();
        let login = warp::path!("api" / "auth" / "login")
            .and(warp::post())
            .and(warp::header::optional::<String>("authorization"))
            .and(warp::addr::remote())
            .and(warp::body::json::<ApiAuthLoginRequest>())
            .map(
                move |authorization: Option<String>,
                      remote: Option<std::net::SocketAddr>,
                      payload: ApiAuthLoginRequest| {
                    use warp::http::StatusCode;

                    let mut auth_state = api_auth_state_for_login.write();
                    let status = auth_state.status_response();

                    if !status.auth_enabled {
                        if is_loopback_remote(remote) {
                            let response = ApiAuthLoginResponse {
                                authenticated: true,
                                auth_enabled: status.auth_enabled,
                                legacy_auth_enabled: status.legacy_auth_enabled,
                                role: ApiRole::Admin,
                                scopes: Vec::new(),
                                mode: "local-loopback".to_string(),
                                access: ApiAuthLoginAccess {
                                    read: true,
                                    write: true,
                                    admin: true,
                                },
                            };
                            return warp::reply::with_status(
                                warp::reply::json(&response),
                                StatusCode::OK,
                            );
                        }
                        return warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "error": "Forbidden",
                                "detail": "Authentication is disabled; this endpoint is restricted to loopback clients only."
                            })),
                            StatusCode::FORBIDDEN,
                        );
                    }

                    let token_from_body = payload.token.and_then(|token| {
                        let trimmed = token.trim();
                        if trimmed.is_empty() {
                            None
                        } else {
                            Some(trimmed.to_string())
                        }
                    });
                    let normalized_authorization = authorization
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string)
                        .or(token_from_body);

                    let Some(grant_match) = auth_state
                        .grant_from_authorization_header_read(normalized_authorization.as_deref())
                    else {
                        return warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "error": "Unauthorized",
                                "detail": "Invalid or missing API token."
                            })),
                            StatusCode::UNAUTHORIZED,
                        );
                    };

                    let ApiAuthGrantMatch {
                        grant,
                        matched_key_index,
                    } = grant_match;
                    let role = grant.role;
                    let scopes = grant.scopes.unwrap_or_default();

                    let read_allowed =
                        role.allows(ApiRole::Reader) && key_scopes_allow(&scopes, ApiScope::Read);
                    if !read_allowed {
                        return warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "error": "Forbidden",
                                "detail": "Token does not grant reader access."
                            })),
                            StatusCode::FORBIDDEN,
                        );
                    }

                    let write_allowed =
                        role.allows(ApiRole::Writer) && key_scopes_allow(&scopes, ApiScope::Write);
                    let admin_allowed =
                        role.allows(ApiRole::Admin) && key_scopes_allow(&scopes, ApiScope::Admin);

                    if let Some(key_index) = matched_key_index {
                        auth_state.touch_key_last_used_by_index(key_index, now_unix_seconds());
                    }

                    let response = ApiAuthLoginResponse {
                        authenticated: true,
                        auth_enabled: status.auth_enabled,
                        legacy_auth_enabled: status.legacy_auth_enabled,
                        role,
                        scopes,
                        mode: if matched_key_index.is_some() {
                            "api-key".to_string()
                        } else {
                            "legacy-token".to_string()
                        },
                        access: ApiAuthLoginAccess {
                            read: read_allowed,
                            write: write_allowed,
                            admin: admin_allowed,
                        },
                    };
                    warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                },
            );

        let api_auth_state_for_status = api_auth_state_for_api.clone();
        let get_access_status = warp::path!("api" / "auth" / "access" / "status")
            .and(warp::get())
            .map(move || {
                use warp::http::StatusCode;
                let auth_state = api_auth_state_for_status.read();
                warp::reply::with_status(
                    warp::reply::json(&auth_state.status_response()),
                    StatusCode::OK,
                )
            });

        let api_auth_state_for_access_roles = api_auth_state_for_api.clone();
        let get_access_roles = warp::path!("api" / "auth" / "access" / "roles")
            .and(warp::get())
            .map(move || {
                use warp::http::StatusCode;
                let auth_state = api_auth_state_for_access_roles.read();
                warp::reply::with_status(
                    warp::reply::json(&auth_state.role_summaries()),
                    StatusCode::OK,
                )
            });

        let api_auth_state_for_list_users = api_auth_state_for_api.clone();
        let list_access_users = warp::path!("api" / "auth" / "access" / "users")
            .and(warp::get())
            .map(move || {
                use warp::http::StatusCode;
                let auth_state = api_auth_state_for_list_users.read();
                warp::reply::with_status(
                    warp::reply::json(&auth_state.list_users()),
                    StatusCode::OK,
                )
            });

        let api_auth_state_for_create_user = api_auth_state_for_api.clone();
        let create_access_user = warp::path!("api" / "auth" / "access" / "users")
            .and(warp::post())
            .and(warp::body::json())
            .map(move |payload: CreateAuthUserRequest| {
                use warp::http::StatusCode;
                let mut auth_state = api_auth_state_for_create_user.write();
                match auth_state.create_user(payload) {
                    Ok(user) => {
                        warp::reply::with_status(warp::reply::json(&user), StatusCode::CREATED)
                    }
                    Err(error) => warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({ "error": error })),
                        StatusCode::BAD_REQUEST,
                    ),
                }
            });

        let api_auth_state_for_update_user = api_auth_state_for_api.clone();
        let update_access_user = warp::path!("api" / "auth" / "access" / "users" / String)
            .and(warp::patch())
            .and(warp::body::json())
            .map(move |user_id: String, payload: UpdateAuthUserRequest| {
                use warp::http::StatusCode;
                let mut auth_state = api_auth_state_for_update_user.write();
                match auth_state.update_user(&user_id, payload) {
                    Ok(user) => warp::reply::with_status(warp::reply::json(&user), StatusCode::OK),
                    Err(error) if error.contains("not found") => warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({ "error": error })),
                        StatusCode::NOT_FOUND,
                    ),
                    Err(error) => warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({ "error": error })),
                        StatusCode::BAD_REQUEST,
                    ),
                }
            });

        let api_auth_state_for_list_keys = api_auth_state_for_api.clone();
        let list_access_keys = warp::path!("api" / "auth" / "access" / "keys")
            .and(warp::get())
            .and(warp::query::<HashMap<String, String>>())
            .map(move |query: HashMap<String, String>| {
                use warp::http::StatusCode;
                let user_id = query.get("user_id").map(String::as_str);
                let auth_state = api_auth_state_for_list_keys.read();
                warp::reply::with_status(
                    warp::reply::json(&auth_state.list_keys(user_id)),
                    StatusCode::OK,
                )
            });

        let api_auth_state_for_create_key = api_auth_state_for_api.clone();
        let create_access_key = warp::path!("api" / "auth" / "access" / "users" / String / "keys")
            .and(warp::post())
            .and(warp::body::json())
            .map(move |user_id: String, payload: CreateApiKeyRequest| {
                use warp::http::StatusCode;
                let mut auth_state = api_auth_state_for_create_key.write();
                match auth_state.create_api_key(&user_id, payload) {
                    Ok(response) => {
                        warp::reply::with_status(warp::reply::json(&response), StatusCode::CREATED)
                    }
                    Err(error) if error.contains("not found") => warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({ "error": error })),
                        StatusCode::NOT_FOUND,
                    ),
                    Err(error) => warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({ "error": error })),
                        StatusCode::BAD_REQUEST,
                    ),
                }
            });

        let api_auth_state_for_revoke_key = api_auth_state_for_api.clone();
        let revoke_access_key = warp::path!("api" / "auth" / "access" / "keys" / String / "revoke")
            .and(warp::post())
            .map(move |key_id: String| {
                use warp::http::StatusCode;
                let mut auth_state = api_auth_state_for_revoke_key.write();
                match auth_state.revoke_api_key(&key_id) {
                    Ok(response) => {
                        warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                    }
                    Err(error) if error.contains("not found") => warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({ "error": error })),
                        StatusCode::NOT_FOUND,
                    ),
                    Err(error) => warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({ "error": error })),
                        StatusCode::BAD_REQUEST,
                    ),
                }
            });

        // Static file serving for web UI (from embedded assets)
        let index_route = warp::path::end().and(warp::get()).and_then(|| async {
            if let Some(content) = UiAssets::get("index.html") {
                Ok::<_, warp::Rejection>(warp::reply::with_header(
                    content.data.into_owned(),
                    "content-type",
                    "text/html; charset=utf-8",
                ))
            } else {
                Err(warp::reject::not_found())
            }
        });

        let ui_static_files = warp::path("ui")
            .and(warp::path::tail())
            .and(warp::get())
            .and_then(|tail: warp::path::Tail| async move {
                let filename = tail.as_str();
                if let Some(content) = UiAssets::get(filename) {
                    let mime_type = mime_guess::from_path(filename)
                        .first_or_octet_stream()
                        .to_string();
                    Ok::<_, warp::Rejection>(warp::reply::with_header(
                        content.data.into_owned(),
                        "content-type",
                        mime_type,
                    ))
                } else {
                    Err(warp::reject::not_found())
                }
            });

        let static_files = ui_static_files;

        let reader_api_routes = list_models
            .or(get_model)
            .or(health)
            .or(hardware)
            .or(system_stats)
            .or(query_rag)
            .or(infer_route)
            .or(get_scaling)
            .map(reply_into_response);
        let reader_api_routes = api_auth_filter(
            ApiRole::Reader,
            ApiScope::Read,
            api_auth_state_for_api.clone(),
        )
        .and(reader_api_routes)
        .map(|response: warp::reply::Response| response);

        let writer_api_routes = list_extensions
            .or(list_marketplace_extensions)
            .or(install_extension)
            .or(uninstall_extension)
            .or(set_extension_config)
            .or(get_extension_config)
            .or(launch_extension)
            .or(sync_extension)
            .map(reply_into_response);
        let writer_api_routes = api_auth_filter(
            ApiRole::Writer,
            ApiScope::Write,
            api_auth_state_for_api.clone(),
        )
        .and(writer_api_routes)
        .map(|response: warp::reply::Response| response);

        let admin_api_routes = package_kapsl
            .or(push_kapsl)
            .or(pull_kapsl)
            .or(start_model)
            .or(stop_model)
            .or(remove_model)
            .or(update_scaling)
            .or(get_role_tokens)
            .or(set_role_tokens)
            .or(get_access_status)
            .or(get_access_roles)
            .or(list_access_users)
            .or(create_access_user)
            .or(update_access_user)
            .or(list_access_keys)
            .or(create_access_key)
            .or(revoke_access_key)
            .map(reply_into_response);
        let admin_api_routes = api_auth_filter(
            ApiRole::Admin,
            ApiScope::Admin,
            api_auth_state_for_api.clone(),
        )
        .and(admin_api_routes)
        .map(|response: warp::reply::Response| response);

        let api_routes = reader_api_routes
            .or(writer_api_routes)
            .unify()
            .or(admin_api_routes)
            .unify()
            .or_else(map_api_auth_rejection);

        let login_route = login.map(reply_into_response);

        let routes = index_route
            .or(static_files)
            .or(metrics_route)
            .or(login_route)
            .or(api_routes);

        log::info!(
            "🌐 Web UI available at http://{}:{}/",
            http_bind_addr_for_api,
            metrics_port
        );
        log::info!(
            "📊 Metrics available at http://{}:{}/metrics",
            http_bind_addr_for_api,
            metrics_port
        );
        log::info!(
            "🔌 API available at http://{}:{}/api/",
            http_bind_addr_for_api,
            metrics_port
        );
        if api_auth_state_for_api.read().auth_enabled() {
            log::info!(
                "   - API auth roles: reader={}, writer={}, admin={} (Authorization: Bearer <api-key>)",
                API_READER_TOKEN_ENV,
                API_WRITER_TOKEN_ENV,
                API_ADMIN_TOKEN_ENV
            );
        }
        log::info!("   - GET /api/models - List all models");
        log::info!("   - GET /api/models/:id - Get model details");
        log::info!("   - POST /api/models/:id/remove - Remove a model");
        log::info!("   - POST /api/models/:id/infer - Tensor or base64 media inference");
        log::info!("   - GET /api/health - System health check");
        log::info!("   - GET /api/hardware - Hardware info");
        log::info!("   - GET /api/system/stats - Runtime process stats (RSS/GPU util)");
        log::info!(
            "   - POST /api/auth/login - Validate token and return effective access (public)"
        );
        log::info!("   - POST /api/engine/package - Create a .aimod package");
        log::info!(
            "   - POST /api/engine/push - Push .aimod to remote backend (default: {})",
            DEFAULT_REMOTE_URL
        );
        log::info!("   - POST /api/engine/pull - Pull .aimod from remote backend");
        log::info!("   - GET /api/extensions - List extensions");
        log::info!(
            "   - GET /api/extensions/marketplace?q=... - Search marketplace extensions (default: {})",
            EXTENSION_MARKETPLACE_URL
        );
        log::info!("   - POST /api/extensions/install - Install extension");
        log::info!("   - POST /api/extensions/:id/uninstall - Uninstall extension");
        log::info!("   - POST /api/extensions/:id/config - Set extension config");
        log::info!("   - GET /api/extensions/:id/config?workspace_id=... - Get extension config");
        log::info!("   - POST /api/extensions/:id/launch - Launch connector");
        log::info!("   - POST /api/extensions/:id/sync - Sync connector docs into local RAG index");
        log::info!("   - GET /api/auth/roles - Read role token config (admin)");
        log::info!("   - POST /api/auth/roles - Update role token config (admin)");
        log::info!("   - GET /api/auth/access/status - Access control summary (admin)");
        log::info!("   - GET /api/auth/access/roles - Role summaries (admin)");
        log::info!("   - GET /api/auth/access/users - List users (admin)");
        log::info!("   - POST /api/auth/access/users - Create user (admin)");
        log::info!("   - PATCH /api/auth/access/users/:id - Update user role/status (admin)");
        log::info!("   - GET /api/auth/access/keys?user_id=... - List API keys (admin)");
        log::info!("   - POST /api/auth/access/users/:id/keys - Create API key (admin)");
        log::info!("   - POST /api/auth/access/keys/:id/revoke - Revoke API key (admin)");
        log::info!("   - POST /api/rag/query - Query indexed RAG chunks\n");

        let bind_addr = (http_bind_addr_for_api, metrics_port);
        match warp::serve(routes).try_bind_ephemeral(bind_addr) {
            Ok((bound_addr, server)) => {
                let _ = http_ready_tx.send(Ok(bound_addr));
                server.await;
            }
            Err(error) => {
                let message = format!(
                    "Failed to bind HTTP API on http://{}:{}/api: {}",
                    http_bind_addr_for_api, metrics_port, error
                );
                let _ = http_ready_tx.send(Err(message.clone()));
                log::error!("{}", message);
            }
        }
    });

    // Spawn auto-scaler background task
    let auto_scaler_clone = auto_scaler.clone();
    let model_registry_for_scaler = model_registry.clone();
    let replica_pools_for_scaler = replica_pools.clone();
    let model_paths_for_scaler = model_paths.clone();
    let device_info_for_scaler = device_info.clone();
    let unique_id_counter_for_scaler = unique_id_counter.clone();
    let shared_metrics_for_scaler = shared_metrics.clone();
    let batch_size_for_scaler = args.batch_size;
    let scheduler_queue_size_for_scaler = args.scheduler_queue_size;
    let scheduler_max_micro_batch_for_scaler = args.scheduler_max_micro_batch;
    let scheduler_queue_delay_ms_for_scaler = args.scheduler_queue_delay_ms;
    let topology_for_scaler = args.topology.clone();
    let tp_degree_for_scaler = args.tp_degree;
    let onnx_tuning_profile_for_scaler = onnx_tuning_profile.clone();

    tokio::spawn(async move {
        use std::time::Duration;
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        let mut last_check = std::time::Instant::now();

        loop {
            interval.tick().await;
            let elapsed = last_check.elapsed();
            last_check = std::time::Instant::now();

            // Check each model for scaling needs
            for model_info in model_registry_for_scaler.list() {
                let base_model_id = model_info.base_model_id;

                // Only process primary models (not replicas)
                if model_info.replica_id != 0 {
                    continue;
                }

                let current_replicas =
                    model_registry_for_scaler.count_active_replicas(base_model_id) as u32;

                // Calculate pool state and update metrics.
                let (
                    total_queue_depth,
                    healthy_replicas,
                    metrics_available,
                    total_model_memory_bytes,
                ) = if let Some(pool) = replica_pools_for_scaler.read().get(&base_model_id) {
                    let (high, low) = pool.get_queue_depth();
                    let healthy = pool.get_healthy_replica_count();
                    let metrics = pool.get_metrics();

                    // Update pool metrics
                    let model_id_str = base_model_id.to_string();
                    shared_metrics_for_scaler
                        .pool_active_replicas
                        .with_label_values(&[&model_id_str])
                        .set(current_replicas as i64);
                    shared_metrics_for_scaler
                        .pool_queue_depth_high
                        .with_label_values(&[&model_id_str])
                        .set(high as i64);
                    shared_metrics_for_scaler
                        .pool_queue_depth_low
                        .with_label_values(&[&model_id_str])
                        .set(low as i64);
                    shared_metrics_for_scaler
                        .pool_healthy_replicas
                        .with_label_values(&[&model_id_str])
                        .set(healthy as i64);

                    shared_metrics_for_scaler
                        .kv_cache_bytes_used
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_bytes_used as i64);
                    shared_metrics_for_scaler
                        .kv_cache_bytes_capacity
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_bytes_capacity as i64);
                    shared_metrics_for_scaler
                        .kv_cache_blocks_total
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_blocks_total as i64);
                    shared_metrics_for_scaler
                        .kv_cache_blocks_free
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_blocks_free as i64);
                    shared_metrics_for_scaler
                        .kv_cache_sequences
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_sequences as i64);
                    shared_metrics_for_scaler
                        .kv_cache_evicted_blocks
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_evicted_blocks as i64);
                    shared_metrics_for_scaler
                        .kv_cache_evicted_sequences
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_evicted_sequences as i64);
                    shared_metrics_for_scaler
                        .kv_cache_packed_layers
                        .with_label_values(&[&model_id_str])
                        .set(metrics.kv_cache_packed_layers as i64);

                    (high + low, healthy as u32, true, metrics.memory_usage)
                } else {
                    (0, 0, false, 0)
                };

                // Check for scale-up
                let should_scale_up = auto_scaler_clone.write().should_scale_up(
                    base_model_id,
                    current_replicas,
                    healthy_replicas,
                    total_queue_depth,
                    elapsed,
                    metrics_available,
                );

                if let Some(target_replicas) = should_scale_up {
                    let onnx_tuning = onnx_tuning_profile_for_scaler.resolve(base_model_id);
                    let capped_target = cap_scale_up_target_by_memory_headroom(
                        current_replicas,
                        target_replicas,
                        total_model_memory_bytes,
                        device_info_for_scaler.total_memory,
                    );
                    if capped_target < target_replicas {
                        log::warn!(
                            "Auto-scaler: Capping model {} scale-up target {} -> {} due to memory headroom",
                            base_model_id,
                            target_replicas,
                            capped_target
                        );
                    }

                    if capped_target <= current_replicas {
                        continue;
                    }

                    let replicas_to_add = capped_target.saturating_sub(current_replicas);
                    log::info!(
                        "Auto-scaler: Model {} queue depth {} exceeds threshold, scaling from {} to {} replicas",
                        base_model_id, total_queue_depth, current_replicas, capped_target
                    );

                    for _ in 0..replicas_to_add {
                        let model_path =
                            if let Some(path) = model_paths_for_scaler.read().get(&base_model_id) {
                                path.clone()
                            } else {
                                continue;
                            };

                        // Get existing replica IDs to avoid collision
                        let replicas = model_registry_for_scaler.list_replicas(base_model_id);
                        let existing_replica_ids: Vec<u32> =
                            replicas.iter().map(|r| r.replica_id).collect();

                        let next_replica_id = auto_scaler_clone
                            .read()
                            .get_next_replica_id(base_model_id, &existing_replica_ids);
                        let unique_id = unique_id_counter_for_scaler.fetch_add(1, Ordering::SeqCst);

                        match scale_up_model(
                            base_model_id,
                            next_replica_id,
                            unique_id,
                            &model_path,
                            &device_info_for_scaler,
                            batch_size_for_scaler,
                            scheduler_queue_size_for_scaler,
                            scheduler_max_micro_batch_for_scaler,
                            scheduler_queue_delay_ms_for_scaler,
                            topology_for_scaler.as_str(),
                            tp_degree_for_scaler,
                            &model_registry_for_scaler,
                            &shared_metrics_for_scaler,
                            onnx_tuning.clone(),
                        )
                        .await
                        {
                            Ok(scheduler) => {
                                // Add new replica to the pool
                                // Clone the pool to avoid holding the lock across await
                                let pool =
                                    replica_pools_for_scaler.read().get(&base_model_id).cloned();
                                if let Some(pool) = pool {
                                    pool.add_replica(next_replica_id, scheduler);
                                }
                            }
                            Err(e) => {
                                log::error!("Failed to scale up model {}: {}", base_model_id, e);
                            }
                        }
                    }

                    // Do not evaluate scale-down in the same cycle after scale-up.
                    continue;
                }

                // Check for scale-down
                let should_scale_down = auto_scaler_clone.write().should_scale_down(
                    base_model_id,
                    current_replicas,
                    healthy_replicas,
                    total_queue_depth,
                    elapsed,
                    metrics_available,
                );

                if let Some(target_replicas) = should_scale_down {
                    let replicas_to_remove = current_replicas.saturating_sub(target_replicas);
                    log::info!(
                        "Auto-scaler: Model {} queue depth {} below threshold, scaling from {} to {} replicas",
                        base_model_id, total_queue_depth, current_replicas, target_replicas
                    );

                    // Remove replicas (highest replica_id first)
                    let replicas = model_registry_for_scaler.list_replicas(base_model_id);
                    let mut replica_ids: Vec<_> = replicas
                        .iter()
                        .filter(|r| r.replica_id > 0 && r.status == ModelStatus::Active)
                        .map(|r| (r.replica_id, r.id))
                        .collect();
                    replica_ids.sort_by(|a, b| b.0.cmp(&a.0)); // Sort descending

                    for (replica_id, unique_id) in
                        replica_ids.iter().take(replicas_to_remove as usize)
                    {
                        if let Err(e) = scale_down_model(
                            base_model_id,
                            *replica_id,
                            *unique_id,
                            &model_registry_for_scaler,
                            &replica_pools_for_scaler,
                        )
                        .await
                        {
                            log::error!("Failed to scale down model {}: {}", base_model_id, e);
                        }
                    }
                }
            }
        }
    });

    let http_bound_addr = match tokio::time::timeout(Duration::from_secs(10), http_ready_rx).await {
        Ok(Ok(Ok(addr))) => addr,
        Ok(Ok(Err(message))) => return Err(message.into()),
        Ok(Err(_)) => {
            return Err("HTTP server task exited before reporting readiness".into());
        }
        Err(_) => {
            return Err(format!(
                "Timed out waiting for HTTP server to start on {}:{}",
                http_bind_addr, metrics_port
            )
            .into());
        }
    };

    let startup_elapsed_ms = startup_started_at.elapsed().as_millis();
    println!("✓ Runtime started in {}ms", startup_elapsed_ms);
    println!("→ Serving on {}", serving_endpoint);
    println!(
        "→ HTTP API on http://{}:{}/api",
        http_bound_addr.ip(),
        http_bound_addr.port()
    );
    println!(
        "→ Metrics on http://{}:{}/metrics",
        http_bound_addr.ip(),
        http_bound_addr.port()
    );

    server
        .run()
        .await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

    Ok(())
}
