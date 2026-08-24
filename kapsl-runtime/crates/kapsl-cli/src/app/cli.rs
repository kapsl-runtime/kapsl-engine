use super::*;

#[derive(Parser, Debug)]
#[command(
    name = "kapsl",
    author,
    version,
    about = "Run, package, and distribute AI models",
    long_about = "Kapsl is a high-performance AI inference runtime and packaging tool.\n\
                  \n\
                  Use `kapsl run` to serve one or more model packages, `kapsl build` to\n\
                  create a portable .aimod package from an ONNX or GGUF model, and\n\
                  `kapsl push`/`pull` to sync packages with a remote registry.",
    after_help = cli_after_help(),
    styles(kapsl_help_styles()),
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Option<KapslCommand>,

    #[command(flatten)]
    pub(crate) run: Args,
}

#[derive(Subcommand, Debug)]
pub(crate) enum KapslCommand {
    /// Start the inference server and load one or more model packages
    Run(Args),
    /// Package a model file or directory into a portable .aimod archive
    Build(BuildCommandArgs),
    /// Upload a .aimod package to a remote registry
    Push(PushCommandArgs),
    /// Download a .aimod package from a remote registry
    Pull(PullCommandArgs),
    /// Log in to a remote registry and save credentials locally
    Login(LoginCommandArgs),
    /// Manage extensions in a running Kapsl Engine
    Extension(ExtensionCommandArgs),
    /// Install optional hardware acceleration support
    Provider(ProviderCommandArgs),
    /// Hot-load a model into an already-running runtime (no restart required)
    AddModel(AddModelCommandArgs),
    /// List models loaded in a running Kapsl Engine
    List(ListCommandArgs),
    /// Stop, unload, and unregister a model from a running Kapsl Engine
    RemoveModel(RemoveModelCommandArgs),
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Run Options")]
pub(crate) struct Args {
    /// Path to one or more .aimod model packages to load at startup (repeatable)
    #[arg(short, long)]
    pub(crate) model: Vec<PathBuf>,

    /// IPC transport used between the runtime and clients.
    /// socket — Unix domain socket (lowest latency, same host only).
    /// tcp    — TCP socket (cross-host).
    /// shm    — Shared memory (highest throughput, same host only).
    /// hybrid — Unix socket control/data path with shared-memory tensor transfer.
    /// auto   — picks the best available transport automatically.
    #[arg(long, default_value = "socket")]
    pub(crate) transport: String,

    /// Unix socket path (used when --transport=socket)
    #[cfg_attr(unix, arg(short, long, default_value = "/tmp/kapsl.sock"))]
    #[cfg_attr(windows, arg(short, long, default_value = r"\\.\pipe\kapsl"))]
    pub(crate) socket: String,

    /// Unix socket for versioned external KV participants such as vLLM.
    /// Omit to disable the external KV control plane.
    #[arg(long, value_name = "PATH")]
    pub(crate) kv_control_socket: Option<PathBuf>,

    /// Maximum lifetime of an external KV lease without a participant heartbeat.
    #[arg(
        long,
        value_name = "MILLISECONDS",
        default_value_t = 30_000,
        value_parser = clap::value_parser!(u64).range(1000..)
    )]
    pub(crate) kv_control_lease_ttl_ms: u64,

    /// Exact conformance-tested shared-pool adapter profiles to allow.
    /// Format: adapter_id,adapter_version,backend_version,profile_id.
    #[arg(
        long,
        value_name = "PROFILE",
        action = clap::ArgAction::Append
    )]
    pub(crate) kv_shared_pool_profile: Vec<String>,

    /// Bind address for the TCP inference server (used when --transport=tcp).
    /// Non-loopback binds require KAPSL_TCP_AUTH_TOKEN.
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) bind: String,

    /// TCP port for the inference server
    #[arg(long, default_value_t = 9096)]
    pub(crate) port: u16,

    /// Maximum number of requests combined into a single inference batch
    #[arg(long, default_value_t = 4)]
    pub(crate) batch_size: usize,

    /// Maximum number of pending requests held in each scheduler priority queue
    #[arg(long, default_value_t = 256)]
    pub(crate) scheduler_queue_size: usize,

    /// Maximum requests combined into a throughput micro-batch before dispatch
    #[arg(long, default_value_t = 4)]
    pub(crate) scheduler_max_micro_batch: usize,

    /// How long (ms) the scheduler waits to accumulate a full micro-batch before flushing early
    #[arg(long, default_value_t = 2)]
    pub(crate) scheduler_queue_delay_ms: u64,

    /// Preset that tunes batch size, transport, and scheduler settings together.
    /// Individual flags (--batch-size etc.) override the preset when specified.
    /// auto       — chooses settings based on detected model size and hardware.
    /// standard   — conservative defaults suitable for most workloads.
    /// balanced   — moderate batching with a mix of latency and throughput.
    /// throughput — aggressive batching optimised for maximum tokens/second.
    /// latency    — batch-size 1, socket transport, minimal queue delay.
    #[arg(long, value_enum, default_value_t = PerformanceProfile::Auto)]
    pub(crate) performance_profile: PerformanceProfile,

    /// Port for the HTTP API, dashboard, and Prometheus metrics server
    #[arg(long, default_value_t = 9095)]
    pub(crate) metrics_port: u16,

    /// Bind address for the HTTP API / dashboard / metrics server.
    /// Defaults to loopback; set to 0.0.0.0 only behind a TLS reverse proxy
    /// and with KAPSL_ALLOW_INSECURE_HTTP=1.
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) http_bind: String,

    /// Root directory for persistent runtime state (RAG data, extensions, auth store).
    /// Overrides KAPSL_RAG_STORAGE_ROOT, KAPSL_EXTENSIONS_ROOT, KAPSL_EXT_CONFIG_ROOT,
    /// and KAPSL_AUTH_STORE_PATH when set.
    #[arg(long, value_name = "DIR")]
    pub(crate) state_dir: Option<PathBuf>,

    /// Multi-device parallelism topology for loaded models.
    /// data-parallel     — each device holds a full model replica (default).
    /// tensor-parallel   — model weights are split across --tp-degree devices.
    /// pipeline-parallel — model layers are distributed across devices.
    /// mixed             — combines tensor and pipeline parallelism.
    #[arg(long, default_value = "data-parallel")]
    pub(crate) topology: String,

    /// Number of devices per tensor-parallel group (used when --topology=tensor-parallel or mixed)
    #[arg(long, default_value_t = 1)]
    pub(crate) tp_degree: usize,

    /// Run as isolated worker process (internal)
    #[arg(long, hide = true)]
    pub(crate) worker: bool,

    /// Model ID for isolated worker process (internal)
    #[arg(long, hide = true)]
    pub(crate) worker_model_id: Option<u32>,

    /// Enable ONNX Runtime memory-pattern optimisation for all ONNX models.
    /// Pre-allocates fixed-shape output buffers to reduce per-call overhead.
    /// Disable if your models have dynamic output shapes.
    #[arg(long, value_name = "BOOL")]
    pub(crate) onnx_memory_pattern: Option<bool>,

    /// Disable the ONNX Runtime CPU memory arena for all ONNX models.
    /// The arena pre-allocates a large block and sub-allocates from it; disabling
    /// can reduce peak RSS at the cost of more frequent allocator calls.
    #[arg(long, value_name = "BOOL")]
    pub(crate) onnx_disable_cpu_mem_arena: Option<bool>,

    /// Number of shape buckets for session reuse across requests with varying input sizes.
    /// Higher values reduce recompilation but increase memory usage.
    #[arg(long, value_name = "N")]
    pub(crate) onnx_session_buckets: Option<usize>,

    /// Rounding granularity (in elements) applied to non-batch dimensions when bucketing.
    /// Larger values create fewer buckets with more padding.
    #[arg(long, value_name = "N")]
    pub(crate) onnx_bucket_dim_granularity: Option<usize>,

    /// Number of leading input dimensions included in the bucket key (beyond batch).
    #[arg(long, value_name = "N")]
    pub(crate) onnx_bucket_max_dims: Option<usize>,

    /// Expected peak concurrent requests for this model, exported in metadata.
    /// Used by clients to size their own thread pools.
    #[arg(long, value_name = "N")]
    pub(crate) onnx_peak_concurrency_hint: Option<u32>,

    /// Shared-memory pool size in MiB for shm/hybrid/auto transports (env: KAPSL_SHM_SIZE_MB)
    #[arg(long, value_name = "MIB", default_value = "256")]
    pub(crate) shm_size_mb: Option<usize>,

    /// Per-model ONNX tuning overrides. Repeat the flag for each model.
    /// Format: `<model_id|*>:key=value[,key=value...]`
    /// Use `*` to apply to all ONNX models.
    /// Keys: memory_pattern, disable_cpu_mem_arena, session_buckets,
    ///       bucket_dim_granularity, bucket_max_dims, peak_concurrency
    /// Example: --onnx-model-tuning 1:memory_pattern=false,session_buckets=8
    #[arg(long, value_name = "SPEC")]
    pub(crate) onnx_model_tuning: Vec<String>,

    /// KV-cache compression bit-width for LLM models: 2, 3, or 4 bits.
    /// 3-bit reduces KV memory by ~2.7× with minimal quality loss.
    /// Omit or set to 0 to keep KV entries in full FP16 (no compression).
    /// Also configurable via KAPSL_LLM_KV_COMPRESSION_BITS.
    #[arg(long, value_name = "BITS")]
    pub(crate) kv_compression_bits: Option<u8>,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Build Options")]
pub(crate) struct BuildCommandArgs {
    /// Build context: a model directory (containing kapsl.yaml) or a bare model file (.onnx, .gguf).
    /// When omitted, the current directory is used as the context.
    #[arg(value_name = "CONTEXT")]
    pub(crate) context: Option<PathBuf>,

    /// Explicit path to the source model file — use when the file lives outside the context directory
    #[arg(long, value_name = "PATH")]
    pub(crate) model: Option<PathBuf>,

    /// Output path for the generated .aimod package (defaults to <project_name>.aimod in the context)
    #[arg(long, value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    /// Override the project name embedded in the package (defaults to the directory or file name)
    #[arg(long)]
    pub(crate) project_name: Option<String>,

    /// Deprecated: legacy combined framework tag (e.g. onnx, gguf, llm). Prefer
    /// --format / --model-type / --task.
    #[arg(long)]
    pub(crate) framework: Option<String>,

    /// Model file format / loader: onnx, gguf, safetensors
    #[arg(long)]
    pub(crate) format: Option<String>,

    /// Model capability class: causal-lm, embedding, seq-classifier, seq2seq, opaque
    #[arg(long = "model-type")]
    pub(crate) model_type: Option<String>,

    /// Serving operation: generate, embed, classify, rerank, forward
    #[arg(long)]
    pub(crate) task: Option<String>,

    /// Override the version string embedded in the package
    #[arg(long)]
    pub(crate) version: Option<String>,

    /// Arbitrary JSON object merged into the package manifest metadata
    #[arg(long, value_name = "JSON")]
    pub(crate) metadata_json: Option<String>,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Push Options")]
pub(crate) struct PushCommandArgs {
    /// Destination in the registry. Format: <repo>/<model>:<label>  (e.g. acme/gpt2:prod)
    #[arg(value_name = "TARGET")]
    pub(crate) target: String,

    /// Path to the .aimod package to upload (defaults to the only .aimod in the current directory)
    #[arg(value_name = "KAPSL")]
    pub(crate) kapsl: Option<PathBuf>,

    /// Explicit package path (alternative to the positional argument)
    #[arg(long, alias = "kapsl-path", value_name = "PATH")]
    pub(crate) model: Option<PathBuf>,

    /// Remote registry URL — overrides KAPSL_REMOTE_URL for this call
    #[arg(long, value_name = "URL")]
    pub(crate) remote_url: Option<String>,

    /// Bearer token for the remote registry (env: KAPSL_REMOTE_TOKEN)
    #[arg(long, value_name = "TOKEN")]
    pub(crate) remote_token: Option<String>,
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
pub(crate) struct PullCommandArgs {
    /// Package to download. Format: <repo>/<model>:<label>  (e.g. acme/gpt2:prod)
    #[arg(value_name = "TARGET")]
    pub(crate) target: Option<String>,

    /// Package to download (alternative to the positional argument)
    #[arg(long, alias = "target-ref", value_name = "TARGET")]
    pub(crate) model: Option<String>,

    /// Directory where the downloaded .aimod file will be saved (defaults to current directory)
    #[arg(long, value_name = "DIR")]
    pub(crate) destination_dir: Option<PathBuf>,

    /// Remote registry URL — overrides KAPSL_REMOTE_URL for this call
    #[arg(long, value_name = "URL")]
    pub(crate) remote_url: Option<String>,

    /// Bearer token for the remote registry (env: KAPSL_REMOTE_TOKEN)
    #[arg(long, value_name = "TOKEN")]
    pub(crate) remote_token: Option<String>,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Login Options")]
pub(crate) struct LoginCommandArgs {
    /// Backend base URL (defaults to KAPSL_REMOTE_URL or https://api.kapsl.net/v1)
    #[arg(long, value_name = "URL")]
    pub(crate) remote_url: Option<String>,

    /// OAuth provider to use
    #[arg(long, value_enum, default_value_t = OAuthProvider::GitHub)]
    pub(crate) provider: OAuthProvider,

    /// Local callback host for browser redirect
    #[arg(long, value_name = "HOST", default_value = "127.0.0.1")]
    pub(crate) callback_host: String,

    /// Local callback port (0 picks an ephemeral free port)
    #[arg(long, value_name = "PORT", default_value_t = 0)]
    pub(crate) callback_port: u16,

    /// Max time to wait for browser login callback
    #[arg(long, value_name = "SECONDS", default_value_t = 180)]
    pub(crate) timeout_seconds: u64,

    /// Print login URL instead of opening a browser automatically
    #[arg(long, default_value_t = false)]
    pub(crate) no_browser: bool,

    /// Use OAuth Device Code flow for headless/SSH environments (GitHub provider only)
    #[arg(
        long = "device-code",
        visible_alias = "headless",
        default_value_t = false
    )]
    pub(crate) device_code: bool,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Add-Model Options")]
pub(crate) struct AddModelCommandArgs {
    /// Path to a .aimod package to load. Repeat to add multiple models in one call.
    #[arg(short, long, required = true, value_name = "PATH")]
    pub(crate) model: Vec<PathBuf>,

    /// HTTP port of the running runtime's API server
    #[arg(long, default_value_t = 9095, value_name = "PORT")]
    pub(crate) http_port: u16,

    /// Hostname or IP of the running runtime's API server
    #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
    pub(crate) http_host: String,

    /// Full base URL of the running runtime (overrides --http-host / --http-port)
    #[arg(long, value_name = "URL")]
    pub(crate) http_url: Option<String>,

    /// Bearer token if the runtime has API authentication enabled
    #[arg(long, value_name = "TOKEN")]
    pub(crate) auth_token: Option<String>,

    /// Parallelism topology for the new model(s) — same values as `kapsl run --topology`
    #[arg(long, default_value = "data-parallel", value_name = "TOPOLOGY")]
    pub(crate) topology: String,

    /// Tensor-parallel device count for the new model(s)
    #[arg(long, default_value_t = 1, value_name = "N")]
    pub(crate) tp_degree: usize,

    /// HTTP request timeout (ms) for the load call — large models may take longer to respond
    #[arg(long, default_value_t = 30000, value_name = "MS")]
    pub(crate) timeout_ms: u64,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "List Options")]
pub(crate) struct ListCommandArgs {
    /// HTTP port of the running runtime's API server
    #[arg(long, default_value_t = 9095, value_name = "PORT")]
    pub(crate) http_port: u16,

    /// Hostname or IP of the running runtime's API server
    #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
    pub(crate) http_host: String,

    /// Full base URL of the running runtime (overrides --http-host / --http-port)
    #[arg(long, value_name = "URL")]
    pub(crate) http_url: Option<String>,

    /// Bearer token if the runtime has API authentication enabled
    #[arg(long, value_name = "TOKEN")]
    pub(crate) auth_token: Option<String>,

    /// Print the complete API response as JSON instead of a table
    #[arg(long)]
    pub(crate) json: bool,

    /// HTTP request timeout (ms)
    #[arg(long, default_value_t = 30000, value_name = "MS")]
    pub(crate) timeout_ms: u64,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Remove-Model Options")]
pub(crate) struct RemoveModelCommandArgs {
    /// ID of the model to remove (shown by `kapsl list`)
    #[arg(value_name = "MODEL_ID")]
    pub(crate) model_id: u32,

    /// HTTP port of the running runtime's API server
    #[arg(long, default_value_t = 9095, value_name = "PORT")]
    pub(crate) http_port: u16,

    /// Hostname or IP of the running runtime's API server
    #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
    pub(crate) http_host: String,

    /// Full base URL of the running runtime (overrides --http-host / --http-port)
    #[arg(long, value_name = "URL")]
    pub(crate) http_url: Option<String>,

    /// Admin bearer token if the runtime has API authentication enabled
    #[arg(long, value_name = "TOKEN")]
    pub(crate) auth_token: Option<String>,

    /// HTTP request timeout (ms)
    #[arg(long, default_value_t = 30000, value_name = "MS")]
    pub(crate) timeout_ms: u64,
}

#[derive(clap::Args, Debug)]
pub(crate) struct ExtensionCommandArgs {
    #[command(subcommand)]
    pub(crate) command: ExtensionSubcommand,
}

#[derive(Subcommand, Debug)]
pub(crate) enum ExtensionSubcommand {
    /// Download and install an extension from Kapsl Hub
    Install(ExtensionInstallCommandArgs),
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Extension Install Options")]
pub(crate) struct ExtensionInstallCommandArgs {
    /// Marketplace extension ID (for example connector.s3)
    #[arg(value_name = "EXTENSION_ID")]
    pub(crate) extension_id: String,

    /// HTTP port of the running runtime's API server
    #[arg(long, default_value_t = 9095, value_name = "PORT")]
    pub(crate) http_port: u16,

    /// Hostname or IP of the running runtime's API server
    #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
    pub(crate) http_host: String,

    /// Full base URL of the running runtime (overrides --http-host / --http-port)
    #[arg(long, value_name = "URL")]
    pub(crate) http_url: Option<String>,

    /// Bearer token if the runtime has API authentication enabled
    #[arg(long, value_name = "TOKEN")]
    pub(crate) auth_token: Option<String>,

    /// Override the Kapsl Hub marketplace endpoint
    #[arg(long, value_name = "URL")]
    pub(crate) marketplace_url: Option<String>,

    /// HTTP request timeout (ms)
    #[arg(long, default_value_t = 30000, value_name = "MS")]
    pub(crate) timeout_ms: u64,
}

#[derive(clap::Args, Debug)]
pub(crate) struct ProviderCommandArgs {
    #[command(subcommand)]
    pub(crate) command: ProviderSubcommand,
}

#[derive(Subcommand, Debug)]
pub(crate) enum ProviderSubcommand {
    /// Download and install a provider pack matching this Kapsl release
    Install(ProviderInstallCommandArgs),
}

#[derive(clap::Args, Debug)]
pub(crate) struct ProviderInstallCommandArgs {
    /// Accelerator runtime to install
    #[arg(value_enum)]
    pub(crate) provider: ProviderPackage,

    /// Reinstall the pack even when a complete matching pack is present
    #[arg(long)]
    pub(crate) force: bool,

    /// Override the destination directory containing kapsl.exe
    #[arg(long, value_name = "DIR", hide = true)]
    pub(crate) install_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ProviderPackage {
    #[value(name = "cuda12", alias = "cuda")]
    Cuda12,
    #[value(name = "tensorrt10", alias = "tensorrt")]
    TensorRt10,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum OAuthProvider {
    #[value(name = "github", alias = "git-hub")]
    GitHub,
    #[value(name = "google")]
    Google,
}

impl OAuthProvider {
    pub(crate) fn route_segment(self) -> &'static str {
        match self {
            Self::GitHub => "github",
            Self::Google => "google",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum PerformanceProfile {
    /// Detect model size and hardware at startup and choose the best preset automatically
    Auto,
    /// Conservative defaults — a safe starting point for unknown workloads
    Standard,
    /// Moderate batching with a balance between throughput and response latency
    Balanced,
    /// Aggressive batching and larger queues optimised for maximum tokens/second
    Throughput,
    /// Batch size 1, socket transport, zero queue delay — minimises time-to-first-token
    Latency,
}

impl PerformanceProfile {
    pub(crate) fn as_str(self) -> &'static str {
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
pub(crate) struct AppliedPerformanceTuning {
    pub(crate) batch_size: Option<usize>,
    pub(crate) transport: Option<String>,
    pub(crate) scheduler_queue_size: Option<usize>,
    pub(crate) scheduler_max_micro_batch: Option<usize>,
    pub(crate) scheduler_queue_delay_ms: Option<u64>,
    pub(crate) media_preprocess: Option<String>,
    pub(crate) rust_log: Option<String>,
    /// Populated when Auto profile is used; emitted after env_logger::init().
    pub(crate) auto_tune_rationale: Option<String>,
}
