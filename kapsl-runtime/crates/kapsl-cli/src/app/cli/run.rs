//! Options for starting an inference runtime.

use std::path::PathBuf;

use clap::ValueEnum;

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Run Options")]
pub(crate) struct Args {
    /// Model package or offline bundle to load. Positional paths may be mixed
    /// with the backward-compatible --model option.
    #[arg(value_name = "MODEL_OR_BUNDLE")]
    pub(crate) input: Vec<PathBuf>,

    /// Path to one or more .aimod model packages to load at startup (repeatable)
    #[arg(short, long)]
    pub(crate) model: Vec<PathBuf>,

    /// Disable network access. Required backend packs must already be cached or
    /// be carried by a .kapsl-bundle.
    #[arg(long)]
    pub(crate) offline: bool,

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

    /// Unix socket for versioned external KV participants. Managed vLLM
    /// configures a private socket automatically.
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
    /// Managed vLLM adds its certified profile automatically.
    /// Format: adapter_id,adapter_version,backend_version,profile_id.
    #[arg(long, value_name = "PROFILE", action = clap::ArgAction::Append)]
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
