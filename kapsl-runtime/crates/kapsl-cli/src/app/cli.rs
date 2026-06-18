use super::*;

pub(crate) fn kapsl_help_styles() -> clap::builder::Styles {
    use clap::builder::styling::{AnsiColor, Effects};
    clap::builder::Styles::styled()
        .header(AnsiColor::Cyan.on_default() | Effects::BOLD)
        .usage(AnsiColor::Cyan.on_default() | Effects::BOLD)
        .literal(AnsiColor::BrightCyan.on_default() | Effects::BOLD)
        .placeholder(AnsiColor::Cyan.on_default())
        .error(AnsiColor::Red.on_default() | Effects::BOLD)
        .invalid(AnsiColor::Yellow.on_default() | Effects::BOLD)
        .valid(AnsiColor::Green.on_default())
}

pub(crate) fn cli_after_help() -> String {
    use std::fmt::Write as _;
    let a = Ansi::new();
    let header = |s: &str| a.bold(&a.teal(s)).into_owned();
    let cmd = |s: &str| a.bold(s).into_owned();
    let comment = |s: &str| a.dim(s).into_owned();

    let mut out = String::new();
    let _ = writeln!(out, "{}", header("Examples:"));
    let _ = writeln!(out, "  {}", comment("# Start the runtime with one model"));
    let _ = writeln!(out, "  {}", cmd("kapsl run --model models/gpt2/gpt2.aimod"));
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}",
        comment("# Load an extra model into an already-running runtime (no restart)")
    );
    let _ = writeln!(
        out,
        "  {}",
        cmd("kapsl add-model --model models/llama/llama.aimod")
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}",
        comment("# Multi-GPU load-balancing across two runtime instances")
    );
    let _ = writeln!(
        out,
        "  {}",
        cmd("kapsl control --runtime gpu0=http://127.0.0.1:9095 --runtime gpu1=http://127.0.0.1:9096")
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}",
        comment("# Package a model directory or single file")
    );
    let _ = writeln!(out, "  {}", cmd("kapsl build ./models/gpt-llm"));
    let _ = writeln!(
        out,
        "  {}",
        cmd("kapsl build ./model.onnx --output ./model.aimod")
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}",
        comment("# Push / pull packages to/from a remote registry")
    );
    let _ = writeln!(out, "  {}", cmd("kapsl push acme/gpt2:prod ./model.aimod"));
    let _ = writeln!(
        out,
        "  {}",
        cmd("kapsl push acme/gpt2:prod ./model.aimod --remote-url oci://ghcr.io")
    );
    let _ = writeln!(
        out,
        "  {}",
        cmd("kapsl pull acme/gpt2:prod --destination-dir ./models")
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}",
        comment("# Authenticate (opens browser; use --device-code for SSH/headless)")
    );
    let _ = writeln!(out, "  {}", cmd("kapsl login"));
    let _ = writeln!(out, "  {}", cmd("kapsl login --device-code"));
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", header("Environment variables:"));
    for (name, desc) in [
        (
            "KAPSL_API_TOKEN",
            "Shared fallback bearer token for /api routes",
        ),
        ("KAPSL_API_TOKEN_READER", "Read-only API token"),
        ("KAPSL_API_TOKEN_WRITER", "Writer API token"),
        ("KAPSL_API_TOKEN_ADMIN", "Admin API token"),
        ("KAPSL_REMOTE_URL", "Default remote registry URL"),
        ("KAPSL_REMOTE_TOKEN", "Bearer token for push/pull"),
        (
            "KAPSL_SHM_SIZE_MB",
            "Shared-memory pool size (MiB) for shm/hybrid transport",
        ),
    ] {
        let padded = format!("{:<26}", name);
        let _ = writeln!(out, "  {}{}", a.teal(&padded), a.dim(desc));
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", header("Compatibility:"));
    let _ = writeln!(out, "  {}", cmd("kapsl --model models/gpt2/gpt2.aimod"));
    let _ = write!(
        out,
        "    {}",
        comment("(equivalent to `kapsl run --model models/gpt2/gpt2.aimod`)")
    );
    out
}

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
    /// Continuously balance load and scale replicas across multiple runtimes
    Control(ControlCommandArgs),
    /// Package a model file or directory into a portable .aimod archive
    Build(BuildCommandArgs),
    /// Upload a .aimod package to a remote registry
    Push(PushCommandArgs),
    /// Download a .aimod package from a remote registry
    Pull(PullCommandArgs),
    /// Log in to a remote registry and save credentials locally
    Login(LoginCommandArgs),
    /// Hot-load a model into an already-running runtime (no restart required)
    AddModel(AddModelCommandArgs),
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
    /// hybrid — shm for local clients, tcp for remote.
    /// auto   — picks the best available transport automatically.
    #[arg(long, default_value = "socket")]
    pub(crate) transport: String,

    /// Unix socket path (used when --transport=socket)
    #[cfg_attr(unix, arg(short, long, default_value = "/tmp/kapsl.sock"))]
    #[cfg_attr(windows, arg(short, long, default_value = r"\\.\pipe\kapsl"))]
    pub(crate) socket: String,

    /// Bind address for the TCP inference server (used when --transport=tcp|hybrid|auto)
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

#[derive(Debug, Clone, Default)]
pub(crate) struct OnnxTuningProfile {
    pub(crate) global: OnnxRuntimeTuning,
    pub(crate) per_model: HashMap<u32, OnnxRuntimeTuning>,
}

pub(crate) fn merge_onnx_runtime_tuning(
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
    pub(crate) fn resolve(&self, model_id: u32) -> OnnxRuntimeTuning {
        if let Some(model_overrides) = self.per_model.get(&model_id) {
            merge_onnx_runtime_tuning(&self.global, model_overrides)
        } else {
            self.global.clone()
        }
    }
}

pub(crate) fn parse_bool_literal(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid boolean '{}'", value)),
    }
}

pub(crate) fn apply_onnx_tuning_pair(
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

pub(crate) fn parse_env_bool_override(name: &str) -> Result<Option<bool>, String> {
    optional_env_var(name)
        .map(|value| parse_bool_literal(&value))
        .transpose()
}

pub(crate) fn parse_env_usize_override(name: &str) -> Result<Option<usize>, String> {
    optional_env_var(name)
        .map(|value| {
            value
                .parse::<usize>()
                .map(|parsed| parsed.max(1))
                .map_err(|e| format!("invalid {} '{}': {}", name, value, e))
        })
        .transpose()
}

pub(crate) fn parse_env_u32_override(name: &str) -> Result<Option<u32>, String> {
    optional_env_var(name)
        .map(|value| {
            value
                .parse::<u32>()
                .map(|parsed| parsed.max(1))
                .map_err(|e| format!("invalid {} '{}': {}", name, value, e))
        })
        .transpose()
}

pub(crate) fn auto_onnx_runtime_tuning(args: &Args) -> OnnxRuntimeTuning {
    let batch_size = args.batch_size.max(1);
    let session_pool = batch_size.min(logical_cpu_cores().max(1)).clamp(1, 4);
    let session_buckets = batch_size.max(4).min(8);
    OnnxRuntimeTuning {
        memory_pattern: Some(true),
        disable_cpu_mem_arena: Some(false),
        session_buckets: Some(session_buckets),
        bucket_dim_granularity: Some(64),
        bucket_max_dims: Some(4),
        peak_concurrency_hint: Some(session_pool as u32),
    }
}

pub(crate) fn env_onnx_runtime_tuning() -> Result<OnnxRuntimeTuning, String> {
    Ok(OnnxRuntimeTuning {
        memory_pattern: parse_env_bool_override(ORT_MEMORY_PATTERN_ENV)?,
        disable_cpu_mem_arena: parse_env_bool_override(ORT_DISABLE_CPU_MEM_ARENA_ENV)?,
        session_buckets: parse_env_usize_override(ORT_SESSION_BUCKETS_ENV)?,
        bucket_dim_granularity: parse_env_usize_override(ORT_BUCKET_DIM_GRANULARITY_ENV)?,
        bucket_max_dims: parse_env_usize_override(ORT_BUCKET_MAX_DIMS_ENV)?,
        peak_concurrency_hint: parse_env_u32_override(MODEL_PEAK_CONCURRENCY_ENV)?,
    })
}

pub(crate) fn parse_onnx_model_tuning_spec(
    spec: &str,
) -> Result<(Option<u32>, OnnxRuntimeTuning), String> {
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

pub(crate) fn build_onnx_tuning_profile(args: &Args) -> Result<OnnxTuningProfile, String> {
    let mut profile = OnnxTuningProfile {
        global: auto_onnx_runtime_tuning(args),
        per_model: HashMap::new(),
    };

    let env_tuning = env_onnx_runtime_tuning()?;
    profile.global = merge_onnx_runtime_tuning(&profile.global, &env_tuning);
    let cli_tuning = OnnxRuntimeTuning {
        memory_pattern: args.onnx_memory_pattern,
        disable_cpu_mem_arena: args.onnx_disable_cpu_mem_arena,
        session_buckets: args.onnx_session_buckets,
        bucket_dim_granularity: args.onnx_bucket_dim_granularity,
        bucket_max_dims: args.onnx_bucket_max_dims,
        peak_concurrency_hint: args.onnx_peak_concurrency_hint,
    };
    profile.global = merge_onnx_runtime_tuning(&profile.global, &cli_tuning);

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

    /// Remote registry URL — overrides KAPSL_REMOTE_URL for this call.
    /// Use an oci:// prefix to push to an OCI-compatible registry (e.g. oci://ghcr.io).
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

    /// Pin to a specific OCI content digest when using an oci:// remote.
    /// Accepts sha256:<hex> or @sha256:<hex>.
    #[arg(long = "ref", value_name = "REF")]
    pub(crate) reference: Option<String>,

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum RuntimeGroupProfile {
    Latency,
    Balanced,
    Throughput,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Control Options")]
pub(crate) struct ControlCommandArgs {
    /// Register a runtime endpoint. Repeat for each instance.
    /// Format: NAME=http://HOST:PORT (e.g. gpu0=http://127.0.0.1:9095)
    #[arg(long = "runtime", value_name = "NAME=URL", required = true)]
    pub(crate) runtimes: Vec<String>,

    /// Override the performance profile for a specific runtime.
    /// Format: NAME=latency|balanced|throughput
    #[arg(long = "runtime-profile", value_name = "NAME=PROFILE")]
    pub(crate) runtime_profiles: Vec<String>,

    /// Set a bearer token for a specific runtime.
    /// Format: NAME=TOKEN (takes precedence over --auth-token for that runtime)
    #[arg(long = "runtime-token", value_name = "NAME=TOKEN")]
    pub(crate) runtime_tokens: Vec<String>,

    /// Shared bearer token applied to all runtimes that have no --runtime-token
    #[arg(long = "auth-token", value_name = "TOKEN")]
    pub(crate) auth_token: Option<String>,

    /// Cap the VRAM budget used for scoring a specific runtime.
    /// Format: NAME=BYTES (e.g. gpu0=17179869184 for 16 GiB)
    #[arg(long = "memory-budget-bytes", value_name = "NAME=BYTES")]
    pub(crate) memory_budget_bytes: Vec<String>,

    /// How often (seconds) to poll runtime metrics and recompute weights
    #[arg(long, default_value_t = 5)]
    pub(crate) interval_seconds: u64,

    /// Per-call HTTP timeout (ms) for polling and weight-update requests
    #[arg(long, default_value_t = 1500)]
    pub(crate) timeout_ms: u64,

    /// Queue depth considered "normal" when computing the pressure score.
    /// Queues deeper than this start to penalise the runtime's weight.
    #[arg(long, default_value_t = 10)]
    pub(crate) queue_target: usize,

    /// Pressure score above which the runtime loses weight each cycle (0.0–1.0)
    #[arg(long, default_value_t = 0.85)]
    pub(crate) high_pressure_score: f64,

    /// Pressure score below which the runtime gains weight each cycle (0.0–1.0)
    #[arg(long, default_value_t = 0.45)]
    pub(crate) low_pressure_score: f64,

    /// GPU utilisation fraction (0.0–1.0) above which the runtime is considered hot
    #[arg(long, default_value_t = 0.92)]
    pub(crate) hot_gpu_utilization: f64,

    /// Memory utilisation fraction (0.0–1.0) above which the runtime is considered hot.
    /// Only active when --memory-budget-bytes is provided for the runtime.
    #[arg(long, default_value_t = 0.90)]
    pub(crate) hot_memory_utilization: f64,

    /// Seconds a runtime must remain overloaded before an extra weight penalty is applied
    #[arg(long, default_value_t = 30)]
    pub(crate) overload_window_seconds: u64,

    /// Seconds a runtime must remain hot before an extra weight penalty is applied
    #[arg(long, default_value_t = 20)]
    pub(crate) hot_window_seconds: u64,

    /// Seconds a runtime stays at weight 0 after a health-check or API failure
    #[arg(long, default_value_t = 30)]
    pub(crate) unhealthy_hold_seconds: u64,

    /// Maximum fractional weight change applied per control cycle (0.0–1.0)
    #[arg(long, default_value_t = 0.10)]
    pub(crate) weight_step: f64,

    /// Minimum weight assigned to eligible runtimes — prevents them from reaching zero under normal load
    #[arg(long, default_value_t = 0.05)]
    pub(crate) weight_floor: f64,

    /// Additional weight fraction removed when the overload or hot window is exceeded
    #[arg(long, default_value_t = 0.20)]
    pub(crate) overload_shift_fraction: f64,

    /// Compute and log scores/weights without posting any updates to the runtimes
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run: bool,

    /// JSON file where each cycle's computed weights and telemetry snapshot are written
    #[arg(
        long,
        value_name = "PATH",
        default_value = "runtime-control-weights.json"
    )]
    pub(crate) weights_file: PathBuf,
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
