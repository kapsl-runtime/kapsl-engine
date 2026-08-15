// Hosted service defaults.
pub(crate) const DEFAULT_REMOTE_URL: &str = "https://api.kapsl.net/v1";
pub(crate) const EXTENSION_MARKETPLACE_URL: &str =
    "https://api.kapsl.net/api/v1/extensions/marketplace";

// HTTP and native transport authentication.
pub(crate) const API_READER_TOKEN_ENV: &str = "KAPSL_API_TOKEN_READER";
pub(crate) const API_WRITER_TOKEN_ENV: &str = "KAPSL_API_TOKEN_WRITER";
pub(crate) const API_ADMIN_TOKEN_ENV: &str = "KAPSL_API_TOKEN_ADMIN";
pub(crate) const AUTH_STORE_PATH_ENV: &str = "KAPSL_AUTH_STORE_PATH";
pub(crate) const DEFAULT_AUTH_STORE_FILENAME: &str = "auth-store.json";
pub(crate) const LOG_SENSITIVE_IDS_ENV: &str = "KAPSL_LOG_SENSITIVE_IDS";
pub(crate) const ALLOW_INSECURE_HTTP_ENV: &str = "KAPSL_ALLOW_INSECURE_HTTP";
pub(crate) const TCP_AUTH_TOKEN_ENV: &str = "KAPSL_TCP_AUTH_TOKEN";

// Runtime storage and external services.
pub(crate) const RAG_STORAGE_ROOT_ENV: &str = "KAPSL_RAG_STORAGE_ROOT";
pub(crate) const REMOTE_URL_ENV: &str = "KAPSL_REMOTE_URL";
pub(crate) const REMOTE_TOKEN_ENV: &str = "KAPSL_REMOTE_TOKEN";
pub(crate) const REMOTE_TOKEN_STORE_PATH_ENV: &str = "KAPSL_REMOTE_TOKEN_STORE_PATH";
pub(crate) const EXTENSION_MARKETPLACE_URL_ENV: &str = "KAPSL_EXTENSION_MARKETPLACE_URL";
pub(crate) const EXTENSIONS_ROOT_ENV: &str = "KAPSL_EXTENSIONS_ROOT";
pub(crate) const EXT_CONFIG_ROOT_ENV: &str = "KAPSL_EXT_CONFIG_ROOT";

// Inference backend tuning.
pub(crate) const LLM_ISOLATE_PROCESS_ENV: &str = "KAPSL_LLM_ISOLATE_PROCESS";
pub(crate) const LLM_ISOLATE_PROCESS_STRICT_ENV: &str = "KAPSL_LLM_ISOLATE_PROCESS_STRICT";
pub(crate) const LLM_ALLOW_SCHEDULER_MICROBATCH_ENV: &str = "KAPSL_LLM_ALLOW_SCHEDULER_MICROBATCH";
pub(crate) const GGUF_MAX_CONCURRENT_ENV: &str = "KAPSL_GGUF_MAX_CONCURRENT";
pub(crate) const GGUF_TARGET_CONCURRENCY_ENV: &str = "KAPSL_GGUF_TARGET_CONCURRENCY";
pub(crate) const GGUF_PREFILL_CHUNK_SIZE_ENV: &str = "KAPSL_GGUF_PREFILL_CHUNK_SIZE";
pub(crate) const ORT_MEMORY_PATTERN_ENV: &str = "KAPSL_ORT_MEMORY_PATTERN";
pub(crate) const ORT_DISABLE_CPU_MEM_ARENA_ENV: &str = "KAPSL_ORT_DISABLE_CPU_MEM_ARENA";
pub(crate) const ORT_SESSION_BUCKETS_ENV: &str = "KAPSL_ORT_SESSION_BUCKETS";
pub(crate) const ORT_BUCKET_DIM_GRANULARITY_ENV: &str = "KAPSL_ORT_BUCKET_DIM_GRANULARITY";
pub(crate) const ORT_BUCKET_MAX_DIMS_ENV: &str = "KAPSL_ORT_BUCKET_MAX_DIMS";
pub(crate) const MODEL_PEAK_CONCURRENCY_ENV: &str = "KAPSL_MODEL_PEAK_CONCURRENCY";
pub(crate) const MODEL_PRIORITY_WEIGHTS_ENV: &str = "KAPSL_MODEL_PRIORITY_WEIGHTS";
pub(crate) const MODEL_LOAD_PARALLELISM_ENV: &str = "KAPSL_MODEL_LOAD_PARALLELISM";
pub(crate) const PROVIDER_POLICY_ENV: &str = "KAPSL_PROVIDER_POLICY";

// Scheduling and runtime pressure policy.
pub(crate) const SCHEDULER_QUEUE_OVERFLOW_POLICY_ENV: &str =
    "KAPSL_SCHEDULER_QUEUE_OVERFLOW_POLICY";
pub(crate) const PRESSURE_MEMORY_CONSERVE_PCT_ENV: &str =
    "KAPSL_SERVER_PRESSURE_MEMORY_CONSERVE_PCT";
pub(crate) const PRESSURE_MEMORY_EMERGENCY_PCT_ENV: &str =
    "KAPSL_SERVER_PRESSURE_MEMORY_EMERGENCY_PCT";
pub(crate) const PRESSURE_GPU_UTIL_CONSERVE_PCT_ENV: &str =
    "KAPSL_SERVER_PRESSURE_GPU_UTIL_CONSERVE_PCT";
pub(crate) const PRESSURE_GPU_UTIL_EMERGENCY_PCT_ENV: &str =
    "KAPSL_SERVER_PRESSURE_GPU_UTIL_EMERGENCY_PCT";
pub(crate) const PRESSURE_GPU_MEM_CONSERVE_PCT_ENV: &str =
    "KAPSL_SERVER_PRESSURE_GPU_MEM_CONSERVE_PCT";
pub(crate) const PRESSURE_GPU_MEM_EMERGENCY_PCT_ENV: &str =
    "KAPSL_SERVER_PRESSURE_GPU_MEM_EMERGENCY_PCT";
pub(crate) const PRESSURE_CONSERVE_MAX_TOKENS_ENV: &str =
    "KAPSL_SERVER_PRESSURE_CONSERVE_MAX_NEW_TOKENS";
pub(crate) const PRESSURE_EMERGENCY_MAX_TOKENS_ENV: &str =
    "KAPSL_SERVER_PRESSURE_EMERGENCY_MAX_NEW_TOKENS";
/// Opt-in co-tenancy guard: when truthy (`1`/`true`/`on`), the monitor loop
/// probes for foreign GPU processes (e.g. a training job on the same card),
/// shrinks the live KV ceiling by their footprint so batching backs off instead
/// of OOMing the neighbor, and excludes their bytes from the runtime-pressure
/// ratio so co-tenant load never truncates request outputs. Default off:
/// single-tenant behavior is byte-for-byte unchanged.
pub(crate) const COTENANCY_GUARD_ENV: &str = "KAPSL_COTENANCY_GUARD";
/// HAMi's own per-process VRAM cap (software vGPU). A HAMi-managed pod sets this
/// — or the per-device `CUDA_DEVICE_MEMORY_LIMIT_<id>` variant — so the engine
/// self-limits its KV cache and reported total to the slice with zero extra
/// config. Value is a byte count or a binary-suffixed size (e.g. `8g`, `2560m`).
pub(crate) const CUDA_DEVICE_MEMORY_LIMIT_ENV: &str = "CUDA_DEVICE_MEMORY_LIMIT";
/// kapsl alias for the per-device VRAM cap, in plain MiB, for non-HAMi
/// deployments that still want cooperative self-limiting.
pub(crate) const KAPSL_GPU_MEMORY_LIMIT_MB_ENV: &str = "KAPSL_GPU_MEMORY_LIMIT_MB";
/// Cooperative process-wide system-memory ceiling, in MiB. The host-memory
/// budget also observes container limits and keeps a safety reserve; this
/// override is useful on bare-metal hosts where no cgroup boundary exists.
pub(crate) const KAPSL_CPU_MEMORY_LIMIT_MB_ENV: &str = "KAPSL_CPU_MEMORY_LIMIT_MB";
/// Comma-separated hard limits for non-CPU, non-CUDA provider domains.
/// Entries use `provider[:device]=size`, for example
/// `metal=8g,directml:0=6g`. An exact device entry overrides the provider-wide
/// fallback. CUDA/TensorRT and CPU retain their dedicated limit variables.
pub(crate) const PROVIDER_MEMORY_LIMITS_ENV: &str = "KAPSL_PROVIDER_MEMORY_LIMITS";
/// Internal worker override used to keep process-local CUDA arenas from each
/// claiming the parent process's full configured pool.
pub(crate) const GPU_DEVICE_POOL_DISABLED_ENV: &str = "KAPSL_GPU_DEVICE_POOL_DISABLED";
/// Selects physical CUDA pool policy: `auto`, `fixed`, or `off`. A per-device
/// `_<device_id>` suffix takes precedence. When omitted, an explicit
/// `KAPSL_GPU_DEVICE_POOL_BYTES` means `fixed`; the shared-KV CUDA application
/// profile otherwise defaults to `auto`.
#[cfg(feature = "gpu-device-pool")]
pub(crate) const GPU_DEVICE_POOL_MODE_ENV: &str = "KAPSL_GPU_DEVICE_POOL_MODE";
/// Explicit operator attestation that each isolated worker receives an
/// exclusive GPU/MIG slice. A configured per-process VRAM cap is also accepted
/// as an isolation boundary.
pub(crate) const ISOLATED_WORKER_GPU_POOL_ENV: &str = "KAPSL_ISOLATED_WORKER_GPU_POOL";
/// Enables the runtime-owned elastic CUDA pool and gives its exact backing
/// allocation size. The full backing capacity is charged against the same safe
/// device budget as backend-owned external weights, so this must leave room for
/// any GGUF/native weights that remain outside the pool. A per-device
/// `_<device_id>` suffix takes precedence.
#[cfg(feature = "gpu-device-pool")]
pub(crate) const GPU_DEVICE_POOL_BYTES_ENV: &str = "KAPSL_GPU_DEVICE_POOL_BYTES";
/// Bytes kept outside an automatically-sized backing pool for backend scratch,
/// native-KV fallback, and later model additions. The default is a bounded
/// fraction of the safe device budget.
#[cfg(feature = "gpu-device-pool")]
pub(crate) const GPU_DEVICE_POOL_UNPOOLED_RESERVE_BYTES_ENV: &str =
    "KAPSL_GPU_DEVICE_POOL_UNPOOLED_RESERVE_BYTES";
#[cfg(feature = "gpu-device-pool")]
pub(crate) const GPU_ONNX_GUARANTEED_BYTES_ENV: &str = "KAPSL_GPU_ONNX_GUARANTEED_BYTES";
#[cfg(feature = "gpu-device-pool")]
pub(crate) const GPU_ONNX_MAX_BYTES_ENV: &str = "KAPSL_GPU_ONNX_MAX_BYTES";
#[cfg(feature = "gpu-device-pool")]
pub(crate) const GPU_GGUF_GUARANTEED_BYTES_ENV: &str = "KAPSL_GPU_GGUF_GUARANTEED_BYTES";
#[cfg(feature = "gpu-device-pool")]
pub(crate) const GPU_GGUF_MAX_BYTES_ENV: &str = "KAPSL_GPU_GGUF_MAX_BYTES";
#[cfg(feature = "gpu-device-pool")]
pub(crate) const GPU_NATIVE_GUARANTEED_BYTES_ENV: &str = "KAPSL_GPU_NATIVE_GUARANTEED_BYTES";
#[cfg(feature = "gpu-device-pool")]
pub(crate) const GPU_NATIVE_MAX_BYTES_ENV: &str = "KAPSL_GPU_NATIVE_MAX_BYTES";
