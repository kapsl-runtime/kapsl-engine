pub(crate) const DEFAULT_REMOTE_URL: &str = "https://api.kapsl.net/v1";
pub(crate) const REMOTE_PLACEHOLDER_URL: &str = "https://placeholder-kapsl-registry.example.com/v1";
pub(crate) const REMOTE_PLACEHOLDER_DIR: &str = ".kapsl-remote-placeholder";
pub(crate) const EXTENSION_MARKETPLACE_URL: &str =
    "https://api.kapsl.net/api/v1/extensions/marketplace";
pub(crate) const API_TOKEN_ENV: &str = "KAPSL_API_TOKEN";
pub(crate) const API_READER_TOKEN_ENV: &str = "KAPSL_API_TOKEN_READER";
pub(crate) const API_WRITER_TOKEN_ENV: &str = "KAPSL_API_TOKEN_WRITER";
pub(crate) const API_ADMIN_TOKEN_ENV: &str = "KAPSL_API_TOKEN_ADMIN";
pub(crate) const AUTH_STORE_PATH_ENV: &str = "KAPSL_AUTH_STORE_PATH";
pub(crate) const DEFAULT_AUTH_STORE_FILENAME: &str = "auth-store.json";
pub(crate) const LOG_SENSITIVE_IDS_ENV: &str = "KAPSL_LOG_SENSITIVE_IDS";
pub(crate) const RAG_STORAGE_ROOT_ENV: &str = "KAPSL_RAG_STORAGE_ROOT";
pub(crate) const REMOTE_URL_ENV: &str = "KAPSL_REMOTE_URL";
pub(crate) const REMOTE_TOKEN_ENV: &str = "KAPSL_REMOTE_TOKEN";
pub(crate) const REMOTE_TOKEN_STORE_PATH_ENV: &str = "KAPSL_REMOTE_TOKEN_STORE_PATH";
pub(crate) const REMOTE_PLACEHOLDER_URL_ENV: &str = "KAPSL_REMOTE_PLACEHOLDER_URL";
pub(crate) const REMOTE_PLACEHOLDER_DIR_ENV: &str = "KAPSL_REMOTE_PLACEHOLDER_DIR";
pub(crate) const EXTENSION_MARKETPLACE_URL_ENV: &str = "KAPSL_EXTENSION_MARKETPLACE_URL";
pub(crate) const ALLOW_INSECURE_HTTP_ENV: &str = "KAPSL_ALLOW_INSECURE_HTTP";
pub(crate) const OCI_REMOTE_PREFIX: &str = "oci://";
pub(crate) const KAPSL_OCI_ARTIFACT_TYPE: &str = "application/vnd.kapsl.aimod.v1";
pub(crate) const KAPSL_OCI_LAYER_TYPE: &str = "application/vnd.kapsl.aimod.v1";
pub(crate) const KAPSL_OCI_CONFIG_TYPE: &str = "application/vnd.kapsl.aimod.config.v1+json";
pub(crate) const OCI_PRECOMPUTE_SHA256_ENV: &str = "KAPSL_OCI_PRECOMPUTE_SHA256";
pub(crate) const ORAS_BIN_ENV: &str = "KAPSL_ORAS_BIN";
pub(crate) const OCI_USERNAME_ENV: &str = "KAPSL_OCI_USERNAME";
pub(crate) const OCI_PASSWORD_ENV: &str = "KAPSL_OCI_PASSWORD";
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
pub(crate) const EXTENSIONS_ROOT_ENV: &str = "KAPSL_EXTENSIONS_ROOT";
pub(crate) const EXT_CONFIG_ROOT_ENV: &str = "KAPSL_EXT_CONFIG_ROOT";
pub(crate) const SCHEDULER_QUEUE_OVERFLOW_POLICY_ENV: &str =
    "KAPSL_SCHEDULER_QUEUE_OVERFLOW_POLICY";
pub(crate) const LEGACY_SCHEDULER_QUEUE_OVERFLOW_POLICY_ENV: &str =
    "KAPSL_LITE_INGRESS_BACKPRESSURE";
pub(crate) const INTER_MODEL_ROUTES_ENV: &str = "KAPSL_INTER_MODEL_ROUTES";
pub(crate) const LEGACY_INTER_MODEL_ROUTES_ENV: &str = "KAPSL_LITE_INTER_MODEL_ROUTES";
pub(crate) const INTER_MODEL_RELAY_MIN_INTERVAL_MS_ENV: &str =
    "KAPSL_INTER_MODEL_RELAY_MIN_INTERVAL_MS";
pub(crate) const LEGACY_INTER_MODEL_RELAY_MIN_INTERVAL_MS_ENV: &str =
    "KAPSL_LITE_INTER_MODEL_RELAY_MIN_INTERVAL_MS";
pub(crate) const INTER_MODEL_RELAY_SESSION_PREFIX: &str = "relay/";
pub(crate) const DEFAULT_INTER_MODEL_RELAY_MIN_INTERVAL_MS: u64 = 2000;
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
/// HAMi's own per-process VRAM cap (software vGPU). A HAMi-managed pod sets this
/// — or the per-device `CUDA_DEVICE_MEMORY_LIMIT_<id>` variant — so the engine
/// self-limits its KV cache and reported total to the slice with zero extra
/// config. Value is a byte count or a binary-suffixed size (e.g. `8g`, `2560m`).
pub(crate) const CUDA_DEVICE_MEMORY_LIMIT_ENV: &str = "CUDA_DEVICE_MEMORY_LIMIT";
/// kapsl alias for the per-device VRAM cap, in plain MiB, for non-HAMi
/// deployments that still want cooperative self-limiting.
pub(crate) const KAPSL_GPU_MEMORY_LIMIT_MB_ENV: &str = "KAPSL_GPU_MEMORY_LIMIT_MB";
