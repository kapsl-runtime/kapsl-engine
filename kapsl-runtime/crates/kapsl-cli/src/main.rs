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
use futures::{stream, StreamExt};
use infer_adapter::{default_request_adapter_registry, parse_inference_request_with_registry};
use kapsl_backends::{BackendFactory, OnnxRuntimeTuning};
use kapsl_core::loader::Manifest;
use kapsl_core::{
    AutoScaler, EngineKind, ModelInfo, ModelRegistry, ModelStatus, PackageLoader, ScalingPolicy,
};
use kapsl_engine_api::{
    BinaryTensorPacket, Engine, EngineError, EngineHandle, EngineMetrics, EngineModelInfo,
    InferenceRequest, TensorDtype,
};
#[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
use kapsl_hal::cross_device_scheduler::CrossDevicePoolScheduler;
use kapsl_hal::device::DeviceInfo;
#[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
use kapsl_hal::gpu_arena::GpuPoolHandle;
use kapsl_ipc::{
    IpcServer, RequestHeader, ResponseHeader, TcpServer, OP_INFER, OP_INFER_STREAM, STATUS_ERR,
    STATUS_OK, STATUS_STREAM_CHUNK, STATUS_STREAM_END,
};
use kapsl_llm::block_manager::{new_shared_allocator, SharedBlockAllocator};
use kapsl_llm::global_scheduler::{EngineHandle as KvEngineHandle, GlobalKvScheduler};
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
use std::io::{BufRead, BufWriter, Cursor, Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{Pid, System};
use tar::{Archive, Builder};
use tokio::sync::Mutex as AsyncMutex;
use warp::{Filter, Reply};

#[cfg(unix)]
use std::os::unix::net::UnixStream;

mod app_support;
mod auth;
mod autoscaler;
mod cli;
mod constants;
mod control;
mod extensions;
mod http_auth;
mod http_engine;
mod http_extensions;
mod http_metrics;
mod http_rag;
mod http_static;
mod http_system;
mod infer_adapter;
mod model_runtime;
mod packaging;
mod rag;
mod runtime_config;
mod runtime_monitor;
mod runtime_support;
mod runtime_tuning;
mod shared_kv;
mod worker;

use app_support::*;
use auth::*;
use autoscaler::*;
use cli::*;
use constants::*;
use control::*;
use extensions::*;
use http_auth::*;
use http_engine::*;
use http_extensions::*;
use http_metrics::*;
use http_rag::*;
use http_static::*;
use http_system::*;
use model_runtime::*;
use packaging::*;
use rag::*;
use runtime_config::*;
use runtime_monitor::*;
use runtime_support::*;
use runtime_tuning::*;
use shared_kv::*;
use worker::*;

type DynError = Box<dyn std::error::Error + Send + Sync>;
type ReplicaPools = Arc<RwLock<HashMap<u32, Arc<ReplicaPool<Scheduler>>>>>;

#[cfg(test)]
mod tests;

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
        Some(KapslCommand::AddModel(args)) => return execute_add_model_command(args),
        Some(KapslCommand::Run(_)) | None => {}
    }

    let runtime_argv = runtime_argv_from_invocation(&raw_argv);
    let (mut args, matches) = parse_runtime_args_and_matches(&runtime_argv)?;
    let applied_tuning = apply_performance_profile(&mut args, &matches);
    let onnx_tuning_profile = Arc::new(
        build_onnx_tuning_profile(&args)
            .map_err(|e| format!("Invalid ONNX tuning configuration: {}", e))?,
    );
    // Propagate --kv-compression-bits to the env var read by kapsl-llm engine.rs.
    // This lets the existing metadata/env override chain pick it up without
    // threading an extra parameter through every load_model call site.
    if let Some(bits) = args.kv_compression_bits {
        if (2..=4).contains(&bits) {
            // SAFETY: single-threaded startup path; no other threads reading env yet.
            unsafe { std::env::set_var("KAPSL_LLM_KV_COMPRESSION_BITS", bits.to_string()) };
        } else {
            eprintln!(
                "Warning: --kv-compression-bits {} is invalid (must be 2, 3, or 4); ignoring",
                bits
            );
        }
    }
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

    print_startup_banner();
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

    // Unified shared KV cache pool and cross-model token budget coordinator.
    let shared_kv = SharedKvStateInner::new(&device_info);

    // Use Arc<RwLock<>> for thread-safe dynamic scheduler management
    let replica_pools: Arc<RwLock<HashMap<u32, Arc<ReplicaPool<Scheduler>>>>> =
        Arc::new(RwLock::new(HashMap::new()));
    // Per-model engine handles for hot-swap stage/swap operations.
    let swap_map: Arc<RwLock<HashMap<u32, Vec<EngineHandle>>>> =
        Arc::new(RwLock::new(HashMap::new()));

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
    let generated_token_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let total_token_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>> =
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
    let shared_kv_for_rebalance = shared_kv.clone();
    tokio::spawn(async move {
        let pid = Pid::from_u32(std::process::id());
        let mut system = System::new();
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        let mut nvidia_smi_retry_after: Option<Instant> = None;
        system.refresh_memory();
        let total_system_memory_bytes = Some(system.total_memory() as usize * 1024);

        loop {
            interval.tick().await;

            // Reclaim KV block quota from any engine whose health changed
            // (e.g. tripped circuit breaker / stalled watchdog), redistributing
            // it to healthy engines.
            shared_kv_for_rebalance.maybe_rebalance_for_health();

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
    let load_parallelism = resolve_model_load_parallelism(args.model.len());
    if args.model.len() > 1 {
        log::info!(
            "Loading {} model packages with parallelism {} ({}=N to override)",
            args.model.len(),
            load_parallelism,
            MODEL_LOAD_PARALLELISM_ENV
        );
    }
    let load_specs = args
        .model
        .iter()
        .map(|model_path| {
            (
                allocate_model_id(&model_id_counter, &recycled_model_ids),
                model_path.clone(),
            )
        })
        .collect::<Vec<_>>();

    let load_results = run_with_loading_async("Loading model packages", {
        let device_info = device_info.clone();
        let shared_kv = shared_kv.clone();
        let model_registry = model_registry.clone();
        let shared_metrics = shared_metrics.clone();
        let topology = args.topology.clone();
        let onnx_tuning_profile = onnx_tuning_profile.clone();
        async move {
            let results = stream::iter(load_specs.into_iter().map(|(model_id, model_path)| {
                let device_info = device_info.clone();
                let shared_kv = shared_kv.clone();
                let model_registry = model_registry.clone();
                let shared_metrics = shared_metrics.clone();
                let topology = topology.clone();
                let onnx_tuning = onnx_tuning_profile.resolve(model_id);
                async move {
                    let result = load_model(
                        model_id,
                        &model_path,
                        &device_info,
                        shared_kv,
                        args.batch_size,
                        args.scheduler_queue_size,
                        args.scheduler_max_micro_batch,
                        args.scheduler_queue_delay_ms,
                        &model_registry,
                        &shared_metrics,
                        &topology,
                        args.tp_degree,
                        onnx_tuning,
                    )
                    .await;
                    (model_id, model_path, result)
                }
            }))
            .buffer_unordered(load_parallelism)
            .collect::<Vec<_>>()
            .await;
            Ok::<_, DynError>(results)
        }
    })
    .await?;

    let mut first_load_error: Option<DynError> = None;
    for (model_id, model_path, result) in load_results {
        match result {
            Ok((pool, handles)) => {
                replica_pools.write().insert(model_id, pool);
                swap_map.write().insert(model_id, handles);
                model_paths.write().insert(model_id, model_path);

                // Register default scaling policy for each model
                auto_scaler
                    .write()
                    .register_policy(model_id, ScalingPolicy::default());
            }
            Err(error) => {
                recycle_model_id(model_id, &recycled_model_ids);
                if first_load_error.is_none() {
                    first_load_error = Some(error);
                }
            }
        }
    }
    if let Some(error) = first_load_error {
        return Err(error);
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
    let generated_token_samples_clone = generated_token_samples.clone();
    let total_token_samples_clone = total_token_samples.clone();
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

    // Clone before the API server spawn so the auto-scaler task can use the same state.
    let shared_kv_for_autoscaler = shared_kv.clone();
    let swap_map_for_autoscaler = swap_map.clone();

    tokio::spawn(async move {
        let metrics_route =
            build_metrics_route(registry_arc.clone(), api_auth_state_for_api.clone());

        // API routes
        let model_registry_for_list = model_registry_clone.clone();
        let replica_pools_for_list = replica_pools_clone.clone();
        let metrics_for_list = shared_metrics_clone.clone();
        let throughput_samples_for_list = throughput_samples_clone.clone();
        let generated_token_samples_for_list = generated_token_samples_clone.clone();
        let total_token_samples_for_list = total_token_samples_clone.clone();
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
                prompt_tokens_total: u64,
                generated_tokens_total: u64,
                generated_tokens_per_sec: f64,
                total_tokens_per_sec: f64,
                decode_steps_total: u64,
                decode_tokens_evaluated_total: u64,
                avg_tokens_evaluated_per_decode_step: f64,
                kv_partial_reuse_hits_total: u64,
                kv_partial_reuse_tokens_saved_total: u64,
                onnx_session_pool_total: usize,
                onnx_session_pool_idle: usize,
                onnx_session_pool_waits_total: u64,
                onnx_session_pool_wait_seconds_total: f64,
                healthy: bool,
            }

            let models = model_registry_for_list.list();
            let mut statuses = Vec::new();
            let now = Instant::now();
            let mut seen_ids = HashSet::new();
            let mut throughput_samples = throughput_samples_for_list.write();
            let mut generated_token_samples = generated_token_samples_for_list.write();
            let mut total_token_samples = total_token_samples_for_list.write();

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

                let (
                    queue_depth,
                    healthy,
                    engine_memory,
                    engine_gpu_util,
                    prompt_tokens_total,
                    generated_tokens_total,
                    decode_steps_total,
                    decode_tokens_evaluated_total,
                    kv_partial_reuse_hits_total,
                    kv_partial_reuse_tokens_saved_total,
                    onnx_session_pool_total,
                    onnx_session_pool_idle,
                    onnx_session_pool_waits_total,
                    onnx_session_pool_wait_seconds_total,
                ) = if let Some(pool) = replica_pools_for_list.read().get(&model.id) {
                    let metrics = pool.get_metrics();
                    (
                        pool.get_queue_depth(),
                        pool.is_healthy(),
                        metrics.memory_usage,
                        metrics.gpu_utilization,
                        metrics.prompt_tokens_total,
                        metrics.generated_tokens_total,
                        metrics.decode_steps_total,
                        metrics.decode_tokens_evaluated_total,
                        metrics.kv_partial_reuse_hits_total,
                        metrics.kv_partial_reuse_tokens_saved_total,
                        metrics.onnx_session_pool_total,
                        metrics.onnx_session_pool_idle,
                        metrics.onnx_session_pool_waits_total,
                        metrics.onnx_session_pool_wait_seconds_total,
                    )
                } else {
                    ((0, 0), true, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)
                };

                // `memory_usage` and `gpu_utilization` are engine-reported metrics only.
                // System-level RSS/GPU stats are exposed separately via GET /api/system/stats.
                let memory_usage = engine_memory;
                let gpu_utilization = engine_gpu_util;
                let throughput = update_throughput(&mut throughput_samples, model.id, total, now);
                let generated_tokens_per_sec = update_throughput(
                    &mut generated_token_samples,
                    model.id,
                    generated_tokens_total,
                    now,
                );
                let total_tokens_per_sec = update_throughput(
                    &mut total_token_samples,
                    model.id,
                    prompt_tokens_total.saturating_add(generated_tokens_total),
                    now,
                );
                let avg_tokens_evaluated_per_decode_step = if decode_steps_total > 0 {
                    decode_tokens_evaluated_total as f64 / decode_steps_total as f64
                } else {
                    0.0
                };

                statuses.push(ModelStatus {
                    info: model,
                    active_inferences: active,
                    total_inferences: total,
                    queue_depth,
                    memory_usage,
                    gpu_utilization,
                    throughput,
                    prompt_tokens_total,
                    generated_tokens_total,
                    generated_tokens_per_sec,
                    total_tokens_per_sec,
                    decode_steps_total,
                    decode_tokens_evaluated_total,
                    avg_tokens_evaluated_per_decode_step,
                    kv_partial_reuse_hits_total,
                    kv_partial_reuse_tokens_saved_total,
                    onnx_session_pool_total,
                    onnx_session_pool_idle,
                    onnx_session_pool_waits_total,
                    onnx_session_pool_wait_seconds_total,
                    healthy,
                });
            }

            throughput_samples.retain(|id, _| seen_ids.contains(id));
            generated_token_samples.retain(|id, _| seen_ids.contains(id));
            total_token_samples.retain(|id, _| seen_ids.contains(id));
            warp::reply::json(&statuses)
        });

        let model_registry_for_get = model_registry_clone.clone();
        let replica_pools_for_get = replica_pools_clone.clone();
        let metrics_for_get = shared_metrics_clone.clone();
        let throughput_samples_for_get = throughput_samples_clone.clone();
        let generated_token_samples_for_get = generated_token_samples_clone.clone();
        let total_token_samples_for_get = total_token_samples_clone.clone();
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
                        prompt_tokens_total: u64,
                        generated_tokens_total: u64,
                        generated_tokens_per_sec: f64,
                        total_tokens_per_sec: f64,
                        decode_steps_total: u64,
                        decode_tokens_evaluated_total: u64,
                        avg_tokens_evaluated_per_decode_step: f64,
                        kv_partial_reuse_hits_total: u64,
                        kv_partial_reuse_tokens_saved_total: u64,
                        onnx_session_pool_total: usize,
                        onnx_session_pool_idle: usize,
                        onnx_session_pool_waits_total: u64,
                        onnx_session_pool_wait_seconds_total: f64,
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

                            let (
                                queue_depth,
                                healthy,
                                engine_memory,
                                engine_gpu_util,
                                prompt_tokens_total,
                                generated_tokens_total,
                                decode_steps_total,
                                decode_tokens_evaluated_total,
                                kv_partial_reuse_hits_total,
                                kv_partial_reuse_tokens_saved_total,
                                onnx_session_pool_total,
                                onnx_session_pool_idle,
                                onnx_session_pool_waits_total,
                                onnx_session_pool_wait_seconds_total,
                            ) = if let Some(pool) = replica_pools_for_get.read().get(&model.id) {
                                let metrics = pool.get_metrics();
                                (
                                    pool.get_queue_depth(),
                                    pool.is_healthy(),
                                    metrics.memory_usage,
                                    metrics.gpu_utilization,
                                    metrics.prompt_tokens_total,
                                    metrics.generated_tokens_total,
                                    metrics.decode_steps_total,
                                    metrics.decode_tokens_evaluated_total,
                                    metrics.kv_partial_reuse_hits_total,
                                    metrics.kv_partial_reuse_tokens_saved_total,
                                    metrics.onnx_session_pool_total,
                                    metrics.onnx_session_pool_idle,
                                    metrics.onnx_session_pool_waits_total,
                                    metrics.onnx_session_pool_wait_seconds_total,
                                )
                            } else {
                                ((0, 0), true, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)
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
                            let (generated_tokens_per_sec, total_tokens_per_sec) = {
                                let now = Instant::now();
                                let mut generated_token_samples =
                                    generated_token_samples_for_get.write();
                                let mut total_token_samples = total_token_samples_for_get.write();
                                (
                                    update_throughput(
                                        &mut generated_token_samples,
                                        model.id,
                                        generated_tokens_total,
                                        now,
                                    ),
                                    update_throughput(
                                        &mut total_token_samples,
                                        model.id,
                                        prompt_tokens_total.saturating_add(generated_tokens_total),
                                        now,
                                    ),
                                )
                            };
                            let avg_tokens_evaluated_per_decode_step = if decode_steps_total > 0 {
                                decode_tokens_evaluated_total as f64 / decode_steps_total as f64
                            } else {
                                0.0
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
                                prompt_tokens_total,
                                generated_tokens_total,
                                generated_tokens_per_sec,
                                total_tokens_per_sec,
                                decode_steps_total,
                                decode_tokens_evaluated_total,
                                avg_tokens_evaluated_per_decode_step,
                                kv_partial_reuse_hits_total,
                                kv_partial_reuse_tokens_saved_total,
                                onnx_session_pool_total,
                                onnx_session_pool_idle,
                                onnx_session_pool_waits_total,
                                onnx_session_pool_wait_seconds_total,
                                healthy,
                            };

                            warp::reply::json(&status)
                        }
                        None => warp::reply::json(&ErrorResponse {
                            error: format!("Model {} not found", model_id),
                        }),
                    }
                });

        let system_routes = build_system_routes(
            model_registry_clone.clone(),
            replica_pools_clone.clone(),
            device_info_for_api.clone(),
            runtime_samples_clone.clone(),
            runtime_pressure_state.clone(),
        );

        let engine_routes = build_engine_routes();

        let extension_routes = build_extension_routes(
            extension_manager.clone(),
            running_connectors.clone(),
            rag_state.clone(),
        );

        let rag_routes = build_rag_routes(rag_state.clone());

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
        let shared_kv_for_start = shared_kv.clone();
        let swap_map_for_start = swap_map.clone();

        let start_model = warp::path!("api" / "models" / "start")
            .and(warp::post())
            .and(warp::body::json())
            .then(move |req: StartModelRequest| {
                let model_registry = model_registry_for_start.clone();
                let replica_pools = replica_pools_for_start.clone();
                let device_info = device_info_for_start.clone();
                let shared_kv = shared_kv_for_start.clone();
                let shared_metrics = shared_metrics_for_start.clone();
                let model_id_counter = model_id_counter_for_start.clone();
                let recycled_model_ids = recycled_model_ids_for_start.clone();
                let model_paths = model_paths_for_start.clone();
                let onnx_tuning_profile = onnx_tuning_profile_for_start.clone();
                let swap_map = swap_map_for_start.clone();

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
                        let shared_kv = shared_kv.clone();
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
                                    shared_kv,
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
                                Ok(Ok((pool, handles))) => {
                                    replica_pools.write().insert(model_id, pool);
                                    swap_map.write().insert(model_id, handles);
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
        let swap_map_for_stop = swap_map.clone();
        let shared_kv_for_stop = shared_kv.clone();

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
                swap_map_for_stop.write().remove(&model_id);
                shared_kv_for_stop.detach_engine_for_model(model_id);
                #[cfg(any(feature = "gguf-native", feature = "gguf-cuda-shared-kv"))]
                shared_kv_for_stop.detach_gpu_pool(model_id);

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
        let swap_map_for_remove = swap_map.clone();
        let shared_kv_for_remove = shared_kv.clone();

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

                force_stop_model_before_remove(
                    base_model_id,
                    &replicas,
                    &model_registry_for_remove,
                    &replica_pools_for_remove,
                    &swap_map_for_remove,
                    &shared_kv_for_remove,
                );
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

        // Shared response types for the three hot-swap endpoints.
        #[derive(Serialize)]
        struct SwapMsgResp {
            message: String,
        }
        #[derive(Serialize)]
        struct SwapErrResp {
            error: String,
        }
        #[derive(Serialize)]
        struct SwapStatusResp {
            model_id: u32,
            supports_swap: bool,
            staged: bool,
        }

        // Returns the engine handles for a model, or an error reply if not found.
        fn lookup_swap_handles(
            swap_map: &Arc<RwLock<HashMap<u32, Vec<EngineHandle>>>>,
            model_id: u32,
        ) -> Result<Vec<EngineHandle>, warp::reply::WithStatus<warp::reply::Json>> {
            let g = swap_map.read();
            match g.get(&model_id) {
                Some(v) => Ok(v.clone()),
                None => Err(warp::reply::with_status(
                    warp::reply::json(&SwapErrResp {
                        error: format!("model {} not found", model_id),
                    }),
                    warp::http::StatusCode::NOT_FOUND,
                )),
            }
        }

        // POST /api/models/:id/stage - Pre-load next model weights into CPU RAM
        let swap_map_for_stage = swap_map.clone();
        let stage_model_route = warp::path!("api" / "models" / u32 / "stage")
            .and(warp::post())
            .and(warp::body::json())
            .then(move |model_id: u32, body: serde_json::Value| {
                let swap_map = swap_map_for_stage.clone();
                async move {
                    use warp::http::StatusCode;
                    let path_str = match body.get("model_path").and_then(|v| v.as_str()) {
                        Some(s) => s.to_string(),
                        None => {
                            return warp::reply::with_status(
                                warp::reply::json(&SwapErrResp {
                                    error: "missing model_path".into(),
                                }),
                                StatusCode::BAD_REQUEST,
                            )
                        }
                    };
                    let handles = match lookup_swap_handles(&swap_map, model_id) {
                        Ok(h) => h,
                        Err(r) => return r,
                    };
                    if !handles.iter().any(|e| e.supports_swap()) {
                        return warp::reply::with_status(
                            warp::reply::json(&SwapErrResp {
                                error: "backend does not support hot-swap".into(),
                            }),
                            StatusCode::BAD_REQUEST,
                        );
                    }
                    let path = std::path::PathBuf::from(&path_str);
                    for engine in &handles {
                        if let Err(e) = engine.stage(&path).await {
                            return warp::reply::with_status(
                                warp::reply::json(&SwapErrResp {
                                    error: e.to_string(),
                                }),
                                StatusCode::INTERNAL_SERVER_ERROR,
                            );
                        }
                    }
                    warp::reply::with_status(
                        warp::reply::json(&SwapMsgResp {
                            message: format!(
                                "model {} staged from {}; call /swap when ready",
                                model_id, path_str
                            ),
                        }),
                        StatusCode::OK,
                    )
                }
            });

        // POST /api/models/:id/swap - Activate staged weights (PCIe transfer only)
        let swap_map_for_swap = swap_map.clone();
        let swap_model_route = warp::path!("api" / "models" / u32 / "swap")
            .and(warp::post())
            .then(move |model_id: u32| {
                let swap_map = swap_map_for_swap.clone();
                async move {
                    use warp::http::StatusCode;
                    let handles = match lookup_swap_handles(&swap_map, model_id) {
                        Ok(h) => h,
                        Err(r) => return r,
                    };
                    for engine in &handles {
                        if let Err(e) = engine.swap().await {
                            return warp::reply::with_status(
                                warp::reply::json(&SwapErrResp {
                                    error: e.to_string(),
                                }),
                                StatusCode::INTERNAL_SERVER_ERROR,
                            );
                        }
                    }
                    warp::reply::with_status(
                        warp::reply::json(&SwapMsgResp {
                            message: format!("model {} swapped to staged weights", model_id),
                        }),
                        StatusCode::OK,
                    )
                }
            });

        // GET /api/models/:id/swap-status - Check hot-swap staging readiness
        let swap_map_for_status = swap_map.clone();
        let swap_status_route = warp::path!("api" / "models" / u32 / "swap-status")
            .and(warp::get())
            .then(move |model_id: u32| {
                let swap_map = swap_map_for_status.clone();
                async move {
                    use warp::http::StatusCode;
                    let handles = match lookup_swap_handles(&swap_map, model_id) {
                        Ok(h) => h,
                        Err(r) => return r,
                    };
                    let supports = handles.iter().any(|e| e.supports_swap());
                    // All swap-capable engines must have weights staged.
                    let staged = supports
                        && handles
                            .iter()
                            .filter(|e| e.supports_swap())
                            .all(|e| e.is_staged());
                    warp::reply::with_status(
                        warp::reply::json(&SwapStatusResp {
                            model_id,
                            supports_swap: supports,
                            staged,
                        }),
                        StatusCode::OK,
                    )
                }
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

        let auth_routes = build_auth_routes(api_auth_state_for_api.clone());

        let static_routes = build_static_routes();

        let reader_api_routes = list_models
            .or(get_model)
            .or(system_routes)
            .or(engine_routes.reader)
            .or(rag_routes)
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

        let writer_api_routes = extension_routes.map(reply_into_response);
        let writer_api_routes = api_auth_filter(
            ApiRole::Writer,
            ApiScope::Write,
            api_auth_state_for_api.clone(),
        )
        .and(writer_api_routes)
        .map(|response: warp::reply::Response| response);

        let admin_api_routes = engine_routes
            .admin
            .or(start_model)
            .or(stop_model)
            .or(remove_model)
            .or(stage_model_route)
            .or(swap_model_route)
            .or(swap_status_route)
            .or(update_scaling)
            .or(auth_routes.admin)
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

        let login_route = auth_routes.login;

        let routes = static_routes
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
        log::info!("   - POST /api/models/:id/stage      - Pre-load next model into CPU RAM (hot-swap phase 1)");
        log::info!("   - GET  /api/models/:id/swap-status - Check if staging is complete");
        log::info!("   - POST /api/models/:id/swap        - Activate staged weights via PCIe transfer (hot-swap phase 2)");
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

    spawn_auto_scaler_task(AutoScalerTaskConfig {
        auto_scaler: auto_scaler.clone(),
        model_registry: model_registry.clone(),
        replica_pools: replica_pools.clone(),
        swap_map: swap_map_for_autoscaler.clone(),
        model_paths: model_paths.clone(),
        device_info: device_info.clone(),
        unique_id_counter: unique_id_counter.clone(),
        shared_metrics: shared_metrics.clone(),
        shared_kv: shared_kv_for_autoscaler,
        batch_size: args.batch_size,
        scheduler_queue_size: args.scheduler_queue_size,
        scheduler_max_micro_batch: args.scheduler_max_micro_batch,
        scheduler_queue_delay_ms: args.scheduler_queue_delay_ms,
        topology: args.topology.clone(),
        tp_degree: args.tp_degree,
        onnx_tuning_profile: onnx_tuning_profile.clone(),
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
    print_startup_ready(
        startup_elapsed_ms,
        &serving_endpoint,
        &http_bound_addr.ip().to_string(),
        http_bound_addr.port(),
    );

    server
        .run()
        .await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

    Ok(())
}
