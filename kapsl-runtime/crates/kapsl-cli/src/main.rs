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
mod http_models;
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
use http_models::*;
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
        let model_routes = build_model_routes(ModelRoutesConfig {
            model_registry: model_registry_clone.clone(),
            replica_pools: replica_pools_clone.clone(),
            shared_metrics: shared_metrics_clone.clone(),
            throughput_samples: throughput_samples_clone.clone(),
            generated_token_samples: generated_token_samples_clone.clone(),
            total_token_samples: total_token_samples_clone.clone(),
            device_info: device_info_for_api.clone(),
            batch_size: args.batch_size,
            scheduler_queue_size: args.scheduler_queue_size,
            scheduler_max_micro_batch: args.scheduler_max_micro_batch,
            scheduler_queue_delay_ms: args.scheduler_queue_delay_ms,
            model_id_counter: model_id_counter.clone(),
            recycled_model_ids: recycled_model_ids.clone(),
            model_paths: model_paths_clone.clone(),
            onnx_tuning_profile: onnx_tuning_profile_for_api.clone(),
            shared_kv: shared_kv.clone(),
            swap_map: swap_map.clone(),
            rag_state: rag_state.clone(),
            inter_model_relay_state: inter_model_relay_state.clone(),
            runtime_pressure_state: runtime_pressure_state.clone(),
            runtime_pressure_config: runtime_pressure_config.clone(),
            auto_scaler: auto_scaler_api.clone(),
            log_sensitive_ids: log_sensitive_ids_for_api,
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

        let auth_routes = build_auth_routes(api_auth_state_for_api.clone());

        let static_routes = build_static_routes();

        let reader_api_routes = model_routes
            .reader
            .or(system_routes)
            .or(engine_routes.reader)
            .or(rag_routes)
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
            .or(model_routes.admin)
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
