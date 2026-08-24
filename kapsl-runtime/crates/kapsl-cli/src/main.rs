use base64::engine::general_purpose::{
    STANDARD as BASE64, URL_SAFE_NO_PAD as BASE64_URL_SAFE_NO_PAD,
};
use base64::Engine as _;
use clap::{parser::ValueSource, ArgGroup, ArgMatches, FromArgMatches, Parser};
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
    BatchingPolicy, BinaryTensorPacket, Engine, EngineError, EngineHandle, EngineMetrics,
    EngineModelInfo, InferenceRequest, TensorDtype,
};
#[cfg(feature = "gpu-device-pool")]
use kapsl_engine_api::{ExternalDeviceMemory, ExternalDeviceMemoryReport};
use kapsl_hal::device::DeviceInfo;
use kapsl_ipc::{IpcServer, TcpServer};
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
use kapsl_shm::{SchedulerSnapshot, ShmServer};
use kapsl_transport::TransportServer;
use parking_lot::{Mutex, RwLock};
use prometheus::Registry;
use rand::rngs::OsRng;
use rand::RngCore;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::future::Future;
use std::io::{BufRead, BufWriter, Cursor, Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{Pid, System};
use tar::{Archive, Builder};
use tokio::sync::Mutex as AsyncMutex;
use warp::Filter;

mod app;
mod features;
mod http;
mod runtime;

use app::*;
use features::*;
use http::*;
use runtime::*;

type DynError = Box<dyn std::error::Error + Send + Sync>;
#[cfg(test)]
mod tests;

fn system_memory_bytes_from_sysinfo(bytes: u64) -> Option<usize> {
    // sysinfo 0.30 reports memory in bytes. Older releases used KiB, which is
    // why multiplying by 1024 here used to be necessary.
    usize::try_from(bytes).ok()
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
        Some(KapslCommand::Extension(args)) => return execute_extension_command(args),
        Some(KapslCommand::Provider(args)) => return execute_provider_command(args),
        Some(KapslCommand::AddModel(args)) => return execute_add_model_command(args),
        Some(KapslCommand::List(args)) => return execute_list_command(args),
        Some(KapslCommand::RemoveModel(args)) => return execute_remove_model_command(args),
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
    let tcp_auth_token = optional_env_var(TCP_AUTH_TOKEN_ENV);
    if args.transport == "tcp" {
        let tcp_bind_addr = parse_bind_ip(&args.bind, IpAddr::from([127, 0, 0, 1]), "bind");
        validate_native_tcp_exposure(tcp_bind_addr, tcp_auth_token.as_deref())?;
        if !tcp_bind_addr.is_loopback() {
            log::warn!(
                "Native TCP inference is bound to {} with token authentication. Traffic remains plaintext; use a trusted network or TLS tunnel.",
                tcp_bind_addr
            );
        }
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
    } else {
        log::warn!(
            "API authentication is disabled. /api routes are restricted to loopback clients only. Create an API key via /api/auth/access/* or set {} / {} / {}.",
            API_READER_TOKEN_ENV,
            API_WRITER_TOKEN_ENV,
            API_ADMIN_TOKEN_ENV
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
    #[cfg(not(unix))]
    if args.kv_control_socket.is_some() {
        return Err("--kv-control-socket currently requires a Unix host".into());
    }
    if args.kv_control_socket.is_none() && !args.kv_shared_pool_profile.is_empty() {
        return Err("--kv-shared-pool-profile requires --kv-control-socket".into());
    }
    #[cfg(not(all(feature = "gpu-device-pool", target_os = "linux")))]
    if !args.kv_shared_pool_profile.is_empty() {
        return Err("--kv-shared-pool-profile requires a Linux gpu-device-pool build".into());
    }
    #[cfg(unix)]
    if let Some(kv_socket) = args.kv_control_socket.as_ref() {
        let inference_uses_socket = matches!(args.transport.as_str(), "socket" | "hybrid")
            || (args.transport == "auto" && !ShmServer::is_available());
        if inference_uses_socket && kv_socket == Path::new(&args.socket) {
            return Err("--kv-control-socket must differ from the inference --socket path".into());
        }
    }

    let registry = Arc::new(Registry::new());
    let model_registry = Arc::new(ModelRegistry::new());
    let models = ModelManager::new(model_registry.clone());

    // Resolve and retain every startup package before choosing the immutable
    // physical GPU pool. This gives automatic sizing a host-only view of
    // external weights without constructing a backend or CUDA/ORT session.
    log::info!("=== Package Planning ===");
    let mut startup_plans = Vec::with_capacity(args.model.len());
    for model_path in &args.model {
        let model_id = models.allocate_model_id();
        let onnx_tuning = onnx_tuning_profile.resolve(model_id);
        let plan = prepare_model_load(
            model_id,
            model_id,
            0,
            model_path,
            &device_info,
            args.batch_size,
            args.scheduler_queue_size,
            args.scheduler_max_micro_batch,
            args.scheduler_queue_delay_ms,
            &args.topology,
            args.tp_degree,
            &onnx_tuning,
        )?;
        startup_plans.push((model_path.clone(), plan));
    }

    // One process-owned facade for logical KV, physical device memory, and
    // pressure state. Pool registration completes before any backend/session
    // construction begins.
    #[cfg(feature = "gpu-device-pool")]
    let resources = {
        let bootstrap =
            device_memory_bootstrap_plan(startup_plans.iter().map(|(_, plan)| plan), &device_info)?;
        RuntimeResources::new_with_device_memory_plan(&device_info, &bootstrap)?
    };
    #[cfg(not(feature = "gpu-device-pool"))]
    let resources = RuntimeResources::new(&device_info)?;

    #[cfg(unix)]
    let mut kv_control_task = if let Some(socket_path) = args.kv_control_socket.as_ref() {
        #[cfg(all(feature = "gpu-device-pool", target_os = "linux"))]
        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            resources.memory().clone(),
            Duration::from_millis(args.kv_control_lease_ttl_ms),
            Some(CudaIpcSharedPoolProvisioner::new(
                resources.memory().clone(),
            )),
            parse_shared_pool_profiles(&args.kv_shared_pool_profile)?,
        )?;
        #[cfg(not(all(feature = "gpu-device-pool", target_os = "linux")))]
        let coordinator = ExternalKvCoordinator::new(
            resources.memory().clone(),
            Duration::from_millis(args.kv_control_lease_ttl_ms),
        )?;
        let control_server = KvControlServer::bind(socket_path, coordinator).await?;
        log::info!(
            "KV participant control: unix://{} (maximum lease TTL={}ms)",
            socket_path.display(),
            args.kv_control_lease_ttl_ms
        );
        Some(tokio::spawn(control_server.run()))
    } else {
        None
    };
    #[cfg(not(unix))]
    let mut kv_control_task: Option<tokio::task::JoinHandle<std::io::Result<()>>> = None;

    let auto_scaler = Arc::new(RwLock::new(AutoScaler::new()));
    let runtime_samples = Arc::new(RwLock::new(RuntimeSamples::default()));
    let throughput_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let generated_token_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let total_token_samples: Arc<RwLock<HashMap<u32, ThroughputSample>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let latency_samples: Arc<RwLock<HashMap<u32, LatencyWindow>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let runtime_pressure_config = resources.pressure().config();
    let runtime_pressure_state = resources.pressure().state();
    let inference_service = InferenceService::new(
        models.clone(),
        resources.pressure().clone(),
        latency_samples.clone(),
    );

    // Create shared metrics instance ONCE for all models
    let shared_metrics = kapsl_monitor::metrics::KapslMetrics::new(&registry);
    #[cfg(feature = "gpu-device-pool")]
    resources.attach_device_memory_metrics(shared_metrics.clone());

    let runtime_samples_for_sampler = runtime_samples.clone();
    let has_cuda_for_sampler = device_info.has_cuda;
    let runtime_pressure_config_for_sampler = runtime_pressure_config.clone();
    let runtime_pressure_state_for_sampler = runtime_pressure_state.clone();
    let kv_for_rebalance = resources.kv().clone();
    let memory_for_sampler = resources.memory().clone();
    let models_for_memory_reconciliation = models.clone();
    // Opt-in co-tenancy guard: probe for foreign GPU processes each tick,
    // shrink the live KV ceiling by their footprint, and exclude their bytes
    // from the pressure ratio. Default off — single-tenant behavior unchanged.
    let cotenancy_guard = optional_env_var(COTENANCY_GUARD_ENV)
        .is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on"));
    // Ceiling observability exists only when the guard does, so the default
    // configuration exports no new metric series and logs nothing new.
    let mut cotenancy_exporter = cotenancy_guard.then(|| CotenancyCeilingExporter::new(&registry));
    let memory_exporter = MemorySnapshotExporter::new(&registry);
    tokio::spawn(async move {
        let pid = Pid::from_u32(std::process::id());
        let mut system = System::new();
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        let mut nvidia_smi_retry_after: Option<Instant> = None;
        system.refresh_memory();
        let total_system_memory_bytes = system_memory_bytes_from_sysinfo(system.total_memory());
        loop {
            interval.tick().await;

            // Reclaim KV block quota from any engine whose health changed
            // (e.g. tripped circuit breaker / stalled watchdog), redistributing
            // it to healthy engines.
            kv_for_rebalance.maybe_rebalance_for_health();

            // Backend-owned host/provider/device allocations can change long
            // after model load (KV growth, provider arenas, compaction, or
            // migration). Resample every live engine before publishing the
            // authority snapshot used by pressure and admission policy.
            models_for_memory_reconciliation.reconcile_memory_reports();

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

            // Co-tenancy: measure foreign VRAM, push it into the live KV
            // ceiling (concurrency lever), and report it to the pressure split
            // (so it never drives output truncation). Shares the nvidia-smi
            // backoff above: while the sampler is backing off, skip the probe
            // too rather than shelling out to a broken nvidia-smi twice.
            let foreign_gpu_memory_bytes = if cotenancy_guard
                && has_cuda_for_sampler
                && nvidia_smi_retry_after.is_none_or(|retry_after| now >= retry_after)
            {
                let foreign = sample_foreign_vram();
                let ceilings = memory_for_sampler.reconcile_external_device_memory(&foreign);
                if ceilings
                    .iter()
                    .any(|sample| sample.smoothed_bytes != sample.previous_bytes)
                {
                    kv_for_rebalance.rebalance_kv_caps();
                }
                if let Some(exporter) = cotenancy_exporter.as_mut() {
                    exporter.observe(&ceilings);
                }
                Some(foreign.values().sum::<usize>())
            } else {
                None
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
                foreign_gpu_memory_bytes,
                collected_at_ms,
            };
            *runtime_samples_for_sampler.write() = snapshot.clone();

            memory_for_sampler.observe_process_memory(process_memory_bytes);
            if let Some(gpu_bytes) = gpu_memory_bytes {
                memory_for_sampler.observe_cuda_memory_total(
                    gpu_bytes.saturating_sub(foreign_gpu_memory_bytes.unwrap_or(0)),
                );
            }
            let memory_snapshot = memory_for_sampler.snapshot();
            memory_exporter.observe(&memory_snapshot);
            let next_state = evaluate_authority_pressure_state(
                &memory_snapshot,
                snapshot.gpu_utilization,
                &runtime_pressure_config_for_sampler,
            );
            let previous_raw =
                runtime_pressure_state_for_sampler.swap(next_state as u8, Ordering::Relaxed);
            let previous = RuntimePressureState::from_u8(previous_raw);
            if previous != next_state {
                log::warn!(
                    "Runtime pressure state changed: {} -> {} (rss={}B total_mem={}B gpu_util={:.2} gpu_mem={:?}/{:?} foreign_gpu_mem={:?})",
                    previous.as_str(),
                    next_state.as_str(),
                    snapshot.process_memory_bytes,
                    snapshot.total_system_memory_bytes.unwrap_or(0),
                    snapshot.gpu_utilization,
                    snapshot.gpu_memory_bytes,
                    snapshot.gpu_memory_total_bytes,
                    snapshot.foreign_gpu_memory_bytes
                );
            }
        }
    });

    // 2. Construct and load the preplanned model backends.
    log::info!("=== Model Loading ===");
    let load_parallelism = resolve_model_load_parallelism(startup_plans.len());
    if startup_plans.len() > 1 {
        log::info!(
            "Loading {} model backends with parallelism {} ({}=N to override)",
            startup_plans.len(),
            load_parallelism,
            MODEL_LOAD_PARALLELISM_ENV
        );
    }

    let load_results = run_with_loading_async("Loading model backends", {
        let device_info = device_info.clone();
        let resources = resources.clone();
        let model_registry = model_registry.clone();
        let shared_metrics = shared_metrics.clone();
        let onnx_tuning_profile = onnx_tuning_profile.clone();
        async move {
            let results = stream::iter(startup_plans.into_iter().map(|(model_path, plan)| {
                let device_info = device_info.clone();
                let resources = resources.clone();
                let model_registry = model_registry.clone();
                let shared_metrics = shared_metrics.clone();
                let model_id = plan.base_model_id();
                let onnx_tuning = onnx_tuning_profile.resolve(model_id);
                async move {
                    let result = load_prepared_model(
                        plan,
                        &device_info,
                        resources,
                        &model_registry,
                        &shared_metrics,
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
                models.install_loaded(model_id, model_path, pool, handles);

                // Register default scaling policy for each model
                auto_scaler
                    .write()
                    .register_policy(model_id, ScalingPolicy::default());
            }
            Err(error) => {
                models.release_model_id(model_id);
                if first_load_error.is_none() {
                    first_load_error = Some(error);
                }
            }
        }
    }
    if let Some(error) = first_load_error {
        return Err(error);
    }

    // Startup loading is done, so no further `std::env::set_var` is sound: from
    // here on the transport server and inference threads are live and may be
    // reading these vars. Models hot-loaded over HTTP inherit the values
    // resolved above instead of re-deriving them. See seal_env_auto_sizing.
    seal_env_auto_sizing();

    log::info!("=== Starting Transport Server ===");
    log::info!("Transport mode: {}", args.transport);

    let get_scheduler_lookup = || {
        let inference_service = inference_service.clone();
        Arc::new(move |model_id: u32| inference_service.scheduler_for_transport(model_id))
            as Arc<dyn Fn(u32) -> Option<Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>
    };
    let get_scheduler_snapshot = || {
        let inference_service = inference_service.clone();
        Arc::new(move || inference_service.scheduler_snapshot()) as SchedulerSnapshot
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
                let tcp_server =
                    TcpServer::new_with_lookup(&args.bind, args.port, get_scheduler_lookup());
                let tcp_server = match tcp_auth_token.as_deref() {
                    Some(token) => tcp_server.with_auth_token(token.to_owned()),
                    None => tcp_server,
                };
                (Box::new(tcp_server), format!("{}:{}", args.bind, args.port))
            }
            "shm" => {
                log::info!("Using shared memory transport");
                let shm_name = format!("/kapsl_shm_{}", std::process::id());
                log::info!("Shared memory: {}", shm_name);
                (
                    Box::new(ShmServer::new_with_lookup_and_registry(
                        &shm_name,
                        shm_size,
                        get_scheduler_lookup(),
                        get_scheduler_snapshot(),
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
                    (
                        Box::new(ShmServer::new_with_lookup_and_registry(
                            &shm_name,
                            shm_size,
                            get_scheduler_lookup(),
                            get_scheduler_snapshot(),
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
    let models_for_api = models.clone();
    let shared_metrics_clone = shared_metrics.clone();
    let metrics_port = args.metrics_port;
    let http_bind_addr_for_api = http_bind_addr;
    let api_auth_state_for_api = api_auth_state.clone();
    let log_sensitive_ids_for_api = log_sensitive_ids;
    let device_info_for_api = device_info.clone(); // Clone Arc for API endpoints
    let auto_scaler_api = auto_scaler.clone();
    let runtime_samples_clone = runtime_samples.clone();
    let throughput_samples_clone = throughput_samples.clone();
    let generated_token_samples_clone = generated_token_samples.clone();
    let total_token_samples_clone = total_token_samples.clone();
    let latency_samples_clone = latency_samples.clone();
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
    let resources_for_api = resources.clone();
    let resources_for_autoscaler = resources.clone();

    tokio::spawn(async move {
        let metrics_route = build_metrics_route(
            registry_arc.clone(),
            api_auth_state_for_api.clone(),
            resources_for_api.clone(),
        );

        // API routes
        let model_routes = build_model_routes(ModelRoutesConfig {
            models: models_for_api.clone(),
            inference: inference_service.clone(),
            shared_metrics: shared_metrics_clone.clone(),
            throughput_samples: throughput_samples_clone.clone(),
            generated_token_samples: generated_token_samples_clone.clone(),
            total_token_samples: total_token_samples_clone.clone(),
            latency_samples: latency_samples_clone.clone(),
            device_info: device_info_for_api.clone(),
            batch_size: args.batch_size,
            scheduler_queue_size: args.scheduler_queue_size,
            scheduler_max_micro_batch: args.scheduler_max_micro_batch,
            scheduler_queue_delay_ms: args.scheduler_queue_delay_ms,
            onnx_tuning_profile: onnx_tuning_profile_for_api.clone(),
            resources: resources_for_api.clone(),
            rag_state: rag_state.clone(),
            auto_scaler: auto_scaler_api.clone(),
            log_sensitive_ids: log_sensitive_ids_for_api,
        });

        // OpenAI-compatible surface. Reader-scoped like `/api/models/:id/infer`,
        // which it delegates to.
        let openai_routes = build_openai_routes(OpenAiRoutesConfig {
            models: models_for_api.clone(),
            inference: inference_service.clone(),
            log_sensitive_ids: log_sensitive_ids_for_api,
        });

        let system_routes = build_system_routes(
            models_for_api.clone(),
            device_info_for_api.clone(),
            runtime_samples_clone.clone(),
            runtime_pressure_state.clone(),
            resources_for_api.clone(),
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
            .or(openai_routes)
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
        models: models.clone(),
        device_info: device_info.clone(),
        shared_metrics: shared_metrics.clone(),
        resources: resources_for_autoscaler,
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

    if let Some(mut control_task) = kv_control_task.take() {
        let mut transport_task = Box::pin(server.run());
        tokio::select! {
            result = &mut transport_task => {
                control_task.abort();
                let _ = control_task.await;
                result.map_err(|error| Box::new(error) as DynError)?;
            }
            result = &mut control_task => {
                let message = match result {
                    Ok(Ok(())) => "KV control listener stopped unexpectedly".to_string(),
                    Ok(Err(error)) => format!("KV control listener failed: {error}"),
                    Err(error) => format!("KV control listener task failed: {error}"),
                };
                return Err(message.into());
            }
        }
    } else {
        server
            .run()
            .await
            .map_err(|error| Box::new(error) as DynError)?;
    }

    Ok(())
}
