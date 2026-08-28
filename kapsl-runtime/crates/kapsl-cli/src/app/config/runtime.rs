//! Translation from CLI and environment input into immutable runtime configuration.

use super::*;
use clap::ArgMatches;

pub(crate) struct ResolvedRuntimeConfig {
    pub(crate) model_paths: Vec<PathBuf>,
    pub(crate) offline: bool,
    pub(crate) state_dir: Option<PathBuf>,
    pub(crate) state_layout: RuntimeStateLayout,
    pub(crate) auth_state: Arc<RwLock<ApiAuthState>>,
    pub(crate) log_sensitive_ids: bool,
    pub(crate) http_bind_addr: IpAddr,
    pub(crate) http_port: u16,
    pub(crate) transport: RuntimeTransportConfig,
    pub(crate) kv_control: KvControlConfig,
    pub(crate) model_loading: ModelLoadingConfig,
    worker: bool,
    worker_model_id: Option<u32>,
    startup_summary: RuntimeStartupSummary,
}

struct RuntimeStartupSummary {
    state_dir: Option<PathBuf>,
    performance_profile: PerformanceProfile,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    applied_tuning: AppliedPerformanceTuning,
}

impl ResolvedRuntimeConfig {
    /// Validate exposure policy after logging has been initialized, then emit
    /// the startup configuration summary from resolved values.
    pub(crate) fn validate_and_log(&self) -> Result<(), DynError> {
        let allow_insecure_http = env_flag(ALLOW_INSECURE_HTTP_ENV);
        if !self.http_bind_addr.is_loopback() && !allow_insecure_http {
            return Err(format!(
                "Refusing to bind HTTP API on non-loopback address {} without {}=1. Use a TLS-terminating reverse proxy if exposing runtime externally.",
                self.http_bind_addr, ALLOW_INSECURE_HTTP_ENV
            )
            .into());
        }
        if !self.http_bind_addr.is_loopback() {
            log::warn!(
                "HTTP API is bound to {}. Traffic is plaintext HTTP; place runtime behind TLS and network ACLs.",
                self.http_bind_addr
            );
        }
        self.transport.validate_tcp_exposure()?;

        let summary = &self.startup_summary;
        if let Some(rationale) = &summary.applied_tuning.auto_tune_rationale {
            log::info!("[auto-tune] {rationale}");
        }
        if let Some(state_dir) = summary.state_dir.as_ref() {
            log::info!("Runtime state directory: {}", state_dir.display());
        }
        print_startup_banner();
        log::info!("🚀 Starting kapsl-runtime...\n");
        log::info!(
            "Performance profile: {} (batch_size={}, transport={}, scheduler_queue_size={}, scheduler_max_micro_batch={}, scheduler_queue_delay_ms={})",
            summary.performance_profile.as_str(),
            summary.batch_size,
            self.transport.mode.as_str(),
            summary.scheduler_queue_size,
            summary.scheduler_max_micro_batch,
            summary.scheduler_queue_delay_ms
        );
        let tuning = &summary.applied_tuning;
        if tuning.batch_size.is_some()
            || tuning.transport.is_some()
            || tuning.scheduler_queue_size.is_some()
            || tuning.scheduler_max_micro_batch.is_some()
            || tuning.scheduler_queue_delay_ms.is_some()
            || tuning.media_preprocess.is_some()
            || tuning.rust_log.is_some()
        {
            log::info!(
                "Applied performance tuning overrides from profile: batch_size={:?}, transport={:?}, scheduler_queue_size={:?}, scheduler_max_micro_batch={:?}, scheduler_queue_delay_ms={:?}, media_preprocess={:?}, rust_log={:?}",
                tuning.batch_size,
                tuning.transport,
                tuning.scheduler_queue_size,
                tuning.scheduler_max_micro_batch,
                tuning.scheduler_queue_delay_ms,
                tuning.media_preprocess,
                tuning.rust_log
            );
        }
        log_auth_configuration(&self.auth_state);
        if !self.log_sensitive_ids {
            log::info!(
                "Sensitive request/session identifiers are redacted in logs (set {}=1 to disable redaction)",
                LOG_SENSITIVE_IDS_ENV
            );
        }
        Ok(())
    }

    pub(crate) fn expand_model_bundles(
        &mut self,
        device_info: &DeviceInfo,
    ) -> Result<(), DynError> {
        self.model_paths = expand_run_bundles(&self.model_paths, device_info)?;
        Ok(())
    }

    pub(crate) fn worker_config(&self) -> Result<Option<WorkerRunConfig>, String> {
        if !self.worker {
            return Ok(None);
        }
        let [model_path] = self.model_paths.as_slice() else {
            return Err("worker mode expects exactly one --model".to_string());
        };
        Ok(Some(WorkerRunConfig::new(
            self.worker_model_id.unwrap_or(0),
            model_path.clone(),
            self.transport.socket_path.clone(),
        )))
    }
}

pub(crate) fn resolve_runtime_config(
    mut args: Args,
    matches: &ArgMatches,
) -> Result<ResolvedRuntimeConfig, DynError> {
    args.model.extend(std::mem::take(&mut args.input));
    configure_llama_cpp_backend_packs(args.offline);
    configure_onnx_backend_packs(args.offline)
        .map_err(|error| format!("Configure lazy ONNX backend packs: {error}"))?;
    let applied_tuning = apply_performance_profile(&mut args, matches);
    propagate_kv_compression_bits(args.kv_compression_bits);

    let state_layout = resolve_runtime_state_layout(args.state_dir.as_deref());
    let auth_state = Arc::new(RwLock::new(ApiAuthState::from_store_path(
        state_layout.auth_store_path.clone(),
    )));
    let log_sensitive_ids = env_flag(LOG_SENSITIVE_IDS_ENV);
    let http_bind_addr = parse_bind_ip(&args.http_bind, IpAddr::from([127, 0, 0, 1]), "http_bind");
    let transport = RuntimeTransportConfig {
        mode: RuntimeTransportMode::parse(&args.transport)?,
        socket_path: args.socket.clone(),
        tcp_bind: args.bind.clone(),
        tcp_port: args.port,
        tcp_auth_token: optional_env_var(TCP_AUTH_TOKEN_ENV),
        shm_size_bytes: args
            .shm_size_mb
            .or_else(|| {
                std::env::var("KAPSL_SHM_SIZE_MB")
                    .ok()
                    .and_then(|value| value.trim().parse::<usize>().ok())
            })
            .unwrap_or(256)
            * 1024
            * 1024,
    };
    let kv_control = KvControlConfig {
        socket_path: args.kv_control_socket.clone(),
        lease_ttl_ms: args.kv_control_lease_ttl_ms,
        shared_pool_profiles: args.kv_shared_pool_profile.clone(),
    };
    let model_loading = ModelLoadingConfig::from_args(&args)
        .map_err(|error| format!("Invalid model loading configuration: {error}"))?;
    let startup_summary = RuntimeStartupSummary {
        state_dir: args.state_dir.clone(),
        performance_profile: args.performance_profile,
        batch_size: args.batch_size,
        scheduler_queue_size: args.scheduler_queue_size,
        scheduler_max_micro_batch: args.scheduler_max_micro_batch,
        scheduler_queue_delay_ms: args.scheduler_queue_delay_ms,
        applied_tuning,
    };

    Ok(ResolvedRuntimeConfig {
        model_paths: args.model,
        offline: args.offline,
        state_dir: args.state_dir,
        state_layout,
        auth_state,
        log_sensitive_ids,
        http_bind_addr,
        http_port: args.metrics_port,
        transport,
        kv_control,
        model_loading,
        worker: args.worker,
        worker_model_id: args.worker_model_id,
        startup_summary,
    })
}

fn propagate_kv_compression_bits(bits: Option<u8>) {
    let Some(bits) = bits else {
        return;
    };
    if (2..=4).contains(&bits) {
        // SAFETY: runtime configuration is resolved before any worker threads
        // or serving tasks are started.
        unsafe { std::env::set_var("KAPSL_LLM_KV_COMPRESSION_BITS", bits.to_string()) };
    } else {
        eprintln!(
            "Warning: --kv-compression-bits {bits} is invalid (must be 2, 3, or 4); ignoring"
        );
    }
}

fn log_auth_configuration(auth_state: &RwLock<ApiAuthState>) {
    let auth_status = auth_state.read().status_response();
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
}
