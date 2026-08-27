//! Ordered construction of worker and server runtime modes.

use super::*;
use futures::{stream, StreamExt};

/// Application bootstrap with hardware supplied by the composition root.
/// Tests can inject a deterministic snapshot without mocking global probing.
pub(crate) struct RuntimeBootstrap {
    config: ResolvedRuntimeConfig,
    device_info: Arc<DeviceInfo>,
}

pub(crate) enum RuntimeBootstrapOutcome {
    Worker {
        config: WorkerRunConfig,
        load_planner: Arc<ModelLoadPlanner>,
    },
    Server(Box<PreparedServerRuntime>),
}

pub(crate) struct PreparedServerRuntime {
    config: ResolvedRuntimeConfig,
    registry: Arc<Registry>,
    resources: Arc<RuntimeResources>,
    model_runtime: Arc<ModelRuntime>,
    inference: Arc<InferenceService>,
    auto_scaler: Arc<RwLock<AutoScaler>>,
    monitor: RuntimeMonitor,
    kv_control_task: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
}

impl RuntimeBootstrap {
    pub(crate) fn new(config: ResolvedRuntimeConfig, device_info: Arc<DeviceInfo>) -> Self {
        Self {
            config,
            device_info,
        }
    }

    pub(crate) async fn prepare(mut self) -> Result<RuntimeBootstrapOutcome, DynError> {
        log_hardware(&self.device_info);
        let load_planner = Arc::new(self.config.model_loading.planner(self.device_info.clone()));
        self.config.expand_model_bundles(&self.device_info)?;

        if let Some(worker_config) = self
            .config
            .worker_config()
            .map_err(|error| format!("Invalid worker configuration: {error}"))?
        {
            return Ok(RuntimeBootstrapOutcome::Worker {
                config: worker_config,
                load_planner,
            });
        }

        preflight_http_bind(self.config.http_bind_addr, self.config.http_port)?;
        self.config.transport.preflight()?;

        let registry = Arc::new(Registry::new());
        let models = ModelManager::new(Arc::new(ModelRegistry::new()));
        let shared_metrics = kapsl_monitor::metrics::KapslMetrics::new(&registry);
        let startup_plans =
            plan_startup_models(&self.config.model_paths, &models, load_planner.as_ref())?;

        #[cfg(feature = "gpu-device-pool")]
        let resources = {
            let bootstrap = device_memory_bootstrap_plan(
                startup_plans.iter().map(|(_, plan)| plan),
                &self.device_info,
            )?;
            RuntimeResources::new_with_device_memory_plan(&self.device_info, &bootstrap)?
        };
        #[cfg(not(feature = "gpu-device-pool"))]
        let resources = RuntimeResources::new(&self.device_info)?;
        #[cfg(feature = "gpu-device-pool")]
        resources.attach_device_memory_metrics(shared_metrics.clone());

        let model_runtime = Arc::new(ModelRuntime::new(
            load_planner,
            resources.clone(),
            models.clone(),
            shared_metrics,
        ));
        prepare_managed_backends(
            &mut self.config,
            &self.device_info,
            &startup_plans,
            &model_runtime,
            &resources,
        )
        .await?;
        let kv_control_task = self.config.kv_control.start(&resources).await?;

        let auto_scaler = Arc::new(RwLock::new(AutoScaler::new()));
        let monitor = RuntimeMonitor::start(RuntimeMonitorConfig {
            device_info: self.device_info,
            resources: resources.clone(),
            models: models.clone(),
            registry: registry.clone(),
        });
        let inference = InferenceService::new_with_metrics(
            models.clone(),
            resources.pressure().clone(),
            monitor.telemetry(),
            model_runtime.shared_metrics().clone(),
        );
        load_startup_models(startup_plans, &model_runtime, &models, &auto_scaler).await?;

        // Serving threads may read sizing variables after this point, so later
        // model loads reuse the already-resolved values.
        seal_env_auto_sizing();

        Ok(RuntimeBootstrapOutcome::Server(Box::new(
            PreparedServerRuntime {
                config: self.config,
                registry,
                resources,
                model_runtime,
                inference,
                auto_scaler,
                monitor,
                kv_control_task,
            },
        )))
    }
}

impl RuntimeBootstrapOutcome {
    pub(crate) async fn run(self, startup_started_at: Instant) -> Result<(), DynError> {
        match self {
            Self::Worker {
                config,
                load_planner,
            } => run_worker(config, load_planner).await,
            Self::Server(runtime) => (*runtime).run(startup_started_at).await,
        }
    }
}

impl PreparedServerRuntime {
    async fn run(self, startup_started_at: Instant) -> Result<(), DynError> {
        let Self {
            config,
            registry,
            resources,
            model_runtime,
            inference,
            auto_scaler,
            monitor,
            kv_control_task,
        } = self;

        let transport =
            RuntimeTransport::build(&config.transport, inference.clone(), registry.clone())?;
        let serving_endpoint = transport.endpoint().to_owned();
        log::info!("✓ Server ready\n");
        log::info!("🎉 kapsl-runtime is running!");
        log::info!("════════════════════════════════════════\n");

        let runtime_samples = monitor.samples();
        let telemetry = monitor.telemetry();
        let runtime_pressure_state = resources.pressure().state();
        let http_server = start_http_server(
            HttpServerConfig {
                bind_addr: config.http_bind_addr,
                port: config.http_port,
                state_layout: config.state_layout,
                auth_state: config.auth_state,
                log_sensitive_ids: config.log_sensitive_ids,
            },
            HttpServerDependencies {
                registry,
                model_runtime: model_runtime.clone(),
                inference,
                telemetry,
                runtime_samples,
                runtime_pressure_state,
                auto_scaler: auto_scaler.clone(),
                resources: resources.clone(),
            },
        )?;
        let autoscaler_task = spawn_auto_scaler_task(AutoScalerTaskConfig {
            auto_scaler,
            model_runtime,
        });
        let http_bound_addr = http_server.bound_addr();
        print_startup_ready(
            startup_started_at.elapsed().as_millis(),
            &serving_endpoint,
            &http_bound_addr.ip().to_string(),
            http_bound_addr.port(),
        );

        RuntimeSupervisor {
            transport,
            kv_control_task,
            http_server,
            monitor,
            autoscaler_task,
            resources,
        }
        .run()
        .await
    }
}

fn log_hardware(device_info: &DeviceInfo) {
    log::info!("=== Hardware Detection ===");
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
                    .find(|device| {
                        matches!(device.backend, kapsl_hal::device::DeviceBackend::Cuda)
                    })
                    .and_then(|device| device.cuda_version.as_deref())
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
}

fn plan_startup_models(
    model_paths: &[PathBuf],
    models: &Arc<ModelManager>,
    load_planner: &ModelLoadPlanner,
) -> Result<Vec<(PathBuf, ModelLoadPlan)>, DynError> {
    log::info!("=== Package Planning ===");
    let mut plans = Vec::with_capacity(model_paths.len());
    for model_path in model_paths {
        let model_id = models.allocate_model_id();
        let plan = load_planner.prepare(model_id, model_id, 0, model_path, None)?;
        plans.push((model_path.clone(), plan));
    }
    Ok(plans)
}

async fn prepare_managed_backends(
    config: &mut ResolvedRuntimeConfig,
    device_info: &DeviceInfo,
    startup_plans: &[(PathBuf, ModelLoadPlan)],
    model_runtime: &ModelRuntime,
    resources: &Arc<RuntimeResources>,
) -> Result<(), DynError> {
    let uses_managed_vllm = startup_plans
        .iter()
        .any(|(_, plan)| plan.uses_managed_vllm());
    let preliminary_admission = model_runtime
        .preflight_managed_vllm_admission(startup_plans)
        .await?;

    if let Some(source) = uses_managed_vllm.then(certified_managed_vllm_source) {
        let lazy_enabled = std::env::var("KAPSL_LAZY_BACKENDS")
            .ok()
            .is_none_or(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
            });
        let validate_lazy_cache = source == CertifiedManagedVllmSource::LazyCache;
        let install_missing = source == CertifiedManagedVllmSource::Missing && lazy_enabled;
        if validate_lazy_cache || install_missing {
            let manager = BackendManager::from_env(config.offline || !lazy_enabled)?;
            manager.ensure_vllm(&BackendTarget::current(device_info))?;
        }
    }

    let managed_vllm = if uses_managed_vllm {
        let prepared = ManagedVllmDeployment::prepare(
            config.state_dir.as_deref(),
            config.kv_control.socket_path.as_deref(),
            config.kv_control.lease_ttl_ms,
        )?;
        config.kv_control.apply_managed_defaults(&prepared);
        Some(prepared.deployment)
    } else {
        None
    };
    drop(preliminary_admission);
    config.kv_control.validate(&config.transport)?;
    if let Some(deployment) = managed_vllm {
        resources.install_managed_vllm(deployment)?;
    }
    Ok(())
}

async fn load_startup_models(
    startup_plans: Vec<(PathBuf, ModelLoadPlan)>,
    model_runtime: &Arc<ModelRuntime>,
    models: &Arc<ModelManager>,
    auto_scaler: &Arc<RwLock<AutoScaler>>,
) -> Result<(), DynError> {
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

    let results = run_with_loading_async("Loading model backends", {
        let model_runtime = model_runtime.clone();
        async move {
            let results = stream::iter(startup_plans.into_iter().map(|(model_path, plan)| {
                let model_runtime = model_runtime.clone();
                let model_id = plan.base_model_id();
                async move {
                    let result = model_runtime.load_prepared_model(plan).await;
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

    let mut first_error = None;
    for (model_id, model_path, result) in results {
        match result {
            Ok((pool, handles)) => {
                models.install_loaded(model_id, model_path, pool, handles);
                auto_scaler
                    .write()
                    .register_policy(model_id, ScalingPolicy::default());
            }
            Err(error) => {
                models.release_model_id(model_id);
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
