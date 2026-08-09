use super::*;

fn create_runtime_backend_for_device(
    manifest: &Manifest,
    provider: &str,
    device_id: usize,
    device_info: &DeviceInfo,
    tuning: &OnnxRuntimeTuning,
    shared_kv: &SharedKvState,
    model_id: u32,
) -> Result<Box<dyn kapsl_engine_api::Engine>, String> {
    #[cfg(not(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    )))]
    let _ = (shared_kv, model_id);
    #[cfg(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    ))]
    let kind = EngineKind::resolve(manifest);

    #[cfg(feature = "gguf-native")]
    if kind.is_gguf() {
        let backend = if let Some(pool) = shared_kv.device_pool(device_id) {
            BackendFactory::create_gguf_native_device_pool(device_id as i32, pool, model_id)?
        } else {
            BackendFactory::create_gguf_native(device_id as i32, None)?
        };
        return Ok(Box::new(backend));
    }

    #[cfg(all(feature = "gguf-cuda-shared-kv", not(feature = "gguf-native")))]
    if kind.is_gguf() {
        let backend = if let Some(pool) = shared_kv.device_pool(device_id) {
            BackendFactory::create_gguf_cuda_device_pool(device_id as i32, pool, model_id)?
        } else {
            BackendFactory::create_gguf_cuda_shared_kv(device_id as i32, None)?
        };
        return Ok(Box::new(backend));
    }

    #[cfg(feature = "native")]
    if kind == EngineKind::Native {
        if let Some(pool) = shared_kv.device_pool(device_id) {
            return BackendFactory::create_native_device_pool(device_id as i32, pool, model_id)
                .map(|backend| Box::new(backend) as Box<dyn kapsl_engine_api::Engine>);
        }
    }

    BackendFactory::create_backend_for_device_with_tuning(
        manifest,
        provider,
        device_id,
        device_info,
        tuning,
    )
}

fn create_runtime_best_backend(
    manifest: &Manifest,
    device_info: &DeviceInfo,
    tuning: &OnnxRuntimeTuning,
    shared_kv: &SharedKvState,
    model_id: u32,
) -> Result<Box<dyn kapsl_engine_api::Engine>, String> {
    #[cfg(not(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    )))]
    let _ = (shared_kv, model_id);
    #[cfg(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    ))]
    {
        let kind = EngineKind::resolve(manifest);
        let device_id = manifest.hardware_requirements.device_id.unwrap_or(0) as usize;

        #[cfg(feature = "gguf-native")]
        if kind.is_gguf() {
            let backend = if let Some(pool) = shared_kv.device_pool(device_id) {
                BackendFactory::create_gguf_native_device_pool(device_id as i32, pool, model_id)?
            } else {
                BackendFactory::create_gguf_native(device_id as i32, None)?
            };
            return Ok(Box::new(backend));
        }

        #[cfg(all(feature = "gguf-cuda-shared-kv", not(feature = "gguf-native")))]
        if kind.is_gguf() {
            let backend = if let Some(pool) = shared_kv.device_pool(device_id) {
                BackendFactory::create_gguf_cuda_device_pool(device_id as i32, pool, model_id)?
            } else {
                BackendFactory::create_gguf_cuda_shared_kv(device_id as i32, None)?
            };
            return Ok(Box::new(backend));
        }

        #[cfg(feature = "native")]
        if kind == EngineKind::Native {
            if let Some(pool) = shared_kv.device_pool(device_id) {
                return BackendFactory::create_native_device_pool(device_id as i32, pool, model_id)
                    .map(|backend| Box::new(backend) as Box<dyn kapsl_engine_api::Engine>);
            }
        }
    }

    BackendFactory::create_best_backend_with_tuning(manifest, device_info, tuning)
}

pub(crate) fn allocate_model_id(counter: &AtomicU32, recycled_ids: &Mutex<Vec<u32>>) -> u32 {
    if let Some(id) = recycled_ids.lock().pop() {
        id
    } else {
        counter.fetch_add(1, Ordering::SeqCst)
    }
}

pub(crate) fn recycle_model_id(model_id: u32, recycled_ids: &Mutex<Vec<u32>>) {
    recycled_ids.lock().push(model_id);
}

pub(crate) async fn run_worker(
    args: &Args,
    device_info: &DeviceInfo,
    onnx_tuning_profile: &OnnxTuningProfile,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if args.model.len() != 1 {
        return Err("Worker mode expects exactly one --model".into());
    }

    let model_id = args.worker_model_id.unwrap_or(0);
    let model_path = &args.model[0];
    let onnx_tuning = onnx_tuning_profile.resolve(model_id);

    let registry = Arc::new(Registry::new());
    let model_registry = Arc::new(ModelRegistry::new());
    let shared_metrics = kapsl_monitor::metrics::KapslMetrics::new(&registry);

    let (pool, _) = load_model(
        model_id,
        model_path,
        device_info,
        SharedKvStateInner::new_runtime(device_info)?,
        args.batch_size,
        args.scheduler_queue_size,
        args.scheduler_max_micro_batch,
        args.scheduler_queue_delay_ms,
        &model_registry,
        &shared_metrics,
        &args.topology,
        args.tp_degree,
        onnx_tuning,
    )
    .await?;

    let mut schedulers = HashMap::new();
    schedulers.insert(model_id, pool as Arc<dyn ReplicaScheduler + Send + Sync>);

    let server = IpcServer::new(&args.socket, schedulers, None);
    log::info!(
        "Worker process serving model {} via IPC socket {}",
        model_id,
        args.socket
    );
    server.run().await?;
    Ok(())
}

/// Load a model
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_model_blocking(
    model_id: u32,
    model_path: &PathBuf,
    device_info: &DeviceInfo,
    shared_kv: SharedKvState,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    model_registry: &ModelRegistry,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    topology: &str,
    tp_degree: usize,
    onnx_tuning: OnnxRuntimeTuning,
) -> Result<
    (Arc<ReplicaPool<Scheduler>>, Vec<EngineHandle>),
    Box<dyn std::error::Error + Send + Sync>,
> {
    // Same work as `load_model`; this variant exists for the HTTP lifecycle
    // path, which runs inside `tokio::task::spawn_blocking`. Driving the async
    // version with the ambient runtime handle keeps one implementation instead
    // of a near-identical sync fork that has to be fixed twice.
    tokio::runtime::Handle::current().block_on(load_model(
        model_id,
        model_path,
        device_info,
        shared_kv,
        batch_size,
        scheduler_queue_size,
        scheduler_max_micro_batch,
        scheduler_queue_delay_ms,
        model_registry,
        shared_metrics,
        topology,
        tp_degree,
        onnx_tuning,
    ))
}

/// Load a model and create its scheduler
#[allow(clippy::too_many_arguments)]
pub(crate) async fn load_model(
    model_id: u32,
    model_path: &PathBuf,
    device_info: &DeviceInfo,
    shared_kv: SharedKvState,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    model_registry: &ModelRegistry,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    topology: &str,
    tp_degree: usize,
    onnx_tuning: OnnxRuntimeTuning,
) -> Result<
    (Arc<ReplicaPool<Scheduler>>, Vec<EngineHandle>),
    Box<dyn std::error::Error + Send + Sync>,
> {
    log::info!(
        "Current directory: {:?}",
        std::env::current_dir().unwrap_or_default()
    );
    let absolute_path = match model_path.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            log::error!(
                "Failed to canonicalize model path {:?}: {} (CWD: {:?})",
                model_path,
                e,
                std::env::current_dir().unwrap_or_default()
            );
            return Err(format!("Invalid model path {:?}: {}", model_path, e).into());
        }
    };
    log::info!("Loading Model ID {}: {:?}", model_id, absolute_path);

    let loader = resolve_package_loader(&absolute_path, model_id)?;
    log::info!("✓ Package loaded");
    log::info!("  Project: {}", loader.manifest.project_name);
    log::info!("  Framework: {}", loader.manifest.framework);
    log::info!("  Version: {}", loader.manifest.version);
    let queue_overflow_policy = resolve_queue_overflow_policy(&loader.manifest);
    log_queue_policy_caveat(queue_overflow_policy);
    log::info!(
        "  Queue overflow policy: {}",
        queue_overflow_policy.as_str()
    );
    let (scheduler_max_micro_batch, scheduler_queue_delay_ms) =
        resolve_scheduler_tuning_for_framework(
            &loader.manifest,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
        );
    let priority_weight = resolve_model_priority_weight(&loader.manifest, model_id);
    #[cfg(feature = "gpu-device-pool")]
    let engine_kind = EngineKind::resolve(&loader.manifest);
    log::info!("  Priority weight: {}", priority_weight);

    let model_file_path = loader.get_model_path();
    export_gguf_auto_sizing_hint(
        &loader.manifest,
        batch_size,
        Some(model_file_path.as_path()),
    );
    let isolate_process = resolve_isolate_process(&loader.manifest);
    let isolate_strict = resolve_isolate_process_strict(&loader.manifest);
    if isolate_process {
        log::info!(
            "✓ Process isolation enabled for Model ID {} (strict={})",
            model_id,
            isolate_strict
        );
    }

    BackendFactory::validate_requirements(&loader.manifest.hardware_requirements, device_info)
        .map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> = format!(
                "Requirements validation failed for model {}: {}",
                model_id, e
            )
            .into();
            err
        })?;

    // Initialize Device Mesh
    use kapsl_hal::device_mesh::DeviceMesh;
    let pipeline_stages = manifest_llm_pipeline_stages(&loader.manifest);
    let EffectiveTopologyChoice {
        mesh_topology,
        worker_topology,
        worker_tp_degree,
        use_pipeline_backend: use_pipeline,
    } = resolve_effective_topology_choice(
        &loader.manifest,
        topology,
        tp_degree,
        pipeline_stages.as_deref(),
    );

    let selected_mesh = select_mesh_devices(&loader.manifest.hardware_requirements, device_info)
        .map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> =
                format!("Failed to select devices for model {}: {}", model_id, e).into();
            err
        })?;
    let logical_provider = selected_mesh.logical_provider;

    let device_mesh =
        DeviceMesh::with_topology(selected_mesh.devices, mesh_topology).map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> =
                format!("Failed to create device mesh: {}", e).into();
            err
        })?;

    log::info!(
        "✓ Device Mesh initialized: {} devices, topology: {:?}",
        device_mesh.world_size,
        device_mesh.topology
    );

    if use_pipeline {
        if let Some(stages) = &pipeline_stages {
            if stages.len() > device_mesh.world_size {
                return Err(format!(
                    "Pipeline stages ({}) exceed available devices ({})",
                    stages.len(),
                    device_mesh.world_size
                )
                .into());
            }
        }
    }

    // Create engines for each device in the mesh
    let mut engines: Vec<EngineHandle> = Vec::new();
    let mut swap_handles: Vec<EngineHandle> = Vec::new();
    let worker = if isolate_process {
        match spawn_worker_process(
            model_id,
            &absolute_path,
            batch_size,
            scheduler_queue_size,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
            worker_topology,
            worker_tp_degree,
            &onnx_tuning,
        ) {
            Ok(worker) => match wait_for_worker_ready_async(&worker, Duration::from_secs(30)).await
            {
                Ok(()) => Some(start_worker_with_supervisor(Arc::new(worker))),
                Err(e) => {
                    worker.kill();
                    if isolate_strict {
                        return Err(format!(
                            "Model {} requires process isolation but the worker was not ready: {}",
                            model_id, e
                        )
                        .into());
                    }
                    log::warn!(
                        "Model {} requested process isolation, but worker was not ready; falling back to in-process load (ISOLATION GUARANTEE DROPPED): {}",
                        model_id,
                        e
                    );
                    None
                }
            },
            Err(e) => {
                if isolate_strict {
                    return Err(format!(
                        "Model {} requires process isolation but the worker failed to spawn: {}",
                        model_id, e
                    )
                    .into());
                }
                log::warn!(
                    "Model {} requested process isolation, but worker spawn failed; falling back to in-process load (ISOLATION GUARANTEE DROPPED): {}",
                    model_id,
                    e
                );
                None
            }
        }
    } else {
        None
    };

    if use_pipeline {
        if let Some(worker) = &worker {
            let backend = RemoteEngine::new(model_id, worker.clone());
            let monitored_backend = MonitoringMiddleware::new_with_metrics(
                backend,
                model_id.to_string(),
                loader.manifest.version.clone(),
                shared_metrics.clone(),
            );
            let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
            let engine_arc: EngineHandle = Arc::from(engine_box);
            swap_handles.push(engine_arc.clone());
            engines.push(engine_arc);
        } else {
            let device_ids: Vec<i32> = (0..device_mesh.world_size)
                .filter_map(|rank| device_mesh.get_device(rank))
                .map(|d| d.id as i32)
                .collect();
            let provider_policy = provider_policy();
            let mut backend = if provider_policy == "manifest" {
                LLMBackend::with_devices(logical_provider.clone(), device_ids.clone())
            } else {
                LLMBackend::with_device_ids(device_ids.clone())
            };
            #[cfg(feature = "gpu-device-pool")]
            let device_memory_admissions = {
                let mut admissions = Vec::new();
                for &device_id in &device_ids {
                    if let Some(admission) = shared_kv.begin_device_memory_admission(
                        device_id as usize,
                        model_id,
                        engine_kind,
                    )? {
                        admissions.push(admission);
                    }
                }
                backend = backend.with_env_allocators(
                    device_ids
                        .iter()
                        .any(|&device_id| shared_kv.uses_env_allocators(device_id as usize)),
                );
                admissions
            };
            let primary_device = device_ids.first().copied().unwrap_or(0) as usize;
            let (kv_pool, kv_blocks_cap, global_sched, sched_engine_id, live_cap) =
                shared_kv.attach_engine(primary_device, model_id, priority_weight);
            backend = backend
                .with_shared_pool(kv_pool)
                .with_kv_blocks_cap(kv_blocks_cap)
                .with_global_scheduler(global_sched, sched_engine_id)
                .with_live_kv_cap(live_cap)
                .with_on_engine_death({
                    let sk = shared_kv.clone();
                    std::sync::Arc::new(move |eid| sk.detach_engine(eid))
                });
            backend.load(&model_file_path).await.map_err(|e| {
                let err: Box<dyn std::error::Error + Send + Sync> =
                    format!("Failed to load pipeline model {}: {}", model_id, e).into();
                err
            })?;
            #[cfg(feature = "gpu-device-pool")]
            for admission in device_memory_admissions {
                admission.commit();
            }

            let monitored_backend = MonitoringMiddleware::new_with_metrics(
                backend,
                model_id.to_string(),
                loader.manifest.version.clone(),
                shared_metrics.clone(),
            );
            let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
            let engine_arc: EngineHandle = Arc::from(engine_box);
            swap_handles.push(engine_arc.clone());
            engines.push(engine_arc);
        }
    } else {
        // We need to iterate over ranks in the mesh
        for rank in 0..device_mesh.world_size {
            if let Some(device) = device_mesh.get_device(rank) {
                if let Some(worker) = &worker {
                    let backend = RemoteEngine::new(model_id, worker.clone());
                    let monitored_backend = MonitoringMiddleware::new_with_metrics(
                        backend,
                        model_id.to_string(),
                        loader.manifest.version.clone(),
                        shared_metrics.clone(),
                    );
                    let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
                    let engine_arc: EngineHandle = Arc::from(engine_box);
                    swap_handles.push(engine_arc.clone());
                    engines.push(engine_arc);
                    continue;
                }
                #[cfg(feature = "gpu-device-pool")]
                let device_memory_admission =
                    shared_kv.begin_device_memory_admission(device.id, model_id, engine_kind)?;

                // Create a backend as a client of the runtime-owned pool when enabled.
                let mut backend = create_runtime_backend_for_device(
                    &loader.manifest,
                    &logical_provider,
                    device.id,
                    device_info,
                    &onnx_tuning,
                    &shared_kv,
                    model_id,
                )?;

                backend.load(&model_file_path).await.map_err(|e| {
                    let err: Box<dyn std::error::Error + Send + Sync> = format!(
                        "Failed to load model {} on device {}: {}",
                        model_id, device.id, e
                    )
                    .into();
                    err
                })?;
                #[cfg(feature = "gpu-device-pool")]
                if let Some(admission) = device_memory_admission {
                    admission.commit();
                }

                let monitored_backend = MonitoringMiddleware::new_with_metrics(
                    backend,
                    model_id.to_string(),
                    loader.manifest.version.clone(),
                    shared_metrics.clone(),
                );

                let engine_box: Box<dyn kapsl_engine_api::Engine> = Box::new(monitored_backend);
                let engine_arc: EngineHandle = Arc::from(engine_box);
                swap_handles.push(engine_arc.clone());
                engines.push(engine_arc);
            }
        }
    }

    log::info!("✓ Loaded {} engine instances", engines.len());

    // Determine device/provider string for registry
    let device_str = device_info.get_best_provider().to_string();
    let optimization_level = loader
        .manifest
        .hardware_requirements
        .graph_optimization_level
        .clone()
        .unwrap_or_else(|| "basic".to_string());

    // Register model in the model registry
    let model_info = ModelInfo::new(
        model_id,
        loader.manifest.project_name.clone(),
        loader.manifest.version.clone(),
        loader.manifest.framework.clone(),
        device_str,
        optimization_level,
        absolute_path.to_string_lossy().to_string(), // Store absolute package path
    );
    model_registry.upsert(model_info);

    // Create Scheduler with the device mesh
    let scheduler = Arc::new(
        Scheduler::new(
            engines,
            batch_size,
            1, // workers per device
            scheduler_queue_size,
            true, // enable fallback
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
            Some(Arc::new(device_mesh)),
        )
        .with_queue_overflow_policy(queue_overflow_policy),
    );

    // Create ReplicaPool and add the primary scheduler
    let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
    pool.add_replica(0, scheduler);

    log::info!("✓ Scheduler started for Model ID {}\n", model_id);

    Ok((Arc::new(pool), swap_handles))
}

/// Scale up a model by adding a new replica
#[allow(clippy::too_many_arguments)]
pub(crate) async fn scale_up_model(
    base_model_id: u32,
    replica_id: u32,
    unique_id: u32,
    model_path: &PathBuf,
    device_info: &DeviceInfo,
    shared_kv: SharedKvState,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    topology: &str,
    tp_degree: usize,
    model_registry: &ModelRegistry,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    onnx_tuning: OnnxRuntimeTuning,
) -> Result<(Arc<Scheduler>, EngineHandle), Box<dyn std::error::Error + Send + Sync>> {
    log::info!(
        "Scaling up Model ID {} - Creating replica #{}",
        base_model_id,
        replica_id
    );

    let absolute_path = model_path
        .canonicalize()
        .map_err(|e| format!("Invalid model path {:?}: {}", model_path, e))?;

    let loader = resolve_package_loader(&absolute_path, base_model_id)?;
    let model_file_path = loader.get_model_path();
    let queue_overflow_policy = resolve_queue_overflow_policy(&loader.manifest);
    log_queue_policy_caveat(queue_overflow_policy);
    let (scheduler_max_micro_batch, scheduler_queue_delay_ms) =
        resolve_scheduler_tuning_for_framework(
            &loader.manifest,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
        );
    let priority_weight = resolve_model_priority_weight(&loader.manifest, base_model_id);
    #[cfg(feature = "gpu-device-pool")]
    let engine_kind = EngineKind::resolve(&loader.manifest);
    let pipeline_stages = manifest_llm_pipeline_stages(&loader.manifest);
    let EffectiveTopologyChoice {
        mesh_topology: _,
        worker_topology,
        worker_tp_degree,
        use_pipeline_backend,
    } = resolve_effective_topology_choice(
        &loader.manifest,
        topology,
        tp_degree,
        pipeline_stages.as_deref(),
    );

    BackendFactory::validate_requirements(&loader.manifest.hardware_requirements, device_info)
        .map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> = format!(
                "Requirements validation failed for replica {}: {}",
                replica_id, e
            )
            .into();
            err
        })?;

    let isolate_process = resolve_isolate_process(&loader.manifest);
    let isolate_strict = resolve_isolate_process_strict(&loader.manifest);
    let worker = if isolate_process {
        match spawn_worker_process(
            unique_id,
            &absolute_path,
            batch_size,
            scheduler_queue_size,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
            worker_topology,
            worker_tp_degree,
            &onnx_tuning,
        ) {
            Ok(worker) => {
                let worker = Arc::new(worker);
                match wait_for_worker_ready_async(worker.as_ref(), Duration::from_secs(30)).await {
                    Ok(()) => Some(start_worker_with_supervisor(worker)),
                    Err(e) => {
                        worker.kill();
                        if isolate_strict {
                            return Err(format!(
                                "Replica {} requires process isolation but the worker was not ready: {}",
                                replica_id, e
                            )
                            .into());
                        }
                        log::warn!(
                            "Replica {} requested process isolation, but worker was not ready; falling back to in-process load (ISOLATION GUARANTEE DROPPED): {}",
                            replica_id,
                            e
                        );
                        None
                    }
                }
            }
            Err(e) => {
                if isolate_strict {
                    return Err(format!(
                        "Replica {} requires process isolation but the worker failed to spawn: {}",
                        replica_id, e
                    )
                    .into());
                }
                log::warn!(
                    "Replica {} requested process isolation, but worker spawn failed; falling back to in-process load (ISOLATION GUARANTEE DROPPED): {}",
                    replica_id,
                    e
                );
                None
            }
        }
    } else {
        None
    };

    let device = device_info.get_best_provider();
    let optimization_level = loader
        .manifest
        .hardware_requirements
        .graph_optimization_level
        .clone()
        .unwrap_or_else(|| "basic".to_string());

    let model_info = ModelInfo::new_replica(
        unique_id,
        replica_id,
        base_model_id,
        loader.manifest.project_name.clone(),
        loader.manifest.version.clone(),
        loader.manifest.framework.clone(),
        device.to_string(),
        optimization_level,
        absolute_path.to_string_lossy().to_string(),
    );
    model_registry.upsert(model_info);

    let engine: Arc<dyn kapsl_engine_api::Engine> = if let Some(worker) = worker {
        let backend = RemoteEngine::new(unique_id, worker);
        let monitored_backend = MonitoringMiddleware::new_with_metrics(
            backend,
            base_model_id.to_string(),
            loader.manifest.version.clone(),
            shared_metrics.clone(),
        );
        Arc::new(monitored_backend)
    } else if use_pipeline_backend {
        let pipeline_selection =
            select_mesh_devices(&loader.manifest.hardware_requirements, device_info).map_err(
                |e| {
                    let err: Box<dyn std::error::Error + Send + Sync> = format!(
                        "Failed to select devices for pipeline replica {}: {}",
                        replica_id, e
                    )
                    .into();
                    err
                },
            )?;
        let device_ids: Vec<i32> = pipeline_selection
            .devices
            .iter()
            .map(|d| d.id as i32)
            .collect();
        let provider_policy = provider_policy();
        let mut backend = if provider_policy == "manifest" {
            LLMBackend::with_devices(pipeline_selection.logical_provider, device_ids.clone())
        } else {
            LLMBackend::with_device_ids(device_ids.clone())
        };
        #[cfg(feature = "gpu-device-pool")]
        let device_memory_admissions = {
            let mut admissions = Vec::new();
            for &device_id in &device_ids {
                if let Some(admission) = shared_kv.begin_device_memory_admission(
                    device_id as usize,
                    base_model_id,
                    engine_kind,
                )? {
                    admissions.push(admission);
                }
            }
            backend = backend.with_env_allocators(
                device_ids
                    .iter()
                    .any(|&device_id| shared_kv.uses_env_allocators(device_id as usize)),
            );
            admissions
        };
        let primary_device = device_ids.first().copied().unwrap_or(0) as usize;
        let (kv_pool, kv_blocks_cap, global_sched, sched_engine_id, live_cap) =
            shared_kv.attach_engine(primary_device, base_model_id, priority_weight);
        backend = backend
            .with_shared_pool(kv_pool)
            .with_kv_blocks_cap(kv_blocks_cap)
            .with_global_scheduler(global_sched, sched_engine_id)
            .with_live_kv_cap(live_cap)
            .with_on_engine_death({
                let sk = shared_kv.clone();
                std::sync::Arc::new(move |eid| sk.detach_engine(eid))
            });
        backend.load(&model_file_path).await.map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> =
                format!("Failed to load pipeline replica {}: {}", replica_id, e).into();
            err
        })?;
        #[cfg(feature = "gpu-device-pool")]
        for admission in device_memory_admissions {
            admission.commit();
        }

        let monitored_backend = MonitoringMiddleware::new_with_metrics(
            backend,
            base_model_id.to_string(),
            loader.manifest.version.clone(),
            shared_metrics.clone(),
        );
        Arc::new(monitored_backend)
    } else {
        #[cfg(feature = "gpu-device-pool")]
        let device_id = loader.manifest.hardware_requirements.device_id.unwrap_or(0) as usize;
        #[cfg(feature = "gpu-device-pool")]
        let device_memory_admission =
            shared_kv.begin_device_memory_admission(device_id, base_model_id, engine_kind)?;
        let mut backend = create_runtime_best_backend(
            &loader.manifest,
            device_info,
            &onnx_tuning,
            &shared_kv,
            base_model_id,
        )?;
        backend.load(&model_file_path).await.map_err(|e| {
            let err: Box<dyn std::error::Error + Send + Sync> =
                format!("Failed to load replica {}: {}", replica_id, e).into();
            err
        })?;
        #[cfg(feature = "gpu-device-pool")]
        if let Some(admission) = device_memory_admission {
            admission.commit();
        }
        let monitored_backend = MonitoringMiddleware::new_with_metrics(
            backend,
            base_model_id.to_string(),
            loader.manifest.version.clone(),
            shared_metrics.clone(),
        );
        Arc::new(monitored_backend)
    };
    let swap_handle = engine.clone();
    let scheduler = Arc::new(
        Scheduler::new(
            vec![engine],
            batch_size,
            1,
            scheduler_queue_size,
            true,
            scheduler_max_micro_batch,
            scheduler_queue_delay_ms,
            None,
        )
        .with_queue_overflow_policy(queue_overflow_policy),
    );

    log::info!(
        "✓ Replica #{} started for Model ID {}",
        replica_id,
        base_model_id
    );

    Ok((scheduler, swap_handle))
}

/// Scale down a model by removing a replica
pub(crate) async fn scale_down_model(
    base_model_id: u32,
    replica_id: u32,
    unique_id: u32,
    model_registry: &ModelRegistry,
    replica_pools: &ReplicaPools,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    log::info!(
        "Scaling down Model ID {} - Removing replica #{}",
        base_model_id,
        replica_id
    );

    // Update status to Stopping
    if let Err(e) = model_registry.set_status(unique_id, ModelStatus::Stopping) {
        log::error!("Failed to set status to Stopping for {}: {}", unique_id, e);
    }

    // Remove replica from pool
    let pool = replica_pools.read().get(&base_model_id).cloned();
    if let Some(pool) = pool {
        let removed = pool.remove_replica(replica_id);
        if !removed {
            let _ = model_registry.set_status(unique_id, ModelStatus::Active);
            return Err(format!(
                "Replica #{} was not present in pool for model {}",
                replica_id, base_model_id
            )
            .into());
        }
    } else {
        let _ = model_registry.set_status(unique_id, ModelStatus::Active);
        return Err(format!("Replica pool not found for model {}", base_model_id).into());
    }

    // Update status to Inactive
    if let Err(e) = model_registry.set_status(unique_id, ModelStatus::Inactive) {
        log::error!("Failed to set status to Inactive for {}: {}", unique_id, e);
    }

    log::info!(
        "✓ Replica #{} stopped for Model ID {}",
        replica_id,
        base_model_id
    );

    Ok(())
}

pub(crate) fn force_stop_model_before_remove(
    base_model_id: u32,
    replicas: &[ModelInfo],
    model_registry: &ModelRegistry,
    replica_pools: &ReplicaPools,
    swap_map: &Arc<RwLock<HashMap<u32, Vec<EngineHandle>>>>,
    shared_kv: &SharedKvState,
) {
    for replica in replicas {
        if let Err(e) = model_registry.set_status(replica.id, ModelStatus::Stopping) {
            log::warn!(
                "Failed to set model {} replica {} to Stopping before remove: {}",
                base_model_id,
                replica.replica_id,
                e
            );
        }
    }

    if let Some(pool) = replica_pools.read().get(&base_model_id).cloned() {
        for replica in replicas {
            if !pool.remove_replica(replica.replica_id) {
                log::debug!(
                    "Replica {} for model {} was not present in the pool during remove",
                    replica.replica_id,
                    base_model_id
                );
            }
        }
    }

    replica_pools.write().remove(&base_model_id);
    swap_map.write().remove(&base_model_id);
    shared_kv.detach_engine_for_model(base_model_id);
    #[cfg(feature = "gpu-device-pool")]
    shared_kv.release_device_memory(base_model_id);

    for replica in replicas {
        if let Err(e) = model_registry.set_status(replica.id, ModelStatus::Inactive) {
            log::warn!(
                "Failed to set model {} replica {} to Inactive before remove: {}",
                base_model_id,
                replica.replica_id,
                e
            );
        }
    }
}

pub(crate) const MEMORY_HEADROOM_FRACTION: f64 = 0.80;

pub(crate) fn cap_scale_up_target_by_memory_headroom(
    current_replicas: u32,
    proposed_target: u32,
    total_model_memory_bytes: usize,
    system_total_memory_kb: u64,
) -> u32 {
    if proposed_target <= current_replicas
        || current_replicas == 0
        || total_model_memory_bytes == 0
        || system_total_memory_kb == 0
    {
        return proposed_target;
    }

    let per_replica_bytes = total_model_memory_bytes as f64 / current_replicas as f64;
    if per_replica_bytes <= 0.0 {
        return proposed_target;
    }

    let budget_bytes = (system_total_memory_kb as f64 * 1024.0 * MEMORY_HEADROOM_FRACTION).max(1.0);
    let max_by_headroom = (budget_bytes / per_replica_bytes).floor() as u32;
    let capped_max = max_by_headroom.max(current_replicas).max(1);
    proposed_target.min(capped_max)
}
