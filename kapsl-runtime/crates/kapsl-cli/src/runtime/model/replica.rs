use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReplicaLoadRole {
    Primary,
    Autoscaled,
}

impl ReplicaLoadRole {
    fn is_primary(self) -> bool {
        self == Self::Primary
    }

    fn selection_context(self, plan: &ModelLoadPlan) -> String {
        match self {
            Self::Primary => format!("model {}", plan.base_model_id),
            Self::Autoscaled => format!("model {} replica {}", plan.base_model_id, plan.replica_id),
        }
    }

    fn load_context(self, plan: &ModelLoadPlan, device_id: Option<usize>) -> String {
        match (self, plan.use_pipeline_backend, device_id) {
            (Self::Primary, true, _) => {
                format!("Failed to load pipeline model {}", plan.base_model_id)
            }
            (Self::Primary, false, Some(device_id)) => format!(
                "Failed to load model {} on device {}",
                plan.base_model_id, device_id
            ),
            (Self::Autoscaled, true, _) => {
                format!("Failed to load pipeline replica {}", plan.replica_id)
            }
            (Self::Autoscaled, false, _) => {
                format!("Failed to load replica {}", plan.replica_id)
            }
            (Self::Primary, false, None) => {
                format!("Failed to load model {}", plan.base_model_id)
            }
        }
    }
}

pub(super) struct LoadedReplica {
    pub(super) scheduler: Arc<Scheduler>,
    pub(super) swap_handles: Vec<EngineHandle>,
    pub(super) model_info: ModelInfo,
}

/// Turn one immutable load plan into a fully loaded scheduler replica.
///
/// Primary and autoscaled loads share backend construction, memory admission,
/// loading, reconciliation, monitoring, and scheduler policy. `role` captures
/// only their intentional layout difference: a primary non-pipeline scheduler
/// spans its selected mesh, while one autoscaled replica remains one engine.
pub(super) async fn load_replica(
    plan: ModelLoadPlan,
    role: ReplicaLoadRole,
    device_info: &DeviceInfo,
    resources: Arc<RuntimeResources>,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    onnx_tuning: &OnnxRuntimeTuning,
) -> Result<LoadedReplica, DynError> {
    log_package_plan(&plan);

    let needs_mesh = role.is_primary() || plan.use_pipeline_backend;
    let (mut device_mesh, mut logical_provider) = if needs_mesh {
        use kapsl_hal::device_mesh::DeviceMesh;

        let selected =
            select_mesh_devices(&plan.loader.manifest.hardware_requirements, device_info).map_err(
                |error| {
                    format!(
                        "Failed to select devices for {}: {}",
                        role.selection_context(&plan),
                        error
                    )
                },
            )?;
        let logical_provider = selected.logical_provider;
        let mesh = DeviceMesh::with_topology(selected.devices, plan.mesh_topology.clone())
            .map_err(|error| format!("Failed to create device mesh: {error}"))?;

        log::info!(
            "✓ Device Mesh initialized: {} devices, topology: {:?}",
            mesh.world_size,
            mesh.topology
        );
        if plan.use_pipeline_backend {
            if let Some(stages) = plan.pipeline_stages.as_ref() {
                if stages.len() > mesh.world_size {
                    return Err(format!(
                        "Pipeline stages ({}) exceed available devices ({})",
                        stages.len(),
                        mesh.world_size
                    )
                    .into());
                }
            }
        }

        (Some(mesh), Some(logical_provider))
    } else {
        (None, None)
    };

    let worker = start_isolated_worker(&plan, onnx_tuning).await?;
    let mut engines = if let Some(worker) = worker {
        let engine_count = if role.is_primary() && !plan.use_pipeline_backend {
            device_mesh
                .as_ref()
                .map(|mesh| mesh.world_size)
                .unwrap_or(1)
        } else {
            1
        };
        (0..engine_count)
            .map(|_| {
                monitor_runtime_backend(
                    Box::new(RemoteEngine::new(plan.runtime_model_id, worker.clone())),
                    plan.base_model_id,
                    &plan.loader.manifest.version,
                    shared_metrics,
                )
            })
            .collect()
    } else if plan.use_pipeline_backend {
        let mesh = device_mesh
            .as_ref()
            .expect("pipeline loads always initialize a device mesh");
        let device_ids: Vec<usize> = (0..mesh.world_size)
            .filter_map(|rank| mesh.get_device(rank))
            .map(|device| device.id)
            .collect();
        let memory_domains: Vec<_> = (0..mesh.world_size)
            .filter_map(|rank| mesh.get_device(rank))
            .map(|device| MemoryDomain::for_provider(&device.backend.to_string(), device.id))
            .collect();
        let backend = create_pipeline_backend(
            &plan,
            logical_provider
                .as_deref()
                .expect("pipeline loads always select a provider"),
            &device_ids,
            &resources,
        );
        let backend = load_runtime_backend(
            backend,
            &plan.model_file_path,
            &memory_domains,
            &resources,
            plan.base_model_id,
            plan.replica_id,
            EngineKind::resolve(&plan.loader.manifest),
            plan.priority_weight,
            &role.load_context(&plan, None),
        )
        .await?;
        vec![monitor_runtime_backend(
            backend,
            plan.base_model_id,
            &plan.loader.manifest.version,
            shared_metrics,
        )]
    } else if role.is_primary() {
        let mesh = device_mesh
            .as_ref()
            .expect("primary loads always initialize a device mesh");
        let provider = logical_provider
            .as_deref()
            .expect("primary loads always select a provider");
        let mut engines = Vec::with_capacity(mesh.world_size);
        for rank in 0..mesh.world_size {
            let Some(device) = mesh.get_device(rank) else {
                continue;
            };
            let backend = create_runtime_backend_for_device(
                &plan.loader.manifest,
                provider,
                device.id,
                device_info,
                onnx_tuning,
                &resources,
                plan.base_model_id,
                plan.replica_id,
            )?;
            let backend = load_runtime_backend(
                backend,
                &plan.model_file_path,
                &[MemoryDomain::for_provider(
                    &device.backend.to_string(),
                    device.id,
                )],
                &resources,
                plan.base_model_id,
                plan.replica_id,
                EngineKind::resolve(&plan.loader.manifest),
                plan.priority_weight,
                &role.load_context(&plan, Some(device.id)),
            )
            .await?;
            engines.push(monitor_runtime_backend(
                backend,
                plan.base_model_id,
                &plan.loader.manifest.version,
                shared_metrics,
            ));
        }
        engines
    } else {
        let selection =
            select_mesh_devices(&plan.loader.manifest.hardware_requirements, device_info);
        let selection = selection.map_err(|error| {
            format!(
                "Failed to select a device for {}: {}",
                role.selection_context(&plan),
                error
            )
        })?;
        logical_provider = Some(selection.logical_provider.clone());
        let device_id = plan
            .loader
            .manifest
            .hardware_requirements
            .device_id
            .unwrap_or(0) as usize;
        let backend = create_runtime_best_backend(
            &plan.loader.manifest,
            device_info,
            onnx_tuning,
            &resources,
            plan.base_model_id,
            plan.replica_id,
        )?;
        let backend = load_runtime_backend(
            backend,
            &plan.model_file_path,
            &[MemoryDomain::for_provider(
                &selection.logical_provider,
                device_id,
            )],
            &resources,
            plan.base_model_id,
            plan.replica_id,
            EngineKind::resolve(&plan.loader.manifest),
            plan.priority_weight,
            &role.load_context(&plan, Some(device_id)),
        )
        .await?;
        vec![monitor_runtime_backend(
            backend,
            plan.base_model_id,
            &plan.loader.manifest.version,
            shared_metrics,
        )]
    };

    log::info!("✓ Loaded {} engine instances", engines.len());
    let swap_handles = engines.clone();
    let scheduler_mesh = if role.is_primary() {
        Some(Arc::new(
            device_mesh
                .take()
                .expect("primary loads always initialize a device mesh"),
        ))
    } else {
        None
    };
    let scheduler = Arc::new(
        Scheduler::new(
            std::mem::take(&mut engines),
            plan.batch_size,
            1,
            plan.scheduler_queue_size,
            true,
            plan.scheduler_max_micro_batch,
            plan.scheduler_queue_delay_ms,
            scheduler_mesh,
        )
        .with_queue_overflow_policy(plan.queue_overflow_policy),
    );

    let selected_provider =
        logical_provider.expect("successful model loads always select an execution provider");
    Ok(LoadedReplica {
        scheduler,
        swap_handles,
        model_info: model_info_for_plan(&plan, role, &selected_provider),
    })
}

fn create_pipeline_backend(
    plan: &ModelLoadPlan,
    logical_provider: &str,
    device_ids: &[usize],
    resources: &RuntimeResources,
) -> Box<dyn kapsl_engine_api::Engine> {
    let backend_device_ids: Vec<i32> = device_ids.iter().map(|&id| id as i32).collect();
    let mut backend = if provider_policy() == "manifest" {
        LLMBackend::with_devices(logical_provider.to_owned(), backend_device_ids.clone())
    } else {
        LLMBackend::with_device_ids(backend_device_ids.clone())
    }
    .with_memory_owner(plan.base_model_id, plan.replica_id);
    #[cfg(feature = "gpu-device-pool")]
    {
        backend = backend.with_env_allocators(
            device_ids
                .iter()
                .any(|&device_id| resources.uses_env_allocators(device_id)),
        );
    }

    let primary_device = device_ids.first().copied().unwrap_or(0);
    let (kv_pool, kv_blocks_cap, global_sched, sched_engine_id, live_cap) =
        resources.kv().attach_engine(
            primary_device,
            plan.base_model_id,
            plan.replica_id,
            plan.priority_weight,
        );
    backend = backend
        .with_shared_pool(kv_pool)
        .with_kv_blocks_cap(kv_blocks_cap)
        .with_global_scheduler(global_sched, sched_engine_id)
        .with_live_kv_cap(live_cap)
        .with_on_engine_death({
            let kv = resources.kv().clone();
            Arc::new(move |engine_id| kv.detach_engine(engine_id))
        });
    Box::new(backend)
}

fn model_info_for_plan(
    plan: &ModelLoadPlan,
    role: ReplicaLoadRole,
    selected_provider: &str,
) -> ModelInfo {
    let manifest = &plan.loader.manifest;
    let optimization_level = manifest
        .hardware_requirements
        .graph_optimization_level
        .clone()
        .unwrap_or_else(|| "basic".to_string());
    // The registry's path is the persisted runtime asset, whose directory also
    // contains package-side tokenizer/config files. ModelManager separately
    // retains `absolute_path` for lifecycle reloads and autoscaling.
    let path = plan.model_file_path.to_string_lossy().to_string();
    let device = selected_provider.to_string();

    let model_info = match role {
        ReplicaLoadRole::Primary => ModelInfo::new(
            plan.base_model_id,
            manifest.project_name.clone(),
            manifest.version.clone(),
            manifest.framework.clone(),
            device,
            optimization_level,
            path,
        ),
        ReplicaLoadRole::Autoscaled => ModelInfo::new_replica(
            plan.runtime_model_id,
            plan.replica_id,
            plan.base_model_id,
            manifest.project_name.clone(),
            manifest.version.clone(),
            manifest.framework.clone(),
            device,
            optimization_level,
            path,
        ),
    };

    model_info.with_model_axes(
        manifest.format.clone(),
        manifest.model_type.clone(),
        manifest.task.clone(),
        manifest.preprocess_kind(),
    )
}

fn log_package_plan(plan: &ModelLoadPlan) {
    log::info!(
        "Loading Model ID {} replica #{}: {:?}",
        plan.base_model_id,
        plan.replica_id,
        plan.absolute_path
    );
    log::info!("✓ Package loaded");
    log::info!("  Project: {}", plan.loader.manifest.project_name);
    log::info!("  Framework: {}", plan.loader.manifest.framework);
    log::info!("  Version: {}", plan.loader.manifest.version);
    log::info!(
        "  Queue overflow policy: {}",
        plan.queue_overflow_policy.as_str()
    );
    log::info!("  Priority weight: {}", plan.priority_weight);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_plan() -> ModelLoadPlan {
        let model_path = std::env::temp_dir().join("selected-provider-model.onnx");
        ModelLoadPlan {
            base_model_id: 7,
            runtime_model_id: 7,
            replica_id: 0,
            absolute_path: model_path.clone(),
            loader: PackageLoader::from_raw_file(&model_path).expect("test package loader"),
            model_file_path: model_path,
            batch_size: 1,
            scheduler_queue_size: 1,
            scheduler_max_micro_batch: 1,
            scheduler_queue_delay_ms: 0,
            queue_overflow_policy: kapsl_scheduler::QueueOverflowPolicy::Block,
            priority_weight: 1,
            pipeline_stages: None,
            mesh_topology: kapsl_hal::device_mesh::MeshTopology::DataParallel,
            worker_topology: "data-parallel",
            worker_tp_degree: 1,
            use_pipeline_backend: false,
            #[cfg(feature = "gpu-device-pool")]
            onnx_peak_concurrency: 1,
            isolate_process: false,
            isolate_strict: false,
        }
    }

    #[test]
    fn model_info_reports_selected_provider() {
        let info = model_info_for_plan(&test_plan(), ReplicaLoadRole::Primary, "cpu");

        assert_eq!(info.device, "cpu");
    }

    #[test]
    fn model_info_reports_runtime_assets_and_manifest_axes() {
        let mut plan = test_plan();
        plan.absolute_path = PathBuf::from("/source/model.aimod");
        plan.model_file_path = PathBuf::from("/cache/model.onnx");
        plan.loader.manifest.format = Some("onnx".to_string());
        plan.loader.manifest.model_type = Some("embedding".to_string());
        plan.loader.manifest.task = Some("embed".to_string());

        let info = model_info_for_plan(&plan, ReplicaLoadRole::Primary, "cpu");

        assert_eq!(info.model_path, "/cache/model.onnx");
        assert_eq!(info.format.as_deref(), Some("onnx"));
        assert_eq!(info.model_type.as_deref(), Some("embedding"));
        assert_eq!(info.task.as_deref(), Some("embed"));
    }
}
