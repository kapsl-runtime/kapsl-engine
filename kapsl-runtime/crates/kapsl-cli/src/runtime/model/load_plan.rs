use super::*;

/// Immutable inputs derived once before any backend allocation starts.
/// Primary loads and scale-up replicas consume the same package, policy, and
/// topology decisions so those paths cannot silently drift apart.
pub(super) struct ModelLoadPlan {
    pub(super) base_model_id: u32,
    pub(super) runtime_model_id: u32,
    pub(super) replica_id: u32,
    pub(super) absolute_path: PathBuf,
    pub(super) loader: PackageLoader,
    pub(super) model_file_path: PathBuf,
    pub(super) batch_size: usize,
    pub(super) scheduler_queue_size: usize,
    pub(super) scheduler_max_micro_batch: usize,
    pub(super) scheduler_queue_delay_ms: u64,
    pub(super) queue_overflow_policy: kapsl_scheduler::QueueOverflowPolicy,
    pub(super) priority_weight: u32,
    pub(super) pipeline_stages: Option<Vec<String>>,
    pub(super) mesh_topology: kapsl_hal::device_mesh::MeshTopology,
    pub(super) worker_topology: &'static str,
    pub(super) worker_tp_degree: usize,
    pub(super) use_pipeline_backend: bool,
    pub(super) isolate_process: bool,
    pub(super) isolate_strict: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_model_load_plan(
    base_model_id: u32,
    runtime_model_id: u32,
    replica_id: u32,
    model_path: &Path,
    device_info: &DeviceInfo,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    topology: &str,
    tp_degree: usize,
) -> Result<ModelLoadPlan, Box<dyn std::error::Error + Send + Sync>> {
    let absolute_path = model_path.canonicalize().map_err(|error| {
        format!(
            "Invalid model path {:?}: {} (CWD: {:?})",
            model_path,
            error,
            std::env::current_dir().unwrap_or_default()
        )
    })?;
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
    let pipeline_stages = manifest_llm_pipeline_stages(&loader.manifest);
    let EffectiveTopologyChoice {
        mesh_topology,
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
        .map_err(|error| {
            format!(
                "Requirements validation failed for model {} replica {}: {}",
                base_model_id, replica_id, error
            )
        })?;
    export_gguf_auto_sizing_hint(
        &loader.manifest,
        batch_size,
        Some(model_file_path.as_path()),
    );

    Ok(ModelLoadPlan {
        base_model_id,
        runtime_model_id,
        replica_id,
        absolute_path,
        model_file_path,
        batch_size,
        scheduler_queue_size,
        scheduler_max_micro_batch,
        scheduler_queue_delay_ms,
        queue_overflow_policy,
        priority_weight,
        pipeline_stages,
        mesh_topology,
        worker_topology,
        worker_tp_degree,
        use_pipeline_backend,
        isolate_process: resolve_isolate_process(&loader.manifest),
        isolate_strict: resolve_isolate_process_strict(&loader.manifest),
        loader,
    })
}

pub(super) async fn start_isolated_worker(
    plan: &ModelLoadPlan,
    onnx_tuning: &OnnxRuntimeTuning,
) -> Result<Option<Arc<WorkerProcess>>, Box<dyn std::error::Error + Send + Sync>> {
    if !plan.isolate_process {
        return Ok(None);
    }

    log::info!(
        "✓ Process isolation enabled for Model ID {} replica #{} (strict={})",
        plan.base_model_id,
        plan.replica_id,
        plan.isolate_strict
    );
    match spawn_worker_process(
        plan.runtime_model_id,
        &plan.absolute_path,
        plan.batch_size,
        plan.scheduler_queue_size,
        plan.scheduler_max_micro_batch,
        plan.scheduler_queue_delay_ms,
        plan.worker_topology,
        plan.worker_tp_degree,
        onnx_tuning,
    ) {
        Ok(worker) => {
            let worker = Arc::new(worker);
            match wait_for_worker_ready_async(worker.as_ref(), Duration::from_secs(30)).await {
                Ok(()) => Ok(Some(start_worker_with_supervisor(worker))),
                Err(error) => {
                    worker.kill();
                    if plan.isolate_strict {
                        Err(format!(
                            "Model {} replica {} requires process isolation but the worker was not ready: {}",
                            plan.base_model_id, plan.replica_id, error
                        )
                        .into())
                    } else {
                        log::warn!(
                            "Model {} replica {} requested process isolation, but worker was not ready; falling back to in-process load (ISOLATION GUARANTEE DROPPED): {}",
                            plan.base_model_id,
                            plan.replica_id,
                            error
                        );
                        Ok(None)
                    }
                }
            }
        }
        Err(error) if plan.isolate_strict => Err(format!(
            "Model {} replica {} requires process isolation but the worker failed to spawn: {}",
            plan.base_model_id, plan.replica_id, error
        )
        .into()),
        Err(error) => {
            log::warn!(
                "Model {} replica {} requested process isolation, but worker spawn failed; falling back to in-process load (ISOLATION GUARANTEE DROPPED): {}",
                plan.base_model_id,
                plan.replica_id,
                error
            );
            Ok(None)
        }
    }
}
