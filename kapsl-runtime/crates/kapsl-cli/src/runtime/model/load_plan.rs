use super::*;

/// Immutable inputs derived once before any backend allocation starts.
/// Primary loads and scale-up replicas consume the same package, policy, and
/// topology decisions so those paths cannot silently drift apart.
pub(crate) struct ModelLoadPlan {
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
    #[cfg(feature = "gpu-device-pool")]
    pub(super) onnx_peak_concurrency: usize,
    pub(super) isolate_process: bool,
    pub(super) isolate_strict: bool,
}

impl ModelLoadPlan {
    pub(crate) fn base_model_id(&self) -> u32 {
        self.base_model_id
    }
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
    onnx_tuning: &OnnxRuntimeTuning,
) -> Result<ModelLoadPlan, Box<dyn std::error::Error + Send + Sync>> {
    #[cfg(not(feature = "gpu-device-pool"))]
    let _ = onnx_tuning;
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
        #[cfg(feature = "gpu-device-pool")]
        onnx_peak_concurrency: if EngineKind::resolve(&loader.manifest).is_onnx_generate() {
            1
        } else {
            onnx_tuning.peak_concurrency_hint.unwrap_or(1).max(1) as usize
        },
        isolate_process: resolve_isolate_process(&loader.manifest),
        isolate_strict: resolve_isolate_process_strict(&loader.manifest),
        loader,
    })
}

#[cfg(feature = "gpu-device-pool")]
pub(crate) fn device_memory_bootstrap_plan<'a>(
    plans: impl IntoIterator<Item = &'a ModelLoadPlan>,
    device_info: &DeviceInfo,
) -> Result<DeviceMemoryBootstrapPlan, String> {
    let mut bootstrap = DeviceMemoryBootstrapPlan::default();
    for plan in plans {
        let selection =
            select_mesh_devices(&plan.loader.manifest.hardware_requirements, device_info)?;
        let cuda_device_ids: Vec<_> = selection
            .devices
            .iter()
            .filter(|device| device.backend.to_string().eq_ignore_ascii_case("cuda"))
            .map(|device| device.id)
            .collect();
        if cuda_device_ids.is_empty() {
            continue;
        }
        if plan.isolate_process {
            // Do not let a new implicit parent allocation starve the child
            // before it loads. Non-strict fallback remains safe: it loads
            // in-process without a runtime-owned pool if worker startup fails.
            for &device_id in &cuda_device_ids {
                bootstrap.mark_isolated_worker(device_id);
            }
            continue;
        }

        let kind = EngineKind::resolve(&plan.loader.manifest);
        let wants_pool = kind.uses_onnx_session()
            || (kind.is_gguf()
                && cfg!(any(
                    feature = "gguf-native",
                    feature = "gguf-cuda-shared-kv"
                )))
            || (kind == EngineKind::Native && cfg!(feature = "native"));
        if wants_pool {
            for &device_id in &cuda_device_ids {
                bootstrap.mark_pool_consumer(device_id);
            }
        }

        // ORT CUDA/TensorRT sessions use the registered environment allocator,
        // so their weights and workspaces are pool demand rather than external
        // bytes. GGUF/native weights remain outside the shared backing pool.
        if kind.uses_onnx_session() && wants_pool {
            let model_bytes = std::fs::metadata(&plan.model_file_path)
                .map_err(|error| {
                    format!(
                        "stat ONNX model {}: {error}",
                        plan.model_file_path.display()
                    )
                })?
                .len() as usize;
            let per_device_model_bytes = if plan.use_pipeline_backend {
                model_bytes.saturating_add(cuda_device_ids.len().saturating_sub(1))
                    / cuda_device_ids.len().max(1)
            } else {
                model_bytes
            };
            // ORT keeps weights plus per-session CUDA arenas/workspaces in the
            // registered allocator. Size both explicitly; serialized model
            // bytes alone are not a peak device-memory plan.
            let workspace_per_session = (per_device_model_bytes / 2).max(256 * 1024 * 1024);
            let pooled_bytes = per_device_model_bytes
                .saturating_add(workspace_per_session)
                .saturating_mul(plan.onnx_peak_concurrency);
            for &device_id in &cuda_device_ids {
                bootstrap.add_pooled_allocation(
                    device_id,
                    format!(
                        "bootstrap-onnx:{}:{}:{}",
                        plan.base_model_id, plan.replica_id, device_id
                    ),
                    pooled_bytes,
                );
            }
            continue;
        }
        let bytes = planned_external_weight_bytes(kind, &plan.model_file_path)?;
        if bytes == 0 {
            continue;
        }
        for &device_id in &cuda_device_ids {
            let allocation_id = if kind.is_gguf() && !cfg!(feature = "gguf-native") {
                // llama.cpp shares immutable weights by canonical model path.
                format!("bootstrap-gguf:{}", plan.model_file_path.display())
            } else {
                // Native backends currently allocate one weight copy per engine.
                format!(
                    "bootstrap-{}:{}:{}:{}",
                    kind.label(),
                    plan.base_model_id,
                    plan.replica_id,
                    device_id
                )
            };
            bootstrap.add_external_allocation(device_id, allocation_id, bytes);
            // GGUF/native execution scratch is not served by the KV pool.
            // Reserve a backend-specific load/context workspace estimate so
            // automatic pool sizing cannot consume that headroom first.
            let scratch_bytes = (bytes / 8).max(256 * 1024 * 1024);
            bootstrap.add_external_allocation(
                device_id,
                format!(
                    "bootstrap-scratch:{}:{}:{}",
                    plan.base_model_id, plan.replica_id, device_id
                ),
                scratch_bytes,
            );
        }
    }
    Ok(bootstrap)
}

#[cfg(feature = "gpu-device-pool")]
fn planned_external_weight_bytes(kind: EngineKind, model_path: &Path) -> Result<usize, String> {
    if kind == EngineKind::Native {
        // Mirror NativeBackend::planned_weight_bytes: a manifest may point at
        // one shard, but the backend loads every sibling safetensors shard.
        let model_dir = if model_path.is_dir() {
            model_path
        } else {
            model_path.parent().unwrap_or(model_path)
        };
        let mut bytes = 0usize;
        for entry in std::fs::read_dir(model_dir)
            .map_err(|error| format!("read model directory {}: {error}", model_dir.display()))?
        {
            let entry = entry.map_err(|error| {
                format!("read model entry under {}: {error}", model_dir.display())
            })?;
            if entry.path().extension().and_then(|value| value.to_str()) == Some("safetensors") {
                bytes = bytes.saturating_add(
                    entry
                        .metadata()
                        .map_err(|error| {
                            format!("stat model shard {}: {error}", entry.path().display())
                        })?
                        .len() as usize,
                );
            }
        }
        return Ok(bytes);
    }

    let bytes = std::fs::metadata(model_path)
        .map_err(|error| format!("stat model {}: {error}", model_path.display()))?
        .len() as usize;
    if kind.is_gguf() && cfg!(feature = "gguf-native") {
        // The native GGUF path may expand quantized tensors while uploading.
        // Two times the source size is a conservative host-only bootstrap bound;
        // post-load reconciliation remains authoritative.
        Ok(bytes.saturating_mul(2))
    } else {
        Ok(bytes)
    }
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

#[cfg(all(test, feature = "gpu-device-pool"))]
mod device_memory_plan_tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("kapsl-{name}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn bootstrap_weight_estimate_uses_the_gguf_file_size() {
        let root = scratch_dir("gguf-bootstrap-plan");
        std::fs::create_dir_all(&root).unwrap();
        let model = root.join("model.gguf");
        std::fs::write(&model, vec![0u8; 4096]).unwrap();

        let bytes = planned_external_weight_bytes(EngineKind::GgufGenerate, &model).unwrap();
        let expected = if cfg!(feature = "gguf-native") {
            8192
        } else {
            4096
        };
        assert_eq!(bytes, expected);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn native_directory_estimate_sums_only_safetensor_shards() {
        let root = scratch_dir("native-bootstrap-plan");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model-1.safetensors"), vec![0u8; 100]).unwrap();
        std::fs::write(root.join("model-2.safetensors"), vec![0u8; 250]).unwrap();
        std::fs::write(root.join("tokenizer.json"), vec![0u8; 999]).unwrap();

        assert_eq!(
            planned_external_weight_bytes(EngineKind::Native, &root).unwrap(),
            350
        );
        assert_eq!(
            planned_external_weight_bytes(EngineKind::Native, &root.join("model-1.safetensors"))
                .unwrap(),
            350,
            "a manifest pointing at one shard must account every sibling shard"
        );

        std::fs::remove_dir_all(root).unwrap();
    }
}
