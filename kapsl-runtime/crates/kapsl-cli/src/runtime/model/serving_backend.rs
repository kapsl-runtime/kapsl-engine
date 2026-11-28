//! One strategy per serving backend.
//!
//! A load plan names the backend that serves its model
//! ([`ResolvedServingBackend`]). Whatever the model runtime does differently
//! for that choice is asked of the backend's [`ServingBackendStrategy`]: how a
//! replica is loaded, which topology requests the backend takes as given, and
//! how the autoscaler should read a queue on it. Inference is already uniform:
//! every backend ends up as an engine behind a scheduler.
//!
//! A new backend adds a variant to [`ResolvedServingBackend`] and a strategy
//! here. The planner, the replica loader and the autoscaler do not change.

use super::*;

/// What the model runtime asks of a serving backend.
#[async_trait::async_trait]
pub(super) trait ServingBackendStrategy: Send + Sync {
    /// Value of `ModelInfo::device` on this backend's replicas. `None` for an
    /// engine inside Kapsl, which reports the execution provider it selected.
    fn device_label(&self) -> Option<&'static str> {
        None
    }

    /// Topology the backend takes from the placement request as given.
    /// `None` leaves the choice to the generic resolution.
    fn topology_override(
        &self,
        _requested_topology: &str,
        _tp_degree: usize,
    ) -> Option<EffectiveTopologyChoice> {
        None
    }

    /// Whether one replica batches concurrent requests itself. A queue on
    /// such a replica is not, on its own, a reason to start another process.
    fn batches_internally(&self) -> bool {
        false
    }

    /// Whether the model's replicas can still grow their KV capacity in place.
    fn has_in_place_kv_headroom(&self, _resources: &RuntimeResources, _model_id: u32) -> bool {
        false
    }

    /// Turn one load plan into a running replica.
    async fn load_replica(
        &self,
        plan: ModelLoadPlan,
        role: ReplicaLoadRole,
        device_info: &DeviceInfo,
        resources: Arc<RuntimeResources>,
        shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    ) -> Result<LoadedReplica, DynError>;
}

const SERVING_BACKENDS: [ResolvedServingBackend; 4] = [
    ResolvedServingBackend::Builtin,
    ResolvedServingBackend::LlamaCpp,
    ResolvedServingBackend::Vllm,
    ResolvedServingBackend::Sglang,
];

/// The strategy of a resolved backend.
pub(super) fn serving_backend_strategy(
    backend: ResolvedServingBackend,
) -> &'static dyn ServingBackendStrategy {
    match backend {
        ResolvedServingBackend::Builtin => &BuiltinBackend,
        ResolvedServingBackend::LlamaCpp => &LlamaCppBackend,
        ResolvedServingBackend::Vllm => &VllmBackend,
        ResolvedServingBackend::Sglang => &SglangBackend,
    }
}

/// The strategy of a loaded model, found by the device label its replicas
/// carry. `None` for a model served by an engine inside Kapsl.
pub(super) fn serving_backend_strategy_for_device(
    device: &str,
) -> Option<&'static dyn ServingBackendStrategy> {
    SERVING_BACKENDS
        .into_iter()
        .map(serving_backend_strategy)
        .find(|strategy| strategy.device_label() == Some(device))
}

/// Formats outside the serving-backend policy (ONNX, the native engine) and
/// packages created before `metadata.serving.backend` existed. The engine runs
/// inside Kapsl or in an isolated worker, built by the backend factory for the
/// model's format.
struct BuiltinBackend;

#[async_trait::async_trait]
impl ServingBackendStrategy for BuiltinBackend {
    async fn load_replica(
        &self,
        plan: ModelLoadPlan,
        role: ReplicaLoadRole,
        device_info: &DeviceInfo,
        resources: Arc<RuntimeResources>,
        shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    ) -> Result<LoadedReplica, DynError> {
        load_in_process_replica(plan, role, device_info, resources, shared_metrics).await
    }
}

/// llama.cpp for GGUF models. It loads like any other engine inside Kapsl:
/// the backend factory picks the llama.cpp pack or a compiled-in GGUF engine.
struct LlamaCppBackend;

#[async_trait::async_trait]
impl ServingBackendStrategy for LlamaCppBackend {
    async fn load_replica(
        &self,
        plan: ModelLoadPlan,
        role: ReplicaLoadRole,
        device_info: &DeviceInfo,
        resources: Arc<RuntimeResources>,
        shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    ) -> Result<LoadedReplica, DynError> {
        load_in_process_replica(plan, role, device_info, resources, shared_metrics).await
    }
}

/// vLLM as a supervised child server whose KV cache Kapsl provisions.
struct VllmBackend;

const VLLM_DEVICE_LABEL: &str = "vllm";

#[async_trait::async_trait]
impl ServingBackendStrategy for VllmBackend {
    fn device_label(&self) -> Option<&'static str> {
        Some(VLLM_DEVICE_LABEL)
    }

    /// A tensor-parallel request is taken as given: the ranks are vLLM's own
    /// workers, one per CUDA device, so the manifest's support for Kapsl's
    /// in-process mesh does not decide it.
    fn topology_override(
        &self,
        requested_topology: &str,
        tp_degree: usize,
    ) -> Option<EffectiveTopologyChoice> {
        if !matches!(requested_topology, "tensor-parallel" | "tensor_parallel") {
            return None;
        }
        let degree = tp_degree.max(1);
        Some(EffectiveTopologyChoice {
            mesh_topology: kapsl_hal::device_mesh::MeshTopology::TensorParallel {
                degree,
                mesh_shape: (1, degree),
            },
            worker_topology: "tensor-parallel",
            worker_tp_degree: degree,
            use_pipeline_backend: false,
        })
    }

    fn batches_internally(&self) -> bool {
        true
    }

    fn has_in_place_kv_headroom(&self, resources: &RuntimeResources, model_id: u32) -> bool {
        resources
            .managed_vllm()
            .is_some_and(|deployment| deployment.model_has_live_resize_headroom(model_id))
    }

    async fn load_replica(
        &self,
        plan: ModelLoadPlan,
        role: ReplicaLoadRole,
        device_info: &DeviceInfo,
        resources: Arc<RuntimeResources>,
        shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    ) -> Result<LoadedReplica, DynError> {
        log_ignored_process_isolation(&plan, "vLLM");
        if plan.use_pipeline_backend {
            return Err("managed vLLM does not support pipeline topology".into());
        }
        let deployment = resources.managed_vllm().ok_or_else(|| {
            "managed vLLM was selected after startup without a configured backend deployment; restart Kapsl with this model in `kapsl run --model ...`"
                .to_string()
        })?;
        let device_ids =
            select_managed_server_devices(&plan, role, device_info, plan.worker_tp_degree)?;
        let backend = ManagedVllmEngine::create(
            deployment,
            &plan.loader.manifest,
            &plan.model_file_path,
            plan.base_model_id,
            plan.replica_id,
            device_ids.clone(),
            plan.worker_tp_degree,
            plan.batch_size.max(1),
            shared_metrics.clone(),
        )?;
        let observer = Arc::new(ManagedVllmSchedulerMetrics {
            metrics: shared_metrics.clone(),
            model: plan.loader.manifest.project_name.clone(),
            replica: plan.replica_id.to_string(),
        });
        let replica = load_managed_server_replica(
            PreparedManagedServer {
                backend: backend.into(),
                device_ids,
                observer: Some(observer),
            },
            VLLM_DEVICE_LABEL,
            &plan,
            role,
            &resources,
            shared_metrics,
        )
        .await?;
        log::info!("✓ Loaded managed vLLM engine instance");
        Ok(replica)
    }
}

/// Reports how long requests wait in the Kapsl scheduler before the vLLM
/// bridge takes them.
struct ManagedVllmSchedulerMetrics {
    metrics: kapsl_monitor::metrics::KapslMetrics,
    model: String,
    replica: String,
}

impl kapsl_scheduler::SchedulerObserver for ManagedVllmSchedulerMetrics {
    fn observe_queue_wait(
        &self,
        _priority: kapsl_scheduler::Priority,
        operation: &'static str,
        elapsed: Duration,
    ) {
        self.metrics
            .managed_vllm
            .bridge_stage_seconds
            .with_label_values(&[
                self.model.as_str(),
                self.replica.as_str(),
                operation,
                "scheduler_queue",
            ])
            .observe(elapsed.as_secs_f64());
    }
}

/// SGLang as a supervised child server on a single CUDA device. Its static
/// footprint is admitted at load, and its requests lease KV capacity through
/// the KV control socket.
struct SglangBackend;

const SGLANG_DEVICE_LABEL: &str = "sglang";

#[async_trait::async_trait]
impl ServingBackendStrategy for SglangBackend {
    fn device_label(&self) -> Option<&'static str> {
        Some(SGLANG_DEVICE_LABEL)
    }

    /// SGLang reports its live KV leases as the sequences it is running.
    fn batches_internally(&self) -> bool {
        true
    }

    async fn load_replica(
        &self,
        plan: ModelLoadPlan,
        role: ReplicaLoadRole,
        device_info: &DeviceInfo,
        resources: Arc<RuntimeResources>,
        shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
    ) -> Result<LoadedReplica, DynError> {
        log_ignored_process_isolation(&plan, "SGLang");
        if plan.use_pipeline_backend || plan.worker_tp_degree != 1 {
            return Err(
                "managed SGLang runs one scheduler rank on one CUDA device; pipeline and tensor-parallel topologies are not supported"
                    .into(),
            );
        }
        let deployment = resources
            .managed_server(ResolvedServingBackend::Sglang)
            .ok_or_else(|| {
                "managed SGLang was selected after startup without a configured backend deployment; restart Kapsl with this model in `kapsl run --model ...`"
                    .to_string()
            })?;
        let device_ids = select_managed_server_devices(&plan, role, device_info, 1)?;
        let backend = create_managed_sglang_engine(
            deployment,
            &plan.loader.manifest,
            &plan.model_file_path,
            plan.base_model_id,
            plan.replica_id,
            device_ids[0],
            plan.batch_size.max(1),
        )?;
        let replica = load_managed_server_replica(
            PreparedManagedServer {
                backend: backend.into(),
                device_ids,
                observer: None,
            },
            SGLANG_DEVICE_LABEL,
            &plan,
            role,
            &resources,
            shared_metrics,
        )
        .await?;
        log::info!("✓ Loaded managed SGLang engine instance");
        Ok(replica)
    }
}

/// A managed server's engine, not yet admitted or loaded.
struct PreparedManagedServer {
    backend: PreparedBackend,
    /// CUDA devices the server's ranks run on, in rank order.
    device_ids: Vec<usize>,
    observer: Option<Arc<dyn kapsl_scheduler::SchedulerObserver>>,
}

/// A managed server is already a supervised child process, so the generic
/// process-isolation request has nothing to add.
fn log_ignored_process_isolation(plan: &ModelLoadPlan, server: &str) {
    if plan.isolate_process {
        log::info!(
            "Ignoring generic process-isolation metadata for model {}: managed {} already runs in a supervised child process",
            plan.base_model_id,
            server
        );
    }
}

/// CUDA devices for the ranks of one managed-server replica.
fn select_managed_server_devices(
    plan: &ModelLoadPlan,
    role: ReplicaLoadRole,
    device_info: &DeviceInfo,
    rank_count: usize,
) -> Result<Vec<usize>, String> {
    let selection = select_mesh_devices(&plan.loader.manifest.hardware_requirements, device_info)
        .map_err(|error| {
        format!(
            "Failed to select a CUDA device for {}: {}",
            role.selection_context(plan),
            error
        )
    })?;
    select_managed_server_device_ids(&selection.devices, rank_count, plan.replica_id)
}

/// Admit and load a managed server's engine, monitor it, and put it behind a
/// scheduler.
///
/// The server batches internally, so the scheduler holds this one engine and
/// one worker whatever the batch size.
async fn load_managed_server_replica(
    engine: PreparedManagedServer,
    device_label: &'static str,
    plan: &ModelLoadPlan,
    role: ReplicaLoadRole,
    resources: &Arc<RuntimeResources>,
    shared_metrics: &kapsl_monitor::metrics::KapslMetrics,
) -> Result<LoadedReplica, DynError> {
    let PreparedManagedServer {
        backend,
        device_ids,
        observer,
    } = engine;
    let memory_domains = device_ids
        .iter()
        .map(|device_id| MemoryDomain::Cuda {
            device_id: *device_id,
        })
        .collect::<Vec<_>>();
    let backend = load_runtime_backend(
        backend,
        &plan.model_file_path,
        &memory_domains,
        resources,
        plan.base_model_id,
        plan.replica_id,
        EngineKind::Native,
        plan.priority_weight,
        &role.load_context(plan, device_ids.first().copied()),
    )
    .await?;
    let engine = monitor_runtime_backend(
        backend,
        plan.base_model_id,
        &plan.loader.manifest.version,
        shared_metrics,
    );
    let swap_handles = vec![engine.clone()];
    let mut scheduler = Scheduler::new(
        vec![engine],
        plan.batch_size,
        1,
        plan.scheduler_queue_size,
        true,
        plan.scheduler_max_micro_batch,
        plan.scheduler_queue_delay_ms,
        None,
    )
    .with_queue_overflow_policy(plan.queue_overflow_policy);
    if let Some(observer) = observer {
        scheduler = scheduler.with_observer(observer);
    }
    Ok(LoadedReplica {
        scheduler: Arc::new(scheduler),
        swap_handles,
        model_info: model_info_for_plan(plan, role, device_label),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_backend_has_a_strategy_and_managed_servers_are_found_by_label() {
        for backend in SERVING_BACKENDS {
            let strategy = serving_backend_strategy(backend);
            // A managed server owns its GPU memory and batches internally; an
            // engine inside Kapsl does neither and reports its provider.
            assert_eq!(strategy.batches_internally(), backend.is_managed_server());
            assert_eq!(
                strategy.device_label().is_some(),
                backend.is_managed_server()
            );
            if let Some(label) = strategy.device_label() {
                assert_eq!(label, backend.as_str());
                let found = serving_backend_strategy_for_device(label).unwrap();
                assert_eq!(found.device_label(), Some(label));
            }
        }
        assert!(serving_backend_strategy_for_device("CUDAExecutionProvider").is_none());
        assert!(serving_backend_strategy_for_device("cpu").is_none());
    }

    #[test]
    fn only_vllm_takes_a_tensor_parallel_request_as_given() {
        let vllm = serving_backend_strategy(ResolvedServingBackend::Vllm);
        for spelling in ["tensor-parallel", "tensor_parallel"] {
            let choice = vllm.topology_override(spelling, 2).unwrap();
            assert_eq!(choice.worker_topology, "tensor-parallel");
            assert_eq!(choice.worker_tp_degree, 2);
            assert!(!choice.use_pipeline_backend);
        }
        // A degree of zero means one rank.
        assert_eq!(
            vllm.topology_override("tensor-parallel", 0)
                .unwrap()
                .worker_tp_degree,
            1
        );
        assert!(vllm.topology_override("pipeline", 2).is_none());
        for backend in [
            ResolvedServingBackend::Builtin,
            ResolvedServingBackend::LlamaCpp,
            ResolvedServingBackend::Sglang,
        ] {
            assert!(serving_backend_strategy(backend)
                .topology_override("tensor-parallel", 2)
                .is_none());
        }
    }
}
