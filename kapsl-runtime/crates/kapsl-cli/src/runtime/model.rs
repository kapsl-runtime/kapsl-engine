use super::*;

mod backend;
mod lifecycle;
pub(crate) mod load_plan;
mod replica;
mod scaling;

use backend::*;
pub(crate) use lifecycle::*;
pub(crate) use load_plan::*;
use replica::*;
pub(crate) use scaling::*;

/// Worker-only values translated from CLI arguments at the application edge.
///
/// Keeping this type in the runtime model layer prevents worker execution from
/// depending on the CLI's broad argument structure.
#[derive(Clone, Debug)]
pub(crate) struct WorkerRunConfig {
    model_id: u32,
    model_path: PathBuf,
    socket_path: String,
}

impl WorkerRunConfig {
    pub(crate) fn new(model_id: u32, model_path: PathBuf, socket_path: String) -> Self {
        Self {
            model_id,
            model_path,
            socket_path,
        }
    }
}

/// Process-scoped facade for model lifecycle execution.
///
/// The composition root creates this once and injects it into startup, HTTP,
/// and autoscaling paths. Callers provide only request-specific values; stable
/// hardware, planning, resource, registry, and metrics dependencies remain
/// owned here.
pub(crate) struct ModelRuntime {
    load_planner: Arc<ModelLoadPlanner>,
    resources: Arc<RuntimeResources>,
    models: Arc<ModelManager>,
    shared_metrics: kapsl_monitor::metrics::KapslMetrics,
}

impl ModelRuntime {
    pub(crate) fn new(
        load_planner: Arc<ModelLoadPlanner>,
        resources: Arc<RuntimeResources>,
        models: Arc<ModelManager>,
        shared_metrics: kapsl_monitor::metrics::KapslMetrics,
    ) -> Self {
        Self {
            load_planner,
            resources,
            models,
            shared_metrics,
        }
    }

    pub(crate) fn device_info(&self) -> &DeviceInfo {
        self.load_planner.device_info()
    }

    pub(crate) fn memory_snapshot(&self) -> MemorySnapshot {
        self.resources.memory().snapshot()
    }

    pub(crate) fn models(&self) -> &Arc<ModelManager> {
        &self.models
    }

    pub(crate) fn shared_metrics(&self) -> &kapsl_monitor::metrics::KapslMetrics {
        &self.shared_metrics
    }

    pub(crate) async fn reclaim_one_lower_priority_model(
        &self,
        failure: &MemoryAdmissionFailure,
    ) -> Option<PriorityReclaimResult> {
        reclaim_one_lower_priority_model(failure, &self.models, &self.resources).await
    }

    pub(crate) fn stop_model_and_replicas(
        &self,
        base_model_id: u32,
    ) -> Result<Vec<ModelInfo>, String> {
        stop_model_and_replicas(base_model_id, &self.models, &self.resources)
    }

    /// Load a model using the ambient Tokio runtime from a blocking lifecycle task.
    pub(crate) fn load_model_blocking(
        &self,
        model_id: u32,
        model_path: &Path,
        placement: &ModelLoadPlacement,
    ) -> Result<(Arc<ReplicaPool<Scheduler>>, Vec<EngineHandle>), DynError> {
        tokio::runtime::Handle::current().block_on(self.load_model(model_id, model_path, placement))
    }

    /// Load the primary replica for a model and create its replica pool.
    pub(crate) async fn load_model(
        &self,
        model_id: u32,
        model_path: &Path,
        placement: &ModelLoadPlacement,
    ) -> Result<(Arc<ReplicaPool<Scheduler>>, Vec<EngineHandle>), DynError> {
        let plan = self
            .load_planner
            .prepare(model_id, model_id, 0, model_path, Some(placement))?;
        self.load_prepared_model(plan).await
    }

    pub(crate) async fn load_prepared_model(
        &self,
        plan: ModelLoadPlan,
    ) -> Result<(Arc<ReplicaPool<Scheduler>>, Vec<EngineHandle>), DynError> {
        let model_id = plan.base_model_id();
        #[cfg(feature = "gpu-device-pool")]
        {
            let bootstrap =
                device_memory_bootstrap_plan(std::iter::once(&plan), self.device_info())?;
            self.resources.ensure_device_pools(&bootstrap)?;
            self.resources
                .attach_device_memory_metrics(self.shared_metrics.clone());
        }
        log::info!(
            "Current directory: {:?}",
            std::env::current_dir().unwrap_or_default()
        );

        let LoadedReplica {
            scheduler,
            swap_handles,
            model_info,
        } = load_replica(
            plan,
            ReplicaLoadRole::Primary,
            self.device_info(),
            self.resources.clone(),
            &self.shared_metrics,
        )
        .await?;
        self.models.registry().upsert(model_info);

        let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
        pool.add_replica(0, scheduler);
        log::info!("✓ Scheduler started for Model ID {}\n", model_id);

        Ok((Arc::new(pool), swap_handles))
    }

    /// Reserve the aggregate managed-vLLM weight/workspace estimate before Kapsl
    /// performs any backend download or starts a backend child. The reservation is
    /// intentionally temporary: ordinary per-model load transactions repeat the
    /// admission after installation and retain the authoritative leases.
    pub(crate) async fn preflight_managed_vllm_admission(
        &self,
        plans: &[(PathBuf, ModelLoadPlan)],
    ) -> Result<Option<MemoryAdmission>, DynError> {
        use kapsl_engine_api::MemoryReport;

        let mut report = MemoryReport::default();
        let mut domains = Vec::new();
        for (_, plan) in plans.iter().filter(|(_, plan)| plan.uses_managed_vllm()) {
            validate_managed_vllm_launch_policy(&plan.loader.manifest)?;
            if plan.use_pipeline_backend || plan.worker_tp_degree != 1 {
                return Err(
                    "managed vLLM currently supports one CUDA device per Kapsl replica; keep --tp-degree=1"
                        .into(),
                );
            }
            let selection = select_mesh_devices(
                &plan.loader.manifest.hardware_requirements,
                self.device_info(),
            )
            .map_err(|error| {
                format!(
                    "Failed to select a CUDA device for preliminary admission of model {}: {error}",
                    plan.base_model_id
                )
            })?;
            let device_id = selection
                .devices
                .iter()
                .find(|device| device.backend.to_string().eq_ignore_ascii_case("cuda"))
                .map(|device| device.id)
                .ok_or("managed vLLM preliminary admission selected no CUDA device")?;
            let model_report = managed_vllm_memory_report(
                &plan.model_file_path,
                &[device_id],
                plan.base_model_id,
                plan.replica_id,
            )?;
            report.allocations.extend(model_report.allocations);
            let domain = MemoryDomain::Cuda { device_id };
            if !domains.contains(&domain) {
                domains.push(domain);
            }
        }
        if report.allocations.is_empty() {
            return Ok(None);
        }

        // One synthetic owner lets all startup models targeting the same device be
        // checked atomically under one device load lock. The admission is dropped
        // after backend installation, before real model IDs acquire their leases.
        let owner = MemoryOwner::new(u32::MAX, u32::MAX);
        let plan = self
            .resources
            .memory()
            .model_load_plan_with_report(&domains, owner, 0, 0, &report)?;
        let admission = self
            .resources
            .memory()
            .begin_load(&plan, EngineKind::Native)
            .await
            .map_err(|error| {
                format!(
                    "preliminary managed-vLLM memory admission rejected before backend download: {error}"
                )
            })?;
        Ok(Some(admission))
    }
}

pub(crate) async fn run_worker(
    config: WorkerRunConfig,
    load_planner: Arc<ModelLoadPlanner>,
) -> Result<(), DynError> {
    let plan = load_planner.prepare(
        config.model_id,
        config.model_id,
        0,
        &config.model_path,
        None,
    )?;
    #[cfg(feature = "gpu-device-pool")]
    let bootstrap =
        device_memory_bootstrap_plan(std::iter::once(&plan), load_planner.device_info())?;

    let registry = Arc::new(Registry::new());
    let model_registry = Arc::new(ModelRegistry::new());
    let models = ModelManager::new(model_registry);
    let shared_metrics = kapsl_monitor::metrics::KapslMetrics::new(&registry);
    #[cfg(feature = "gpu-device-pool")]
    let resources =
        RuntimeResources::new_with_device_memory_plan(load_planner.device_info(), &bootstrap)?;
    #[cfg(not(feature = "gpu-device-pool"))]
    let resources = RuntimeResources::new(load_planner.device_info())?;
    let memory_for_reconciliation = resources.memory().clone();
    let model_runtime = ModelRuntime::new(load_planner, resources, models, shared_metrics);
    let (pool, handles) = model_runtime.load_prepared_model(plan).await?;

    // Isolated workers do not run the parent HTTP/system monitor. Keep their
    // process-local authority current as provider arenas and KV allocations
    // change, using the same cadence as the parent sampler.
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        loop {
            interval.tick().await;
            for handle in &handles {
                let _ = handle.actual_memory();
            }
            if let Some(rss) = super::host_memory::process_rss_bytes() {
                memory_for_reconciliation.observe_process_memory(rss);
            }
        }
    });

    let mut schedulers = HashMap::new();
    schedulers.insert(
        config.model_id,
        pool as Arc<dyn ReplicaScheduler + Send + Sync>,
    );

    let server = IpcServer::new(&config.socket_path, schedulers, None);
    log::info!(
        "Worker process serving model {} via IPC socket {}",
        config.model_id,
        config.socket_path
    );
    server.run().await?;
    Ok(())
}
