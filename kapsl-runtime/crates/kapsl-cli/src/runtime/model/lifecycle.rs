use super::*;
use std::cmp::Reverse;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PriorityReclaimAction {
    Replica { replica_id: u32 },
    Model,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PriorityReclaimResult {
    pub(crate) model_id: u32,
    pub(crate) priority_weight: u32,
    pub(crate) action: PriorityReclaimAction,
    pub(crate) released_bytes: usize,
}

fn model_authority_bytes(snapshot: &MemorySnapshot, model_id: u32) -> usize {
    snapshot
        .rows
        .iter()
        .filter(|row| row.owner.model_id == model_id)
        .map(MemorySnapshotRow::used_bytes)
        .fold(0usize, usize::saturating_add)
}

/// Reclaim one idle, strictly lower-weight victim after a typed memory
/// admission failure. Extra replicas are removed before the primary model is
/// unloaded. The global reclamation mutex serializes competing high-priority
/// loads; model lifecycle locks keep stop/swap/autoscale from interleaving.
pub(crate) async fn reclaim_one_lower_priority_model(
    failure: &MemoryAdmissionFailure,
    models: &Arc<ModelManager>,
    resources: &Arc<RuntimeResources>,
) -> Option<PriorityReclaimResult> {
    let _reclamation = resources.priority().lock_reclamation().await;
    let snapshot = resources.memory().snapshot();
    let candidates = resources.priority().reclaim_candidates(
        &snapshot,
        failure.owner().model_id,
        failure.priority_weight(),
        failure.domains(),
    );

    for candidate in candidates {
        let Some(pool) = models.pool(candidate.model_id) else {
            continue;
        };
        let (high, low) = pool.get_queue_depth();
        if high.saturating_add(low) != 0 {
            continue;
        }

        let _victim_lifecycle = models.lock_lifecycle(candidate.model_id).await;
        let Some(pool) = models.pool(candidate.model_id) else {
            continue;
        };
        let (high, low) = pool.get_queue_depth();
        if high.saturating_add(low) != 0 {
            continue;
        }

        let before = model_authority_bytes(&snapshot, candidate.model_id);
        let mut extra_replicas = models
            .registry()
            .list_replicas(candidate.model_id)
            .into_iter()
            .filter(|replica| replica.replica_id > 0 && replica.status == ModelStatus::Active)
            .collect::<Vec<_>>();
        extra_replicas.sort_by_key(|replica| Reverse(replica.replica_id));

        let action = if let Some(replica) = extra_replicas.first() {
            if let Err(error) =
                scale_down_model(candidate.model_id, replica.replica_id, replica.id, models).await
            {
                log::warn!(
                    "Priority arbiter could not scale down model {} replica {}: {}",
                    candidate.model_id,
                    replica.replica_id,
                    error,
                );
                continue;
            }
            PriorityReclaimAction::Replica {
                replica_id: replica.replica_id,
            }
        } else {
            if let Err(error) = stop_model_and_replicas(candidate.model_id, models, resources) {
                log::warn!(
                    "Priority arbiter could not stop model {}: {}",
                    candidate.model_id,
                    error,
                );
                continue;
            }
            PriorityReclaimAction::Model
        };

        // Scheduler executors release their final engine `Arc` after their
        // closed queues drain. Wait briefly for that ordinary asynchronous drop
        // so the retry observes returned authority capacity, while remaining
        // bounded if an in-flight backend is slow to tear down.
        let mut after = model_authority_bytes(&resources.memory().snapshot(), candidate.model_id);
        for _ in 0..20 {
            if after < before {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
            after = model_authority_bytes(&resources.memory().snapshot(), candidate.model_id);
        }
        let released_bytes = before.saturating_sub(after);
        log::warn!(
            "[priority-arbiter] {:?} for model {} (weight {}) released {} authority bytes for model {} (weight {})",
            action,
            candidate.model_id,
            candidate.priority_weight,
            released_bytes,
            failure.owner().model_id,
            failure.priority_weight(),
        );
        return Some(PriorityReclaimResult {
            model_id: candidate.model_id,
            priority_weight: candidate.priority_weight,
            action,
            released_bytes,
        });
    }
    None
}

/// Stop every runtime replica for one logical model and retain its registry
/// entries and package path so it can be started again.
///
/// Callers must hold the model's lifecycle lock. Both the stop and remove HTTP
/// paths delegate here so pool teardown, KV detachment, and registry status
/// transitions cannot drift apart.
pub(crate) fn stop_model_and_replicas(
    base_model_id: u32,
    models: &ModelManager,
    resources: &RuntimeResources,
) -> Result<Vec<ModelInfo>, String> {
    let replicas = models.registry().list_replicas(base_model_id);

    // A managed child imports Kapsl-owned CUDA IPC memory. Prove that its
    // complete process group has exited and retire the participant before
    // removing scheduler/engine handles that release ordinary model authority.
    // If the fence fails, preserve the existing runtime and its charges.
    if let Some(deployment) = resources.managed_vllm() {
        match deployment.shutdown_model(base_model_id) {
            Ok(count) if count > 0 => log::info!(
                "Stopped {} managed vLLM runtime(s) for model {}",
                count,
                base_model_id
            ),
            Ok(_) => {}
            Err(error) => {
                return Err(format!(
                    "managed vLLM teardown for model {} did not complete: {}",
                    base_model_id, error
                ));
            }
        }
    }

    for replica in &replicas {
        if let Err(error) = models
            .registry()
            .set_status(replica.id, ModelStatus::Stopping)
        {
            log::warn!(
                "Failed to set model {} replica {} to Stopping: {}",
                base_model_id,
                replica.replica_id,
                error
            );
        }
    }

    if let Some(pool) = models.pool(base_model_id) {
        for replica in &replicas {
            if !pool.remove_replica(replica.replica_id) {
                log::debug!(
                    "Replica {} for model {} was not present while stopping",
                    replica.replica_id,
                    base_model_id
                );
            }
        }
    }

    models.stop_runtime(base_model_id);
    resources.kv().detach_engine_for_model(base_model_id);

    for replica in &replicas {
        if let Err(error) = models
            .registry()
            .set_status(replica.id, ModelStatus::Inactive)
        {
            log::warn!(
                "Failed to set model {} replica {} to Inactive: {}",
                base_model_id,
                replica.replica_id,
                error
            );
        }
    }

    Ok(replicas)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LeasedTestEngine {
        lease: std::sync::Mutex<Option<MemoryLease>>,
        priority: std::sync::Mutex<Option<ModelPriorityLease>>,
    }

    #[async_trait::async_trait]
    impl Engine for LeasedTestEngine {
        async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
            Ok(())
        }

        fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
            Ok(request.input.clone())
        }

        fn infer_stream(&self, request: &InferenceRequest) -> kapsl_engine_api::EngineStream {
            let packet = request.input.clone();
            Box::pin(futures::stream::once(async move { Ok(packet) }))
        }

        fn unload(&mut self) {
            self.lease.get_mut().unwrap().take();
            self.priority.get_mut().unwrap().take();
        }

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        fn health_check(&self) -> Result<(), EngineError> {
            Ok(())
        }
    }

    fn test_device_info() -> DeviceInfo {
        DeviceInfo {
            cpu_cores: 1,
            total_memory: 1024,
            os_type: "test".to_string(),
            os_release: "test".to_string(),
            has_cuda: false,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: Vec::new(),
        }
    }

    #[tokio::test]
    async fn stop_transitions_primary_and_autoscaled_replicas_together() {
        let registry = Arc::new(ModelRegistry::new());
        registry.register(ModelInfo::new(
            7,
            "model".to_string(),
            "1".to_string(),
            "onnx".to_string(),
            "cpu".to_string(),
            "basic".to_string(),
            "/tmp/model.aimod".to_string(),
        ));
        registry.register(ModelInfo::new_replica(
            1001,
            1,
            7,
            "model".to_string(),
            "1".to_string(),
            "onnx".to_string(),
            "cpu".to_string(),
            "basic".to_string(),
            "/tmp/model.aimod".to_string(),
        ));

        let models = ModelManager::new(registry.clone());
        let pool: Arc<ReplicaPool<Scheduler>> =
            Arc::new(ReplicaPool::new(PoolStrategy::LeastLoaded));
        models.install_loaded(7, PathBuf::from("/tmp/model.aimod"), pool, Vec::new());
        let resources = RuntimeResources::new(&test_device_info()).expect("runtime resources");
        let _lifecycle_guard = models.lock_lifecycle(7).await;

        let stopped = stop_model_and_replicas(7, &models, &resources).unwrap();
        let mut stopped_ids: Vec<_> = stopped.into_iter().map(|model| model.id).collect();
        stopped_ids.sort_unstable();

        assert_eq!(stopped_ids, vec![7, 1001]);
        assert_eq!(registry.count_active_replicas(7), 0);
        assert_eq!(registry.get(7).unwrap().status, ModelStatus::Inactive);
        assert_eq!(registry.get(1001).unwrap().status, ModelStatus::Inactive);
        assert!(!models.contains_pool(7));
        assert_eq!(
            models.model_path(7),
            Some(PathBuf::from("/tmp/model.aimod"))
        );
    }

    #[tokio::test]
    async fn typed_admission_failure_unloads_only_idle_lower_weight_model() {
        let device_info = DeviceInfo {
            total_memory: 10 * 1024 * 1024,
            ..test_device_info()
        };
        let resources = RuntimeResources::new(&device_info).expect("runtime resources");
        let registry = Arc::new(ModelRegistry::new());
        registry.register(ModelInfo::new(
            1,
            "low".to_string(),
            "1".to_string(),
            "onnx".to_string(),
            "cpu".to_string(),
            "basic".to_string(),
            "/tmp/low.aimod".to_string(),
        ));
        let models = ModelManager::new(registry.clone());
        let owner = MemoryOwner::new(1, 0);
        let mut victim_plan = MemoryPlan::new();
        victim_plan.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::PersistentWeights,
            1024,
        ));
        let lease = resources.memory().admit(&victim_plan).expect("lease");
        let priority = resources
            .priority()
            .register(owner, 1, [MemoryDomain::Host]);
        let engine: EngineHandle = Arc::new(LeasedTestEngine {
            lease: std::sync::Mutex::new(Some(lease)),
            priority: std::sync::Mutex::new(Some(priority)),
        });
        let scheduler = Arc::new(Scheduler::new(vec![engine], 1, 1, 8, true, 1, 0, None));
        let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
        pool.add_replica(0, scheduler);
        models.install_loaded(
            1,
            PathBuf::from("/tmp/low.aimod"),
            Arc::new(pool),
            Vec::new(),
        );

        let mut request_plan = MemoryPlan::new();
        request_plan.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            MemoryOwner::new(9, 0),
            MemoryAllocationClass::PersistentWeights,
            2048,
        ));
        let failure = MemoryAdmissionFailure::new(
            MemoryOwner::new(9, 0),
            5,
            &request_plan,
            "synthetic pressure",
        );
        let reclaimed = reclaim_one_lower_priority_model(&failure, &models, &resources)
            .await
            .expect("lower-priority victim");

        assert_eq!(reclaimed.model_id, 1);
        assert_eq!(reclaimed.action, PriorityReclaimAction::Model);
        assert!(!models.contains_pool(1));
        assert_eq!(registry.get(1).unwrap().status, ModelStatus::Inactive);
        assert!(resources
            .memory()
            .snapshot()
            .rows
            .iter()
            .all(|row| row.owner.model_id != 1));
    }
}
