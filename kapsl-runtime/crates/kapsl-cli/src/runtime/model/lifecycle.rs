use super::*;

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
) -> Vec<ModelInfo> {
    let replicas = models.registry().list_replicas(base_model_id);

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

    replicas
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let stopped = stop_model_and_replicas(7, &models, &resources);
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
}
