#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_model_registry_basic() {
        let registry = ModelRegistry::new();

        let model = ModelInfo::new(
            0,
            "test_model".to_string(),
            "1.0.0".to_string(),
            "onnx".to_string(),
            "CPU".to_string(),
            "all".to_string(),
            "/path/to/model.onnx".to_string(),
        );

        registry.register(model.clone());

        assert_eq!(registry.count(), 1);
        let retrieved = registry.get(0).unwrap();
        assert_eq!(retrieved.name, "test_model");
        assert_eq!(retrieved.version, "1.0.0");
    }

    #[test]
    fn test_model_registry_multiple() {
        let registry = ModelRegistry::new();

        for i in 0..5 {
            let model = ModelInfo::new(
                i,
                format!("model_{}", i),
                "1.0.0".to_string(),
                "onnx".to_string(),
                "CPU".to_string(),
                "basic".to_string(),
                format!("/path/to/model_{}.onnx", i),
            );
            registry.register(model);
        }

        assert_eq!(registry.count(), 5);
        let models = registry.list();
        assert_eq!(models.len(), 5);
    }

    #[test]
    fn test_model_registry_unregister() {
        let registry = ModelRegistry::new();

        let model = ModelInfo::new(
            0,
            "test_model".to_string(),
            "1.0.0".to_string(),
            "onnx".to_string(),
            "CPU".to_string(),
            "all".to_string(),
            "/path/to/model.onnx".to_string(),
        );

        registry.register(model);
        assert_eq!(registry.count(), 1);

        let removed = registry.unregister(0);
        assert!(removed.is_some());
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_model_registry_thread_safety() {
        use std::thread;

        let registry = ModelRegistry::new();
        let mut handles = vec![];

        for i in 0..10 {
            let registry_clone = registry.clone();
            let handle = thread::spawn(move || {
                let model = ModelInfo::new(
                    i,
                    format!("model_{}", i),
                    "1.0.0".to_string(),
                    "onnx".to_string(),
                    "CPU".to_string(),
                    "basic".to_string(),
                    format!("/path/to/model_{}.onnx", i),
                );
                registry_clone.register(model);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(registry.count(), 10);
    }

    #[test]
    fn test_model_registry_events() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let registry = ModelRegistry::new();
        let event_count = Arc::new(AtomicUsize::new(0));
        let count_clone = event_count.clone();

        registry.subscribe(move |_event| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });

        let model = ModelInfo::new(
            0,
            "test".into(),
            "1".into(),
            "onnx".into(),
            "cpu".into(),
            "all".into(),
            "path".into(),
        );
        registry.register(model);
        // Should trigger Registered event
        assert_eq!(event_count.load(Ordering::SeqCst), 1);

        registry.set_status(0, ModelStatus::Stopping).unwrap();
        // StatusChanged
        assert_eq!(event_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_model_registry_metrics() {
        let registry = ModelRegistry::new();
        let model = ModelInfo::new(
            0,
            "test".into(),
            "1".into(),
            "onnx".into(),
            "cpu".into(),
            "all".into(),
            "path".into(),
        );
        registry.register(model);

        let metric = ModelMetric {
            request_count: 100,
            ..Default::default()
        };
        registry.update_metric(0, metric);

        let retrieved = registry.get_metric(0).unwrap();
        assert_eq!(retrieved.request_count, 100);
    }

    #[test]
    fn test_rollback_status() {
        let registry = ModelRegistry::new();
        let model = ModelInfo::new(
            0,
            "test".into(),
            "1".into(),
            "onnx".into(),
            "cpu".into(),
            "all".into(),
            "path".into(),
        );
        registry.register(model); // Status: Active (default)

        registry.set_status(0, ModelStatus::Stopping).unwrap();
        assert_eq!(registry.get(0).unwrap().status, ModelStatus::Stopping);

        let success = registry.rollback_status(0);
        assert!(success);
        assert_eq!(registry.get(0).unwrap().status, ModelStatus::Active);
    }

    #[test]
    fn test_state_transitions() {
        let registry = ModelRegistry::new();
        let model = ModelInfo::new(
            0,
            "test".into(),
            "1".into(),
            "onnx".into(),
            "cpu".into(),
            "all".into(),
            "path".into(),
        );
        registry.register(model);
        // Initial: Active

        // Allowed: Active -> Stopping
        assert!(registry.set_status(0, ModelStatus::Stopping).is_ok());

        // Allowed: Stopping -> Inactive
        assert!(registry.set_status(0, ModelStatus::Inactive).is_ok());

        // Allowed: Inactive -> Starting
        assert!(registry.set_status(0, ModelStatus::Starting).is_ok());

        // Allowed: Starting -> Active
        assert!(registry.set_status(0, ModelStatus::Active).is_ok());

        // Invalid: Active -> Inactive (must stop first)
        assert!(registry.set_status(0, ModelStatus::Inactive).is_err());

        // Invalid: Active -> Starting (makes no sense)
        // Wait, maybe Active -> Starting (restart) is okay? No, usually Stop -> Start.
        // My validation logic default deny everything else.
        // Let's check logic: (Starting, Active) -> OK.
        // (Active, Starting) -> _ => false.
        assert!(registry.set_status(0, ModelStatus::Starting).is_err());
    }

    #[test]
    fn test_upsert_preserves_status_and_replica_fields() {
        let registry = ModelRegistry::new();
        let model = ModelInfo::new(
            1,
            "base".into(),
            "1".into(),
            "onnx".into(),
            "cpu".into(),
            "all".into(),
            "path".into(),
        );
        registry.register(model);

        registry.set_status(1, ModelStatus::Stopping).unwrap();

        let updated = ModelInfo::new(
            1,
            "base-updated".into(),
            "2".into(),
            "onnx".into(),
            "cpu".into(),
            "basic".into(),
            "new-path".into(),
        );
        registry.upsert(updated);

        let retrieved = registry.get(1).unwrap();
        assert_eq!(retrieved.name, "base-updated");
        assert_eq!(retrieved.version, "2");
        assert_eq!(retrieved.status, ModelStatus::Stopping);
        assert_eq!(retrieved.replica_id, 0);
        assert_eq!(retrieved.base_model_id, 1);
    }

    #[test]
    fn test_list_replicas_and_active_count() {
        let registry = ModelRegistry::new();

        let base = ModelInfo::new(
            1,
            "model".into(),
            "1".into(),
            "onnx".into(),
            "cpu".into(),
            "all".into(),
            "path".into(),
        );
        let replica1 = ModelInfo::new_replica(
            2,
            1,
            1,
            "model".into(),
            "1".into(),
            "onnx".into(),
            "cpu".into(),
            "all".into(),
            "path".into(),
        );
        let replica2 = ModelInfo::new_replica(
            3,
            2,
            1,
            "model".into(),
            "1".into(),
            "onnx".into(),
            "cpu".into(),
            "all".into(),
            "path".into(),
        );

        registry.register(base);
        registry.register(replica1);
        registry.register(replica2);

        registry.set_status(2, ModelStatus::Stopping).unwrap();
        registry.set_status(2, ModelStatus::Inactive).unwrap();

        let replicas = registry.list_replicas(1);
        assert_eq!(replicas.len(), 3);

        let active_count = registry.count_active_replicas(1);
        assert_eq!(active_count, 2);

        let active_models = registry.list_by_status(ModelStatus::Active);
        assert_eq!(active_models.len(), 2);
    }
}
