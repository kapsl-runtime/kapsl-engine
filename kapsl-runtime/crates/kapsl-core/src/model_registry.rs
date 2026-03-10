use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Status of a model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    /// Model is currently active and accepting requests
    #[default]
    Active,
    /// Model is inactive and not accepting requests
    Inactive,
    /// Model is in the process of starting
    Starting,
    /// Model is in the process of stopping
    Stopping,
    /// Model is loading
    Loading,
}

/// Information about a loaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique model ID
    pub id: u32,
    /// Model name from manifest
    pub name: String,
    /// Model version
    pub version: String,
    /// Framework (e.g., "onnx", "tensorflow")
    pub framework: String,
    /// Execution provider/device (e.g., "CPU", "CUDA", "Metal")
    pub device: String,
    /// Graph optimization level
    pub optimization_level: String,
    /// Timestamp when model was loaded (Unix timestamp in seconds)
    pub loaded_at: u64,
    /// Model file path
    pub model_path: String,
    /// Current status of the model
    pub status: ModelStatus,
    /// Replica ID for this specific instance (0 for primary)
    pub replica_id: u32,
    /// Base model ID this is a replica of (same as id for primary)
    pub base_model_id: u32,
}

impl ModelInfo {
    /// Create a new ModelInfo instance
    pub fn new(
        id: u32,
        name: String,
        version: String,
        framework: String,
        device: String,
        optimization_level: String,
        model_path: String,
    ) -> Self {
        Self {
            id,
            name,
            version,
            framework,
            device,
            optimization_level,
            loaded_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model_path,
            status: ModelStatus::Active,
            replica_id: 0,     // Default to 0 for primary instance
            base_model_id: id, // Primary instance uses its own ID as base
        }
    }

    /// Create a new replica of an existing model
    #[allow(clippy::too_many_arguments)]
    pub fn new_replica(
        unique_id: u32,
        replica_id: u32,
        base_model_id: u32,
        name: String,
        version: String,
        framework: String,
        device: String,
        optimization_level: String,
        model_path: String,
    ) -> Self {
        Self {
            id: unique_id,
            name,
            version,
            framework,
            device,
            optimization_level,
            loaded_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model_path,
            status: ModelStatus::Active,
            replica_id,
            base_model_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelEvent {
    Registered(u32),
    Unregistered(u32),
    StatusChanged {
        id: u32,
        old_status: ModelStatus,
        new_status: ModelStatus,
    },
    // MetricsUpdated(u32), // Optional: might be too noisy
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetric {
    pub model_latency_ms_p50: f32,
    pub model_latency_ms_p95: f32,
    pub model_latency_ms_p99: f32,
    pub model_latency_ms_max: f32,
    pub request_count: u64,
}

/// Thread-safe registry for tracking loaded models
// Callback type for event listeners
type EventListener = Box<dyn Fn(&ModelEvent) + Send + Sync>;

/// Thread-safe registry for tracking loaded models
#[derive(Clone)]
pub struct ModelRegistry {
    models: Arc<RwLock<HashMap<u32, ModelInfo>>>,
    // Metrics storage
    metrics: Arc<RwLock<HashMap<u32, ModelMetric>>>,
    // Status history for rollback (Latest -> Oldest)
    history: Arc<RwLock<HashMap<u32, Vec<ModelStatus>>>>,
    // Event listeners
    listeners: Arc<RwLock<Vec<EventListener>>>,
}

impl ModelRegistry {
    /// Create a new empty ModelRegistry
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(HashMap::new())),
            listeners: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a new model
    pub fn register(&self, model_info: ModelInfo) {
        let id = model_info.id;
        {
            let mut models = self.models.write();
            models.insert(id, model_info.clone());
        }
        // Initialize metrics
        {
            let mut metrics = self.metrics.write();
            metrics.insert(id, ModelMetric::default());
        }
        // Initialize history
        {
            let mut history = self.history.write();
            history.insert(id, Vec::new());
        }
        self.notify_listeners(&ModelEvent::Registered(id));
    }

    /// Register or update model metadata while preserving existing status.
    pub fn upsert(&self, mut model_info: ModelInfo) {
        let id = model_info.id;
        {
            let mut models = self.models.write();
            if let Some(existing) = models.get_mut(&id) {
                let status = existing.status;
                let replica_id = existing.replica_id;
                let base_model_id = existing.base_model_id;
                model_info.status = status;
                model_info.replica_id = replica_id;
                model_info.base_model_id = base_model_id;
                *existing = model_info;
                return;
            }
            models.insert(id, model_info);
        }
        {
            let mut metrics = self.metrics.write();
            metrics.insert(id, ModelMetric::default());
        }
        {
            let mut history = self.history.write();
            history.insert(id, Vec::new());
        }
        self.notify_listeners(&ModelEvent::Registered(id));
    }

    /// Get information about a specific model
    pub fn get(&self, model_id: u32) -> Option<ModelInfo> {
        let models = self.models.read();
        models.get(&model_id).cloned()
    }

    /// Get all registered models
    pub fn list(&self) -> Vec<ModelInfo> {
        let models = self.models.read();
        models.values().cloned().collect()
    }

    /// Get the number of registered models
    pub fn count(&self) -> usize {
        let models = self.models.read();
        models.len()
    }

    /// Remove a model from the registry
    pub fn unregister(&self, model_id: u32) -> Option<ModelInfo> {
        let result = {
            let mut models = self.models.write();
            models.remove(&model_id)
        };

        if result.is_some() {
            // Clean up metrics and history
            self.metrics.write().remove(&model_id);
            self.history.write().remove(&model_id);
            self.notify_listeners(&ModelEvent::Unregistered(model_id));
        }

        result
    }

    /// Validate if a state transition is allowed
    fn validate_transition(from: ModelStatus, to: ModelStatus) -> bool {
        match (from, to) {
            // Can always go to same status (noop)
            (a, b) if a == b => true,
            // To Stopping: Allowed from Active, Starting (e.g. abort)
            (_, ModelStatus::Stopping) => true,
            // To Inactive: From Stopping (normal), Starting (failure), Active (force kill?)
            (ModelStatus::Stopping, ModelStatus::Inactive) => true,
            (ModelStatus::Starting, ModelStatus::Inactive) => true, // Start failure

            // To Starting: From Inactive
            (ModelStatus::Inactive, ModelStatus::Starting) => true,

            // To Active: From Starting (success), Stopping (abort stop), Inactive (restart)
            (ModelStatus::Starting, ModelStatus::Active) => true,
            (ModelStatus::Stopping, ModelStatus::Active) => true, // Resume/Abort stop
            (ModelStatus::Inactive, ModelStatus::Active) => true, // Direct restart

            // Default deny
            _ => false,
        }
    }

    /// Update the status of a model
    pub fn set_status(&self, model_id: u32, status: ModelStatus) -> Result<ModelStatus, String> {
        let mut models = self.models.write();
        if let Some(model) = models.get_mut(&model_id) {
            let old_status = model.status;

            if !Self::validate_transition(old_status, status) {
                return Err(format!(
                    "Invalid state transition from {:?} to {:?}",
                    old_status, status
                ));
            }

            if old_status != status {
                // Save to history
                let mut history = self.history.write();
                if let Some(h_vec) = history.get_mut(&model_id) {
                    h_vec.push(old_status);
                }

                model.status = status;

                // Notify
                drop(models); // Drop lock before notifying
                self.notify_listeners(&ModelEvent::StatusChanged {
                    id: model_id,
                    old_status,
                    new_status: status,
                });
                return Ok(old_status);
            }
            Ok(old_status) // No change needed, return current
        } else {
            Err(format!("Model {} not found", model_id))
        }
    }

    /// List models filtered by status
    pub fn list_by_status(&self, status: ModelStatus) -> Vec<ModelInfo> {
        let models = self.models.read();
        models
            .values()
            .filter(|m| m.status == status)
            .cloned()
            .collect()
    }

    /// Get all replicas of a specific base model
    pub fn list_replicas(&self, base_model_id: u32) -> Vec<ModelInfo> {
        let models = self.models.read();
        models
            .values()
            .filter(|m| m.base_model_id == base_model_id)
            .cloned()
            .collect()
    }

    /// Count active replicas for a specific base model
    pub fn count_active_replicas(&self, base_model_id: u32) -> usize {
        let models = self.models.read();
        models
            .values()
            .filter(|m| m.base_model_id == base_model_id && m.status == ModelStatus::Active)
            .count()
    }

    // --- Metrics Methods ---

    pub fn update_metric(&self, model_id: u32, metric: ModelMetric) {
        let mut metrics = self.metrics.write();
        metrics.insert(model_id, metric);
    }

    pub fn get_metric(&self, model_id: u32) -> Option<ModelMetric> {
        let metrics = self.metrics.read();
        metrics.get(&model_id).cloned()
    }

    // --- Event Methods ---

    pub fn subscribe<F>(&self, callback: F)
    where
        F: Fn(&ModelEvent) + Send + Sync + 'static,
    {
        let mut listeners = self.listeners.write();
        listeners.push(Box::new(callback));
    }

    fn notify_listeners(&self, event: &ModelEvent) {
        let listeners = self.listeners.read();
        for listener in listeners.iter() {
            listener(event);
        }
    }

    // --- Versioning & Rollback Methods ---

    /// Revert model status to the previous state
    pub fn rollback_status(&self, model_id: u32) -> bool {
        let mut history = self.history.write();
        if let Some(h_vec) = history.get_mut(&model_id) {
            if let Some(prev_status) = h_vec.pop() {
                // Drop history lock before acquiring models lock to avoid deadlock order issues?
                // Actually `set_status` acquires models -> history. Here we seek to acquire models.
                // If we hold history, then call set_status (which takes models then history), we deadlock.
                // So we must release history lock, then call set_status.
                drop(history);

                // We use set_status which will push the current (wrong) status to history again?
                // Rollback implies "pop and restore". `set_status` pushes old status.
                // If we pop `prev`, current is `curr`. `set_status(prev)` makes `curr` go to history.
                // So history becomes `[..., curr]`. This acts as "undo/redo" stack effectively.
                // If we want permanent forget, we should manually update.
                // Let's stick to set_status for simplicity and event notification.
                return self.set_status(model_id, prev_status).is_ok();
            }
        }
        false
    }

    /// Get all versions of a model by name
    pub fn get_versions(&self, name: &str) -> Vec<ModelInfo> {
        let models = self.models.read();
        models
            .values()
            .filter(|m| m.name == name)
            .cloned()
            .collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[path = "model_registry_tests.rs"]
mod model_registry_tests;
