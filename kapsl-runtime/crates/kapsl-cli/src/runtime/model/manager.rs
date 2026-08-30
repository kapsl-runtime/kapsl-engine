//! Runtime-owned model registry state and lifecycle serialization.

use super::*;
use tokio::sync::OwnedMutexGuard;

/// Runtime-owned state for one logical model. The lifecycle mutex survives
/// asynchronous loads and explicit-ID reuse, so stop/remove/swap serialize
/// with every operation for that identity.
pub(crate) struct ModelEntry {
    path: Option<PathBuf>,
    pool: Option<Arc<ReplicaPool<Scheduler>>>,
    swap_handles: Vec<EngineHandle>,
    lifecycle: Arc<AsyncMutex<()>>,
}

impl ModelEntry {
    fn empty() -> Self {
        Self {
            path: None,
            pool: None,
            swap_handles: Vec::new(),
            lifecycle: Arc::new(AsyncMutex::new(())),
        }
    }
}

/// Single owner for model identity, runtime replicas, hot-swap handles, paths,
/// and lifecycle serialization.
pub(crate) struct ModelManager {
    registry: Arc<ModelRegistry>,
    entries: RwLock<HashMap<u32, ModelEntry>>,
    next_model_id: AtomicU32,
    recycled_model_ids: Mutex<Vec<u32>>,
    next_replica_unique_id: AtomicU32,
}

impl ModelManager {
    pub(crate) fn new(registry: Arc<ModelRegistry>) -> Arc<Self> {
        Arc::new(Self {
            registry,
            entries: RwLock::new(HashMap::new()),
            next_model_id: AtomicU32::new(0),
            recycled_model_ids: Mutex::new(Vec::new()),
            next_replica_unique_id: AtomicU32::new(1000),
        })
    }

    pub(crate) fn registry(&self) -> &Arc<ModelRegistry> {
        &self.registry
    }

    pub(crate) fn allocate_model_id(&self) -> u32 {
        loop {
            let candidate = self
                .recycled_model_ids
                .lock()
                .pop()
                .unwrap_or_else(|| self.next_model_id.fetch_add(1, Ordering::SeqCst));
            if self.registry.get(candidate).is_none() && !self.contains_pool(candidate) {
                return candidate;
            }
        }
    }

    pub(crate) fn release_model_id(&self, model_id: u32) {
        // Keep the entry (and therefore its lifecycle mutex) alive. A recycled
        // ID can be allocated before the releasing request has returned; using
        // a fresh mutex here would let the next load overlap that request.
        self.clear_runtime_entry(model_id, true);
        let mut recycled = self.recycled_model_ids.lock();
        if !recycled.contains(&model_id) {
            recycled.push(model_id);
        }
    }

    pub(crate) fn next_replica_unique_id(&self) -> u32 {
        loop {
            let candidate = self.next_replica_unique_id.fetch_add(1, Ordering::SeqCst);
            if self.registry.get(candidate).is_none() {
                return candidate;
            }
        }
    }

    pub(crate) async fn lock_lifecycle(&self, model_id: u32) -> OwnedMutexGuard<()> {
        let lifecycle = {
            let mut entries = self.entries.write();
            entries
                .entry(model_id)
                .or_insert_with(ModelEntry::empty)
                .lifecycle
                .clone()
        };
        lifecycle.lock_owned().await
    }

    pub(crate) fn contains_pool(&self, model_id: u32) -> bool {
        self.entries
            .read()
            .get(&model_id)
            .is_some_and(|entry| entry.pool.is_some())
    }

    pub(crate) fn pool(&self, model_id: u32) -> Option<Arc<ReplicaPool<Scheduler>>> {
        self.entries
            .read()
            .get(&model_id)
            .and_then(|entry| entry.pool.clone())
    }

    pub(crate) fn pools(&self) -> Vec<(u32, Arc<ReplicaPool<Scheduler>>)> {
        self.entries
            .read()
            .iter()
            .filter_map(|(model_id, entry)| entry.pool.clone().map(|pool| (*model_id, pool)))
            .collect()
    }

    pub(crate) fn model_path(&self, model_id: u32) -> Option<PathBuf> {
        self.entries
            .read()
            .get(&model_id)
            .and_then(|entry| entry.path.clone())
    }

    pub(crate) fn install_loaded(
        &self,
        model_id: u32,
        path: PathBuf,
        pool: Arc<ReplicaPool<Scheduler>>,
        swap_handles: Vec<EngineHandle>,
    ) {
        let mut entries = self.entries.write();
        let entry = entries.entry(model_id).or_insert_with(ModelEntry::empty);
        entry.path = Some(path);
        entry.pool = Some(pool);
        entry.swap_handles = swap_handles;
    }

    pub(crate) fn add_swap_handle(&self, model_id: u32, handle: EngineHandle) {
        self.entries
            .write()
            .entry(model_id)
            .or_insert_with(ModelEntry::empty)
            .swap_handles
            .push(handle);
    }

    pub(crate) fn pop_swap_handle(&self, model_id: u32) {
        if let Some(entry) = self.entries.write().get_mut(&model_id) {
            entry.swap_handles.pop();
        }
    }

    pub(crate) fn swap_handles(&self, model_id: u32) -> Option<Vec<EngineHandle>> {
        self.entries
            .read()
            .get(&model_id)
            .filter(|entry| !entry.swap_handles.is_empty())
            .map(|entry| entry.swap_handles.clone())
    }

    /// Resample every live backend's retained cross-domain footprint. The
    /// runtime memory wrapper reconciles the returned report into its elastic
    /// lease before this call returns.
    pub(crate) fn reconcile_memory_reports(&self) -> usize {
        let handles = self
            .entries
            .read()
            .values()
            .flat_map(|entry| entry.swap_handles.iter().cloned())
            .collect::<Vec<_>>();
        for handle in &handles {
            let _ = handle.actual_memory();
        }
        handles.len()
    }

    /// Stop runtime execution while retaining the package path and lifecycle
    /// entry so the same logical model can be started again.
    pub(crate) fn stop_runtime(&self, model_id: u32) {
        self.clear_runtime_entry(model_id, false);
    }

    pub(crate) fn remove(&self, model_id: u32) {
        // Preserve the lifecycle mutex for explicit-ID reuse, just as
        // `release_model_id` does for automatically assigned IDs.
        self.clear_runtime_entry(model_id, true);
    }

    fn clear_runtime_entry(&self, model_id: u32, clear_path: bool) {
        let mut entries = self.entries.write();
        let entry = entries.entry(model_id).or_insert_with(ModelEntry::empty);
        if clear_path {
            entry.path = None;
        }
        entry.pool = None;
        entry.swap_handles.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn lifecycle_lock_serializes_same_model_but_not_different_models() {
        let manager = ModelManager::new(Arc::new(ModelRegistry::new()));
        let first = manager.lock_lifecycle(7).await;
        assert!(manager
            .entries
            .read()
            .get(&7)
            .unwrap()
            .lifecycle
            .clone()
            .try_lock_owned()
            .is_err());
        let other = manager.lock_lifecycle(8).await;
        drop(other);
        drop(first);
        let again = manager.lock_lifecycle(7).await;
        drop(again);
    }

    #[test]
    fn recycled_model_ids_are_reused() {
        let manager = ModelManager::new(Arc::new(ModelRegistry::new()));
        let first = manager.allocate_model_id();
        manager.release_model_id(first);
        assert_eq!(manager.allocate_model_id(), first);
    }

    #[test]
    fn generated_ids_skip_explicit_registry_ids() {
        let registry = Arc::new(ModelRegistry::new());
        registry.upsert(ModelInfo::new(
            0,
            "explicit".to_string(),
            "1".to_string(),
            "onnx".to_string(),
            "cpu".to_string(),
            "basic".to_string(),
            "/tmp/model".to_string(),
        ));
        let manager = ModelManager::new(registry);

        assert_eq!(manager.allocate_model_id(), 1);
    }

    #[tokio::test]
    async fn releasing_an_id_keeps_its_lifecycle_lock() {
        let manager = ModelManager::new(Arc::new(ModelRegistry::new()));
        let model_id = manager.allocate_model_id();
        let guard = manager.lock_lifecycle(model_id).await;

        manager.release_model_id(model_id);
        assert_eq!(manager.allocate_model_id(), model_id);
        assert!(manager
            .entries
            .read()
            .get(&model_id)
            .unwrap()
            .lifecycle
            .clone()
            .try_lock_owned()
            .is_err());

        drop(guard);
    }
}
