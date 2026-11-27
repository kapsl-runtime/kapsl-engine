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
    /// Request activity of the installed load; present exactly when `pool` is.
    activity: Option<Arc<ModelActivity>>,
    /// Topology requested through the start API; `None` uses the planner's
    /// defaults. Kept across a stop so an idle reload repeats the same load.
    placement: Option<ModelLoadPlacement>,
    /// Set while the idle unloader has this model stopped. Any load, explicit
    /// stop, or removal clears it.
    idle_unloaded: Option<IdleUnloaded>,
    /// The idle unloader is about to stop this model; the stop turns this
    /// into `idle_unloaded`.
    idle_unload_pending: bool,
    /// Idle timeout given for this model through the start API. It wins over
    /// startup configuration; the inner `None` keeps the model loaded.
    idle_timeout_override: Option<Option<Duration>>,
}

impl ModelEntry {
    fn empty() -> Self {
        Self {
            path: None,
            pool: None,
            swap_handles: Vec::new(),
            lifecycle: Arc::new(AsyncMutex::new(())),
            activity: None,
            placement: None,
            idle_unloaded: None,
            idle_unload_pending: false,
            idle_timeout_override: None,
        }
    }

    fn reloads_on_request(&self) -> bool {
        self.pool.is_none()
            && self.path.is_some()
            && self
                .idle_unloaded
                .is_some_and(|idle| idle.reload_on_request)
    }
}

#[derive(Clone, Copy, Debug)]
struct IdleUnloaded {
    /// Policy when the model was stopped: reload it for the next request.
    reload_on_request: bool,
}

/// Process-wide idle unloading policy, fixed at startup. The default keeps
/// every model loaded until it is stopped explicitly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct IdleUnloadConfig {
    /// Timeout for models without their own setting; `None` keeps them loaded.
    pub(crate) default_timeout: Option<Duration>,
    /// Per-model timeouts by model ID; `None` keeps that model loaded.
    pub(crate) model_timeouts: HashMap<u32, Option<Duration>>,
    /// Reload an idle-unloaded model when the next request for it arrives.
    pub(crate) reload_on_request: bool,
}

/// Single owner for model identity, runtime replicas, hot-swap handles, paths,
/// and lifecycle serialization.
pub(crate) struct ModelManager {
    registry: Arc<ModelRegistry>,
    entries: RwLock<HashMap<u32, ModelEntry>>,
    next_model_id: AtomicU32,
    recycled_model_ids: Mutex<Vec<u32>>,
    next_replica_unique_id: AtomicU32,
    idle_unload: RwLock<IdleUnloadConfig>,
}

impl ModelManager {
    pub(crate) fn new(registry: Arc<ModelRegistry>) -> Arc<Self> {
        Arc::new(Self {
            registry,
            entries: RwLock::new(HashMap::new()),
            next_model_id: AtomicU32::new(0),
            recycled_model_ids: Mutex::new(Vec::new()),
            next_replica_unique_id: AtomicU32::new(1000),
            idle_unload: RwLock::new(IdleUnloadConfig::default()),
        })
    }

    /// Install the startup idle unloading policy. Called once, before any
    /// model is loaded.
    pub(crate) fn configure_idle_unload(&self, config: IdleUnloadConfig) {
        *self.idle_unload.write() = config;
    }

    /// How long this model may stay loaded without requests. The start API's
    /// per-model value wins, then startup per-model configuration, then the
    /// process default. `None` keeps the model loaded.
    pub(crate) fn idle_timeout(&self, model_id: u32) -> Option<Duration> {
        if let Some(timeout) = self
            .entries
            .read()
            .get(&model_id)
            .and_then(|entry| entry.idle_timeout_override)
        {
            return timeout;
        }
        let config = self.idle_unload.read();
        match config.model_timeouts.get(&model_id) {
            Some(timeout) => *timeout,
            None => config.default_timeout,
        }
    }

    /// Shortest idle timeout among the loaded models, if any has one.
    pub(crate) fn shortest_idle_timeout(&self) -> Option<Duration> {
        self.pools()
            .into_iter()
            .filter_map(|(model_id, _)| self.idle_timeout(model_id))
            .min()
    }

    /// Set this model's own idle timeout; `None` keeps it loaded.
    pub(crate) fn set_idle_timeout_override(&self, model_id: u32, timeout: Option<Duration>) {
        self.entries
            .write()
            .entry(model_id)
            .or_insert_with(ModelEntry::empty)
            .idle_timeout_override = Some(timeout);
    }

    /// Announce that the idle unloader is about to stop this model. The stop
    /// then marks the model idle-unloaded in the same step that drops its
    /// pool, so no request sees it as plainly "not loaded" in between.
    /// Callers hold the model's lifecycle lock and call
    /// [`Self::cancel_idle_unload`] if the stop fails.
    pub(crate) fn begin_idle_unload(&self, model_id: u32) {
        if let Some(entry) = self.entries.write().get_mut(&model_id) {
            entry.idle_unload_pending = true;
        }
    }

    /// Withdraw [`Self::begin_idle_unload`] after a stop that did not happen.
    pub(crate) fn cancel_idle_unload(&self, model_id: u32) {
        if let Some(entry) = self.entries.write().get_mut(&model_id) {
            entry.idle_unload_pending = false;
        }
    }

    /// Whether the model is stopped because it was idle.
    pub(crate) fn is_idle_unloaded(&self, model_id: u32) -> bool {
        self.entries
            .read()
            .get(&model_id)
            .is_some_and(|entry| entry.pool.is_none() && entry.idle_unloaded.is_some())
    }

    /// Forget that the model was stopped for being idle, so it stays down
    /// until it is started explicitly. Returns whether it was marked.
    pub(crate) fn clear_idle_unloaded(&self, model_id: u32) -> bool {
        self.entries
            .write()
            .get_mut(&model_id)
            .is_some_and(|entry| entry.idle_unloaded.take().is_some())
    }

    /// Whether the model is stopped for being idle and a request reloads it.
    pub(crate) fn reloads_on_request(&self, model_id: u32) -> bool {
        self.entries
            .read()
            .get(&model_id)
            .is_some_and(ModelEntry::reloads_on_request)
    }

    /// Whether a request for this model can be served: it is loaded, or it
    /// was stopped for being idle and reloads on request.
    pub(crate) fn can_serve(&self, model_id: u32) -> bool {
        self.entries
            .read()
            .get(&model_id)
            .is_some_and(|entry| entry.pool.is_some() || entry.reloads_on_request())
    }

    /// IDs of every model for which [`Self::can_serve`] holds.
    pub(crate) fn servable_model_ids(&self) -> Vec<u32> {
        self.entries
            .read()
            .iter()
            .filter(|(_, entry)| entry.pool.is_some() || entry.reloads_on_request())
            .map(|(model_id, _)| *model_id)
            .collect()
    }

    /// Remember the topology a start request asked for.
    pub(crate) fn remember_load_placement(&self, model_id: u32, placement: ModelLoadPlacement) {
        self.entries
            .write()
            .entry(model_id)
            .or_insert_with(ModelEntry::empty)
            .placement = Some(placement);
    }

    /// Topology of the model's last API-requested load, if it had one.
    pub(crate) fn load_placement(&self, model_id: u32) -> Option<ModelLoadPlacement> {
        self.entries
            .read()
            .get(&model_id)
            .and_then(|entry| entry.placement.clone())
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

    /// Take the model's lifecycle lock only if no load, stop, swap, or scale
    /// operation holds it. `None` also covers a model with no entry.
    pub(crate) fn try_lock_lifecycle(&self, model_id: u32) -> Option<OwnedMutexGuard<()>> {
        let lifecycle = self.entries.read().get(&model_id)?.lifecycle.clone();
        lifecycle.try_lock_owned().ok()
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

    /// Resolve the live pool for one inference request and count the request
    /// as in flight until the returned guard is dropped. `None` when the model
    /// is not loaded or the idle unloader is stopping it.
    pub(crate) fn pool_for_request(
        &self,
        model_id: u32,
    ) -> Option<(Arc<ReplicaPool<Scheduler>>, ModelActivityGuard)> {
        let entries = self.entries.read();
        let entry = entries.get(&model_id)?;
        let pool = entry.pool.clone()?;
        let guard = entry.activity.as_ref()?.try_begin_request()?;
        Some((pool, guard))
    }

    /// Request activity of the model's installed load, if it is loaded.
    pub(crate) fn activity(&self, model_id: u32) -> Option<Arc<ModelActivity>> {
        self.entries
            .read()
            .get(&model_id)
            .and_then(|entry| entry.activity.clone())
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
        entry.activity = Some(ModelActivity::new());
        entry.idle_unloaded = None;
        entry.idle_unload_pending = false;
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
        let reload_on_request = self.idle_unload.read().reload_on_request;
        let mut entries = self.entries.write();
        let entry = entries.entry(model_id).or_insert_with(ModelEntry::empty);
        // Only a stop the idle unloader announced leaves the model
        // idle-unloaded; every other stop or removal clears that state.
        let stopped_for_idle = std::mem::take(&mut entry.idle_unload_pending) && !clear_path;
        if clear_path {
            // The identity is gone: drop everything a later model reusing
            // this ID must not inherit.
            entry.path = None;
            entry.placement = None;
            entry.idle_timeout_override = None;
        }
        entry.pool = None;
        entry.swap_handles.clear();
        entry.activity = None;
        entry.idle_unloaded = stopped_for_idle.then_some(IdleUnloaded { reload_on_request });
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

    #[test]
    fn each_install_gets_fresh_request_activity() {
        let manager = ModelManager::new(Arc::new(ModelRegistry::new()));
        let install = |manager: &ModelManager| {
            manager.install_loaded(
                3,
                PathBuf::from("/tmp/model.aimod"),
                Arc::new(ReplicaPool::new(PoolStrategy::LeastLoaded)),
                Vec::new(),
            )
        };
        assert!(manager.pool_for_request(3).is_none());

        install(&manager);
        let (_pool, stale_guard) = manager.pool_for_request(3).expect("loaded");
        let first = manager.activity(3).expect("activity");
        assert!(manager.try_lock_lifecycle(3).is_some());

        manager.stop_runtime(3);
        assert!(manager.activity(3).is_none());
        assert!(manager.pool_for_request(3).is_none());

        install(&manager);
        let second = manager.activity(3).expect("activity");
        assert!(!Arc::ptr_eq(&first, &second));
        // A guard from the previous load must not keep the new load busy.
        assert_eq!(second.in_flight(), 0);
        drop(stale_guard);
        assert_eq!(second.in_flight(), 0);

        let later = Instant::now() + Duration::from_secs(120);
        assert!(second.try_claim_idle(Duration::from_secs(60), later));
        assert!(manager.pool_for_request(3).is_none());
        second.release_idle_claim();
        assert!(manager.pool_for_request(3).is_some());
    }

    #[test]
    fn idle_state_follows_the_model_lifecycle() {
        let manager = ModelManager::new(Arc::new(ModelRegistry::new()));
        manager.configure_idle_unload(IdleUnloadConfig {
            reload_on_request: true,
            ..IdleUnloadConfig::default()
        });
        let install = |manager: &ModelManager| {
            manager.install_loaded(
                3,
                PathBuf::from("/tmp/model.aimod"),
                Arc::new(ReplicaPool::new(PoolStrategy::LeastLoaded)),
                Vec::new(),
            )
        };
        let half_minute = Some(Duration::from_secs(30));
        install(&manager);
        manager.remember_load_placement(3, ModelLoadPlacement::new("tensor-parallel", 2));
        manager.set_idle_timeout_override(3, half_minute);

        // An explicit stop is not an idle unload: nothing reloads it.
        manager.stop_runtime(3);
        assert!(!manager.is_idle_unloaded(3));
        assert!(!manager.can_serve(3));

        // A stop the idle unloader announced leaves the model reloadable.
        install(&manager);
        manager.begin_idle_unload(3);
        assert!(!manager.is_idle_unloaded(3), "still loaded");
        manager.stop_runtime(3);
        assert!(manager.is_idle_unloaded(3));
        assert!(manager.reloads_on_request(3));
        assert!(manager.can_serve(3));
        // What the reload needs survives the stop.
        assert_eq!(manager.load_placement(3).map(|p| p.tp_degree), Some(2));
        assert_eq!(manager.idle_timeout(3), half_minute);

        // Any load clears the mark.
        install(&manager);
        assert!(!manager.is_idle_unloaded(3));
        assert!(!manager.reloads_on_request(3));
        assert!(manager.can_serve(3));

        // Stopping an idle-unloaded model explicitly keeps it down.
        manager.begin_idle_unload(3);
        manager.stop_runtime(3);
        assert!(manager.clear_idle_unloaded(3));
        assert!(!manager.clear_idle_unloaded(3));
        assert!(!manager.can_serve(3));

        // A withdrawn announcement does not mark a later, explicit stop.
        install(&manager);
        manager.begin_idle_unload(3);
        manager.cancel_idle_unload(3);
        manager.stop_runtime(3);
        assert!(!manager.is_idle_unloaded(3));

        // Removal forgets everything a later model with this ID must not inherit.
        install(&manager);
        manager.begin_idle_unload(3);
        manager.remove(3);
        assert!(!manager.is_idle_unloaded(3));
        assert!(!manager.can_serve(3));
        assert!(manager.load_placement(3).is_none());
        assert_eq!(manager.idle_timeout(3), None);
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
