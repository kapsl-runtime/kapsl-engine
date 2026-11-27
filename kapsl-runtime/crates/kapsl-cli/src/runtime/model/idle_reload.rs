//! Opt-in reload of idle-unloaded models on the next request.
//!
//! With `--idle-reload`, a model the idle unloader stopped is loaded again
//! when a request for it arrives. The request waits for the load. When the
//! memory is not available (for example a training job on the same GPU is
//! using it) the request fails with an error that says so, and the model
//! stays stopped until a later request can load it.

use super::*;
use futures::future::{BoxFuture, FutureExt, Shared};

/// How long requests fail fast with the last reload error before another
/// load is attempted. A denied load is not free, and during a long training
/// run every inference request would otherwise retry it.
pub(crate) const IDLE_RELOAD_RETRY_SECONDS: u64 = 10;

/// A loaded model that is ready to install.
pub(crate) type LoadedModel = (Arc<ReplicaPool<Scheduler>>, Vec<EngineHandle>);

/// Why a request could not be served by reloading its model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum IdleReloadError {
    /// The model is neither loaded nor waiting for an idle reload.
    NotLoaded,
    /// The memory authority denied the load: the model's memory is in use.
    MemoryUnavailable { message: String },
    /// The load failed for another reason.
    Failed { message: String },
}

impl IdleReloadError {
    fn from_load_error(model_id: u32, error: &(dyn std::error::Error + 'static)) -> Self {
        if error.downcast_ref::<MemoryAdmissionFailure>().is_some() {
            Self::MemoryUnavailable {
                message: format!(
                    "model {model_id} was stopped for being idle and cannot be reloaded right now: {error}. \
                     The memory it needs is in use, possibly by another process on the same device such as a training job. \
                     Retry when that memory is free."
                ),
            }
        } else {
            Self::failed(model_id, error)
        }
    }

    fn failed(model_id: u32, error: impl std::fmt::Display) -> Self {
        Self::Failed {
            message: format!(
                "model {model_id} was stopped for being idle and could not be reloaded: {error}"
            ),
        }
    }

    /// Seconds after which a retry may succeed, for errors worth retrying.
    pub(crate) fn retry_after_seconds(&self) -> Option<u64> {
        matches!(self, Self::MemoryUnavailable { .. }).then_some(IDLE_RELOAD_RETRY_SECONDS)
    }

    /// The closest engine error, for callers that are not HTTP routes.
    pub(crate) fn into_engine_error(self) -> EngineError {
        match self {
            Self::NotLoaded => EngineError::ModelNotLoaded,
            Self::MemoryUnavailable { message } => EngineError::resource_exhausted(message),
            Self::Failed { message } => EngineError::backend(message),
        }
    }
}

impl std::fmt::Display for IdleReloadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotLoaded => formatter.write_str("model is not loaded"),
            Self::MemoryUnavailable { message } | Self::Failed { message } => {
                formatter.write_str(message)
            }
        }
    }
}

type ReloadResult = Result<(), IdleReloadError>;
type LoadFuture = BoxFuture<'static, Result<LoadedModel, IdleReloadError>>;
type LoadFn = dyn Fn(u32, PathBuf, Option<ModelLoadPlacement>) -> LoadFuture + Send + Sync;

enum ReloadState {
    /// One reload is running; every request for the model waits on it.
    Loading(Shared<BoxFuture<'static, ReloadResult>>),
    /// The last reload failed; requests get its error until the retry time.
    Failed { at: Instant, error: IdleReloadError },
}

/// Reloads idle-unloaded models, one load per model at a time.
pub(crate) struct IdleReloader {
    models: Arc<ModelManager>,
    load: Box<LoadFn>,
    retry_after: Duration,
    state: Mutex<HashMap<u32, ReloadState>>,
}

impl IdleReloader {
    /// Reload through the same load transaction as the start API.
    pub(crate) fn for_runtime(model_runtime: Arc<ModelRuntime>) -> Arc<Self> {
        let models = model_runtime.models().clone();
        Self::new(
            models,
            Duration::from_secs(IDLE_RELOAD_RETRY_SECONDS),
            move |model_id, path, placement| {
                let model_runtime = model_runtime.clone();
                async move {
                    let placement = placement.unwrap_or_else(|| model_runtime.default_placement());
                    match model_runtime
                        .load_model_with_reclaim(model_id, &path, &placement)
                        .await
                    {
                        Ok(Ok(loaded)) => Ok(loaded),
                        Ok(Err(error)) => {
                            Err(IdleReloadError::from_load_error(model_id, error.as_ref()))
                        }
                        Err(error) => Err(IdleReloadError::failed(
                            model_id,
                            format!("loader task failed: {error}"),
                        )),
                    }
                }
                .boxed()
            },
        )
    }

    pub(crate) fn new(
        models: Arc<ModelManager>,
        retry_after: Duration,
        load: impl Fn(u32, PathBuf, Option<ModelLoadPlacement>) -> LoadFuture + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            models,
            load: Box::new(load),
            retry_after,
            state: Mutex::new(HashMap::new()),
        })
    }

    /// Make the model loaded, reloading it when the idle unloader stopped it.
    ///
    /// Call this when the model's pool could not be acquired. Concurrent
    /// callers share one load. The load runs in its own task, so a caller
    /// that goes away (a disconnected client) does not cancel it.
    pub(crate) async fn wake(self: &Arc<Self>, model_id: u32) -> ReloadResult {
        // A model the idle unloader is stopping right now still has its pool
        // but admits no requests; the reload below waits for that stop.
        if !self.models.contains_pool(model_id) && !self.models.reloads_on_request(model_id) {
            return Err(IdleReloadError::NotLoaded);
        }
        let reload = {
            let mut state = self.state.lock();
            match state.get(&model_id) {
                Some(ReloadState::Loading(reload)) => reload.clone(),
                Some(ReloadState::Failed { at, error }) if at.elapsed() < self.retry_after => {
                    return Err(error.clone());
                }
                _ => {
                    // The state lock is held until the entry is inserted, so
                    // the task cannot finish and clear it first.
                    let reloader = Arc::clone(self);
                    let task = tokio::spawn(async move {
                        let result = reloader.reload(model_id).await;
                        reloader.finish(model_id, &result);
                        result
                    });
                    let reloader = Arc::clone(self);
                    let reload = async move {
                        match task.await {
                            Ok(result) => result,
                            Err(error) => {
                                // The task died before it could record its
                                // outcome; do not leave the model "loading".
                                let result = Err(IdleReloadError::failed(
                                    model_id,
                                    format!("reload task failed: {error}"),
                                ));
                                reloader.finish(model_id, &result);
                                result
                            }
                        }
                    }
                    .boxed()
                    .shared();
                    state.insert(model_id, ReloadState::Loading(reload.clone()));
                    reload
                }
            }
        };
        reload.await
    }

    fn finish(&self, model_id: u32, result: &ReloadResult) {
        let mut state = self.state.lock();
        match result {
            Err(
                error
                @ (IdleReloadError::MemoryUnavailable { .. } | IdleReloadError::Failed { .. }),
            ) => {
                state.insert(
                    model_id,
                    ReloadState::Failed {
                        at: Instant::now(),
                        error: error.clone(),
                    },
                );
            }
            Ok(()) | Err(IdleReloadError::NotLoaded) => {
                state.remove(&model_id);
            }
        }
    }

    async fn reload(&self, model_id: u32) -> ReloadResult {
        // Serializes with the idle unloader, explicit start/stop/remove, hot
        // swap, and autoscaling for this model.
        let _lifecycle = self.models.lock_lifecycle(model_id).await;
        if self.models.contains_pool(model_id) {
            return Ok(());
        }
        if !self.models.reloads_on_request(model_id) {
            // Stopped explicitly or removed while this request waited.
            return Err(IdleReloadError::NotLoaded);
        }
        let Some(path) = self.models.model_path(model_id) else {
            return Err(IdleReloadError::NotLoaded);
        };
        let placement = self.models.load_placement(model_id);
        let registry = self.models.registry();
        let _ = registry.set_status(model_id, ModelStatus::Starting);
        log::info!(
            "[idle-reload] reloading model {} for an incoming request",
            model_id
        );
        let started = Instant::now();
        match (self.load)(model_id, path.clone(), placement).await {
            Ok((pool, handles)) => {
                self.models.install_loaded(model_id, path, pool, handles);
                let _ = registry.set_status(model_id, ModelStatus::Active);
                log::info!(
                    "[idle-reload] model {} is loaded again after {:.1}s",
                    model_id,
                    started.elapsed().as_secs_f64()
                );
                Ok(())
            }
            Err(error) => {
                // The model keeps its idle-unloaded mark, so a later request
                // tries again once the retry time has passed.
                let _ = registry.set_status(model_id, ModelStatus::Inactive);
                log::warn!("[idle-reload] {}", error);
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_ID: u32 = 7;

    fn idle_unloaded_model(reload_on_request: bool) -> Arc<ModelManager> {
        let registry = Arc::new(ModelRegistry::new());
        registry.register(ModelInfo::new(
            MODEL_ID,
            "model".to_string(),
            "1".to_string(),
            "onnx".to_string(),
            "cpu".to_string(),
            "basic".to_string(),
            "/tmp/model.aimod".to_string(),
        ));
        let models = ModelManager::new(registry);
        models.configure_idle_unload(IdleUnloadConfig {
            reload_on_request,
            ..IdleUnloadConfig::default()
        });
        models.install_loaded(
            MODEL_ID,
            PathBuf::from("/tmp/model.aimod"),
            empty_pool(),
            Vec::new(),
        );
        models.begin_idle_unload(MODEL_ID);
        models.stop_runtime(MODEL_ID);
        // The registry only reaches Inactive through Stopping, as a real
        // stop does.
        for status in [ModelStatus::Stopping, ModelStatus::Inactive] {
            models
                .registry()
                .set_status(MODEL_ID, status)
                .expect("a stop moves through Stopping to Inactive");
        }
        models
    }

    fn empty_pool() -> Arc<ReplicaPool<Scheduler>> {
        Arc::new(ReplicaPool::new(PoolStrategy::LeastLoaded))
    }

    /// A loader that counts its calls and resolves with `outcome` once
    /// `release` is notified.
    fn counting_reloader(
        models: &Arc<ModelManager>,
        retry_after: Duration,
        outcome: Result<(), IdleReloadError>,
    ) -> (
        Arc<IdleReloader>,
        Arc<AtomicUsize>,
        Arc<tokio::sync::Notify>,
    ) {
        let loads = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(tokio::sync::Notify::new());
        let reloader = IdleReloader::new(models.clone(), retry_after, {
            let loads = loads.clone();
            let release = release.clone();
            move |_, _, _| {
                loads.fetch_add(1, Ordering::SeqCst);
                let release = release.clone();
                let outcome = outcome.clone();
                async move {
                    release.notified().await;
                    outcome.map(|()| (empty_pool(), Vec::new()))
                }
                .boxed()
            }
        });
        (reloader, loads, release)
    }

    fn memory_unavailable() -> IdleReloadError {
        IdleReloadError::MemoryUnavailable {
            message: "memory is in use".to_string(),
        }
    }

    #[tokio::test]
    async fn concurrent_requests_share_one_reload() {
        let models = idle_unloaded_model(true);
        let (reloader, loads, release) =
            counting_reloader(&models, Duration::from_secs(10), Ok(()));

        let waiters: Vec<_> = (0..5)
            .map(|_| {
                let reloader = reloader.clone();
                tokio::spawn(async move { reloader.wake(MODEL_ID).await })
            })
            .collect();
        while loads.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            models.registry().get(MODEL_ID).unwrap().status,
            ModelStatus::Starting
        );
        release.notify_one();
        for waiter in waiters {
            assert_eq!(waiter.await.unwrap(), Ok(()));
        }

        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(models.contains_pool(MODEL_ID));
        assert!(!models.is_idle_unloaded(MODEL_ID));
        assert_eq!(
            models.registry().get(MODEL_ID).unwrap().status,
            ModelStatus::Active
        );
        // Nothing is left to reload; the loaded model answers immediately.
        assert_eq!(reloader.wake(MODEL_ID).await, Ok(()));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_request_that_goes_away_does_not_cancel_the_reload() {
        let models = idle_unloaded_model(true);
        let (reloader, loads, release) =
            counting_reloader(&models, Duration::from_secs(10), Ok(()));

        let impatient = tokio::spawn({
            let reloader = reloader.clone();
            async move { reloader.wake(MODEL_ID).await }
        });
        while loads.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        impatient.abort();
        let _ = impatient.await;

        release.notify_one();
        assert_eq!(reloader.wake(MODEL_ID).await, Ok(()));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(models.contains_pool(MODEL_ID));
    }

    #[tokio::test]
    async fn a_denied_reload_fails_fast_until_the_retry_time() {
        let models = idle_unloaded_model(true);
        let (reloader, loads, release) = counting_reloader(
            &models,
            Duration::from_millis(400),
            Err(memory_unavailable()),
        );

        release.notify_one();
        assert_eq!(reloader.wake(MODEL_ID).await, Err(memory_unavailable()));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(
            models.reloads_on_request(MODEL_ID),
            "the model stays reloadable"
        );
        assert_eq!(
            models.registry().get(MODEL_ID).unwrap().status,
            ModelStatus::Inactive
        );

        // Within the retry time the cached error is returned without loading.
        assert_eq!(reloader.wake(MODEL_ID).await, Err(memory_unavailable()));
        assert_eq!(loads.load(Ordering::SeqCst), 1);

        tokio::time::sleep(Duration::from_millis(500)).await;
        release.notify_one();
        assert_eq!(reloader.wake(MODEL_ID).await, Err(memory_unavailable()));
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn models_that_do_not_reload_on_request_are_left_stopped() {
        let stays_down = idle_unloaded_model(false);
        let (reloader, loads, _) = counting_reloader(&stays_down, Duration::from_secs(10), Ok(()));
        assert_eq!(
            reloader.wake(MODEL_ID).await,
            Err(IdleReloadError::NotLoaded)
        );
        assert_eq!(reloader.wake(99).await, Err(IdleReloadError::NotLoaded));

        // An explicit stop of an idle-unloaded model cancels its reload.
        let stopped = idle_unloaded_model(true);
        assert!(stopped.clear_idle_unloaded(MODEL_ID));
        let (reloader, stopped_loads, _) =
            counting_reloader(&stopped, Duration::from_secs(10), Ok(()));
        assert_eq!(
            reloader.wake(MODEL_ID).await,
            Err(IdleReloadError::NotLoaded)
        );

        assert_eq!(loads.load(Ordering::SeqCst), 0);
        assert_eq!(stopped_loads.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_reload_waits_for_the_unloader_and_repeats_the_requested_placement() {
        let models = idle_unloaded_model(true);
        models.remember_load_placement(MODEL_ID, ModelLoadPlacement::new("tensor-parallel", 2));
        let seen = Arc::new(Mutex::new(None));
        let reloader = IdleReloader::new(models.clone(), Duration::from_secs(10), {
            let seen = seen.clone();
            move |model_id, path, placement| {
                *seen.lock() = Some((model_id, path, placement.map(|p| (p.topology, p.tp_degree))));
                async { Ok::<LoadedModel, IdleReloadError>((empty_pool(), Vec::new())) }.boxed()
            }
        });

        // The reload queues behind whoever holds the lifecycle lock.
        let lifecycle = models.lock_lifecycle(MODEL_ID).await;
        let waiting = tokio::spawn({
            let reloader = reloader.clone();
            async move { reloader.wake(MODEL_ID).await }
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(seen.lock().is_none());
        drop(lifecycle);

        assert_eq!(waiting.await.unwrap(), Ok(()));
        assert_eq!(
            seen.lock().clone(),
            Some((
                MODEL_ID,
                PathBuf::from("/tmp/model.aimod"),
                Some(("tensor-parallel".to_string(), 2))
            ))
        );
    }

    #[test]
    fn a_memory_admission_denial_becomes_the_memory_in_use_error() {
        let mut plan = MemoryPlan::new();
        plan.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            MemoryOwner::new(MODEL_ID, 0),
            MemoryAllocationClass::PersistentWeights,
            2048,
        ));
        let denial: DynError = Box::new(MemoryAdmissionFailure::new(
            MemoryOwner::new(MODEL_ID, 0),
            1,
            &plan,
            "synthetic pressure",
        ));
        let error = IdleReloadError::from_load_error(MODEL_ID, denial.as_ref());
        assert!(matches!(error, IdleReloadError::MemoryUnavailable { .. }));
        let message = error.to_string();
        assert!(
            message.contains("cannot be reloaded right now"),
            "{message}"
        );
        assert!(message.contains("training job"), "{message}");

        let other: DynError = "backend crashed".into();
        assert!(matches!(
            IdleReloadError::from_load_error(MODEL_ID, other.as_ref()),
            IdleReloadError::Failed { .. }
        ));
    }

    #[test]
    fn memory_denials_explain_themselves_and_map_to_retryable_errors() {
        let denied = memory_unavailable();
        assert_eq!(
            denied.retry_after_seconds(),
            Some(IDLE_RELOAD_RETRY_SECONDS)
        );
        assert!(matches!(
            denied.into_engine_error(),
            EngineError::ResourceExhausted { .. }
        ));

        let failed = IdleReloadError::failed(MODEL_ID, "backend crashed");
        assert!(failed
            .to_string()
            .contains("could not be reloaded: backend crashed"));
        assert_eq!(failed.retry_after_seconds(), None);
        assert!(matches!(
            failed.into_engine_error(),
            EngineError::Backend { .. }
        ));
        assert!(matches!(
            IdleReloadError::NotLoaded.into_engine_error(),
            EngineError::ModelNotLoaded
        ));
    }
}
