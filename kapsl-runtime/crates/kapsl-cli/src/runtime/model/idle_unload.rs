//! Opt-in idle unloading.
//!
//! A model with an idle timeout that has served no inference for that long
//! is stopped so another process sharing the machine (for example a training
//! job on the same GPU) can use its memory. Timeouts come from
//! `--idle-unload-seconds`, per-model startup settings, or the start API;
//! without any of them every model stays loaded. A stopped model stays
//! registered with its package path. It is started again through the start
//! API, or by the next request when idle reload is enabled.

use super::lifecycle::model_authority_bytes;
use super::*;

pub(crate) struct IdleUnloadTaskConfig {
    pub(crate) model_runtime: Arc<ModelRuntime>,
}

/// The task always runs so that a timeout set later through the start API
/// takes effect; a pass over models without timeouts does nothing.
pub(crate) fn spawn_idle_unload_task(config: IdleUnloadTaskConfig) -> tokio::task::JoinHandle<()> {
    let IdleUnloadTaskConfig { model_runtime } = config;
    tokio::spawn(async move {
        loop {
            let shortest = model_runtime.models().shortest_idle_timeout();
            tokio::time::sleep(idle_unload_check_interval(shortest)).await;
            model_runtime.unload_idle_models(Instant::now());
        }
    })
}

/// Check often enough that a model is stopped within a few seconds of its
/// deadline, without polling more than once a second.
fn idle_unload_check_interval(shortest_timeout: Option<Duration>) -> Duration {
    const SLOWEST: Duration = Duration::from_secs(5);
    shortest_timeout.map_or(SLOWEST, |timeout| {
        timeout.clamp(Duration::from_secs(1), SLOWEST)
    })
}

/// Stop every model whose installed load has had no request in flight for at
/// least its idle timeout before `now`. Returns the IDs that were stopped.
///
/// A model is skipped, and checked again on the next pass, while a load,
/// stop, swap, or scale operation holds its lifecycle lock, or while work is
/// queued on it. The activity claim closes the model to new requests before
/// it is stopped, so a request arriving at the deadline either runs to
/// completion on the loaded model or finds it stopped.
pub(crate) fn unload_idle_models(
    now: Instant,
    models: &ModelManager,
    resources: &RuntimeResources,
) -> Vec<u32> {
    let model_ids: Vec<u32> = models
        .pools()
        .into_iter()
        .map(|(model_id, _)| model_id)
        .collect();
    let mut unloaded = Vec::new();
    for model_id in model_ids {
        let Some(timeout) = models.idle_timeout(model_id) else {
            continue;
        };
        let idle_long_enough = models
            .activity(model_id)
            .and_then(|activity| activity.idle_for(now))
            .is_some_and(|idle_for| idle_for >= timeout);
        if !idle_long_enough {
            continue;
        }
        let Some(_lifecycle) = models.try_lock_lifecycle(model_id) else {
            continue;
        };
        // Re-resolve under the lock: the load seen above may have been replaced.
        let (Some(pool), Some(activity)) = (models.pool(model_id), models.activity(model_id))
        else {
            continue;
        };
        let (high, low) = pool.get_queue_depth();
        drop(pool);
        let Some(idle_for) = activity.idle_for(now) else {
            continue;
        };
        if high.saturating_add(low) != 0 || !activity.try_claim_idle(timeout, now) {
            continue;
        }

        let held_bytes = model_authority_bytes(&resources.memory().snapshot(), model_id);
        // Announced first, so dropping the pool and marking the model
        // idle-unloaded is one step for concurrent requests.
        models.begin_idle_unload(model_id);
        match stop_model_and_replicas(model_id, models, resources) {
            Ok(_) => {
                log::info!(
                    "[idle-unload] stopped model {} after {}s without inference requests ({} bytes of tracked memory); {}",
                    model_id,
                    idle_for.as_secs(),
                    held_bytes,
                    if models.reloads_on_request(model_id) {
                        "the next request for it reloads it"
                    } else {
                        "start it again with POST /api/models/start"
                    },
                );
                unloaded.push(model_id);
            }
            Err(error) => {
                models.cancel_idle_unload(model_id);
                activity.release_idle_claim();
                log::warn!(
                    "[idle-unload] could not stop idle model {}; it stays loaded: {}",
                    model_id,
                    error
                );
            }
        }
    }
    unloaded
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_ID: u32 = 7;
    const TIMEOUT: Duration = Duration::from_secs(60);

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

    fn load(models: &ModelManager, model_id: u32) {
        models.registry().register(ModelInfo::new(
            model_id,
            format!("model-{model_id}"),
            "1".to_string(),
            "onnx".to_string(),
            "cpu".to_string(),
            "basic".to_string(),
            "/tmp/model.aimod".to_string(),
        ));
        models.install_loaded(
            model_id,
            PathBuf::from("/tmp/model.aimod"),
            Arc::new(ReplicaPool::new(PoolStrategy::LeastLoaded)),
            Vec::new(),
        );
    }

    /// One loaded model under a process-wide idle timeout.
    fn loaded_model() -> (Arc<ModelManager>, Arc<RuntimeResources>) {
        loaded_models(
            &[MODEL_ID],
            IdleUnloadConfig {
                default_timeout: Some(TIMEOUT),
                ..IdleUnloadConfig::default()
            },
        )
    }

    fn loaded_models(
        model_ids: &[u32],
        config: IdleUnloadConfig,
    ) -> (Arc<ModelManager>, Arc<RuntimeResources>) {
        let models = ModelManager::new(Arc::new(ModelRegistry::new()));
        models.configure_idle_unload(config);
        for model_id in model_ids {
            load(&models, *model_id);
        }
        let resources = RuntimeResources::new(&test_device_info()).expect("runtime resources");
        (models, resources)
    }

    fn after(idle: Duration) -> Instant {
        Instant::now() + idle
    }

    fn sorted(mut model_ids: Vec<u32>) -> Vec<u32> {
        model_ids.sort_unstable();
        model_ids
    }

    #[test]
    fn check_interval_follows_the_shortest_timeout_within_one_to_five_seconds() {
        let seconds = Duration::from_secs;
        assert_eq!(idle_unload_check_interval(None), seconds(5));
        assert_eq!(idle_unload_check_interval(Some(seconds(1))), seconds(1));
        assert_eq!(idle_unload_check_interval(Some(seconds(3))), seconds(3));
        assert_eq!(idle_unload_check_interval(Some(seconds(3600))), seconds(5));
    }

    #[test]
    fn idle_model_is_stopped_and_stays_restartable() {
        let (models, resources) = loaded_model();

        assert!(unload_idle_models(Instant::now(), &models, &resources).is_empty());
        assert!(models.contains_pool(MODEL_ID));

        assert_eq!(
            unload_idle_models(after(TIMEOUT * 2), &models, &resources),
            [MODEL_ID]
        );
        assert!(!models.contains_pool(MODEL_ID));
        assert_eq!(
            models.registry().get(MODEL_ID).unwrap().status,
            ModelStatus::Inactive
        );
        assert_eq!(
            models.model_path(MODEL_ID),
            Some(PathBuf::from("/tmp/model.aimod"))
        );
        // Without idle reload it stays down and is not served.
        assert!(models.is_idle_unloaded(MODEL_ID));
        assert!(!models.reloads_on_request(MODEL_ID));
        assert!(!models.can_serve(MODEL_ID));
        // Nothing is left to stop on the next pass.
        assert!(unload_idle_models(after(TIMEOUT * 2), &models, &resources).is_empty());
    }

    #[test]
    fn with_idle_reload_a_stopped_model_stays_servable() {
        let (models, resources) = loaded_models(
            &[MODEL_ID],
            IdleUnloadConfig {
                default_timeout: Some(TIMEOUT),
                reload_on_request: true,
                ..IdleUnloadConfig::default()
            },
        );

        assert_eq!(
            unload_idle_models(after(TIMEOUT * 2), &models, &resources),
            [MODEL_ID]
        );
        assert!(!models.contains_pool(MODEL_ID));
        assert!(models.is_idle_unloaded(MODEL_ID));
        assert!(models.reloads_on_request(MODEL_ID));
        assert!(models.can_serve(MODEL_ID));
        assert_eq!(models.servable_model_ids(), [MODEL_ID]);
    }

    #[test]
    fn models_stay_loaded_when_nothing_sets_a_timeout() {
        let (models, resources) = loaded_models(&[MODEL_ID], IdleUnloadConfig::default());

        assert_eq!(models.shortest_idle_timeout(), None);
        assert!(unload_idle_models(after(TIMEOUT * 1000), &models, &resources).is_empty());
        assert!(models.contains_pool(MODEL_ID));
    }

    #[test]
    fn each_model_follows_its_own_timeout() {
        const DEFAULTED: u32 = 1;
        const PINNED: u32 = 2;
        const QUICK: u32 = 3;
        let (models, resources) = loaded_models(
            &[DEFAULTED, PINNED, QUICK],
            IdleUnloadConfig {
                default_timeout: Some(TIMEOUT),
                model_timeouts: HashMap::from([(PINNED, None), (QUICK, Some(TIMEOUT / 6))]),
                reload_on_request: false,
            },
        );
        assert_eq!(models.shortest_idle_timeout(), Some(TIMEOUT / 6));

        assert_eq!(
            unload_idle_models(after(TIMEOUT / 2), &models, &resources),
            [QUICK]
        );
        assert_eq!(
            unload_idle_models(after(TIMEOUT * 2), &models, &resources),
            [DEFAULTED]
        );
        // A model configured with 0 is never unloaded.
        assert!(unload_idle_models(after(TIMEOUT * 1000), &models, &resources).is_empty());
        assert!(models.contains_pool(PINNED));
    }

    #[test]
    fn a_start_api_timeout_wins_over_startup_configuration() {
        const KEPT: u32 = 1;
        const ADDED: u32 = 2;
        let (models, resources) = loaded_models(
            &[KEPT, ADDED],
            IdleUnloadConfig {
                default_timeout: Some(TIMEOUT),
                model_timeouts: HashMap::from([(ADDED, None)]),
                reload_on_request: false,
            },
        );
        models.set_idle_timeout_override(KEPT, None);
        models.set_idle_timeout_override(ADDED, Some(TIMEOUT / 6));
        assert_eq!(models.idle_timeout(KEPT), None);
        assert_eq!(models.idle_timeout(ADDED), Some(TIMEOUT / 6));

        assert_eq!(
            sorted(unload_idle_models(after(TIMEOUT * 2), &models, &resources)),
            [ADDED]
        );
        assert!(models.contains_pool(KEPT));

        // A timeout can enable unloading for one model with no default at all.
        let (models, resources) = loaded_models(&[KEPT, ADDED], IdleUnloadConfig::default());
        models.set_idle_timeout_override(ADDED, Some(TIMEOUT));
        assert_eq!(
            unload_idle_models(after(TIMEOUT * 2), &models, &resources),
            [ADDED]
        );
    }

    #[test]
    fn an_in_flight_request_keeps_its_model_loaded() {
        let (models, resources) = loaded_model();
        let (_pool, request) = models.pool_for_request(MODEL_ID).expect("loaded");

        assert!(unload_idle_models(after(TIMEOUT * 2), &models, &resources).is_empty());
        assert!(models.contains_pool(MODEL_ID));

        drop(request);
        assert!(
            unload_idle_models(after(TIMEOUT / 2), &models, &resources).is_empty(),
            "the idle timer restarts when the last request ends"
        );
        assert_eq!(
            unload_idle_models(after(TIMEOUT * 2), &models, &resources),
            [MODEL_ID]
        );
    }

    #[tokio::test]
    async fn a_running_lifecycle_operation_defers_the_unload() {
        let (models, resources) = loaded_model();
        let operation = models.lock_lifecycle(MODEL_ID).await;

        assert!(unload_idle_models(after(TIMEOUT * 2), &models, &resources).is_empty());
        assert!(models.contains_pool(MODEL_ID));

        drop(operation);
        assert_eq!(
            unload_idle_models(after(TIMEOUT * 2), &models, &resources),
            [MODEL_ID]
        );
    }
}
