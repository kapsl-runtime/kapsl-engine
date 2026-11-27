//! Opt-in idle unloading.
//!
//! When enabled with `--idle-unload-seconds`, a model that has served no
//! inference for the configured time is stopped so another process sharing
//! the machine (for example a training job on the same GPU) can use its
//! memory. The model stays registered with its package path and can be
//! started again through the normal start API.

use super::lifecycle::model_authority_bytes;
use super::*;

pub(crate) struct IdleUnloadTaskConfig {
    pub(crate) timeout: Duration,
    pub(crate) model_runtime: Arc<ModelRuntime>,
}

pub(crate) fn spawn_idle_unload_task(config: IdleUnloadTaskConfig) -> tokio::task::JoinHandle<()> {
    let IdleUnloadTaskConfig {
        timeout,
        model_runtime,
    } = config;
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(idle_unload_check_interval(timeout));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            model_runtime.unload_idle_models(timeout, Instant::now());
        }
    })
}

/// Check often enough that a model is stopped within a few seconds of its
/// deadline, without polling more than once a second.
fn idle_unload_check_interval(timeout: Duration) -> Duration {
    timeout.clamp(Duration::from_secs(1), Duration::from_secs(5))
}

/// Stop every model whose installed load has had no request in flight for at
/// least `timeout` before `now`. Returns the IDs that were stopped.
///
/// A model is skipped, and checked again on the next tick, while a load,
/// stop, swap, or scale operation holds its lifecycle lock, or while work is
/// queued on it. The activity claim closes the model to new requests before
/// it is stopped, so a request arriving at the deadline either runs to
/// completion on the loaded model or sees it as not loaded.
pub(crate) fn unload_idle_models(
    timeout: Duration,
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
        match stop_model_and_replicas(model_id, models, resources) {
            Ok(_) => {
                log::info!(
                    "[idle-unload] stopped model {} after {}s without inference requests ({} bytes of tracked memory); start it again with POST /api/models/start",
                    model_id,
                    idle_for.as_secs(),
                    held_bytes,
                );
                unloaded.push(model_id);
            }
            Err(error) => {
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

    fn loaded_model() -> (Arc<ModelManager>, Arc<RuntimeResources>) {
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
        models.install_loaded(
            MODEL_ID,
            PathBuf::from("/tmp/model.aimod"),
            Arc::new(ReplicaPool::new(PoolStrategy::LeastLoaded)),
            Vec::new(),
        );
        let resources = RuntimeResources::new(&test_device_info()).expect("runtime resources");
        (models, resources)
    }

    fn after_timeout() -> Instant {
        Instant::now() + TIMEOUT * 2
    }

    #[test]
    fn check_interval_stays_between_one_and_five_seconds() {
        assert_eq!(
            idle_unload_check_interval(Duration::from_secs(1)),
            Duration::from_secs(1)
        );
        assert_eq!(
            idle_unload_check_interval(Duration::from_secs(3)),
            Duration::from_secs(3)
        );
        assert_eq!(
            idle_unload_check_interval(Duration::from_secs(3600)),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn idle_model_is_stopped_and_stays_restartable() {
        let (models, resources) = loaded_model();

        assert!(unload_idle_models(TIMEOUT, Instant::now(), &models, &resources).is_empty());
        assert!(models.contains_pool(MODEL_ID));

        assert_eq!(
            unload_idle_models(TIMEOUT, after_timeout(), &models, &resources),
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
        // Nothing is left to stop on the next tick.
        assert!(unload_idle_models(TIMEOUT, after_timeout(), &models, &resources).is_empty());
    }

    #[test]
    fn an_in_flight_request_keeps_its_model_loaded() {
        let (models, resources) = loaded_model();
        let (_pool, request) = models.pool_for_request(MODEL_ID).expect("loaded");

        assert!(unload_idle_models(TIMEOUT, after_timeout(), &models, &resources).is_empty());
        assert!(models.contains_pool(MODEL_ID));

        drop(request);
        let ended = Instant::now();
        assert!(
            unload_idle_models(TIMEOUT, ended + TIMEOUT / 2, &models, &resources).is_empty(),
            "the idle timer restarts when the last request ends"
        );
        assert_eq!(
            unload_idle_models(TIMEOUT, ended + TIMEOUT * 2, &models, &resources),
            [MODEL_ID]
        );
    }

    #[tokio::test]
    async fn a_running_lifecycle_operation_defers_the_unload() {
        let (models, resources) = loaded_model();
        let operation = models.lock_lifecycle(MODEL_ID).await;

        assert!(unload_idle_models(TIMEOUT, after_timeout(), &models, &resources).is_empty());
        assert!(models.contains_pool(MODEL_ID));

        drop(operation);
        assert_eq!(
            unload_idle_models(TIMEOUT, after_timeout(), &models, &resources),
            [MODEL_ID]
        );
    }
}
