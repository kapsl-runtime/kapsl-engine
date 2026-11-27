use super::*;

impl ModelRuntime {
    /// Scale up a model by adding a new replica.
    pub(crate) async fn scale_up_model(
        &self,
        base_model_id: u32,
        replica_id: u32,
        unique_id: u32,
        model_path: &Path,
    ) -> Result<(Arc<Scheduler>, EngineHandle), DynError> {
        #[cfg(feature = "gpu-device-pool")]
        self.resources
            .attach_device_memory_metrics(self.shared_metrics.clone());
        log::info!(
            "Scaling up Model ID {} - Creating replica #{}",
            base_model_id,
            replica_id
        );

        let planner = Arc::clone(&self.load_planner);
        let package_path = model_path.to_path_buf();
        let plan = run_off_runtime("replica load planning", move || {
            planner.prepare(base_model_id, unique_id, replica_id, &package_path, None)
        })
        .await?;
        #[cfg(feature = "gpu-device-pool")]
        {
            let bootstrap =
                device_memory_bootstrap_plan(std::iter::once(&plan), self.device_info())?;
            self.resources.ensure_device_pools(&bootstrap)?;
        }
        let LoadedReplica {
            scheduler,
            mut swap_handles,
            model_info,
        } = load_replica(
            plan,
            ReplicaLoadRole::Autoscaled,
            self.device_info(),
            self.resources.clone(),
            &self.shared_metrics,
        )
        .await?;
        if swap_handles.len() != 1 {
            return Err(format!(
                "Autoscaled replica {} for model {} produced {} engines; exactly one is required",
                replica_id,
                base_model_id,
                swap_handles.len()
            )
            .into());
        }
        let swap_handle = swap_handles.pop().expect("length checked above");
        self.models.registry().upsert(model_info);

        log::info!(
            "✓ Replica #{} started for Model ID {}",
            replica_id,
            base_model_id
        );

        Ok((scheduler, swap_handle))
    }

    /// Scale down a model by removing a replica.
    pub(crate) async fn scale_down_model(
        &self,
        base_model_id: u32,
        replica_id: u32,
        unique_id: u32,
    ) -> Result<(), DynError> {
        scale_down_model(base_model_id, replica_id, unique_id, &self.models).await
    }
}

/// Runs blocking lifecycle work on the blocking pool instead of the calling
/// async worker.
///
/// Planning a replica opens its package, which unpacks the whole archive:
/// about twenty seconds for a 3 GB model. The autoscaler's timer wakes it on
/// the worker that was polling the runtime's I/O and timer driver, and a
/// worker with a single task does not wake a sleeping peer to take the driver
/// over. Blocking there stops every socket and timer in the process until the
/// work returns. That includes the KV control socket, whose clients give up
/// after two seconds, so a busy vLLM replica lost its leases and died whenever
/// a second replica was being prepared.
async fn run_off_runtime<T, F>(what: &'static str, work: F) -> Result<T, DynError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, DynError> + Send + 'static,
{
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|error| format!("{what} task failed before completion: {error}"))?
}

/// Internal implementation retained for priority-reclamation paths that run
/// before a complete [`ModelRuntime`] is available in unit tests.
pub(crate) async fn scale_down_model(
    base_model_id: u32,
    replica_id: u32,
    unique_id: u32,
    models: &ModelManager,
) -> Result<(), DynError> {
    log::info!(
        "Scaling down Model ID {} - Removing replica #{}",
        base_model_id,
        replica_id
    );

    if let Err(error) = models
        .registry()
        .set_status(unique_id, ModelStatus::Stopping)
    {
        log::error!("Failed to set status to Stopping for {unique_id}: {error}");
    }

    let pool = models.pool(base_model_id);
    if let Some(pool) = pool {
        let removed = pool.remove_replica(replica_id);
        if !removed {
            let _ = models.registry().set_status(unique_id, ModelStatus::Active);
            return Err(format!(
                "Replica #{} was not present in pool for model {}",
                replica_id, base_model_id
            )
            .into());
        }
    } else {
        let _ = models.registry().set_status(unique_id, ModelStatus::Active);
        return Err(format!("Replica pool not found for model {base_model_id}").into());
    }

    // Scale-up appends one hot-swap handle per replica; scale-down removes
    // replicas in descending ID order, so the matching handle is the tail.
    models.pop_swap_handle(base_model_id);

    if let Err(error) = models
        .registry()
        .set_status(unique_id, ModelStatus::Inactive)
    {
        log::error!("Failed to set status to Inactive for {unique_id}: {error}");
    }

    log::info!(
        "✓ Replica #{} stopped for Model ID {}",
        replica_id,
        base_model_id
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// On a single-threaded runtime nothing else runs while a task blocks its
    /// thread, so a ticker that keeps advancing proves the work left it.
    #[tokio::test]
    async fn blocking_lifecycle_work_leaves_the_runtime_running() {
        let ticks = Arc::new(AtomicUsize::new(0));
        let ticker = tokio::spawn({
            let ticks = Arc::clone(&ticks);
            async move {
                loop {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    ticks.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        let value = run_off_runtime("test work", || {
            std::thread::sleep(Duration::from_millis(250));
            Ok(7)
        })
        .await
        .unwrap();
        ticker.abort();

        assert_eq!(value, 7);
        assert!(
            ticks.load(Ordering::Relaxed) >= 10,
            "the runtime stalled while lifecycle work ran"
        );
    }

    #[tokio::test]
    async fn failed_lifecycle_work_names_the_step() {
        let error = run_off_runtime::<(), _>("test work", || panic!("lifecycle work panicked"))
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("test work task failed before completion"),
            "{error}"
        );
    }
}
