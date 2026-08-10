use super::*;

pub(crate) fn build_system_routes(
    models: Arc<ModelManager>,
    device_info_for_api: Arc<DeviceInfo>,
    runtime_samples_clone: Arc<RwLock<RuntimeSamples>>,
    runtime_pressure_state: Arc<AtomicU8>,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let models_for_health = models.clone();
    let health = warp::path!("api" / "health").and(warp::get()).map(move || {
        let total = models_for_health.registry().count();
        let mut healthy = 0;
        let mut unhealthy = 0;

        for model in models_for_health.registry().list() {
            if let Some(pool) = models_for_health.pool(model.id) {
                if pool.is_healthy() {
                    healthy += 1;
                } else {
                    unhealthy += 1;
                }
            } else {
                healthy += 1;
            }
        }

        let overall_status = if unhealthy == 0 {
            "healthy"
        } else {
            "degraded"
        };
        let response = json!({
            "status": overall_status.to_string(),
            "total_models": total,
            "healthy_models": healthy,
            "unhealthy_models": unhealthy,
        });
        warp::reply::json(&response)
    });

    // Hardware info endpoint
    let device_info_for_hw = device_info_for_api.clone();
    let hardware = warp::path!("api" / "hardware")
        .and(warp::get())
        .map(move || warp::reply::json(&*device_info_for_hw));

    // System-level runtime stats (process RSS, GPU utilization, etc).
    let runtime_samples_for_system_stats = runtime_samples_clone.clone();
    let runtime_pressure_state_for_system_stats = runtime_pressure_state.clone();
    let system_stats = warp::path!("api" / "system" / "stats")
        .and(warp::get())
        .map(move || {
            #[derive(Serialize)]
            struct SystemStatsResponse {
                pid: u32,
                process_memory_bytes: usize,
                total_system_memory_bytes: Option<usize>,
                gpu_utilization: f64,
                gpu_memory_bytes: Option<usize>,
                gpu_memory_total_bytes: Option<usize>,
                /// VRAM held by co-tenant processes sharing the GPU. `None` when
                /// the co-tenancy probe is disabled or unavailable.
                foreign_gpu_memory_bytes: Option<usize>,
                pressure_state: String,
                collected_at_ms: u64,
            }

            let samples = runtime_samples_for_system_stats.read().clone();
            let pressure_state = RuntimePressureState::from_u8(
                runtime_pressure_state_for_system_stats.load(Ordering::Relaxed),
            );
            warp::reply::json(&SystemStatsResponse {
                pid: std::process::id(),
                process_memory_bytes: samples.process_memory_bytes,
                total_system_memory_bytes: samples.total_system_memory_bytes,
                gpu_utilization: samples.gpu_utilization,
                gpu_memory_bytes: samples.gpu_memory_bytes,
                gpu_memory_total_bytes: samples.gpu_memory_total_bytes,
                foreign_gpu_memory_bytes: samples.foreign_gpu_memory_bytes,
                pressure_state: pressure_state.as_str().to_string(),
                collected_at_ms: samples.collected_at_ms,
            })
        });

    health
        .or(hardware)
        .or(system_stats)
        .map(reply_into_response)
        .boxed()
}
