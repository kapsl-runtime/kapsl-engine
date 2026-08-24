use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Serialize)]
struct ModelMemoryUsageResponse {
    model_id: u32,
    owner_kind: &'static str,
    name: String,
    replica_count: usize,
    planned_bytes: usize,
    reserved_bytes: usize,
    committed_bytes: usize,
    observed_bytes: usize,
    used_bytes: usize,
    percentage: f64,
}

#[derive(Debug, Serialize)]
struct MemoryDomainUsageResponse {
    domain: String,
    budget_bytes: usize,
    planned_bytes: usize,
    reserved_bytes: usize,
    committed_bytes: usize,
    observed_bytes: usize,
    used_bytes: usize,
    available_bytes: usize,
}

#[derive(Debug, Serialize)]
struct MemoryAuthorityResponse {
    /// Sum of live, workload-attributed rows. A row contributes the largest of
    /// reserved, committed, or observed bytes so accounting states are not
    /// double-counted while a backend transitions between them.
    model_bytes: usize,
    domain_used_bytes: usize,
    domain_budget_bytes: usize,
    domain_available_bytes: usize,
    foreign_pressure_active: bool,
    models: Vec<ModelMemoryUsageResponse>,
    domains: Vec<MemoryDomainUsageResponse>,
}

#[derive(Default)]
struct ModelMemoryAggregate {
    replicas: BTreeSet<u32>,
    planned_bytes: usize,
    reserved_bytes: usize,
    committed_bytes: usize,
    observed_bytes: usize,
    used_bytes: usize,
}

fn is_host_domain(domain: &MemoryDomain) -> bool {
    matches!(
        domain,
        MemoryDomain::Host | MemoryDomain::HostPinned { .. } | MemoryDomain::HostMapped { .. }
    )
}

fn summarize_memory_authority(
    models: &ModelManager,
    snapshot: MemorySnapshot,
) -> MemoryAuthorityResponse {
    let mut by_model = BTreeMap::<(bool, u32), ModelMemoryAggregate>::new();
    for row in &snapshot.rows {
        let aggregate = by_model
            .entry((row.owner.is_external_kv(), row.owner.model_id))
            .or_default();
        aggregate.replicas.insert(row.owner.replica_id);
        aggregate.planned_bytes = aggregate.planned_bytes.saturating_add(row.planned_bytes);
        aggregate.reserved_bytes = aggregate.reserved_bytes.saturating_add(row.reserved_bytes);
        aggregate.committed_bytes = aggregate
            .committed_bytes
            .saturating_add(row.committed_bytes);
        aggregate.observed_bytes = aggregate.observed_bytes.saturating_add(row.observed_bytes);
        aggregate.used_bytes = aggregate.used_bytes.saturating_add(row.used_bytes());
    }

    let model_bytes = by_model.values().fold(0usize, |total, model| {
        total.saturating_add(model.used_bytes)
    });
    let mut model_usage = by_model
        .into_iter()
        .map(|((external_kv, model_id), aggregate)| {
            let name = if external_kv {
                format!(
                    "External KV participant {}",
                    model_id & !MemoryOwner::EXTERNAL_KV_BASE
                )
            } else {
                models
                    .registry()
                    .get(model_id)
                    .map(|model| model.name)
                    .unwrap_or_else(|| format!("Model {model_id}"))
            };
            let percentage = if model_bytes == 0 {
                0.0
            } else {
                aggregate.used_bytes as f64 * 100.0 / model_bytes as f64
            };
            ModelMemoryUsageResponse {
                model_id,
                owner_kind: if external_kv {
                    "external_kv_participant"
                } else {
                    "model"
                },
                name,
                replica_count: aggregate.replicas.len(),
                planned_bytes: aggregate.planned_bytes,
                reserved_bytes: aggregate.reserved_bytes,
                committed_bytes: aggregate.committed_bytes,
                observed_bytes: aggregate.observed_bytes,
                used_bytes: aggregate.used_bytes,
                percentage,
            }
        })
        .collect::<Vec<_>>();
    model_usage.sort_by(|left, right| {
        right
            .used_bytes
            .cmp(&left.used_bytes)
            .then_with(|| left.model_id.cmp(&right.model_id))
    });

    let mut domain_used_bytes = 0usize;
    let mut domain_budget_bytes = 0usize;
    let mut domain_available_bytes = 0usize;
    // Host, pinned-host, and mapped-host rows share one physical manager and
    // budget. Collapse that family for the headline totals exactly as the
    // authority's pressure calculation does, while retaining each raw domain
    // below for diagnostics.
    let host_budget = snapshot
        .domain(&MemoryDomain::Host)
        .map(|domain| domain.budget_bytes)
        .or_else(|| {
            snapshot
                .domains
                .iter()
                .find(|domain| is_host_domain(&domain.domain))
                .map(|domain| domain.budget_bytes)
        });
    let (host_reserved, host_committed, host_attributed_observed) = snapshot
        .rows
        .iter()
        .filter(|row| is_host_domain(&row.domain))
        .fold((0usize, 0usize, 0usize), |totals, row| {
            (
                totals.0.saturating_add(row.reserved_bytes),
                totals.1.saturating_add(row.committed_bytes),
                totals.2.saturating_add(row.observed_bytes),
            )
        });
    let host_sampled_observed = snapshot
        .domain(&MemoryDomain::Host)
        .map(|domain| domain.observed_bytes)
        .unwrap_or(0);
    let host_used = host_reserved
        .max(host_committed)
        .max(host_attributed_observed.max(host_sampled_observed));
    let mut host_accounted = false;
    let domains = snapshot
        .domains
        .into_iter()
        .map(|domain| {
            let used_bytes = domain.used_bytes();
            if is_host_domain(&domain.domain) {
                if !host_accounted {
                    let budget_bytes = host_budget.unwrap_or(domain.budget_bytes);
                    domain_used_bytes = domain_used_bytes.saturating_add(host_used);
                    domain_budget_bytes = domain_budget_bytes.saturating_add(budget_bytes);
                    domain_available_bytes = domain_available_bytes
                        .saturating_add(budget_bytes.saturating_sub(host_used));
                    host_accounted = true;
                }
            } else {
                domain_used_bytes = domain_used_bytes.saturating_add(used_bytes);
                domain_budget_bytes = domain_budget_bytes.saturating_add(domain.budget_bytes);
                domain_available_bytes =
                    domain_available_bytes.saturating_add(domain.available_bytes);
            }
            MemoryDomainUsageResponse {
                domain: domain.domain.to_string(),
                budget_bytes: domain.budget_bytes,
                planned_bytes: domain.planned_bytes,
                reserved_bytes: domain.reserved_bytes,
                committed_bytes: domain.committed_bytes,
                observed_bytes: domain.observed_bytes,
                used_bytes,
                available_bytes: domain.available_bytes,
            }
        })
        .collect();

    MemoryAuthorityResponse {
        model_bytes,
        domain_used_bytes,
        domain_budget_bytes,
        domain_available_bytes,
        foreign_pressure_active: snapshot.foreign_pressure_active,
        models: model_usage,
        domains,
    }
}

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
    memory_authority: MemoryAuthorityResponse,
}

pub(crate) fn build_system_routes(
    models: Arc<ModelManager>,
    device_info_for_api: Arc<DeviceInfo>,
    runtime_samples_clone: Arc<RwLock<RuntimeSamples>>,
    runtime_pressure_state: Arc<AtomicU8>,
    resources: Arc<RuntimeResources>,
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
    let models_for_system_stats = models.clone();
    let resources_for_system_stats = resources.clone();
    let system_stats = warp::path!("api" / "system" / "stats")
        .and(warp::get())
        .map(move || {
            let samples = runtime_samples_for_system_stats.read().clone();
            let pressure_state = RuntimePressureState::from_u8(
                runtime_pressure_state_for_system_stats.load(Ordering::Relaxed),
            );
            let memory_authority = summarize_memory_authority(
                &models_for_system_stats,
                resources_for_system_stats.memory().snapshot(),
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
                memory_authority,
            })
        });

    health
        .or(hardware)
        .or(system_stats)
        .map(reply_into_response)
        .boxed()
}
