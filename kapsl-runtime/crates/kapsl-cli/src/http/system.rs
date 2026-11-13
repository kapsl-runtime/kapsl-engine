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
    /// What a new allocation on this domain would actually be granted: the
    /// stricter of the authority's room and the CUDA device ledger's (which
    /// also charges idle pool backing).
    available_bytes: usize,
    /// The authority's own room, before the ledger gate. Equal to
    /// `available_bytes` except on CUDA devices with an idle pool backing.
    authority_available_bytes: usize,
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

/// `owner_kind` reported for a memory owner in `/api/system/stats`.
fn owner_kind_label(owner: MemoryOwner) -> &'static str {
    if owner.is_external_kv() {
        "external_kv_participant"
    } else if owner.is_runtime() {
        "runtime"
    } else {
        "model"
    }
}

fn summarize_memory_authority(
    models: &ModelManager,
    snapshot: MemorySnapshot,
    grantable: impl Fn(&MemoryDomain, usize) -> usize,
) -> MemoryAuthorityResponse {
    let mut by_model = BTreeMap::<(&'static str, u32), ModelMemoryAggregate>::new();
    for row in &snapshot.rows {
        let aggregate = by_model
            .entry((owner_kind_label(row.owner), row.owner.model_id))
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
        .map(|((owner_kind, model_id), aggregate)| {
            let name = if owner_kind == "external_kv_participant" {
                format!(
                    "External KV participant {}",
                    model_id & !MemoryOwner::EXTERNAL_KV_BASE
                )
            } else if owner_kind == "runtime" {
                format!(
                    "Runtime overhead (CUDA {})",
                    model_id & !MemoryOwner::RUNTIME_BASE
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
                owner_kind,
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
                domain_available_bytes = domain_available_bytes
                    .saturating_add(grantable(&domain.domain, domain.available_bytes));
            }
            MemoryDomainUsageResponse {
                domain: domain.domain.to_string(),
                budget_bytes: domain.budget_bytes,
                planned_bytes: domain.planned_bytes,
                reserved_bytes: domain.reserved_bytes,
                committed_bytes: domain.committed_bytes,
                observed_bytes: domain.observed_bytes,
                used_bytes,
                available_bytes: if is_host_domain(&domain.domain) {
                    domain.available_bytes
                } else {
                    grantable(&domain.domain, domain.available_bytes)
                },
                authority_available_bytes: domain.available_bytes,
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
    /// CUDA driver / runtime versions (fields are `null` when unknown).
    cuda: CudaVersions,
    /// Freshness of the GPU samples behind CUDA admission: `fresh`, `stale`
    /// (a safety margin is applied) or `failed` (no new CUDA growth admitted).
    gpu_sampling: &'static str,
    /// Seconds since the last GPU sample when `gpu_sampling` is not `fresh`.
    gpu_sampling_age_secs: Option<u64>,
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
            let memory = resources_for_system_stats.memory();
            let sampling = memory.cuda_sampling_state();
            let memory_authority = summarize_memory_authority(
                &models_for_system_stats,
                memory.snapshot(),
                |domain, available| memory.grantable_bytes(domain, available),
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
                cuda: cuda_versions().clone(),
                gpu_sampling: sampling.as_str(),
                gpu_sampling_age_secs: sampling.age().map(|age| age.as_secs()),
                memory_authority,
            })
        });

    health
        .or(hardware)
        .or(system_stats)
        .map(reply_into_response)
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_kinds_are_labelled_for_the_stats_response() {
        assert_eq!(owner_kind_label(MemoryOwner::new(3, 0)), "model");
        assert_eq!(owner_kind_label(MemoryOwner::runtime(0)), "runtime");
        assert_eq!(
            owner_kind_label(MemoryOwner::external_kv(2).unwrap()),
            "external_kv_participant"
        );
    }

    #[test]
    fn available_bytes_is_what_the_ledger_would_grant_not_the_authority_figure() {
        // The 2026-10-24 L4 numbers (MEM-12): authority room 18_267_513_642,
        // ledger room 17_410_267_546 (it also charges the idle pool backing).
        const AUTHORITY: usize = 18_267_513_642;
        const LEDGER: usize = 17_410_267_546;
        let cuda = MemoryDomain::Cuda { device_id: 0 };
        let snapshot = MemorySnapshot {
            rows: Vec::new(),
            domains: vec![MemoryDomainSnapshot {
                domain: cuda.clone(),
                budget_bytes: 22_000_000_000,
                planned_bytes: 0,
                reserved_bytes: 0,
                committed_bytes: 0,
                observed_bytes: 0,
                available_bytes: AUTHORITY,
            }],
            foreign_pressure_active: false,
        };
        let models = ModelManager::new(Arc::new(ModelRegistry::new()));
        let response = summarize_memory_authority(&models, snapshot, |domain, available| {
            assert_eq!(domain, &cuda);
            crate::runtime::memory::stricter_available(available, Some(LEDGER))
        });
        assert_eq!(response.domains[0].available_bytes, LEDGER);
        assert_eq!(response.domains[0].authority_available_bytes, AUTHORITY);
        assert_eq!(response.domain_available_bytes, LEDGER);
    }

    #[test]
    fn stricter_available_takes_the_smaller_gate() {
        use crate::runtime::memory::stricter_available;
        assert_eq!(stricter_available(10, Some(7)), 7);
        assert_eq!(stricter_available(5, Some(7)), 5);
        assert_eq!(stricter_available(10, None), 10);
    }
}
