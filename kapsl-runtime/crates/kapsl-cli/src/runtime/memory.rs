//! Backend-neutral memory admission and lifetime ownership.
//!
//! A [`MemoryPlan`] describes memory before a backend allocates it. The
//! [`MemoryAuthority`] routes each claim to the manager for its domain, and a
//! committed [`MemoryLease`] retains those reservations until the owning
//! model/replica is unloaded. Provider domains are accounting adapters: they
//! participate in ownership and lifetime now, without pretending that the
//! runtime physically allocates their memory.

use super::host_memory::{HostMemoryLease, HostMemoryLoadAdmission, HostMemoryManager};
use super::tuning::{
    device_vram_cap_bytes, effective_ceiling_bytes, parse_cuda_memory_limit, smooth_ceiling_bytes,
};
use crate::app::constants::PROVIDER_MEMORY_LIMITS_ENV;
use kapsl_core::EngineKind;
#[cfg(any(feature = "gpu-device-pool", test))]
use kapsl_engine_api::{ExternalDeviceMemory, ExternalDeviceMemoryReport};
use kapsl_engine_api::{
    MemoryAllocationClass as BackendMemoryAllocationClass,
    MemoryAllocationSource as BackendMemoryAllocationSource, MemoryDomain as BackendMemoryDomain,
    MemoryReport as BackendMemoryReport,
};
use kapsl_hal::device::DeviceInfo;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(feature = "gpu-device-pool")]
use super::device_memory::{
    DeviceMemoryAdmission, DeviceMemoryBootstrapPlan, DeviceMemoryLease, DeviceMemoryManager,
    DeviceMemorySwapAdmission, DeviceMemoryTransientLease,
};

/// An independently-budgeted memory domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum MemoryDomain {
    Host,
    HostPinned {
        provider: String,
        device_id: Option<usize>,
    },
    HostMapped {
        provider: String,
        device_id: Option<usize>,
    },
    Cuda {
        device_id: usize,
    },
    /// A provider that is not managed by the host or CUDA allocators.
    Provider {
        provider: String,
        device_id: Option<usize>,
    },
}

impl fmt::Display for MemoryDomain {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host => formatter.write_str("host"),
            Self::HostPinned {
                provider,
                device_id: Some(device_id),
            } => write!(formatter, "host-pinned:{provider}:{device_id}"),
            Self::HostPinned {
                provider,
                device_id: None,
            } => write!(formatter, "host-pinned:{provider}"),
            Self::HostMapped {
                provider,
                device_id: Some(device_id),
            } => write!(formatter, "host-mapped:{provider}:{device_id}"),
            Self::HostMapped {
                provider,
                device_id: None,
            } => write!(formatter, "host-mapped:{provider}"),
            Self::Cuda { device_id } => write!(formatter, "cuda:{device_id}"),
            Self::Provider {
                provider,
                device_id: Some(device_id),
            } => write!(formatter, "{provider}:{device_id}"),
            Self::Provider {
                provider,
                device_id: None,
            } => formatter.write_str(provider),
        }
    }
}

impl MemoryDomain {
    fn from_backend(domain: &BackendMemoryDomain) -> Self {
        match domain {
            BackendMemoryDomain::Host => Self::Host,
            BackendMemoryDomain::HostPinned {
                provider,
                device_id,
            } => Self::HostPinned {
                provider: provider.clone(),
                device_id: *device_id,
            },
            BackendMemoryDomain::HostMapped {
                provider,
                device_id,
            } => Self::HostMapped {
                provider: provider.clone(),
                device_id: *device_id,
            },
            BackendMemoryDomain::Cuda { device_id } => Self::Cuda {
                device_id: *device_id,
            },
            BackendMemoryDomain::Provider {
                provider,
                device_id,
            } => {
                let provider = normalize_provider_name(provider);
                Self::Provider {
                    provider,
                    device_id: *device_id,
                }
            }
        }
    }

    pub(crate) fn for_provider(provider: &str, device_id: usize) -> Self {
        match normalize_provider_name(provider).as_str() {
            "cpu" => Self::Host,
            "cuda" | "tensorrt" => Self::Cuda { device_id },
            "metal" => Self::Provider {
                provider: "metal".to_string(),
                device_id: Some(device_id),
            },
            provider => Self::Provider {
                provider: provider.to_string(),
                device_id: Some(device_id),
            },
        }
    }
}

fn normalize_provider_name(provider: &str) -> String {
    match provider.trim().to_ascii_lowercase().as_str() {
        "coreml" => "metal".to_string(),
        "dml" => "directml".to_string(),
        provider => provider.to_string(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProviderLimitKey {
    provider: String,
    device_id: Option<usize>,
}

impl ProviderLimitKey {
    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Provider {
            provider: self.provider.clone(),
            device_id: self.device_id,
        }
    }
}

/// Operator-declared hard ceilings for accounting-only provider adapters.
/// Exact device entries take precedence over provider-wide fallbacks.
#[derive(Debug, Clone, Default)]
struct ProviderMemoryLimits {
    bytes: HashMap<ProviderLimitKey, usize>,
}

impl ProviderMemoryLimits {
    fn from_env() -> Result<Self, String> {
        let Some(raw) = std::env::var(PROVIDER_MEMORY_LIMITS_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            return Ok(Self::default());
        };
        Self::parse(&raw)
    }

    fn parse(raw: &str) -> Result<Self, String> {
        let mut bytes = HashMap::new();
        for raw_entry in raw.split(',') {
            let entry = raw_entry.trim();
            if entry.is_empty() {
                continue;
            }
            let (raw_domain, raw_limit) = entry.split_once('=').ok_or_else(|| {
                format!(
                    "invalid {PROVIDER_MEMORY_LIMITS_ENV} entry `{entry}`; expected provider[:device]=size"
                )
            })?;
            let raw_domain = raw_domain.trim();
            let (provider, device_id) = match raw_domain.rsplit_once(':') {
                Some((provider, device)) => {
                    let device_id = device.trim().parse::<usize>().map_err(|_| {
                        format!(
                            "invalid provider device `{raw_domain}` in {PROVIDER_MEMORY_LIMITS_ENV}"
                        )
                    })?;
                    (provider, Some(device_id))
                }
                None => (raw_domain, None),
            };
            let provider = normalize_provider_name(provider);
            if provider.is_empty() {
                return Err(format!(
                    "empty provider name in {PROVIDER_MEMORY_LIMITS_ENV} entry `{entry}`"
                ));
            }
            if matches!(provider.as_str(), "cpu" | "cuda" | "tensorrt") {
                return Err(format!(
                    "{PROVIDER_MEMORY_LIMITS_ENV} does not configure `{provider}`; use the dedicated CPU/CUDA memory limit"
                ));
            }
            let limit = parse_cuda_memory_limit(raw_limit).ok_or_else(|| {
                format!(
                    "invalid provider memory size `{}` in {PROVIDER_MEMORY_LIMITS_ENV}; use a positive byte count or k/m/g suffix",
                    raw_limit.trim()
                )
            })?;
            let key = ProviderLimitKey {
                provider,
                device_id,
            };
            if bytes.insert(key.clone(), limit).is_some() {
                return Err(format!(
                    "duplicate provider memory limit for {} in {PROVIDER_MEMORY_LIMITS_ENV}",
                    key.domain()
                ));
            }
        }
        Ok(Self { bytes })
    }

    fn limit_for_domain(&self, domain: &MemoryDomain) -> Option<usize> {
        let MemoryDomain::Provider {
            provider,
            device_id,
        } = domain
        else {
            return None;
        };
        self.bytes
            .get(&ProviderLimitKey {
                provider: normalize_provider_name(provider),
                device_id: *device_id,
            })
            .copied()
            .or_else(|| {
                self.bytes
                    .get(&ProviderLimitKey {
                        provider: normalize_provider_name(provider),
                        device_id: None,
                    })
                    .copied()
            })
    }
}

/// Why an allocation exists. Budgets and metrics can aggregate these without
/// knowing which backend produced them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MemoryAllocationClass {
    PersistentWeights,
    ModelSession,
    KvCache,
    TransientWorkspace,
    BlockTable,
    RequestTransient,
    ExternallyOwned,
}

impl MemoryAllocationClass {
    fn from_backend(class: BackendMemoryAllocationClass) -> Self {
        match class {
            BackendMemoryAllocationClass::PersistentWeights => Self::PersistentWeights,
            BackendMemoryAllocationClass::ModelSession => Self::ModelSession,
            BackendMemoryAllocationClass::KvCache => Self::KvCache,
            BackendMemoryAllocationClass::TransientWorkspace => Self::TransientWorkspace,
            BackendMemoryAllocationClass::BlockTable => Self::BlockTable,
            BackendMemoryAllocationClass::RequestTransient => Self::RequestTransient,
            BackendMemoryAllocationClass::ExternallyOwned => Self::ExternallyOwned,
        }
    }
}

impl fmt::Display for MemoryAllocationClass {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::PersistentWeights => "persistent-weights",
            Self::ModelSession => "model-session",
            Self::KvCache => "kv-cache",
            Self::TransientWorkspace => "transient-workspace",
            Self::BlockTable => "block-table",
            Self::RequestTransient => "request-transient",
            Self::ExternallyOwned => "externally-owned",
        })
    }
}

/// Stable workload identity carried through every admission adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MemoryOwnerKind {
    Model,
    ExternalKv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MemoryOwner {
    kind: MemoryOwnerKind,
    pub(crate) model_id: u32,
    pub(crate) replica_id: u32,
}

impl MemoryOwner {
    pub(crate) const EXTERNAL_KV_BASE: u32 = 1 << 31;

    pub(crate) const fn new(model_id: u32, replica_id: u32) -> Self {
        Self {
            kind: MemoryOwnerKind::Model,
            model_id,
            replica_id,
        }
    }

    pub(crate) fn external_kv(participant_slot: u32) -> Option<Self> {
        (participant_slot < Self::EXTERNAL_KV_BASE).then_some(Self {
            kind: MemoryOwnerKind::ExternalKv,
            model_id: Self::EXTERNAL_KV_BASE | participant_slot,
            replica_id: 0,
        })
    }

    pub(crate) const fn is_external_kv(self) -> bool {
        matches!(self.kind, MemoryOwnerKind::ExternalKv)
    }

    pub(crate) const fn external_kv_slot(self) -> Option<u32> {
        if self.is_external_kv() {
            Some(self.model_id & !Self::EXTERNAL_KV_BASE)
        } else {
            None
        }
    }
}

impl fmt::Display for MemoryOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(slot) = self.external_kv_slot() {
            write!(formatter, "external KV participant {slot}")
        } else {
            write!(
                formatter,
                "model {} replica {}",
                self.model_id, self.replica_id
            )
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MemoryRowKey {
    domain: MemoryDomain,
    owner: MemoryOwner,
    class: MemoryAllocationClass,
}

impl MemoryRowKey {
    fn from_claim(claim: &MemoryClaim) -> Self {
        Self {
            domain: claim.domain.clone(),
            owner: claim.owner,
            class: claim.class,
        }
    }
}

/// Unified per-owner view used by admission, pressure policy, and metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MemorySnapshotRow {
    pub(crate) domain: MemoryDomain,
    pub(crate) owner: MemoryOwner,
    pub(crate) class: MemoryAllocationClass,
    pub(crate) planned_bytes: usize,
    pub(crate) reserved_bytes: usize,
    pub(crate) committed_bytes: usize,
    pub(crate) observed_bytes: usize,
}

impl MemorySnapshotRow {
    /// Live footprint without double-counting accounting states during a
    /// reservation-to-observation transition.
    pub(crate) fn used_bytes(&self) -> usize {
        self.reserved_bytes
            .max(self.committed_bytes)
            .max(self.observed_bytes)
    }
}

/// One independently-budgeted domain in a [`MemorySnapshot`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MemoryDomainSnapshot {
    pub(crate) domain: MemoryDomain,
    pub(crate) budget_bytes: usize,
    pub(crate) planned_bytes: usize,
    pub(crate) reserved_bytes: usize,
    pub(crate) committed_bytes: usize,
    pub(crate) observed_bytes: usize,
    pub(crate) available_bytes: usize,
}

impl MemoryDomainSnapshot {
    pub(crate) fn used_bytes(&self) -> usize {
        self.reserved_bytes
            .max(self.committed_bytes)
            .max(self.observed_bytes)
    }
}

/// Atomic point-in-time accounting across every memory adapter.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MemorySnapshot {
    pub(crate) rows: Vec<MemorySnapshotRow>,
    pub(crate) domains: Vec<MemoryDomainSnapshot>,
    pub(crate) foreign_pressure_active: bool,
}

impl MemorySnapshot {
    pub(crate) fn domain(&self, domain: &MemoryDomain) -> Option<&MemoryDomainSnapshot> {
        self.domains.iter().find(|item| &item.domain == domain)
    }

    pub(crate) fn max_host_ratio(&self) -> Option<f64> {
        let is_host = |domain: &MemoryDomain| {
            matches!(
                domain,
                MemoryDomain::Host
                    | MemoryDomain::HostPinned { .. }
                    | MemoryDomain::HostMapped { .. }
            )
        };
        let budget = self
            .domain(&MemoryDomain::Host)
            .map(|item| item.budget_bytes)
            .or_else(|| {
                self.domains
                    .iter()
                    .find(|item| is_host(&item.domain))
                    .map(|item| item.budget_bytes)
            })?;
        if budget == 0 {
            return None;
        }
        let (reserved, committed, attributed_observed) = self
            .rows
            .iter()
            .filter(|row| is_host(&row.domain))
            .fold((0usize, 0usize, 0usize), |acc, row| {
                (
                    acc.0.saturating_add(row.reserved_bytes),
                    acc.1.saturating_add(row.committed_bytes),
                    acc.2.saturating_add(row.observed_bytes),
                )
            });
        let sampled_observed = self
            .domain(&MemoryDomain::Host)
            .map(|item| item.observed_bytes)
            .unwrap_or(0);
        Some(
            reserved
                .max(committed)
                .max(attributed_observed.max(sampled_observed)) as f64
                / budget as f64,
        )
    }

    /// Highest pressure ratio across accelerator/provider domains. CUDA and
    /// accounting-only providers use identical pressure semantics once a
    /// provider budget is known.
    pub(crate) fn max_device_ratio(&self) -> Option<f64> {
        self.domains
            .iter()
            .filter(|item| {
                matches!(
                    item.domain,
                    MemoryDomain::Cuda { .. } | MemoryDomain::Provider { .. }
                )
            })
            .filter(|item| item.budget_bytes > 0)
            .map(|item| {
                item.budget_bytes.saturating_sub(item.available_bytes) as f64
                    / item.budget_bytes as f64
            })
            .reduce(f64::max)
    }

    pub(crate) fn cap_replica_target(
        &self,
        model_id: u32,
        current_replicas: u32,
        proposed_target: u32,
    ) -> u32 {
        if proposed_target <= current_replicas || current_replicas == 0 {
            return proposed_target;
        }
        let mut additions = proposed_target - current_replicas;
        let mut saw_memory = false;
        for domain in &self.domains {
            let model_bytes = self
                .rows
                .iter()
                .filter(|row| row.owner.model_id == model_id && row.domain == domain.domain)
                .map(|row| row.committed_bytes.max(row.observed_bytes))
                .fold(0usize, usize::saturating_add);
            if model_bytes == 0 {
                continue;
            }
            saw_memory = true;
            let per_replica = model_bytes.saturating_add(current_replicas as usize - 1)
                / current_replicas as usize;
            if per_replica > 0 {
                additions = additions.min((domain.available_bytes / per_replica) as u32);
            }
        }
        if saw_memory {
            current_replicas.saturating_add(additions)
        } else {
            proposed_target
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct MemoryAccountingBytes {
    planned: usize,
    reserved: usize,
    committed: usize,
    observed: usize,
}

#[derive(Default)]
struct AuthorityAccounting {
    leases: HashMap<u64, HashMap<MemoryRowKey, MemoryAccountingBytes>>,
    observed_domains: HashMap<MemoryDomain, usize>,
}

#[derive(Debug, Clone)]
struct MemoryDomainPolicy {
    domain: MemoryDomain,
    budget_bytes: usize,
}

/// Whether bytes are runtime-managed or only accounted by an adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MemoryClaimSource {
    Runtime { allocation_id: Option<String> },
    External { allocation_id: String },
}

/// One independently-owned claim in a memory plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MemoryClaim {
    pub(crate) domain: MemoryDomain,
    pub(crate) owner: MemoryOwner,
    pub(crate) class: MemoryAllocationClass,
    pub(crate) bytes: usize,
    pub(crate) source: MemoryClaimSource,
}

impl MemoryClaim {
    pub(crate) fn runtime(
        domain: MemoryDomain,
        owner: MemoryOwner,
        class: MemoryAllocationClass,
        bytes: usize,
    ) -> Self {
        Self {
            domain,
            owner,
            class,
            bytes,
            source: MemoryClaimSource::Runtime {
                allocation_id: None,
            },
        }
    }

    pub(crate) fn runtime_allocation(
        domain: MemoryDomain,
        owner: MemoryOwner,
        class: MemoryAllocationClass,
        allocation_id: impl Into<String>,
        bytes: usize,
    ) -> Self {
        Self {
            domain,
            owner,
            class,
            bytes,
            source: MemoryClaimSource::Runtime {
                allocation_id: Some(allocation_id.into()),
            },
        }
    }

    pub(crate) fn external(
        domain: MemoryDomain,
        owner: MemoryOwner,
        class: MemoryAllocationClass,
        allocation_id: impl Into<String>,
        bytes: usize,
    ) -> Self {
        Self {
            domain,
            owner,
            class,
            bytes,
            source: MemoryClaimSource::External {
                allocation_id: allocation_id.into(),
            },
        }
    }
}

/// Backend-neutral declaration of a workload's host and device memory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MemoryPlan {
    claims: Vec<MemoryClaim>,
}

impl MemoryPlan {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn push(&mut self, claim: MemoryClaim) -> &mut Self {
        self.claims.push(claim);
        self
    }

    pub(crate) fn claims(&self) -> &[MemoryClaim] {
        &self.claims
    }

    pub(crate) fn extend(&mut self, other: Self) -> &mut Self {
        self.claims.extend(other.claims);
        self
    }

    pub(crate) fn contains_class(&self, class: MemoryAllocationClass) -> bool {
        self.claims.iter().any(|claim| claim.class == class)
    }

    #[cfg(test)]
    pub(crate) fn request_transient(owner: MemoryOwner, bytes: usize) -> Self {
        Self {
            claims: vec![MemoryClaim::runtime(
                MemoryDomain::Host,
                owner,
                MemoryAllocationClass::RequestTransient,
                bytes,
            )],
        }
    }

    pub(crate) fn from_backend_report(owner: MemoryOwner, report: &BackendMemoryReport) -> Self {
        let claims = report
            .allocations
            .iter()
            .map(|allocation| {
                let domain = MemoryDomain::from_backend(&allocation.domain);
                let class = MemoryAllocationClass::from_backend(allocation.class);
                match allocation.source {
                    BackendMemoryAllocationSource::RuntimeManaged => {
                        MemoryClaim::runtime_allocation(
                            domain,
                            owner,
                            class,
                            allocation.allocation_id.clone(),
                            allocation.bytes,
                        )
                    }
                    BackendMemoryAllocationSource::BackendManaged => MemoryClaim::external(
                        domain,
                        owner,
                        class,
                        allocation.allocation_id.clone(),
                        allocation.bytes,
                    ),
                }
            })
            .collect();
        Self { claims }
    }

    /// Request-time authority. Runtime-managed CUDA rows are enforced by the
    /// admitted workload's pool quota at allocation time; backend-managed CUDA
    /// rows receive a transient device-budget reservation. Host, pinned,
    /// mapped, and provider rows are retained for the request lifetime.
    pub(crate) fn request_from_backend_report(
        owner: MemoryOwner,
        report: &BackendMemoryReport,
    ) -> Self {
        let mut plan = Self::from_backend_report(owner, report);
        plan.claims.retain(|claim| {
            matches!(
                claim.domain,
                MemoryDomain::Host
                    | MemoryDomain::HostPinned { .. }
                    | MemoryDomain::HostMapped { .. }
                    | MemoryDomain::Provider { .. }
            ) || (cfg!(feature = "gpu-device-pool")
                && matches!(claim.domain, MemoryDomain::Cuda { .. }))
        });
        plan
    }

    #[cfg(test)]
    pub(crate) fn external_cuda_report(
        owner: MemoryOwner,
        report: &ExternalDeviceMemoryReport,
    ) -> Self {
        let mut plan = Self::new();
        for allocation in &report.allocations {
            plan.push(MemoryClaim::external(
                MemoryDomain::Cuda {
                    device_id: allocation.device_id,
                },
                owner,
                classify_external_allocation(&allocation.allocation_id),
                allocation.allocation_id.clone(),
                allocation.bytes,
            ));
        }
        plan
    }

    #[cfg(feature = "gpu-device-pool")]
    fn external_report_for_cuda(&self, device_id: usize) -> ExternalDeviceMemoryReport {
        ExternalDeviceMemoryReport {
            allocations: self
                .claims
                .iter()
                .filter_map(|claim| match (&claim.domain, &claim.source) {
                    (
                        MemoryDomain::Cuda {
                            device_id: candidate,
                        },
                        MemoryClaimSource::External { allocation_id },
                    ) if *candidate == device_id => Some(ExternalDeviceMemory {
                        allocation_id: allocation_id.clone(),
                        device_id,
                        bytes: claim.bytes,
                    }),
                    _ => None,
                })
                .collect(),
        }
    }

    #[cfg(any(feature = "gpu-device-pool", test))]
    fn cuda_device_ids(&self) -> Vec<usize> {
        let mut device_ids: Vec<_> = self
            .claims
            .iter()
            .filter_map(|claim| match claim.domain {
                MemoryDomain::Cuda { device_id } => Some(device_id),
                _ => None,
            })
            .collect();
        device_ids.sort_unstable();
        device_ids.dedup();
        device_ids
    }
}

/// Typed boundary for a model load rejected by `MemoryAuthority`.
///
/// Lifecycle policy may downcast this error to reclaim an idle, strictly
/// lower-priority model and retry. Backend/model failures remain ordinary
/// errors and can never trigger an unrelated unload.
#[derive(Debug)]
pub(crate) struct MemoryAdmissionFailure {
    owner: MemoryOwner,
    priority_weight: u32,
    domains: Vec<MemoryDomain>,
    message: String,
}

impl MemoryAdmissionFailure {
    pub(crate) fn new(
        owner: MemoryOwner,
        priority_weight: u32,
        plan: &MemoryPlan,
        message: impl Into<String>,
    ) -> Self {
        let mut domains = plan
            .claims()
            .iter()
            .filter(|claim| claim.bytes > 0)
            .map(|claim| claim.domain.clone())
            .collect::<Vec<_>>();
        domains.sort_by_key(ToString::to_string);
        domains.dedup();
        Self {
            owner,
            priority_weight: priority_weight.max(1),
            domains,
            message: message.into(),
        }
    }

    pub(crate) fn owner(&self) -> MemoryOwner {
        self.owner
    }

    pub(crate) fn priority_weight(&self) -> u32 {
        self.priority_weight
    }

    pub(crate) fn domains(&self) -> &[MemoryDomain] {
        &self.domains
    }
}

impl fmt::Display for MemoryAdmissionFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "memory admission failed for {} (priority weight {}): {}",
            self.owner, self.priority_weight, self.message
        )
    }
}

impl std::error::Error for MemoryAdmissionFailure {}

#[cfg(any(feature = "gpu-device-pool", test))]
pub(crate) fn classify_external_allocation(allocation_id: &str) -> MemoryAllocationClass {
    let lowered = allocation_id.to_ascii_lowercase();
    if lowered.contains("block-table") || lowered.contains("block_table") {
        MemoryAllocationClass::BlockTable
    } else if lowered.contains("scratch") || lowered.contains("workspace") {
        MemoryAllocationClass::TransientWorkspace
    } else if lowered.contains("kv") || lowered.contains("past") || lowered.contains("present") {
        MemoryAllocationClass::KvCache
    } else if lowered.contains("session") {
        MemoryAllocationClass::ModelSession
    } else if lowered.contains("weight")
        || lowered.contains("model")
        || lowered.contains("gguf")
        || lowered.contains("native")
    {
        MemoryAllocationClass::PersistentWeights
    } else {
        MemoryAllocationClass::ExternallyOwned
    }
}

#[cfg(feature = "gpu-device-pool")]
fn external_cuda_report_from_backend(report: &BackendMemoryReport) -> ExternalDeviceMemoryReport {
    ExternalDeviceMemoryReport {
        allocations: report
            .allocations
            .iter()
            .filter_map(|allocation| {
                let BackendMemoryDomain::Cuda { device_id } = &allocation.domain else {
                    return None;
                };
                if allocation.source != BackendMemoryAllocationSource::BackendManaged {
                    return None;
                }
                Some(ExternalDeviceMemory {
                    allocation_id: allocation.allocation_id.clone(),
                    device_id: *device_id,
                    bytes: allocation.bytes,
                })
            })
            .collect(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProviderAllocationKey {
    domain: MemoryDomain,
    allocation_id: String,
}

#[derive(Debug)]
struct ProviderAllocation {
    class: MemoryAllocationClass,
    bytes: usize,
    owners: HashMap<MemoryOwner, ProviderOwnership>,
}

#[derive(Debug, Clone, Copy)]
struct ProviderOwnership {
    refs: usize,
    bytes: usize,
}

struct ProviderMemoryLedger {
    allocations: Mutex<HashMap<ProviderAllocationKey, ProviderAllocation>>,
    budgets: HashMap<MemoryDomain, usize>,
    next_transient_allocation: std::sync::atomic::AtomicU64,
}

impl Default for ProviderMemoryLedger {
    fn default() -> Self {
        Self::with_budgets(HashMap::new())
    }
}

impl ProviderMemoryLedger {
    fn with_budgets(budgets: HashMap<MemoryDomain, usize>) -> Self {
        Self {
            allocations: Mutex::new(HashMap::new()),
            budgets,
            next_transient_allocation: AtomicU64::new(0),
        }
    }

    fn budget_for_domain(&self, domain: &MemoryDomain) -> Option<usize> {
        self.budgets.get(domain).copied().or_else(|| {
            let MemoryDomain::Provider { provider, .. } = domain else {
                return None;
            };
            self.budgets
                .get(&MemoryDomain::Provider {
                    provider: provider.clone(),
                    device_id: None,
                })
                .copied()
        })
    }

    fn domain_bytes(
        allocations: &HashMap<ProviderAllocationKey, ProviderAllocation>,
        domain: &MemoryDomain,
    ) -> usize {
        allocations
            .iter()
            .filter(|(key, _)| &key.domain == domain)
            .map(|(_, allocation)| allocation.bytes)
            .fold(0usize, usize::saturating_add)
    }

    fn enforce_budget(
        &self,
        allocations: &HashMap<ProviderAllocationKey, ProviderAllocation>,
        domain: &MemoryDomain,
        current_allocation_bytes: usize,
        target_allocation_bytes: usize,
    ) -> Result<(), String> {
        let Some(budget) = self.budget_for_domain(domain) else {
            return Ok(());
        };
        let current = Self::domain_bytes(allocations, domain);
        let projected = current
            .saturating_sub(current_allocation_bytes)
            .saturating_add(target_allocation_bytes);
        if projected > budget {
            return Err(format!(
                "provider memory admission rejected in {domain}: current={current} allocation_current={current_allocation_bytes} allocation_target={target_allocation_bytes} projected={projected} budget={budget} bytes"
            ));
        }
        Ok(())
    }

    fn reserve(
        self: &Arc<Self>,
        claim: &MemoryClaim,
    ) -> Result<Option<ProviderMemoryLease>, String> {
        let MemoryDomain::Provider { .. } = claim.domain else {
            return Ok(None);
        };
        if claim.bytes == 0 {
            return Ok(None);
        }
        let mut allocation_id = match &claim.source {
            MemoryClaimSource::External { allocation_id } => allocation_id.clone(),
            MemoryClaimSource::Runtime { allocation_id } => {
                allocation_id.clone().unwrap_or_else(|| {
                    format!(
                        "runtime:{}:{}:{}",
                        claim.owner.model_id, claim.owner.replica_id, claim.class
                    )
                })
            }
        };
        if claim.class == MemoryAllocationClass::RequestTransient {
            use std::sync::atomic::Ordering;
            allocation_id = format!(
                "{}:request:{}",
                allocation_id,
                self.next_transient_allocation
                    .fetch_add(1, Ordering::Relaxed)
            );
        }
        let key = ProviderAllocationKey {
            domain: claim.domain.clone(),
            allocation_id,
        };
        let mut allocations = self.allocations.lock();
        if let Some(existing) = allocations.get(&key) {
            if existing.class != claim.class {
                return Err(format!(
                    "provider allocation `{}` in {} is already classified as {}, not {} for {}",
                    key.allocation_id, key.domain, existing.class, claim.class, claim.owner
                ));
            }
            let current_owner_bytes = existing
                .owners
                .get(&claim.owner)
                .map(|ownership| ownership.bytes)
                .unwrap_or(0);
            let target_owner_bytes = current_owner_bytes.saturating_add(claim.bytes);
            let other_owner_bytes = existing
                .owners
                .iter()
                .filter(|(owner, _)| **owner != claim.owner)
                .map(|(_, ownership)| ownership.bytes)
                .max()
                .unwrap_or(0);
            let target_allocation_bytes = target_owner_bytes.max(other_owner_bytes);
            self.enforce_budget(
                &allocations,
                &key.domain,
                existing.bytes,
                target_allocation_bytes,
            )?;
            let existing = allocations
                .get_mut(&key)
                .expect("provider allocation checked above");
            let ownership = existing
                .owners
                .entry(claim.owner)
                .or_insert(ProviderOwnership { refs: 0, bytes: 0 });
            ownership.refs = ownership.refs.saturating_add(1);
            ownership.bytes = ownership.bytes.saturating_add(claim.bytes);
            existing.bytes = existing
                .owners
                .values()
                .map(|owner| owner.bytes)
                .max()
                .unwrap_or(0);
        } else {
            self.enforce_budget(&allocations, &key.domain, 0, claim.bytes)?;
            let mut owners = HashMap::new();
            owners.insert(
                claim.owner,
                ProviderOwnership {
                    refs: 1,
                    bytes: claim.bytes,
                },
            );
            allocations.insert(
                key.clone(),
                ProviderAllocation {
                    class: claim.class,
                    bytes: claim.bytes,
                    owners,
                },
            );
        }
        Ok(Some(ProviderMemoryLease {
            ledger: Arc::clone(self),
            key,
            owner: claim.owner,
            class: claim.class,
            bytes: claim.bytes,
        }))
    }

    fn release(&self, key: &ProviderAllocationKey, owner: MemoryOwner, bytes: usize) {
        let mut allocations = self.allocations.lock();
        let remove = allocations.get_mut(key).is_some_and(|allocation| {
            let remove_owner = allocation.owners.get_mut(&owner).is_some_and(|ownership| {
                ownership.refs = ownership.refs.saturating_sub(1);
                ownership.bytes = ownership.bytes.saturating_sub(bytes);
                ownership.refs == 0 || ownership.bytes == 0
            });
            if remove_owner {
                allocation.owners.remove(&owner);
            }
            allocation.bytes = allocation
                .owners
                .values()
                .map(|owner| owner.bytes)
                .max()
                .unwrap_or(0);
            allocation.owners.is_empty()
        });
        if remove {
            allocations.remove(key);
        }
    }

    fn resize(
        &self,
        key: &ProviderAllocationKey,
        owner: MemoryOwner,
        current_bytes: usize,
        target_bytes: usize,
    ) -> Result<(), String> {
        let mut allocations = self.allocations.lock();
        let allocation = allocations.get(key).ok_or_else(|| {
            format!(
                "provider allocation `{}` in {} no longer exists for {}",
                key.allocation_id, key.domain, owner
            )
        })?;
        let ownership = allocation.owners.get(&owner).ok_or_else(|| {
            format!(
                "provider allocation `{}` in {} is not owned by {}",
                key.allocation_id, key.domain, owner
            )
        })?;
        let target_owner_bytes = ownership
            .bytes
            .saturating_sub(current_bytes)
            .saturating_add(target_bytes);
        let other_owner_bytes = allocation
            .owners
            .iter()
            .filter(|(candidate, _)| **candidate != owner)
            .map(|(_, ownership)| ownership.bytes)
            .max()
            .unwrap_or(0);
        let target_allocation_bytes = target_owner_bytes.max(other_owner_bytes);
        self.enforce_budget(
            &allocations,
            &key.domain,
            allocation.bytes,
            target_allocation_bytes,
        )?;
        let allocation = allocations
            .get_mut(key)
            .expect("provider allocation checked above");
        let ownership = allocation
            .owners
            .get_mut(&owner)
            .expect("provider ownership checked above");
        ownership.bytes = target_owner_bytes;
        allocation.bytes = allocation
            .owners
            .values()
            .map(|owner| owner.bytes)
            .max()
            .unwrap_or(0);
        Ok(())
    }

    #[cfg(test)]
    fn allocation_count(&self) -> usize {
        self.allocations.lock().len()
    }
}

struct ProviderMemoryLease {
    ledger: Arc<ProviderMemoryLedger>,
    key: ProviderAllocationKey,
    owner: MemoryOwner,
    class: MemoryAllocationClass,
    bytes: usize,
}

impl ProviderMemoryLease {
    fn matches(&self, claim: &MemoryClaim) -> bool {
        self.key.domain == claim.domain && self.owner == claim.owner && self.class == claim.class
    }

    fn bytes(&self) -> usize {
        self.bytes
    }

    fn resize(&mut self, target_bytes: usize) -> Result<(), String> {
        self.ledger
            .resize(&self.key, self.owner, self.bytes, target_bytes)?;
        self.bytes = target_bytes;
        Ok(())
    }
}

impl Drop for ProviderMemoryLease {
    fn drop(&mut self) {
        self.ledger.release(&self.key, self.owner, self.bytes);
    }
}

/// One device's externally-observed, authority-owned KV ceiling transition.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CeilingSample {
    pub(crate) device_id: usize,
    pub(crate) foreign_bytes: usize,
    pub(crate) target_bytes: usize,
    pub(crate) smoothed_bytes: usize,
    pub(crate) previous_bytes: usize,
    pub(crate) squeezed: bool,
}

fn ceiling_is_squeezed(device_id: usize, declared: usize, live_bytes: usize) -> bool {
    let _ = device_id;
    live_bytes < declared.saturating_mul(9) / 10
}

/// Process-wide authority spanning host, CUDA, and provider adapter domains.
pub(crate) struct MemoryAuthority {
    host: Arc<HostMemoryManager>,
    #[cfg(feature = "gpu-device-pool")]
    cuda: Option<Arc<DeviceMemoryManager>>,
    providers: Arc<ProviderMemoryLedger>,
    /// Serializes cross-domain lease mutations with unified snapshots so a
    /// reader never observes only one side of an atomic grow/shrink/reconcile.
    operations: Mutex<()>,
    accounting: Mutex<AuthorityAccounting>,
    next_lease_id: AtomicU64,
    device_domains: HashMap<usize, MemoryDomainPolicy>,
    domain_budgets: HashMap<MemoryDomain, usize>,
    live_ceiling: Mutex<HashMap<usize, Arc<AtomicUsize>>>,
}

fn authority_domain_policies(
    device_info: &DeviceInfo,
    host_budget: super::host_memory::HostMemoryBudget,
    provider_limits: &ProviderMemoryLimits,
) -> (
    HashMap<usize, MemoryDomainPolicy>,
    HashMap<MemoryDomain, usize>,
) {
    let mut devices = HashMap::new();
    let mut domains = HashMap::new();
    domains.insert(MemoryDomain::Host, host_budget.safe_bytes);

    for device in &device_info.devices {
        let provider = device.backend.to_string();
        let is_cpu = provider.eq_ignore_ascii_case("cpu");
        let domain = MemoryDomain::for_provider(&provider, device.id);
        let mut budget_bytes = if is_cpu {
            host_budget.safe_bytes
        } else {
            (device.memory_mb as usize).saturating_mul(1024 * 1024)
        };
        if matches!(domain, MemoryDomain::Cuda { .. }) {
            budget_bytes =
                device_vram_cap_bytes(device.id).map_or(budget_bytes, |cap| budget_bytes.min(cap));
            budget_bytes = effective_ceiling_bytes(device.id, budget_bytes, 0);
        } else if matches!(domain, MemoryDomain::Provider { .. }) {
            if let Some(limit) = provider_limits.limit_for_domain(&domain) {
                budget_bytes = if budget_bytes == 0 {
                    limit
                } else {
                    budget_bytes.min(limit)
                };
            }
        }
        if budget_bytes == 0 {
            continue;
        }
        domains
            .entry(domain.clone())
            .and_modify(|current| *current = (*current).max(budget_bytes))
            .or_insert(budget_bytes);

        // Device IDs are backend-local in HAL. Prefer an accelerator when a
        // CPU and accelerator both use index zero, which is the placement the
        // LLM pipeline selects as its primary memory domain.
        let replace = devices
            .get(&device.id)
            .is_none_or(|current: &MemoryDomainPolicy| {
                matches!(current.domain, MemoryDomain::Host) && !is_cpu
            });
        if replace {
            devices.insert(
                device.id,
                MemoryDomainPolicy {
                    domain,
                    budget_bytes,
                },
            );
        }
    }
    // A configured adapter may not be visible to the generic HAL probe (for
    // example an ONNX execution provider loaded dynamically). Retain its limit
    // so backend reports still enter a bounded authority domain.
    for (key, limit) in &provider_limits.bytes {
        domains
            .entry(key.domain())
            .and_modify(|current| *current = (*current).min(*limit))
            .or_insert(*limit);
    }
    (devices, domains)
}

fn provider_domain_budgets(budgets: &HashMap<MemoryDomain, usize>) -> HashMap<MemoryDomain, usize> {
    budgets
        .iter()
        .filter(|(domain, _)| matches!(domain, MemoryDomain::Provider { .. }))
        .map(|(domain, bytes)| (domain.clone(), *bytes))
        .collect()
}

impl MemoryAuthority {
    /// Build the policy/accounting half of a CUDA authority without opening a
    /// physical device. Synthetic-device unit tests use this constructor so
    /// their deliberately unique IDs cannot be mistaken for CUDA ordinals.
    #[cfg(all(test, feature = "gpu-device-pool"))]
    pub(crate) fn new_accounting_only_for_test(
        device_info: &DeviceInfo,
    ) -> Result<Arc<Self>, String> {
        let host = HostMemoryManager::new(device_info);
        let provider_limits = ProviderMemoryLimits::from_env()?;
        let (device_domains, domain_budgets) =
            authority_domain_policies(device_info, host.budget(), &provider_limits);
        let provider_budgets = provider_domain_budgets(&domain_budgets);
        Ok(Arc::new(Self {
            host,
            cuda: None,
            providers: Arc::new(ProviderMemoryLedger::with_budgets(provider_budgets)),
            operations: Mutex::new(()),
            accounting: Mutex::new(AuthorityAccounting::default()),
            next_lease_id: AtomicU64::new(1),
            device_domains,
            domain_budgets,
            live_ceiling: Mutex::new(HashMap::new()),
        }))
    }

    #[cfg_attr(feature = "gpu-device-pool", allow(dead_code))]
    pub(crate) fn new(device_info: &DeviceInfo) -> Result<Arc<Self>, String> {
        #[cfg(feature = "gpu-device-pool")]
        {
            Self::new_with_cuda_plan(device_info, &DeviceMemoryBootstrapPlan::default())
        }
        #[cfg(not(feature = "gpu-device-pool"))]
        {
            let host = HostMemoryManager::new(device_info);
            let provider_limits = ProviderMemoryLimits::from_env()?;
            let (device_domains, domain_budgets) =
                authority_domain_policies(device_info, host.budget(), &provider_limits);
            let provider_budgets = provider_domain_budgets(&domain_budgets);
            Ok(Arc::new(Self {
                host,
                providers: Arc::new(ProviderMemoryLedger::with_budgets(provider_budgets)),
                operations: Mutex::new(()),
                accounting: Mutex::new(AuthorityAccounting::default()),
                next_lease_id: AtomicU64::new(1),
                device_domains,
                domain_budgets,
                live_ceiling: Mutex::new(HashMap::new()),
            }))
        }
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn new_with_cuda_plan(
        device_info: &DeviceInfo,
        bootstrap: &DeviceMemoryBootstrapPlan,
    ) -> Result<Arc<Self>, String> {
        let host = HostMemoryManager::new(device_info);
        let cuda = DeviceMemoryManager::from_env_with_plan(device_info, bootstrap)?;
        let provider_limits = ProviderMemoryLimits::from_env()?;
        let (mut device_domains, mut domain_budgets) =
            authority_domain_policies(device_info, host.budget(), &provider_limits);
        if let Some(manager) = cuda.as_ref() {
            for (device_id, budget_bytes) in manager.domain_budgets() {
                let domain = MemoryDomain::Cuda { device_id };
                device_domains.insert(
                    device_id,
                    MemoryDomainPolicy {
                        domain: domain.clone(),
                        budget_bytes,
                    },
                );
                domain_budgets.insert(domain, budget_bytes);
            }
        }
        let provider_budgets = provider_domain_budgets(&domain_budgets);
        Ok(Arc::new(Self {
            host,
            cuda,
            providers: Arc::new(ProviderMemoryLedger::with_budgets(provider_budgets)),
            operations: Mutex::new(()),
            accounting: Mutex::new(AuthorityAccounting::default()),
            next_lease_id: AtomicU64::new(1),
            device_domains,
            domain_budgets,
            live_ceiling: Mutex::new(HashMap::new()),
        }))
    }

    pub(crate) fn domain_for_device(&self, device_id: usize) -> MemoryDomain {
        self.device_domains
            .get(&device_id)
            .map(|policy| policy.domain.clone())
            .unwrap_or(MemoryDomain::Host)
    }

    #[cfg(all(feature = "gpu-device-pool", any(target_os = "linux", test)))]
    pub(crate) fn cuda_device(
        &self,
        device_id: usize,
    ) -> Result<Arc<cudarc::driver::CudaDevice>, String> {
        self.cuda
            .as_ref()
            .ok_or_else(|| "runtime has no CUDA memory authority".to_string())?
            .cuda_device(device_id)
    }

    /// KV policy capacity for one HAL device. The authority owns the physical
    /// budget; callers only translate these bytes into logical blocks.
    pub(crate) fn kv_budget_bytes(&self, device_id: usize) -> usize {
        let Some(policy) = self.device_domains.get(&device_id) else {
            return self
                .host
                .budget()
                .class_limit(MemoryAllocationClass::KvCache);
        };
        if matches!(policy.domain, MemoryDomain::Host) {
            return self
                .host
                .budget()
                .class_limit(MemoryAllocationClass::KvCache);
        }
        let ceiling = self
            .live_ceiling
            .lock()
            .get(&device_id)
            .map(|value| value.load(Ordering::Relaxed))
            .unwrap_or(policy.budget_bytes);
        ceiling / 2
    }

    pub(crate) fn kv_device_budgets(&self) -> Vec<(usize, usize)> {
        let mut budgets: Vec<_> = self
            .device_domains
            .keys()
            .copied()
            .map(|device_id| (device_id, self.kv_budget_bytes(device_id)))
            .collect();
        budgets.sort_unstable_by_key(|(device_id, _)| *device_id);
        budgets
    }

    /// Hard admission budget for a backend-declared physical domain. External
    /// KV participants use this during registration so an unknown device or an
    /// unbounded provider fails before its first request.
    pub(crate) fn domain_budget_bytes(&self, domain: &MemoryDomain) -> usize {
        self.budget_for_domain(domain)
    }

    pub(crate) fn supports_external_leases(&self, domain: &MemoryDomain) -> bool {
        if self.domain_budget_bytes(domain) == 0 {
            return false;
        }
        match domain {
            MemoryDomain::Cuda { .. } => {
                #[cfg(feature = "gpu-device-pool")]
                {
                    self.cuda.is_some()
                }
                #[cfg(not(feature = "gpu-device-pool"))]
                {
                    false
                }
            }
            MemoryDomain::Host
            | MemoryDomain::HostPinned { .. }
            | MemoryDomain::HostMapped { .. }
            | MemoryDomain::Provider { .. } => true,
        }
    }

    pub(crate) fn observe_process_memory(&self, bytes: usize) {
        self.accounting
            .lock()
            .observed_domains
            .insert(MemoryDomain::Host, bytes);
    }

    pub(crate) fn observe_cuda_memory_total(&self, bytes: usize) {
        let devices: Vec<_> = self
            .device_domains
            .iter()
            .filter_map(|(&device_id, policy)| {
                matches!(policy.domain, MemoryDomain::Cuda { .. })
                    .then_some((device_id, policy.budget_bytes))
            })
            .collect();
        let total_budget = devices
            .iter()
            .map(|(_, budget)| *budget)
            .fold(0usize, usize::saturating_add);
        if total_budget == 0 {
            return;
        }
        let mut accounting = self.accounting.lock();
        let mut assigned = 0usize;
        for (index, (device_id, budget)) in devices.iter().enumerate() {
            let observed = if index + 1 == devices.len() {
                bytes.saturating_sub(assigned)
            } else {
                bytes.saturating_mul(*budget) / total_budget
            };
            assigned = assigned.saturating_add(observed);
            #[cfg(feature = "gpu-device-pool")]
            let observed = observed.saturating_sub(
                self.cuda
                    .as_ref()
                    .map(|manager| manager.pool_capacity_bytes(*device_id))
                    .unwrap_or(0),
            );
            accounting.observed_domains.insert(
                MemoryDomain::Cuda {
                    device_id: *device_id,
                },
                observed,
            );
        }
    }

    /// Reconcile externally-observed device pressure into the authority. This
    /// is the only source of the soft ceiling consumed by KV policy.
    pub(crate) fn reconcile_external_device_memory(
        &self,
        foreign: &HashMap<usize, usize>,
    ) -> Vec<CeilingSample> {
        let _operation = self.operations.lock();
        let mut changed = false;
        let mut samples = Vec::with_capacity(self.device_domains.len());
        {
            let mut live = self.live_ceiling.lock();
            for (&device_id, policy) in &self.device_domains {
                if !matches!(policy.domain, MemoryDomain::Cuda { .. }) {
                    continue;
                }
                let declared = policy.budget_bytes;
                let foreign_bytes = foreign.get(&device_id).copied().unwrap_or(0);
                let target = declared.saturating_sub(foreign_bytes);
                let atom = live
                    .entry(device_id)
                    .or_insert_with(|| Arc::new(AtomicUsize::new(declared)));
                let previous = atom.load(Ordering::Relaxed);
                let smoothed = smooth_ceiling_bytes(previous, target);
                if smoothed != previous {
                    atom.store(smoothed, Ordering::Relaxed);
                    changed = true;
                }
                samples.push(CeilingSample {
                    device_id,
                    foreign_bytes,
                    target_bytes: target,
                    smoothed_bytes: smoothed,
                    previous_bytes: previous,
                    squeezed: ceiling_is_squeezed(device_id, declared, smoothed),
                });
            }
        }
        if changed {
            log::debug!("[memory-authority] external device ceilings reconciled");
        }
        samples
    }

    pub(crate) fn foreign_pressure_active(&self) -> bool {
        let live = self.live_ceiling.lock();
        live.iter().any(|(device_id, value)| {
            self.device_domains.get(device_id).is_some_and(|policy| {
                ceiling_is_squeezed(
                    *device_id,
                    policy.budget_bytes,
                    value.load(Ordering::Relaxed),
                )
            })
        })
    }

    fn register_lease(&self, claims: &[MemoryClaim]) -> u64 {
        let _operation = self.operations.lock();
        self.register_lease_locked(claims)
    }

    /// Register one lease while the caller holds [`Self::operations`].
    ///
    /// Startup reservations sometimes need to choose among several whole,
    /// geometry-aligned candidates and reserve the selected candidate in the
    /// same authority transaction. Splitting registration from the locked
    /// form would either deadlock on the non-reentrant operation mutex or
    /// reopen the admission race between selection and reservation.
    fn register_lease_locked(&self, claims: &[MemoryClaim]) -> u64 {
        let lease_id = self.next_lease_id.fetch_add(1, Ordering::Relaxed);
        let mut rows = HashMap::<MemoryRowKey, MemoryAccountingBytes>::new();
        for claim in claims {
            let bytes = rows.entry(MemoryRowKey::from_claim(claim)).or_default();
            bytes.planned = bytes.planned.saturating_add(claim.bytes);
            bytes.reserved = bytes.reserved.saturating_add(claim.bytes);
        }
        self.accounting.lock().leases.insert(lease_id, rows);
        lease_id
    }

    fn replace_lease_rows(
        &self,
        lease_id: u64,
        planned: &HashMap<MemoryRowKey, usize>,
        claims: &[MemoryClaim],
        committed: &HashMap<MemoryRowKey, usize>,
        observed: &HashMap<MemoryRowKey, usize>,
        mark_committed: bool,
    ) {
        let mut rows = HashMap::<MemoryRowKey, MemoryAccountingBytes>::new();
        for (key, bytes) in planned {
            rows.entry(key.clone()).or_default().planned = *bytes;
        }
        for claim in claims {
            let row = rows.entry(MemoryRowKey::from_claim(claim)).or_default();
            row.reserved = row.reserved.saturating_add(claim.bytes);
        }
        for (key, bytes) in committed {
            rows.entry(key.clone()).or_default().committed = *bytes;
        }
        for (key, bytes) in observed {
            rows.entry(key.clone()).or_default().observed = *bytes;
        }
        if mark_committed {
            for row in rows.values_mut() {
                row.committed = row.committed.max(row.reserved);
            }
        }
        self.accounting.lock().leases.insert(lease_id, rows);
    }

    fn unregister_lease(&self, lease_id: u64) {
        let _operation = self.operations.lock();
        self.unregister_lease_locked(lease_id);
    }

    fn unregister_lease_locked(&self, lease_id: u64) {
        self.accounting.lock().leases.remove(&lease_id);
    }

    pub(crate) fn snapshot(&self) -> MemorySnapshot {
        let _operation = self.operations.lock();
        self.snapshot_unlocked()
    }

    fn snapshot_unlocked(&self) -> MemorySnapshot {
        #[cfg(feature = "gpu-device-pool")]
        let pool_observations = self
            .cuda
            .as_ref()
            .map(|manager| manager.observed_pool_claims())
            .unwrap_or_default();
        #[cfg(feature = "gpu-device-pool")]
        let pool_observed_by_domain = pool_observations.iter().fold(
            HashMap::<MemoryDomain, usize>::new(),
            |mut domains, claim| {
                let bytes = domains.entry(claim.domain.clone()).or_default();
                *bytes = bytes.saturating_add(claim.bytes);
                domains
            },
        );
        let accounting = self.accounting.lock();
        let mut rows = HashMap::<MemoryRowKey, MemoryAccountingBytes>::new();
        for lease in accounting.leases.values() {
            for (key, value) in lease {
                let aggregate = rows.entry(key.clone()).or_default();
                aggregate.planned = aggregate.planned.saturating_add(value.planned);
                aggregate.reserved = aggregate.reserved.saturating_add(value.reserved);
                aggregate.committed = aggregate.committed.saturating_add(value.committed);
                aggregate.observed = aggregate.observed.saturating_add(value.observed);
            }
        }
        #[cfg(feature = "gpu-device-pool")]
        for claim in &pool_observations {
            let row = rows.entry(MemoryRowKey::from_claim(claim)).or_default();
            row.observed = row.observed.max(claim.bytes);
        }
        let mut snapshot_rows: Vec<_> = rows
            .iter()
            .map(|(key, value)| MemorySnapshotRow {
                domain: key.domain.clone(),
                owner: key.owner,
                class: key.class,
                planned_bytes: value.planned,
                reserved_bytes: value.reserved,
                committed_bytes: value.committed,
                observed_bytes: value.observed,
            })
            .collect();
        snapshot_rows.sort_by_key(|row| {
            (
                row.domain.to_string(),
                row.owner.model_id,
                row.owner.replica_id,
                row.class.to_string(),
            )
        });

        let mut domains = self.domain_budgets.clone();
        for (device_id, ceiling) in self.live_ceiling.lock().iter() {
            domains.insert(
                MemoryDomain::Cuda {
                    device_id: *device_id,
                },
                ceiling.load(Ordering::Relaxed),
            );
        }
        for row in &snapshot_rows {
            domains
                .entry(row.domain.clone())
                .or_insert_with(|| self.budget_for_domain(&row.domain));
        }
        for domain in accounting.observed_domains.keys() {
            domains
                .entry(domain.clone())
                .or_insert_with(|| self.budget_for_domain(domain));
        }
        let mut domain_rows = Vec::with_capacity(domains.len());
        for (domain, budget_bytes) in domains {
            let (planned, reserved, committed, attributed_observed) = snapshot_rows
                .iter()
                .filter(|row| row.domain == domain)
                .fold((0usize, 0usize, 0usize, 0usize), |acc, row| {
                    (
                        acc.0.saturating_add(row.planned_bytes),
                        acc.1.saturating_add(row.reserved_bytes),
                        acc.2.saturating_add(row.committed_bytes),
                        acc.3.saturating_add(row.observed_bytes),
                    )
                });
            let sampled_observed = accounting
                .observed_domains
                .get(&domain)
                .copied()
                .unwrap_or(0);
            #[cfg(feature = "gpu-device-pool")]
            let sampled_observed = if matches!(domain, MemoryDomain::Cuda { .. }) {
                sampled_observed
                    .saturating_add(pool_observed_by_domain.get(&domain).copied().unwrap_or(0))
            } else {
                sampled_observed
            };
            let observed = sampled_observed.max(attributed_observed);
            let used = reserved.max(committed).max(observed);
            domain_rows.push(MemoryDomainSnapshot {
                domain,
                budget_bytes,
                planned_bytes: planned,
                reserved_bytes: reserved,
                committed_bytes: committed,
                observed_bytes: observed,
                available_bytes: budget_bytes.saturating_sub(used),
            });
        }
        domain_rows.sort_by_key(|row| row.domain.to_string());
        drop(accounting);
        MemorySnapshot {
            rows: snapshot_rows,
            domains: domain_rows,
            foreign_pressure_active: self.foreign_pressure_active(),
        }
    }

    /// Validate a prospective lease increase against the exact state exposed
    /// to pressure and autoscaling policy. Callers hold `operations`, making
    /// this check and the following adapter reservations one transaction.
    fn preflight_growth_locked(&self, plan: &MemoryPlan) -> Result<(), String> {
        let snapshot = self.snapshot_unlocked();
        self.preflight_growth_against_snapshot_locked(plan, &snapshot)
    }

    fn preflight_growth_against_snapshot_locked(
        &self,
        plan: &MemoryPlan,
        snapshot: &MemorySnapshot,
    ) -> Result<(), String> {
        if plan.claims().iter().all(|claim| claim.bytes == 0) {
            return Ok(());
        }
        let mut additions_by_domain = HashMap::<MemoryDomain, usize>::new();
        for claim in plan.claims() {
            let bytes = additions_by_domain.entry(claim.domain.clone()).or_default();
            *bytes = bytes.saturating_add(claim.bytes);
        }
        for (domain, requested) in additions_by_domain {
            if requested == 0 {
                continue;
            }
            let Some(current) = snapshot.domain(&domain) else {
                if matches!(
                    domain,
                    MemoryDomain::HostPinned { .. }
                        | MemoryDomain::HostMapped { .. }
                        | MemoryDomain::Provider { .. }
                ) {
                    // Provider-specific limits are intentionally deferred; the
                    // provider adapter still owns and reports the allocation.
                    // Pinned/mapped rows are enforced by the aggregate host
                    // snapshot immediately below.
                    continue;
                }
                return Err(format!(
                    "memory authority has no budget for {domain}: requested={requested} bytes"
                ));
            };
            if current.budget_bytes == 0 && matches!(domain, MemoryDomain::Provider { .. }) {
                continue;
            }
            if requested > current.available_bytes {
                return Err(format!(
                    "memory authority rejected growth in {domain}: requested={requested} available={} planned={} reserved={} committed={} observed={} budget={} bytes",
                    current.available_bytes,
                    current.planned_bytes,
                    current.reserved_bytes,
                    current.committed_bytes,
                    current.observed_bytes,
                    current.budget_bytes,
                ));
            }
        }

        #[cfg(feature = "gpu-device-pool")]
        if let Some(manager) = self.cuda.as_ref() {
            let mut additions_by_workload = HashMap::<(usize, MemoryOwner), usize>::new();
            for claim in plan.claims().iter().filter(|claim| {
                matches!(claim.domain, MemoryDomain::Cuda { .. })
                    && matches!(claim.source, MemoryClaimSource::Runtime { .. })
                    && claim.bytes > 0
            }) {
                let MemoryDomain::Cuda { device_id } = claim.domain else {
                    unreachable!("filtered CUDA claim")
                };
                let Some(pool_available) = manager.runtime_allocatable(claim)? else {
                    // Model-load admission establishes the workload quota after
                    // this central reservation; request paths always find it.
                    continue;
                };
                let reserved = snapshot
                    .rows
                    .iter()
                    .filter(|row| row.domain == claim.domain && row.owner == claim.owner)
                    .map(|row| row.reserved_bytes.max(row.committed_bytes))
                    .fold(0usize, usize::saturating_add);
                let observed = snapshot
                    .rows
                    .iter()
                    .filter(|row| row.domain == claim.domain && row.owner == claim.owner)
                    .map(|row| row.observed_bytes)
                    .fold(0usize, usize::saturating_add);
                let outstanding = reserved.saturating_sub(observed);
                let planned = additions_by_workload
                    .entry((device_id, claim.owner))
                    .or_default();
                let available = pool_available
                    .saturating_sub(outstanding)
                    .saturating_sub(*planned);
                if claim.bytes > available {
                    return Err(format!(
                        "CUDA memory authority rejected growth for {}: requested={} available={} outstanding={} pool_allocatable={} device={} bytes",
                        claim.owner,
                        claim.bytes,
                        available,
                        outstanding,
                        pool_available,
                        device_id,
                    ));
                }
                *planned = planned.saturating_add(claim.bytes);
            }
        }

        // Host, pinned, and mapped rows have distinct metric domains but draw
        // from one physical budget and the same two class groups.
        let is_host = |domain: &MemoryDomain| {
            matches!(
                domain,
                MemoryDomain::Host
                    | MemoryDomain::HostPinned { .. }
                    | MemoryDomain::HostMapped { .. }
            )
        };
        let host_additions = plan
            .claims()
            .iter()
            .filter(|claim| is_host(&claim.domain))
            .map(|claim| claim.bytes)
            .fold(0usize, usize::saturating_add);
        if host_additions > 0 {
            let host_reserved = snapshot
                .rows
                .iter()
                .filter(|row| is_host(&row.domain))
                .map(|row| row.reserved_bytes)
                .fold(0usize, usize::saturating_add);
            let host_committed = snapshot
                .rows
                .iter()
                .filter(|row| is_host(&row.domain))
                .map(|row| row.committed_bytes)
                .fold(0usize, usize::saturating_add);
            let attributed_observed = snapshot
                .rows
                .iter()
                .filter(|row| is_host(&row.domain))
                .map(|row| row.observed_bytes)
                .fold(0usize, usize::saturating_add);
            let sampled_observed = snapshot
                .domain(&MemoryDomain::Host)
                .map(|domain| domain.observed_bytes)
                .unwrap_or(0);
            let used = host_reserved
                .max(host_committed)
                .max(attributed_observed.max(sampled_observed));
            let host_available = self.host.budget().safe_bytes.saturating_sub(used);
            if host_additions > host_available {
                return Err(format!(
                    "host memory authority rejected growth: requested={host_additions} available={host_available} reserved={host_reserved} committed={host_committed} observed={} budget={} bytes",
                    attributed_observed.max(sampled_observed),
                    self.host.budget().safe_bytes,
                ));
            }

            for kv_group in [false, true] {
                let group_additions = plan
                    .claims()
                    .iter()
                    .filter(|claim| {
                        is_host(&claim.domain)
                            && (claim.class == MemoryAllocationClass::KvCache) == kv_group
                    })
                    .map(|claim| claim.bytes)
                    .fold(0usize, usize::saturating_add);
                let group_used = snapshot
                    .rows
                    .iter()
                    .filter(|row| {
                        is_host(&row.domain)
                            && (row.class == MemoryAllocationClass::KvCache) == kv_group
                    })
                    .map(MemorySnapshotRow::used_bytes)
                    .fold(0usize, usize::saturating_add);
                let class = if kv_group {
                    MemoryAllocationClass::KvCache
                } else {
                    MemoryAllocationClass::ModelSession
                };
                let class_limit = self.host.budget().class_limit(class);
                if group_additions > class_limit.saturating_sub(group_used) {
                    return Err(format!(
                        "host memory authority rejected {} growth: requested={} used={} class_budget={} bytes",
                        if kv_group { "KV" } else { "model/request" },
                        group_additions,
                        group_used,
                        class_limit,
                    ));
                }
            }
        }
        Ok(())
    }

    fn budget_for_domain(&self, domain: &MemoryDomain) -> usize {
        match domain {
            MemoryDomain::Host
            | MemoryDomain::HostPinned { .. }
            | MemoryDomain::HostMapped { .. } => self.host.budget().safe_bytes,
            MemoryDomain::Cuda { device_id } => self
                .live_ceiling
                .lock()
                .get(device_id)
                .map(|value| value.load(Ordering::Relaxed))
                .or_else(|| self.domain_budgets.get(domain).copied())
                .unwrap_or(0),
            MemoryDomain::Provider { provider, .. } => self
                .domain_budgets
                .get(domain)
                .copied()
                .or_else(|| {
                    self.domain_budgets
                        .get(&MemoryDomain::Provider {
                            provider: provider.clone(),
                            device_id: None,
                        })
                        .copied()
                })
                .unwrap_or(0),
        }
    }

    /// Build a load plan from the selected devices and the backend's external
    /// allocation report. Empty CUDA reports describe pool-backed allocations;
    /// non-CUDA providers are represented by accounting-only adapter claims.
    #[cfg(test)]
    pub(crate) fn model_load_plan(
        &self,
        domains: &[MemoryDomain],
        owner: MemoryOwner,
        session_bytes: usize,
        workspace_bytes: usize,
        external_report: &ExternalDeviceMemoryReport,
    ) -> Result<MemoryPlan, String> {
        if domains.is_empty() {
            return Err(format!("memory plan for {owner} has no target domains"));
        }
        let mut plan = MemoryPlan::new();
        let mut selected = Vec::new();
        for domain in domains {
            if !selected.contains(domain) {
                selected.push(domain.clone());
            }
        }
        let mut host_planned = false;

        for domain in selected {
            match &domain {
                MemoryDomain::Host
                | MemoryDomain::HostPinned { .. }
                | MemoryDomain::HostMapped { .. } => {
                    if host_planned {
                        continue;
                    }
                    host_planned = true;
                    plan.push(MemoryClaim::runtime(
                        domain.clone(),
                        owner,
                        MemoryAllocationClass::ModelSession,
                        session_bytes,
                    ));
                    plan.push(MemoryClaim::runtime(
                        domain.clone(),
                        owner,
                        MemoryAllocationClass::TransientWorkspace,
                        workspace_bytes,
                    ));
                    plan.push(MemoryClaim::runtime(
                        domain.clone(),
                        owner,
                        MemoryAllocationClass::KvCache,
                        0,
                    ));
                }
                MemoryDomain::Cuda { device_id } => {
                    let external: Vec<_> = external_report
                        .allocations
                        .iter()
                        .filter(|allocation| allocation.device_id == *device_id)
                        .collect();
                    if external.is_empty() {
                        plan.push(MemoryClaim::runtime(
                            domain.clone(),
                            owner,
                            MemoryAllocationClass::PersistentWeights,
                            session_bytes,
                        ));
                        plan.push(MemoryClaim::runtime(
                            domain.clone(),
                            owner,
                            MemoryAllocationClass::TransientWorkspace,
                            workspace_bytes,
                        ));
                    } else {
                        for allocation in external {
                            plan.push(MemoryClaim::external(
                                domain.clone(),
                                owner,
                                classify_external_allocation(&allocation.allocation_id),
                                allocation.allocation_id.clone(),
                                allocation.bytes,
                            ));
                        }
                    }
                    // Elastic KV has no fixed byte reservation at model-load
                    // time, but still belongs to this model/replica lease.
                    plan.push(MemoryClaim::runtime(
                        domain.clone(),
                        owner,
                        MemoryAllocationClass::KvCache,
                        0,
                    ));
                }
                MemoryDomain::Provider { .. } => {
                    plan.push(MemoryClaim::runtime(
                        domain.clone(),
                        owner,
                        MemoryAllocationClass::PersistentWeights,
                        session_bytes,
                    ));
                    plan.push(MemoryClaim::runtime(
                        domain.clone(),
                        owner,
                        MemoryAllocationClass::TransientWorkspace,
                        workspace_bytes,
                    ));
                    plan.push(MemoryClaim::runtime(
                        domain.clone(),
                        owner,
                        MemoryAllocationClass::KvCache,
                        0,
                    ));
                }
            }
        }
        Ok(plan)
    }

    /// Build a load plan from the backend-neutral report. Explicit backend
    /// rows retain their domain, class, source, and stable allocation ID;
    /// selected domains missing from a legacy report receive conservative
    /// model/session and workspace estimates.
    pub(crate) fn model_load_plan_with_report(
        &self,
        domains: &[MemoryDomain],
        owner: MemoryOwner,
        session_bytes: usize,
        workspace_bytes: usize,
        report: &BackendMemoryReport,
    ) -> Result<MemoryPlan, String> {
        if domains.is_empty() {
            return Err(format!("memory plan for {owner} has no target domains"));
        }
        let mut selected = Vec::new();
        for domain in domains {
            if !selected.contains(domain) {
                selected.push(domain.clone());
            }
        }
        let mut plan = MemoryPlan::from_backend_report(owner, report);
        plan.claims.retain(|claim| {
            matches!(
                claim.domain,
                MemoryDomain::Host
                    | MemoryDomain::HostPinned { .. }
                    | MemoryDomain::HostMapped { .. }
            ) || selected.contains(&claim.domain)
        });

        for domain in selected {
            let has_model = plan.claims.iter().any(|claim| {
                claim.domain == domain
                    && matches!(
                        claim.class,
                        MemoryAllocationClass::PersistentWeights
                            | MemoryAllocationClass::ModelSession
                    )
            });
            let has_workspace = plan.claims.iter().any(|claim| {
                claim.domain == domain && claim.class == MemoryAllocationClass::TransientWorkspace
            });
            let has_kv = plan.claims.iter().any(|claim| {
                claim.domain == domain && claim.class == MemoryAllocationClass::KvCache
            });
            if !has_model {
                plan.push(MemoryClaim::runtime(
                    domain.clone(),
                    owner,
                    if matches!(
                        domain,
                        MemoryDomain::Host
                            | MemoryDomain::HostPinned { .. }
                            | MemoryDomain::HostMapped { .. }
                    ) {
                        MemoryAllocationClass::ModelSession
                    } else {
                        MemoryAllocationClass::PersistentWeights
                    },
                    session_bytes,
                ));
            }
            if !has_workspace {
                plan.push(MemoryClaim::runtime(
                    domain.clone(),
                    owner,
                    MemoryAllocationClass::TransientWorkspace,
                    workspace_bytes,
                ));
            }
            if !has_kv {
                plan.push(MemoryClaim::runtime(
                    domain,
                    owner,
                    MemoryAllocationClass::KvCache,
                    0,
                ));
            }
        }
        Ok(plan)
    }

    /// Admit every claim whose adapter is synchronous. If a later claim fails,
    /// already-created leases are dropped and the plan is atomic to callers.
    pub(crate) fn admit(self: &Arc<Self>, plan: &MemoryPlan) -> Result<MemoryLease, String> {
        let mut lease = MemoryLease::new(Arc::clone(self), Vec::new());
        lease.grow(plan)?;
        Ok(lease)
    }

    /// Admit the first candidate that fits, preserving caller order.
    ///
    /// All candidates are evaluated and the winner is reserved while holding
    /// the same authority operation lock. This is intended for plans such as
    /// an exact external KV cache where capacity may be reduced only in whole,
    /// precomputed geometry units; callers provide candidates from most to
    /// least preferred and include their hard minimum as the final entry.
    /// Returning the snapshot used for the decision makes the grant auditable.
    #[allow(dead_code)]
    pub(crate) fn admit_first_fitting(
        self: &Arc<Self>,
        candidates: &[MemoryPlan],
    ) -> Result<(MemoryLease, usize, MemorySnapshot), String> {
        if candidates.is_empty() {
            return Err("memory admission requires at least one candidate".to_string());
        }

        let _operation = self.operations.lock();
        // Candidate order is part of one auditable decision. Observed-domain
        // samplers do not all participate in `operations`, so freeze their
        // point-in-time values once instead of allowing a later fallback to
        // be judged against a different snapshot from the preferred plan.
        let decision_snapshot = self.snapshot_unlocked();
        let mut failures = Vec::with_capacity(candidates.len());

        for (index, candidate) in candidates.iter().enumerate() {
            if candidate.claims().iter().all(|claim| claim.bytes == 0) {
                failures.push(format!(
                    "candidate {index}: memory admission candidate has no positive-byte claims"
                ));
                continue;
            }
            if let Err(error) =
                self.preflight_growth_against_snapshot_locked(candidate, &decision_snapshot)
            {
                failures.push(format!("candidate {index}: {error}"));
                continue;
            }

            let mut lease = MemoryLease::new_locked(Arc::clone(self), Vec::new());
            match lease.grow_pre_admitted_locked(candidate) {
                Ok(()) => return Ok((lease, index, decision_snapshot.clone())),
                Err(error) => {
                    // `grow_impl_locked` stages physical adapter leases and
                    // publishes claims only after every reservation succeeds,
                    // so a failed candidate leaves this registered lease empty.
                    self.unregister_lease_locked(lease.lease_id);
                    lease.released = true;
                    failures.push(format!("candidate {index}: {error}"));
                }
            }
        }

        Err(format!(
            "memory authority rejected every candidate: {}",
            failures.join("; ")
        ))
    }

    /// Begin a model-load transaction. Host and provider claims are reserved
    /// first; CUDA adapters then serialize load sampling per device.
    pub(crate) async fn begin_load(
        self: &Arc<Self>,
        plan: &MemoryPlan,
        kind: EngineKind,
    ) -> Result<MemoryAdmission, String> {
        let non_cuda_plan = MemoryPlan {
            claims: plan
                .claims()
                .iter()
                .filter(|claim| !matches!(claim.domain, MemoryDomain::Cuda { .. }))
                .cloned()
                .collect(),
        };
        let tracks_host_load = non_cuda_plan.claims().iter().any(|claim| {
            matches!(
                claim.domain,
                MemoryDomain::Host
                    | MemoryDomain::HostPinned { .. }
                    | MemoryDomain::HostMapped { .. }
            ) && claim.bytes != 0
                && matches!(
                    claim.class,
                    MemoryAllocationClass::PersistentWeights
                        | MemoryAllocationClass::ModelSession
                        | MemoryAllocationClass::TransientWorkspace
                        | MemoryAllocationClass::ExternallyOwned
                )
        });
        let host_load = if tracks_host_load {
            Some(self.host.begin_load_reconciliation().await)
        } else {
            None
        };
        let lease = self.admit(&non_cuda_plan)?;
        #[allow(unused_mut)]
        let mut admission = MemoryAdmission {
            _authority: Arc::clone(self),
            lease: Some(lease),
            host_load,
            cuda_plan: MemoryPlan {
                claims: plan
                    .claims()
                    .iter()
                    .filter(|claim| matches!(claim.domain, MemoryDomain::Cuda { .. }))
                    .cloned()
                    .collect(),
            },
            #[cfg(feature = "gpu-device-pool")]
            cuda: Vec::new(),
            reconciled: false,
        };

        if !admission.cuda_plan.claims().is_empty() {
            #[cfg(feature = "gpu-device-pool")]
            admission
                .lease
                .as_mut()
                .expect("memory admission lease")
                .add_unbacked_claims(admission.cuda_plan.claims())?;
            #[cfg(not(feature = "gpu-device-pool"))]
            return Err("CUDA load requires the CUDA memory authority".to_string());
        }

        #[cfg(not(feature = "gpu-device-pool"))]
        let _ = kind;

        #[cfg(feature = "gpu-device-pool")]
        if let Some(manager) = self.cuda.as_ref() {
            for device_id in admission.cuda_plan.cuda_device_ids() {
                let report = admission.cuda_plan.external_report_for_cuda(device_id);
                let owner = owner_for_device(&admission.cuda_plan, device_id)?;
                if let Some(cuda) = manager
                    .begin_admission(device_id, owner, kind, &report)
                    .await?
                {
                    admission.cuda.push(cuda);
                }
            }
        }
        Ok(admission)
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) async fn begin_swap(
        self: &Arc<Self>,
        plan: &MemoryPlan,
    ) -> Result<MemorySwapLease, String> {
        let non_cuda_plan = MemoryPlan {
            claims: plan
                .claims()
                .iter()
                .filter(|claim| !matches!(claim.domain, MemoryDomain::Cuda { .. }))
                .cloned()
                .collect(),
        };
        let cuda_plan = MemoryPlan {
            claims: plan
                .claims()
                .iter()
                .filter(|claim| matches!(claim.domain, MemoryDomain::Cuda { .. }))
                .cloned()
                .collect(),
        };
        let mut lease = self.admit(&non_cuda_plan)?;
        if !cuda_plan.claims().is_empty() {
            lease.add_unbacked_claims(cuda_plan.claims())?;
        }

        let mut cuda = Vec::new();
        if !cuda_plan.claims().is_empty() {
            let manager = self
                .cuda
                .as_ref()
                .ok_or_else(|| "CUDA swap requires the CUDA memory authority".to_string())?;
            for device_id in cuda_plan.cuda_device_ids() {
                let report = cuda_plan.external_report_for_cuda(device_id);
                let owner = owner_for_device(&cuda_plan, device_id)?;
                if let Some(admission) = manager
                    .begin_swap_admission(device_id, owner, &report)
                    .await?
                {
                    cuda.push(admission);
                }
            }
        }
        Ok(MemorySwapLease {
            _lease: lease,
            _cuda: cuda,
        })
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn cuda_pool(
        &self,
        device_id: usize,
    ) -> Option<Arc<kapsl_hal::gpu_arena::GpuDevicePool>> {
        self.cuda
            .as_ref()
            .and_then(|manager| manager.pool(device_id))
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn uses_cuda_environment_allocator(&self, device_id: usize) -> bool {
        self.cuda
            .as_ref()
            .is_some_and(|manager| manager.has_pool(device_id))
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn ensure_cuda_pools(
        &self,
        bootstrap: &DeviceMemoryBootstrapPlan,
    ) -> Result<(), String> {
        if let Some(manager) = self.cuda.as_ref() {
            manager.ensure_pools_for_plan(bootstrap)?;
        }
        Ok(())
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn attach_cuda_metrics(&self, metrics: kapsl_monitor::metrics::KapslMetrics) {
        if let Some(manager) = self.cuda.as_ref() {
            manager.attach_metrics(metrics);
        }
    }

    pub(crate) fn refresh_cuda_pool_metrics(&self) {
        #[cfg(feature = "gpu-device-pool")]
        if let Some(manager) = self.cuda.as_ref() {
            manager.refresh_pool_metrics();
        }
    }
}

#[cfg(any(feature = "gpu-device-pool", test))]
fn owner_for_device(plan: &MemoryPlan, device_id: usize) -> Result<MemoryOwner, String> {
    let mut owners = plan.claims().iter().filter_map(|claim| match claim.domain {
        MemoryDomain::Cuda {
            device_id: candidate,
        } if candidate == device_id => Some(claim.owner),
        _ => None,
    });
    let owner = owners
        .next()
        .ok_or_else(|| format!("CUDA device {device_id} plan has no owner"))?;
    if owners.any(|candidate| candidate != owner) {
        return Err(format!(
            "CUDA device {device_id} plan mixes multiple model/replica owners"
        ));
    }
    Ok(owner)
}

/// In-flight load transaction. Dropping it rolls back every domain.
#[must_use = "reconcile and commit the memory admission after backend load"]
pub(crate) struct MemoryAdmission {
    _authority: Arc<MemoryAuthority>,
    lease: Option<MemoryLease>,
    host_load: Option<HostMemoryLoadAdmission>,
    cuda_plan: MemoryPlan,
    #[cfg(feature = "gpu-device-pool")]
    cuda: Vec<DeviceMemoryAdmission>,
    reconciled: bool,
}

impl MemoryAdmission {
    pub(crate) fn reconcile(&mut self, actual_report: &BackendMemoryReport) -> Result<(), String> {
        if let Some(mut host_load) = self.host_load.take() {
            let lease = self.lease.as_mut().expect("memory admission lease");
            host_load.reconcile(&mut lease.host)?;
        }

        #[cfg(feature = "gpu-device-pool")]
        {
            let external_cuda = external_cuda_report_from_backend(actual_report);
            for admission in &mut self.cuda {
                admission.reconcile(&external_cuda)?;
            }
        }
        if let Some(lease) = self.lease.as_mut() {
            lease.record_loaded_report(actual_report)?;
        }
        reconcile_backend_claim_bytes(&mut self.cuda_plan.claims, actual_report);
        self.reconciled = true;
        Ok(())
    }

    pub(crate) fn commit(mut self) -> MemoryLease {
        assert!(
            self.reconciled,
            "memory admission must be reconciled before commit"
        );
        let mut lease = self.lease.take().expect("memory admission lease");
        #[cfg(feature = "gpu-device-pool")]
        for item in std::mem::take(&mut self.cuda) {
            let item = item.commit();
            debug_assert!(self
                .cuda_plan
                .claims()
                .iter()
                .any(|claim| claim.owner == item.owner()));
            lease.cuda.push(item);
        }
        lease.mark_committed();
        lease
    }
}

/// RAII ownership of all admitted memory in a plan.
pub(crate) struct MemoryLease {
    authority: Arc<MemoryAuthority>,
    lease_id: u64,
    claims: Vec<MemoryClaim>,
    planned: HashMap<MemoryRowKey, usize>,
    committed: HashMap<MemoryRowKey, usize>,
    observed: HashMap<MemoryRowKey, usize>,
    host: Vec<HostMemoryLease>,
    providers: Vec<ProviderMemoryLease>,
    #[cfg(feature = "gpu-device-pool")]
    cuda: Vec<DeviceMemoryLease>,
    #[cfg(feature = "gpu-device-pool")]
    cuda_transient: Vec<DeviceMemoryTransientLease>,
    released: bool,
}

impl MemoryLease {
    fn new(authority: Arc<MemoryAuthority>, claims: Vec<MemoryClaim>) -> Self {
        let lease_id = authority.register_lease(&claims);
        Self::from_registered(authority, lease_id, claims)
    }

    #[allow(dead_code)]
    fn new_locked(authority: Arc<MemoryAuthority>, claims: Vec<MemoryClaim>) -> Self {
        let lease_id = authority.register_lease_locked(&claims);
        Self::from_registered(authority, lease_id, claims)
    }

    fn from_registered(
        authority: Arc<MemoryAuthority>,
        lease_id: u64,
        claims: Vec<MemoryClaim>,
    ) -> Self {
        let planned = aggregate_claim_rows(&claims);
        Self {
            authority,
            lease_id,
            claims,
            planned,
            committed: HashMap::new(),
            observed: HashMap::new(),
            host: Vec::new(),
            providers: Vec::new(),
            #[cfg(feature = "gpu-device-pool")]
            cuda: Vec::new(),
            #[cfg(feature = "gpu-device-pool")]
            cuda_transient: Vec::new(),
            released: false,
        }
    }

    pub(crate) fn claims(&self) -> &[MemoryClaim] {
        &self.claims
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    pub(crate) fn backend_claim_templates_for_class(
        &self,
        class: MemoryAllocationClass,
    ) -> Vec<MemoryClaim> {
        let mut claims = Vec::new();
        for claim in self.claims.iter().filter(|claim| {
            claim.class == class
                && match &claim.source {
                    MemoryClaimSource::Runtime { allocation_id } => allocation_id.is_some(),
                    MemoryClaimSource::External { .. } => true,
                }
        }) {
            if !claims
                .iter()
                .any(|existing: &MemoryClaim| existing.domain == claim.domain)
            {
                let mut claim = claim.clone();
                claim.bytes = 0;
                claims.push(claim);
            }
        }
        claims
    }

    pub(crate) fn reserved_bytes_for_class(&self, class: MemoryAllocationClass) -> usize {
        self.claims
            .iter()
            .filter(|claim| claim.class == class)
            .map(|claim| claim.bytes)
            .fold(0usize, usize::saturating_add)
    }

    /// Move a request-time reservation into committed accounting while keeping
    /// the same physical authority leases alive.
    pub(crate) fn commit_capacity(&mut self) {
        self.mark_committed();
    }

    /// Atomically reserve additional bytes before the backing allocator may
    /// grow. If any domain rejects the request, temporary reservations are
    /// dropped and the lease remains unchanged.
    pub(crate) fn grow(&mut self, plan: &MemoryPlan) -> Result<(), String> {
        let authority = Arc::clone(&self.authority);
        let _operation = authority.operations.lock();
        self.grow_locked(plan)
    }

    fn grow_locked(&mut self, plan: &MemoryPlan) -> Result<(), String> {
        self.grow_impl_locked(plan, true)
    }

    fn grow_pre_admitted_locked(&mut self, plan: &MemoryPlan) -> Result<(), String> {
        self.grow_impl_locked(plan, false)
    }

    fn grow_impl_locked(
        &mut self,
        plan: &MemoryPlan,
        requires_preflight: bool,
    ) -> Result<(), String> {
        if plan.claims().is_empty() {
            return Ok(());
        }
        self.validate_growth_plan(plan)?;
        if requires_preflight {
            self.authority.preflight_growth_locked(plan)?;
        }

        let mut staged_host = Vec::new();
        let mut staged_providers = Vec::new();
        #[cfg(feature = "gpu-device-pool")]
        let mut staged_cuda = Vec::new();

        for claim in plan.claims().iter().filter(|claim| claim.bytes > 0) {
            match &claim.domain {
                MemoryDomain::Host
                | MemoryDomain::HostPinned { .. }
                | MemoryDomain::HostMapped { .. } => {
                    if let Some(lease) = self.authority.host.admit(claim)? {
                        staged_host.push(lease);
                    }
                }
                MemoryDomain::Provider { .. } => {
                    if let Some(lease) = self.authority.providers.reserve(claim)? {
                        staged_providers.push(lease);
                    }
                }
                MemoryDomain::Cuda { .. } => {
                    #[cfg(feature = "gpu-device-pool")]
                    {
                        let manager = self.authority.cuda.as_ref().ok_or_else(|| {
                            format!(
                                "{} growth for {} has no CUDA memory authority",
                                claim.domain, claim.owner
                            )
                        })?;
                        if let Some(lease) = manager.admit_transient(claim)? {
                            staged_cuda.push(lease);
                        }
                    }
                    #[cfg(not(feature = "gpu-device-pool"))]
                    return Err(format!(
                        "{} growth for {} requires CUDA memory authority",
                        claim.domain, claim.owner
                    ));
                }
            }
        }

        self.host.extend(staged_host);
        self.providers.extend(staged_providers);
        #[cfg(feature = "gpu-device-pool")]
        self.cuda_transient.extend(staged_cuda);
        for claim in plan.claims() {
            self.add_claim_bytes(claim, claim.bytes);
            let planned = self
                .planned
                .entry(MemoryRowKey::from_claim(claim))
                .or_default();
            *planned = planned.saturating_add(claim.bytes);
        }
        self.refresh_accounting(false);
        Ok(())
    }

    /// Return bytes to every backing authority immediately. A shrink never
    /// drops below the last observed physical footprint for the row.
    pub(crate) fn shrink(&mut self, plan: &MemoryPlan) -> Result<(), String> {
        let authority = Arc::clone(&self.authority);
        let _operation = authority.operations.lock();
        self.shrink_locked(plan)
    }

    /// Return externally owned KV bytes after the caller has fenced every
    /// importer and physically released the backing. Unlike a speculative
    /// shrink, this is authoritative evidence that the old observed footprint
    /// no longer exists, so the observation floor may move down with it.
    pub(crate) fn shrink_after_external_release(
        &mut self,
        plan: &MemoryPlan,
    ) -> Result<(), String> {
        let authority = Arc::clone(&self.authority);
        let _operation = authority.operations.lock();
        if plan.claims().iter().any(|claim| {
            claim.class != MemoryAllocationClass::KvCache
                || !matches!(claim.source, MemoryClaimSource::External { .. })
        }) {
            return Err(
                "externally fenced shrink accepts only external KV-cache claims".to_string(),
            );
        }
        self.validate_growth_plan(plan)?;
        let current = aggregate_claim_rows(&self.claims);
        let reductions = aggregate_claim_rows(plan.claims());
        for (key, reduction) in reductions {
            let current_bytes = current.get(&key).copied().unwrap_or(0);
            if reduction > current_bytes {
                return Err(format!(
                    "externally fenced shrink exceeds the live {} allocation for {}",
                    key.class, key.owner
                ));
            }
            let target = current_bytes - reduction;
            self.observed.insert(key.clone(), target);
            self.resize_adapter_row(&key, target)?;
            self.set_claim_row_bytes(&key, target);
            if let Some(committed) = self.committed.get_mut(&key) {
                *committed = (*committed).min(target);
            }
            if let Some(planned) = self.planned.get_mut(&key) {
                *planned = planned.saturating_sub(reduction);
            }
        }
        self.refresh_accounting(false);
        Ok(())
    }

    fn shrink_locked(&mut self, plan: &MemoryPlan) -> Result<(), String> {
        if plan.claims().is_empty() {
            return Ok(());
        }
        self.validate_growth_plan(plan)?;
        let current = aggregate_claim_rows(&self.claims);
        let reductions = aggregate_claim_rows(plan.claims());
        for (key, reduction) in reductions {
            let current_bytes = current.get(&key).copied().unwrap_or(0);
            let observed_floor = self.observed.get(&key).copied().unwrap_or(0);
            let target = current_bytes.saturating_sub(reduction).max(observed_floor);
            self.resize_adapter_row(&key, target)?;
            self.set_claim_row_bytes(&key, target);
            if let Some(committed) = self.committed.get_mut(&key) {
                *committed = (*committed).min(target);
            }
            if let Some(planned) = self.planned.get_mut(&key) {
                *planned = planned.saturating_sub(reduction);
            }
        }
        self.refresh_accounting(false);
        Ok(())
    }

    /// Reconcile the lease to a backend's authoritative cross-domain report.
    /// Domain changes are treated as migrations: the destination is admitted
    /// before the source reservation is released.
    pub(crate) fn reconcile(&mut self, report: &BackendMemoryReport) -> Result<(), String> {
        let authority = Arc::clone(&self.authority);
        let _operation = authority.operations.lock();
        self.reconcile_locked(report)
    }

    fn reconcile_locked(&mut self, report: &BackendMemoryReport) -> Result<(), String> {
        // An empty report is the legacy/decorator signal for "no unified
        // memory telemetry", not proof that a still-loaded backend owns zero
        // bytes. Keep the admitted fallback rows until unload drops the lease.
        // This is especially important for ONNX task wrappers using a shared
        // CUDA allocator: their legacy external report is intentionally empty.
        if report.allocations.is_empty() {
            return Ok(());
        }
        let owner = self.single_owner()?;
        let actual_plan = MemoryPlan::from_backend_report(owner, report);
        let actual = aggregate_claim_rows(actual_plan.claims());

        #[cfg(feature = "gpu-device-pool")]
        {
            let external_cuda = external_cuda_report_from_backend(report);
            let result = self
                .cuda
                .iter_mut()
                .try_for_each(|lease| lease.reconcile_report(&external_cuda));
            if let Err(error) = result {
                // The backend already owns these bytes. Preserve that physical
                // observation even when the authority cannot raise its
                // reservation, so pressure and every subsequent admission see
                // the overage instead of stale planned values.
                self.observed = actual;
                self.refresh_accounting(false);
                return Err(error);
            }
        }

        let current = aggregate_claim_rows(&self.claims);

        let mut growth = MemoryPlan::new();
        for (key, actual_bytes) in &actual {
            let current_bytes = current.get(key).copied().unwrap_or(0);
            if *actual_bytes > current_bytes {
                let mut claim = actual_plan
                    .claims()
                    .iter()
                    .find(|claim| MemoryRowKey::from_claim(claim) == *key)
                    .cloned()
                    .expect("aggregated backend row has a claim template");
                claim.bytes = actual_bytes - current_bytes;
                growth.push(claim);
            }
        }
        // These bytes already exist in the backend and are already represented
        // by the observed snapshot. Reserve through each physical adapter
        // without charging the central preflight a second time. Host, CUDA,
        // and provider adapters still enforce their hard domain/class limits.
        if let Err(error) = self.grow_pre_admitted_locked(&growth) {
            // Reconciliation is observational after the backend has allocated.
            // A rejected reservation must not erase evidence of the physical
            // footprint; retaining it drives pressure and closes admission.
            self.observed = actual;
            self.refresh_accounting(false);
            return Err(error);
        }

        // Publish the new observed floors before shrinking so a concurrent or
        // stale report can never release authority below known physical use.
        self.observed = actual.clone();
        let after_growth = aggregate_claim_rows(&self.claims);
        let mut shrink = MemoryPlan::new();
        for (key, current_bytes) in after_growth {
            let target = actual.get(&key).copied().unwrap_or(0);
            if current_bytes > target {
                shrink.push(MemoryClaim::runtime(
                    key.domain.clone(),
                    key.owner,
                    key.class,
                    current_bytes - target,
                ));
            }
        }
        self.shrink_locked(&shrink)?;
        self.committed = actual;
        self.refresh_accounting(false);
        Ok(())
    }

    fn validate_growth_plan(&self, plan: &MemoryPlan) -> Result<(), String> {
        let owners: Vec<_> = self.claims.iter().map(|claim| claim.owner).collect();
        for claim in plan.claims() {
            if !owners.is_empty() && !owners.contains(&claim.owner) {
                return Err(format!(
                    "cannot resize lease for {} with a claim owned by {}",
                    owners[0], claim.owner
                ));
            }
        }
        Ok(())
    }

    fn single_owner(&self) -> Result<MemoryOwner, String> {
        let Some(owner) = self.claims.first().map(|claim| claim.owner) else {
            return Err("cannot reconcile an ownerless memory lease".to_string());
        };
        if self.claims.iter().any(|claim| claim.owner != owner) {
            return Err("cannot reconcile a memory lease containing multiple owners".to_string());
        }
        Ok(owner)
    }

    fn add_claim_bytes(&mut self, claim: &MemoryClaim, bytes: usize) {
        let key = MemoryRowKey::from_claim(claim);
        if let Some(existing) = self
            .claims
            .iter_mut()
            .find(|existing| MemoryRowKey::from_claim(existing) == key)
        {
            existing.bytes = existing.bytes.saturating_add(bytes);
        } else {
            let mut claim = claim.clone();
            claim.bytes = bytes;
            self.claims.push(claim);
        }
    }

    fn ensure_claim_template(&mut self, template: &MemoryClaim) {
        let key = MemoryRowKey::from_claim(template);
        if self
            .claims
            .iter()
            .all(|claim| MemoryRowKey::from_claim(claim) != key)
        {
            let mut template = template.clone();
            template.bytes = 0;
            self.claims.push(template);
        }
    }

    fn set_claim_row_bytes(&mut self, key: &MemoryRowKey, target: usize) {
        let mut assigned = false;
        for claim in &mut self.claims {
            if MemoryRowKey::from_claim(claim) != *key {
                continue;
            }
            if assigned {
                claim.bytes = 0;
            } else {
                claim.bytes = target;
                assigned = true;
            }
        }
        if !assigned && target > 0 {
            self.claims.push(MemoryClaim::runtime(
                key.domain.clone(),
                key.owner,
                key.class,
                target,
            ));
        }
    }

    fn resize_adapter_row(&mut self, key: &MemoryRowKey, target: usize) -> Result<(), String> {
        let probe = MemoryClaim::runtime(key.domain.clone(), key.owner, key.class, target);
        match key.domain {
            MemoryDomain::Host
            | MemoryDomain::HostPinned { .. }
            | MemoryDomain::HostMapped { .. } => {
                resize_host_leases(&mut self.host, &probe, target)?;
            }
            MemoryDomain::Provider { .. } => {
                resize_provider_leases(&mut self.providers, &probe, target)?;
            }
            MemoryDomain::Cuda { .. } => {
                #[cfg(feature = "gpu-device-pool")]
                resize_cuda_transient_leases(&mut self.cuda_transient, &probe, target)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "gpu-device-pool")]
    fn add_unbacked_claims(&mut self, claims: &[MemoryClaim]) -> Result<(), String> {
        let authority = Arc::clone(&self.authority);
        let _operation = authority.operations.lock();
        let plan = MemoryPlan {
            claims: claims.to_vec(),
        };
        self.validate_growth_plan(&plan)?;
        self.authority.preflight_growth_locked(&plan)?;
        for claim in claims {
            self.add_claim_bytes(claim, claim.bytes);
            let planned = self
                .planned
                .entry(MemoryRowKey::from_claim(claim))
                .or_default();
            *planned = planned.saturating_add(claim.bytes);
        }
        self.refresh_accounting(false);
        Ok(())
    }

    fn mark_committed(&mut self) {
        let authority = Arc::clone(&self.authority);
        let _operation = authority.operations.lock();
        self.committed = aggregate_claim_rows(&self.claims);
        for (key, bytes) in &self.committed {
            self.observed.entry(key.clone()).or_insert(*bytes);
        }
        self.refresh_accounting(true);
    }

    fn record_loaded_report(&mut self, report: &BackendMemoryReport) -> Result<(), String> {
        let authority = Arc::clone(&self.authority);
        let _operation = authority.operations.lock();
        self.record_loaded_report_locked(report)
    }

    fn record_loaded_report_locked(&mut self, report: &BackendMemoryReport) -> Result<(), String> {
        if report.allocations.is_empty() {
            return Ok(());
        }
        let owner = self.single_owner()?;
        let actual_plan = MemoryPlan::from_backend_report(owner, report);
        let actual = aggregate_claim_rows(actual_plan.claims());
        for (key, target) in &actual {
            let probe = MemoryClaim::runtime(key.domain.clone(), key.owner, key.class, *target);
            match key.domain {
                MemoryDomain::Host
                | MemoryDomain::HostPinned { .. }
                | MemoryDomain::HostMapped { .. } => {
                    let reserved = self
                        .host
                        .iter()
                        .filter(|lease| lease.matches(&probe))
                        .map(HostMemoryLease::bytes)
                        .fold(0usize, usize::saturating_add);
                    if *target > reserved {
                        let mut delta = probe.clone();
                        delta.bytes = target - reserved;
                        if let Some(lease) = self.authority.host.admit(&delta)? {
                            self.host.push(lease);
                        }
                    }
                }
                MemoryDomain::Provider { .. } => {
                    let reserved = self
                        .providers
                        .iter()
                        .filter(|lease| lease.matches(&probe))
                        .map(ProviderMemoryLease::bytes)
                        .fold(0usize, usize::saturating_add);
                    if *target > reserved {
                        let mut delta = probe;
                        delta.bytes = target - reserved;
                        if let Some(lease) = self.authority.providers.reserve(&delta)? {
                            self.providers.push(lease);
                        }
                    }
                }
                MemoryDomain::Cuda { .. } => {}
            }
        }
        for template in actual_plan.claims() {
            self.ensure_claim_template(template);
        }
        let existing: Vec<_> = aggregate_claim_rows(&self.claims).into_keys().collect();
        for key in existing {
            self.set_claim_row_bytes(&key, actual.get(&key).copied().unwrap_or(0));
        }
        for (key, bytes) in &actual {
            self.set_claim_row_bytes(key, *bytes);
        }

        // Host load reconciliation may have conservatively raised one row to
        // the measured RSS delta. Retain that physical reservation in the
        // unified snapshot even when a backend under-reports host allocations.
        let host_rows = self
            .host
            .iter()
            .map(HostMemoryLease::claim)
            .collect::<Vec<_>>();
        for (key, bytes) in aggregate_claim_rows(&host_rows) {
            let reported = aggregate_claim_rows(&self.claims)
                .get(&key)
                .copied()
                .unwrap_or(0);
            self.set_claim_row_bytes(&key, reported.max(bytes));
        }
        self.observed = actual;
        self.committed = aggregate_claim_rows(&self.claims);
        self.refresh_accounting(false);
        Ok(())
    }

    #[cfg(any(test, feature = "gpu-device-pool"))]
    fn reconcile_pre_admitted_locked(
        &mut self,
        report: &BackendMemoryReport,
        target_planned: HashMap<MemoryRowKey, usize>,
    ) -> Result<(), String> {
        #[cfg(feature = "gpu-device-pool")]
        {
            let external_cuda = external_cuda_report_from_backend(report);
            for lease in &mut self.cuda {
                lease.reconcile_report(&external_cuda)?;
            }
        }

        let owner = self.single_owner()?;
        let actual_plan = MemoryPlan::from_backend_report(owner, report);
        let actual = aggregate_claim_rows(actual_plan.claims());
        let current = aggregate_claim_rows(&self.claims);

        // Activation has already freed the source allocation. Release rows
        // that contracted or migrated before attaching the pre-admitted target
        // to this persistent lease; `operations` hides the transfer from
        // snapshots and competing admissions.
        for (key, current_bytes) in &current {
            let target = actual.get(key).copied().unwrap_or(0);
            if *current_bytes > target {
                self.resize_adapter_row(key, target)?;
                self.set_claim_row_bytes(key, target);
                if let Some(committed) = self.committed.get_mut(key) {
                    *committed = (*committed).min(target);
                }
                if let Some(observed) = self.observed.get_mut(key) {
                    *observed = (*observed).min(target);
                }
            }
        }
        self.refresh_accounting(false);

        let after_shrink = aggregate_claim_rows(&self.claims);
        let mut non_cuda_growth = MemoryPlan::new();
        let mut cuda_growth = Vec::new();
        for (key, target) in &actual {
            let current_bytes = after_shrink.get(key).copied().unwrap_or(0);
            if *target <= current_bytes {
                continue;
            }
            let mut claim = actual_plan
                .claims()
                .iter()
                .find(|claim| MemoryRowKey::from_claim(claim) == *key)
                .cloned()
                .expect("aggregated backend row has a claim template");
            claim.bytes = target - current_bytes;
            if matches!(key.domain, MemoryDomain::Cuda { .. }) {
                cuda_growth.push(claim);
            } else {
                non_cuda_growth.push(claim);
            }
        }
        self.grow_pre_admitted_locked(&non_cuda_growth)?;
        for claim in cuda_growth {
            self.add_claim_bytes(&claim, claim.bytes);
        }

        let existing: Vec<_> = aggregate_claim_rows(&self.claims).into_keys().collect();
        for key in existing {
            self.set_claim_row_bytes(&key, actual.get(&key).copied().unwrap_or(0));
        }
        for (key, bytes) in &actual {
            self.set_claim_row_bytes(key, *bytes);
        }
        self.planned = target_planned;
        self.observed = actual.clone();
        self.committed = actual;
        self.refresh_accounting(false);
        Ok(())
    }

    #[cfg(any(test, feature = "gpu-device-pool"))]
    fn release_locked(&mut self) -> Result<(), String> {
        if self.released {
            return Ok(());
        }
        self.observed.clear();
        let release = MemoryPlan {
            claims: self.claims.clone(),
        };
        self.shrink_locked(&release)?;
        self.authority.unregister_lease_locked(self.lease_id);
        self.released = true;
        Ok(())
    }

    fn refresh_accounting(&self, mark_committed: bool) {
        self.authority.replace_lease_rows(
            self.lease_id,
            &self.planned,
            &self.claims,
            &self.committed,
            &self.observed,
            mark_committed,
        );
    }
}

impl Drop for MemoryLease {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.observed.clear();
        let release = MemoryPlan {
            claims: self.claims.clone(),
        };
        if let Err(error) = self.shrink(&release) {
            log::warn!(
                "[memory-authority] lease {} release reconciliation failed: {}",
                self.lease_id,
                error
            );
        }
        self.authority.unregister_lease(self.lease_id);
    }
}

fn aggregate_claim_rows(claims: &[MemoryClaim]) -> HashMap<MemoryRowKey, usize> {
    let mut rows = HashMap::new();
    for claim in claims {
        let bytes = rows
            .entry(MemoryRowKey::from_claim(claim))
            .or_insert(0usize);
        *bytes = bytes.saturating_add(claim.bytes);
    }
    rows
}

fn resize_host_leases(
    leases: &mut Vec<HostMemoryLease>,
    claim: &MemoryClaim,
    target: usize,
) -> Result<(), String> {
    let mut current = leases
        .iter()
        .filter(|lease| lease.matches(claim))
        .map(HostMemoryLease::bytes)
        .fold(0usize, usize::saturating_add);
    if target > current {
        return Err(format!(
            "host lease for {} class={} is missing {} admitted bytes",
            claim.owner,
            claim.class,
            target - current
        ));
    }
    let mut index = leases.len();
    while current > target && index > 0 {
        index -= 1;
        if !leases[index].matches(claim) {
            continue;
        }
        let release = (current - target).min(leases[index].bytes());
        let retained = leases[index].bytes() - release;
        leases[index].resize(retained)?;
        current -= release;
        if retained == 0 {
            leases.swap_remove(index);
        }
    }
    Ok(())
}

fn resize_provider_leases(
    leases: &mut Vec<ProviderMemoryLease>,
    claim: &MemoryClaim,
    target: usize,
) -> Result<(), String> {
    let mut matching: Vec<_> = leases
        .iter()
        .enumerate()
        .filter(|(_, lease)| lease.matches(claim))
        .map(|(index, lease)| (index, lease.bytes()))
        .collect();
    let mut current = matching
        .iter()
        .map(|(_, bytes)| *bytes)
        .fold(0usize, usize::saturating_add);
    while current > target {
        let Some((index, bytes)) = matching.pop() else {
            break;
        };
        let release = (current - target).min(bytes);
        let retained = bytes - release;
        leases[index].resize(retained)?;
        current -= release;
    }
    leases.retain(|lease| lease.bytes() != 0);
    if current != target {
        return Err(format!(
            "provider lease for {} class={} could not resize to {} bytes",
            claim.owner, claim.class, target
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu-device-pool")]
fn resize_cuda_transient_leases(
    leases: &mut Vec<DeviceMemoryTransientLease>,
    claim: &MemoryClaim,
    target: usize,
) -> Result<(), String> {
    let mut matching: Vec<_> = leases
        .iter()
        .enumerate()
        .filter(|(_, lease)| lease.matches(claim))
        .map(|(index, lease)| (index, lease.bytes()))
        .collect();
    let mut current = matching
        .iter()
        .map(|(_, bytes)| *bytes)
        .fold(0usize, usize::saturating_add);
    while current > target {
        let Some((index, bytes)) = matching.pop() else {
            break;
        };
        let release = (current - target).min(bytes);
        let retained = bytes - release;
        leases[index].resize(retained)?;
        current -= release;
    }
    leases.retain(|lease| lease.bytes() != 0);
    if current != target && !matches!(claim.source, MemoryClaimSource::Runtime { .. }) {
        return Err(format!(
            "CUDA lease for {} class={} could not resize to {} bytes",
            claim.owner, claim.class, target
        ));
    }
    Ok(())
}

fn reconcile_backend_claim_bytes(claims: &mut [MemoryClaim], report: &BackendMemoryReport) {
    for claim in claims {
        let expected_source = match &claim.source {
            MemoryClaimSource::Runtime { .. } => BackendMemoryAllocationSource::RuntimeManaged,
            MemoryClaimSource::External { .. } => BackendMemoryAllocationSource::BackendManaged,
        };
        let expected_id = match &claim.source {
            MemoryClaimSource::Runtime { allocation_id } => allocation_id.as_deref(),
            MemoryClaimSource::External { allocation_id } => Some(allocation_id.as_str()),
        };
        let mut matched = false;
        let mut actual_bytes = 0usize;
        for allocation in &report.allocations {
            if MemoryDomain::from_backend(&allocation.domain) != claim.domain
                || MemoryAllocationClass::from_backend(allocation.class) != claim.class
                || allocation.source != expected_source
                || expected_id.is_some_and(|id| id != allocation.allocation_id.as_str())
            {
                continue;
            }
            matched = true;
            actual_bytes = actual_bytes.saturating_add(allocation.bytes);
        }
        if matched {
            claim.bytes = actual_bytes;
        }
    }
}

/// Short-lived lease for the second copy of externally-owned weights during
/// activation. It also retains the per-device CUDA serialization guards.
#[cfg(feature = "gpu-device-pool")]
#[must_use = "hold the swap lease until backend activation finishes"]
pub(crate) struct MemorySwapLease {
    _lease: MemoryLease,
    _cuda: Vec<DeviceMemorySwapAdmission>,
}

#[cfg(feature = "gpu-device-pool")]
impl MemorySwapLease {
    /// Atomically transfer the already-admitted target footprint onto the
    /// persistent model lease after activation. The temporary peak remains
    /// owned until the persistent rows and long-lived CUDA admission match the
    /// backend's actual report.
    pub(crate) fn finish(
        mut self,
        persistent: &mut MemoryLease,
        report: &BackendMemoryReport,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self._lease.authority, &persistent.authority) {
            return Err("cannot transfer a swap lease between memory authorities".to_string());
        }
        let authority = Arc::clone(&persistent.authority);
        let _operation = authority.operations.lock();
        let target_planned = self._lease.planned.clone();
        self._lease.release_locked()?;
        persistent.reconcile_pre_admitted_locked(report, target_planned)?;
        self._cuda.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::host_memory::HostMemoryBudget;
    use kapsl_hal::device::{Device, DeviceBackend};

    const MIB: usize = 1024 * 1024;
    const GIB: usize = 1024 * 1024 * 1024;

    fn cpu_device_info() -> DeviceInfo {
        DeviceInfo {
            cpu_cores: 1,
            total_memory: 10 * 1024 * 1024,
            os_type: "test".to_string(),
            os_release: "test".to_string(),
            has_cuda: false,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: vec![Device {
                id: 0,
                name: "test-cpu".to_string(),
                backend: DeviceBackend::Cpu,
                memory_mb: 0,
                compute_units: 1,
                pci_bus_id: None,
                partition_id: None,
                driver_version: None,
                cuda_version: None,
                compute_capability: None,
                utilization_gpu_pct: None,
                temperature_c: None,
                supports_fp16: false,
                supports_int8: true,
            }],
        }
    }

    fn backend_report(
        allocation_id: &str,
        domain: BackendMemoryDomain,
        class: BackendMemoryAllocationClass,
        bytes: usize,
    ) -> BackendMemoryReport {
        BackendMemoryReport {
            allocations: vec![kapsl_engine_api::MemoryAllocation {
                allocation_id: allocation_id.to_string(),
                domain,
                class,
                source: BackendMemoryAllocationSource::BackendManaged,
                bytes,
            }],
        }
    }

    fn authority_with_provider_budget(domain: MemoryDomain, budget: usize) -> Arc<MemoryAuthority> {
        let host = HostMemoryManager::new(&cpu_device_info());
        let domain_budgets = HashMap::from([
            (MemoryDomain::Host, host.budget().safe_bytes),
            (domain.clone(), budget),
        ]);
        Arc::new(MemoryAuthority {
            host,
            #[cfg(feature = "gpu-device-pool")]
            cuda: None,
            providers: Arc::new(ProviderMemoryLedger::with_budgets(HashMap::from([(
                domain, budget,
            )]))),
            operations: Mutex::new(()),
            accounting: Mutex::new(AuthorityAccounting::default()),
            next_lease_id: AtomicU64::new(1),
            device_domains: HashMap::new(),
            domain_budgets,
            live_ceiling: Mutex::new(HashMap::new()),
        })
    }

    #[test]
    fn first_fitting_admission_selects_and_reserves_under_one_snapshot() {
        let domain = MemoryDomain::Provider {
            provider: "metal".to_string(),
            device_id: Some(0),
        };
        let authority = authority_with_provider_budget(domain.clone(), 100);
        let owner = MemoryOwner::new(71, 2);
        let candidate = |bytes: usize| {
            let mut plan = MemoryPlan::new();
            plan.push(MemoryClaim::external(
                domain.clone(),
                owner,
                MemoryAllocationClass::KvCache,
                "managed-vllm-provisional",
                bytes,
            ));
            plan
        };
        let candidates = vec![candidate(120), candidate(80), candidate(40)];

        let (mut lease, selected, decision) = authority
            .admit_first_fitting(&candidates)
            .expect("the second whole candidate should fit");
        assert_eq!(selected, 1);
        assert_eq!(decision.domain(&domain).unwrap().available_bytes, 100);
        assert_eq!(
            authority
                .snapshot()
                .domain(&domain)
                .unwrap()
                .available_bytes,
            20
        );
        assert_eq!(
            lease.reserved_bytes_for_class(MemoryAllocationClass::KvCache),
            80
        );

        lease.commit_capacity();
        drop(lease);
        assert_eq!(
            authority
                .snapshot()
                .domain(&domain)
                .unwrap()
                .available_bytes,
            100
        );
    }

    #[test]
    fn first_fitting_admission_rejects_all_without_leaking_a_lease() {
        let domain = MemoryDomain::Provider {
            provider: "directml".to_string(),
            device_id: Some(4),
        };
        let authority = authority_with_provider_budget(domain.clone(), 64);
        let owner = MemoryOwner::new(72, 0);
        let candidates = [80usize, 65].map(|bytes| {
            let mut plan = MemoryPlan::new();
            plan.push(MemoryClaim::external(
                domain.clone(),
                owner,
                MemoryAllocationClass::KvCache,
                "managed-vllm-provisional",
                bytes,
            ));
            plan
        });

        let error = match authority.admit_first_fitting(&candidates) {
            Ok(_) => panic!("every candidate exceeds the hard budget"),
            Err(error) => error,
        };
        assert!(error.contains("rejected every candidate"));
        assert!(authority.snapshot().rows.is_empty());
        assert_eq!(
            authority
                .snapshot()
                .domain(&domain)
                .unwrap()
                .available_bytes,
            64
        );
    }

    #[test]
    fn first_fitting_admission_rejects_empty_and_zero_candidates() {
        let domain = MemoryDomain::Provider {
            provider: "metal".to_string(),
            device_id: Some(0),
        };
        let authority = authority_with_provider_budget(domain.clone(), 64);
        let owner = MemoryOwner::new(73, 0);
        let mut zero = MemoryPlan::new();
        zero.push(MemoryClaim::external(
            domain,
            owner,
            MemoryAllocationClass::KvCache,
            "managed-vllm-provisional",
            0,
        ));

        assert!(authority.admit_first_fitting(&[]).is_err());
        let error = match authority.admit_first_fitting(&[MemoryPlan::new(), zero]) {
            Ok(_) => panic!("zero-byte candidates cannot satisfy a hard minimum"),
            Err(error) => error,
        };
        assert!(error.contains("no positive-byte claims"));
        assert!(authority.snapshot().rows.is_empty());
    }

    #[test]
    fn concurrent_first_fitting_admissions_have_one_winner() {
        let domain = MemoryDomain::Provider {
            provider: "metal".to_string(),
            device_id: Some(0),
        };
        let authority = authority_with_provider_budget(domain.clone(), 100);
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut workers = Vec::new();
        for replica_id in 0..2 {
            let authority = authority.clone();
            let domain = domain.clone();
            let barrier = barrier.clone();
            workers.push(std::thread::spawn(move || {
                let mut plan = MemoryPlan::new();
                plan.push(MemoryClaim::external(
                    domain,
                    MemoryOwner::new(74, replica_id),
                    MemoryAllocationClass::KvCache,
                    "managed-vllm-provisional",
                    80,
                ));
                barrier.wait();
                let admission = authority.admit_first_fitting(&[plan]);
                barrier.wait();
                admission.is_ok()
            }));
        }
        barrier.wait();
        barrier.wait();
        let winners = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .filter(|won| *won)
            .count();
        assert_eq!(winners, 1);
        assert_eq!(
            authority
                .snapshot()
                .domain(&domain)
                .unwrap()
                .available_bytes,
            100
        );
    }

    #[test]
    fn provider_limits_parse_aliases_and_device_overrides() {
        let limits = ProviderMemoryLimits::parse("coreml=8g,directml=6g,dml:1=4g").unwrap();
        assert_eq!(
            limits.limit_for_domain(&MemoryDomain::Provider {
                provider: "metal".to_string(),
                device_id: Some(0),
            }),
            Some(8 * GIB)
        );
        assert_eq!(
            limits.limit_for_domain(&MemoryDomain::Provider {
                provider: "directml".to_string(),
                device_id: Some(1),
            }),
            Some(4 * GIB)
        );
        assert_eq!(
            limits.limit_for_domain(&MemoryDomain::Provider {
                provider: "directml".to_string(),
                device_id: Some(2),
            }),
            Some(6 * GIB)
        );
        assert!(ProviderMemoryLimits::parse("cuda=8g").is_err());
        assert!(ProviderMemoryLimits::parse("metal=8g,coreml=4g").is_err());
        assert!(ProviderMemoryLimits::parse("metal=unbounded").is_err());
    }

    #[test]
    fn provider_limits_clamp_probed_domains_and_register_unprobed_adapters() {
        let mut info = cpu_device_info();
        info.devices.push(Device {
            id: 2,
            name: "test-metal".to_string(),
            backend: DeviceBackend::Metal,
            memory_mb: 16,
            compute_units: 1,
            pci_bus_id: None,
            partition_id: None,
            driver_version: None,
            cuda_version: None,
            compute_capability: None,
            utilization_gpu_pct: None,
            temperature_c: None,
            supports_fp16: true,
            supports_int8: true,
        });
        let limits = ProviderMemoryLimits::parse("metal=8m,directml:4=6m").unwrap();
        let (_, domains) = authority_domain_policies(
            &info,
            HostMemoryBudget {
                limit_bytes: 10 * MIB,
                safe_bytes: 8 * MIB,
            },
            &limits,
        );
        assert_eq!(
            domains[&MemoryDomain::Provider {
                provider: "metal".to_string(),
                device_id: Some(2),
            }],
            8 * MIB
        );
        assert_eq!(
            domains[&MemoryDomain::Provider {
                provider: "directml".to_string(),
                device_id: Some(4),
            }],
            6 * MIB
        );
    }

    #[test]
    fn provider_ledger_enforces_limits_and_counts_elastic_deltas_additively() {
        let domain = MemoryDomain::Provider {
            provider: "metal".to_string(),
            device_id: Some(0),
        };
        let ledger = Arc::new(ProviderMemoryLedger::with_budgets(HashMap::from([(
            domain.clone(),
            100,
        )])));
        let owner = MemoryOwner::new(7, 0);
        let claim = |allocation: &str, bytes| {
            MemoryClaim::external(
                domain.clone(),
                owner,
                MemoryAllocationClass::KvCache,
                allocation,
                bytes,
            )
        };
        let first = ledger.reserve(&claim("arena", 60)).unwrap().unwrap();
        let mut delta = ledger.reserve(&claim("arena", 30)).unwrap().unwrap();
        assert!(ledger.reserve(&claim("other", 20)).is_err());

        delta.resize(10).unwrap();
        let other = ledger.reserve(&claim("other", 20)).unwrap().unwrap();
        assert_eq!(ledger.allocation_count(), 2);

        drop(other);
        drop(delta);
        drop(first);
        assert_eq!(ledger.allocation_count(), 0);
    }

    #[test]
    fn provider_reconciliation_enforces_cap_and_retains_overage_observation() {
        let domain = MemoryDomain::Provider {
            provider: "directml".to_string(),
            device_id: Some(0),
        };
        let authority = authority_with_provider_budget(domain.clone(), 100);
        let owner = MemoryOwner::new(8, 0);
        let mut plan = MemoryPlan::new();
        plan.push(MemoryClaim::external(
            domain,
            owner,
            MemoryAllocationClass::ModelSession,
            "directml:arena",
            40,
        ));
        let mut lease = authority.admit(&plan).unwrap();
        lease.mark_committed();

        let report = |bytes| {
            backend_report(
                "directml:arena",
                BackendMemoryDomain::Provider {
                    provider: "directml".to_string(),
                    device_id: Some(0),
                },
                BackendMemoryAllocationClass::ModelSession,
                bytes,
            )
        };
        assert!(lease.reconcile(&report(120)).is_err());
        let snapshot = authority.snapshot();
        let row = snapshot.rows.iter().find(|row| row.owner == owner).unwrap();
        assert_eq!(row.reserved_bytes, 40);
        assert_eq!(row.observed_bytes, 120);
        let domain = snapshot
            .domain(&MemoryDomain::Provider {
                provider: "directml".to_string(),
                device_id: Some(0),
            })
            .unwrap();
        assert_eq!(domain.available_bytes, 0);

        lease.reconcile(&report(80)).unwrap();
        let row = authority
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner == owner)
            .unwrap();
        assert_eq!(row.reserved_bytes, 80);
        assert_eq!(row.committed_bytes, 80);
        assert_eq!(row.observed_bytes, 80);
    }

    #[test]
    fn rejected_continuous_reconciliation_still_publishes_physical_observation() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(11, 0);
        let mut plan = MemoryPlan::new();
        plan.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::ModelSession,
            MIB,
        ));
        let mut lease = authority.admit(&plan).unwrap();
        lease.mark_committed();

        let over_limit = backend_report(
            "session",
            BackendMemoryDomain::Host,
            BackendMemoryAllocationClass::ModelSession,
            5 * GIB,
        );
        assert!(lease.reconcile(&over_limit).is_err());
        let row = authority
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner == owner)
            .unwrap();
        assert_eq!(row.reserved_bytes, MIB);
        assert_eq!(row.committed_bytes, MIB);
        assert_eq!(row.observed_bytes, 5 * GIB);

        // Once usage returns inside the class limit, the same lease recovers
        // and its reservation catches up to the current physical report.
        let recovered = backend_report(
            "session",
            BackendMemoryDomain::Host,
            BackendMemoryAllocationClass::ModelSession,
            2 * GIB,
        );
        lease.reconcile(&recovered).unwrap();
        let row = authority
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner == owner)
            .unwrap();
        assert_eq!(row.reserved_bytes, 2 * GIB);
        assert_eq!(row.committed_bytes, 2 * GIB);
        assert_eq!(row.observed_bytes, 2 * GIB);
    }

    #[test]
    fn provider_domains_participate_in_device_pressure() {
        let snapshot = MemorySnapshot {
            domains: vec![MemoryDomainSnapshot {
                domain: MemoryDomain::Provider {
                    provider: "directml".to_string(),
                    device_id: Some(0),
                },
                budget_bytes: 100,
                planned_bytes: 90,
                reserved_bytes: 90,
                committed_bytes: 90,
                observed_bytes: 90,
                available_bytes: 10,
            }],
            ..MemorySnapshot::default()
        };
        assert_eq!(snapshot.max_device_ratio(), Some(0.9));
    }

    #[test]
    fn host_plan_preserves_owner_and_allocation_classes() {
        let owner = MemoryOwner::new(7, 3);
        let mut plan = MemoryPlan::new();
        plan.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::ModelSession,
            2 * GIB,
        ));
        plan.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::TransientWorkspace,
            GIB,
        ));
        assert_eq!(plan.claims().len(), 2);
        assert!(plan.claims().iter().all(|claim| claim.owner == owner));
        assert_eq!(plan.claims()[0].class, MemoryAllocationClass::ModelSession);
        assert_eq!(
            plan.claims()[1].class,
            MemoryAllocationClass::TransientWorkspace
        );
    }

    #[test]
    fn external_report_maps_domains_and_classes() {
        let owner = MemoryOwner::new(9, 2);
        let plan = MemoryPlan::external_cuda_report(
            owner,
            &ExternalDeviceMemoryReport {
                allocations: vec![
                    ExternalDeviceMemory {
                        allocation_id: "weights:model".to_string(),
                        device_id: 4,
                        bytes: 100,
                    },
                    ExternalDeviceMemory {
                        allocation_id: "weights:model:scratch".to_string(),
                        device_id: 4,
                        bytes: 20,
                    },
                ],
            },
        );
        assert_eq!(plan.cuda_device_ids(), vec![4]);
        assert_eq!(
            plan.claims()[0].class,
            MemoryAllocationClass::PersistentWeights
        );
        assert_eq!(
            plan.claims()[1].class,
            MemoryAllocationClass::TransientWorkspace
        );
        assert_eq!(owner_for_device(&plan, 4).unwrap(), owner);
    }

    #[test]
    fn backend_report_preserves_domains_sources_and_stable_ids() {
        let owner = MemoryOwner::new(21, 6);
        let report = BackendMemoryReport {
            allocations: vec![
                kapsl_engine_api::MemoryAllocation {
                    allocation_id: "pinned-input".to_string(),
                    domain: BackendMemoryDomain::HostPinned {
                        provider: "cuda".to_string(),
                        device_id: Some(3),
                    },
                    class: BackendMemoryAllocationClass::RequestTransient,
                    source: BackendMemoryAllocationSource::BackendManaged,
                    bytes: 64,
                },
                kapsl_engine_api::MemoryAllocation {
                    allocation_id: "pooled-weights".to_string(),
                    domain: BackendMemoryDomain::Cuda { device_id: 3 },
                    class: BackendMemoryAllocationClass::PersistentWeights,
                    source: BackendMemoryAllocationSource::RuntimeManaged,
                    bytes: 128,
                },
            ],
        };
        let plan = MemoryPlan::from_backend_report(owner, &report);
        assert_eq!(plan.claims().len(), 2);
        assert_eq!(
            plan.claims()[0].domain,
            MemoryDomain::HostPinned {
                provider: "cuda".to_string(),
                device_id: Some(3),
            }
        );
        assert_eq!(
            plan.claims()[0].source,
            MemoryClaimSource::External {
                allocation_id: "pinned-input".to_string(),
            }
        );
        assert_eq!(
            plan.claims()[1].source,
            MemoryClaimSource::Runtime {
                allocation_id: Some("pooled-weights".to_string()),
            }
        );
        assert!(plan.claims().iter().all(|claim| claim.owner == owner));
    }

    #[test]
    fn coreml_backend_reports_join_the_selected_metal_domain() {
        let owner = MemoryOwner::new(31, 4);
        let report = BackendMemoryReport {
            allocations: vec![kapsl_engine_api::MemoryAllocation {
                allocation_id: "coreml-weights".to_string(),
                domain: BackendMemoryDomain::Provider {
                    provider: "coreml".to_string(),
                    device_id: Some(2),
                },
                class: BackendMemoryAllocationClass::PersistentWeights,
                source: BackendMemoryAllocationSource::BackendManaged,
                bytes: 512,
            }],
        };
        let plan = MemoryPlan::from_backend_report(owner, &report);
        assert_eq!(
            plan.claims()[0].domain,
            MemoryDomain::Provider {
                provider: "metal".to_string(),
                device_id: Some(2),
            }
        );
    }

    #[test]
    fn request_report_retains_every_host_domain_and_provider_adapter() {
        let owner = MemoryOwner::new(22, 1);
        let report = BackendMemoryReport {
            allocations: vec![
                kapsl_engine_api::MemoryAllocation {
                    allocation_id: "host".to_string(),
                    domain: BackendMemoryDomain::Host,
                    class: BackendMemoryAllocationClass::RequestTransient,
                    source: BackendMemoryAllocationSource::BackendManaged,
                    bytes: 10,
                },
                kapsl_engine_api::MemoryAllocation {
                    allocation_id: "mapped".to_string(),
                    domain: BackendMemoryDomain::HostMapped {
                        provider: "cuda".to_string(),
                        device_id: Some(0),
                    },
                    class: BackendMemoryAllocationClass::RequestTransient,
                    source: BackendMemoryAllocationSource::BackendManaged,
                    bytes: 20,
                },
                kapsl_engine_api::MemoryAllocation {
                    allocation_id: "provider".to_string(),
                    domain: BackendMemoryDomain::Provider {
                        provider: "metal".to_string(),
                        device_id: Some(0),
                    },
                    class: BackendMemoryAllocationClass::RequestTransient,
                    source: BackendMemoryAllocationSource::BackendManaged,
                    bytes: 30,
                },
            ],
        };
        let plan = MemoryPlan::request_from_backend_report(owner, &report);
        assert_eq!(plan.claims().len(), 3);
        assert!(plan
            .claims()
            .iter()
            .all(|claim| claim.class == MemoryAllocationClass::RequestTransient));
    }

    #[test]
    fn host_lease_releases_all_claims_atomically() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(11, 5);
        let plan = authority
            .model_load_plan(
                &[MemoryDomain::Host],
                owner,
                2 * GIB,
                GIB,
                &ExternalDeviceMemoryReport::default(),
            )
            .unwrap();
        let lease = authority.admit(&plan).unwrap();
        assert_eq!(authority.host.reserved_bytes(), 3 * GIB);
        assert_eq!(lease.claims().len(), 3);
        drop(lease);
        assert_eq!(authority.host.reserved_bytes(), 0);
    }

    #[test]
    fn model_plan_classifies_every_domain() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(13, 2);
        let cuda = MemoryDomain::Cuda { device_id: 0 };
        let provider = MemoryDomain::Provider {
            provider: "metal".to_string(),
            device_id: Some(0),
        };
        let plan = authority
            .model_load_plan(
                &[MemoryDomain::Host, cuda.clone(), provider.clone()],
                owner,
                100,
                20,
                &ExternalDeviceMemoryReport::default(),
            )
            .unwrap();

        let classes_for = |domain: &MemoryDomain| {
            plan.claims()
                .iter()
                .filter(|claim| &claim.domain == domain)
                .map(|claim| (claim.class, claim.bytes))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            classes_for(&MemoryDomain::Host),
            vec![
                (MemoryAllocationClass::ModelSession, 100),
                (MemoryAllocationClass::TransientWorkspace, 20),
                (MemoryAllocationClass::KvCache, 0),
            ]
        );
        for domain in [&cuda, &provider] {
            assert_eq!(
                classes_for(domain),
                vec![
                    (MemoryAllocationClass::PersistentWeights, 100),
                    (MemoryAllocationClass::TransientWorkspace, 20),
                    (MemoryAllocationClass::KvCache, 0),
                ]
            );
        }
        assert!(plan.claims().iter().all(|claim| claim.owner == owner));
    }

    #[test]
    fn compatibility_kv_templates_require_a_backend_allocation_identity() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(13, 3);
        let mut synthetic = MemoryPlan::new();
        synthetic.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::KvCache,
            0,
        ));
        let synthetic_lease = authority.admit(&synthetic).unwrap();
        assert!(synthetic_lease
            .backend_claim_templates_for_class(MemoryAllocationClass::KvCache)
            .is_empty());

        let mut reported = MemoryPlan::new();
        reported.push(MemoryClaim::runtime_allocation(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::KvCache,
            "backend:kv",
            0,
        ));
        let reported_lease = authority.admit(&reported).unwrap();
        assert_eq!(
            reported_lease
                .backend_claim_templates_for_class(MemoryAllocationClass::KvCache)
                .len(),
            1
        );
    }

    #[test]
    fn backend_rows_outside_selected_device_domains_are_ignored() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let report = BackendMemoryReport {
            allocations: vec![kapsl_engine_api::MemoryAllocation {
                allocation_id: "legacy-cuda-estimate".to_string(),
                domain: BackendMemoryDomain::Cuda { device_id: 0 },
                class: BackendMemoryAllocationClass::PersistentWeights,
                source: BackendMemoryAllocationSource::BackendManaged,
                bytes: GIB,
            }],
        };
        let plan = authority
            .model_load_plan_with_report(
                &[MemoryDomain::Host],
                MemoryOwner::new(3, 0),
                100,
                20,
                &report,
            )
            .unwrap();
        assert!(plan
            .claims()
            .iter()
            .all(|claim| !matches!(claim.domain, MemoryDomain::Cuda { .. })));
        assert_eq!(plan.claims().len(), 3);
    }

    #[test]
    fn provider_adapter_reference_counts_stable_allocations() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let plan_for = |replica_id| {
            let mut plan = MemoryPlan::new();
            plan.push(MemoryClaim::external(
                MemoryDomain::Provider {
                    provider: "metal".to_string(),
                    device_id: Some(0),
                },
                MemoryOwner::new(4, replica_id),
                MemoryAllocationClass::PersistentWeights,
                "weights:shared",
                GIB,
            ));
            plan
        };
        let first = authority.admit(&plan_for(1)).unwrap();
        let second = authority.admit(&plan_for(2)).unwrap();
        assert_eq!(authority.providers.allocation_count(), 1);
        drop(first);
        assert_eq!(authority.providers.allocation_count(), 1);
        drop(second);
        assert_eq!(authority.providers.allocation_count(), 0);
    }

    #[test]
    fn provider_request_allocations_are_counted_additively() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(5, 1);
        let mut plan = MemoryPlan::new();
        plan.push(MemoryClaim::external(
            MemoryDomain::Provider {
                provider: "metal".to_string(),
                device_id: Some(0),
            },
            owner,
            MemoryAllocationClass::RequestTransient,
            "request:staging",
            100,
        ));
        let first = authority.admit(&plan).unwrap();
        let second = authority.admit(&plan).unwrap();
        assert_eq!(authority.providers.allocation_count(), 2);
        drop(first);
        assert_eq!(authority.providers.allocation_count(), 1);
        drop(second);
        assert_eq!(authority.providers.allocation_count(), 0);
    }

    #[test]
    fn mixed_owners_in_one_cuda_domain_are_rejected() {
        let mut plan = MemoryPlan::new();
        for owner in [MemoryOwner::new(1, 0), MemoryOwner::new(1, 1)] {
            plan.push(MemoryClaim::external(
                MemoryDomain::Cuda { device_id: 0 },
                owner,
                MemoryAllocationClass::ExternallyOwned,
                format!("allocation:{}", owner.replica_id),
                1,
            ));
        }
        assert!(owner_for_device(&plan, 0)
            .unwrap_err()
            .contains("mixes multiple"));
    }

    #[test]
    fn typed_domains_distinguish_cpu_zero_from_cuda_zero() {
        let host = MemoryDomain::for_provider("cpu", 0);
        let cuda = MemoryDomain::for_provider("tensorrt", 0);
        assert_eq!(host, MemoryDomain::Host);
        assert_eq!(cuda, MemoryDomain::Cuda { device_id: 0 });
        assert_ne!(host, cuda);
    }

    #[test]
    fn request_transient_is_owned_and_released() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(12, 4);
        let lease = authority
            .admit(&MemoryPlan::request_transient(owner, GIB))
            .unwrap();
        assert_eq!(authority.host.reserved_bytes(), GIB);
        assert_eq!(lease.claims()[0].owner, owner);
        assert_eq!(
            lease.claims()[0].class,
            MemoryAllocationClass::RequestTransient
        );
        drop(lease);
        assert_eq!(authority.host.reserved_bytes(), 0);
    }

    #[test]
    fn elastic_host_lease_grows_shrinks_and_updates_one_snapshot() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(44, 2);
        let mut lease = authority.admit(&MemoryPlan::new()).unwrap();
        let mut growth = MemoryPlan::new();
        growth.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::KvCache,
            2 * GIB,
        ));
        lease.grow(&growth).unwrap();

        let snapshot = authority.snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.owner == owner && row.class == MemoryAllocationClass::KvCache)
            .unwrap();
        assert_eq!(row.planned_bytes, 2 * GIB);
        assert_eq!(row.reserved_bytes, 2 * GIB);
        assert_eq!(authority.host.reserved_bytes(), 2 * GIB);

        let mut shrink = MemoryPlan::new();
        shrink.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::KvCache,
            GIB,
        ));
        lease.shrink(&shrink).unwrap();
        assert_eq!(authority.host.reserved_bytes(), GIB);
        assert_eq!(
            authority.snapshot().rows[0].reserved_bytes,
            GIB,
            "snapshot and host adapter must move atomically"
        );

        drop(lease);
        assert_eq!(authority.host.reserved_bytes(), 0);
        assert!(authority.snapshot().rows.is_empty());
    }

    #[test]
    fn pressure_ratio_includes_in_flight_reservations() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(44, 3);
        let lease = authority
            .admit(&MemoryPlan::request_transient(owner, 2 * GIB))
            .unwrap();

        let snapshot = authority.snapshot();
        let expected = 2.0 / 8.0;
        assert!((snapshot.max_host_ratio().unwrap() - expected).abs() < f64::EPSILON);

        drop(lease);
        assert_eq!(authority.snapshot().max_host_ratio(), Some(0.0));
    }

    #[test]
    fn rejected_elastic_growth_leaves_existing_capacity_unchanged() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(45, 0);
        let mut lease = authority.admit(&MemoryPlan::new()).unwrap();
        let mut admitted = MemoryPlan::new();
        admitted.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::KvCache,
            4 * GIB,
        ));
        lease.grow(&admitted).unwrap();

        let mut over_budget = MemoryPlan::new();
        over_budget.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::KvCache,
            1,
        ));
        assert!(lease.grow(&over_budget).is_err());
        assert_eq!(authority.host.reserved_bytes(), 4 * GIB);
        let row = authority
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner == owner)
            .unwrap();
        assert_eq!(row.reserved_bytes, 4 * GIB);
    }

    #[test]
    fn observed_snapshot_pressure_is_an_admission_gate() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        authority.observe_process_memory(8 * GIB);
        let owner = MemoryOwner::new(45, 1);
        let mut lease = authority.admit(&MemoryPlan::new()).unwrap();
        let mut growth = MemoryPlan::new();
        growth.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::KvCache,
            1,
        ));

        let error = lease.grow(&growth).unwrap_err();
        assert!(error.contains("available=0"));
        assert_eq!(authority.host.reserved_bytes(), 0);
    }

    #[test]
    fn reconcile_releases_capacity_and_rejected_regrowth_is_atomic() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(46, 3);
        let mut lease = authority.admit(&MemoryPlan::new()).unwrap();
        let mut initial = MemoryPlan::new();
        initial.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::KvCache,
            3 * GIB,
        ));
        lease.grow(&initial).unwrap();
        lease.mark_committed();

        let report = |bytes| BackendMemoryReport {
            allocations: vec![kapsl_engine_api::MemoryAllocation {
                allocation_id: "host-kv".to_string(),
                domain: BackendMemoryDomain::Host,
                class: BackendMemoryAllocationClass::KvCache,
                source: BackendMemoryAllocationSource::BackendManaged,
                bytes,
            }],
        };
        lease.reconcile(&report(GIB)).unwrap();
        assert_eq!(authority.host.reserved_bytes(), GIB);
        let row = authority.snapshot().rows.into_iter().next().unwrap();
        assert_eq!(row.planned_bytes, GIB);
        assert_eq!(row.reserved_bytes, GIB);
        assert_eq!(row.committed_bytes, GIB);
        assert_eq!(row.observed_bytes, GIB);

        assert!(lease.reconcile(&report(5 * GIB)).is_err());
        assert_eq!(authority.host.reserved_bytes(), GIB);
        let row = authority.snapshot().rows.into_iter().next().unwrap();
        assert_eq!(row.reserved_bytes, GIB);
        assert_eq!(row.committed_bytes, GIB);
        assert_eq!(
            row.observed_bytes,
            5 * GIB,
            "a rejected reservation must still publish the physical overage"
        );
    }

    #[test]
    fn empty_backend_report_preserves_admitted_fallback_rows() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(46, 4);
        let mut plan = MemoryPlan::new();
        plan.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::ModelSession,
            GIB,
        ));
        plan.push(MemoryClaim::runtime(
            MemoryDomain::Host,
            owner,
            MemoryAllocationClass::TransientWorkspace,
            MIB,
        ));
        let mut lease = authority.admit(&plan).unwrap();
        lease.mark_committed();
        let before = authority.snapshot().rows;

        lease
            .record_loaded_report(&BackendMemoryReport::default())
            .unwrap();
        lease.reconcile(&BackendMemoryReport::default()).unwrap();

        let rows = authority.snapshot().rows;
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows.iter().map(|row| row.planned_bytes).sum::<usize>(),
            GIB + MIB
        );
        assert_eq!(rows, before);
    }

    #[test]
    fn preadmitted_swap_transfers_to_the_persistent_lease_without_a_gap() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(47, 0);
        let plan = |bytes| {
            let mut plan = MemoryPlan::new();
            plan.push(MemoryClaim::runtime(
                MemoryDomain::Host,
                owner,
                MemoryAllocationClass::ModelSession,
                bytes,
            ));
            plan
        };
        let mut persistent = authority.admit(&plan(GIB)).unwrap();
        persistent.mark_committed();
        let mut target = authority.admit(&plan(2 * GIB)).unwrap();
        assert_eq!(authority.host.reserved_bytes(), 3 * GIB);

        let report = BackendMemoryReport {
            allocations: vec![kapsl_engine_api::MemoryAllocation {
                allocation_id: "target-session".to_string(),
                domain: BackendMemoryDomain::Host,
                class: BackendMemoryAllocationClass::ModelSession,
                source: BackendMemoryAllocationSource::BackendManaged,
                bytes: 2 * GIB,
            }],
        };
        let target_planned = target.planned.clone();
        {
            let _operation = authority.operations.lock();
            target.release_locked().unwrap();
            persistent
                .reconcile_pre_admitted_locked(&report, target_planned)
                .unwrap();
        }

        assert_eq!(authority.host.reserved_bytes(), 2 * GIB);
        let row = authority.snapshot().rows.into_iter().next().unwrap();
        assert_eq!(row.planned_bytes, 2 * GIB);
        assert_eq!(row.reserved_bytes, 2 * GIB);
        assert_eq!(row.committed_bytes, 2 * GIB);
        assert_eq!(row.observed_bytes, 2 * GIB);
        drop((target, persistent));
        assert_eq!(authority.host.reserved_bytes(), 0);
    }

    #[test]
    fn reconcile_migrates_authority_before_releasing_the_source() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let owner = MemoryOwner::new(48, 2);
        let mut lease = authority
            .admit(&MemoryPlan {
                claims: vec![MemoryClaim::runtime(
                    MemoryDomain::Host,
                    owner,
                    MemoryAllocationClass::KvCache,
                    GIB,
                )],
            })
            .unwrap();
        lease.mark_committed();
        let pinned = MemoryDomain::HostPinned {
            provider: "cuda".to_string(),
            device_id: Some(0),
        };
        lease
            .reconcile(&BackendMemoryReport {
                allocations: vec![kapsl_engine_api::MemoryAllocation {
                    allocation_id: "migrated-host-kv".to_string(),
                    domain: BackendMemoryDomain::HostPinned {
                        provider: "cuda".to_string(),
                        device_id: Some(0),
                    },
                    class: BackendMemoryAllocationClass::KvCache,
                    source: BackendMemoryAllocationSource::BackendManaged,
                    bytes: GIB,
                }],
            })
            .unwrap();

        assert_eq!(authority.host.reserved_bytes(), GIB);
        let snapshot = authority.snapshot();
        assert!(snapshot
            .rows
            .iter()
            .any(|row| row.domain == pinned && row.reserved_bytes == GIB));
        assert!(lease.claims().iter().any(|claim| {
            claim.domain == pinned
                && matches!(claim.source, MemoryClaimSource::External { .. })
                && claim.bytes == GIB
        }));
    }

    #[test]
    fn model_plan_requires_an_explicit_domain() {
        let authority = MemoryAuthority::new(&cpu_device_info()).unwrap();
        let error = authority
            .model_load_plan(
                &[],
                MemoryOwner::new(1, 0),
                1,
                1,
                &ExternalDeviceMemoryReport::default(),
            )
            .unwrap_err();
        assert!(error.contains("no target domains"));
    }
}
