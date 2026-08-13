//! Backend-neutral memory admission and lifetime ownership.
//!
//! A [`MemoryPlan`] describes memory before a backend allocates it. The
//! [`MemoryAuthority`] routes each claim to the manager for its domain, and a
//! committed [`MemoryLease`] retains those reservations until the owning
//! model/replica is unloaded. Provider domains are accounting adapters: they
//! participate in ownership and lifetime now, without pretending that the
//! runtime physically allocates their memory.

use super::host_memory::{HostMemoryLease, HostMemoryLoadAdmission, HostMemoryManager};
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
                let provider = match provider.trim().to_ascii_lowercase().as_str() {
                    // CoreML executes in the Metal device domain selected by
                    // the runtime. Treat both names as one admission domain so
                    // an explicit backend report is not discarded and replaced
                    // by a legacy estimate.
                    "coreml" | "metal" => "metal".to_string(),
                    provider => provider.to_string(),
                };
                Self::Provider {
                    provider,
                    device_id: *device_id,
                }
            }
        }
    }

    pub(crate) fn for_provider(provider: &str, device_id: usize) -> Self {
        match provider.trim().to_ascii_lowercase().as_str() {
            "cpu" => Self::Host,
            "cuda" | "tensorrt" => Self::Cuda { device_id },
            "coreml" | "metal" => Self::Provider {
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
pub(crate) struct MemoryOwner {
    pub(crate) model_id: u32,
    pub(crate) replica_id: u32,
}

impl MemoryOwner {
    pub(crate) const fn new(model_id: u32, replica_id: u32) -> Self {
        Self {
            model_id,
            replica_id,
        }
    }
}

impl fmt::Display for MemoryOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "model {} replica {}",
            self.model_id, self.replica_id
        )
    }
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
    owners: HashMap<MemoryOwner, usize>,
}

#[derive(Default)]
struct ProviderMemoryLedger {
    allocations: Mutex<HashMap<ProviderAllocationKey, ProviderAllocation>>,
    next_transient_allocation: std::sync::atomic::AtomicU64,
}

impl ProviderMemoryLedger {
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
        if let Some(existing) = allocations.get_mut(&key) {
            if existing.class != claim.class {
                return Err(format!(
                    "provider allocation `{}` in {} is already classified as {}, not {} for {}",
                    key.allocation_id, key.domain, existing.class, claim.class, claim.owner
                ));
            }
            let refs = existing.owners.entry(claim.owner).or_default();
            *refs = refs.saturating_add(1);
            existing.bytes = existing.bytes.max(claim.bytes);
        } else {
            let mut owners = HashMap::new();
            owners.insert(claim.owner, 1);
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
        }))
    }

    fn release(&self, key: &ProviderAllocationKey, owner: MemoryOwner) {
        let mut allocations = self.allocations.lock();
        let remove = allocations.get_mut(key).is_some_and(|allocation| {
            let remove_owner = allocation.owners.get_mut(&owner).is_some_and(|refs| {
                *refs = refs.saturating_sub(1);
                *refs == 0
            });
            if remove_owner {
                allocation.owners.remove(&owner);
            }
            allocation.owners.is_empty()
        });
        if remove {
            allocations.remove(key);
        }
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
}

impl Drop for ProviderMemoryLease {
    fn drop(&mut self) {
        self.ledger.release(&self.key, self.owner);
    }
}

/// Process-wide authority spanning host, CUDA, and provider adapter domains.
pub(crate) struct MemoryAuthority {
    host: Arc<HostMemoryManager>,
    #[cfg(feature = "gpu-device-pool")]
    cuda: Option<Arc<DeviceMemoryManager>>,
    providers: Arc<ProviderMemoryLedger>,
}

impl MemoryAuthority {
    #[cfg_attr(feature = "gpu-device-pool", allow(dead_code))]
    pub(crate) fn new(device_info: &DeviceInfo) -> Result<Arc<Self>, String> {
        #[cfg(feature = "gpu-device-pool")]
        {
            Self::new_with_cuda_plan(device_info, &DeviceMemoryBootstrapPlan::default())
        }
        #[cfg(not(feature = "gpu-device-pool"))]
        {
            Ok(Arc::new(Self {
                host: HostMemoryManager::new(device_info),
                providers: Arc::new(ProviderMemoryLedger::default()),
            }))
        }
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn new_with_cuda_plan(
        device_info: &DeviceInfo,
        bootstrap: &DeviceMemoryBootstrapPlan,
    ) -> Result<Arc<Self>, String> {
        Ok(Arc::new(Self {
            host: HostMemoryManager::new(device_info),
            cuda: DeviceMemoryManager::from_env_with_plan(device_info, bootstrap)?,
            providers: Arc::new(ProviderMemoryLedger::default()),
        }))
    }

    pub(crate) fn host_budget(&self) -> super::host_memory::HostMemoryBudget {
        self.host.budget()
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
    pub(crate) fn admit(&self, plan: &MemoryPlan) -> Result<MemoryLease, String> {
        let mut lease = MemoryLease::new(plan.claims.clone());
        for claim in plan.claims() {
            match &claim.domain {
                MemoryDomain::Host
                | MemoryDomain::HostPinned { .. }
                | MemoryDomain::HostMapped { .. } => {
                    if let Some(host) = self.host.admit(claim)? {
                        lease.host.push(host);
                    }
                }
                MemoryDomain::Provider { .. } => {
                    if let Some(provider) = self.providers.reserve(claim)? {
                        lease.providers.push(provider);
                    }
                }
                MemoryDomain::Cuda { .. } => {
                    #[cfg(feature = "gpu-device-pool")]
                    {
                        let manager = self.cuda.as_ref().ok_or_else(|| {
                            format!(
                                "{} claim for {} has no CUDA memory authority",
                                claim.domain, claim.owner
                            )
                        })?;
                        if let Some(transient) = manager.admit_transient(claim)? {
                            lease.cuda_transient.push(transient);
                        }
                    }
                    #[cfg(not(feature = "gpu-device-pool"))]
                    return Err(format!(
                        "{} claim for {} requires CUDA memory authority",
                        claim.domain, claim.owner
                    ));
                }
            }
        }
        Ok(lease)
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
        let mut cuda = Vec::new();
        if let Some(manager) = self.cuda.as_ref() {
            for device_id in plan.cuda_device_ids() {
                let report = plan.external_report_for_cuda(device_id);
                let owner = owner_for_device(plan, device_id)?;
                if let Some(admission) = manager
                    .begin_swap_admission(device_id, owner, &report)
                    .await?
                {
                    cuda.push(admission);
                }
            }
        }
        Ok(MemorySwapLease {
            _claims: plan.claims.clone(),
            _cuda: cuda,
        })
    }

    #[cfg(any(
        feature = "native",
        feature = "gguf-native",
        feature = "gguf-cuda-shared-kv"
    ))]
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
            reconcile_backend_claim_bytes(&mut lease.claims, actual_report);
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
        lease.claims.extend(self.cuda_plan.claims.clone());
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
        lease
    }
}

/// RAII ownership of all admitted memory in a plan.
pub(crate) struct MemoryLease {
    claims: Vec<MemoryClaim>,
    host: Vec<HostMemoryLease>,
    providers: Vec<ProviderMemoryLease>,
    #[cfg(feature = "gpu-device-pool")]
    cuda: Vec<DeviceMemoryLease>,
    #[cfg(feature = "gpu-device-pool")]
    cuda_transient: Vec<DeviceMemoryTransientLease>,
}

impl MemoryLease {
    fn new(claims: Vec<MemoryClaim>) -> Self {
        Self {
            claims,
            host: Vec::new(),
            providers: Vec::new(),
            #[cfg(feature = "gpu-device-pool")]
            cuda: Vec::new(),
            #[cfg(feature = "gpu-device-pool")]
            cuda_transient: Vec::new(),
        }
    }

    pub(crate) fn claims(&self) -> &[MemoryClaim] {
        &self.claims
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    #[cfg(feature = "gpu-device-pool")]
    pub(crate) fn reconcile_report(&mut self, report: &BackendMemoryReport) -> Result<(), String> {
        let external_cuda = external_cuda_report_from_backend(report);
        for lease in &mut self.cuda {
            lease.reconcile_report(&external_cuda)?;
        }
        reconcile_backend_claim_bytes(&mut self.claims, report);
        Ok(())
    }
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
    _claims: Vec<MemoryClaim>,
    _cuda: Vec<DeviceMemorySwapAdmission>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_hal::device::{Device, DeviceBackend};

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
