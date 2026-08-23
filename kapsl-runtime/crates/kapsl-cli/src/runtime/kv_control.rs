//! Out-of-process KV participant control plane.
//!
//! The transport is deliberately small: one newline-delimited JSON request per
//! Unix connection. Policy and byte accounting live in `ExternalKvCoordinator`;
//! framing only decodes the versioned `kapsl-kv-abi` envelopes.

use super::memory::{
    MemoryAllocationClass, MemoryAuthority, MemoryClaim, MemoryDomain, MemoryLease, MemoryOwner,
    MemoryPlan,
};
use kapsl_kv_abi::{
    dispatch_control_request, KvCacheOwnership, KvCommitRequest, KvContractError,
    KvControlRequestEnvelope, KvControlResponse, KvControlResponseEnvelope, KvGroupLease,
    KvIntegrationTier, KvLease, KvMemoryDomain, KvMetadataMode, KvParticipantRegistration,
    KvReserveRequest, KvSequenceKey, KAPSL_KV_ABI_VERSION,
};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_FRAME_BYTES: usize = 1024 * 1024;
const MAX_CONNECTIONS: usize = 64;

struct ParticipantRecord {
    registration: KvParticipantRegistration,
    owner: MemoryOwner,
}

struct ExternalLeaseRecord {
    public: KvLease,
    request: KvReserveRequest,
    participant_id: String,
    ttl: Duration,
    expires_at: Instant,
    memory: MemoryLease,
}

#[derive(Default)]
struct CoordinatorState {
    participants: HashMap<String, ParticipantRecord>,
    leases: HashMap<String, ExternalLeaseRecord>,
    sequences: HashMap<(String, KvSequenceKey), String>,
}

/// Runtime implementation of the backend-neutral KV coordinator contract.
///
/// This listener accepts only `kv_connected`/opaque participants. A
/// `shared_pool` backend needs a negotiated data plane and cannot be promoted
/// merely by reaching this control socket.
pub(crate) struct ExternalKvCoordinator {
    memory: Arc<MemoryAuthority>,
    state: Mutex<CoordinatorState>,
    next_participant_slot: AtomicU32,
    next_lease_id: AtomicU64,
    maximum_lease_ttl: Duration,
}

impl ExternalKvCoordinator {
    pub(crate) fn new(
        memory: Arc<MemoryAuthority>,
        maximum_lease_ttl: Duration,
    ) -> Result<Arc<Self>, String> {
        if maximum_lease_ttl.is_zero() {
            return Err("KV control lease TTL must be non-zero".to_string());
        }
        Ok(Arc::new(Self {
            memory,
            state: Mutex::new(CoordinatorState::default()),
            next_participant_slot: AtomicU32::new(0),
            next_lease_id: AtomicU64::new(1),
            maximum_lease_ttl,
        }))
    }

    pub(crate) fn expire_stale(&self) -> usize {
        let now = Instant::now();
        let expired = {
            let mut state = self.state.lock();
            let lease_ids = state
                .leases
                .iter()
                .filter_map(|(lease_id, lease)| {
                    (lease.expires_at <= now).then_some(lease_id.clone())
                })
                .collect::<Vec<_>>();
            let mut expired = Vec::with_capacity(lease_ids.len());
            for lease_id in lease_ids {
                if let Some(lease) = state.leases.remove(&lease_id) {
                    state
                        .sequences
                        .remove(&(lease.participant_id.clone(), lease.request.sequence.clone()));
                    expired.push(lease);
                }
            }
            expired
        };
        let count = expired.len();
        drop(expired);
        if count > 0 {
            log::warn!("[kv-control] expired {count} stale capacity lease(s)");
        }
        count
    }

    #[cfg(test)]
    fn participant_count(&self) -> usize {
        self.state.lock().participants.len()
    }

    #[cfg(test)]
    fn lease_count(&self) -> usize {
        self.state.lock().leases.len()
    }

    fn participant_registration(
        &self,
        participant_id: &str,
    ) -> Result<(KvParticipantRegistration, MemoryOwner), KvContractError> {
        self.state
            .lock()
            .participants
            .get(participant_id)
            .map(|record| (record.registration.clone(), record.owner))
            .ok_or_else(|| KvContractError::NotFound {
                message: format!("KV participant '{participant_id}' is not registered"),
            })
    }

    fn effective_ttl(&self, requested_ms: Option<u64>) -> Result<Duration, KvContractError> {
        let requested = requested_ms
            .map(Duration::from_millis)
            .unwrap_or(self.maximum_lease_ttl);
        if requested > self.maximum_lease_ttl {
            return Err(KvContractError::invalid_request(format!(
                "requested lease TTL {}ms exceeds the runtime maximum {}ms",
                requested.as_millis(),
                self.maximum_lease_ttl.as_millis()
            )));
        }
        Ok(requested)
    }
}

impl kapsl_kv_abi::KvCoordinator for ExternalKvCoordinator {
    fn register(&self, registration: &KvParticipantRegistration) -> Result<(), KvContractError> {
        registration.validate()?;
        if registration.capabilities.tier != KvIntegrationTier::KvConnected
            || registration.capabilities.metadata_mode != KvMetadataMode::Opaque
            || registration.capabilities.ownership != KvCacheOwnership::Backend
        {
            return Err(KvContractError::invalid_capabilities(
                "the external control listener accepts only kv_connected/opaque/backend-owned participants",
            ));
        }

        for domain in registration
            .capacity_model
            .groups
            .iter()
            .flat_map(|group| &group.memory_domains)
        {
            let runtime_domain = runtime_memory_domain(domain)?;
            if !self.memory.supports_external_leases(&runtime_domain) {
                return Err(KvContractError::invalid_capabilities(format!(
                    "runtime has no bounded external-lease authority for KV domain {runtime_domain}"
                )));
            }
        }

        self.expire_stale();
        let mut state = self.state.lock();
        if let Some(existing) = state.participants.get(&registration.participant_id) {
            if existing.registration == *registration {
                return Ok(());
            }
            if state
                .leases
                .values()
                .any(|lease| lease.participant_id == registration.participant_id)
            {
                return Err(KvContractError::invalid_request(format!(
                    "participant '{}' cannot change registration while leases are active",
                    registration.participant_id
                )));
            }
            let owner = existing.owner;
            state.participants.insert(
                registration.participant_id.clone(),
                ParticipantRecord {
                    registration: registration.clone(),
                    owner,
                },
            );
            return Ok(());
        }

        let slot = self.next_participant_slot.fetch_add(1, Ordering::Relaxed);
        let owner = MemoryOwner::external_kv(slot).ok_or_else(|| KvContractError::Internal {
            message: "external KV participant owner space exhausted".to_string(),
        })?;
        state.participants.insert(
            registration.participant_id.clone(),
            ParticipantRecord {
                registration: registration.clone(),
                owner,
            },
        );
        log::info!(
            "[kv-control] registered participant '{}' backend={} model={} tier=kv_connected metadata=opaque",
            registration.participant_id,
            registration.backend,
            registration.model_fingerprint
        );
        Ok(())
    }

    fn reserve(
        &self,
        participant_id: &str,
        request: &KvReserveRequest,
    ) -> Result<KvLease, KvContractError> {
        request.validate()?;
        self.expire_stale();

        let (registration, owner) = self.participant_registration(participant_id)?;
        if let Some(prefix) = &request.prefix {
            if prefix.model_fingerprint != registration.model_fingerprint {
                return Err(KvContractError::invalid_request(
                    "prefix model fingerprint does not match the registered participant",
                ));
            }
            return Err(KvContractError::unsupported("reserve_prefix"));
        }

        let sequence_key = (participant_id.to_string(), request.sequence.clone());
        {
            let state = self.state.lock();
            if let Some(existing_id) = state.sequences.get(&sequence_key) {
                let existing = state
                    .leases
                    .get(existing_id)
                    .expect("sequence index must reference a live lease");
                if existing.request == *request {
                    return Ok(existing.public.clone());
                }
                return Err(KvContractError::invalid_request(
                    "sequence already has a lease with a different reservation",
                ));
            }
        }

        let known_groups = registration
            .capacity_model
            .groups
            .iter()
            .map(|group| group.group_id.as_str())
            .collect::<std::collections::HashSet<_>>();
        if request
            .groups
            .iter()
            .any(|group| !known_groups.contains(group.group_id.as_str()))
        {
            return Err(KvContractError::invalid_request(
                "reservation references an unregistered cache group",
            ));
        }
        let bytes_by_domain = registration
            .capacity_model
            .bytes_by_domain_for_reservations(&request.groups)
            .ok_or_else(|| KvContractError::CapacityExhausted {
                message: "reservation exceeds the participant capacity model or byte accounting overflowed"
                    .to_string(),
            })?;

        let numeric_id = self.next_lease_id.fetch_add(1, Ordering::Relaxed);
        let lease_id = format!("kapsl-kv-{numeric_id:016x}");
        let mut plan = MemoryPlan::new();
        for (domain, bytes) in bytes_by_domain {
            let bytes = usize::try_from(bytes).map_err(|_| KvContractError::CapacityExhausted {
                message: "KV reservation does not fit the runtime address space".to_string(),
            })?;
            plan.push(MemoryClaim::external(
                runtime_memory_domain(&domain)?,
                owner,
                MemoryAllocationClass::KvCache,
                format!("kv-participant:{participant_id}:{lease_id}"),
                bytes,
            ));
        }
        let memory = self
            .memory
            .admit(&plan)
            .map_err(|message| KvContractError::CapacityExhausted { message })?;

        let ttl = self.effective_ttl(request.ttl_ms)?;
        let expires_at = Instant::now() + ttl;
        let public = KvLease {
            lease_id: lease_id.clone(),
            sequence: request.sequence.clone(),
            groups: request
                .groups
                .iter()
                .map(|group| KvGroupLease {
                    group_id: group.group_id.clone(),
                    token_capacity: group.token_capacity,
                    blocks: Vec::new(),
                })
                .collect(),
            expires_at_unix_ms: Some(unix_ms_after(ttl)),
        };
        public.validate()?;

        let record = ExternalLeaseRecord {
            public: public.clone(),
            request: request.clone(),
            participant_id: participant_id.to_string(),
            ttl,
            expires_at,
            memory,
        };
        let mut state = self.state.lock();
        if state
            .participants
            .get(participant_id)
            .is_none_or(|current| current.registration != registration)
        {
            return Err(KvContractError::invalid_request(
                "participant registration changed while the reservation was being admitted; retry",
            ));
        }
        if let Some(existing_id) = state.sequences.get(&sequence_key) {
            let existing = state
                .leases
                .get(existing_id)
                .expect("sequence index must reference a live lease");
            if existing.request == *request {
                return Ok(existing.public.clone());
            }
            return Err(KvContractError::invalid_request(
                "sequence acquired a conflicting lease concurrently",
            ));
        }
        state.sequences.insert(sequence_key, lease_id.clone());
        state.leases.insert(lease_id, record);
        Ok(public)
    }

    fn commit(
        &self,
        participant_id: &str,
        request: &KvCommitRequest,
    ) -> Result<(), KvContractError> {
        request.validate()?;
        self.expire_stale();
        if request.prefix.is_some() {
            return Err(KvContractError::unsupported("commit_prefix"));
        }
        let mut state = self.state.lock();
        let lease =
            state
                .leases
                .get_mut(&request.lease_id)
                .ok_or_else(|| KvContractError::NotFound {
                    message: format!("KV lease '{}' does not exist", request.lease_id),
                })?;
        ensure_lease_owner(lease, participant_id)?;
        if lease
            .public
            .groups
            .iter()
            .any(|group| request.computed_tokens > group.token_capacity)
        {
            return Err(KvContractError::invalid_request(
                "computed_tokens exceeds the reserved token capacity",
            ));
        }
        lease.memory.commit_capacity();
        Ok(())
    }

    fn touch(&self, participant_id: &str, lease_id: &str) -> Result<(), KvContractError> {
        self.expire_stale();
        let mut state = self.state.lock();
        let lease = state
            .leases
            .get_mut(lease_id)
            .ok_or_else(|| KvContractError::NotFound {
                message: format!("KV lease '{lease_id}' does not exist"),
            })?;
        ensure_lease_owner(lease, participant_id)?;
        lease.expires_at = Instant::now() + lease.ttl;
        lease.public.expires_at_unix_ms = Some(unix_ms_after(lease.ttl));
        Ok(())
    }

    fn heartbeat(&self, participant_id: &str) -> Result<(), KvContractError> {
        self.expire_stale();
        let mut state = self.state.lock();
        if !state.participants.contains_key(participant_id) {
            return Err(KvContractError::NotFound {
                message: format!("KV participant '{participant_id}' is not registered"),
            });
        }
        let now = Instant::now();
        for lease in state
            .leases
            .values_mut()
            .filter(|lease| lease.participant_id == participant_id)
        {
            lease.expires_at = now + lease.ttl;
            lease.public.expires_at_unix_ms = Some(unix_ms_after(lease.ttl));
        }
        Ok(())
    }

    fn release(&self, participant_id: &str, lease_id: &str) -> Result<(), KvContractError> {
        self.expire_stale();
        let released = {
            let mut state = self.state.lock();
            let lease = state
                .leases
                .get(lease_id)
                .ok_or_else(|| KvContractError::NotFound {
                    message: format!("KV lease '{lease_id}' does not exist"),
                })?;
            ensure_lease_owner(lease, participant_id)?;
            let released = state.leases.remove(lease_id).expect("lease checked above");
            state.sequences.remove(&(
                released.participant_id.clone(),
                released.request.sequence.clone(),
            ));
            released
        };
        drop(released);
        Ok(())
    }
}

fn ensure_lease_owner(
    lease: &ExternalLeaseRecord,
    participant_id: &str,
) -> Result<(), KvContractError> {
    if lease.participant_id == participant_id {
        Ok(())
    } else {
        Err(KvContractError::NotFound {
            message: "KV lease does not belong to this participant".to_string(),
        })
    }
}

fn runtime_memory_domain(domain: &KvMemoryDomain) -> Result<MemoryDomain, KvContractError> {
    let device_id = |value: u32| {
        usize::try_from(value).map_err(|_| {
            KvContractError::invalid_capabilities(
                "KV memory-domain device ID does not fit the runtime address space",
            )
        })
    };
    Ok(match domain {
        KvMemoryDomain::Host => MemoryDomain::Host,
        KvMemoryDomain::HostPinned {
            provider,
            device_id: id,
        } => MemoryDomain::HostPinned {
            provider: provider.clone(),
            device_id: id.map(device_id).transpose()?,
        },
        KvMemoryDomain::HostMapped {
            provider,
            device_id: id,
        } => MemoryDomain::HostMapped {
            provider: provider.clone(),
            device_id: id.map(device_id).transpose()?,
        },
        KvMemoryDomain::Cuda { device_id: id } => MemoryDomain::Cuda {
            device_id: device_id(*id)?,
        },
        KvMemoryDomain::Provider {
            provider,
            device_id: id,
        } => MemoryDomain::Provider {
            provider: provider.clone(),
            device_id: id.map(device_id).transpose()?,
        },
    })
}

fn unix_ms_after(duration: Duration) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let later = now.saturating_add(duration.as_millis());
    u64::try_from(later).unwrap_or(u64::MAX)
}

#[cfg(unix)]
mod unix {
    use super::*;
    use std::io;
    use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
    use std::path::{Path, PathBuf};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{UnixListener, UnixStream};
    use tokio::sync::Semaphore;

    #[derive(Clone, Copy)]
    struct SocketIdentity {
        device: u64,
        inode: u64,
    }

    pub(crate) struct KvControlServer {
        listener: UnixListener,
        path: PathBuf,
        identity: SocketIdentity,
        coordinator: Arc<ExternalKvCoordinator>,
        permits: Arc<Semaphore>,
        max_frame_bytes: usize,
    }

    impl KvControlServer {
        pub(crate) async fn bind(
            path: impl AsRef<Path>,
            coordinator: Arc<ExternalKvCoordinator>,
        ) -> Result<Self, String> {
            let path = path.as_ref().to_path_buf();
            if !path.is_absolute() {
                return Err("KV control socket path must be absolute".to_string());
            }
            let parent = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .ok_or_else(|| "KV control socket path has no parent directory".to_string())?;
            if !parent.is_dir() {
                return Err(format!(
                    "KV control socket parent directory {} does not exist",
                    parent.display()
                ));
            }

            remove_stale_socket(&path).await?;
            let listener = UnixListener::bind(&path).map_err(|error| {
                format!(
                    "failed to bind KV control socket {}: {error}",
                    path.display()
                )
            })?;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).map_err(
                |error| {
                    format!(
                        "failed to secure KV control socket {}: {error}",
                        path.display()
                    )
                },
            )?;
            let metadata = std::fs::metadata(&path).map_err(|error| {
                format!(
                    "failed to inspect KV control socket {}: {error}",
                    path.display()
                )
            })?;
            let identity = SocketIdentity {
                device: metadata.dev(),
                inode: metadata.ino(),
            };
            Ok(Self {
                listener,
                path,
                identity,
                coordinator,
                permits: Arc::new(Semaphore::new(MAX_CONNECTIONS)),
                max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            })
        }

        pub(crate) async fn run(self) -> io::Result<()> {
            let sweep_period = (self.coordinator.maximum_lease_ttl / 2)
                .max(Duration::from_millis(250))
                .min(Duration::from_secs(5));
            let mut sweep = tokio::time::interval(sweep_period);
            loop {
                tokio::select! {
                    accepted = self.listener.accept() => {
                        let (stream, _) = accepted?;
                        let permit = Arc::clone(&self.permits)
                            .acquire_owned()
                            .await
                            .map_err(|_| io::Error::other("KV control connection limiter closed"))?;
                        let coordinator = Arc::clone(&self.coordinator);
                        let max_frame_bytes = self.max_frame_bytes;
                        tokio::spawn(async move {
                            let _permit = permit;
                            if let Err(error) = handle_connection(stream, coordinator, max_frame_bytes).await {
                                log::warn!("[kv-control] connection failed: {error}");
                            }
                        });
                    }
                    _ = sweep.tick() => {
                        self.coordinator.expire_stale();
                    }
                }
            }
        }
    }

    impl Drop for KvControlServer {
        fn drop(&mut self) {
            let same_socket = std::fs::metadata(&self.path).is_ok_and(|metadata| {
                metadata.file_type().is_socket()
                    && metadata.dev() == self.identity.device
                    && metadata.ino() == self.identity.inode
            });
            if same_socket {
                let _ = std::fs::remove_file(&self.path);
            }
        }
    }

    async fn remove_stale_socket(path: &Path) -> Result<(), String> {
        let metadata = match std::fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(format!(
                    "failed to inspect KV control socket {}: {error}",
                    path.display()
                ));
            }
        };
        if !metadata.file_type().is_socket() {
            return Err(format!(
                "refusing to replace non-socket path {}",
                path.display()
            ));
        }
        if UnixStream::connect(path).await.is_ok() {
            return Err(format!(
                "KV control socket {} is already served by another process",
                path.display()
            ));
        }
        std::fs::remove_file(path).map_err(|error| {
            format!(
                "failed to remove stale KV control socket {}: {error}",
                path.display()
            )
        })
    }

    async fn handle_connection(
        mut stream: UnixStream,
        coordinator: Arc<ExternalKvCoordinator>,
        max_frame_bytes: usize,
    ) -> io::Result<()> {
        let response = match tokio::time::timeout(
            Duration::from_secs(5),
            read_frame(&mut stream, max_frame_bytes),
        )
        .await
        {
            Ok(Ok(frame)) => match serde_json::from_slice::<KvControlRequestEnvelope>(&frame) {
                Ok(envelope) => dispatch_control_request(coordinator.as_ref(), envelope),
                Err(error) => error_response(
                    request_id_from_invalid_frame(&frame),
                    KvContractError::invalid_request(format!(
                        "invalid KV control envelope: {error}"
                    )),
                ),
            },
            Ok(Err(error)) => error_response(
                String::new(),
                KvContractError::Transport {
                    message: error.to_string(),
                },
            ),
            Err(_) => error_response(
                String::new(),
                KvContractError::Transport {
                    message: "timed out reading KV control frame".to_string(),
                },
            ),
        };
        let mut encoded = serde_json::to_vec(&response).map_err(io::Error::other)?;
        encoded.push(b'\n');
        stream.write_all(&encoded).await?;
        match stream.shutdown().await {
            Err(error) if error.kind() == io::ErrorKind::NotConnected => Ok(()),
            result => result,
        }
    }

    async fn read_frame(stream: &mut UnixStream, max_frame_bytes: usize) -> io::Result<Vec<u8>> {
        let mut frame = Vec::new();
        let mut buffer = [0u8; 8192];
        loop {
            let read = stream.read(&mut buffer).await?;
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "peer closed before a newline-delimited KV control frame",
                ));
            }
            if let Some(newline) = buffer[..read].iter().position(|byte| *byte == b'\n') {
                frame.extend_from_slice(&buffer[..newline]);
                if frame.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "KV control frame is empty",
                    ));
                }
                if frame.len() > max_frame_bytes {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "KV control frame exceeds maximum size",
                    ));
                }
                return Ok(frame);
            }
            frame.extend_from_slice(&buffer[..read]);
            if frame.len() > max_frame_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "KV control frame exceeds maximum size",
                ));
            }
        }
    }

    fn request_id_from_invalid_frame(frame: &[u8]) -> String {
        serde_json::from_slice::<serde_json::Value>(frame)
            .ok()
            .and_then(|value| value.get("request_id")?.as_str().map(str::to_string))
            .unwrap_or_default()
    }

    fn error_response(request_id: String, error: KvContractError) -> KvControlResponseEnvelope {
        KvControlResponseEnvelope {
            abi_version: KAPSL_KV_ABI_VERSION,
            request_id,
            response: KvControlResponse::Error { error },
        }
    }

    pub(crate) use KvControlServer as Server;
}

#[cfg(unix)]
pub(crate) use unix::Server as KvControlServer;

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_hal::device::{Device, DeviceBackend, DeviceInfo};
    use kapsl_kv_abi::{
        KvBackendCapabilities, KvCapacityGroup, KvCapacityModel, KvControlRequest,
        KvGroupReservation, KvMemoryDomain, KvSequenceKey,
    };

    fn test_memory() -> Arc<MemoryAuthority> {
        let info = DeviceInfo {
            cpu_cores: 1,
            total_memory: 1024 * 1024,
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
        };
        MemoryAuthority::new(&info).expect("test memory authority")
    }

    fn registration() -> KvParticipantRegistration {
        KvParticipantRegistration {
            participant_id: "vllm:test:scheduler".to_string(),
            backend: "vllm".to_string(),
            model_fingerprint: "sha256:test".to_string(),
            capabilities: KvBackendCapabilities::opaque_connected(),
            capacity_model: KvCapacityModel {
                groups: vec![KvCapacityGroup {
                    group_id: "vllm.group.0".to_string(),
                    pool_id: "vllm.pool.0".to_string(),
                    allocation_granularity_tokens: 16,
                    bytes_per_allocation: 4096,
                    memory_domains: vec![KvMemoryDomain::Host],
                    max_allocations: Some(1024),
                }],
            },
            topology: None,
        }
    }

    fn reservation(ttl_ms: Option<u64>) -> KvReserveRequest {
        KvReserveRequest {
            sequence: KvSequenceKey {
                request_id: "request-1".to_string(),
                sequence_id: "request-1".to_string(),
            },
            groups: vec![KvGroupReservation {
                group_id: "vllm.group.0".to_string(),
                token_capacity: 32,
                minimum_blocks: None,
            }],
            prefix: None,
            priority: 0,
            ttl_ms,
        }
    }

    #[test]
    fn coordinator_lifecycle_is_reflected_in_memory_authority() {
        use kapsl_kv_abi::KvCoordinator as _;

        let memory = test_memory();
        let coordinator =
            ExternalKvCoordinator::new(memory.clone(), Duration::from_secs(30)).unwrap();
        coordinator.register(&registration()).unwrap();
        assert_eq!(coordinator.participant_count(), 1);

        let lease = coordinator
            .reserve("vllm:test:scheduler", &reservation(None))
            .unwrap();
        let duplicate = coordinator
            .reserve("vllm:test:scheduler", &reservation(None))
            .unwrap();
        assert_eq!(lease.lease_id, duplicate.lease_id);
        assert_eq!(coordinator.lease_count(), 1);
        let row = memory
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner.is_external_kv())
            .expect("external KV memory row");
        assert_eq!(row.reserved_bytes, 8192);

        coordinator
            .commit(
                "vllm:test:scheduler",
                &KvCommitRequest {
                    lease_id: lease.lease_id.clone(),
                    computed_tokens: 17,
                    prefix: None,
                },
            )
            .unwrap();
        let committed = memory
            .snapshot()
            .rows
            .into_iter()
            .find(|row| row.owner.is_external_kv())
            .unwrap();
        assert_eq!(committed.committed_bytes, 8192);

        coordinator
            .touch("vllm:test:scheduler", &lease.lease_id)
            .unwrap();
        coordinator.heartbeat("vllm:test:scheduler").unwrap();
        coordinator
            .release("vllm:test:scheduler", &lease.lease_id)
            .unwrap();
        assert_eq!(coordinator.lease_count(), 0);
        assert!(memory
            .snapshot()
            .rows
            .iter()
            .all(|row| !row.owner.is_external_kv()));
    }

    #[test]
    fn stale_lease_returns_capacity_to_the_authority() {
        use kapsl_kv_abi::KvCoordinator as _;

        let memory = test_memory();
        let coordinator =
            ExternalKvCoordinator::new(memory.clone(), Duration::from_millis(5)).unwrap();
        coordinator.register(&registration()).unwrap();
        coordinator
            .reserve("vllm:test:scheduler", &reservation(Some(5)))
            .unwrap();
        std::thread::sleep(Duration::from_millis(15));
        assert_eq!(coordinator.expire_stale(), 1);
        assert_eq!(coordinator.lease_count(), 0);
        assert!(memory
            .snapshot()
            .rows
            .iter()
            .all(|row| !row.owner.is_external_kv()));
    }

    #[test]
    fn registration_and_ttl_fail_closed_without_bounded_authority() {
        use kapsl_kv_abi::KvCoordinator as _;

        let memory = test_memory();
        let coordinator =
            ExternalKvCoordinator::new(memory.clone(), Duration::from_secs(30)).unwrap();
        let mut cuda_registration = registration();
        cuda_registration.capacity_model.groups[0].memory_domains =
            vec![KvMemoryDomain::Cuda { device_id: 0 }];
        assert!(matches!(
            coordinator.register(&cuda_registration),
            Err(KvContractError::InvalidCapabilities { .. })
        ));

        coordinator.register(&registration()).unwrap();
        assert!(matches!(
            coordinator.reserve("vllm:test:scheduler", &reservation(Some(30_001))),
            Err(KvContractError::InvalidRequest { .. })
        ));
        assert_eq!(coordinator.lease_count(), 0);
        assert!(memory
            .snapshot()
            .rows
            .iter()
            .all(|row| !row.owner.is_external_kv()));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_server_dispatches_the_versioned_wire_envelope() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::net::UnixStream;

        let coordinator =
            ExternalKvCoordinator::new(test_memory(), Duration::from_secs(30)).unwrap();
        let path = std::env::temp_dir().join(format!(
            "kapsl-kv-control-test-{}-{}.sock",
            std::process::id(),
            coordinator.next_lease_id.load(Ordering::Relaxed)
        ));
        let server = KvControlServer::bind(&path, coordinator.clone())
            .await
            .unwrap();
        let task = tokio::spawn(server.run());

        let envelope = KvControlRequestEnvelope {
            abi_version: KAPSL_KV_ABI_VERSION,
            request_id: "rpc-register".to_string(),
            request: KvControlRequest::Register {
                registration: registration(),
            },
        };
        let mut stream = UnixStream::connect(&path).await.unwrap();
        let mut frame = serde_json::to_vec(&envelope).unwrap();
        frame.push(b'\n');
        stream.write_all(&frame).await.unwrap();
        let mut reader = BufReader::new(stream);
        let mut response = Vec::new();
        reader.read_until(b'\n', &mut response).await.unwrap();
        let response: KvControlResponseEnvelope = serde_json::from_slice(&response).unwrap();
        assert_eq!(response.request_id, "rpc-register");
        assert!(matches!(response.response, KvControlResponse::Registered));
        assert_eq!(coordinator.participant_count(), 1);

        task.abort();
        let _ = task.await;
    }
}
