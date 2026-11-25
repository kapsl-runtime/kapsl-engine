//! Backend-neutral launch, readiness, and supervision of a managed server.
//!
//! A managed server is an OpenAI-compatible child process whose KV memory is
//! governed through the KV control socket. Everything that does not depend on
//! which server it is lives here: the deployment, the process lifecycle, the
//! health and KV readiness gate, supervised restart, and the OpenAI bridge. A
//! serving backend adds a [`ManagedServerDriver`].
//!
//! Managed vLLM predates this module and still carries its own lifecycle in
//! the parent module. The HTTP bridge, the readiness fence, and the process
//! helpers are shared with it from there.

use super::*;
use std::sync::Weak;

/// KV participant state that decides whether a managed server may take
/// traffic, and that its engine reports as metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ManagedKvStatus {
    /// Identifies one registration. A different value means the participant
    /// registered again.
    pub(super) revision: u64,
    pub(super) capacity_bytes: u64,
    pub(super) leased_bytes: u64,
    pub(super) active_leases: u64,
}

/// What a serving backend supplies for the runtime to launch and supervise it.
pub(super) trait ManagedServerDriver: Send + Sync + 'static {
    /// Name used in log lines and errors, for example `SGLang`.
    fn name(&self) -> &'static str;

    /// Label of the child's archived output, for example `sglang`.
    fn log_label(&self) -> &'static str;

    /// Build the command that starts the server. Called for every start.
    fn command(&self) -> Result<Command, EngineError>;

    /// Path the server answers with HTTP 2xx once it can serve.
    fn health_path(&self) -> &'static str {
        "/health"
    }

    /// State of the KV participant launched under `participant_base`.
    ///
    /// `Ok(None)` means it is not ready yet. An error is a lifecycle violation:
    /// the server must not become routable and is restarted.
    fn kv_status(
        &self,
        coordinator: &ExternalKvCoordinator,
        participant_base: &str,
    ) -> Result<Option<ManagedKvStatus>, String>;
}

/// Participant configuration for a server whose ranks run on `device_ids`.
///
/// This is the one place the runtime builds the document, so every managed
/// backend hands its adapter the same fields. Callers set the shared-pool
/// options and validate the result.
pub(super) fn managed_kv_participant_config(
    control_endpoint: &str,
    participant_base: &str,
    model_fingerprint: &str,
    mode: kapsl_kv_abi::KvParticipantMode,
    device_ids: &[usize],
    lease_ttl_ms: u64,
) -> Result<kapsl_kv_abi::KvParticipantConfig, String> {
    let mut memory_domains = Vec::with_capacity(device_ids.len());
    let mut rank_device_map = std::collections::BTreeMap::new();
    for (rank, device_id) in device_ids.iter().enumerate() {
        let device_id = u32::try_from(*device_id)
            .map_err(|_| format!("CUDA device {device_id} does not fit the KV contract"))?;
        memory_domains.push(kapsl_kv_abi::KvMemoryDomain::Cuda { device_id });
        rank_device_map.insert(rank.to_string(), device_id);
    }
    Ok(kapsl_kv_abi::KvParticipantConfig {
        control_endpoint: control_endpoint.to_string(),
        participant_id: participant_base.to_string(),
        model_fingerprint: model_fingerprint.to_string(),
        mode,
        memory_domains,
        rank_device_map,
        lease_ttl_ms,
        timeout_ms: None,
        live_resize: false,
        vmm_conformance: false,
        provisioning_grant: None,
    })
}

/// Runtime-wide state shared by every replica of one managed server backend.
pub(crate) struct ManagedServerDeployment {
    backend: ResolvedServingBackend,
    name: &'static str,
    pub(super) python: PathBuf,
    pub(super) control_endpoint: String,
    pub(super) runtime_root: PathBuf,
    pub(super) lease_ttl_ms: u64,
    runtimes: Mutex<HashMap<u32, Vec<Weak<ManagedServerRuntime>>>>,
    coordinator: RwLock<Option<Arc<ExternalKvCoordinator>>>,
}

/// Deployment plus the KV control socket it expects the runtime to serve.
pub(crate) struct PreparedManagedServerDeployment {
    pub(crate) deployment: Arc<ManagedServerDeployment>,
    pub(crate) control_socket: PathBuf,
}

impl ManagedServerDeployment {
    /// Create the private runtime directory and settle the control socket.
    ///
    /// `requested_control_socket` is the operator's socket, or the one another
    /// managed backend already chose: every backend shares one coordinator.
    pub(super) fn prepare(
        backend: ResolvedServingBackend,
        name: &'static str,
        python: PathBuf,
        state_dir: Option<&Path>,
        requested_control_socket: Option<&Path>,
        lease_ttl_ms: u64,
    ) -> Result<PreparedManagedServerDeployment, String> {
        let directory = format!("managed-{}", backend.as_str());
        let runtime_root = match state_dir {
            Some(state_dir) => state_dir
                .join(&directory)
                .join(std::process::id().to_string()),
            None => std::env::temp_dir().join(format!(
                "kapsl-{}-{}",
                backend.as_str(),
                std::process::id()
            )),
        };
        std::fs::create_dir_all(&runtime_root).map_err(|error| {
            format!(
                "create managed {name} runtime directory {}: {error}",
                runtime_root.display()
            )
        })?;
        #[cfg(unix)]
        std::fs::set_permissions(&runtime_root, Permissions::from_mode(0o700)).map_err(
            |error| {
                format!(
                    "secure managed {name} runtime directory {}: {error}",
                    runtime_root.display()
                )
            },
        )?;
        let control_socket = match requested_control_socket {
            Some(path) => absolute_path(path)?,
            None => runtime_root.join("kv-control.sock"),
        };
        let deployment = Arc::new(Self {
            backend,
            name,
            python,
            control_endpoint: format!("unix://{}", control_socket.display()),
            runtime_root,
            lease_ttl_ms,
            runtimes: Mutex::new(HashMap::new()),
            coordinator: RwLock::new(None),
        });
        Ok(PreparedManagedServerDeployment {
            deployment,
            control_socket,
        })
    }

    pub(crate) fn backend(&self) -> ResolvedServingBackend {
        self.backend
    }

    pub(crate) fn name(&self) -> &'static str {
        self.name
    }

    pub(crate) fn install_coordinator(
        &self,
        coordinator: Arc<ExternalKvCoordinator>,
    ) -> Result<(), String> {
        let mut installed = self.coordinator.write();
        if let Some(current) = installed.as_ref() {
            if Arc::ptr_eq(current, &coordinator) {
                return Ok(());
            }
            return Err(format!(
                "managed {} control coordinator is already installed",
                self.name
            ));
        }
        *installed = Some(coordinator);
        Ok(())
    }

    pub(super) fn coordinator(&self) -> Result<Arc<ExternalKvCoordinator>, String> {
        self.coordinator
            .read()
            .clone()
            .ok_or_else(|| format!("managed {} control coordinator is not installed", self.name))
    }

    fn retire_participants_after_backend_exit(
        &self,
        participant_base: &str,
    ) -> Result<usize, String> {
        self.coordinator()?
            .retire_participants_after_backend_exit(participant_base)
    }

    fn register_runtime(&self, model_id: u32, runtime: &Arc<ManagedServerRuntime>) {
        let mut runtimes = self.runtimes.lock();
        let model_runtimes = runtimes.entry(model_id).or_default();
        model_runtimes.retain(|candidate| candidate.strong_count() > 0);
        model_runtimes.push(Arc::downgrade(runtime));
    }

    /// Stop every replica of a logical model at the lifecycle boundary and
    /// retire its participants, without waiting for scheduler handles to drop.
    pub(crate) fn shutdown_model(&self, model_id: u32) -> Result<usize, String> {
        let runtimes = self
            .runtimes
            .lock()
            .remove(&model_id)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|runtime| runtime.upgrade())
            .collect::<Vec<_>>();

        let mut failures = Vec::new();
        for runtime in &runtimes {
            if !runtime.shutdown(self) {
                failures.push(runtime.process.spec.participant_base.clone());
            }
        }
        if failures.is_empty() {
            return Ok(runtimes.len());
        }

        self.runtimes
            .lock()
            .entry(model_id)
            .or_default()
            .extend(runtimes.iter().map(Arc::downgrade));
        Err(format!(
            "managed {} teardown could not prove backend exit for participant(s): {}",
            self.name,
            failures.join(", ")
        ))
    }

    /// Fence every managed process group before the core runtime exits.
    pub(crate) fn shutdown_all(&self) -> Result<usize, String> {
        let model_ids = self.runtimes.lock().keys().copied().collect::<Vec<_>>();
        let mut stopped = 0usize;
        let mut failures = Vec::new();
        for model_id in model_ids {
            match self.shutdown_model(model_id) {
                Ok(count) => stopped = stopped.saturating_add(count),
                Err(error) => failures.push(format!("model {model_id}: {error}")),
            }
        }
        if failures.is_empty() {
            Ok(stopped)
        } else {
            Err(failures.join("; "))
        }
    }
}

/// Identity and limits of one supervised server process.
#[derive(Clone, Debug)]
pub(super) struct ManagedServerSpec {
    pub(super) served_model_name: String,
    /// Base of the KV participant ID the server registers under.
    pub(super) participant_base: String,
    /// Private `http://127.0.0.1:<port>` origin of the server.
    pub(super) endpoint: String,
    pub(super) log_path: PathBuf,
    pub(super) startup_timeout: Duration,
    pub(super) target_concurrency: usize,
}

pub(super) struct ManagedServerProcess {
    driver: Box<dyn ManagedServerDriver>,
    spec: ManagedServerSpec,
    bridge: ManagedVllmHttpBridge,
    lifecycle: Mutex<()>,
    child: Mutex<Option<Child>>,
    shutdown: AtomicBool,
    restarts: AtomicU32,
    readiness: AtomicU8,
    readiness_epoch: AtomicU64,
    kv_readiness_fence: Arc<ManagedVllmKvReadinessFence>,
    published_kv_readiness_epoch: AtomicU64,
    /// Participant registration that made the current process generation
    /// routable; zero until the first publication of a generation.
    published_kv_revision: AtomicU64,
    generation: AtomicU64,
}

impl ManagedServerProcess {
    pub(super) fn new(
        driver: Box<dyn ManagedServerDriver>,
        spec: ManagedServerSpec,
    ) -> Result<Self, String> {
        let bridge = ManagedVllmHttpBridge::new(&spec.endpoint).map_err(|error| {
            format!(
                "managed {} endpoint {} is not usable: {error}",
                driver.name(),
                spec.endpoint
            )
        })?;
        Ok(Self {
            driver,
            spec,
            bridge,
            lifecycle: Mutex::new(()),
            child: Mutex::new(None),
            shutdown: AtomicBool::new(false),
            restarts: AtomicU32::new(0),
            readiness: AtomicU8::new(ManagedVllmReplicaState::Planned as u8),
            readiness_epoch: AtomicU64::new(0),
            kv_readiness_fence: Arc::new(ManagedVllmKvReadinessFence::new()),
            published_kv_readiness_epoch: AtomicU64::new(0),
            published_kv_revision: AtomicU64::new(0),
            generation: AtomicU64::new(0),
        })
    }

    fn name(&self) -> &'static str {
        self.driver.name()
    }

    fn spawn_child(&self) -> Result<(), EngineError> {
        // Serialize spawn with termination, so terminate() cannot observe no
        // child while Command::spawn is in progress and leave it running.
        let _lifecycle = self.lifecycle.lock();
        if self.shutdown.load(Ordering::Acquire) {
            return Err(EngineError::backend(format!(
                "managed {} process has been shut down",
                self.name()
            )));
        }
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.published_kv_revision.store(0, Ordering::Release);
        self.readiness
            .store(ManagedVllmReplicaState::Starting as u8, Ordering::Release);
        self.readiness_epoch.fetch_add(1, Ordering::AcqRel);
        let mut command = match self.driver.command() {
            Ok(command) => command,
            Err(error) => {
                self.mark_suspect_locked();
                return Err(error);
            }
        };
        #[cfg(unix)]
        command.process_group(0);
        let child = match crate::observability::spawn_logged_child(
            &mut command,
            &self.spec.log_path,
            self.driver.log_label(),
        ) {
            Ok(child) => child,
            Err(error) => {
                self.mark_suspect_locked();
                return Err(EngineError::backend(format!(
                    "launch managed {} with {}: {error}",
                    self.name(),
                    command.get_program().to_string_lossy()
                )));
            }
        };
        log::info!(
            "Managed {} process started: pid={} endpoint={} log={}",
            self.name(),
            child.id(),
            self.spec.endpoint,
            self.spec.log_path.display()
        );
        *self.child.lock() = Some(child);
        Ok(())
    }

    fn state(&self) -> ManagedVllmReplicaState {
        ManagedVllmReplicaState::from_u8(self.readiness.load(Ordering::Acquire))
    }

    fn mark_suspect(&self) {
        let _lifecycle = self.lifecycle.lock();
        self.mark_suspect_locked();
    }

    fn mark_suspect_locked(&self) {
        if self.state() != ManagedVllmReplicaState::Stopped {
            self.readiness
                .store(ManagedVllmReplicaState::Suspect as u8, Ordering::Release);
            // Increment even for Suspect -> Suspect so a transport failure
            // that races an in-flight health probe fences that stale result.
            self.readiness_epoch.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn readiness_snapshot(&self) -> (u64, u64, u64) {
        let _lifecycle = self.lifecycle.lock();
        (
            self.generation.load(Ordering::Acquire),
            self.readiness_epoch.load(Ordering::Acquire),
            self.kv_readiness_fence.snapshot(),
        )
    }

    /// Publish Routable for the lifecycle revision a probe observed.
    ///
    /// A generation stays bound to the first participant registration that
    /// made it routable. A different registration under the same generation
    /// means a second process registered without a supervised restart.
    fn mark_routable(
        &self,
        generation: u64,
        readiness_epoch: u64,
        kv_readiness_epoch: u64,
        kv_revision: u64,
    ) -> bool {
        let _lifecycle = self.lifecycle.lock();
        if self.shutdown.load(Ordering::Acquire)
            || self.generation.load(Ordering::Acquire) != generation
            || self.readiness_epoch.load(Ordering::Acquire) != readiness_epoch
            || self.kv_readiness_fence.snapshot() != kv_readiness_epoch
        {
            return false;
        }
        let published_revision = self.published_kv_revision.load(Ordering::Acquire);
        if published_revision != 0 && published_revision != kv_revision {
            self.mark_suspect_locked();
            return false;
        }
        match self.state() {
            ManagedVllmReplicaState::Starting
            | ManagedVllmReplicaState::Suspect
            | ManagedVllmReplicaState::Routable => {
                if self.state() == ManagedVllmReplicaState::Routable
                    && self.published_kv_readiness_epoch.load(Ordering::Acquire)
                        == kv_readiness_epoch
                {
                    return true;
                }
                self.published_kv_revision
                    .store(kv_revision, Ordering::Release);
                self.published_kv_readiness_epoch
                    .store(kv_readiness_epoch, Ordering::Release);
                self.readiness
                    .store(ManagedVllmReplicaState::Routable as u8, Ordering::Release);
                // Retirement advances the shared epoch before it changes
                // coordinator state. Recheck after publication so one that
                // overlaps this critical section cannot leave Routable usable.
                if self.kv_readiness_fence.snapshot() != kv_readiness_epoch {
                    self.mark_suspect_locked();
                    return false;
                }
                self.readiness_epoch.fetch_add(1, Ordering::AcqRel);
                true
            }
            ManagedVllmReplicaState::Planned | ManagedVllmReplicaState::Stopped => false,
        }
    }

    fn ensure_routable(&self) -> Result<(), EngineError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(EngineError::backend(format!(
                "managed {} process has been shut down",
                self.name()
            )));
        }
        let state = self.state();
        let published_kv_epoch = self.published_kv_readiness_epoch.load(Ordering::Acquire);
        let current_kv_epoch = self.kv_readiness_fence.snapshot();
        if state != ManagedVllmReplicaState::Routable || published_kv_epoch != current_kv_epoch {
            if state == ManagedVllmReplicaState::Routable && published_kv_epoch != current_kv_epoch
            {
                self.mark_suspect();
            }
            return Err(EngineError::backend(format!(
                "managed {} replica is not routable (state={:?}, generation={})",
                self.name(),
                self.state(),
                self.generation.load(Ordering::Acquire),
            )));
        }
        if let Some(status) = self.try_wait()? {
            self.mark_suspect();
            return Err(EngineError::backend(format!(
                "managed {} process exited with {status}; log: {}",
                self.name(),
                self.spec.log_path.display()
            )));
        }
        Ok(())
    }

    fn try_wait(&self) -> Result<Option<ExitStatus>, EngineError> {
        let mut child = self.child.lock();
        let Some(child) = child.as_mut() else {
            return Err(EngineError::backend(format!(
                "managed {} process has not been started",
                self.name()
            )));
        };
        child.try_wait().map_err(|error| {
            EngineError::backend(format!("inspect managed {} process: {error}", self.name()))
        })
    }

    async fn probe_health(&self) -> Result<(), EngineError> {
        if let Some(status) = self.try_wait()? {
            self.mark_suspect();
            return Err(EngineError::backend(format!(
                "managed {} process exited with {status}; log: {}",
                self.name(),
                self.spec.log_path.display()
            )));
        }
        self.bridge
            .check_health(
                self.driver.health_path(),
                ManagedVllmRequestTimeouts::new(HEALTH_TIMEOUT, HEALTH_TIMEOUT, HEALTH_TIMEOUT),
                None,
            )
            .await
            .map_err(|error| self.bridge_error(error, "health check"))
    }

    fn kv_status(
        &self,
        deployment: &ManagedServerDeployment,
    ) -> Result<Option<ManagedKvStatus>, String> {
        let coordinator = deployment.coordinator()?;
        self.driver
            .kv_status(&coordinator, &self.spec.participant_base)
    }

    async fn wait_ready(&self, deployment: &ManagedServerDeployment) -> Result<(), EngineError> {
        let deadline = Instant::now() + self.spec.startup_timeout;
        let generation = self.generation.load(Ordering::Acquire);
        loop {
            if let Some(status) = self.try_wait()? {
                self.mark_suspect();
                return Err(EngineError::backend(format!(
                    "managed {} exited before readiness with {status}; log: {}",
                    self.name(),
                    self.spec.log_path.display()
                )));
            }
            let (probe_generation, readiness_epoch, kv_readiness_epoch) = self.readiness_snapshot();
            if probe_generation != generation {
                return Err(EngineError::backend(format!(
                    "managed {} readiness observed a different process generation",
                    self.name()
                )));
            }
            if self.probe_health().await.is_ok() {
                let status = self.kv_status(deployment).map_err(|error| {
                    self.mark_suspect();
                    EngineError::backend(format!(
                        "managed {} KV participant is not acceptable: {error}; log: {}",
                        self.name(),
                        self.spec.log_path.display()
                    ))
                })?;
                if let Some(status) = status {
                    if self.mark_routable(
                        generation,
                        readiness_epoch,
                        kv_readiness_epoch,
                        status.revision,
                    ) {
                        return Ok(());
                    }
                    // A concurrent lifecycle transition fenced this probe. A
                    // shutdown or new generation is terminal; a same-generation
                    // suspect transition gets a fresh probe.
                    if self.shutdown.load(Ordering::Acquire)
                        || self.generation.load(Ordering::Acquire) != generation
                    {
                        return Err(EngineError::backend(format!(
                            "managed {} readiness completed for a stale process generation",
                            self.name()
                        )));
                    }
                }
            }
            if Instant::now() >= deadline {
                self.mark_suspect();
                return Err(EngineError::timeout(format!(
                    "timed out after {}s waiting for managed {} to answer {} and register its KV participant; log: {}",
                    self.spec.startup_timeout.as_secs(),
                    self.name(),
                    self.driver.health_path(),
                    self.spec.log_path.display()
                )));
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    /// Stop the whole process group. `false` means it could not be proved
    /// dead, so its KV participant must not be retired.
    fn stop_child(&self) -> bool {
        self.mark_suspect();
        let mut child_guard = self.child.lock();
        let Some(child) = child_guard.as_mut() else {
            return true;
        };
        #[cfg(unix)]
        {
            let process_group = child.id() as i32;
            let _ = child.try_wait();
            if !process_group_alive(process_group) {
                return true;
            }
            unsafe {
                let _ = libc::kill(-process_group, libc::SIGTERM);
            }

            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                let _ = child.try_wait();
                if !process_group_alive(process_group) {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            unsafe {
                let _ = libc::kill(-process_group, libc::SIGKILL);
            }
            let _ = child.kill();
            let _ = child.wait();

            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline {
                if !process_group_alive(process_group) {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            log::error!(
                "managed {} process group {} still exists after forced termination; retaining its KV participant",
                self.name(),
                process_group,
            );
            false
        }
        #[cfg(not(unix))]
        {
            if child.try_wait().ok().flatten().is_some() {
                return true;
            }
            let _ = child.kill();
            let _ = child.wait();
            true
        }
    }

    fn terminate(&self) -> bool {
        {
            let _lifecycle = self.lifecycle.lock();
            self.shutdown.store(true, Ordering::Release);
            self.readiness
                .store(ManagedVllmReplicaState::Stopped as u8, Ordering::Release);
            self.readiness_epoch.fetch_add(1, Ordering::AcqRel);
        }
        self.stop_child()
    }

    fn bridge_error(&self, error: ManagedVllmBridgeError, action: &str) -> EngineError {
        managed_server_bridge_error(self.name(), action, error)
    }

    /// Map a bridge failure of a client request, fencing the replica when the
    /// transport itself failed.
    fn request_error(&self, error: ManagedVllmBridgeError, action: &str) -> EngineError {
        if bridge_error_is_transport_failure(&error) {
            self.mark_suspect();
        }
        self.bridge_error(error, action)
    }
}

impl Drop for ManagedServerProcess {
    fn drop(&mut self) {
        let _ = self.terminate();
    }
}

/// Describe a bridge failure without the bridge's own vLLM wording.
fn managed_server_bridge_error(
    name: &str,
    action: &str,
    error: ManagedVllmBridgeError,
) -> EngineError {
    let context = format!("managed {name} {action}");
    match error {
        ManagedVllmBridgeError::Cancelled => {
            EngineError::cancelled(format!("{context}: request was cancelled"))
        }
        ManagedVllmBridgeError::HeaderTimeout => {
            EngineError::timeout(format!("{context}: response headers timed out"))
        }
        ManagedVllmBridgeError::IdleBodyTimeout => {
            EngineError::timeout(format!("{context}: response body became idle"))
        }
        ManagedVllmBridgeError::TotalTimeout => {
            EngineError::timeout(format!("{context}: request timed out"))
        }
        ManagedVllmBridgeError::UpstreamStatus { status, body } => EngineError::backend(format!(
            "{context}: upstream HTTP {status}: {}",
            String::from_utf8_lossy(&body),
        )),
        ManagedVllmBridgeError::SseBufferExceeded { limit } => EngineError::backend(format!(
            "{context}: SSE event exceeded the {limit}-byte buffer limit"
        )),
        ManagedVllmBridgeError::ResponseBodyExceeded { limit } => EngineError::backend(format!(
            "{context}: response body exceeded the {limit}-byte buffer limit"
        )),
        error => EngineError::backend(format!("{context}: {error}")),
    }
}

/// Keep one managed server routable: probe it, and restart it under the
/// process-group and participant fences when it exits or stops answering.
fn start_managed_server_supervisor(
    process: Arc<ManagedServerProcess>,
    deployment: Arc<ManagedServerDeployment>,
) {
    tokio::spawn(async move {
        let name = process.name();
        let model = process.spec.served_model_name.clone();
        let participant_base = process.spec.participant_base.clone();
        let mut consecutive_failures = 0u8;
        loop {
            tokio::time::sleep(SUPERVISOR_INTERVAL).await;
            if process.shutdown.load(Ordering::Acquire) {
                break;
            }

            let exited = matches!(process.try_wait(), Ok(Some(_)) | Err(_));
            if !exited {
                // Bind the probe to the lifecycle revision observed before
                // dispatch, so a response that overlaps a transport failure
                // cannot restore Routable on a newer or suspect revision.
                let (generation, readiness_epoch, kv_readiness_epoch) =
                    process.readiness_snapshot();
                if process.probe_health().await.is_ok() {
                    match process.kv_status(&deployment) {
                        Ok(Some(status)) => {
                            if process.mark_routable(
                                generation,
                                readiness_epoch,
                                kv_readiness_epoch,
                                status.revision,
                            ) {
                                consecutive_failures = 0;
                                continue;
                            }
                            let published = process.published_kv_revision.load(Ordering::Acquire);
                            if published == 0 || published == status.revision {
                                // A concurrent lifecycle transition fenced this
                                // probe; the next one decides.
                                continue;
                            }
                            log::error!(
                                "[managed-supervisor] {name} model {model} registered KV participant revision {} while revision {published} was routable; restarting",
                                status.revision,
                            );
                            consecutive_failures = MAX_CONSECUTIVE_HEALTH_FAILURES;
                        }
                        Ok(None) => {}
                        Err(error) => {
                            log::error!(
                                "[managed-supervisor] {name} model {model} KV participant is not acceptable; restarting: {error}"
                            );
                            consecutive_failures = MAX_CONSECUTIVE_HEALTH_FAILURES;
                        }
                    }
                }
                process.mark_suspect();
                consecutive_failures = consecutive_failures.saturating_add(1);
                if consecutive_failures < MAX_CONSECUTIVE_HEALTH_FAILURES {
                    continue;
                }
                log::error!(
                    "[managed-supervisor] {name} endpoint {} failed its HTTP/KV readiness checks",
                    process.spec.endpoint,
                );
            }

            // A restart is a lifecycle boundary: reap the whole old process
            // group, then retire its participant, before a new process can
            // register. The fence may wait through TERM/KILL grace periods, so
            // it runs off the async workers.
            let stopped = match tokio::task::spawn_blocking({
                let process = process.clone();
                move || process.stop_child()
            })
            .await
            {
                Ok(stopped) => stopped,
                Err(error) => {
                    log::error!(
                        "[managed-supervisor] {name} model {model} process-group fence task failed; restart is disabled: {error}"
                    );
                    break;
                }
            };
            if !stopped {
                log::error!(
                    "[managed-supervisor] {name} model {model} process group could not be reaped; restart is disabled"
                );
                break;
            }
            if let Err(error) = deployment.retire_participants_after_backend_exit(&participant_base)
            {
                log::error!(
                    "[managed-supervisor] {name} model {model} participant retirement failed; restart is disabled: {error}"
                );
                break;
            }

            if process.restarts.load(Ordering::Acquire) >= MAX_RESTARTS {
                log::error!(
                    "[managed-supervisor] {name} model {model} exceeded {MAX_RESTARTS} restarts; giving up (log: {})",
                    process.spec.log_path.display()
                );
                break;
            }
            let attempt = process.restarts.fetch_add(1, Ordering::AcqRel) + 1;
            log::warn!(
                "[managed-supervisor] restarting {name} model {model} (attempt {attempt}/{MAX_RESTARTS})"
            );
            tokio::time::sleep(SUPERVISOR_BACKOFF).await;
            if process.shutdown.load(Ordering::Acquire) {
                break;
            }
            match process.spawn_child() {
                Ok(()) => match process.wait_ready(&deployment).await {
                    Ok(()) => {
                        consecutive_failures = 0;
                        log::info!(
                            "[managed-supervisor] {name} model {model} restarted successfully"
                        );
                    }
                    Err(error) => log::error!(
                        "[managed-supervisor] {name} model {model} did not become ready: {error}"
                    ),
                },
                Err(error) => log::error!(
                    "[managed-supervisor] failed to restart {name} model {model}: {error}"
                ),
            }
        }
    });
}

pub(super) struct ManagedServerRuntime {
    process: Arc<ManagedServerProcess>,
    participant_retired: Mutex<bool>,
}

impl ManagedServerRuntime {
    fn shutdown(&self, deployment: &ManagedServerDeployment) -> bool {
        if !self.process.terminate() {
            return false;
        }

        let mut retired = self.participant_retired.lock();
        if *retired {
            return true;
        }
        let participant_base = &self.process.spec.participant_base;
        match deployment.retire_participants_after_backend_exit(participant_base) {
            Ok(retired_count) => {
                *retired = true;
                if retired_count > 0 {
                    log::info!(
                        "Retired {retired_count} managed {} KV participant(s) under {participant_base} after backend exit",
                        self.process.name(),
                    );
                }
                true
            }
            Err(error) => {
                log::error!(
                    "failed to retire managed {} KV participant {participant_base}: {error}",
                    self.process.name(),
                );
                false
            }
        }
    }
}

/// Engine facade over one supervised OpenAI-compatible server.
pub(super) struct ManagedServerEngine {
    deployment: Arc<ManagedServerDeployment>,
    process: Arc<ManagedServerProcess>,
    runtime: Arc<ManagedServerRuntime>,
    memory_report: MemoryReport,
    loaded: AtomicBool,
    requests: Arc<AtomicU64>,
    errors: Arc<AtomicU64>,
    generated_tokens: Arc<AtomicU64>,
}

impl ManagedServerEngine {
    /// Bind a prepared process to its deployment. `memory_report` is the
    /// static footprint admitted for the server at model load.
    pub(super) fn create(
        deployment: Arc<ManagedServerDeployment>,
        process: ManagedServerProcess,
        model_id: u32,
        memory_report: MemoryReport,
    ) -> Result<Box<dyn Engine>, String> {
        let process = Arc::new(process);
        deployment.coordinator()?.register_managed_readiness_fence(
            &process.spec.participant_base,
            Arc::downgrade(&process.kv_readiness_fence),
        )?;
        let runtime = Arc::new(ManagedServerRuntime {
            process: process.clone(),
            participant_retired: Mutex::new(false),
        });
        deployment.register_runtime(model_id, &runtime);
        Ok(Box::new(Self {
            deployment,
            process,
            runtime,
            memory_report,
            loaded: AtomicBool::new(false),
            requests: Arc::new(AtomicU64::new(0)),
            errors: Arc::new(AtomicU64::new(0)),
            generated_tokens: Arc::new(AtomicU64::new(0)),
        }))
    }

    fn shutdown_managed_backend(&self) {
        self.loaded.store(false, Ordering::Release);
        self.runtime.shutdown(&self.deployment);
    }

    fn infer_inner(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let name = self.process.name();
        self.process.ensure_routable()?;
        if request
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(EngineError::cancelled(format!(
                "managed {name} request was cancelled before dispatch"
            )));
        }
        let (path, payload) =
            managed_server_request_payload(&self.process.spec.served_model_name, request, false)?;
        let chat = path.contains("chat");
        let mut response = self
            .process
            .bridge
            .post_json_sync(
                &path,
                payload.as_bytes(),
                managed_vllm_request_timeouts(request),
            )
            .map_err(|error| self.process.request_error(error, "request dispatch"))?;
        let status = response.status();
        let body = response.body_mut().read_to_string().map_err(|error| {
            self.process
                .request_error(map_ureq_body_error(error), "response read")
        })?;
        if !status.is_success() {
            return Err(EngineError::backend(format!(
                "managed {name} returned upstream HTTP {status}: {body}"
            )));
        }
        let value: serde_json::Value = serde_json::from_str(&body).map_err(|error| {
            EngineError::backend(format!("decode managed {name} response: {error}"))
        })?;
        let text = managed_server_completion_text(name, &value, chat)?;
        self.generated_tokens
            .fetch_add(estimate_vllm_tokens(&text), Ordering::Relaxed);
        managed_server_output_packet(name, text)
    }

    fn kv_status(&self) -> Option<ManagedKvStatus> {
        self.process.kv_status(&self.deployment).ok().flatten()
    }
}

#[async_trait::async_trait]
impl Engine for ManagedServerEngine {
    fn planned_memory(&self, _path: &Path) -> Result<MemoryReport, EngineError> {
        Ok(self.memory_report.clone())
    }

    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        let name = self.process.name();
        // Nothing may be registered under this base before its first process
        // starts, so every registration seen afterwards is the child's own.
        self.deployment
            .retire_participants_after_backend_exit(&self.process.spec.participant_base)
            .map_err(|error| {
                EngineError::backend(format!(
                    "clear stale managed {name} KV participants before launch: {error}"
                ))
            })?;
        if let Err(error) = self.process.spawn_child() {
            self.shutdown_managed_backend();
            return Err(error);
        }
        if let Err(error) = self.process.wait_ready(&self.deployment).await {
            self.shutdown_managed_backend();
            return Err(error);
        }
        self.loaded.store(true, Ordering::Release);
        start_managed_server_supervisor(self.process.clone(), self.deployment.clone());
        log::info!(
            "✓ Managed {name} ready for model {} at {}",
            self.process.spec.served_model_name,
            self.process.spec.endpoint
        );
        Ok(())
    }

    fn actual_memory(&self) -> MemoryReport {
        if self.loaded.load(Ordering::Acquire) {
            self.memory_report.clone()
        } else {
            MemoryReport::default()
        }
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let result = self.infer_inner(request);
        if result.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn supports_openai_wire(&self) -> bool {
        true
    }

    async fn infer_openai_wire(
        &self,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireResponse, EngineError> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let result = async {
            if request.format != OpenAiWireFormat::Json {
                return Err(EngineError::invalid_input(format!(
                    "managed {} unary wire inference requires JSON format",
                    self.process.name()
                )));
            }
            request.validate(usize::MAX)?;
            self.process.ensure_routable()?;
            let response = self
                .process
                .bridge
                .post_json_buffered(
                    request.endpoint.path(),
                    request.body.clone(),
                    managed_vllm_wire_request_timeouts(request),
                    request.cancellation.clone(),
                    MAX_OPENAI_WIRE_RESPONSE_BYTES,
                )
                .await
                .map_err(|error| self.process.request_error(error, "wire request dispatch"))?;
            if !response.status.is_success() {
                self.errors.fetch_add(1, Ordering::Relaxed);
            }
            Ok(OpenAiWireResponse {
                head: managed_vllm_wire_response_head(response.status, &response.headers)?,
                body: response.body,
            })
        }
        .await;
        if result.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    async fn infer_openai_wire_stream(
        &self,
        request: &OpenAiWireRequest,
    ) -> Result<OpenAiWireStreamResponse, EngineError> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let result = async {
            if request.format != OpenAiWireFormat::ServerSentEvents {
                return Err(EngineError::invalid_input(format!(
                    "managed {} streaming wire inference requires SSE format",
                    self.process.name()
                )));
            }
            request.validate(usize::MAX)?;
            self.process.ensure_routable()?;
            let response = self
                .process
                .bridge
                .post_json_raw(
                    request.endpoint.path(),
                    request.body.clone(),
                    managed_vllm_wire_request_timeouts(request),
                    request.cancellation.clone(),
                )
                .await
                .map_err(|error| self.process.request_error(error, "wire stream dispatch"))?;
            if !response.status.is_success() {
                self.errors.fetch_add(1, Ordering::Relaxed);
            }
            Ok(OpenAiWireStreamResponse {
                head: managed_vllm_wire_response_head(response.status, &response.headers)?,
                body: relay_managed_server_wire_stream(
                    response.body,
                    self.process.clone(),
                    self.errors.clone(),
                ),
            })
        }
        .await;
        if result.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn self_batches(&self) -> bool {
        true
    }

    fn batching_policy(&self) -> BatchingPolicy {
        let mut policy = BatchingPolicy::delegated();
        policy.max_requests = self.process.spec.target_concurrency.max(1);
        policy
    }

    fn infer_stream(&self, request: &InferenceRequest) -> EngineStream {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let (path, payload) = match managed_server_request_payload(
            &self.process.spec.served_model_name,
            request,
            true,
        ) {
            Ok(request) => request,
            Err(error) => {
                self.errors.fetch_add(1, Ordering::Relaxed);
                return Box::pin(stream::once(async move { Err(error) }));
            }
        };
        let chat = path.contains("chat");
        let process = self.process.clone();
        let cancellation = request.cancellation.clone();
        let timeouts = managed_vllm_request_timeouts(request);
        let errors = self.errors.clone();
        let generated_tokens = self.generated_tokens.clone();
        let start = async move {
            if let Err(error) = process.ensure_routable() {
                errors.fetch_add(1, Ordering::Relaxed);
                return Box::pin(stream::once(async move { Err(error) })) as EngineStream;
            }
            let response = match process
                .bridge
                .post_json_sse(&path, payload.into_bytes(), timeouts, cancellation)
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    errors.fetch_add(1, Ordering::Relaxed);
                    let error = process.request_error(error, "stream dispatch");
                    return Box::pin(stream::once(async move { Err(error) })) as EngineStream;
                }
            };
            translate_managed_server_stream(
                response.events,
                process,
                errors,
                generated_tokens,
                chat,
            )
        };
        Box::pin(stream::once(start).flatten())
    }

    fn unload(&mut self) {
        self.shutdown_managed_backend();
    }

    fn metrics(&self) -> EngineMetrics {
        let requests = self.requests.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);
        let kv = self.kv_status();
        let memory_usage = self
            .actual_memory()
            .allocations
            .iter()
            .map(|allocation| allocation.bytes)
            .fold(0_usize, usize::saturating_add);
        EngineMetrics {
            memory_usage,
            batch_size: self.process.spec.target_concurrency.max(1),
            error_rate: if requests == 0 {
                0.0
            } else {
                errors as f64 / requests as f64
            },
            kv_cache_bytes_used: kv.map_or(0, |status| metric_usize(status.leased_bytes)),
            kv_cache_bytes_capacity: kv.map_or(0, |status| metric_usize(status.capacity_bytes)),
            kv_cache_sequences: kv.map_or(0, |status| metric_usize(status.active_leases)),
            generated_tokens_total: self.generated_tokens.load(Ordering::Relaxed),
            ..Default::default()
        }
    }

    fn health_check(&self) -> Result<(), EngineError> {
        self.process.ensure_routable()
    }
}

impl Drop for ManagedServerEngine {
    fn drop(&mut self) {
        self.shutdown_managed_backend();
    }
}

/// Translate a Kapsl inference request into an OpenAI completion or chat
/// request. A chat request arrives as the JSON envelope the OpenAI route
/// builds for managed servers.
fn managed_server_request_payload(
    served_model_name: &str,
    request: &InferenceRequest,
    stream: bool,
) -> Result<(String, String), EngineError> {
    if request.input.dtype != TensorDtype::Utf8 {
        return Err(EngineError::invalid_input(format!(
            "a managed server expects a UTF-8 prompt tensor, got {}",
            request.input.dtype
        )));
    }
    let prompt = std::str::from_utf8(&request.input.data)
        .map_err(|error| EngineError::invalid_input(format!("prompt is not UTF-8: {error}")))?;
    let envelope = serde_json::from_str::<serde_json::Value>(prompt).ok();
    let messages = envelope.as_ref().and_then(|value| {
        value
            .get(MANAGED_VLLM_CHAT_MARKER)
            .and_then(serde_json::Value::as_bool)
            .filter(|enabled| *enabled)
            .and_then(|_| value.get("messages"))
            .cloned()
    });
    let chat = messages.is_some();
    let mut payload = match messages {
        Some(messages) => serde_json::json!({
            "model": served_model_name,
            "messages": messages,
            "stream": stream,
        }),
        None => serde_json::json!({
            "model": served_model_name,
            "prompt": prompt,
            "stream": stream,
        }),
    };
    let endpoint_path = if chat {
        if let Some(stop) = envelope
            .as_ref()
            .and_then(|value| value.get("stop"))
            .filter(|stop| !stop.as_array().is_some_and(Vec::is_empty))
        {
            payload["stop"] = stop.clone();
        }
        "/v1/chat/completions"
    } else {
        "/v1/completions"
    };
    let metadata = request.metadata.as_ref();
    payload["max_tokens"] = serde_json::json!(metadata
        .and_then(|value| value.max_new_tokens)
        .unwrap_or(256));
    if let Some(value) = metadata.and_then(|value| value.temperature) {
        payload["temperature"] = serde_json::json!(value);
    }
    if let Some(value) = metadata.and_then(|value| value.top_p) {
        payload["top_p"] = serde_json::json!(value);
    }
    if let Some(value) = metadata.and_then(|value| value.top_k) {
        payload["top_k"] = serde_json::json!(value);
    }
    if let Some(value) = metadata.and_then(|value| value.repetition_penalty) {
        payload["repetition_penalty"] = serde_json::json!(value);
    }
    if let Some(value) = metadata.and_then(|value| value.seed) {
        payload["seed"] = serde_json::json!(value);
    }
    if let Some(value) = metadata.and_then(|value| value.stop_token_ids.as_ref()) {
        payload["stop_token_ids"] = serde_json::json!(value);
    }
    serde_json::to_string(&payload)
        .map(|body| (endpoint_path.to_string(), body))
        .map_err(|error| EngineError::backend(format!("serialize managed server request: {error}")))
}

fn managed_server_output_packet(
    name: &str,
    text: String,
) -> Result<BinaryTensorPacket, EngineError> {
    if text.is_empty() {
        return Err(EngineError::backend(format!(
            "managed {name} returned an empty completion"
        )));
    }
    let data = text.into_bytes();
    BinaryTensorPacket::new(vec![1, data.len() as i64], TensorDtype::Utf8, data)
}

fn managed_server_completion_text(
    name: &str,
    value: &serde_json::Value,
    chat: bool,
) -> Result<String, EngineError> {
    let choice = value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| EngineError::backend(format!("managed {name} response has no choices")))?;
    let text = if chat {
        choice
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(serde_json::Value::as_str)
    } else {
        choice.get("text").and_then(serde_json::Value::as_str)
    };
    text.map(str::to_string).ok_or_else(|| {
        EngineError::backend(format!("managed {name} response has no textual completion"))
    })
}

fn relay_managed_server_wire_stream(
    upstream: ManagedVllmByteStream,
    process: Arc<ManagedServerProcess>,
    errors: Arc<AtomicU64>,
) -> OpenAiWireStream {
    Box::pin(upstream.map(move |item| {
        item.map_err(|error| {
            errors.fetch_add(1, Ordering::Relaxed);
            process.request_error(error, "wire stream read")
        })
    }))
}

struct ManagedServerTranslatedStream {
    upstream: ManagedVllmSseStream,
    process: Arc<ManagedServerProcess>,
    errors: Arc<AtomicU64>,
    generated_tokens: Arc<AtomicU64>,
    chat: bool,
    terminal: bool,
}

impl ManagedServerTranslatedStream {
    fn fail(
        mut self,
        error: EngineError,
    ) -> Option<(Result<BinaryTensorPacket, EngineError>, Self)> {
        self.errors.fetch_add(1, Ordering::Relaxed);
        self.terminal = true;
        Some((Err(error), self))
    }
}

/// Turn an OpenAI SSE completion stream into Kapsl text packets.
fn translate_managed_server_stream(
    upstream: ManagedVllmSseStream,
    process: Arc<ManagedServerProcess>,
    errors: Arc<AtomicU64>,
    generated_tokens: Arc<AtomicU64>,
    chat: bool,
) -> EngineStream {
    let state = ManagedServerTranslatedStream {
        upstream,
        process,
        errors,
        generated_tokens,
        chat,
        terminal: false,
    };
    Box::pin(stream::unfold(state, |mut state| async move {
        if state.terminal {
            return None;
        }
        let name = state.process.name();
        loop {
            let data = match state.upstream.next().await {
                Some(Ok(data)) => data,
                Some(Err(error)) => {
                    let error = state.process.request_error(error, "stream read");
                    return state.fail(error);
                }
                None => {
                    return state.fail(EngineError::backend(format!(
                        "managed {name} stream ended before the terminal [DONE] event"
                    )));
                }
            };
            if data == b"[DONE]" {
                return None;
            }
            if data.is_empty() {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_slice(&data) {
                Ok(value) => value,
                Err(error) => {
                    return state.fail(EngineError::backend(format!(
                        "decode managed {name} stream event: {error}"
                    )));
                }
            };
            if let Some(upstream_error) = value.get("error") {
                let error = EngineError::backend(format!(
                    "managed {name} stream returned an upstream error: {upstream_error}"
                ));
                return state.fail(error);
            }
            let choice = value
                .get("choices")
                .and_then(serde_json::Value::as_array)
                .and_then(|choices| choices.first());
            let text = if state.chat {
                choice
                    .and_then(|choice| choice.get("delta"))
                    .and_then(|delta| delta.get("content"))
                    .and_then(serde_json::Value::as_str)
            } else {
                choice
                    .and_then(|choice| choice.get("text"))
                    .and_then(serde_json::Value::as_str)
            };
            let Some(text) = text.filter(|text| !text.is_empty()) else {
                continue;
            };
            state
                .generated_tokens
                .fetch_add(estimate_vllm_tokens(text), Ordering::Relaxed);
            return Some((managed_server_output_packet(name, text.to_string()), state));
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDriver;

    impl ManagedServerDriver for TestDriver {
        fn name(&self) -> &'static str {
            "TestServer"
        }

        fn log_label(&self) -> &'static str {
            "test-server"
        }

        fn command(&self) -> Result<Command, EngineError> {
            Err(EngineError::backend("the test driver starts no process"))
        }

        fn kv_status(
            &self,
            _coordinator: &ExternalKvCoordinator,
            _participant_base: &str,
        ) -> Result<Option<ManagedKvStatus>, String> {
            Ok(None)
        }
    }

    fn process() -> ManagedServerProcess {
        ManagedServerProcess::new(
            Box::new(TestDriver),
            ManagedServerSpec {
                served_model_name: "model".to_string(),
                participant_base: "kapsl-model-1-0".to_string(),
                endpoint: "http://127.0.0.1:9".to_string(),
                log_path: std::env::temp_dir().join("kapsl-managed-server-test.log"),
                startup_timeout: Duration::from_secs(1),
                target_concurrency: 4,
            },
        )
        .expect("process")
    }

    /// Put a process in the state `spawn_child` leaves behind, without a child.
    fn start_generation(process: &ManagedServerProcess) -> (u64, u64, u64) {
        process.generation.fetch_add(1, Ordering::AcqRel);
        process.published_kv_revision.store(0, Ordering::Release);
        process
            .readiness
            .store(ManagedVllmReplicaState::Starting as u8, Ordering::Release);
        process.readiness_epoch.fetch_add(1, Ordering::AcqRel);
        process.readiness_snapshot()
    }

    #[test]
    fn a_failed_launch_leaves_the_replica_unroutable() {
        let process = process();
        assert_eq!(process.state(), ManagedVllmReplicaState::Planned);
        assert!(process.spawn_child().is_err());
        assert_eq!(process.state(), ManagedVllmReplicaState::Suspect);
        assert!(process.ensure_routable().is_err());
    }

    #[test]
    fn routable_is_bound_to_the_observed_lifecycle_revision() {
        let process = process();
        let (generation, readiness_epoch, kv_epoch) = start_generation(&process);

        // A transport failure after the probe was taken fences its result.
        process.mark_suspect();
        assert!(!process.mark_routable(generation, readiness_epoch, kv_epoch, 7));
        assert_eq!(process.state(), ManagedVllmReplicaState::Suspect);

        let (generation, readiness_epoch, kv_epoch) = process.readiness_snapshot();
        assert!(process.mark_routable(generation, readiness_epoch, kv_epoch, 7));
        assert_eq!(process.state(), ManagedVllmReplicaState::Routable);

        // Participant retirement advances the fence; the published state is
        // then stale for dispatch and for a probe taken before it.
        let (generation, readiness_epoch, kv_epoch) = process.readiness_snapshot();
        process.kv_readiness_fence.advance();
        assert!(!process.mark_routable(generation, readiness_epoch, kv_epoch, 7));
        assert_eq!(
            process.published_kv_readiness_epoch.load(Ordering::Acquire),
            kv_epoch
        );
        assert_ne!(process.kv_readiness_fence.snapshot(), kv_epoch);
    }

    #[test]
    fn a_generation_accepts_only_the_registration_that_made_it_routable() {
        let process = process();
        let (generation, readiness_epoch, kv_epoch) = start_generation(&process);
        assert!(process.mark_routable(generation, readiness_epoch, kv_epoch, 7));

        // The same registration keeps the replica routable.
        let (generation, readiness_epoch, kv_epoch) = process.readiness_snapshot();
        assert!(process.mark_routable(generation, readiness_epoch, kv_epoch, 7));

        // A second registration under a live generation is a second process.
        let (generation, readiness_epoch, kv_epoch) = process.readiness_snapshot();
        assert!(!process.mark_routable(generation, readiness_epoch, kv_epoch, 8));
        assert_eq!(process.state(), ManagedVllmReplicaState::Suspect);

        // A supervised restart starts a generation that accepts a new one.
        let (generation, readiness_epoch, kv_epoch) = start_generation(&process);
        assert!(process.mark_routable(generation, readiness_epoch, kv_epoch, 8));
        assert_eq!(process.state(), ManagedVllmReplicaState::Routable);
    }

    #[test]
    fn a_stopped_process_never_becomes_routable() {
        let process = process();
        let (generation, readiness_epoch, kv_epoch) = start_generation(&process);
        assert!(process.terminate());
        assert!(!process.mark_routable(generation, readiness_epoch, kv_epoch, 7));
        assert_eq!(process.state(), ManagedVllmReplicaState::Stopped);
        assert!(process.spawn_child().is_err());
        assert_eq!(process.state(), ManagedVllmReplicaState::Stopped);
    }

    /// The vLLM connector config as the runtime built it by hand before the
    /// shared participant configuration type existed.
    fn hand_built_vllm_extra_config(
        device_ids: &[usize],
        grant: Option<&kapsl_kv_abi::KvProvisioningGrant>,
        live_resize: bool,
        vmm_conformance: bool,
    ) -> serde_json::Value {
        let memory_domains = device_ids
            .iter()
            .map(|device_id| serde_json::json!({"kind": "cuda", "device_id": device_id}))
            .collect::<Vec<_>>();
        let rank_device_map = device_ids
            .iter()
            .enumerate()
            .map(|(rank, device_id)| (rank.to_string(), serde_json::json!(device_id)))
            .collect::<serde_json::Map<_, _>>();
        let mut extra = serde_json::json!({
            "kapsl_control_endpoint": "unix:///tmp/kapsl.sock",
            "kapsl_participant_id": "kapsl-model-1-0",
            "kapsl_model_fingerprint": "sha256:model",
            "kapsl_kv_mode": "shared_pool",
            "kapsl_memory_domains": memory_domains,
            "kapsl_rank_device_map": rank_device_map,
            "kapsl_lease_ttl_ms": 30_000,
            "kapsl_live_resize": live_resize,
            "kapsl_vmm_conformance": vmm_conformance,
        });
        if let Some(grant) = grant {
            extra["kapsl_provisioning_grant"] = serde_json::to_value(grant).unwrap();
        }
        extra
    }

    #[test]
    fn vllm_connector_config_is_unchanged_by_the_shared_participant_type() {
        let grant = kapsl_kv_abi::KvProvisioningGrant {
            token: "kvg1_token".to_string(),
            geometry_digest: format!("sha256:{}", "a".repeat(64)),
            authority_generation: 7,
            expires_at_unix_ms: 42,
        };
        for (device_ids, grant, live_resize, vmm_conformance) in [
            (&[0usize][..], None, false, false),
            (&[0, 2][..], None, false, false),
            (&[1][..], Some(&grant), false, false),
            (&[3, 1][..], Some(&grant), true, false),
            (&[0][..], Some(&grant), true, true),
        ] {
            let encoded = build_kv_transfer_config(
                "unix:///tmp/kapsl.sock",
                "kapsl-model-1-0",
                "sha256:model",
                device_ids,
                30_000,
                grant,
                live_resize,
                vmm_conformance,
            )
            .unwrap();
            let value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
            assert_eq!(
                value,
                serde_json::json!({
                    "kv_connector": "KapslConnectorV1",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "kapsl_vllm_connector",
                    "kv_connector_extra_config": hand_built_vllm_extra_config(
                        device_ids,
                        grant,
                        live_resize,
                        vmm_conformance,
                    ),
                })
            );
        }
    }

    fn request(prompt: &str) -> InferenceRequest {
        let data = prompt.as_bytes().to_vec();
        InferenceRequest::new(
            BinaryTensorPacket::new(vec![1, data.len() as i64], TensorDtype::Utf8, data).unwrap(),
        )
    }

    #[test]
    fn request_payload_distinguishes_prompts_from_the_chat_envelope() {
        let (path, body) =
            managed_server_request_payload("served", &request("plain prompt"), false).unwrap();
        let body: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(path, "/v1/completions");
        assert_eq!(body["model"], "served");
        assert_eq!(body["prompt"], "plain prompt");
        assert_eq!(body["stream"], false);
        assert_eq!(body["max_tokens"], 256);

        let envelope = serde_json::json!({
            MANAGED_VLLM_CHAT_MARKER: true,
            "messages": [{"role": "user", "content": "hello"}],
            "stop": ["END"],
        })
        .to_string();
        let (path, body) =
            managed_server_request_payload("served", &request(&envelope), true).unwrap();
        let body: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(path, "/v1/chat/completions");
        assert_eq!(body["messages"][0]["content"], "hello");
        assert_eq!(body["stop"][0], "END");
        assert_eq!(body["stream"], true);
        assert!(body.get("prompt").is_none());
    }

    #[test]
    fn completion_text_reads_chat_and_text_choices() {
        let chat = serde_json::json!({"choices": [{"message": {"content": "hi"}}]});
        assert_eq!(
            managed_server_completion_text("TestServer", &chat, true).unwrap(),
            "hi"
        );
        let text = serde_json::json!({"choices": [{"text": "there"}]});
        assert_eq!(
            managed_server_completion_text("TestServer", &text, false).unwrap(),
            "there"
        );
        let error = managed_server_completion_text("TestServer", &serde_json::json!({}), true)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("managed TestServer response has no choices"),
            "{error}"
        );
        assert!(managed_server_output_packet("TestServer", String::new()).is_err());
    }

    #[test]
    fn bridge_errors_are_reported_under_the_server_name() {
        for (error, expected) in [
            (
                ManagedVllmBridgeError::HeaderTimeout,
                "managed TestServer request dispatch: response headers timed out",
            ),
            (
                ManagedVllmBridgeError::TotalTimeout,
                "managed TestServer request dispatch: request timed out",
            ),
            (
                ManagedVllmBridgeError::Cancelled,
                "managed TestServer request dispatch: request was cancelled",
            ),
        ] {
            let message =
                managed_server_bridge_error("TestServer", "request dispatch", error).to_string();
            assert!(message.contains(expected), "{message}");
            assert!(!message.contains("vLLM"), "{message}");
        }
    }

    #[tokio::test]
    async fn translated_stream_yields_text_until_done_and_fails_on_truncation() {
        let events = |items: Vec<&'static str>| -> ManagedVllmSseStream {
            Box::pin(futures::stream::iter(
                items.into_iter().map(|item| Ok(item.as_bytes().to_vec())),
            ))
        };
        let run = |upstream: ManagedVllmSseStream| async move {
            let errors = Arc::new(AtomicU64::new(0));
            let items = translate_managed_server_stream(
                upstream,
                Arc::new(process()),
                errors.clone(),
                Arc::new(AtomicU64::new(0)),
                true,
            )
            .collect::<Vec<_>>()
            .await;
            (items, errors.load(Ordering::Relaxed))
        };

        let (items, errors) = run(events(vec![
            r#"{"choices":[{"delta":{"role":"assistant"}}]}"#,
            r#"{"choices":[{"delta":{"content":"hel"}}]}"#,
            r#"{"choices":[{"delta":{"content":"lo"}}]}"#,
            "[DONE]",
        ]))
        .await;
        let text = items
            .iter()
            .map(|item| match item {
                Ok(packet) => String::from_utf8(packet.data.clone()).unwrap(),
                Err(error) => panic!("unexpected stream error: {error}"),
            })
            .collect::<String>();
        assert_eq!((text.as_str(), errors), ("hello", 0));

        let (items, errors) = run(events(vec![r#"{"choices":[{"delta":{"content":"x"}}]}"#])).await;
        assert_eq!((items.len(), errors), (2, 1));
        match &items[1] {
            Err(error) => assert!(error
                .to_string()
                .contains("ended before the terminal [DONE] event")),
            Ok(_) => panic!("a truncated stream must end with an error"),
        }
    }
}
