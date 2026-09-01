//! Resolved external KV-control configuration and listener startup.

use super::*;

// Coordinator and accounting types are portable; control.rs gates the Unix
// socket transport internally so non-Unix fail-closed paths still compile.
mod control;
#[cfg(all(feature = "gpu-device-pool", any(target_os = "linux", test)))]
mod cuda_ipc;
mod shared;

pub(crate) use control::*;
#[cfg(all(feature = "gpu-device-pool", any(target_os = "linux", test)))]
pub(crate) use cuda_ipc::*;
pub(crate) use shared::*;

#[derive(Clone, Debug)]
pub(crate) struct KvControlConfig {
    pub(crate) socket_path: Option<PathBuf>,
    pub(crate) lease_ttl_ms: u64,
    pub(crate) shared_pool_profiles: Vec<String>,
}

impl KvControlConfig {
    pub(crate) fn apply_managed_defaults(&mut self, prepared: &PreparedManagedVllmDeployment) {
        self.socket_path = Some(prepared.control_socket.clone());
        for profile in &prepared.shared_pool_profiles {
            if !self
                .shared_pool_profiles
                .iter()
                .any(|candidate| candidate == profile)
            {
                self.shared_pool_profiles.push(profile.clone());
            }
        }
    }

    pub(crate) fn validate(&self, transport: &RuntimeTransportConfig) -> Result<(), DynError> {
        #[cfg(not(unix))]
        if self.socket_path.is_some() {
            return Err("--kv-control-socket currently requires a Unix host".into());
        }
        if self.socket_path.is_none() && !self.shared_pool_profiles.is_empty() {
            return Err("--kv-shared-pool-profile requires --kv-control-socket".into());
        }
        #[cfg(not(all(feature = "gpu-device-pool", target_os = "linux")))]
        if !self.shared_pool_profiles.is_empty() {
            return Err("--kv-shared-pool-profile requires a Linux gpu-device-pool build".into());
        }
        #[cfg(unix)]
        if let Some(kv_socket) = self.socket_path.as_ref() {
            if transport.mode.uses_ipc_socket() && kv_socket == Path::new(&transport.socket_path) {
                return Err(
                    "--kv-control-socket must differ from the inference --socket path".into(),
                );
            }
        }
        Ok(())
    }

    #[cfg(unix)]
    pub(crate) async fn start(
        &self,
        resources: &Arc<RuntimeResources>,
    ) -> Result<Option<tokio::task::JoinHandle<std::io::Result<()>>>, DynError> {
        let Some(socket_path) = self.socket_path.as_ref() else {
            return Ok(None);
        };

        #[cfg(all(feature = "gpu-device-pool", any(target_os = "linux", test)))]
        let coordinator = ExternalKvCoordinator::new_with_shared_pool_provisioner(
            resources.memory().clone(),
            Duration::from_millis(self.lease_ttl_ms),
            Some(CudaIpcSharedPoolProvisioner::new(
                resources.memory().clone(),
            )),
            parse_shared_pool_profiles(&self.shared_pool_profiles)?,
        )?;
        #[cfg(not(all(feature = "gpu-device-pool", any(target_os = "linux", test))))]
        let coordinator = ExternalKvCoordinator::new(
            resources.memory().clone(),
            Duration::from_millis(self.lease_ttl_ms),
        )?;

        if let Some(deployment) = resources.managed_vllm() {
            deployment.install_coordinator(coordinator.clone())?;
        }
        let control_server = KvControlServer::bind(socket_path, coordinator).await?;
        log::info!(
            "KV participant control: unix://{} (maximum lease TTL={}ms)",
            socket_path.display(),
            self.lease_ttl_ms
        );
        Ok(Some(tokio::spawn(control_server.run())))
    }

    #[cfg(not(unix))]
    pub(crate) async fn start(
        &self,
        _resources: &Arc<RuntimeResources>,
    ) -> Result<Option<tokio::task::JoinHandle<std::io::Result<()>>>, DynError> {
        Ok(None)
    }
}
