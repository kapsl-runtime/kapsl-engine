//! Construction and lifecycle state for the native inference transport.

use super::*;

/// Supported native inference transports after CLI parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RuntimeTransportMode {
    Socket,
    Tcp,
    Shm,
    Hybrid,
    Auto,
}

impl RuntimeTransportMode {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "socket" => Ok(Self::Socket),
            "tcp" => Ok(Self::Tcp),
            "shm" => Ok(Self::Shm),
            "hybrid" => Ok(Self::Hybrid),
            "auto" => Ok(Self::Auto),
            other => Err(format!(
                "Invalid transport mode: {other}. Use 'socket', 'tcp', 'shm', 'hybrid', or 'auto'"
            )),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Socket => "socket",
            Self::Tcp => "tcp",
            Self::Shm => "shm",
            Self::Hybrid => "hybrid",
            Self::Auto => "auto",
        }
    }

    pub(crate) fn uses_ipc_socket(self) -> bool {
        matches!(self, Self::Socket | Self::Hybrid)
            || (self == Self::Auto && !ShmServer::is_available())
    }
}

/// Immutable settings needed to validate and construct the native transport.
#[derive(Clone, Debug)]
pub(crate) struct RuntimeTransportConfig {
    pub(crate) mode: RuntimeTransportMode,
    pub(crate) socket_path: String,
    pub(crate) tcp_bind: String,
    pub(crate) tcp_port: u16,
    pub(crate) tcp_auth_token: Option<String>,
    pub(crate) shm_size_bytes: usize,
}

impl RuntimeTransportConfig {
    pub(crate) fn preflight(&self) -> Result<(), DynError> {
        if self.mode.uses_ipc_socket() {
            preflight_ipc_socket(&self.socket_path)?;
        }
        Ok(())
    }

    pub(crate) fn validate_tcp_exposure(&self) -> Result<(), DynError> {
        if self.mode != RuntimeTransportMode::Tcp {
            return Ok(());
        }
        let bind_addr = parse_bind_ip(&self.tcp_bind, IpAddr::from([127, 0, 0, 1]), "bind");
        validate_native_tcp_exposure(bind_addr, self.tcp_auth_token.as_deref())?;
        if !bind_addr.is_loopback() {
            log::warn!(
                "Native TCP inference is bound to {} with token authentication. Traffic remains plaintext; use a trusted network or TLS tunnel.",
                bind_addr
            );
        }
        Ok(())
    }
}

/// Constructed native transport and the endpoint advertised at startup.
pub(crate) struct RuntimeTransport {
    server: Arc<dyn TransportServer>,
    endpoint: String,
}

impl RuntimeTransport {
    pub(crate) fn build(
        config: &RuntimeTransportConfig,
        inference: Arc<InferenceService>,
        registry: Arc<Registry>,
    ) -> Result<Self, DynError> {
        log::info!("=== Starting Transport Server ===");
        log::info!("Transport mode: {}", config.mode.as_str());

        let scheduler_lookup = || {
            let inference = inference.clone();
            Arc::new(move |model_id: u32| inference.scheduler_for_transport(model_id))
                as Arc<dyn Fn(u32) -> Option<Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>
        };
        let scheduler_snapshot = || {
            let inference = inference.clone();
            Arc::new(move || inference.scheduler_snapshot()) as SchedulerSnapshot
        };

        let (server, endpoint): (Arc<dyn TransportServer>, String) = match config.mode {
            RuntimeTransportMode::Socket => {
                log::info!("Socket: {}", config.socket_path);
                (
                    Arc::new(IpcServer::new_with_lookup(
                        &config.socket_path,
                        scheduler_lookup(),
                        None,
                    )),
                    config.socket_path.clone(),
                )
            }
            RuntimeTransportMode::Tcp => {
                log::info!("TCP Address: {}:{}", config.tcp_bind, config.tcp_port);
                let server = TcpServer::new_with_lookup(
                    &config.tcp_bind,
                    config.tcp_port,
                    scheduler_lookup(),
                );
                let server = match config.tcp_auth_token.as_deref() {
                    Some(token) => server.with_auth_token(token.to_owned()),
                    None => server,
                };
                (
                    Arc::new(server),
                    format!("{}:{}", config.tcp_bind, config.tcp_port),
                )
            }
            RuntimeTransportMode::Shm => {
                log::info!("Using shared memory transport");
                let shm_name = process_shm_name();
                log::info!("Shared memory: {shm_name}");
                (
                    Arc::new(ShmServer::new_with_lookup_and_registry(
                        &shm_name,
                        config.shm_size_bytes,
                        scheduler_lookup(),
                        scheduler_snapshot(),
                        Some(registry.clone()),
                    )),
                    shm_name,
                )
            }
            RuntimeTransportMode::Hybrid => {
                log::info!("Using hybrid transport (Socket + SHM)");
                log::info!("Socket: {}", config.socket_path);
                let shm_name = process_shm_name();
                log::info!("Shared memory: {shm_name}");
                let shm_manager = Arc::new(
                    ShmManager::create(&shm_name, config.shm_size_bytes)
                        .map_err(|error| format!("Failed to create SHM manager: {error}"))?,
                );
                (
                    Arc::new(IpcServer::new_with_lookup(
                        &config.socket_path,
                        scheduler_lookup(),
                        Some(shm_manager),
                    )),
                    format!("{} (shm: {})", config.socket_path, shm_name),
                )
            }
            RuntimeTransportMode::Auto if ShmServer::is_available() => {
                log::info!("Auto-selecting transport: shared memory");
                let shm_name = process_shm_name();
                log::info!("Shared memory: {shm_name}");
                (
                    Arc::new(ShmServer::new_with_lookup_and_registry(
                        &shm_name,
                        config.shm_size_bytes,
                        scheduler_lookup(),
                        scheduler_snapshot(),
                        Some(registry),
                    )),
                    shm_name,
                )
            }
            RuntimeTransportMode::Auto => {
                log::info!("Auto-selecting transport: socket");
                log::info!("Socket: {}", config.socket_path);
                (
                    Arc::new(IpcServer::new_with_lookup(
                        &config.socket_path,
                        scheduler_lookup(),
                        None,
                    )),
                    config.socket_path.clone(),
                )
            }
        };

        Ok(Self { server, endpoint })
    }

    pub(crate) fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub(crate) async fn run(&self) -> Result<(), kapsl_transport::TransportError> {
        self.server.run().await
    }
}

fn process_shm_name() -> String {
    format!("/kapsl_shm_{}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_transport_modes() {
        assert_eq!(
            RuntimeTransportMode::parse("socket").unwrap(),
            RuntimeTransportMode::Socket
        );
        assert_eq!(
            RuntimeTransportMode::parse("AUTO").unwrap(),
            RuntimeTransportMode::Auto
        );
        assert!(RuntimeTransportMode::parse("invalid").is_err());
    }
}
