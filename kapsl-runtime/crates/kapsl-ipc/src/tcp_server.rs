use async_trait::async_trait;
use kapsl_scheduler::ReplicaScheduler;
use kapsl_transport::{TransportError, TransportServer};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::TcpListener;

/// TCP-based server for network communication across different computers
pub struct TcpServer {
    bind_addr: String,
    port: u16,
    scheduler_lookup: crate::server::SchedulerLookup,
}

impl TcpServer {
    pub fn new(
        bind_addr: &str,
        port: u16,
        schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
    ) -> Self {
        let schedulers = Arc::new(schedulers);
        let scheduler_lookup: crate::server::SchedulerLookup =
            Arc::new(move |model_id| schedulers.get(&model_id).cloned());
        Self::new_with_lookup(bind_addr, port, scheduler_lookup)
    }

    pub fn new_with_lookup(
        bind_addr: &str,
        port: u16,
        scheduler_lookup: crate::server::SchedulerLookup,
    ) -> Self {
        Self {
            bind_addr: bind_addr.to_string(),
            port,
            scheduler_lookup,
        }
    }

    async fn run_internal(&self) -> std::io::Result<()> {
        let addr = format!("{}:{}", self.bind_addr, self.port);
        let listener = TcpListener::bind(&addr).await?;
        let scheduler_lookup = self.scheduler_lookup.clone();

        log::info!("TCP Server listening on {}", addr);
        log::info!("TCP Server bound to {}", addr);

        loop {
            let (stream, peer_addr) = listener.accept().await?;
            let scheduler_lookup = scheduler_lookup.clone();

            log::info!("New TCP connection from {}", peer_addr);

            tokio::spawn(async move {
                if let Err(e) =
                    crate::server::handle_connection(stream, scheduler_lookup, None).await
                {
                    log::error!("Connection error: {}", e);
                }
                log::info!("TCP connection closed from {}", peer_addr);
            });
        }
    }
}

#[async_trait]
impl TransportServer for TcpServer {
    async fn run(&self) -> Result<(), TransportError> {
        self.run_internal().await.map_err(TransportError::Io)
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        // TCP listeners don't need explicit cleanup
        Ok(())
    }

    fn transport_type(&self) -> &'static str {
        "tcp"
    }
}
