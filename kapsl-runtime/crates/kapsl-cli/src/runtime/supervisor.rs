//! Ownership and shutdown coordination for long-lived runtime tasks.

use super::*;

/// All process-scoped serving tasks that must terminate together.
pub(crate) struct RuntimeSupervisor {
    pub(crate) transport: RuntimeTransport,
    pub(crate) kv_control_task: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    pub(crate) http_server: HttpServerHandle,
    pub(crate) monitor: RuntimeMonitor,
    pub(crate) autoscaler_task: tokio::task::JoinHandle<()>,
    pub(crate) resources: Arc<RuntimeResources>,
}

impl RuntimeSupervisor {
    pub(crate) async fn run(self) -> Result<(), DynError> {
        let Self {
            transport,
            mut kv_control_task,
            mut http_server,
            monitor,
            autoscaler_task,
            resources,
        } = self;

        let mut transport_task = Box::pin(transport.run());
        let mut shutdown_signal = Box::pin(runtime_shutdown_signal());
        let mut kv_control_exit = Box::pin(async {
            match kv_control_task.as_mut() {
                Some(task) => Some(task.await),
                None => std::future::pending().await,
            }
        });

        let outcome = tokio::select! {
            result = &mut transport_task => {
                result.map_err(|error| Box::new(error) as DynError)
            }
            result = &mut kv_control_exit => {
                let message = match result.expect("pending KV future cannot complete") {
                    Ok(Ok(())) => "KV control listener stopped unexpectedly".to_string(),
                    Ok(Err(error)) => format!("KV control listener failed: {error}"),
                    Err(error) => format!("KV control listener task failed: {error}"),
                };
                Err(message.into())
            }
            result = http_server.wait() => {
                let message = match result {
                    Ok(()) => "HTTP server stopped unexpectedly".to_string(),
                    Err(error) => format!("HTTP server task failed: {error}"),
                };
                Err(message.into())
            }
            signal = &mut shutdown_signal => {
                let signal = signal?;
                log::info!("Received {signal}; shutting down managed backends");
                shutdown_managed_backends(&resources)
            }
        };

        // Release futures borrowing task handles before coordinated cleanup.
        drop(kv_control_exit);
        drop(transport_task);
        if let Some(task) = kv_control_task {
            task.abort();
            let _ = task.await;
        }
        http_server.abort();
        monitor.abort();
        autoscaler_task.abort();
        outcome
    }
}

#[cfg(unix)]
async fn runtime_shutdown_signal() -> Result<&'static str, std::io::Error> {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {
        result = tokio::signal::ctrl_c() => {
            result?;
            Ok("SIGINT")
        }
        _ = terminate.recv() => Ok("SIGTERM"),
    }
}

#[cfg(not(unix))]
async fn runtime_shutdown_signal() -> Result<&'static str, std::io::Error> {
    tokio::signal::ctrl_c().await?;
    Ok("interrupt")
}

fn shutdown_managed_backends(resources: &RuntimeResources) -> Result<(), DynError> {
    let Some(deployment) = resources.managed_vllm() else {
        return Ok(());
    };
    let stopped = deployment
        .shutdown_all()
        .map_err(|error| format!("managed backend shutdown failed: {error}"))?;
    if stopped > 0 {
        log::info!("Stopped {stopped} managed vLLM runtime(s) during core shutdown");
    }
    Ok(())
}
