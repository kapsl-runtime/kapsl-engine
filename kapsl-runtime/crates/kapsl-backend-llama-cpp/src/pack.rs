//! Per-handle pack state: the GGUF backend, the runtime that drives it, and
//! the cancellation monitor, all living from `initialize` to `shutdown`.

use crate::cancellation::{
    spawn_monitor, ActiveCancellations, CancellationProbe, CancellationRegistration,
};
use crate::logger::HostLogger;
#[cfg(kapsl_llama_external_pool_sdk)]
use crate::shared_pool::{external_pool_factory, SharedPoolHost};
use kapsl_backend_abi::*;
use kapsl_engine_api::{
    CancellationToken, Engine as _, InferenceRequest, MemoryDomain, MemoryReport,
};
use kapsl_llm::GgufBackend;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

pub(crate) struct PackState {
    pub(crate) runtime: tokio::runtime::Runtime,
    pub(crate) backend: RwLock<GgufBackend>,
    pub(crate) cancellations: ActiveCancellations,
    monitor_stop: Arc<AtomicBool>,
    monitor: Option<std::thread::JoinHandle<()>>,
    pub(crate) logger: HostLogger,
    profile: u32,
}

impl PackState {
    pub(crate) fn new(config: &KapslLlamaConfigV1) -> Result<Self, String> {
        let expected_profile = if cfg!(feature = "cuda12") {
            KAPSL_LLAMA_PROFILE_CUDA12
        } else {
            KAPSL_LLAMA_PROFILE_CPU
        };
        if config.profile != expected_profile {
            return Err(format!(
                "pack profile mismatch: library profile={} requested={}",
                expected_profile, config.profile
            ));
        }
        let host_callbacks = if config.host.is_null() {
            None
        } else {
            // SAFETY: initialize validates and consumes the borrowed host table
            // synchronously. Only its stable callback/context values are copied.
            let host = unsafe { &*config.host };
            if host.struct_size < std::mem::size_of::<KapslLlamaHostCallbacksV1>() as u32 {
                return Err("host callback table is smaller than ABI v1".to_string());
            }
            Some(*host)
        };
        let logger = if let Some(host) = host_callbacks {
            HostLogger {
                user_data: host.user_data as usize,
                callback: host.log,
            }
        } else {
            HostLogger {
                user_data: 0,
                callback: None,
            }
        };
        #[cfg(kapsl_llama_external_pool_sdk)]
        let backend = {
            if config.require_shared_pool == 0 {
                return Err(
                    "signed shared-pool pack requires Kapsl core to set require_shared_pool"
                        .to_string(),
                );
            }
            let host = host_callbacks.ok_or_else(|| {
                "runtime-owned shared-pool pack requires a host callback table".to_string()
            })?;
            let shared_host = SharedPoolHost {
                user_data: host.user_data as usize,
                create: host
                    .create_shared_pool
                    .ok_or_else(|| "host callback table lacks create_shared_pool".to_string())?,
                destroy: host
                    .destroy_shared_pool
                    .ok_or_else(|| "host callback table lacks destroy_shared_pool".to_string())?,
                bytes: host
                    .shared_pool_bytes
                    .ok_or_else(|| "host callback table lacks shared_pool_bytes".to_string())?,
            };
            GgufBackend::new_cuda_external_kv_pool(
                config.device_id as usize,
                external_pool_factory(shared_host),
            )
        };
        #[cfg(not(kapsl_llama_external_pool_sdk))]
        let backend = {
            if config.require_shared_pool != 0 {
                return Err(
                    "this pack exposes llama.cpp native KV only; a runtime-owned shared-pool pack is required"
                        .to_string(),
                );
            }
            GgufBackend::new_on_device(config.device_id as usize)
        };
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("kapsl-llama-pack")
            .build()
            .map_err(|error| format!("create llama.cpp pack runtime: {error}"))?;
        let cancellations = ActiveCancellations::default();
        let monitor_stop = Arc::new(AtomicBool::new(false));
        let monitor = spawn_monitor(Arc::clone(&cancellations), Arc::clone(&monitor_stop))
            .map_err(|error| format!("start cancellation monitor: {error}"))?;
        logger.emit(KAPSL_LOG_INFO, "initialized native llama.cpp backend pack");
        Ok(Self {
            runtime,
            backend: RwLock::new(backend),
            cancellations,
            monitor_stop,
            monitor: Some(monitor),
            logger,
            profile: config.profile,
        })
    }

    pub(crate) fn normalize_memory(&self, mut report: MemoryReport) -> MemoryReport {
        if self.profile == KAPSL_LLAMA_PROFILE_CPU {
            for allocation in &mut report.allocations {
                if matches!(allocation.domain, MemoryDomain::Cuda { .. }) {
                    allocation.domain = MemoryDomain::Host;
                }
            }
        }
        report
    }

    pub(crate) fn register_request(
        &self,
        wire: &KapslLlamaRequestV1,
        request: &mut InferenceRequest,
    ) -> CancellationRegistration {
        let token = CancellationToken::new();
        request.cancellation = Some(token.clone());
        self.cancellations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                wire.request_id,
                CancellationProbe {
                    context: wire.cancellation_context as usize,
                    callback: wire.is_cancelled,
                    token,
                },
            );
        CancellationRegistration {
            request_id: wire.request_id,
            active: Arc::clone(&self.cancellations),
        }
    }

    pub(crate) fn stop(&mut self) {
        self.backend
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .unload();
        self.monitor_stop.store(true, Ordering::Release);
        if let Some(monitor) = self.monitor.take() {
            let _ = monitor.join();
        }
        self.logger
            .emit(KAPSL_LOG_INFO, "shut down native llama.cpp backend pack");
    }
}
