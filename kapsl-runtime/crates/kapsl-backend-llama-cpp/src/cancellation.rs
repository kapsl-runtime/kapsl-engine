//! Request cancellation: the probes registered for in-flight requests and the
//! monitor thread that polls the host on their behalf.

use kapsl_backend_abi::*;
use kapsl_engine_api::CancellationToken;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

#[derive(Clone)]
pub(crate) struct CancellationProbe {
    pub(crate) context: usize,
    pub(crate) callback: Option<KapslRequestCancelledFn>,
    pub(crate) token: CancellationToken,
}

/// Probes for the requests currently inside an ABI call, keyed by request id.
pub(crate) type ActiveCancellations = Arc<Mutex<HashMap<u64, CancellationProbe>>>;

/// Unregisters its request's probe when the ABI call that created it returns.
pub(crate) struct CancellationRegistration {
    pub(crate) request_id: u64,
    pub(crate) active: ActiveCancellations,
}

impl Drop for CancellationRegistration {
    fn drop(&mut self) {
        self.active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.request_id);
    }
}

/// Start the thread that polls each active request's host callback and cancels
/// its token once the host reports the request cancelled. It runs until `stop`
/// is set.
pub(crate) fn spawn_monitor(
    active: ActiveCancellations,
    stop: Arc<AtomicBool>,
) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("kapsl-llama-cancel".to_string())
        .spawn(move || {
            while !stop.load(Ordering::Acquire) {
                {
                    let active = active
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    for (request_id, probe) in active.iter() {
                        let cancelled = probe.callback.is_some_and(|callback| {
                            // SAFETY: the request context remains alive until
                            // its synchronous ABI call unregisters this probe.
                            unsafe { callback(probe.context as *mut c_void, *request_id) != 0 }
                        });
                        if cancelled {
                            probe.token.cancel();
                        }
                    }
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        })
}
