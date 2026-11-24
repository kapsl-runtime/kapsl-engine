//! Log sink that forwards pack messages to the host's callback.

use kapsl_backend_abi::*;
use std::ffi::c_void;

#[derive(Clone, Copy)]
pub(crate) struct HostLogger {
    pub(crate) user_data: usize,
    pub(crate) callback: Option<KapslLogFn>,
}

unsafe impl Send for HostLogger {}
unsafe impl Sync for HostLogger {}

impl HostLogger {
    pub(crate) fn emit(self, level: u32, message: &str) {
        if let Some(callback) = self.callback {
            // SAFETY: the host promises that its callback context remains live
            // until shutdown returns. The message borrow lasts for this call.
            unsafe {
                callback(
                    self.user_data as *mut c_void,
                    level,
                    KapslSlice::from_bytes(message.as_bytes()),
                );
            }
        }
    }
}
