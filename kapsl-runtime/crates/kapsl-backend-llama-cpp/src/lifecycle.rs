//! Lifecycle entry points: create the pack, load and unload its model, check
//! its health, and shut it down.

use crate::ffi::{clear_output, engine_error, path_from_slice, state, with_ffi_error};
use crate::pack::PackState;
use kapsl_backend_abi::*;
use kapsl_engine_api::Engine as _;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub(crate) unsafe extern "C" fn initialize(
    config: *const KapslLlamaConfigV1,
    handle_out: *mut *mut c_void,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    // SAFETY: outputs are ABI caller-owned.
    unsafe { clear_output(error_out) };
    if !handle_out.is_null() {
        // SAFETY: checked non-null.
        unsafe { *handle_out = std::ptr::null_mut() };
    }
    // SAFETY: all raw pointer access is validated inside the closure.
    unsafe {
        with_ffi_error(error_out, || {
            if config.is_null() || handle_out.is_null() {
                return Err((
                    KAPSL_STATUS_INVALID_ARGUMENT,
                    "initialize requires config and handle output".to_string(),
                ));
            }
            let config = &*config;
            if config.struct_size < std::mem::size_of::<KapslLlamaConfigV1>() as u32 {
                return Err((
                    KAPSL_STATUS_INCOMPATIBLE_ABI,
                    "llama.cpp config is smaller than ABI v1".to_string(),
                ));
            }
            let pack = PackState::new(config).map_err(|message| {
                let status = if config.require_shared_pool != 0 {
                    KAPSL_STATUS_UNSUPPORTED
                } else {
                    KAPSL_STATUS_BACKEND_ERROR
                };
                (status, message)
            })?;
            *handle_out = Box::into_raw(Box::new(pack)).cast();
            Ok(())
        })
    }
}

pub(crate) unsafe extern "C" fn load_model(
    handle: *mut c_void,
    model_path: KapslSlice,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe {
        with_ffi_error(error_out, || {
            let pack = state(handle)?;
            let path = path_from_slice(model_path)?;
            pack.logger.emit(
                KAPSL_LOG_INFO,
                &format!("loading GGUF model {}", path.display()),
            );
            pack.runtime
                .block_on(
                    pack.backend
                        .write()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .load(&path),
                )
                .map_err(engine_error)
        })
    }
}

pub(crate) unsafe extern "C" fn health_check(
    handle: *mut c_void,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe {
        with_ffi_error(error_out, || {
            state(handle)?
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .health_check()
                .map_err(engine_error)
        })
    }
}

pub(crate) unsafe extern "C" fn unload(handle: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        if let Ok(pack) = state(handle) {
            pack.backend
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .unload();
        }
    }));
}

pub(crate) unsafe extern "C" fn shutdown(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        // SAFETY: shutdown consumes the handle exactly once.
        let mut pack = Box::from_raw(handle as *mut PackState);
        pack.stop();
    }));
}

#[cfg(all(test, not(kapsl_llama_external_pool_sdk)))]
mod tests {
    use super::*;
    use crate::ffi::free_buffer;

    #[test]
    fn shared_pool_requirement_fails_closed() {
        let config = KapslLlamaConfigV1 {
            struct_size: std::mem::size_of::<KapslLlamaConfigV1>() as u32,
            profile: if cfg!(feature = "cuda12") {
                KAPSL_LLAMA_PROFILE_CUDA12
            } else {
                KAPSL_LLAMA_PROFILE_CPU
            },
            device_id: 0,
            model_id: 1,
            replica_id: 0,
            require_shared_pool: 1,
            host: std::ptr::null(),
        };
        let mut handle = std::ptr::null_mut();
        let mut error = KapslOwnedBuffer::empty();
        let status = unsafe { initialize(&config, &mut handle, &mut error) };
        assert_eq!(status, KAPSL_STATUS_UNSUPPORTED);
        assert!(handle.is_null());
        assert!(!error.ptr.is_null());
        unsafe { free_buffer(error) };
    }
}
