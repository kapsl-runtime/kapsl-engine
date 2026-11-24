//! Plumbing shared by the ABI entry points: owned-buffer hand-off, error and
//! panic containment, and decoding of borrowed handle, request and path
//! arguments.

use crate::pack::PackState;
use kapsl_backend_abi::*;
use kapsl_engine_api::InferenceRequest;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

fn owned_buffer(bytes: Vec<u8>) -> KapslOwnedBuffer {
    let mut bytes = std::mem::ManuallyDrop::new(bytes);
    KapslOwnedBuffer {
        ptr: bytes.as_mut_ptr(),
        len: bytes.len(),
        capacity: bytes.capacity(),
    }
}

pub(crate) unsafe fn clear_output(output: *mut KapslOwnedBuffer) {
    if !output.is_null() {
        // SAFETY: caller supplied writable ABI output storage.
        unsafe { *output = KapslOwnedBuffer::empty() };
    }
}

pub(crate) unsafe fn write_output(
    output: *mut KapslOwnedBuffer,
    bytes: Vec<u8>,
) -> Result<(), i32> {
    if output.is_null() {
        return Err(KAPSL_STATUS_INVALID_ARGUMENT);
    }
    // SAFETY: checked non-null and required writable by the ABI.
    unsafe { *output = owned_buffer(bytes) };
    Ok(())
}

unsafe fn write_error(output: *mut KapslOwnedBuffer, message: impl AsRef<str>) {
    if !output.is_null() {
        // SAFETY: caller supplied writable ABI output storage.
        unsafe { *output = owned_buffer(message.as_ref().as_bytes().to_vec()) };
    }
}

pub(crate) unsafe extern "C" fn free_buffer(buffer: KapslOwnedBuffer) {
    if buffer.ptr.is_null() {
        return;
    }
    // SAFETY: buffers originate from `owned_buffer` in this library and are
    // returned at most once through this function.
    unsafe {
        drop(Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.capacity));
    }
}

pub(crate) unsafe fn with_ffi_error(
    error_out: *mut KapslOwnedBuffer,
    operation: impl FnOnce() -> Result<(), (i32, String)>,
) -> i32 {
    // SAFETY: forwarded from the ABI caller.
    unsafe { clear_output(error_out) };
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => KAPSL_STATUS_OK,
        Ok(Err((status, message))) => {
            // SAFETY: forwarded from the ABI caller.
            unsafe { write_error(error_out, message) };
            status
        }
        Err(_) => {
            // SAFETY: forwarded from the ABI caller.
            unsafe { write_error(error_out, "native llama.cpp pack panicked") };
            KAPSL_STATUS_PANIC
        }
    }
}

pub(crate) unsafe fn state<'a>(handle: *mut c_void) -> Result<&'a PackState, (i32, String)> {
    if handle.is_null() {
        return Err((
            KAPSL_STATUS_INVALID_ARGUMENT,
            "null pack handle".to_string(),
        ));
    }
    // SAFETY: handles originate from `initialize` and live until `shutdown`.
    Ok(unsafe { &*(handle as *mut PackState) })
}

pub(crate) unsafe fn request_from_wire(
    request: *const KapslLlamaRequestV1,
) -> Result<(KapslLlamaRequestV1, InferenceRequest), (i32, String)> {
    if request.is_null() {
        return Err((KAPSL_STATUS_INVALID_ARGUMENT, "null request".to_string()));
    }
    // SAFETY: checked non-null; the ABI requires a full v1 request.
    let wire = unsafe { *request };
    if wire.struct_size < std::mem::size_of::<KapslLlamaRequestV1>() as u32
        || wire.wire_format != KAPSL_LLAMA_CPP_WIRE_FORMAT_JSON_V1
    {
        return Err((
            KAPSL_STATUS_INCOMPATIBLE_ABI,
            "unsupported request struct or wire format".to_string(),
        ));
    }
    // SAFETY: request bytes are borrowed for the duration of this call.
    let bytes = unsafe { wire.request_json.as_bytes() }.ok_or_else(|| {
        (
            KAPSL_STATUS_INVALID_ARGUMENT,
            "request JSON has a null pointer".to_string(),
        )
    })?;
    let parsed = serde_json::from_slice(bytes).map_err(|error| {
        (
            KAPSL_STATUS_INVALID_ARGUMENT,
            format!("decode inference request JSON: {error}"),
        )
    })?;
    Ok((wire, parsed))
}

pub(crate) unsafe fn path_from_slice(path: KapslSlice) -> Result<PathBuf, (i32, String)> {
    // SAFETY: path is borrowed for the current ABI call.
    let bytes = unsafe { path.as_bytes() }.ok_or_else(|| {
        (
            KAPSL_STATUS_INVALID_ARGUMENT,
            "model path has a null pointer".to_string(),
        )
    })?;
    let text = std::str::from_utf8(bytes).map_err(|error| {
        (
            KAPSL_STATUS_INVALID_ARGUMENT,
            format!("model path is not UTF-8: {error}"),
        )
    })?;
    Ok(Path::new(text).to_path_buf())
}

pub(crate) fn engine_error(error: kapsl_engine_api::EngineError) -> (i32, String) {
    let status = if matches!(error, kapsl_engine_api::EngineError::Cancelled { .. }) {
        KAPSL_STATUS_CANCELLED
    } else {
        KAPSL_STATUS_BACKEND_ERROR
    };
    (status, error.to_string())
}

pub(crate) fn json_error(error: serde_json::Error) -> (i32, String) {
    (
        KAPSL_STATUS_BACKEND_ERROR,
        format!("encode native-pack JSON: {error}"),
    )
}
