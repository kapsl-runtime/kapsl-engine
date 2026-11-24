//! Inference entry points: blocking and streaming generation, and
//! host-requested cancellation of an in-flight request.

use crate::ffi::{
    clear_output, engine_error, json_error, request_from_wire, state, with_ffi_error, write_output,
};
use futures::StreamExt;
use kapsl_backend_abi::*;
use kapsl_engine_api::Engine as _;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub(crate) unsafe extern "C" fn infer(
    handle: *mut c_void,
    request: *const KapslLlamaRequestV1,
    packet_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_output(packet_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = state(handle)?;
            let (wire, mut request) = request_from_wire(request)?;
            let _registration = pack.register_request(&wire, &mut request);
            let packet = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .infer(&request)
                .map_err(engine_error)?;
            let bytes = serde_json::to_vec(&packet).map_err(json_error)?;
            write_output(packet_out, bytes)
                .map_err(|status| (status, "null inference packet output".to_string()))
        })
    }
}

pub(crate) unsafe extern "C" fn infer_stream(
    handle: *mut c_void,
    request: *const KapslLlamaRequestV1,
    user_data: *mut c_void,
    on_chunk: Option<KapslLlamaStreamChunkFn>,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe {
        with_ffi_error(error_out, || {
            let pack = state(handle)?;
            let callback = on_chunk.ok_or_else(|| {
                (
                    KAPSL_STATUS_INVALID_ARGUMENT,
                    "stream callback is required".to_string(),
                )
            })?;
            let (wire, mut request) = request_from_wire(request)?;
            let request_id = wire.request_id;
            let _registration = pack.register_request(&wire, &mut request);
            let mut stream = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .infer_stream(&request);
            pack.runtime.block_on(async {
                while let Some(item) = stream.next().await {
                    let packet = item.map_err(engine_error)?;
                    let bytes = serde_json::to_vec(&packet).map_err(json_error)?;
                    let status = callback(user_data, request_id, KapslSlice::from_bytes(&bytes));
                    if status != KAPSL_STATUS_OK {
                        if let Some(probe) = pack
                            .cancellations
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .get(&request_id)
                            .cloned()
                        {
                            probe.token.cancel();
                        }
                        return Err((
                            KAPSL_STATUS_CANCELLED,
                            "stream consumer cancelled request".to_string(),
                        ));
                    }
                }
                Ok(())
            })
        })
    }
}

pub(crate) unsafe extern "C" fn cancel(handle: *mut c_void, request_id: u64) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let pack = state(handle)?;
        let active = pack
            .cancellations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(probe) = active.get(&request_id) else {
            return Err((
                KAPSL_STATUS_INVALID_ARGUMENT,
                "unknown request id".to_string(),
            ));
        };
        probe.token.cancel();
        Ok(())
    }));
    match result {
        Ok(Ok(())) => KAPSL_STATUS_OK,
        Ok(Err((status, _))) => status,
        Err(_) => KAPSL_STATUS_PANIC,
    }
}
