//! Read-only JSON report entry points: planned and actual memory, metrics,
//! model info, KV capability documents, and batching policy.

use crate::ffi::{
    clear_output, engine_error, json_error, path_from_slice, request_from_wire, state,
    with_ffi_error, write_output,
};
use crate::pack::PackState;
use kapsl_backend_abi::*;
use kapsl_engine_api::Engine as _;
use kapsl_llm::GgufBackend;
use serde::Serialize;
use std::ffi::c_void;

pub(crate) unsafe extern "C" fn planned_memory(
    handle: *mut c_void,
    model_path: KapslSlice,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    // SAFETY: outputs are ABI caller-owned.
    unsafe { clear_output(report_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = state(handle)?;
            let path = path_from_slice(model_path)?;
            let report = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .planned_memory(&path)
                .map_err(engine_error)?;
            let report = pack.normalize_memory(report);
            let bytes = serde_json::to_vec(&report).map_err(json_error)?;
            write_output(report_out, bytes)
                .map_err(|status| (status, "null memory report output".to_string()))
        })
    }
}

pub(crate) unsafe extern "C" fn planned_request_memory(
    handle: *mut c_void,
    request: *const KapslLlamaRequestV1,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_output(report_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = state(handle)?;
            let (_, request) = request_from_wire(request)?;
            let report = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .planned_request_memory(&request);
            let bytes = serde_json::to_vec(&report).map_err(json_error)?;
            write_output(report_out, bytes)
                .map_err(|status| (status, "null request memory output".to_string()))
        })
    }
}

pub(crate) unsafe extern "C" fn actual_memory(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    json_report(handle, report_out, error_out, |pack, backend| {
        serde_json::to_vec(&pack.normalize_memory(backend.actual_memory()))
    })
}

pub(crate) unsafe extern "C" fn metrics(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    json_report(handle, report_out, error_out, |_pack, backend| {
        serde_json::to_vec(&backend.metrics())
    })
}

pub(crate) unsafe extern "C" fn model_info(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    json_report(handle, report_out, error_out, |_pack, backend| {
        serde_json::to_vec(&backend.model_info())
    })
}

pub(crate) unsafe extern "C" fn kv_capabilities(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    // kapsl-engine-api 0.1.3 predates the neutral KV capability methods used by
    // newer cores. Native-KV packs are conservatively unmanaged; a future
    // shared-pool-capable pack will return the versioned capability document.
    json_report(handle, report_out, error_out, |_pack, _backend| {
        serde_json::to_vec(&serde_json::Value::Null)
    })
}

pub(crate) unsafe extern "C" fn kv_topology(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    json_report(handle, report_out, error_out, |_pack, _backend| {
        serde_json::to_vec(&serde_json::Value::Null)
    })
}

#[derive(Serialize)]
struct PackBatchingPolicy {
    max_batch: usize,
    self_batches: bool,
}

pub(crate) unsafe extern "C" fn batching_policy(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    json_report(handle, report_out, error_out, |_pack, backend| {
        serde_json::to_vec(&PackBatchingPolicy {
            max_batch: backend.max_batch(),
            self_batches: backend.self_batches(),
        })
    })
}

unsafe fn json_report(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
    make: impl FnOnce(&PackState, &GgufBackend) -> Result<Vec<u8>, serde_json::Error>,
) -> i32 {
    unsafe { clear_output(report_out) };
    unsafe {
        with_ffi_error(error_out, || {
            let pack = state(handle)?;
            let backend = pack
                .backend
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let bytes = make(pack, &backend).map_err(json_error)?;
            write_output(report_out, bytes)
                .map_err(|status| (status, "null JSON report output".to_string()))
        })
    }
}
