//! Native llama.cpp pack behind Kapsl's stable C function-table ABI.
//!
//! The pack owns all Rust implementation details. Core sees only opaque
//! handles, versioned C structs, JSON wire messages, and explicit buffer
//! ownership. CUDA builds can expose either the runtime-owned shared-pool path
//! or the separately signed native-KV rollback profile.
//!
//! Generation uses the legacy `kapsl_llama_cpp_backend_v1` table. The CPU pack
//! also exports the backend-neutral `kapsl_backend_v1` table for the pooled
//! GGUF tasks (embed, rerank, classify); see `pooled`.

mod cancellation;
mod ffi;
mod inference;
mod lifecycle;
mod logger;
mod pack;
#[cfg(not(any(feature = "cuda12", kapsl_llama_external_pool_sdk)))]
mod pooled;
mod reports;
#[cfg(kapsl_llama_external_pool_sdk)]
mod shared_pool;

#[cfg(not(any(feature = "cuda12", kapsl_llama_external_pool_sdk)))]
pub use pooled::kapsl_backend_v1;

use kapsl_backend_abi::*;

#[cfg(kapsl_llama_external_pool_sdk)]
#[no_mangle]
pub static KAPSL_LLAMA_CPP_KV_MODE_V1: [u8; b"KAPSL_LLAMA_CPP_KV_MODE=shared_pool\0".len()] =
    *b"KAPSL_LLAMA_CPP_KV_MODE=shared_pool\0";

#[cfg(not(kapsl_llama_external_pool_sdk))]
#[no_mangle]
pub static KAPSL_LLAMA_CPP_KV_MODE_V1: [u8; b"KAPSL_LLAMA_CPP_KV_MODE=native\0".len()] =
    *b"KAPSL_LLAMA_CPP_KV_MODE=native\0";

#[cfg(kapsl_llama_external_pool_sdk)]
const CAPABILITIES: u64 = KAPSL_LLAMA_CAP_CUDA
    | KAPSL_LLAMA_CAP_SHARED_POOL
    | KAPSL_LLAMA_CAP_STREAMING
    | KAPSL_LLAMA_CAP_CANCELLATION
    | KAPSL_LLAMA_CAP_MEMORY_REPORTING;

#[cfg(not(kapsl_llama_external_pool_sdk))]
const CAPABILITIES: u64 = if cfg!(feature = "cuda12") {
    KAPSL_LLAMA_CAP_CUDA
        | KAPSL_LLAMA_CAP_NATIVE_KV
        | KAPSL_LLAMA_CAP_STREAMING
        | KAPSL_LLAMA_CAP_CANCELLATION
        | KAPSL_LLAMA_CAP_MEMORY_REPORTING
} else {
    KAPSL_LLAMA_CAP_CPU
        | KAPSL_LLAMA_CAP_NATIVE_KV
        | KAPSL_LLAMA_CAP_STREAMING
        | KAPSL_LLAMA_CAP_CANCELLATION
        | KAPSL_LLAMA_CAP_MEMORY_REPORTING
};

static API_V1: KapslLlamaCppApiV1 = KapslLlamaCppApiV1 {
    magic: KAPSL_LLAMA_CPP_ENTRYPOINT_MAGIC,
    abi_version: KAPSL_LLAMA_CPP_ABI_VERSION,
    struct_size: std::mem::size_of::<KapslLlamaCppApiV1>() as u32,
    wire_format: KAPSL_LLAMA_CPP_WIRE_FORMAT_JSON_V1,
    capabilities: CAPABILITIES,
    initialize: Some(lifecycle::initialize),
    planned_memory: Some(reports::planned_memory),
    load_model: Some(lifecycle::load_model),
    planned_request_memory: Some(reports::planned_request_memory),
    infer: Some(inference::infer),
    infer_stream: Some(inference::infer_stream),
    cancel: Some(inference::cancel),
    actual_memory: Some(reports::actual_memory),
    metrics: Some(reports::metrics),
    model_info: Some(reports::model_info),
    kv_capabilities: Some(reports::kv_capabilities),
    kv_topology: Some(reports::kv_topology),
    batching_policy: Some(reports::batching_policy),
    health_check: Some(lifecycle::health_check),
    unload: Some(lifecycle::unload),
    shutdown: Some(lifecycle::shutdown),
    free_buffer: Some(ffi::free_buffer),
};

/// Return the immutable ABI v1 function table.
#[no_mangle]
pub extern "C" fn kapsl_llama_cpp_backend_v1() -> *const KapslLlamaCppApiV1 {
    &API_V1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entrypoint_has_complete_native_pack_contract() {
        let api = unsafe { &*kapsl_llama_cpp_backend_v1() };
        assert_eq!(api.magic, KAPSL_LLAMA_CPP_ENTRYPOINT_MAGIC);
        assert_eq!(api.abi_version, KAPSL_LLAMA_CPP_ABI_VERSION);
        assert_eq!(api.struct_size as usize, std::mem::size_of_val(api));
        #[cfg(not(kapsl_llama_external_pool_sdk))]
        assert_ne!(api.capabilities & KAPSL_LLAMA_CAP_NATIVE_KV, 0);
        #[cfg(kapsl_llama_external_pool_sdk)]
        assert_ne!(api.capabilities & KAPSL_LLAMA_CAP_SHARED_POOL, 0);
        assert_ne!(api.capabilities & KAPSL_LLAMA_CAP_STREAMING, 0);
        #[cfg(not(kapsl_llama_external_pool_sdk))]
        assert_eq!(api.capabilities & KAPSL_LLAMA_CAP_SHARED_POOL, 0);
        assert!(api.initialize.is_some());
        assert!(api.load_model.is_some());
        assert!(api.infer.is_some());
        assert!(api.infer_stream.is_some());
        assert!(api.cancel.is_some());
        assert!(api.planned_memory.is_some());
        assert!(api.actual_memory.is_some());
        assert!(api.metrics.is_some());
        assert!(api.batching_policy.is_some());
        assert!(api.unload.is_some());
        assert!(api.shutdown.is_some());
    }
}
