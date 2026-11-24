//! Adapter that hands Kapsl core's runtime-owned shared KV pool to llama.cpp
//! as an external KV pool. Compiled only into shared-pool packs.

use kapsl_backend_abi::*;
use kapsl_llm::gguf_backend::{
    GgufExternalKvPool, GgufExternalKvPoolFactory, GgufExternalKvPoolGeometry,
};
use llama_cpp_sys_2::{llama_kapsl_kv_pool_desc, LLAMA_KAPSL_KV_DTYPE_F16};
use std::ffi::c_void;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(crate) struct SharedPoolHost {
    pub(crate) user_data: usize,
    pub(crate) create: KapslCreateSharedPoolFn,
    pub(crate) destroy: KapslDestroySharedPoolFn,
    pub(crate) bytes: KapslSharedPoolBytesFn,
}

struct SharedPoolBridge {
    descriptor: KapslSharedPoolDescriptorV1,
    host_user_data: usize,
    destroy: KapslDestroySharedPoolFn,
}

unsafe impl Send for SharedPoolBridge {}

impl Drop for SharedPoolBridge {
    fn drop(&mut self) {
        // SAFETY: the core callback context remains live through pack shutdown,
        // and this is the one matching destroy for the successful create call.
        unsafe {
            (self.destroy)(
                self.host_user_data as *mut c_void,
                self.descriptor.pool_context,
            );
        }
    }
}

unsafe fn shared_bridge<'a>(user_data: *mut c_void) -> Option<&'a SharedPoolBridge> {
    if user_data.is_null() {
        None
    } else {
        // SAFETY: the bridge Box is the external pool's drop guard and therefore
        // outlives the llama context that invokes these callbacks.
        Some(unsafe { &*(user_data as *const SharedPoolBridge) })
    }
}

unsafe extern "C" fn bridge_reserve(
    user_data: *mut c_void,
    session_id: u64,
    tokens_needed: u32,
    block_table_device_out: *mut *mut u32,
    blocks_out: *mut u32,
) -> bool {
    unsafe { shared_bridge(user_data) }
        .and_then(|bridge| bridge.descriptor.reserve.map(|callback| (bridge, callback)))
        .is_some_and(|(bridge, callback)| unsafe {
            callback(
                bridge.descriptor.pool_context,
                session_id,
                tokens_needed,
                block_table_device_out,
                blocks_out,
            ) != 0
        })
}

unsafe extern "C" fn bridge_reserve_sequence(
    user_data: *mut c_void,
    sequence_id: u64,
    tokens_needed: u32,
    block_table_device_out: *mut *mut u32,
    blocks_out: *mut u32,
) -> bool {
    unsafe { shared_bridge(user_data) }
        .and_then(|bridge| {
            bridge
                .descriptor
                .reserve_sequence
                .map(|callback| (bridge, callback))
        })
        .is_some_and(|(bridge, callback)| unsafe {
            callback(
                bridge.descriptor.pool_context,
                sequence_id,
                tokens_needed,
                block_table_device_out,
                blocks_out,
            ) != 0
        })
}

unsafe extern "C" fn bridge_commit_sequences(
    user_data: *mut c_void,
    block_table_device_out: *mut *mut u32,
) -> bool {
    unsafe { shared_bridge(user_data) }
        .and_then(|bridge| {
            bridge
                .descriptor
                .commit_sequences
                .map(|callback| (bridge, callback))
        })
        .is_some_and(|(bridge, callback)| unsafe {
            callback(bridge.descriptor.pool_context, block_table_device_out) != 0
        })
}

unsafe extern "C" fn bridge_release(user_data: *mut c_void, sequence_id: u64) {
    if let Some((bridge, callback)) = unsafe { shared_bridge(user_data) }
        .and_then(|bridge| bridge.descriptor.release.map(|callback| (bridge, callback)))
    {
        unsafe { callback(bridge.descriptor.pool_context, sequence_id) };
    }
}

unsafe extern "C" fn bridge_touch(user_data: *mut c_void, sequence_id: u64) -> bool {
    unsafe { shared_bridge(user_data) }
        .and_then(|bridge| bridge.descriptor.touch.map(|callback| (bridge, callback)))
        .is_some_and(|(bridge, callback)| unsafe {
            callback(bridge.descriptor.pool_context, sequence_id) != 0
        })
}

fn checked_u32(value: usize, field: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{field} exceeds llama.cpp ABI range"))
}

pub(crate) fn external_pool_factory(host: SharedPoolHost) -> GgufExternalKvPoolFactory {
    Arc::new(move |geometry: GgufExternalKvPoolGeometry| {
        let wire_geometry = KapslSharedPoolGeometryV1 {
            struct_size: std::mem::size_of::<KapslSharedPoolGeometryV1>() as u32,
            device_id: checked_u32(geometry.device_id, "device id")?,
            requested_blocks: u64::try_from(geometry.requested_blocks)
                .map_err(|_| "requested blocks exceed Kapsl ABI range".to_string())?,
            block_size_tokens: checked_u32(geometry.block_size_tokens, "block size")?,
            num_layers: checked_u32(geometry.num_layers, "layer count")?,
            num_kv_heads: checked_u32(geometry.num_kv_heads, "KV head count")?,
            key_head_dim: checked_u32(geometry.key_head_dim, "key head dimension")?,
            value_head_dim: checked_u32(geometry.value_head_dim, "value head dimension")?,
            element_bytes: 2,
            max_sequences: checked_u32(geometry.max_sequences, "sequence count")?,
            max_blocks_per_sequence: checked_u32(
                geometry.max_blocks_per_sequence,
                "per-sequence block count",
            )?,
            model_fingerprint: geometry.model_fingerprint,
        };
        let mut descriptor = KapslSharedPoolDescriptorV1 {
            struct_size: 0,
            pool_context: std::ptr::null_mut(),
            device_base: std::ptr::null_mut(),
            addressable_blocks: 0,
            block_table_device: std::ptr::null_mut(),
            block_table_layer_stride: 0,
            block_table_sequence_stride: 0,
            sequence_slots: 0,
            reserve: None,
            reserve_sequence: None,
            commit_sequences: None,
            release: None,
            touch: None,
        };
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: all arguments are stack-owned for this synchronous host call.
        let status = unsafe {
            (host.create)(
                host.user_data as *mut c_void,
                &wire_geometry,
                &mut descriptor,
                &mut error,
            )
        };
        if !error.ptr.is_null() {
            return Err(
                "host create_shared_pool returned an invalid cross-allocator error buffer"
                    .to_string(),
            );
        }
        if status != KAPSL_STATUS_OK {
            return Err(format!(
                "host create_shared_pool failed with status {status}"
            ));
        }
        if descriptor.struct_size < std::mem::size_of::<KapslSharedPoolDescriptorV1>() as u32
            || descriptor.pool_context.is_null()
            || descriptor.device_base.is_null()
        {
            if !descriptor.pool_context.is_null() {
                unsafe { (host.destroy)(host.user_data as *mut c_void, descriptor.pool_context) };
            }
            return Err("host returned an incomplete shared-pool descriptor".to_string());
        }
        let num_blocks = match u32::try_from(descriptor.addressable_blocks) {
            Ok(blocks) => blocks,
            Err(_) => {
                // SAFETY: create succeeded, so its descriptor requires exactly
                // one matching destroy even though it cannot be represented by
                // llama.cpp's u32 block-count ABI.
                unsafe { (host.destroy)(host.user_data as *mut c_void, descriptor.pool_context) };
                return Err("shared pool has more than u32::MAX addressable blocks".to_string());
            }
        };
        let bridge = Box::new(SharedPoolBridge {
            descriptor,
            host_user_data: host.user_data,
            destroy: host.destroy,
        });
        let bridge_pointer = (&*bridge as *const SharedPoolBridge).cast_mut().cast();
        let raw = llama_kapsl_kv_pool_desc {
            user_data: bridge_pointer,
            device_id: wire_geometry.device_id,
            block_size: wire_geometry.block_size_tokens,
            num_blocks,
            num_kv_heads: wire_geometry.num_kv_heads,
            head_dim: wire_geometry.key_head_dim,
            dtype: LLAMA_KAPSL_KV_DTYPE_F16,
            device_base: descriptor.device_base,
            block_table_device: descriptor.block_table_device,
            block_table_layer_stride: descriptor.block_table_layer_stride,
            n_layers: wire_geometry.num_layers,
            max_blocks_per_seq: wire_geometry.max_blocks_per_sequence,
            block_table_seq_stride: descriptor.block_table_sequence_stride,
            n_seq_slots: descriptor.sequence_slots,
            model_fingerprint: wire_geometry.model_fingerprint,
            reserve: Some(bridge_reserve),
            reserve_seq: descriptor
                .reserve_sequence
                .map(|_| bridge_reserve_sequence as _),
            commit_seq: descriptor
                .commit_sequences
                .map(|_| bridge_commit_sequences as _),
            release: Some(bridge_release),
            touch: descriptor.touch.map(|_| bridge_touch as _),
            reserve_prefix: None,
            promote_prefix: None,
            needs_restore: None,
        };
        let bytes_host = host;
        let pool_context = descriptor.pool_context as usize;
        let usage_bytes: Arc<dyn Fn() -> usize + Send + Sync> = Arc::new(move || {
            // SAFETY: Kapsl core retains the host context through pack shutdown.
            let bytes = unsafe {
                (bytes_host.bytes)(
                    bytes_host.user_data as *mut c_void,
                    pool_context as *mut c_void,
                )
            };
            usize::try_from(bytes).unwrap_or(usize::MAX)
        });
        // SAFETY: bridge owns the raw callback context and its Drop invokes the
        // matching core destroy callback after llama.cpp releases the context.
        unsafe { GgufExternalKvPool::from_raw(geometry, raw, bridge, usage_bytes) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};

    struct AdapterHostState {
        creates: AtomicUsize,
        destroys: AtomicUsize,
        addressable_blocks: AtomicU64,
    }

    unsafe extern "C" fn adapter_reserve(
        _pool_context: *mut c_void,
        _session_id: u64,
        _tokens_needed: u32,
        _block_table_device_out: *mut *mut u32,
        _blocks_out: *mut u32,
    ) -> u32 {
        1
    }

    unsafe extern "C" fn adapter_commit(
        _pool_context: *mut c_void,
        _block_table_device_out: *mut *mut u32,
    ) -> u32 {
        1
    }

    unsafe extern "C" fn adapter_release(_pool_context: *mut c_void, _sequence_id: u64) {}

    unsafe extern "C" fn adapter_create(
        user_data: *mut c_void,
        geometry: *const KapslSharedPoolGeometryV1,
        descriptor_out: *mut KapslSharedPoolDescriptorV1,
        _error_out: *mut KapslOwnedBuffer,
    ) -> i32 {
        let state = unsafe { &*(user_data as *const AdapterHostState) };
        let geometry = unsafe { &*geometry };
        state.creates.fetch_add(1, AtomicOrdering::SeqCst);
        let configured_blocks = state.addressable_blocks.load(AtomicOrdering::SeqCst);
        let addressable_blocks = if configured_blocks == 0 {
            geometry.requested_blocks
        } else {
            configured_blocks
        };
        unsafe {
            descriptor_out.write(KapslSharedPoolDescriptorV1 {
                struct_size: std::mem::size_of::<KapslSharedPoolDescriptorV1>() as u32,
                pool_context: user_data,
                device_base: std::ptr::dangling_mut::<c_void>(),
                addressable_blocks,
                block_table_device: std::ptr::dangling_mut::<u32>(),
                block_table_layer_stride: geometry.max_blocks_per_sequence,
                block_table_sequence_stride: geometry
                    .num_layers
                    .saturating_mul(geometry.max_blocks_per_sequence),
                sequence_slots: geometry.max_sequences,
                reserve: Some(adapter_reserve),
                reserve_sequence: Some(adapter_reserve),
                commit_sequences: Some(adapter_commit),
                release: Some(adapter_release),
                touch: None,
            });
        }
        KAPSL_STATUS_OK
    }

    unsafe extern "C" fn adapter_destroy(user_data: *mut c_void, _pool_context: *mut c_void) {
        let state = unsafe { &*(user_data as *const AdapterHostState) };
        state.destroys.fetch_add(1, AtomicOrdering::SeqCst);
    }

    unsafe extern "C" fn adapter_bytes(_user_data: *mut c_void, _pool_context: *mut c_void) -> u64 {
        4096
    }

    fn adapter_geometry() -> GgufExternalKvPoolGeometry {
        GgufExternalKvPoolGeometry {
            device_id: 0,
            requested_blocks: 64,
            block_size_tokens: 16,
            num_layers: 8,
            num_kv_heads: 4,
            key_head_dim: 64,
            value_head_dim: 64,
            max_sequences: 2,
            max_blocks_per_sequence: 32,
            model_fingerprint: 0xC0DE,
        }
    }

    fn adapter_host(state: &AdapterHostState) -> SharedPoolHost {
        SharedPoolHost {
            user_data: (state as *const AdapterHostState) as usize,
            create: adapter_create,
            destroy: adapter_destroy,
            bytes: adapter_bytes,
        }
    }

    #[test]
    fn external_pool_adapter_destroys_the_core_pool_with_its_guard() {
        let state = AdapterHostState {
            creates: AtomicUsize::new(0),
            destroys: AtomicUsize::new(0),
            addressable_blocks: AtomicU64::new(0),
        };
        let pool = external_pool_factory(adapter_host(&state))(adapter_geometry())
            .expect("valid core descriptor");
        assert_eq!(state.creates.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(state.destroys.load(AtomicOrdering::SeqCst), 0);
        drop(pool);
        assert_eq!(state.destroys.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn external_pool_adapter_cleans_up_unrepresentable_block_counts() {
        let state = AdapterHostState {
            creates: AtomicUsize::new(0),
            destroys: AtomicUsize::new(0),
            addressable_blocks: AtomicU64::new(u64::from(u32::MAX) + 1),
        };
        let result = external_pool_factory(adapter_host(&state))(adapter_geometry());
        assert!(result.is_err());
        assert_eq!(state.creates.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(state.destroys.load(AtomicOrdering::SeqCst), 1);
    }
}
