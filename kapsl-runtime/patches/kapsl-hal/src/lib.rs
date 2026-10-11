#[path = "memory/block_range.rs"]
pub mod block_range;
#[path = "memory/cpu_store.rs"]
pub mod cpu_block_store;
#[cfg(feature = "cuda")]
#[path = "memory/cross_device_scheduler.rs"]
pub mod cross_device_scheduler;
#[path = "device/info.rs"]
pub mod device;
#[path = "device/mesh.rs"]
pub mod device_mesh;
#[cfg(feature = "cuda")]
#[path = "memory/gpu_arena.rs"]
pub mod gpu_arena;
#[cfg(feature = "cuda")]
pub use gpu_arena::arena_region as gpu_arena_region;
#[cfg(feature = "cuda")]
#[path = "memory/gpu_ipc_region.rs"]
pub mod gpu_ipc_region;
#[path = "memory/gpu_region.rs"]
pub mod gpu_region;
#[cfg(feature = "cuda")]
#[path = "compute/gpu_tensor.rs"]
pub mod gpu_tensor;
#[cfg(all(feature = "cuda", unix))]
#[path = "memory/gpu_vmm_region.rs"]
pub mod gpu_vmm_region;
#[path = "compute/kernel.rs"]
pub mod kernel;
#[path = "communication/mock.rs"]
pub mod mock_comm;
#[cfg(feature = "nccl")]
#[path = "communication/nccl.rs"]
pub mod nccl_comm;
#[cfg(feature = "cuda")]
#[path = "memory/prefix_cache.rs"]
pub mod prefix_cache;
#[path = "compute/tensor.rs"]
pub mod tensor;

#[cfg(test)]
#[path = "device/mesh_tests.rs"]
mod device_mesh_tests;
#[cfg(test)]
#[path = "device/tests.rs"]
mod device_tests;
#[cfg(test)]
#[path = "compute/kernel_tests.rs"]
mod kernel_tests;
#[cfg(test)]
#[path = "compute/tensor_tests.rs"]
mod tensor_tests;
