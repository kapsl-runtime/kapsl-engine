# Kapsl HAL GPU regions

HAL owns physical GPU operations. The runtime owns admission, budget grants,
workload identities, worker registration and coordinated capacity changes.

| Region | Access | Capacity | Allocation contract |
| --- | --- | --- | --- |
| `GpuArenaRegion` | Local | Fixed slab | Existing aligned suballocations and owner quotas |
| `GpuIpcRegion` | Dedicated CUDA IPC export | Fixed | One isolated backing allocation |
| `GpuVmmRegion` | Dedicated POSIX FD export | Explicit growth and tail release | Stable virtual reservation, multiple physical segments |

The CUDA types require the `cuda` feature. The VMM API additionally requires Unix
file descriptors; exported backing requires a supporting CUDA driver. The runtime
enables its IPC/VMM transport on Linux. The bindings also compile on Unix hosts
without a CUDA driver for consumer checks. `gpu_region` provides portable
capability and observation types. Host
unit tests compile the exported-region implementations with fake drivers so
their lifetime and failure contracts can be tested without a GPU.

## Arena compatibility

The physical slab implementation lives in `memory/gpu_arena_region.rs` and is
available as `kapsl_hal::gpu_arena_region::GpuArenaRegion`.
`kapsl_hal::gpu_arena::GpuDevicePool` remains a type alias for that exact type.
Existing allocator callbacks, `GpuPoolBuffer`, owner admission queries and
`GpuKvPoolView` retain their APIs and single-base-pointer layout.

The extraction preserves HAL 0.3.1's KV initialization behavior:
clear on the retained stream, wait for completion before exposing an allocation,
and retain the extent and charge if initialization fails.

## Exported backing lifecycle

IPC setup is deliberately split:

```rust,ignore
use kapsl_hal::gpu_ipc_region::GpuIpcRegion;

// The runtime has already admitted physical capacity and retains its grant.
let region = GpuIpcRegion::allocate(device, bytes)?;
region.initialize_zeroed()?;
let raw_handle = region.export_handle()?;
// The adapter encodes raw_handle.as_bytes() for its authorized participant.
```

The caller retains `region` on initialization errors until cleanup succeeds.
Export and local pointer access are rejected until initialization's CUDA fence
completes. Calling initialization again after success does not clear live data.
`zero_range_after_fence` clears an explicitly retired extent; the caller must
first fence every process and stream that could access it.

VMM setup reserves addresses before committing physical backing:

```rust,ignore
use kapsl_hal::gpu_vmm_region::GpuVmmRegion;

let region = GpuVmmRegion::reserve(device, maximum_bytes)?;
let first = region.grow_to(minimum_bytes)?;
let first_fd = region.export_segment(first.id)?;
// Each admitted growth is another explicit operation.
if initial_bytes > minimum_bytes {
    let headroom = region.grow_to(initial_bytes)?;
    let headroom_fd = region.export_segment(headroom.id)?;
    // Pass headroom_fd to the participant through the runtime's protocol.
}
```

All capacities must meet device granularity. Each successful growth returns a
backend-neutral segment identity, byte offset and length. Segment identities are
not reused after shrink and cannot export another region's handle. The runtime
adapter supplies binding names, worker generations, descriptor indices and KV
block geometry. Only the runtime decides when new capacity is schedulable.

`tail_segments` and `shrink_boundary` expose whole-segment boundaries. The runtime
enforces its minimum capacity and retirement rules, closes exported descriptors,
collects worker unmap acknowledgments and fences GPU work before calling
`release_tail_after_fence`. Invalid targets are rejected before any driver
operation. Physical segments disappear from accounting only after handle release
succeeds; an unmap alone does not release the committed-byte observation.

Both region types provide unsafe `release_after_fence` methods: the caller must
prove that local and remote users, mappings and exported handles are retired.
Calls are serialized with export and resize, and successful release is idempotent.
A failed release preserves state for retry. Keep the region **and its budget
grant** until successful cleanup; the HAL cannot own or release runtime grants.

Dropping a region is not an importer acknowledgment. Unexported regions attempt
local cleanup. Exported regions abandoned without explicit fenced release retain
their backing and CUDA context and log an error. This prevents a Rust destructor
from freeing memory that another process may still use; it is not a reclamation
policy or a substitute for the runtime's quarantine/retirement protocol.

## Observations and failure handling

`GpuRegion::physical_snapshot` reports committed, mapped and virtual bytes.
Exported regions also report cleared and fenced `ready_bytes` for the parent
context. Arenas leave readiness unknown because their allocation contracts
govern initialization separately. Logical free bytes and workload quotas remain
in the arena's existing `snapshot`; none of these observations is an additional
budget charge.

Exported regions expose an `observer()` that retains only synchronized state.
Keeping this observer does not keep CUDA backing alive. Released snapshots have
zero physical/virtual capacity and `released = true`.

Growth records physical handles before mapping or initialization. If rollback
fails, `GpuRegionError::Rollback` preserves both errors, and snapshots retain the
remaining handles, mappings and unready bytes. New growth and export are rejected
while cleanup is pending. Retry the original tail boundary, or retire the entire
region after all users are fenced. Partial release remembers completed unmaps
and handle releases, so retry does not repeat them. Failure to release the virtual
reservation remains visible even after all physical handles have been released.

## Validation and release order

Run the host region and compatibility tests with CUDA bindings enabled:

```sh
cargo test -p kapsl-hal --features cuda,cudarc/cuda-12060 -- \
  gpu_region gpu_ipc_region gpu_vmm_region \
  gpu_arena::initialization_tests gpu_arena::admission_tests \
  gpu_arena::tests::aligned gpu_arena::tests::range_free \
  gpu_arena::tests::quota gpu_arena::tests::pool_owner_scope \
  gpu_arena::tests::device_pool_snapshot --test-threads=1
cargo test -p kapsl-hal --no-default-features
```

The new IPC/VMM hardware smoke tests are ignored by default. On a Linux CUDA
host, run them explicitly with `cargo test -p kapsl-hal --features
cuda,cudarc/cuda-12060 -- hardware_ --ignored --test-threads=1`. Full worker
isolation, remote mapping and resize under queued inference still require the
runtime's GPU conformance environment.

This source change prepares HAL 0.3.2. Publish the merged source under
`kapsl-hal-v0.3.2` before migrating the engine's published dependency. The engine continues
using its existing exported-backing implementation until that coordinated update.
The common per-device allocation facade and region selection are later steps.
