# Native allocator host regression tests

The generic native host implements the scoped allocator extension published in
`kapsl-backend-abi = 0.2.0`. Its frozen ABI-v1 prefix points to a retained
`KapslBackendHostScopedAllocatorV1`; the advertised size covers the extension,
and the scoped allocator version is 1. This implementation needs no SDK change.

The host checks device, model and replica identity before allocation. Model and
replica scopes are admitted during initialization and load. Request scopes must
refer to live engine-dispatched request IDs; a batch scope must contain distinct
IDs from the same dispatch. Cancellation revokes new request allocations.
Retired scope IDs cannot be rebound, including after unload/reload. Scoped
adapters cannot use the legacy callback to omit request attribution.

Allocation scope, request IDs and class remain attached to each live handle,
including arenas retained after inference returns. The host's live-memory report
uses those records for governed CUDA bytes. Physical allocation requires an
engine-admitted pool owner, and the existing pool enforces aggregate quotas and
other owners' reservations. Free validates the instance's handle, address and
byte count and synchronizes before returning memory to the pool. Failed frees
stay charged and can be retried.

Unload drains calls and stops new allocations. Successful adapter unload permits
the host to reclaim outstanding handles; failed adapter unload retains memory
until a successful retry or shutdown. Reload and inference are blocked while
cleanup is incomplete. A synchronization failure never makes a range reusable;
terminal cleanup failure retains its charge and storage until process teardown.

## Run locally

From the repository root:

```sh
cargo test --manifest-path kapsl-runtime/Cargo.toml -p kapsl backend::native --locked
cargo test --manifest-path kapsl-runtime/Cargo.toml -p kapsl --features gpu-device-pool --locked -- --test-threads=1
```

Both commands are host-only. The second compiles the production CUDA pool
binding and runs its host tests without a driver, toolkit or GPU. The normal
Linux/macOS/Windows PR test matrix also includes the native host tests.

`tests/native-adapter` builds a test-only `cdylib` against the
published ABI. The tests load it through the production library loader and use
the production initialization, inference and lifecycle bridge. Only the physical
allocator is replaced with a fake admitted pool. The fixture is a dev dependency
and is not built or packaged by engine release builds.

| Advertised surface | Executed assertions |
| --- | --- |
| CUDA / governed / scoped allocator | Extended table size, versions and every required callback; valid model, replica, request and batch allocations; malformed and foreign scopes; quota denial; invalid frees; synchronization and cleanup failures |
| Concurrent inference | Single and batch calls overlap on one instance with separate ownership; unrelated dispatches cannot become one allocation batch |
| Batching | Multiple outputs and the matching release callback; declared batching policy and metrics |
| Streaming | Borrowed chunks, bounded delivery, cancellation on consumer drop, and rejection of a chunk for another request |
| Cancellation | Single, batch and stream cancellation reaches the adapter; cancelled scopes cannot allocate; retained arenas stay charged |
| Memory reporting | Planned model and request reports; live host ledger; metrics retain leaked bytes during failed cleanup |
| Required lifecycle and reports | Descriptor validation, model information, health, load/unload/reload, initialization/load/inference failure cleanup |

Every advertised optional function and every required ABI function is removed
in turn to verify host rejection. KV participation is not advertised by this
fixture; adding that capability without its functions is also rejected.

These tests exercise the loader after the signed-installation boundary. The fake
library is a local test artifact, not a published signed pack. They establish no
ORT provider, platform, output or performance qualification. Stable signed-pack
qualification remains a separate release gate, including the existing 1.5x
startup threshold and infrastructure teardown checks.
