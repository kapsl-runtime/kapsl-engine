# Lazy backend packs

Kapsl resolves deployment backends from the model contract and the detected
host. Clients do not install or start a backend directly, and an `.aimod`
package cannot provide a download URL or executable path.

## Online use

```bash
curl -fsSL https://downloads.kapsl.net/install-beta.sh | sh
kapsl run model.aimod
```

Before model initialization, Kapsl validates the package contract, selects the
backend, and performs preliminary memory admission. If the selected backend is
a lazy pack, Kapsl then downloads its runtime-version-specific artifact,
verifies the signed index, artifact signature, checksum, platform, accelerator,
ABI, and installed files, and atomically activates it. A second run uses the
validated cache.

The beta rollout publishes managed vLLM, one standard-ABI ORT CPU candidate,
two legacy ONNX accelerator profiles, and two in-process llama.cpp profiles:

```text
llama-cpp/cpu
llama-cpp/cuda12
onnx/cpu
onnx/cuda12
onnx/tensorrt10
vllm/cu130-flash-attn
```

An ONNX model resolves exactly one profile after provider policy and declared
fallbacks are resolved. The CPU candidate contains and activates no CUDA
libraries. Its signed manifest declares `adapter_abi: kapsl-backend-v1`; the
legacy CUDA/TensorRT provider bundles omit that field and cannot be routed to
the generic host accidentally. While the candidate gate is off, the standard
CPU pack remains dormant and the embedded ORT implementation is the rollback.
TensorRT is never an automatic fastest-provider upgrade: the package must name
`tensorrt` as its preferred provider or an allowed fallback.
GGUF models never trigger lazy ONNX or vLLM pack installation, and ONNX models
never download vLLM.

GGUF models resolve a llama.cpp pack only when requested. Kapsl loads its
versioned C function table from a canonical pack-local absolute path; Rust
trait objects never cross the library boundary. ABI v1 covers initialization,
load/unload, inference and streaming, cancellation, memory/allocation reports,
metrics, logging, shared-pool callbacks, and shutdown. The CPU pack is the
default lazy profile for the portable Linux x86_64 core.

During the CUDA beta transition, the established eager
`gguf-cuda-shared-kv` build remains the certified reference and rollback path.
The shared-pool `llama-cpp/cuda12` candidate calls the versioned host callback
table and allocates KV blocks and its device block table from the same
Kapsl-owned `GpuDevicePool` used by in-process ONNX. The pack wraps that raw
descriptor inside llama.cpp; it does not create another pool, CUDA process, or
tensor-IPC path. Its signed manifest declares `kv_mode: shared_pool`, and core
and pack both fail closed if either the callbacks or capability bit are absent.

A separately signed `kv_mode: native` CUDA pack remains available as the
backend-owned-KV rollback. It is rejected unless an operator explicitly sets
`KAPSL_LLAMA_CPP_ALLOW_NATIVE_KV=1`; that override cannot authorize a pack
whose signed mode is `shared_pool`. Release jobs build the shared candidate
only from an exact certified `kapsl-sdk` commit. The public-candidate GPU gate
then checks first-use download/cache reuse, allocation ownership and cleanup,
single-process CUDA participation, and the 2% throughput / 5% latency budgets
before the shared pack is promoted.
The CUDA archive carries its resolved non-driver CUDA dependency closure and
redistribution notices beside the entrypoint with an `$ORIGIN` runpath; host
NVIDIA driver libraries are explicitly excluded.

ONNX Runtime remains in the Kapsl process. Provider objects are opened from
canonical pack-local absolute paths and retained for the process lifetime;
Kapsl does not modify process-wide `LD_LIBRARY_PATH`. The pack carries a
versioned native entrypoint descriptor, ORT libraries, the selected execution
provider, its user-space accelerator dependency closure, compatibility/memory
metadata, and license notices. Linux x86_64 is the first published ONNX pack
platform; other platforms retain their eager in-process provider layout during
the beta rollout.

The backend-neutral native-pack host is available as a migration and
certification gate through `KAPSL_GENERIC_NATIVE_PACKS=1`. Only a pack whose
signed `adapter_abi` is `kapsl-backend-v1` is eligible. With that gate on, the
pack must export the published ABI v1 entrypoint and Kapsl will not construct
the embedded ORT backend if loading or initialization fails. A legacy pack
continues through its provider loader even when CPU certification enables the
generic gate, so the CPU migration cannot change CUDA/TensorRT routing. The
signed accelerator profile must exactly match the adapter's CPU, CUDA, or
TensorRT capability bits. CUDA and TensorRT adapters must allocate device
memory through the runtime-owned `GpuDevicePool` callbacks.

The standard-ABI ORT family also binds the versioned pack identity exactly:
`cpu` maps to accelerator `cpu`, `cuda12` maps to `cuda`, and `tensorrt10`
maps to `tensorrt`. Provider aliases are normalized only at the engine policy
boundary. The adapter receives the signed canonical provider, and activation
rejects a descriptor unless its one compiled profile, ABI, wire format,
execution mode, and governed-memory declaration match the signed manifest and
static function table.

This path stays in-process: tensor buffers cross the adapter boundary as
borrowed views, and ORT's allocator forwards directly to the same Kapsl-owned
pool. It introduces no backend RPC, CUDA IPC, tensor serialization, or second
GPU allocation authority. The host supplies the canonical signed-pack root and
the resolved per-model ORT tuning in initialization options, so the adapter can
resolve only pack-local runtime libraries and does not reread competing process
configuration. The gate defaults off until the out-of-tree ORT adapter has
passed CPU parity, GPU memory-ownership, unload/reload, and stable release
conformance. An invalid gate value is an error rather than a request to fall
back.

Release jobs build the CPU candidate from the exact `kapsl-integrations`
commit in `.github/ort-integration.lock`. The adapter's committed Rust
toolchain is installed and verified independently of the engine toolchain;
the resulting archive records its source commit and is accepted only after the
engine validates its payload, provenance, file hashes, and standard ABI marker.
The same exact checkout now exposes a prepared accelerator handoff for
`cuda12` and `tensorrt10`. It authenticates Microsoft's official ORT GPU
archive, closes and normalizes every non-driver CUDA/TensorRT dependency, and
emits the same standard-ABI manifest/provenance contract. Release workflows
continue publishing the legacy accelerator rollback until an official stable
release completes real GPU ownership, unload, reproducibility, and teardown
qualification; preparing an archive does not promote it.
The engine's existing Ed25519 backend-index publisher remains the sole owner of
official release signing.

Host CI pins the canonical integrations-owned parity entrypoint by path and
SHA-256, plus its tiny ONNX model source, in
`.github/ort-cpu-parity.lock.json`. The exact integrations commit in
`.github/ort-integration.lock` supplies both the adapter and that
conformance contract. CI builds the pack, constructs a signed offline bundle,
preinstalls that bundle through the normal backend manager, and runs two full
ABBA embedded/candidate blocks. Four captures per route make the startup median
resistant to one host-scheduler outlier without weakening its gate. The
dedicated ORT CPU conformance workflow
builds the adapter once for both release-handoff validation and the longer
performance comparison; installer smoke remains a separate quick job with no
performance thresholds. This CPU-only forward-path conformance does not
provision a GPU and does not by itself authorize removing embedded ORT. The
broader retirement gate still requires every supported CPU task class plus the
separate CUDA/TensorRT memory ownership and lifecycle suites.

When a native adapter advertises ABI v1 cancellation, Kapsl bridges each
request's `CancellationToken` to the adapter's `cancel(request_id)` hook on one
process-wide event-driven cancellation runtime. This does not poll and does not
create an operating-system thread per request. The borrowed request callback
remains available for cancellation that races initial dispatch, while the
explicit hook can interrupt a backend run already in progress. Model load,
unload, and shutdown take an exclusive cancellation guard so a late task
cannot race lifecycle mutation or call a retired adapter handle.

Inspect a decision without running the model:

```bash
kapsl backend-plan model.aimod
```

The JSON result includes `selected_backend`, `profile`, `installed`,
`download_required`, `download_bytes`, `memory_admission`, and
`execution_mode`. Obvious GPU overcapacity is reported without fetching the
backend artifact or starting a backend process.

Administrative commands are available for deterministic preparation and cache
maintenance:

```bash
kapsl backend ensure model.aimod
kapsl backend list
kapsl backend list --json
kapsl backend prune
kapsl backend prune --old-versions
```

## Cache and installation safety

The default cache is:

```text
~/.local/share/kapsl/backends/
└── <runtime-version>/
    └── <backend>/<profile>/
```

Installation is serialized by a filesystem lock. Downloads enter a staging
directory, are bounded by their signed sizes, and are verified before an atomic
rename makes the pack visible. Interrupted, corrupt, mismatched, or concurrent
installs cannot leave a usable partial pack. Kapsl fails closed; it does not
silently substitute a different backend.

Official binaries embed one or more Ed25519 backend-index public keys. The
release pipeline confirms that the private signing key matches an embedded
public key before publishing an index. Only HTTPS artifact URLs from that
signed index are accepted in production.

## Offline bundles

Prepare a bundle on a connected machine:

```bash
kapsl bundle model.aimod --output model.kapsl-bundle
```

Prepare for a different deployment target:

```bash
kapsl bundle model.aimod \
  --target linux-x86_64-cuda \
  --output model.kapsl-bundle
```

Multiple models share one copy of each required backend pack:

```bash
kapsl bundle model-a.aimod model-b.aimod \
  --output production.kapsl-bundle
```

If the signed release files are already on the preparation host, bundle them
without downloading the archives again:

```bash
KAPSL_BACKEND_INDEX_PATH=/release/backend-index.json \
KAPSL_BACKEND_PUBLIC_KEYS="$PUBLIC_KEYS" \
kapsl bundle model.aimod \
  --backend-artifacts-dir /release \
  --target linux-x86_64-cpu \
  --output model.kapsl-bundle
```

The local directory is only a source for offline-bundle creation. Kapsl maps
the filename from the signed HTTPS index entry, confines it to that directory,
and verifies the signed size and digest before copying it into the bundle. It
does not permit local artifacts during an ordinary online or offline model
run.

Copy the resulting file to the offline host and run it directly:

```bash
kapsl run model.kapsl-bundle
```

To populate and validate the backend cache without starting a server, use the
same verified activation path:

```bash
kapsl backend ensure model.kapsl-bundle --offline
```

Kapsl verifies the bundle checksums, signed backend index, signed pack
artifacts, runtime version, target, model contracts, and required-pack closure.
It installs included packs through the same atomic backend manager, then passes
the embedded `.aimod` paths to the normal memory-admission and startup flow.
The bundle's verified index is anchored into the ordinary backend cache, so a
subsequent offline validation never needs to fetch the index. A different
signed index claiming the same immutable runtime release is rejected.
There is no weaker raw-archive import path.

For an ordinary `.aimod` run that must not use the network, use `--offline` or
set `KAPSL_OFFLINE=1`. If its pack is not already cached, Kapsl reports the
missing backend and suggests `kapsl bundle` or `kapsl backend ensure` on a
connected machine.

## Operator overrides

| Variable | Purpose |
|----------|---------|
| `KAPSL_BACKEND_CACHE_DIR` | Override the backend cache root. |
| `KAPSL_BUNDLE_CACHE_DIR` | Override the verified extracted-bundle cache. |
| `KAPSL_OFFLINE=1` | Disable backend-index and artifact network access. |
| `KAPSL_LAZY_BACKENDS=0` | Disable automatic lazy installation and require a preinstalled backend. |
| `KAPSL_LAZY_ONNX_PACKS=0` | Keep the eager/legacy ONNX provider layout during the compatibility window. Linux x86_64 defaults to lazy ONNX packs. |
| `KAPSL_GENERIC_NATIVE_PACKS=1` | Enable a signed `kapsl-backend-v1` ONNX candidate. It defaults off, leaving that pack dormant and embedded ORT active; once enabled, candidate load/initialization failures are fail-closed. |
| `KAPSL_LAZY_LLAMA_CPP_PACKS=0` | Keep the eager/compiled llama.cpp layout. The portable Linux x86_64 core defaults to lazy CPU packs when no eager GGUF feature is compiled. |
| `KAPSL_LLAMA_CPP_ALLOW_NATIVE_KV=1` | Explicitly allow a signed CUDA pack whose `kv_mode` is `native`. Shared-pool packs and the eager shared-KV profile do not require or consume this rollback override. |
| `KAPSL_PROVIDER_PATH` | Additional Kapsl provider-manifest roots. Verified lazy ONNX pack roots are appended automatically; this is not a loader search path. |
| `KAPSL_BACKEND_INDEX_URL` | Override the signed index URL for a private mirror. |
| `KAPSL_BACKEND_INDEX_PATH` | Read a local signed index and adjacent `.sig` file. |
| `KAPSL_BACKEND_PUBLIC_KEYS` | Add trusted raw Ed25519 public keys for development or controlled key rotation. Official release keys are embedded. |

`KAPSL_VLLM_PYTHON` and `KAPSL_VLLM_BUNDLE` remain development and legacy
layout overrides. Every discovered vLLM environment still has to match the
certified Python, PyTorch, CUDA, vLLM, connector, and shared-pool profile tuple.
