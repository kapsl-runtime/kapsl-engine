# Backend-neutral removal inventory

Status: classification only; no runtime behavior is changed by this document.

Audit base: `origin/develop` at
`84159f0a95387ac6ef4820c68c92c1b3825990a4` (2026-09-02).

The CI-gating correction is tracked separately in engine PR #197. That change
must land before another behavior-changing backend migration PR. Embedded ORT
remains the rollback path until a signed accelerator pack passes an official
stable-release qualification.

## Classification

Each inventory row uses one or more of these classifications:

1. **Keep**: engine-neutral host functionality that remains in the engine.
2. **Move**: backend-specific functionality that moves to
   `kapsl-integrations`.
3. **Retire after qualification**: temporary embedded rollback code that may
   be deleted only after the signed ORT accelerator route passes stable-release
   qualification.
4. **Compatibility**: parsing or translation for old model manifests, flags,
   environment variables, or installation layouts. Compatibility code must be
   isolated from the resolver and generic host contracts and given an explicit
   removal policy.
5. **Rewrite**: tests or documentation that must be rewritten around generic
   backend IDs, capabilities, and signed packs.

A mixed row is intentional. For example, `backend/native.rs` contains a
backend-neutral ABI host that stays and ONNX-specific validation that moves.

## Audit method and dependency result

The requested source scan was run across `kapsl-runtime/crates`. The literal
`ort::` check needs a token boundary because the broad expression also matches
the suffix of `transport::`:

```bash
rg -n '(^|[^[:alnum:]_])ort::' kapsl-runtime/crates
```

The audited patterns currently have this footprint:

| Pattern | Matches | Files |
|---|---:|---:|
| literal `ort::` | 4 | 1 |
| `onnxruntime` | 6 | 3 |
| `ort_pool_allocator` | 3 | 1 |
| `OnnxRuntimeTuning` | 32 | 5 |
| `ResolvedServingBackend` | 31 | 5 |
| `Vllm` | 526 | 13 |
| `LlamaCpp` | 92 | 7 |

The requested Cargo command reports that there is no package named
`kapsl-cli`; the binary package is named `kapsl`. The equivalent command is:

```bash
cargo tree --manifest-path kapsl-runtime/Cargo.toml -p kapsl
```

Its inverse dependency result is:

```text
ort 2.0.0-rc.11
├── kapsl 0.2.3
├── kapsl-backends 0.3.0
│   └── kapsl 0.2.3
└── kapsl-llm 0.3.0
    ├── kapsl 0.2.3
    └── kapsl-backends 0.3.0
        └── kapsl 0.2.3
```

Removing only the direct `ort` entry from the engine manifest would therefore
leave ORT in the runtime graph. Published `kapsl-backends` 0.3.0 has an
unconditional target-specific ORT dependency. It also depends on
`kapsl-llm` 0.3.0 without disabling default features; those defaults include
`onnx`. The engine's own `kapsl-llm` dependency also enables those defaults.

This requires a publish-first SDK change. Either the published SDK crates must
make ONNX optional and default-off for an engine consumer, or the engine must
stop consuming those crates before the Phase 7 dependency acceptance check.
The engine must consume exact published versions; a path dependency or Cargo
patch is not an acceptable bridge.

## Contract findings that block deletion

### Native ABI

Published `kapsl-backend-abi` 0.1.0 already covers the principal native host
surface: discovery, CPU/CUDA/TensorRT capability bits, lifecycle, inference,
batching, streaming, cancellation, memory reports, metrics, health, optional KV
reports, governed allocation/free callbacks, and device synchronization.

The following gaps must be resolved in the SDK contract phase:

- `KapslDeviceAllocationRequestV1` identifies device, allocation class, model,
  and replica, but has no request, generation, epoch, or other allocation-scope
  identity. The current engine rejects nonzero `flags` and `reserved` values,
  so those fields cannot be repurposed informally. Accelerator generation needs
  a versioned, unambiguous ownership/lifetime scope before it can be certified.
- `describe` returns JSON, but the engine currently validates a fixed ONNX
  descriptor shape rather than a published schema for formats, model types,
  tasks, and required capabilities.
- The signed `BackendPackManifest` records backend/profile, platform,
  accelerator, execution mode, memory, entrypoint, hashes, signature, licenses,
  and optional installed-file hashes. It does not declare supported formats,
  model types, tasks, batching, streaming, allocation, or KV capability
  requirements, and it has no explicit build-provenance field. The target
  resolver cannot be implemented from the current manifest alone.
- The ABI is an in-process native contract. No generic managed-backend launch
  and inference protocol is published yet.

The `kapsl-kv-abi` contract should remain limited to KV-memory participation.
It must not become the lifecycle or inference protocol for vLLM, SGLang, or
other managed backends.

### Current native host leaks

The generic native host in `backend/native.rs` is structurally reusable, but it
still:

- accepts `kapsl_backends::OnnxRuntimeTuning` and emits `onnx_tuning` options;
- validates `manifest.backend == "onnx"` against fixed profile aliases;
- constructs the active pack engine with a hard-coded ONNX identity; and
- maps ONNX and llama.cpp names into product-specific `PoolBackend` variants.

`KAPSL_GENERIC_NATIVE_PACKS` also defaults to disabled. That is bridge policy,
not a permanent resolver rule: signed ORT packs become the primary route only
after the required profiles and contract are ready, while embedded ORT remains
an explicit rollback until stable qualification.

The host should accept opaque, signed adapter options and generic backend IDs.
Memory ownership remains engine-governed, but attribution cannot require adding
a new engine enum variant for each backend.

## Source inventory

### Signed-pack hosting, selection, and lifecycle

| Surface | Class | Disposition |
|---|---|---|
| `crates/kapsl-cli/src/backend/manager.rs` | 1, 2, 4, 5 | Keep signature verification, trusted keys, safe extraction, cache locking, quarantine, platform/ABI checks, and generic execution modes. Replace `plan_vllm`, `plan_onnx`, `plan_llama_cpp`, fixed profile enums/constants, and llama-only `kv_mode` validation with descriptor-driven planning. |
| `crates/kapsl-cli/src/backend/bundle.rs` | 1, 2, 4, 5 | Keep verified offline bundle creation/extraction. Replace product-specific closure selection and target mapping with generic signed descriptors. |
| `crates/kapsl-cli/src/backend/selection.rs` | 1, 2, 4, 5 | Keep model-contract validation, explicit-pin failure, deterministic selection, and selection-reason reporting. Replace `ServingBackendPolicy` and `ResolvedServingBackend::{Builtin,LlamaCpp,Vllm}` branches with backend IDs and capability sets. Preserve old aliases only in a compatibility translator. |
| `crates/kapsl-cli/src/runtime/model/load_plan.rs` | 1, 2, 3, 4, 5 | Keep topology, admission, and load-plan ownership. Replace `BackendLoadTuning::Onnx` and `uses_managed_vllm` with opaque adapter configuration and generic execution-mode/capability data. |
| `crates/kapsl-cli/src/runtime/model/backend.rs` | 1, 2, 3, 5 | Keep `MemoryTrackedEngine`, load transactions, reconciliation, and engine-governed admission. Move compute factories and task adapters; retain embedded ONNX construction only through the qualified rollback window. |
| `crates/kapsl-cli/src/runtime/model/replica.rs` | 1, 2, 5 | Keep replica lifecycle and scheduling. Replace direct `ManagedVllmEngine` construction and product-specific metrics with the generic managed host. |
| `crates/kapsl-cli/src/runtime/resources/mod.rs` | 1, 2, 5 | Keep shared runtime resources. Replace the single `managed_vllm` deployment slot with a generic managed-deployment registry. |

`kapsl-core::EngineKind` still encodes ONNX task variants, GGUF, and native
SafeTensors routing. Its model-contract normalization is useful host behavior,
but product-specific engine selection must become requirements data. Legacy
manifest inference belongs in classification 4; resolver branching does not.

### ORT-specific surface

The exact ORT/tuning scan is fully covered by these files:

| File | Class | Disposition |
|---|---|---|
| `crates/kapsl-cli/src/backend/onnx.rs` | 2, 3, 4, 5 | Move fixed CPU/CUDA12/TensorRT10 profiles, provider translation, ORT sidecar entrypoint, and provider validation to integrations. Fold reusable pack activation into the generic manager. Retire the legacy provider-only activator and embedded fallback only after stable qualification. The four direct `ort::` calls are here. |
| `crates/kapsl-cli/src/backend/native.rs` | 1, 2, 3, 5 | Keep ABI loading, table validation, lifecycle, inference, cancellation, reports, callbacks, synchronization, and leak reclamation. Remove the ONNX descriptor branch and typed tuning after the integration contract is published. |
| `crates/kapsl-cli/src/app/config/onnx_session.rs` | 2, 4, 5 | Move ORT argument/configuration translation to the ORT adapter. If old flags remain temporarily, translate them to opaque adapter options at one compatibility boundary. |
| `crates/kapsl-cli/src/runtime/model/load_plan.rs` | 1, 2, 3, 4, 5 | Remove typed `OnnxRuntimeTuning` from the generic plan; see the lifecycle table. |
| `crates/kapsl-cli/src/runtime/model/backend.rs` | 1, 2, 3, 5 | Remove embedded ONNX factories, including the generation path, only after the signed pack covers every retained task. |
| `crates/kapsl-cli/src/runtime/memory/device.rs` | 1, 3, 5 | Retain `GpuDevicePool`, quotas, ownership, admission, metrics, and generic callbacks. Retire the three `kapsl_backends::ort_pool_allocator` registrations and replace product enums with generic attribution after qualification. |
| `crates/kapsl-cli/src/runtime/serving/worker.rs` | 1, 2, 4, 5 | Keep process isolation, supervision, restart, transport, cancellation, and GPU boundary enforcement. Replace ONNX-specific child flags with generic signed adapter options. |
| `crates/kapsl-cli/src/features/providers/installer.rs` and sibling `features/providers/*` | 2, 3, 4, 5 | These install legacy CUDA/TensorRT provider DLL packs and expose provider-specific CLI text. Move provider packaging/validation to the ORT integration; retain only an explicitly deprecated compatibility command if required. |

Inline tests in these owning files inherit classification 5 and must be
rewritten against signed descriptors, generic host behavior, or an explicit
legacy-compatibility fixture.

Additional ORT ownership outside the exact scan:

- `crates/kapsl-cli/Cargo.toml`: direct `ort`, `kapsl-backends`, and
  default-featured `kapsl-llm` dependencies; `gpu-device-pool` also enables
  `kapsl-backends/onnx-cuda-pool` (classes 2 and 3).
- `crates/kapsl-cli/src/app/cli/run.rs`, `app/config/model_loading.rs`, and ORT
  environment constants: backend-specific CLI/config translation (classes 2,
  4, and 5).
- `crates/kapsl-cli/src/main.rs`: imports `BackendFactory` and `LLMBackend` for
  embedded runtime construction (classes 2 and 3).
- Published `kapsl-backends` owns the ONNX adapters, preprocessing, provider
  compatibility, factory, and pool allocator. These must no longer be runtime
  dependencies of the neutral engine (classes 2 and 3).

### llama.cpp-specific surface

| Surface | Class | Disposition |
|---|---|---|
| `crates/kapsl-backend-llama-cpp/*` | 2, 5 | Move the adapter crate, build logic, and implementation into `kapsl-integrations`; publish/package it independently. |
| `crates/kapsl-cli/src/backend/llama_cpp/mod.rs` | 1, 2, 4, 5 | Move profile selection, GGUF/KV estimation, llama-specific ABI validation, and adapter bridge. Reuse the generic native host for library loading, lifecycle, inference, reporting, and cancellation. |
| `crates/kapsl-cli/src/backend/llama_cpp/shared_pool.rs` | 1, 2, 5 | Keep engine KV/memory authority and generic callbacks; move llama-specific callback translation and capability interpretation. |
| `backend/{manager,bundle,selection}.rs` | 1, 2, 4, 5 | Remove `LlamaCppBackendPackProfile` and product branches; retain old backend spellings only as explicit pins translated to a generic ID. |
| Cargo features `gguf-native`, `gguf-cuda`, and `gguf-cuda-shared-kv` | 2, 4, 5 | Replace engine-linked compute features with signed pack capabilities. Provide a migration message for obsolete build flags. |
| `crates/kapsl-cli/src/tests/packaging_tests.rs` | 5 | Replace llama.cpp-specific packaging expectations with signed-descriptor and integration-artifact contract fixtures. |

### vLLM-specific surface

| Surface | Class | Disposition |
|---|---|---|
| `crates/kapsl-cli/src/runtime/managed/mod.rs` | 1, 2, 5 | Move vLLM version pins, Python discovery/probing, connector schema, planner invocation, KV geometry/sizing, CLI arguments, environment, chat translation, and token estimation. Extract a generic supervised-process host with readiness, restart, process-group teardown, health, metrics, cancellation, and protocol transport. |
| `crates/kapsl-cli/src/runtime/managed/bridge.rs` | 1, 2, 5 | Retain generic HTTP/SSE transport, timeouts, cancellation, and wire-error handling; rename and remove vLLM response assumptions. Backend-specific request/response translation moves. |
| `crates/kapsl-cli/src/app/startup/bootstrap.rs` | 1, 2, 5 | Replace `ensure_vllm`, `ManagedVllmDeployment::prepare`, and vLLM preflight with generic selected-pack preparation and admission. |
| `crates/kapsl-cli/src/http/openai/chat.rs` | 1, 2, 4, 5 | Keep the public OpenAI endpoint and generic streaming relay. Move `ManagedVllmOpenAiMode`, vLLM detection, and translation; isolate the old translated mode as rollback compatibility if retained. |
| `crates/kapsl-cli/src/runtime/kv/control.rs` | 1, 2, 5 | Keep KV admission, grants, leases, readiness fencing, snapshots, resize accounting, and teardown. Rename vLLM-specific types/methods and replace the hard-coded `registration.backend != "vllm"` check with grant-bound generic identity/capabilities. |
| `crates/kapsl-cli/src/runtime/kv/mod.rs` | 1, 2, 5 | Keep KV-control startup and validation. Replace `PreparedManagedVllmDeployment` defaults and the `resources.managed_vllm()` hook with generic managed participants. |
| `crates/kapsl-cli/src/runtime/{model/load_plan.rs,model/replica.rs,resources/mod.rs}` | 1, 2, 5 | Replace vLLM-specific topology and deployment types with the managed protocol. |
| `backend/{manager,bundle,selection}.rs` | 1, 2, 4, 5 | Remove the managed-vLLM profile constant and product-specific planning. The string `vllm` may remain as an explicit backend ID, not as a resolver branch. |
| `crates/kapsl-cli/src/tests/packaging_tests.rs` | 5 | Replace managed-vLLM packaging expectations with generic managed-pack contract fixtures. |

The generic managed protocol must cover launch configuration, model/request
lifecycle, inference and streaming, cancellation, health/readiness, metrics,
memory reports, teardown, and optional KV-ABI participation. It must not assume
Python, an OpenAI HTTP server, CUDA, or vLLM-specific planner JSON.

### Remaining built-in compute and SDK coupling

Full neutrality requires more than ORT retirement:

- `main.rs` and `runtime/model/backend.rs` use
  `kapsl_backends::BackendFactory` and `kapsl_llm::LLMBackend` for embedded
  compute. SafeTensors/GGUF implementations move to integrations (class 2).
- `kapsl-llm` also supplies prompt templates, model-asset discovery, RAG prompt
  helpers, shared allocators, and scheduler types to HTTP/RAG/runtime modules.
  Neutral utilities must move to appropriate published core, loader, RAG, or
  engine-API crates before the engine can drop `kapsl-llm`; they must not keep a
  compute crate in the runtime graph merely for helpers (classes 1, 2, and 5).
- The final engine must remove its runtime dependency on `kapsl-backends` and
  stop compiling backend implementations through Cargo features.

## Packaging, release, conformance, and documentation inventory

Backend compilation and packaging currently live in the engine release. The
integration repositories must produce immutable signed artifacts; the engine
release should verify and consume them.

| Owner today | Class | Files/actions |
|---|---|---|
| ORT packaging | 2, 3, 5 | `.github/scripts/collect-ort-sidecars.sh`, `onnx-backend-pack-entrypoint.c`, `package-linux-onnx-backend-packs.sh`, `package-linux-ort-cpu-backend.sh`, `package-linux-ort-accelerator-backends.sh`, `package-linux-provider-packs.sh`, and their contract/package tests. Move compilation/provider validation to integrations; keep legacy embedded sidecars only through stable qualification. |
| llama.cpp packaging | 2, 5 | `.github/scripts/package-linux-llama-cpp-backend-packs.sh`, performance scripts, package/contract tests, and `.github/workflows/lazy-llama-cpp-backend-pack-gpu-certification.yml`. Move adapter build, package, and backend conformance to integrations. |
| vLLM packaging | 2, 5 | `.github/scripts/bootstrap-vllm-backend.sh`, `package-linux-vllm-backend.sh`, the managed lock file, SDK/wheel verification, managed conformance/benchmark scripts, and `.github/workflows/vllm-shared-pool-conformance.yml`. Move backend-specific implementation to integrations; keep engine-side stable gating and generic host conformance. |
| Index and model-package validation | 1, 2, 4, 5 | Keep signing/index generation and model-contract validation in `.github/scripts/generate-backend-index.py` and `validate_model_package_backend.py`, but replace hard-coded ORT/llama.cpp/vLLM matrices with descriptor-driven fixtures. |
| Mixed-backend memory tests | 1, 2, 5 | Keep generic memory-isolation and teardown assertions from `certify-mixed-backend-concurrency.py` and GPU-pool integration tests; replace named backend construction with signed test packs. Real-device execution remains stable-only. |
| Engine release assembly | 1, 2, 3, 5 | `.github/workflows/{release-runtime-installers,beta-runtime-installers}.yml` currently check out backend sources and build ORT, llama.cpp, and vLLM artifacts. Change these to consume immutable signed integration artifacts. Embedded ORT assembly remains only until the bridge qualification succeeds. |
| Installers and images | 2, 3, 4, 5 | `installers/install.sh`, `docker/Dockerfile.{cpu,cuda,tensorrt}`, and Docker release scripts contain ORT sidecar and managed-vLLM assumptions. Make pack installation descriptor-driven; preserve old flags/layouts only through a documented compatibility path. |
| Docs | 4, 5 | `docs/backend-packs.md`, `configuration.md`, `deployment.md`, `http-api.md`, `model-packaging.md`, and `VLLM_MEMORY_AND_BRIDGE_PLAN.md` describe product-specific routing and flags. Rewrite them as each migration lands; do not document embedded fallback as automatic. |

Backend-specific correctness and performance suites should move with their
adapters. Engine PRs retain host-only ABI, pack loading, signature/tamper,
lifecycle, cancellation, packaging-contract, and generic fake-backend tests.
Full CPU performance and real GPU qualification remain stable/manual gates as
defined by PR #197; a PR must never provision a real GPU.

## Required sequencing

1. Merge the isolated CI-gating correction in PR #197.
2. Version and publish any required SDK contracts/features, promote SDK
   `develop` to `main`, and verify crates.io availability.
3. Build and host-test signed ORT `onnx/cpu`, `onnx/cuda12`, and
   `onnx/tensorrt10` packs in integrations. Do not silently substitute CPU.
4. Add descriptor-driven selection and the signed-pack primary route while
   retaining embedded ORT as an explicit, logged rollback.
5. Run an official stable release qualification. It must prove the signed
   accelerator pack was selected and teardown completed.
6. Only then retire the direct and transitive embedded ORT surfaces.
7. Migrate llama.cpp, publish the generic managed protocol, migrate vLLM and
   SGLang, and move remaining built-in compute in separate PRs.

No step may use a sibling path dependency, temporary Cargo patch, published
branch rebase, reused stable tag, or PR/branch GPU run.

## Acceptance queries

Phase 7 is not complete until all of these are true:

```bash
cargo tree --manifest-path kapsl-runtime/Cargo.toml -p kapsl -i ort
rg -n '(^|[^[:alnum:]_])ort::' kapsl-runtime/crates
rg -n 'ort_pool_allocator|OnnxRuntimeTuning' kapsl-runtime/crates
```

The first command must report no `ort` package and the source scans must be
empty in runtime code. An ONNX model with no compatible signed pack must return
a clear `no compatible backend` error, never activate embedded ORT, and never
fall back from CUDA/TensorRT to CPU. CPU results must remain equivalent, the
native ORT route must stay in-process without an added tensor-serialization or
process boundary, and accelerator allocations must continue through the
engine's governed callbacks.

Engine-wide neutrality additionally requires:

```bash
cargo tree --manifest-path kapsl-runtime/Cargo.toml -p kapsl -i kapsl-backends
rg -n 'ResolvedServingBackend|ManagedVllm|LlamaCppBackendPackProfile|OnnxBackendPackProfile' \
  kapsl-runtime/crates/kapsl-cli/src
```

Product names may remain in compatibility parsing, explicit configuration, and
reporting, but not in resolver, loader, lifecycle, or process-launch branches.
The definitive test is a signed fake backend for a known model contract that
can be installed, selected, loaded, exercised, and unloaded without modifying
or recompiling `kapsl-engine`.
