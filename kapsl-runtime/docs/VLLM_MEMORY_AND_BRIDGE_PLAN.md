# Managed vLLM memory and bridge remediation plan

Status: all five implementation phases are present on
`feature/vllm-complete-remediation` and host-verified. The pinned self-hosted
GPU workflow is the remaining acceptance step; it must pass the native
attention, exact-memory, live-resize, semantic, restart, 1->2->1 scaling,
mixed-backend, reclamation, and direct-vLLM performance gates before any pull
request or release allowlist is created.

Date: 2026-08-26

## Executive decision

Managed vLLM should stop using a fixed fraction of each GPU as its default
memory policy. Kapsl should calculate an exact KV-cache requirement from the
model's certified cache geometry, the configured context length, and a target
per-replica concurrency. It should reserve those bytes transactionally through
`MemoryAuthority`, pass the granted byte count to vLLM, and require the
connector's CUDA-IPC registration to match that grant exactly.

The OpenAI bridge should remain behind Kapsl authentication, scheduling,
priority, session affinity, memory admission, and lifecycle supervision, but it
should stop translating every request and streamed token through multiple JSON
and thread/channel layers. A typed OpenAI wire path should forward one
normalized request body and relay vLLM's already OpenAI-compatible response
bytes with bounded asynchronous backpressure.

These are two related but independent fixes:

1. **Exact initial KV sizing** removes the large idle reservation and makes
   every new replica fit the memory that Kapsl has actually granted.
2. **The OpenAI wire fast path** removes avoidable CPU, latency, connection, and
   allocation overhead from the out-of-process bridge.

Fixed exact pools retain one startup-sized CUDA-IPC backing. The optional
Phase-5 elastic profile instead reserves a stable CUDA virtual address and maps
or unmaps Kapsl-owned physical segments behind it, with synchronized native
vLLM tail-block growth and retirement.

## What the benchmarked pre-remediation implementation did

### Pre-remediation memory path

At the benchmarked commit, `ManagedVllmSettings` defaulted
`gpu_memory_utilization` to `0.5` and accepted values from `0.1` through `0.9`.
`ManagedVllmProcess::build_command` passed that value to every managed vLLM process as
`--gpu-memory-utilization`.

The resulting allocation path is:

```text
0.5 of physical GPU memory
        |
        v
vLLM profiles weights, non-Torch memory, and peak activations
        |
        v
vLLM assigns the remainder to a fixed KVCacheConfig
        |
        v
KapslConnectorV1 reports that exact geometry to Kapsl
        |
        v
Kapsl allocates one dedicated CUDA-IPC backing of that size
        |
        v
the connector replaces vLLM's packed KV allocator with tensor views
into the Kapsl-owned backing
```

There is no second PyTorch KV allocation in `shared_pool` mode. The connector's
allocation hook intentionally replaces vLLM's raw packed-buffer allocation.
The memory problem is over-sizing, not duplicate KV storage.

The current `shared_pool` name also needs precise interpretation:

- Kapsl owns and accounts for the physical allocation.
- The Kapsl process exports it and the vLLM process imports it through CUDA IPC.
- All requests in that vLLM instance share vLLM's native block allocator over
  the backing.
- The backing is nevertheless dedicated to one participant/pool/device.
- It is not taken from Kapsl's general CUDA allocator slab.
- It is not dynamically shared with llama.cpp, another vLLM replica, or another
  model.
- Request leases grant aggregate logical capacity; vLLM still chooses native
  block indices.

Consequently, unused blocks remain physically resident and appear as used VRAM
in NVML until the participant is retired and the whole backing is released.

### Pre-remediation OpenAI bridge path

At the benchmarked commit, managed vLLM chat requests followed this path:

```text
client JSON
  -> Warp parses ChatCompletionRequest
  -> Kapsl builds a JSON marker envelope
  -> envelope is copied into a UTF-8 BinaryTensorPacket
  -> InferenceService / scheduler / replica selection
  -> ManagedVllmEngine parses the envelope into serde_json::Value
  -> ManagedVllmEngine builds and serializes another OpenAI request
  -> blocking GET /health before the inference request
  -> new ureq Agent and loopback TCP POST
  -> one OS thread per streaming request
  -> BufReader splits SSE into lines
  -> every data event is parsed into serde_json::Value
  -> content is copied into a BinaryTensorPacket
  -> unbounded MPSC channel back to Tokio
  -> the OpenAI route copies text into its stop-sequence buffer
  -> Kapsl serializes a new chat.completion.chunk event
  -> client
```

The supervisor already performs periodic health checks, so the synchronous
health request on every inference duplicates lifecycle monitoring. Constructing
a new HTTP agent for each request also prevents reliable keep-alive reuse. The
streaming implementation adds one operating-system thread and an unbounded
channel per request, and both sides of the bridge decode and re-encode JSON that
is already in the requested wire format.

## Evidence and observed impact

The 2026-08-26 RTX 4090 benchmark used Kapsl engine commit
`c53ea92562eacf308f094134f4e4bcf673f2186e`, SDK commit
`a7b719ee383ed9acb0626aaa46f91702f2bec210`, vLLM
`0.26.1rc1.dev1130+g2ec6f0d71`, and connector `0.5.0`.

For the 0.5B BF16 model on a 23.52 GiB visible device, vLLM reported:

- target memory at `gpu_memory_utilization=0.5`: 11.76 GiB;
- weights plus non-Torch memory: 1.26 GiB;
- peak activations: 0.08 GiB;
- KV cache: 10.42 GiB;
- cache capacity: 910,480 tokens, or approximately 889 full 1,024-token
  sequences.

Kapsl authorized a matching 11,187,978,240-byte CUDA-IPC allocation. During the
concurrency-16 workload, vLLM reported no more than approximately 0.5% KV-cache
usage. Most of the 10.42 GiB backing was therefore idle but unavailable to any
other backend.

Against direct vLLM with matching engine settings, one Kapsl replica showed:

- output-token throughput changes of -3.2%, -4.7%, -4.4%, and -8.9% at
  concurrency 1, 4, 8, and 16;
- median output-token throughput change: -4.6%;
- median p95 latency change: +4.1%;
- median TTFT change: +19.6%;
- device-wide VRAM delta: +0.382 GiB.

Those aggregate measurements do not individually attribute cost to health
checks, connection churn, JSON translation, thread scheduling, or Kapsl's CUDA
context. Instrumentation must be added before claiming the contribution of any
single item. The source path nevertheless contains avoidable work in all of
those categories.

Default autoscaling also attempted a second managed vLLM replica. The retained
run contains 36,926 connection-refused log entries and 41,858 AIPerf empty-stream
errors. That run does not prove that fixed memory sizing was the sole cause, but
two processes each independently targeting 50% of the same device leave no
safe room for CUDA contexts, workspaces, Kapsl allocations, or another model.
The current policy is structurally unsafe for autoscaling even without that
failure result.

## Detailed problem statement

### P1. The fraction is an engine-local policy, not a Kapsl memory decision

Every managed process interprets `0.5` against the complete visible device.
The value is not divided by replica count, adjusted for co-resident models, or
recalculated from the authority's committed and observed state. A scale-up
therefore repeats the same claim rather than asking Kapsl what remains.

### P2. A small model receives a cache sized from card capacity rather than work

After vLLM profiles the model, all remaining bytes below the percentage target
become KV capacity. For a small model on a large GPU, model size has little
effect on the final footprint. The benchmark allocated enough cache for roughly
889 maximum-length requests while exercising only 16 concurrent requests.

### P3. Kapsl learns the exact geometry only after vLLM has chosen the size

The connector receives a completed `KVCacheConfig`. It can replace the physical
allocation, validate tensor aliases, and account the exact geometry, but it
cannot retroactively change `num_blocks`. Ownership moved to Kapsl; sizing
authority did not.

### P4. The planned model report omits the future KV backing

`managed_vllm_memory_report` currently estimates SafeTensors weights and a
workspace. The separate external-participant registration later charges the KV
backing. This avoids double accounting once the backend is running, but there
is no transactional reservation connecting model admission, the selected KV
budget, child startup, and connector registration. Concurrent model loads or
replica starts can all plan against the same apparent headroom before the first
connector allocation is visible.

### P5. The current backing is fixed and participant-private

The CUDA-IPC provisioner allocates one isolated contiguous region per
participant/pool/device. vLLM creates fixed packed tensor views and a fixed
native block table over it. Free vLLM blocks can be reused by that vLLM process,
but cannot be returned to another backend while the process remains loaded.
This reduces neither long-lived idle reservation nor cross-backend
fragmentation in the way a genuinely elastic common pool would.

### P6. Restart and autoscale reuse a static process specification

The supervisor rebuilds the same command after a restart. Even if device
pressure changed, the process asks for the original percentage again. A new
replica likewise receives the package default rather than a per-replica grant.

### P7. The request bridge performs a second control request per inference

`infer` and `infer_stream` call `/health` immediately before sending the real
request. This adds one serialization point and one loopback request to every
inference while a separate supervisor already checks health every two seconds.
A process can still fail after the health request, so this does not eliminate
the normal request-failure race.

### P8. HTTP connection pools are not reused effectively

The request path constructs a new `ureq::Agent` for each inference so it can set
a request timeout. Health checks construct another agent. Request-level timeout
overrides can be applied to a shared agent; rebuilding the agent is unnecessary.

### P9. Streaming is thread-per-request with an unbounded handoff

Each stream starts a blocking operating-system thread, reads line-oriented SSE,
and sends packets through an unbounded Tokio channel. At increasing concurrency
this creates avoidable scheduler activity, memory growth risk under downstream
backpressure, and delayed cancellation when the reader is blocked.

### P10. OpenAI JSON is decoded and encoded repeatedly

Kapsl already receives an OpenAI chat request, and vLLM already returns OpenAI
chat responses. The current abstraction reduces the exchange to prompt text and
token strings, requiring an internal marker envelope and reconstructing every
response event. It also makes Kapsl usage counts approximate when vLLM has exact
token counts.

### P11. Replica readiness is not one atomic state

Process liveness, HTTP readiness, connector attachment/activation, scheduler
routability, and supervisor restart state are separate observations. A replica
must not be added to routing or remain routable unless the child is healthy and
its exact Kapsl KV binding is attached and active. Fixing bridge health checks
without defining this state would trade overhead for intermittent routing
failures.

## Goals

1. Make Kapsl the source of truth for the initial KV byte budget.
2. Size cache capacity from declared work: context length and target concurrent
   sequences, not an arbitrary fraction of card capacity.
3. Reserve model, workspace, and KV capacity without races or double charging.
4. Recalculate and independently admit every new or restarted replica.
5. Keep the connector's no-copy, no-second-allocation property.
6. Keep authentication, model resolution, priority, session affinity,
   cancellation, memory admission, scheduling, and lifecycle supervision in
   front of vLLM.
7. Remove per-request health calls, connection-pool churn, thread-per-stream,
   unbounded bridge buffering, and JSON event reconstruction from the managed
   OpenAI fast path.
8. Preserve exact upstream HTTP/SSE errors and exact usage where possible.
9. Provide metrics that distinguish granted, occupied, active, and idle KV
   capacity as well as each bridge latency stage.
10. Retain explicit rollback paths until GPU and performance gates pass.

## Non-goals for the first delivery

- Sharing model weights, CUDA contexts, or general vLLM workspaces across
  processes.
- Silently moving KV data between llama.cpp and vLLM; their cache formats are
  different.
- Live in-place resizing of vLLM's current single packed allocation.
- Bypassing Kapsl admission or routing by exposing vLLM's listener publicly.
- Changing attention backends without a new certified connector profile.
- Claiming that the existing Kapsl general CUDA slab and a managed vLLM
  participant already form one physical pool; they do not.

## Proposed memory architecture

### 1. Replace the default fraction with an explicit KV policy

Add a package-level policy with `auto` as the default:

```yaml
metadata:
  serving:
    backend: vllm
    vllm:
      max_model_len: 4096
      kv_cache:
        mode: auto
        target_concurrency: 16
        headroom_percent: 20
        min_bytes: 268435456
        max_bytes: 2147483648
```

Proposed semantics:

- `mode: auto` calculates an exact byte request.
- `target_concurrency` is per replica. If omitted, use the resolved Kapsl batch
  or delegated-concurrency target rather than GPU size.
- `headroom_percent` covers block rounding, prefix-cache retention, and normal
  request-length variance. It must not turn into a percentage of total VRAM.
- `min_bytes` is an optional operator floor, but it cannot be lower than the
  cache required for one `max_model_len` sequence.
- `max_bytes` is an optional anti-hoarding cap.
- `mode: fixed` requires an exact `bytes` value and validates that it can hold
  at least one maximum-length request.
- `gpu_memory_utilization` becomes an explicitly selected legacy mode during a
  migration window. It must not remain the implicit default.

Do not silently reinterpret an explicitly authored
`gpu_memory_utilization`. Emit a deprecation warning with the calculated
exact-byte recommendation, and require `kv_cache.mode: legacy_fraction` if the
operator wants old behavior after the compatibility window.

### 2. Obtain certified cache geometry before allocating the pool

Kapsl must know bytes per cache block before launching the final API process.
Hand-maintaining model-family formulas in Rust is unsafe for hybrid cache
groups, sliding-window models, MLA, cache dtype changes, tensor parallelism,
and future vLLM layouts.

Add a version-pinned planning entry point to the certified connector bundle,
for example:

```text
python -m kapsl_vllm_connector.plan \
  --model <hf-directory> \
  --max-model-len <n> \
  --tensor-parallel-size <n> \
  --attention-backend FLASH_ATTN
```

It should emit machine-readable JSON containing at minimum:

- connector, vLLM, profile, and layout versions;
- cache groups and layer membership;
- block size in tokens;
- bytes per block per rank/device and total bytes per block;
- cache dtype and element size;
- minimum blocks required for one `max_model_len` sequence;
- any model-specific alignment or fixed overhead;
- whether the geometry is supported by the certified shared-pool profile.

The helper must use the same pinned vLLM cache-spec code as the runtime. It
must fail closed when the layout cannot be planned without a new profile. A
simple Rust formula may exist only as a tested diagnostic cross-check for
standard MHA/GQA models, not as the authority.

The certified environment validation must include this planner and its output
schema version.

### 3. Calculate desired blocks from work, not card size

For every rank/device:

```text
sequence_blocks = ceil(max_model_len / block_size_tokens)
base_blocks     = sequence_blocks * target_concurrency
prefix_blocks   = configured prefix-retention allowance
desired_blocks  = round_up(base_blocks + prefix_blocks, profile_alignment)
desired_blocks  = add_bounded_headroom(desired_blocks)
desired_bytes   = sum_over_cache_groups(desired_blocks * bytes_per_block)
```

The calculation must use checked integer arithmetic. It must account for cache
groups independently when their geometry differs. Tensor-parallel ranks must
receive the exact per-device value reported by the planner.

The minimum valid grant is enough blocks for one full `max_model_len` request
on every required cache group. If that minimum does not fit, model loading must
fail before the child starts. Kapsl must not silently reduce the advertised
context length.

If the full target does not fit but the minimum does, Kapsl may reduce the
effective target concurrency to the largest whole number of full sequences
that fits. The resolved value must be logged and exported as a metric. An
operator-configured strict flag should turn this reduction into a load failure
for capacity-sensitive deployments.

### 4. Reserve the budget transactionally before child startup

Introduce a provisional external-KV reservation in `MemoryAuthority`.
Selection and reservation must occur under the same authority operation lock:

1. Take the current domain snapshot, including planned, reserved, committed,
   observed, pool, and foreign-pressure state.
2. Reserve this replica's weight/workspace report.
3. Subtract the configured driver/runtime safety reserve.
4. Clamp the desired KV plan to the remaining bytes without violating the
   one-sequence minimum.
5. Create a single-use reservation token bound to:
   - participant base ID;
   - model fingerprint;
   - replica ID;
   - exact device/rank map;
   - exact per-pool byte count and geometry digest;
   - certified adapter/profile tuple;
   - expiry time and authority generation.
6. Keep the reservation charged while the child starts.

This token closes the race between concurrent startup models and autoscaled
replicas. An in-process deployment map alone is insufficient because it would
not arbitrate with llama.cpp, ONNX, provider allocations, or other
`MemoryAuthority` consumers.

### 5. Pass exact bytes to vLLM

Launch the pinned vLLM build with its exact-byte cache option, shown by the
current build as:

```text
--kv-cache-memory-bytes <granted-bytes>
```

Remove the implicit `--gpu-memory-utilization 0.5` from the auto path. Before
shipping, certify the exact precedence and startup checks of this flag against
the pinned vLLM wheel; no assumption about a newer or older upstream CLI is
acceptable.

The process command, resolved plan, and startup log must expose:

- desired bytes;
- granted bytes;
- effective target concurrency;
- bytes and blocks per cache group;
- authority snapshot used for the decision;
- whether the plan was auto, fixed, or legacy.

### 6. Transfer, rather than duplicate, the reservation at registration

Include the single-use token in `kv_connector_extra_config`. During
`shared_pool` registration, `ExternalKvCoordinator` must verify that the
participant's reported topology and requested allocation exactly match the
token.

On success, transfer the existing authority charge into the provisioned
CUDA-IPC backing. Do not release and reacquire between those operations, and do
not create a second charge. Reject registration if any of these differ:

- participant/model/profile identity;
- rank or device mapping;
- cache-group geometry digest;
- requested byte count;
- token generation, lifetime, or prior-use state.

The provisioner then allocates, zeroes, exports, attaches, and activates the
same way it does today. The existing tensor-alias and no-second-allocation
checks remain mandatory.

### 7. Make replica readiness atomic

Represent managed replica state explicitly, for example:

```text
Planned -> Reserved -> Starting -> HttpReady -> KvAttached -> Active -> Routable
                                      |                         |
                                      +------ failure ----------+
```

A replica becomes routable only after:

- the child process is alive;
- the HTTP server is ready;
- every expected worker binding is attached;
- the connector profile and imported byte size are verified;
- the external participant is active;
- the scheduler has published the new readiness generation.

The supervisor must atomically clear `Routable` before stopping or restarting a
child. Request dispatch checks this local state and process generation; it does
not issue a `/health` request. Connection failures still propagate normally
and also trip the readiness/circuit-breaker state.

### 8. Recalculate every scale-up and restart

Autoscaling must call the same planner and authority reservation transaction as
the primary load. It must never copy the first replica's byte count without
checking current headroom.

For a restart:

1. mark the old generation unroutable;
2. fence and reap the complete process group;
3. retire the participant and release or quarantine its backing according to
   existing safety rules;
4. obtain a new memory snapshot and grant;
5. build a new command with the new exact byte count;
6. publish the replica only after the full readiness state completes.

If process exit or CUDA detach cannot be proven, retain the old backing and do
not start a replacement against the same budget.

### 9. Export actionable memory metrics

Add per-model/replica/device metrics for:

- requested and granted KV bytes;
- minimum required bytes;
- effective target concurrency;
- total, allocated, active, and idle blocks;
- backing bytes versus logical leased bytes;
- planning reductions and rejection reasons;
- provisional reservation age/state;
- restart generation and quarantine bytes.

NVML used memory must remain a separate device-wide measurement. Free blocks
inside an allocated CUDA-IPC backing are reusable by that participant but are
not device-free.

## Proposed bridge architecture

### 1. Add a typed internal OpenAI operation

Replace the JSON marker convention with an explicit, versioned protocol request
in `kapsl-engine-api`. The exact API shape can follow existing transport
constraints, but it must distinguish:

- ordinary tensor inference;
- an OpenAI chat/completions wire request;
- streaming versus one-shot response;
- the expected output format.

The type should carry the normalized upstream path and one serialized request
body. It should not carry client authorization or allow an arbitrary host; the
managed engine supplies its own private endpoint.

Because `InferenceRequest` is serialized by sequence-based transports, adding
fields requires a coordinated engine-api/transport version bump and backward
compatibility tests. Do not insert a field in the middle of an existing bincode
layout, and do not replace one magic byte prefix with another undocumented
one.

### 2. Keep all Kapsl ingress policy before the fast path

The public route continues to perform:

- authentication and authorization;
- payload size limits and OpenAI schema validation;
- model alias resolution;
- unsupported-parameter checks;
- request ID assignment for internal logs/traces;
- session scoping and affinity;
- priority resolution;
- pressure and memory admission;
- scheduler and replica selection;
- cancellation ownership.

Only after those steps should it normalize `model` to the selected vLLM served
name and serialize the upstream request once. vLLM must receive supported stop
sequences and streaming usage options directly so Kapsl does not have to
reimplement them on the response path.

### 3. Use one persistent asynchronous client per managed process

Replace the per-request blocking client with a connection-pooled asynchronous
HTTP client owned by `ManagedVllmProcess` or its bridge object.

Requirements:

- connection reuse across requests;
- request-level timeouts without rebuilding the pool;
- bounded connect, header, idle-body, and total deadlines;
- HTTP/1.1 streaming without response buffering;
- immediate cancellation when the downstream stream is dropped;
- bounded buffers and natural backpressure;
- no OS thread per request;
- no unbounded MPSC channel.

Prefer a Unix-domain socket if the exact pinned vLLM server supports a stable
and certifiable UDS option. Otherwise retain private loopback TCP with a
persistent pool. UDS is an optimization, not a prerequisite for eliminating
the dominant translation and thread costs.

### 4. Remove the inference-time health request

The bridge should perform a local readiness-generation and child-liveness check
before dispatch. The periodic supervisor remains responsible for active health
probing. The inference POST itself is the authoritative result for that
request.

On connect/reset/timeout failures:

- return the correct typed engine error;
- atomically mark the generation suspect or unroutable according to the circuit
  breaker;
- let the supervisor confirm and restart;
- never mask the failure as an empty successful stream.

Readiness polling during initial startup and periodic supervisor checks should
reuse their own persistent client or the same bridge pool with short
request-level deadlines.

### 5. Relay OpenAI responses without token translation

For the typed OpenAI operation:

- one-shot responses return the upstream status and JSON body without parsing
  and rebuilding a new completion;
- streaming responses relay upstream SSE byte chunks directly;
- downstream backpressure controls upstream reads;
- disconnecting the downstream client drops/cancels the upstream body;
- hop-by-hop headers are removed, while safe content-type, cache, request-ID,
  and retry headers are preserved according to an allowlist;
- non-2xx upstream error bodies retain their OpenAI error shape and status.

The relay must not assume that network chunks align with SSE events. It should
forward bytes, not lines. A small bounded validator may inspect the terminal
usage event for metrics, but metrics extraction must not sit in the forwarding
critical path or rewrite content.

vLLM-generated completion IDs and exact token usage should be preserved. Kapsl's
own request ID remains in logs/traces and may be returned in an HTTP header
without rewriting every SSE event.

### 6. Preserve the native tensor path

Native `/infer`, non-OpenAI callers, and engines other than managed vLLM keep
the existing `BinaryTensorPacket` behavior. If a managed vLLM request arrives
as an ordinary prompt tensor, the compatibility path may still construct a
completion request and return text. Only explicitly typed OpenAI operations are
wire-relayed.

### 7. Define stop, usage, and cancellation ownership once

For the wire fast path:

- vLLM owns stop-sequence enforcement because it receives the original stop
  parameters;
- vLLM owns exact prompt/completion usage counts;
- Kapsl owns client-disconnect cancellation and request-lifetime admission;
- Kapsl must not apply a second stop filter or synthesize a second successful
  finish event;
- connector request leases remain held until upstream completion or confirmed
  cancellation.

Before enabling passthrough, conformance tests must verify stop strings that
span token boundaries, `stream_options.include_usage`, cancellation, and all
supported sampling fields against the pinned vLLM build.

### 8. Add bridge-stage instrumentation

Measure at least:

- ingress parse/validation duration;
- scheduler queue duration;
- upstream dispatch duration;
- time to upstream headers;
- time to first upstream byte;
- time from first upstream byte to first downstream byte;
- total relayed bytes and chunks;
- active bridge streams;
- pooled connections/connect attempts;
- cancellation and upstream error counts;
- legacy versus wire-fast-path requests.

These measurements are required to attribute the existing 4-9% throughput and
TTFT gaps rather than inferring causality from source inspection.

## True live KV elasticity: Phase-5 implementation

The fixed CUDA-IPC profile remains intentionally immutable. The separately
versioned CUDA VMM profile implements live physical growth and shrink while the
model stays loaded. It satisfies the following requirements:

1. Reserve a stable virtual-address range and map/unmap Kapsl-owned CUDA memory
   granules with CUDA virtual memory management, or teach the certified vLLM
   attention path to consume multiple backing segments.
2. Extend the connector contract so Kapsl can offer and revoke block ranges
   after activation.
3. Extend vLLM's block manager so its total block count can grow and so only
   currently free blocks can be retired.
4. Fence attention kernels and scheduler state before unmapping or reassigning
   any block.
5. Zero a block before it crosses participant/security ownership.
6. Quarantine ambiguous releases exactly as the current expiry path does.
7. Re-run native FlashAttention write/read, guard, exhaustion, reuse, and detach
   conformance for the new segmented or virtual-memory profile.

The implementation uses that stable contiguous virtual representation. ABI 1.5
describes the minimum/mapped/maximum blocks, VMM granularity and segments, and
generation-bound resize stages. Growth maps and zeroes workers first, then
publishes scheduler blocks; shrink retires a native-free scheduler tail first,
then unmaps workers and lowers physical accounting. Target and speculative
forwards share a resize lock. Timeouts and ambiguous detach fence routing and
retain the charge.

The elastic profile remains disabled outside an explicit package policy and
exact conformance allowlist. The GPU workflow must prove stable addresses,
zeroed segments, zero second PyTorch allocation, native attention before and
after resize, minimum-floor shrink, churn, reclamation, and quarantine behavior
before deployment. Process autoscaling still prefers internal batching and
elastic headroom before adding duplicated processes.

## Implementation phases

| Phase | Implementation state | Remaining gate |
|---|---|---|
| 0: measurement/contracts | Implemented; host fixtures pass | GPU baseline evidence |
| 1: exact initial budget | Implemented; host transaction/lifecycle tests pass | Exact multi-rank GPU allocation |
| 2: async cleanup | Implemented; fake-upstream/backpressure tests pass | Same-host GPU observation |
| 3: wire fast path | Implemented; API/transport/scheduler/runtime tests pass | Semantic and performance matrix |
| 4: autoscaling/rollout | Implemented; unit and public-API harnesses present | Real 1->2->1 and restart cycle |
| 5: live resize | Implemented as ABI 1.5 CUDA VMM profile; host state-machine tests pass | Native GPU VMM/churn/reclamation matrix |

The remaining entries are acceptance execution, not missing implementation.
They are all enforced by `vllm-shared-pool-conformance.yml`; failure prevents
creation of the elastic allowlist and the final pull request.

### Phase 0: measurement and contracts

- Add bridge-stage timings and connection/thread counters to the legacy path.
- Add requested/granted/active/idle KV metrics.
- Capture a semantic baseline with identical prompts, sampling parameters, and
  exact output lengths for direct and Kapsl vLLM.
- Specify the planner JSON schema, reservation-token lifecycle, typed OpenAI
  operation, and readiness state machine.
- Decide the default per-replica target concurrency and strict-reduction policy.

Exit gate: the current throughput/TTFT gap and the 10.42 GiB pool are visible in
component metrics, and all new wire/authority contracts have versioned test
fixtures.

### Phase 1: exact initial KV budget

- Add the certified cache-geometry planner.
- Add `kv_cache` manifest policy and legacy migration parsing.
- Add checked block/byte planning.
- Add transactional provisional external-KV reservations.
- Pass exact `--kv-cache-memory-bytes` bytes.
- Transfer the reservation during connector registration.
- Make attachment, activation, and scheduler publication one readiness path.
- Re-plan restarts and scale-ups.

Exit gate: no default fraction is passed; the physical IPC backing equals the
authority grant within certified alignment; two admitted replicas cannot exceed
the authority budget.

### Phase 2: low-risk bridge cleanup

- Reuse persistent clients for readiness and inference.
- Remove per-request `/health`.
- Replace blocking thread-per-stream I/O with bounded asynchronous streaming.
- Make cancellation drop the upstream operation promptly.
- Preserve existing response translation temporarily as an A/B control.

Exit gate: no inference health request, no per-stream OS thread, no unbounded
channel, and no behavioral regression in the existing OpenAI response tests.

### Phase 3: OpenAI wire fast path

- Add the versioned typed protocol operation across engine API, transport,
  scheduler, and runtime.
- Serialize one normalized upstream request.
- Relay one-shot status/body and streaming SSE bytes directly.
- Preserve upstream errors, usage, and response IDs.
- Keep the legacy tensor path behind a rollback flag.

Exit gate: direct and Kapsl clients observe equivalent OpenAI semantics, and
the bridge no longer parses/reconstructs each normal stream event.

### Phase 4: autoscaling policy and production rollout

- Prefer vLLM continuous-batching capacity before process replication.
- Gate scale-up on a successful exact memory reservation.
- Keep new replicas unroutable through HTTP and KV activation.
- Canary auto sizing and the wire path separately.
- Remove the implicit 0.5 default after the compatibility window.

Exit gate: repeated 1->2->1 scaling and forced process restarts complete without
connection-refused streams, authority leaks, or quarantined memory under clean
shutdown.

### Phase 5: optional live KV resizing

- Prototype CUDA VMM or segmented cache backing.
- Add block add/retire synchronization to the connector and vLLM integration.
- Create a new adapter profile and full GPU conformance matrix.
- Enable only after fragmentation, churn, and security-isolation tests pass.

## Expected code areas

### `kapsl-engine`

- `kapsl-runtime/crates/kapsl-cli/src/runtime/managed_vllm.rs`
  - settings migration, exact command arguments, shared async bridge client,
    readiness generation, restart re-planning, and metrics;
- `kapsl-runtime/crates/kapsl-cli/src/runtime/model/replica.rs`
  - pass resolved concurrency and authority inputs; publish only active
    replicas;
- `kapsl-runtime/crates/kapsl-cli/src/runtime/model.rs`
  - aggregate startup preflight and concurrent-load behavior;
- `kapsl-runtime/crates/kapsl-cli/src/runtime/memory.rs`
  - provisional external-KV reservation and atomic transfer semantics;
- `kapsl-runtime/crates/kapsl-cli/src/runtime/kv_control.rs`
  - token validation, topology/grant matching, activation readiness, and
    transfer to the provisioner;
- CUDA-IPC provisioner code
  - adopt a precharged reservation without double accounting;
- `kapsl-runtime/crates/kapsl-cli/src/http/openai/chat.rs`
  - typed wire request and direct response/SSE relay;
- `kapsl-runtime/crates/kapsl-cli/src/http/openai/types.rs`
  - normalized request fields and response-header policy;
- configuration/model-packaging documentation
  - new policy, deprecation, observability, and operational examples.

### `kapsl-sdk`

- `integrations/vllm/src/kapsl_vllm_connector/`
  - certified geometry planner, grant token, exact topology validation, and
    future resize contract;
- `crates/kapsl-engine-api/`
  - versioned OpenAI protocol operation and, if needed, response stream types;
- transport and scheduler crates
  - compatibility-safe serialization and operation routing;
- conformance workflows
  - exact-byte allocation, no-second-allocation, native attention, lifecycle,
    and optional UDS coverage.

### `kapsl-benchmarks`

- direct/Kapsl vLLM profiles with byte-identical upstream request semantics;
- memory sampling and new Kapsl metrics collection;
- replica scale/restart and mixed-backend scenarios;
- bridge CPU, connection, thread, TTFT, and backpressure tests.

## Test plan

### Host unit tests

- manifest parsing for auto, fixed, and explicit legacy modes;
- rejection of conflicting fields and zero/overflow values;
- checked block rounding and cache-group summation;
- minimum one-sequence capacity;
- target-concurrency reduction and strict rejection;
- min/max/headroom behavior;
- multi-rank per-device grants;
- reservation-token identity, expiry, replay, mismatch, and rollback;
- no double charge during reservation transfer;
- release on child spawn failure, readiness timeout, clean exit, and restart;
- no release after ambiguous process-group or transport fencing;
- readiness state transitions and generation checks;
- old package migration warnings and explicit legacy command construction.

### Connector tests

- planner output equals vLLM cache-spec geometry for certified model families;
- planned bytes equal the eventual `KVCacheConfig` packed size;
- connector rejects a byte, block, layer, rank, dtype, layout, or profile
  mismatch;
- imported tensors remain bounded aliases of the Kapsl allocation;
- PyTorch allocation delta remains zero for KV construction;
- token cannot be reused by a second engine ID;
- activation cannot occur before every worker attaches.

### Fake-upstream bridge integration tests

- exactly one inference POST and zero inference-time health GETs;
- persistent connection reuse across sequential requests;
- concurrent streams do not create one OS thread each;
- arbitrary TCP chunk boundaries are relayed byte-for-byte;
- slow clients apply bounded backpressure rather than growing an unbounded
  queue;
- disconnect cancels/drops the upstream request promptly;
- one-shot and streaming non-2xx OpenAI errors preserve status and body;
- no empty-success conversion after a mid-stream failure;
- model alias is normalized without forwarding credentials;
- priority, session affinity, queueing, and admission still execute;
- native tensor inference does not enter the wire path;
- legacy and fast paths produce semantically equivalent completions.

### GPU integration and conformance tests

- physical backing bytes match the exact grant;
- no second KV allocation appears in PyTorch or NVML deltas;
- one full `max_model_len` request succeeds at the minimum grant;
- target concurrency succeeds without vLLM block exhaustion;
- two replicas receive independent grants and stay within the safe device
  budget;
- mixed llama.cpp/vLLM load leaves ungranted VRAM available;
- repeated load/unload and 1->2->1 scaling return cleanly to baseline;
- forced child crash and restart do not reuse an unfenced pool;
- prefix caching, stop sequences, streaming usage, and cancellation work on the
  exact certified profile;
- FlashAttention native writes and paged reads still operate directly on the
  imported allocation.

### Fragmentation and churn tests for the later elastic phase

Repeat cache growth, shrink, load, unload, and scale cycles while scraping:

- `kapsl_gpu_device_pool_allocated_bytes`;
- free bytes and free ranges;
- largest free range;
- fragmentation ratio;
- external participant backing and quarantine bytes;
- NVML used/free memory;
- largest subsequent allocation and OOM outcome.

The existing benchmark did not measure those allocator metrics and cannot be
used as proof of fragmentation improvement.

## Performance and memory acceptance criteria

Use identical request bodies, tokenizer/model, prompt lengths, output lengths,
sampling parameters, warmup, measurement windows, and trial counts for direct
and Kapsl paths. Do not compare runs whose actual output sequence lengths
differ.

### Memory gates

- The default auto policy passes an exact KV byte count and no implicit 0.5
  fraction.
- The provisioned CUDA-IPC backing equals the authority grant within one
  certified allocation-alignment unit.
- Active vLLM KV tensors consume no second physical cache allocation.
- Idle cache above the declared target concurrency is not provisioned.
- Every replica's weights, workspace, KV backing, and safety reserve fit inside
  the authority's current device budget before the child starts.
- A rejected replica leaves no process, port/socket, provisional reservation,
  participant, or CUDA backing behind.
- On the benchmark's 0.5B/1,024-token/concurrency-16 case, cache capacity should
  be on the order of the declared workload plus bounded headroom, not 910,480
  tokens. The exact byte gate comes from certified geometry rather than a
  hard-coded model-specific target.

### Bridge gates

- Output-token throughput loss versus direct vLLM is no more than 2% at
  concurrency 1, 4, 8, and 16 after statistical confidence is established.
- Added median TTFT is no more than 5 ms and added p95 TTFT is no more than
  10 ms on the same-host benchmark.
- There is no per-request health call and no OS thread per stream.
- Buffering is bounded and demonstrated under a deliberately slow client.
- Streaming and one-shot outputs retain vLLM's exact usage and OpenAI error
  semantics.
- Kapsl authentication, priority, session affinity, cancellation, scheduling,
  and memory admission remain covered by integration tests.

If the 2% throughput target is missed, retain the measured fast path and use
the stage timers to decide whether the remaining cost is scheduler dispatch,
transport, serialization, vLLM connector admission, or unavoidable process
separation. Do not remove Kapsl policy controls merely to satisfy the number.

## Rollout and rollback

1. Ship metrics first with no behavior change.
2. Add exact sizing behind `KAPSL_VLLM_MEMORY_MODE=exact`, with
   `legacy-fraction` as rollback.
3. Canary exact sizing on one certified model family, then mixed models and
   autoscaling.
4. Add the async translated bridge behind
   `KAPSL_VLLM_BRIDGE_MODE=async-translated`.
5. Add the typed wire path as `wire`, compare all three modes, and retain
   `legacy` for one release.
6. Make `exact` and `wire` defaults only after GPU, semantic, and performance
   gates pass.
7. Remove the implicit 0.5 default; keep explicit legacy parsing for the
   documented compatibility window.
8. Bump the connector version/profile whenever the planning, grant, allocation,
   or live-resize contract changes, and require fresh signed conformance.

Rollback must never reinterpret an exact grant as a fraction. It should stop
the affected replica, retire its participant safely, and start a fresh legacy
generation only after memory is released or quarantined.

## Resolved implementation decisions

1. Omitted `target_concurrency` follows the resolved per-replica Kapsl
   batch/delegated-concurrency target; operators may override it in the package.
2. Auto sizing uses 20% bounded headroom and no implicit prefix-retention pool.
3. Whole-sequence concurrency reduction is allowed by default and observable;
   `strict: true` rejects it.
4. The pinned planner constructs the executor, registers backend-customized
   specs, resolves layouts/groups, calls vLLM's own cache-config builder, and
   exits before `initialize_from_config` performs final KV allocation.
5. The bridge uses persistent pooled private loopback TCP. UDS is not required
   by the certified profile.
6. `OpenAiWireRequest` and versioned transport envelopes carry normalized
   request bytes; raw unary/SSE response frames preserve upstream semantics.
7. Autoscaling defers process replication while native continuous batching or
   elastic physical headroom remains, except that an explicit minimum-replica
   floor is authoritative.
8. Live elasticity uses CUDA VMM physical segments behind one stable BLNHC
   virtual address.

## Definition of done

The remediation is complete when a managed vLLM replica starts from an exact,
transactionally granted KV budget; its connector allocation matches that grant
without duplication; scale-up and restart repeat the same current-state
admission; the replica is routed only after full HTTP/KV activation; and
OpenAI requests traverse Kapsl policy while their request and response bodies
cross the private process boundary once, asynchronously, with bounded
backpressure and no per-event JSON reconstruction.

Live resize is complete only under its separate definition: free vLLM blocks
can be safely unmapped or reassigned to another participant while the model
remains loaded, with native-attention conformance and quarantine guarantees.
