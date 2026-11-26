# SGLang shared pool and pool topology: design

Status: proposal. Nothing here is implemented. SGLang runs under Kapsl only as
an opaque KV participant (kapsl-engine #273, kapsl-integrations #44,
kapsl-sdk #223), and that path has not run on a GPU yet.

Date: 2026-11-26

## Decision in brief

1. **Do not give each backend type its own pool.** Keep one memory authority
   per GPU for every backend, and one isolated elastic region per replica.
   What moves between replicas, whatever their backend, is physical memory in
   2 MiB granules, not KV blocks.
2. **Give SGLang the same elastic profile vLLM has**, with Kapsl owning the
   physical memory, instead of copying kvcached's model where each server
   allocates its own.
3. **Take one thing from kvcached**: mapping by page anywhere in the pool, so a
   busy replica can give memory back. Kapsl's elastic profile can only retire a
   free tail. This is a later phase and applies to vLLM as well.

The sections below give the reasons, what was read to reach them, and what is
still unverified.

## One pool per backend type, or shared?

There are three things that can be shared or separate, and the answer differs
for each.

| Layer | Per backend type | Per replica | One per GPU | Recommendation |
|---|---|---|---|---|
| Budget and admission (who may use how many bytes) | static split between vLLM, SGLang, llama.cpp | no global view | Kapsl today, kvcached | one per GPU |
| Supply of physical granules | idle vLLM memory cannot serve SGLang | each server allocates its own (kvcached) | Kapsl elastic profile | one per GPU, owned by the runtime |
| Address range and tensor layout | impossible: geometry is per model | Kapsl today, kvcached | impossible across processes | one per replica |

Why not a pool per backend type:

- It is a static split. In the 2026-11-25 benchmark, three models with static
  splits held 21.6 to 23.9 GB idle on a 24 GB card; with elastic sharing they
  held 11.3 GB (Kapsl live resize) and 12.3 GB (kvcached).
- A type is the wrong boundary. KV geometry (layers, KV heads, head size,
  dtype) belongs to the model, not to the server that runs it. Two vLLM models
  have no more in common than a vLLM model and an SGLang model.
- kvcached does not separate by type either. Its ledger is keyed only by the
  physical GPU, so vLLM and SGLang processes charge the same one. That is read
  from its code; its README does not describe mixed use.

Why not one allocation that every backend attaches to:

- CUDA IPC exports a whole allocation. An importer would see every other
  model's KV. `runtime/memory/gpu_shared_pool.rs` already keeps exported
  backings separate from the general arena for this reason.
- Blocks are not interchangeable between models. Only bytes are.

So the unit of sharing is the granule, and "shared" means one budget and one
supply, with a private mapping per replica.

### Limit of the current layout

The general arena that in-process engines use (embedded ONNX Runtime, the
compiled-in GGUF engines, the native engine) is one immutable allocation sized
at startup and charged to the ledger in full (`runtime/memory/device_budget.rs`,
`runtime/memory/gpu_pool.rs`). Memory inside it cannot flow to a vLLM or SGLang
replica, and the reverse. Budget is shared across all backends; granule supply
is shared only among out-of-process elastic replicas. Moving the arena onto
granules is a separate project and is not part of this design.

## Kapsl and kvcached side by side

kvcached facts are from its `main` branch (README states v0.1.6): `csrc/page.cpp`,
`csrc/page_allocator.cpp`, `csrc/allocator.cpp`, `csrc/gpu_budget.cpp`,
`docs/coordinator.md`, and the two `integration/*/patches.py` files, the last
three read in part.

| | Kapsl elastic profile (vLLM today) | kvcached |
|---|---|---|
| Who creates physical memory | the runtime; segments are exported to the server as OS handles | each server process (`cuMemCreate` in its own context) |
| Who enforces the budget | the runtime, before memory exists | each process checks a ledger file per GPU in `/dev/shm` under `flock`; cooperative |
| Unit | granule-aligned segments of blocks | 2 MiB pages (`KVCACHED_PAGE_SIZE_MB`) |
| Mapping shape | one contiguous prefix; grow at the end, retire a free tail | any page independently; unmapped slots read a zero page |
| Backends under one budget | vLLM, plus in-process engines in the ledger | vLLM and SGLang |
| Giving memory back while busy | no: shrink is admitted only with no live lease (#271) | yes: freed pages beyond a small reserve go back to the driver |
| Per-replica policy | minimum, maximum, grow and shrink thresholds | limit, minimum, priority, headroom; lower priorities are reclaimed first |
| Memory crossing processes | zeroed by the runtime before another owner sees it | returned to the driver; whether pages are cleared first was not checked |
| Failure of the coordinator | replicas lose their leases and are fenced | none exists; a dead process is found by its file lock |
| Cost on the request path | synchronous scheduling for resize (7 to 15% on plain vLLM), resize round trips | allocation inside the process |
| SGLang versions | 0.5.23 pinned (opaque tier) | `>=0.5.11,<0.5.21`; 0.5.23 is outside the range |

Reading of the comparison:

- The two agree on topology: one budget per GPU, one mapping per replica, no
  per-type pools.
- kvcached is lighter because nothing crosses a process boundary. The price is
  that the budget holds only if every process on the GPU cooperates, and that
  it patches the server's allocator classes, which ties it to a version range.
- Kapsl's central ownership is what allows exact admission, zeroing between
  owners, and fenced revocation. Those are the product claims in
  `kapsl-sdk/docs/backend-kv-integration.md` and should stay.
- Kapsl is weaker in one place that matters in practice: a replica under
  steady load keeps whatever it grew to.

## What SGLang 0.5.23 provides

Read from the SGLang source at tag `v0.5.23` (`python/sglang/srt/`). Not run.

- **Layout.** `MHATokenToKVPool` holds one K buffer and one V buffer per layer,
  each indexed by token slot (`mem_cache/memory_pool.py`). vLLM's certified
  profile is one packed tensor. A step of N tokens therefore touches
  2 x layers buffers.
- **Its own VMM backing.** `mem_cache/kv_vmm_backing.py` reserves one address
  range per device (`KvVmmArena`), creates the K and V tensors in it as normal
  torch tensors, and backs a prefix of each buffer (`KvVmmBufferOwner`,
  `ensure_prefix`, `finalize`). It is off by default behind
  `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`.
- **What that backing is for.** Sizing the pool once after CUDA graph capture
  (`managers/scheduler.py`, `post_capture_resize_kv_pool`). `commit_range` only
  grows, and no unmap exists. It is a mechanism to build on, not elasticity.
- **Allocator resize.** The allocators have `resize(config)`
  (`mem_cache/allocator/base.py` and others). Whether it can run after startup,
  and whether it can shrink, is not verified.
- **Radix cache.** Finished requests' tokens stay in the pool until evicted,
  so "no live lease" does not mean "free tail".
- **No KV connector interface.** The adapter attaches through SGLang's plugin
  hooks, which fail open. The adapter's launcher is built to fail closed
  instead, and a shared-pool tier has to keep that property.

## Design

### Topology

- One participant per SGLang replica, as now (`<base>:scheduler`), in
  `shared_pool` mode with `live_resize`.
- One isolated elastic region per replica: an address range reserved by SGLang,
  backed by segments the runtime creates, zeroes and exports.
- The same `MemoryAuthority` and the same `GpuSharedPoolProvisioner` that serve
  vLLM. No SGLang-specific pool, ledger or budget.

### Layout mapping

One logical block is one token step across every buffer. For a model whose
buffers all have the same row size, the step that keeps every buffer on a
granule boundary is `granularity / gcd(granularity, row_bytes)` tokens.

Example, computed from the model configuration and not measured: Qwen2.5-1.5B
has 28 layers, 2 KV heads of 128, fp16, so 512 bytes per token per buffer.
One 2 MiB granule holds 4,096 tokens in one buffer; a step is 4,096 tokens and
maps 56 granules, 112 MiB.

The KV ABI describes an elastic pool as one range with a minimum, mapped and
maximum block count. SGLang needs one resize to cover 2 x layers ranges. Two
ways to express it:

| Option | Change | Cost |
|---|---|---|
| A. Striped pool: one binding with a stripe count and stride | new optional field in the pool descriptor (ABI minor) | one generation, one acknowledgement per resize |
| B. One binding per buffer, resized together | none in the ABI | 56 bindings for a 28-layer model, and a resize that half-applies has to be undone |

Recommended: A.

### Memory source

`KvVmmArena` maps memory it creates itself. The adapter replaces that one step:
it imports the segments the runtime exported and maps them into the same
address range. Tensor creation, allocator and attention stay SGLang's.

This is the difference from kvcached, which replaces SGLang's pool and
allocator classes. A smaller patch surface is the reason to prefer SGLang's own
arena as the seam.

### Resize

Growth, driven as for vLLM by a refused reservation or by utilization:

1. the runtime creates and zeroes the segments and exports them;
2. the adapter maps them at the end of each buffer's backed prefix;
3. the adapter extends the allocator's token range;
4. the adapter acknowledges, and the runtime activates the capacity.

Shrink, first delivery:

1. admitted only when the replica holds no lease, as for vLLM;
2. the adapter evicts radix-cache entries that occupy the tail to retire;
3. the adapter shrinks the allocator's token range, then unmaps the tail of
   every buffer and acknowledges;
4. the runtime releases the segments.

SGLang at TP 1 runs scheduler and model in one process, so the worker and
scheduler stages of the vLLM protocol collapse into one acknowledgement.

A resize must not overlap a forward pass. vLLM needed `--no-async-scheduling`
for that and pays 7 to 15% for it. Where SGLang's equivalent boundary is
(its overlap mode runs scheduling and forward on separate streams) is an open
question and decides the cost.

### Engine

- `runtime/kv/control.rs` checks provisioning grants against the literal backend
  `vllm`. Key that on the certified profile instead.
- Growth and shrink policy lives in the vLLM supervisor
  (`start_managed_vllm_supervisor`). It moves to the shared managed-server
  supervision so SGLang uses the same code. This is the second step of the
  backend strategy refactor (#274 is the first).
- At load, SGLang's static footprint stops including the KV pool: the server
  starts with the address range reserved and the minimum mapped.
- A certified profile id and allowlist entry for SGLang, as for vLLM.

### Later: mapping by page

To let a busy replica give memory back, the mapped state has to be a set of
ranges, not one count, and the server's allocator has to prefer low slots or
tolerate holes. kvcached shows both are workable. It needs:

- the ABI to carry mapped ranges (the striped descriptor above should be
  designed with this in mind);
- allocator changes in each backend, which is the invasive part;
- a decline message, so a server can refuse a range it cannot free.

It also removes the reason shrink waits for an idle replica in vLLM.

### Not taken from kvcached

- Server-owned physical memory with a cooperative ledger. It is simpler and
  faster to build, and it is roughly the opaque tier plus a resizable capacity.
  It gives up enforcement and zeroing between owners.
- Priorities with preemption of running requests. Kapsl has per-model priority
  weights and reclaim; mapping them onto elastic regions is separate work.

## Phases

| Phase | Work | Needs a GPU |
|---|---|---|
| 0 | Opaque tier on a real server (#273, #44): hooks fire, leases are taken, admission holds | yes |
| 1 | Probe, no Kapsl changes: SGLang's VMM backing on our pinned build; allocator `resize` after startup; unmap of a tail; attention correct before and after | yes |
| 2 | ABI: striped pool descriptor, SGLang profile | no |
| 3 | Adapter: import and map exported segments, growth | yes |
| 4 | Engine: profile-keyed grants, shared supervision, load plan without the KV pool | partly |
| 5 | Shrink when idle, with radix eviction | yes |
| 6 | Mapping by page, for SGLang and vLLM | yes |

Phase 1 decides whether phases 3 to 5 are possible on SGLang's own seam or need
allocator replacement as kvcached does. Nothing after it should start before it
passes.

## Open questions

1. Can SGLang's allocator `resize` run on a live server, in both directions?
2. Where is the safe boundary for a resize relative to a forward pass, and what
   does it cost in throughput?
3. Does the VMM backing work with the attention backends and CUDA graphs our
   launch flags select? It is off by default upstream.
4. Hybrid and sliding-window pools (`MLATokenToKVPool`, SWA, Mamba) are out of
   scope for the first profile. kvcached covers them with separate patches.
5. Tensor-parallel SGLang: one region per rank, resized together, as for vLLM.
   Not designed here.
6. Is the 2 x layers granule round-up acceptable for small models, or should
   small pools use a smaller step with partial last granules?
