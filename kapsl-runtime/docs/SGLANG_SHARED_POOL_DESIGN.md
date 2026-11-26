# SGLang shared pool and pool topology: design

Status: proposal. Nothing here is implemented. SGLang runs under Kapsl only as
an opaque KV participant (kapsl-engine #273, kapsl-integrations #44,
kapsl-sdk #223). Phases 0 and 1 ran on an RTX 3090 on 2026-11-26; their
results are under "Phases".

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
4. **Back SGLang's per-layer buffers as they are (striped).** The phase 1
   probe found both layouts correct. Striped attaches to SGLang's own
   virtual-memory backing, which cost nothing measurable, and live growth and
   idle retirement worked on it. Packed needs no KV ABI change but replaces
   how SGLang builds its pool, and showed a possible throughput cost in one
   run. Striped needs a minor ABI addition (a stripe count), which is the next
   decision.

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
- **Allocator resize.** `resize(config)` in `mem_cache/allocator/base.py` sets
  the new size and then calls `clear()`, which drops every allocation. It is a
  startup step. A live pool needs its own way to add free slots at the top and
  to retire a free tail.
- **Radix cache.** Finished requests' tokens stay in the pool until evicted,
  so "no live lease" does not mean "free tail". `Scheduler.flush_cache()`
  resets the cache and the allocator, and only on an idle server.
- **Mapping memory it did not create.** `utils/cuda_vmm_utils.py` has
  `VmmReservation.map_existing`, which maps a caller-owned allocation into a
  reservation, and `import_peer_handle` and `import_and_map_alloc` for
  shareable handles received as file descriptors. SGLang uses them between its
  own ranks. They are what an adapter needs to map segments the runtime
  exported.
- **Its own packed layout is gone in this release.**
  `PageMajorMHATokenToKVPool` is marked non-constructible ("the strided 4-D
  view builder and its write kernel are gone") and server arguments reject it.
  The class comment points to `--enable-unified-memory` for a page-major
  layout with per-layer views. In the normal path the K and V buffers of a
  layer are two halves of one allocation, so that paged kernels reach V from a
  single base.
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

### Layout: the open choice

The KV ABI describes an elastic pool as one contiguous range of blocks whose
mapped prefix is covered by segments. Each segment is one CUDA allocation. The
driver maps an allocation from offset zero only, and unmaps only a whole
mapping (CUDA Driver API, `cuMemMap` and `cuMemUnmap`). One segment therefore
backs one contiguous piece of one buffer.

SGLang keeps 2 x layers buffers. There are two ways to put them on that
contract.

| | Striped: SGLang's buffers as they are | Packed: one range, per-layer views |
|---|---|---|
| How | each buffer gets its own backed prefix inside SGLang's arena | one tensor holds every layer's K and V for a token; each `k_buffer[i]` and `v_buffer[i]` is a strided view of it |
| Patch surface in SGLang | small: replace where `KvVmmArena` gets its memory | larger: replace how `MHATokenToKVPool` creates its buffers |
| KV ABI | new optional stripe count, and a stripe index per segment (minor version) | unchanged: it is the layout the vLLM profile already uses |
| Segments per growth step | one per buffer: 56 for a 28-layer model | one |
| Smallest step, Qwen2.5-1.5B | 4,096 tokens, 112 MiB | 512 tokens, 14 MiB |
| Main risk | allocation and handle count grows with layers x steps | attention backends, CUDA graphs and copy paths must accept non-contiguous per-layer buffers |
| Evidence | SGLang's own startup path uses the arena | kvcached's default CUDA layout maps one compound page across all layers, on SGLang up to 0.5.20; how it derives the per-layer tensors was not read |

The step sizes are computed from the model configuration, not measured: 28
layers, 2 KV heads of 128, fp16, so 512 bytes per token per buffer and 28,672
bytes per token across all 56. A 2 MiB granule holds 4,096 tokens of one
buffer. Packed, the smallest step that is a whole number of granules is 512
tokens (7 granules).

Striped keeps the SGLang patch small and costs an ABI change and many small
allocations. Packed costs nothing in Kapsl and more in SGLang.

Phase 1 settled correctness for both (results under "Phases"). Striped is
recommended: it runs on SGLang's own backing at no measured cost, and growth
and retirement were demonstrated on it. Packed also gave identical outputs,
including with CUDA graphs, so SGLang's attention path accepts strided
per-layer views. But it means replacing how `MHATokenToKVPool` creates its
buffers, and one load run was markedly slower than the normal layout.
SGLang's own `--enable-unified-memory` is not an option: it accepts only
hybrid Mamba and sliding-window models.

### Memory source

In both layouts the runtime creates, zeroes and exports the segments, and the
adapter imports and maps them into an address range SGLang reserved. Striped,
that replaces the one step where `KvVmmArena` maps memory it created itself.
Packed, the adapter also creates the tensor and the per-layer views.

### Resize

Growth, driven as for vLLM by a refused reservation or by utilization:

1. the runtime creates and zeroes the segments and exports them;
2. the adapter maps them at the end of each buffer's backed prefix;
3. the adapter adds the new token slots to the allocator's free list (its own
   code; `resize` would clear the allocator);
4. the adapter acknowledges, and the runtime activates the capacity.

Shrink, first delivery:

1. admitted only when the replica holds no lease, as for vLLM;
2. the adapter evicts radix-cache entries that occupy the tail to retire;
3. the adapter shrinks the allocator's token range, then unmaps the tail of
   every buffer and acknowledges;
4. the runtime releases the segments.

SGLang at TP 1 runs scheduler and model in one process, so the worker and
scheduler stages of the vLLM protocol collapse into one acknowledgement.

Growth needs no fence. Phase 1 grew the pool twice while 2 and 4 requests
were running under SGLang's default overlap scheduling, from the scheduler
thread between steps; every request completed and later outputs were
unchanged. So growth does not need what vLLM pays for with
`--no-async-scheduling` (7 to 15%). Retirement ran on an idle server only.

SGLang keeps the pool's capacity in six places, and a resize has to update
all of them in one step: the pool's `size`, the allocator's `size` (and
`num_pages` on paged allocators), `ModelRunner.max_total_num_tokens`,
`Scheduler.max_total_num_tokens`, and the scheduler's `pool_stats_observer`
and `invariant_checker`. The observer is a frozen, slotted dataclass. Missing
one trips SGLang's idle leak check ("pool memory leak detected") and the
scheduler exits; the probe hit exactly that once. The list is specific to
SGLang 0.5.23 and has to be re-checked on every version bump.

The reservation is fixed when the pool is created. SGLang sizes it from the
first `config_from_budget` call and backs only what the post-capture call
asks for, so the most a replica can ever grow to is chosen at startup. The
arena also backs 2 MiB of every buffer at creation: 96 MiB for the 48 buffers
of the test model, the floor of a striped replica.

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

- the ABI to carry mapped ranges (if the striped descriptor is added, it
  should be designed with this in mind);
- allocator changes in each backend, which is the invasive part;
- a decline message, so a server can refuse a range it cannot free.

It also removes the reason shrink waits for an idle replica in vLLM.

### Not taken from kvcached

- Server-owned physical memory with a cooperative ledger. It is simpler and
  faster to build, and it is roughly the opaque tier plus a resizable capacity.
  It gives up enforcement and zeroing between owners.
- Priorities with preemption of running requests. Kapsl has per-model priority
  weights and reclaim; mapping them onto elastic regions is separate work.

## Making the in-process pool resizable

A separate question: can the pool that llama.cpp and ONNX Runtime share inside
Kapsl resize the way a vLLM pool does?

Today it is one `cudaMallocAsync` allocation per device (`GpuArenaRegion` in
kapsl-hal) with a byte-range allocator on top, sized once and charged in full.
Both ways to change that stay inside one process, so neither needs what makes
the vLLM profile heavy: exported handles, acknowledgements, fencing by
generation.

| | A. More than one arena | B. The arena on a VMM region |
|---|---|---|
| How | grow by installing another arena region; shrink by retiring one that holds no lease | reserve the address range once; map and unmap segments under the range allocator |
| Exists today | `GpuDevicePool` is already a catalogue of regions: `install_arena`, `acquire_region` choosing among candidates, `retire_arena` | `GpuVmmRegion` in kapsl-hal (reserved range, `grow_to`, `shrink_boundary`), used for vLLM pools |
| Missing | a policy that adds an arena when admission needs room and retires an empty one | an allocator that hands out only mapped bytes and knows which segments are entirely free |
| Unit that comes back | a whole arena with nothing left in it | any segment with nothing left in it |

What can come back in either:

- llama.cpp KV pages: yes. They are separate arena allocations, taken and
  released as sequences come and go. This is the same kind of memory a vLLM
  pool returns.
- ONNX Runtime workspaces: only when ONNX Runtime's own arena releases them.
- Model weights: no, not while the model is loaded.

So the gain is real for llama.cpp KV and small for ONNX. Fragmentation decides
how much of it is reachable: one long-lived allocation pins its segment or its
arena. Keeping long-lived allocations (weights) apart from short-lived ones (KV
pages, workspaces) is what makes shrinking effective.

Not verified, and it decides between A and B: how the llama.cpp KV store and
the ONNX pool client take extents from the arena, and whether either can use
more than one region. The statement about KV pages comes from the allocator's
documentation comments, not from tracing an allocation.

## Phases

| Phase | Work | Needs a GPU |
|---|---|---|
| 0 | Opaque tier on a real server (#273, #44): hooks fire, leases are taken, admission holds. **Passed 2026-11-26.** | yes |
| 1 | Probe, no Kapsl changes: both layouts on the pinned build, growth and tail retirement on a live server, attention output unchanged. **Passed 2026-11-26.** | yes |
| 2 | SGLang profile and the striped pool descriptor in the KV ABI | no |
| 3 | Adapter: import and map exported segments, growth | yes |
| 4 | Engine: profile-keyed grants, shared supervision, load plan without the KV pool | partly |
| 5 | Shrink when idle, with radix eviction | yes |
| 6 | Mapping by page, for SGLang and vLLM | yes |

### Setup of the runs

One RTX 3090 (24 GB, driver 595.99.02, CUDA 13.2), SGLang 0.5.23 with torch
2.14.1 (CUDA 13.0) and flashinfer 0.7.2 as the attention backend,
Qwen2.5-0.5B-Instruct (24 layers, 2 KV heads of 64, bf16, page size 1).
Correctness: 8 fixed prompts, greedy, 48 tokens, compared token for token
with an unmodified server after flushing its prefix cache. Load: 1,589-token
prompts, 128 output tokens, output tokens per second. Every number is a
single run. The probe is an SGLang plugin plus driver scripts, kept outside
this repository.

### Phase 0 results

- The first run found an adapter bug: SGLang 0.5.23 reports the KV pool size
  as NumPy integers, and adapter 0.2.0 refused the pool at registration, so
  SGLang shut down (correctly, rather than serve ungoverned). Fixed in
  kapsl-integrations 9f02230.
- With the fix: the participant registered, outputs through Kapsl were 8/8
  identical, and 100/100 requests succeeded at concurrency 1, 8 and 16.
  24 requests needing about 41,000 tokens against a 16,384-token pool all
  completed, queueing for leases instead of failing.
- Cost of the opaque tier: about 10% at concurrency 8 and 16 (1,478 / 1,464
  tokens/s through Kapsl against 1,636 / 1,626 for plain SGLang with the same
  16,384-token pool). Sending the load straight to the managed server gave
  the same numbers as going through Kapsl, so the cost is the adapter's lease
  calls inside SGLang's scheduler, not Kapsl's HTTP bridge.
- In two runs, about 9 leases were still counted for up to 45 s after a
  saturating load; an instrumented run (72 leases granted, 72 returned, none
  counted afterwards) did not reproduce it. Unexplained; the lease lifetime
  cleared them.

### Phase 1 results

| Check | Result |
|---|---|
| 1. Striped (`SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1`) | Pass. 8/8 identical. The arena's handles are POSIX file descriptors, so they can be exported and imported. Throughput 890 / 1,881 / 2,908 tokens/s at concurrency 4 / 8 / 16 against 863 / 1,809 / 2,862 for the normal pool: no measurable cost. |
| 2a. Packed, plugin: per-layer K and V as strided views of one tensor | Pass for correctness. 8/8 identical with CUDA graphs on; no extra GPU memory (3,366 against 3,412 MiB). One load run at concurrency 8 gave 1,025 tokens/s against 1,664 to 1,809 for the normal layout in other runs: a possible cost, not measured in a matched sweep. |
| 2b. Packed, `--enable-unified-memory` | Not applicable: SGLang accepts it only for hybrid Mamba and sliding-window models. |
| 3. Growth on a live server | Pass. 4,096 to 16,384 to 32,768 tokens with 2 and 4 requests running; 192 MiB mapped per step in 7.6 to 7.7 ms; all requests completed; 8/8 identical afterwards. |
| 4. Tail retirement | Pass. 32,768 to 16,384 to 4,096 tokens on an idle server; 192 MiB unmapped per step in 9 to 12 ms; GPU memory dropped by about that; 8/8 identical afterwards. A boundary inside one mapping was refused, as CUDA requires. The probe flushed the whole prefix cache first; evicting only the tail is still to be written. |
| 5. Cost | The virtual-memory backing: none measurable (check 1). Growth: no fence needed (check 3). |

### Phase 1 checks as specified

Run on the pinned SGLang build with the launch flags Kapsl uses, one MHA model
and greedy decoding. Outputs are compared token for token with an unmodified
server.

1. **Striped.** Start with `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1`. Pass: same
   outputs, and the arena's backed bytes match the pool size.
2. **Packed.** Two variants: SGLang's own `--enable-unified-memory`, and a
   plugin that makes the per-layer buffers views of one tensor. Pass: same
   outputs with CUDA graphs on, and GPU memory no higher than the baseline,
   which rules out a hidden copy of the pool.
3. **Growth on a live server.** Back more of the pool and add the new slots to
   the allocator between scheduler steps while requests run. Pass: running
   requests finish with the same outputs and new requests use the added slots.
4. **Tail retirement.** After requests finish, evict the radix cache above a
   boundary, remove those slots from the allocator and unmap. Pass: GPU memory
   drops by the unmapped amount and later requests are correct.
5. **Cost.** Throughput at concurrency 1 to 16 against the unmodified server,
   with whatever boundary check 3 needed between a resize and a forward pass.

Check 1 or 2 failing removes that layout. Check 3 or 4 failing means SGLang's
allocator has to be replaced, as kvcached does, and the plan is costed again.

## Open questions

1. Adding and retiring slots on a live server works for the token allocator
   (page size 1) by editing its free list and the six capacity fields
   directly. The paged and sliding-window allocators are untested.
2. Growth needs no boundary against a forward pass. Retirement was only tried
   on an idle server; retiring while other requests run still needs a design.
3. The virtual-memory backing works with flashinfer and CUDA graphs on an
   RTX 3090. Other attention backends (FA3 on Hopper, triton) are untested.
4. Hybrid and sliding-window pools (`MLATokenToKVPool`, SWA, Mamba) are out of
   scope for the first profile. kvcached covers them with separate patches.
5. Tensor-parallel SGLang: one region per rank, resized together, as for vLLM.
   Not designed here.
6. Striped layout only: one resize passes 2 x layers handles in a single
   control message. Is that acceptable for deep models, and is a 112 MiB step
   too coarse for small ones?
