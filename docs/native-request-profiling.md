# Native request diagnosis

`KAPSL_REQUEST_PROFILING=1` enables bounded timing for manual diagnosis.
`engine.dispatch` records request planning, the memory admission callback and
backend execution. `engine.native` separates request conversion, call-lock
acquisition, ownership registration, cancellation registration, adapter calls,
result conversion and return cleanup. Request-memory reports have their own
conversion and adapter-call timings. Existing scheduler queue-wait metrics and
client request durations cover time outside these engine calls.

Each model/replica collector reserves at most 8,192 records across its complete
lifetime. Disabled collectors perform no request timing or sample allocation.
Only operation names, numeric ownership/request IDs and durations are captured.
No tensor data, prompts, session IDs or user metadata enter the profile.

Records remain in memory during inference. Unload the model before terminating
the process to emit `KAPSL_REQUEST_PROFILE` JSON through normal logging; abrupt
termination can lose samples. Native and adapter profiles share ABI request
IDs. Dispatch records use ID zero and can be matched chronologically for serial
workloads. Each collector publishes its own clock origin for correlation.
Nested measurements must not be added to their enclosing durations.

Use the analyzer in `kapsl-integrations/integrations/ort/conformance/` and retain
raw logs, binary/source hashes, model hashes and host resource observations.
Its default windows assume 40 warmups followed by 1,000-request serial trials.
The ORT integration also has an ignored CPU-only direct-adapter diagnostic that
excludes the engine, for comparison with actual signed-pack hosting.

Profiling is diagnostic only. Existing qualification runs keep this switch
disabled, preserve every trial, and retain the 1.5× startup threshold and other
gates. This change does not alter allocator ownership, memory admission,
cancellation, pack verification or platform compatibility policy.
