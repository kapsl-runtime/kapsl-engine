# Logging

The `kapsl` executable installs one process-wide `tracing` subscriber, including
a bridge for existing `log` calls. HTTP, MCP, scheduling, memory governance,
native transport dependencies, and backend lifecycle diagnostics use the same
filter, formatter, and stderr sink. Libraries do not install a logger.

`KAPSL_LOG_FORMAT=text` is the default; `json` emits one JSON object per event.
`RUST_LOG` controls levels and targets (default `warn`). The throughput profile
continues to select `warn` when `RUST_LOG` is unset. Invalid logging settings
fail startup. CLI subcommands also initialize this pipeline.
Runtime startup banners and progress spinners become structured events in
JSON mode, keeping them out of the JSON stream as plain terminal text.

For request auditing in a log collector:

```bash
KAPSL_LOG_FORMAT=json \
RUST_LOG=warn,kapsl::access=info,kapsl::authorization=debug \
kapsl run --mcp-port 9097 2>>kapsl.jsonl
```

`kapsl::access` records completed HTTP and MCP requests with `protocol`,
`method`, `status`, and `elapsed_us`. It omits paths, query strings, headers,
credentials, and bodies. `kapsl::authorization` records policy evaluations
with required role, scope, and outcome. Warp can evaluate several role groups
while selecting a route, so evaluation records are not request counts; use
the completion records for request totals. These logs are diagnostics, not a
durable or tamper-resistant compliance audit store.

Managed vLLM stdout and stderr are independently drained in bounded 8 KiB
chunks, including unterminated lines. Raw bytes are still appended to the
existing per-model `vllm.log`; new files are owner-readable/writable on Unix.
Chunks can also enter the central sink with:

```bash
RUST_LOG=warn,kapsl::backend_output=debug KAPSL_LOG_FORMAT=json kapsl run model.aimod
```

Backend output carries `backend`, `stream`, and `output` fields. It is raw
diagnostic text and may contain model inputs or credentials printed by the
backend; enable it only when appropriate. Other dependency debug/trace logs
may likewise contain sensitive data. The safe field selection above applies
to Kapsl's access and authorization events, not arbitrary backend output.

Use the process supervisor or your log collector for retention and rotation.
Native per-request authorization audit events require the SDK callback
described in [Authentication](authentication.md); native library diagnostics
already flow through this subscriber.
