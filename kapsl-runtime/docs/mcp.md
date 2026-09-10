# MCP Server

`kapsl-runtime` can expose a Model Context Protocol (MCP) Streamable HTTP
endpoint. MCP is a northbound protocol adapter: requests still enter the normal
Kapsl scheduler, runtime-pressure policy, and memory governor before reaching a
backend.

The adapter is disabled at compile time and runtime by default. Build it with:

```bash
cargo build -p kapsl --features mcp-server
```

Start a listener by passing an MCP port:

```bash
kapsl run model.aimod --mcp-port 9097
```

The endpoint is `http://127.0.0.1:9097/mcp`. It uses the HTTP API's
`--http-bind` address and exposure policy. A non-loopback bind requires
`KAPSL_ALLOW_INSECURE_HTTP=1` and should sit behind a TLS-terminating reverse
proxy. Set `KAPSL_MCP_ALLOWED_HOSTS` to a comma-separated list of additional
reverse-proxy authorities. The MCP transport validates `Host` headers to
prevent DNS-rebinding attacks.

## Tools

- `kapsl_list_models` lists models that are registered and have a live replica
  pool.
- `kapsl_infer` accepts a numeric `model_id` and a backend-neutral tensor or
  media inference envelope. It submits the resulting request through the same
  `InferenceService` used by the native and HTTP ingress paths.

MCP inference intentionally ignores client-provided session IDs in this first
version. Kapsl normally derives the internal session namespace from the
authenticated credential. The MCP facade will enable sessions after principal
identity is part of its request context.

## Authorization

Every MCP HTTP request uses the same bearer tokens, API keys, reader role, and
`api:read` scope policy as the HTTP inference routes. When authentication is
disabled, only loopback clients are accepted. Credentials are supplied using
the standard HTTP header:

```text
Authorization: Bearer <api-key>
```

Login, HTTP routes, gRPC, and MCP all use the engine's shared policy evaluator.
MCP request completion events use the process-wide logging subscriber;
see [Logging](logging.md) for JSON output and audit filters.

The adapter performs no backend selection and imports no backend-specific
crate. Backend removal or replacement therefore does not change the MCP
protocol boundary.
