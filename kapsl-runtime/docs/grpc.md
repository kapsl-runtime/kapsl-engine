# gRPC API

Build with the optional `grpc-server` feature and choose a listener port:

```sh
cargo build --locked -p kapsl --features grpc-server
RUST_LOG=info target/debug/kapsl run --grpc-port 9096 --model model.aimod
```

`--grpc-bind` defaults to `127.0.0.1`. `--grpc-max-message-bytes` defaults to
16777216 (16 MiB), for both requests and individual response messages.
The listener is disabled unless `--grpc-port` is supplied. These flags are
available only in builds with the feature; existing installer profiles do not
enable it automatically.

For external serving, terminate TLS at an HTTP/2-capable reverse proxy. The
backend listener is plaintext HTTP/2; non-loopback binds require
`KAPSL_ALLOW_INSECURE_GRPC=1`. The proxy should supply ordinary gRPC forwarding,
not gRPC-Web translation. Browser applications can continue using HTTP SSE.

## Ownership

`kapsl-sdk/crates/kapsl-grpc` contains the protobuf schemas, generated clients,
tensor conversion, and reusable server. `kapsl-communication` optionally
re-exports it. Engine's `runtime/serving/grpc.rs` supplies:

- Live model and replica-pool discovery.
- The shared `InferenceService`, which applies scheduler and memory-pressure
  policy and invokes the existing governed backend boundary.
- The shared API authorization evaluator and credential-scoped session IDs.
- Startup configuration and supervised shutdown.

The listener uses the existing process logger. `RUST_LOG=kapsl::access=info`
enables gRPC handler completion records containing method, gRPC status, and
elapsed time through stream termination. Credentials, input/output data, model
names, and request/session IDs are omitted. Backend failure details are not
returned to clients. Requests rejected by the gRPC decoder before reaching a
handler are handled by the transport itself.

HTTP, login, and gRPC consult the same live credential state. Use
`authorization: Bearer TOKEN` metadata with the existing role tokens or API
keys. All currently exposed RPCs require Reader / API Read access. With no
active credentials, only actual loopback peers are accepted. Revocations,
expiry, role changes, and scopes apply to subsequent RPCs on existing
connections; active streams retain their admission until completion or
cancellation. Native socket/TCP/SHM authentication keeps its existing policy.

## Compatibility and clients

The standard service provides KServe V2 health, metadata, and unary
`ModelInfer`. The `kapsl.v1.KapslInference` service adds `ListModels` and
`InferStream`, which accepts one inference request and returns typed response
packets until EOF or a gRPC error.

Supported tensor datatypes are FP16, FP32, FP64, INT32, INT64, UINT8, and scalar
UTF-8 BYTES. This initial adapter supports single-output models. Unknown
versions, unsupported types/options, and malformed tensor encodings are
rejected explicitly.

Triton's bidirectional streaming and management extensions are not part of this
implementation. Neither WebSocket nor Kapsl governance-control RPCs are added.
See the SDK's `docs/grpc.md`, proto files, and Python streaming example for
client generation and parameter details.

The engine uses [`kapsl-grpc 0.3.0`](https://crates.io/crates/kapsl-grpc/0.3.0)
from crates.io. Its backend-neutral API dependency is shared with the engine's
other published SDK crates.

## Python SDK and native transport upgrade

Python `kapsl-sdk` 0.2.0 bundles `KapslGrpcClient` and `AsyncKapslGrpcClient`
behind the `grpc` extra. They provide discovery, unary inference, typed server
streaming, deadlines, and cancellation without consumer-side proto generation.
Use the configured gRPC port when constructing either client.

This engine also uses native `kapsl-transport`, `kapsl-ipc`, and `kapsl-shm`
0.4.0. Native tensor requests require the versioned `KIRQ` envelope; old
encodings are rejected before metadata decoding. SHM/hybrid require region and
protocol version 3, with allocation leases and response mailboxes. Upgrade and
restart Python clients and the engine together to recreate SHM regions. Native
TCP retains its separately configured `KAPSL_TCP_AUTH_TOKEN` policy.

See the SDK's `docs/python-sdk-0.2.md` for the supported version matrix and
stream ownership examples. No legacy request decoder or SHM notification queue
is retained in this transport release.
