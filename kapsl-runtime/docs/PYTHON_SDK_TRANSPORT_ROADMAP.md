# Python SDK Transport Roadmap

This document describes a forward-compatible path for remote transports in `kapsl_runtime`.

## Current State

- Python `KapslClient` supports:
  - local socket/pipe transport
  - raw TCP transport (`tcp://host:port`)
- Wire protocol is the existing binary IPC frame.
- TCP transport is not encrypted and has no transport-layer authentication.

## Goals

- Keep one Python API surface while supporting local and remote runtimes.
- Add secure remote transport for production deployments.
- Maintain backward compatibility for existing socket users.

## Phase 1: Hardened TCP

- Add client-side connection/read timeouts in Python bindings.
- Add optional auth token in transport metadata (or dedicated auth frame).
- Add runtime flag to reject unauthenticated TCP requests by default when bound non-loopback.
- Add integration tests for cross-machine TCP transport.

## Phase 2: TLS for Binary Protocol

- Add `tls://host:port` endpoint support in Python and Rust clients.
- TLS modes:
  - server verification with CA bundle
  - optional mTLS with client cert/key
- Suggested Python API extension:
  - `KapslClient(protocol="tcp", host=..., port=..., tls=True, ca_cert=..., client_cert=..., client_key=...)`
- Keep framing protocol unchanged so only transport layer changes.

## Phase 3: gRPC Transport (Optional)

- Define a stable gRPC service:
  - `Infer`
  - `InferStream`
  - health/readiness methods
- Map existing request/response schema to protobuf messages.
- Add a gRPC-backed `TransportClient` implementation in Rust runtime.
- Add Python adapter so `KapslClient(..., protocol="grpc")` routes to grpc transport internally.

## Compatibility Policy

- `KapslClient("/tmp/kapsl.sock")` and `KapslClient("tcp://...")` remain valid.
- New security and protocol options are additive keyword arguments.
- Existing inference methods (`infer`, `infer_stream`) remain unchanged.
