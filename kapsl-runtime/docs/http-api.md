# HTTP API Reference

The runtime exposes a REST API on `http://127.0.0.1:9095` by default. All endpoints use JSON unless noted. Authentication details are in [Authentication](./authentication.md).

## Base URL

```
http://<host>:<port>/api
```

Default: `http://127.0.0.1:9095/api`

---

## Health

### GET /api/health

Check that the runtime is up.

**Auth**: none required

```bash
curl http://127.0.0.1:9095/api/health
```

```json
{"status": "ok"}
```

---

## Models

### GET /api/models

List all loaded models.

**Auth**: reader

```bash
curl http://127.0.0.1:9095/api/models \
  -H "Authorization: Bearer <token>"
```

### GET /api/models/:id

Get details for one model.

**Auth**: reader

### POST /api/models/start

Load a model from disk.

**Auth**: admin

```json
{
  "model_path": "/path/to/model.aimod",
  "model_id": 1,
  "topology": "data-parallel",
  "tp_degree": 1
}
```

### POST /api/models/:id/stop

Stop a model (keeps it registered).

**Auth**: admin

### POST /api/models/:id/remove

Unload and remove a model from the registry.

**Auth**: admin

### GET /api/models/:id/scaling

Read the auto-scaling policy for a model.

**Auth**: reader

### POST /api/models/:id/scaling

Update the auto-scaling policy.

**Auth**: admin

```json
{
  "min_replicas": 1,
  "max_replicas": 4,
  "target_queue_depth": 5,
  "scale_down_threshold": 2,
  "cooldown_seconds": 300
}
```

---

## Inference

### POST /api/models/:id/infer

Run inference on a loaded model.

**Auth**: reader

#### Tensor payload

```json
{
  "input": {
    "shape": [1, 1, 28, 28],
    "dtype": "float32",
    "data": [0, 0, 0, 63]
  }
}
```

Large tensors can be sent as base64 to avoid JSON number overhead:

```json
{
  "input": {
    "shape": [1, 1, 28, 28],
    "dtype": "float32",
    "data_base64": "AAAAPw=="
  }
}
```

Additional named inputs (for multi-input models):

```json
{
  "input": { "shape": [1, 10], "dtype": "int64", "data": [...] },
  "additional_inputs": [
    {
      "name": "style",
      "tensor": { "shape": [1, 1, 256], "dtype": "float32", "data": [...] }
    }
  ]
}
```

Full optional fields:

```json
{
  "input": { ... },
  "additional_inputs": [],
  "session_id": "session-abc",
  "metadata": {
    "request_id": "req-1",
    "timeout_ms": 5000,
    "priority": 1,
    "force_cpu": false,
    "auth_token": "token-if-not-in-header"
  }
}
```

#### Media payload (images / video)

```json
{
  "media": {
    "kind": "image",
    "mime_type": "image/png",
    "base64": "<base64-encoded-image>"
  },
  "tensor_options": {
    "target_width": 224,
    "target_height": 224,
    "layout": "nchw",
    "channels": "rgb",
    "dtype": "float32",
    "normalize": "zero_to_one"
  }
}
```

| `normalize` value | Range | Description |
|------------------|-------|-------------|
| `zero_to_one` | [0, 1] | Divide pixels by 255 |
| `minus_one_to_one` | [−1, 1] | Standard normalisation |
| `imagenet` | mean/std | ImageNet channel normalisation |
| `none` | raw | No normalisation |

Video fields: `frame_count`, `frame_stride`, `start_time_ms`, `end_time_ms` (requires `ffmpeg`).

#### dtype values

`float32`, `float64`, `float16`, `int32`, `int64`, `uint8`, `string`

---

## RAG

### POST /api/rag/query

Query indexed documents with vector search.

**Auth**: reader

```json
{
  "workspace_id": "my-workspace",
  "query": "What is the capital of France?",
  "top_k": 4,
  "min_score": 0.1
}
```

---

## System

### GET /api/hardware

List detected hardware accelerators (GPUs, etc.).

**Auth**: reader

### GET /api/system/stats

Runtime statistics: queue depth, active requests, throughput.

**Auth**: reader

### GET /metrics

Prometheus metrics endpoint.

**Auth**: admin

---

## Package management

### POST /api/engine/package

Build an `.aimod` package from a model file.

**Auth**: admin

```json
{
  "model_path": "/absolute/path/to/model.onnx",
  "output_path": "/absolute/path/to/model.aimod"
}
```

### POST /api/engine/push

Push an `.aimod` package to a remote registry.

**Auth**: admin

```json
{
  "package_path": "/path/to/model.aimod",
  "remote_url": "https://registry.kapsl.ai"
}
```

### POST /api/engine/pull

Pull an `.aimod` package from a remote registry.

**Auth**: admin

```json
{
  "package_id": "my-model:v1.0",
  "output_path": "/path/to/model.aimod",
  "remote_url": "https://registry.kapsl.ai"
}
```

---

## Extensions

### GET /api/extensions

List installed extensions.

**Auth**: writer

### GET /api/extensions/marketplace

Search the extension marketplace.

**Auth**: writer
**Query**: `?q=search-term`

### POST /api/extensions/install

Install an extension from the marketplace or a local path.

**Auth**: writer

```json
{"extension_id": "connector.s3"}
```

### POST /api/extensions/:id/uninstall

Uninstall an extension.

**Auth**: writer

### POST /api/extensions/:id/config

Update extension configuration.

**Auth**: writer

### POST /api/extensions/:id/launch

Start an extension sidecar process.

**Auth**: writer

### POST /api/extensions/:id/sync

Trigger a RAG sync for a connector extension.

**Auth**: writer

---

## Authentication management

### POST /api/auth/login

Validate a token and return session info. Used by the web dashboard on login.

**Auth**: none (presents token in body or `Authorization` header)

```json
{"token": "my-api-key"}
```

Response:

```json
{
  "authenticated": true,
  "role": "admin",
  "mode": "api-key",
  "access": {"read": true, "write": true, "admin": true}
}
```

### GET /api/auth/access/status

Summary of the current auth configuration.

**Auth**: admin

### GET /api/auth/access/users

List all users.

**Auth**: admin

### POST /api/auth/access/users

Create a new user.

**Auth**: admin

```json
{"username": "alice", "role": "reader"}
```

### POST /api/auth/access/users/:id/keys

Create an API key for a user.

**Auth**: admin

```json
{"name": "service-key", "role": "reader"}
```

### POST /api/auth/access/keys/:id/revoke

Revoke an API key.

**Auth**: admin

### GET /api/auth/roles

Read legacy role-token configuration.

**Auth**: admin

### POST /api/auth/roles

Update legacy role-token configuration.

**Auth**: admin
