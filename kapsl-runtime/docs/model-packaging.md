# Model Packaging

Models are distributed as `.aimod` archives — self-contained packages that bundle model weights, a manifest, and optional metadata. The runtime loads `.aimod` files directly; it does not accept raw weight files.

## Build a package

### Via CLI

```bash
kapsl package \
  --model /path/to/model.onnx \
  --output /path/to/model.aimod
```

### Via HTTP API (admin)

```bash
curl -X POST http://127.0.0.1:9095/api/engine/package \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/absolute/path/to/model.onnx",
    "output_path": "/absolute/path/to/model.aimod"
  }'
```

## Package registry

### Push to registry

```bash
# CLI
kapsl push \
  --package /path/to/model.aimod \
  --remote https://registry.kapsl.ai

# HTTP API
curl -X POST http://127.0.0.1:9095/api/engine/push \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "package_path": "/path/to/model.aimod",
    "remote_url": "https://registry.kapsl.ai"
  }'
```

### Pull from registry

```bash
# CLI
kapsl pull \
  --package my-model:v1.2 \
  --remote https://registry.kapsl.ai \
  --output /path/to/model.aimod

# HTTP API
curl -X POST http://127.0.0.1:9095/api/engine/pull \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "package_id": "my-model:v1.2",
    "output_path": "/path/to/model.aimod",
    "remote_url": "https://registry.kapsl.ai"
  }'
```

## Load a model at runtime

A running server can load additional models without restart:

```bash
curl -X POST http://127.0.0.1:9095/api/models/start \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/model.aimod",
    "model_id": 1
  }'
```

| Field | Required | Description |
|-------|----------|-------------|
| `model_path` | Yes | Absolute path to the `.aimod` file |
| `model_id` | No | Integer ID to assign; auto-assigned if omitted |
| `topology` | No | `"data-parallel"` (default) |
| `tp_degree` | No | Tensor-parallelism degree (default `1`) |

## Supported model formats inside a package

| Format | Extension | Notes |
|--------|-----------|-------|
| ONNX | `.onnx` | Recommended — widest backend support |
| GGUF | `.gguf` | LLMs (llama.cpp compatible) |
| SafeTensors | `.safetensors` | Native-enabled runtime or managed vLLM on Linux with the certified bundle |

PyTorch (`.pt`, `.pth`), TensorFlow (`.pb`), and unknown primary model
extensions are rejected during build and load. Kapsl never treats these bytes
as ONNX. Export the model to ONNX, GGUF, or a supported SafeTensors deployment
first.

## Serving backend policy

Packages may declare a deployment-time policy at
`metadata.serving.backend`. The supported values are `auto`, `llama_cpp`, and
`vllm`. Omitting the field preserves the backend-factory behavior used before
this policy existed; an old package is never redirected to vLLM merely because
it is loaded on a CUDA host.

The build command writes the field without replacing other metadata:

```bash
# A GGUF package that must use llama.cpp.
kapsl build ./model.gguf \
  --format gguf --model-type causal-lm --task generate \
  --serving-backend llama_cpp

# A deployment-portable policy. On a CUDA host with the certified bundle, this
# SafeTensors contract resolves to managed vLLM; without CUDA it requires a
# native-enabled Kapsl binary.
kapsl build ./model.safetensors \
  --format safetensors --model-type causal-lm --task generate \
  --serving-backend auto
```

Selection is deterministic:

| Policy | Package/hardware | Selected target |
|--------|------------------|-----------------|
| omitted | any | Existing built-in backend factory |
| `llama_cpp` | GGUF | llama.cpp; other formats are rejected |
| `vllm` | CUDA + explicit SafeTensors/causal-lm/generate axes | Managed vLLM; other contracts are rejected |
| `auto` | GGUF | llama.cpp |
| `auto` | CUDA + explicit SafeTensors/causal-lm/generate axes | Managed vLLM |
| `auto` | SafeTensors without a matching external target | Built-in native backend when compiled; otherwise rejected before ONNX Runtime |
| `auto` | anything else | Existing built-in backend factory |

Inspect the decision on the current host, or override CUDA detection when
planning for another deployment host:

```bash
kapsl backend-plan ./model.aimod
kapsl backend-plan ./model.aimod --cuda true
```

The command emits machine-readable JSON including `selected_backend`,
`external_process`, and the reason for the decision. `external_process` is true
for vLLM because Kapsl supervises a separate Python process; it does not mean
the user starts or addresses a second server. Run the package normally:

```bash
kapsl run --model ./model.aimod
```

Kapsl generates the `KapslConnectorV1` configuration and shared-pool allowlist,
waits for vLLM readiness, and keeps the Kapsl HTTP/native endpoints in front.
The exact bundle and fail-closed checks are described in
[Configuration](./configuration.md#serving-backend-policy).

## Model lifecycle

```
load (via --model flag or /api/models/start)
  └── serving (model is active)
        ├── POST /api/models/:id/stop   → stopped (idle, not unloaded)
        └── POST /api/models/:id/remove → removed from registry
```

```bash
# Stop a loaded model (keeps it in registry)
curl -X POST http://127.0.0.1:9095/api/models/0/stop \
  -H "Authorization: Bearer <admin-token>"

# Remove it entirely
curl -X POST http://127.0.0.1:9095/api/models/0/remove \
  -H "Authorization: Bearer <admin-token>"
```

## Auto-scaling policy

Each model can have a scaling policy controlling how many replicas run in parallel:

```bash
curl -X POST http://127.0.0.1:9095/api/models/0/scaling \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "min_replicas": 1,
    "max_replicas": 4,
    "target_queue_depth": 5,
    "scale_down_threshold": 2,
    "cooldown_seconds": 300
  }'
```

| Field | Default | Description |
|-------|---------|-------------|
| `min_replicas` | `1` | Minimum replicas always running |
| `max_replicas` | `4` | Upper replica limit |
| `target_queue_depth` | `5` | Scale up when queue exceeds this |
| `scale_down_threshold` | `2` | Scale down when queue falls below this |
| `cooldown_seconds` | `300` | Minimum time between scale-down events |

## Model caching

The runtime uses a content-addressed cache for unpacked model weights. Environment variables:

| Variable | Description |
|----------|-------------|
| `KAPSL_MODEL_CACHE_DIR` | Cache root (default: `.kapsl-model-cache/` next to the `.aimod` file) |
| `KAPSL_MODEL_CACHE_MAX_MIB` | Maximum cache size in MiB; LRU eviction enforced |
| `KAPSL_MODEL_CACHE_RESERVED_FREE_MIB` | Minimum free disk space to maintain |
