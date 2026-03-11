# Extensions & RAG

Extensions are installable connectors that plug external data sources into the runtime's retrieval-augmented generation (RAG) pipeline. Once installed, an extension can sync documents from a source (S3, Azure Blob, local filesystem, etc.) into a vector store that the runtime uses to augment inference requests.

## How it works

```
Data source (S3 / Azure / local)
    │  sync
    ▼
Extension connector (sidecar or Wasm)
    │  indexes
    ▼
Vector store (kapsl-rag)
    │  retrieval
    ▼
Model inference (augmented prompt)
```

1. An extension connector pulls documents from a data source.
2. The runtime indexes the documents into its vector store.
3. On inference, the runtime retrieves relevant document chunks and injects them into the request.

## Browse the marketplace

```bash
curl "http://127.0.0.1:9095/api/extensions/marketplace?q=s3" \
  -H "Authorization: Bearer <writer-token>"
```

## Install an extension

### From the marketplace

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/install \
  -H "Authorization: Bearer <writer-token>" \
  -H "Content-Type: application/json" \
  -d '{"extension_id": "connector.s3"}'
```

### From a local directory (admin / developer mode)

Use the web dashboard with **Developer Features** enabled, or via API:

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/install \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"local_path": "/path/to/extension-dir"}'
```

The directory must contain a `rag-extension.toml` manifest.

## List installed extensions

```bash
curl http://127.0.0.1:9095/api/extensions \
  -H "Authorization: Bearer <writer-token>"
```

## Configure an extension

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/connector.s3/config \
  -H "Authorization: Bearer <writer-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "my-workspace",
    "config": {
      "bucket": "my-bucket",
      "region": "us-east-1",
      "prefix": "docs/"
    }
  }'
```

## Start a connector

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/connector.s3/launch \
  -H "Authorization: Bearer <writer-token>"
```

## Trigger a sync

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/connector.s3/sync \
  -H "Authorization: Bearer <writer-token>" \
  -H "Content-Type: application/json" \
  -d '{"workspace_id": "my-workspace"}'
```

## Query augmented inference

Once documents are synced, include a `rag` block in your inference request:

```bash
curl -X POST http://127.0.0.1:9095/api/models/0/infer \
  -H "Authorization: Bearer <reader-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "shape": [1, 32],
      "dtype": "string",
      "data": [72, 101, 108, 108, 111]
    },
    "rag": {
      "workspace_id": "my-workspace",
      "top_k": 4,
      "min_score": 0.1,
      "max_context_tokens": 768
    }
  }'
```

Or query the RAG store directly without inference:

```bash
curl -X POST http://127.0.0.1:9095/api/rag/query \
  -H "Authorization: Bearer <reader-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "my-workspace",
    "query": "What is the refund policy?",
    "top_k": 4
  }'
```

## Uninstall an extension

```bash
curl -X POST http://127.0.0.1:9095/api/extensions/connector.s3/uninstall \
  -H "Authorization: Bearer <writer-token>"
```

## Extension manifest format

Custom extensions must include a `rag-extension.toml`:

```toml
[extension]
id = "connector.my-source"
name = "My Data Source"
version = "1.0.0"
description = "Syncs documents from My Data Source"
runtime = "sidecar"   # or "wasm"

[[capabilities]]
type = "sync"

[[auth]]
method = "api-key"
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `KAPSL_EXTENSIONS_ROOT` | Directory where extensions are installed |
| `KAPSL_EXT_CONFIG_ROOT` | Directory for extension configuration files |
| `KAPSL_RAG_STORAGE_ROOT` | Directory for the vector store data |
| `KAPSL_EXTENSION_MARKETPLACE_URL` | Override the marketplace endpoint |
