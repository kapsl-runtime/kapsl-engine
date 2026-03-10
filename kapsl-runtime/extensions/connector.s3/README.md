# S3 RAG Extension

This extension adds an S3 connector for the kapsl RAG extension runtime. It can:

- validate S3 connection/configuration
- list an S3-backed source
- sync object metadata as document deltas
- fetch full document bytes from S3

## Directory Layout

- `rag-extension.toml`: extension manifest consumed by `ExtensionRegistry`
- `connector`: sidecar entrypoint executable
- `src/connector.py`: protocol implementation (stdin/stdout JSON)
- `requirements.txt`: Python dependencies

## Requirements

- Python 3.9+
- `pip install -r requirements.txt`
- AWS credentials via one of:
  - static keys in config (`access_key_id`, `secret_access_key`, optional `session_token`)
  - AWS profile (`profile`)
  - standard AWS environment/instance credentials chain

## Config Schema

Required:

- `bucket` (string)

Optional:

- `region`
- `prefix`
- `endpoint_url`
- `profile`
- `access_key_id`
- `secret_access_key`
- `session_token`
- `source_id`
- `source_name`
- `max_keys` (1-1000)
- `max_sync_documents` (>=1)
- `include_extensions` (array of suffixes like `.md`, `.pdf`)
- `force_path_style` (for S3-compatible endpoints)

See `extensions/s3-rag-extension/example-config.json` for a starter config.

## Install Into Runtime

```bash
curl -sS -X POST http://localhost:9095/api/extensions/install \
  -H 'Content-Type: application/json' \
  -d '{"path":"/Users/kiennguyen/Documents/Code/idx/framework/extensions/s3-rag-extension"}'
```

## Set Workspace Config

```bash
curl -sS -X POST http://localhost:9095/api/extensions/connector.s3/config \
  -H 'Content-Type: application/json' \
  -d '{
    "workspace_id":"default",
    "config":{
      "bucket":"your-bucket",
      "region":"us-east-1",
      "prefix":"docs/",
      "include_extensions":[".md", ".txt", ".pdf"]
    }
  }'
```

## Launch Connector

```bash
curl -sS -X POST http://localhost:9095/api/extensions/connector.s3/launch \
  -H 'Content-Type: application/json' \
  -d '{"workspace_id":"default"}'
```

## Notes

- `ValidateConfig` performs `head_bucket` and fails fast on bad credentials/bucket access.
- `Sync` returns only `upsert` deltas. Delete detection is not tracked yet.
- Document IDs are S3 object keys.
