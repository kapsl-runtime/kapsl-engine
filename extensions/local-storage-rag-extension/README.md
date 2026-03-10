# Local Storage RAG Extension

This extension adds a local filesystem connector for the kapsl RAG extension runtime.

## Features

- validates `root_path` and optional `prefix`
- lists one local source (`ListSources`)
- syncs files as document deltas (`Sync`)
- fetches file bytes as base64 documents (`FetchDocument`)

## Files

- `rag-extension.toml`: extension manifest
- `connector`: executable sidecar entrypoint
- `src/connector.py`: JSON-over-stdio protocol implementation
- `example-config.json`: sample runtime config

## Config

Required:

- `root_path` (absolute path to local content)

Optional:

- `prefix`
- `source_id`
- `source_name`
- `include_extensions`
- `max_sync_documents`
- `follow_symlinks`

## Install Into Engine

```bash
curl -sS -X POST http://localhost:9095/api/extensions/install \
  -H 'Content-Type: application/json' \
  -d '{"path":"/Users/kiennguyen/Documents/Code/idx/framework/extensions/local-storage-rag-extension"}'
```

## Configure Workspace

```bash
curl -sS -X POST http://localhost:9095/api/extensions/connector.local-storage/config \
  -H 'Content-Type: application/json' \
  -d '{
    "workspace_id":"default",
    "config":{
      "root_path":"/Users/kiennguyen/Documents/Code/idx/framework",
      "prefix":"docs",
      "include_extensions":[".md", ".txt"],
      "max_sync_documents":1000,
      "follow_symlinks":false
    }
  }'
```

## Launch

```bash
curl -sS -X POST http://localhost:9095/api/extensions/connector.local-storage/launch \
  -H 'Content-Type: application/json' \
  -d '{"workspace_id":"default"}'
```

## Notes

- `Sync` currently emits `upsert` deltas only.
- Document IDs are relative paths under `root_path`.
