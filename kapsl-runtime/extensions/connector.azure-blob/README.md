# Azure Blob RAG Extension

This extension adds an Azure Blob Storage connector for the kapsl RAG extension runtime.

## Features

- validates container access (`ValidateConfig`)
- lists one Blob-backed source (`ListSources`)
- syncs blob metadata as document deltas (`Sync`)
- fetches blob bytes as base64 documents (`FetchDocument`)

## Files

- `rag-extension.toml`: extension manifest
- `connector`: executable sidecar entrypoint
- `src/connector.py`: JSON-over-stdio protocol implementation
- `requirements.txt`: Python dependencies
- `example-config.json`: sample runtime config

## Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

## Config

Required:

- `container`

Also provide one of:

- `connection_string`
- `account_url` (and optionally `credential`)

Optional:

- `prefix`
- `source_id`
- `source_name`
- `include_extensions`
- `max_sync_documents`

## Install Into Engine

```bash
curl -sS -X POST http://localhost:9095/api/extensions/install \
  -H 'Content-Type: application/json' \
  -d '{"path":"/Users/kiennguyen/Documents/Code/idx/framework/extensions/azure-blob-rag-extension"}'
```

## Configure Workspace

```bash
curl -sS -X POST http://localhost:9095/api/extensions/connector.azure-blob/config \
  -H 'Content-Type: application/json' \
  -d '{
    "workspace_id":"default",
    "config":{
      "container":"my-rag-container",
      "connection_string":"DefaultEndpointsProtocol=https;AccountName=...",
      "prefix":"knowledge/",
      "include_extensions":[".md", ".txt", ".pdf"]
    }
  }'
```

## Launch

```bash
curl -sS -X POST http://localhost:9095/api/extensions/connector.azure-blob/launch \
  -H 'Content-Type: application/json' \
  -d '{"workspace_id":"default"}'
```

## Notes

- `Sync` currently emits `upsert` deltas only.
- Blob `id` is the blob name.
