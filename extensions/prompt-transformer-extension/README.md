# Prompt Format Transformer Extension

This extension rewrites a plain text prompt into a model-specific chat prompt format with the right end-of-turn or EOS marker before inference.

## Features

- wraps plain prompts into Gemma, ChatML, or Llama 3 chat formats
- supports a custom format with configurable prefix/suffix tokens
- leaves already formatted prompts unchanged
- integrates with `/api/models/:id/infer` via the `prompt_transform` request block

## Files

- `rag-extension.toml`: extension manifest
- `connector`: executable sidecar entrypoint
- `src/connector.py`: JSON-over-stdio prompt transformer
- `example-config.json`: sample runtime config

## Config

Required:

- `format`: one of `gemma`, `chatml`, `llama3`, or `custom`

Optional:

- `trim_input`
- `bos_token`
- `user_prefix`
- `user_suffix`
- `assistant_prefix`
- `think_suffix`

## Built-in Formats

- `gemma`: `<bos>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`
- `chatml`: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
- `llama3`: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`

## Install Into Engine

```bash
curl -sS -X POST http://localhost:9095/api/extensions/install \
  -H 'Content-Type: application/json' \
  -d '{"path":"/Users/kiennguyen/Documents/Code/kapsl/kapsl-engine/extensions/prompt-transformer-extension"}'
```

## Configure Workspace

```bash
curl -sS -X POST http://localhost:9095/api/extensions/transformer.prompt-format/config \
  -H 'Content-Type: application/json' \
  -d '{
    "workspace_id":"default",
    "config":{
      "format":"gemma",
      "trim_input":true
    }
  }'
```

## Infer With Transformation

```bash
curl -sS -X POST http://localhost:9095/api/models/0/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "input":{
      "shape":[1,1],
      "dtype":"string",
      "data_base64":"V2hhdCBpcyBLSFYgY2FjaGU/"
    },
    "prompt_transform":{
      "workspace_id":"default",
      "extension_id":"transformer.prompt-format"
    }
  }'
```
