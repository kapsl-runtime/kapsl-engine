# OpenAI-Compatible API

The Kapsl runtime serves an OpenAI-compatible surface alongside its native
`/api` routes. Any client that speaks the OpenAI chat completions protocol can
be pointed at a running Kapsl runtime by changing its base URL.

```bash
export OPENAI_BASE_URL=http://127.0.0.1:9095/v1
export OPENAI_API_KEY=unused   # or your Kapsl API token, if auth is enabled
```

## Endpoints

| Endpoint | Notes |
| --- | --- |
| `GET /v1/models` | Models that are loaded and backed by a live pool |
| `GET /v1/models/{model}` | Single model, same resolution rules as below |
| `POST /v1/chat/completions` | Streaming and non-streaming |

These routes sit behind the same reader-scope API auth as
`/api/models/:id/infer`. When the runtime is started with auth enabled, send
the token as `Authorization: Bearer <token>` — which is what OpenAI clients
already do with `OPENAI_API_KEY`.

## Quick check

```bash
curl http://127.0.0.1:9095/v1/models

curl http://127.0.0.1:9095/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [{"role": "user", "content": "Say hello in five words."}]
  }'
```

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9095/v1", api_key="unused")

stream = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "Say hello in five words."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Naming a model

OpenAI clients address models by string; the native API addresses them by a
numeric id. The `model` field accepts, in order:

1. the name shown by `GET /v1/models` (case-insensitive)
2. `name:version`
3. a bare numeric model id, the same one `/api/models/:id/infer` takes

Replicas share a name with their primary; the primary is always chosen. If the
name matches nothing, the error lists the models that are actually loaded
rather than returning a bare "not found".

## Parameters

Honored, mapped onto the runtime sampler:

`max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `seed`, `stop`,
`stream`, `stream_options.include_usage`, `user`.

Also accepted as extensions, since the sampler supports them directly and other
compatible servers expose them: `top_k`, `repetition_penalty`.

`user` doubles as the KV-affinity session key, so a client that sets it keeps
hitting the same replica and reusing its prefix cache. An explicit
`X-Kapsl-Session` header takes precedence.

Rejected rather than silently ignored:

- `n` greater than 1 — the runtime returns a single choice per request.
- Non-text content parts (for example `image_url`) in a message.

Accepted and ignored: any other OpenAI field, including `frequency_penalty`,
`presence_penalty`, `logit_bias`, `response_format`, `tools`, and
`tool_choice`. Clients send these by default, and failing the request over them
would defeat the point of the surface. **They have no effect** — do not rely on
them to constrain output.

## Streaming

`stream: true` emits real `chat.completion.chunk` deltas as tokens are
generated, over the same scheduler path as `/api/models/:id/infer/stream`, and
terminates with `data: [DONE]`. Time-to-first-token is what the client
observes; deltas are not a finished completion chopped up after the fact.

Disconnecting mid-stream cancels the underlying generation rather than leaving
it running to completion.

A failure that happens after the stream is live cannot change the status line,
so it is sent in band as a `data: {"error": {...}}` event followed by
`[DONE]` — never as a `finish_reason: "stop"` chunk, which would report a
backend failure to the client as a successful empty answer.

## Known limitations

These are real gaps, not rough edges to discover in production:

- **`usage` token counts are estimates.** The scheduler returns generated text,
  not token counts. Streaming responses count emitted tokens exactly for
  `completion_tokens`; everything else is a ~4-characters-per-token estimate.
  Do not bill on these numbers.
- **Multi-turn prompts are rendered from the model *name*.** A single user
  message is passed to the backend untouched, so GGUF models apply the real
  chat template embedded in the file. A multi-turn conversation has to be
  flattened into one prompt before it reaches the backend, and the template
  family is guessed from the model name (Gemma, ChatML/Qwen, Llama 2, Llama 3).
  An unrecognized name falls back to a plain `role: content` transcript, which
  works but is not what the model was tuned on. Naming packages after their
  base model materially improves multi-turn quality.
- **`stop` is enforced at the API layer**, not in the sampler. Output is
  truncated at the first stop string, and streaming cancels generation at that
  point. Non-streaming requests still pay for the tokens generated past it.
- **No function/tool calling, no `logprobs`, no embeddings**, and no
  `/v1/completions`. Only `/v1/chat/completions`.
