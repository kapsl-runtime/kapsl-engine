# OpenAI-Compatible API

The Kapsl runtime serves OpenAI-compatible text generation and embeddings
alongside its native `/api` routes. Clients using Chat Completions, stateless
text-only Responses, or Embeddings can be pointed at a running Kapsl runtime by
changing their base URL.

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
| `POST /v1/responses` | Text-only, stateless, streaming and non-streaming |
| `POST /v1/embeddings` | ONNX models packaged with `task=embed` |

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

curl http://127.0.0.1:9095/v1/responses \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "input": "Say hello in five words."
  }'

curl http://127.0.0.1:9095/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{
    "model": "all-minilm-l6-v2",
    "input": ["first document", "second document"]
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

response = client.responses.create(
    model="qwen2.5-7b-instruct",
    input="Say hello in five words.",
    store=False,
)
print(response.output_text)

embeddings = client.embeddings.create(
    model="all-minilm-l6-v2",
    input=["first document", "second document"],
)
print(embeddings.data[0].embedding)
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

## Chat Completions parameters

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

## Embeddings parameters

`POST /v1/embeddings` accepts every OpenAI input form: one string, an array of
strings, one integer token-ID array, or an array of token-ID arrays. String
inputs use the `tokenizer.json` packaged beside the model; token-ID inputs skip
tokenization. Models must be ONNX packages declared with `task=embed`, which
routes execution through Kapsl's masked-mean pooling and normalization backend.

`encoding_format` supports `float` (the default) and `base64`. Base64 values are
the little-endian bytes of the float32 vector. `user` is accepted as a replica
affinity key, with `X-Kapsl-Session` taking precedence. `usage.prompt_tokens`
and `usage.total_tokens` are exact for this endpoint and exclude padding.

`dimensions` may be used to request a vector no larger than the model's native
output. Kapsl shortens the native vector and L2-normalizes the result; it cannot
turn a model that was not trained for dimension shortening into one that was.

The endpoint enforces the OpenAI request ceilings: 8,192 tokens per input,
300,000 tokens across a request, and at most 2,048 inputs. A smaller tokenizer
or fixed model sequence limit wins. Empty and mixed-type inputs are rejected
with an OpenAI-shaped `400` error.

## Responses parameters

`POST /v1/responses` accepts a text `input` string or a text-only message array.
Message content may be a string or an array of `input_text` / `output_text`
parts. `instructions`, `max_output_tokens`, `temperature`, `top_p`, `seed`,
`stream`, and `metadata` are supported. `user`, `prompt_cache_key`, and
`safety_identifier` are accepted and used for replica/KV-cache affinity.
`top_k` and `repetition_penalty` are accepted as Kapsl extensions.

Responses are currently stateless. Omitted `store` and `store: false` are
accepted and the returned object reports `store: false`. Explicit `store: true`,
`previous_response_id`, Conversations, background mode, tools, reasoning
controls, structured output, logprobs, and non-text content are rejected with an
OpenAI-shaped `400` error rather than silently ignored.

Streaming uses the Responses API lifecycle events, including
`response.created`, `response.output_text.delta`, and `response.completed`.
Unlike Chat Completions, a Responses stream ends after its terminal typed event
and does not emit `data: [DONE]`.

## Chat Completions streaming

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
- **Embeddings currently require ONNX `task=embed` packages.** GGUF embedding,
  unusual encoder input tensors, and fixed batch sizes other than one are not
  supported. String input also requires a packaged Hugging Face
  `tokenizer.json`; token-ID input remains available without it.
- **No function/tool calling, structured output, or `logprobs`**, and no
  `/v1/completions`. `/v1/responses` supports plain text generation only.
