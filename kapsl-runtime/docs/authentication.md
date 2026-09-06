# Authentication

`kapsl-runtime` supports role-based access control. Protected `/api` endpoints,
`/metrics`, and MCP use one engine-owned authorization evaluator. The public
`/api/auth/login` route validates credentials through that same evaluator.

The shared policy enforces roles, API-key scopes, expiry, revocation, suspended
users, and the authentication-disabled loopback fallback. Adapters supply the
credential and actual socket peer address; forwarded address headers do not
establish local trust. Policy changes apply to subsequent requests without a
restart. No backend implements this policy.

## Roles

| Role | Inference | Model management | Admin operations |
|------|-----------|-----------------|-----------------|
| `reader` | Yes | No | No |
| `writer` | Yes | Extensions, RAG sync | No |
| `admin` | Yes | Yes (all) | Yes (auth, metrics) |

## Enabling authentication

By default, the runtime only accepts connections from loopback (`127.0.0.1` / `::1`). To enable token-based auth for remote access:

### Option 1 — Environment variables (recommended for production)

Set tokens before starting the runtime:

```bash
export KAPSL_API_TOKEN_ADMIN="your-admin-secret"
export KAPSL_API_TOKEN_WRITER="your-writer-secret"
export KAPSL_API_TOKEN_READER="your-reader-secret"

kapsl --model model.aimod
```

### Option 2 — CLI flag at startup

```bash
kapsl --model model.aimod --admin-token "your-admin-secret"
```

### Option 3 — Auth store (persistent, managed via API)

The runtime stores users and API keys in `~/.kapsl/auth-store.json` (override with `KAPSL_AUTH_STORE_PATH`). Manage them via the web dashboard or the API (see below).

## Using a token in requests

### HTTP API

```bash
curl http://127.0.0.1:9095/api/models \
  -H "Authorization: Bearer your-token"
```

### MCP server

When the runtime is built with `--features mcp-server` and started with
`--mcp-port`, the Streamable HTTP endpoint uses the same bearer credentials.
All currently exposed MCP tools require reader access and `api:read` scope.

### kapsl-sdk (Python)

```python
from kapsl_sdk import KapslClient

client = KapslClient("tcp://127.0.0.1:9096", api_token="your-token")
```

The SDK attaches the token to every inference request automatically.

Native TCP currently uses the dedicated `KAPSL_TCP_AUTH_TOKEN`, verified by
`kapsl-ipc` on each request. It does **not** consult the HTTP/MCP API-key store,
roles, or scopes. The engine's central auth module also owns the native TCP
exposure check: a non-loopback bind requires a nonempty dedicated token.
Unix sockets and shared memory retain their local OS access controls.

Unifying native per-request API-key authorization requires an SDK transport
callback carrying the credential and peer context before request dispatch
(including OpenAI wire requests). The current `kapsl-ipc` dependency only
provides a static token setter; the engine does not duplicate its wire server.

Authorization evaluations and completed HTTP/MCP requests use the shared
logging pipeline. See [Logging](logging.md) for filtering and JSON output.

## Managing users and API keys

The auth management API requires an admin token.

### List users

```bash
curl http://127.0.0.1:9095/api/auth/access/users \
  -H "Authorization: Bearer <admin-token>"
```

### Create a user

```bash
curl -X POST http://127.0.0.1:9095/api/auth/access/users \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "role": "reader"}'
```

### Create an API key for a user

```bash
curl -X POST http://127.0.0.1:9095/api/auth/access/users/{user_id}/keys \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-service-key", "role": "reader"}'
```

The response includes the generated key. Store it securely — it cannot be retrieved again.

### Revoke an API key

```bash
curl -X POST http://127.0.0.1:9095/api/auth/access/keys/{key_id}/revoke \
  -H "Authorization: Bearer <admin-token>"
```

## Auth status

```bash
curl http://127.0.0.1:9095/api/auth/access/status \
  -H "Authorization: Bearer <admin-token>"
```

Returns a summary: auth enabled/disabled, number of users, and role-token configuration.

## Local (unauthenticated) mode

When no auth tokens are configured, the runtime runs in loopback-only mode. All `/api` endpoints accept connections from `127.0.0.1` / `::1` without a token. This is the default for local development.

The web dashboard detects this and authenticates automatically when accessed from the local machine.

## Security recommendations

- Do not expose the HTTP port to the internet without TLS and authentication
- Use `KAPSL_ALLOW_INSECURE_HTTP=1` only when behind a TLS-terminating reverse proxy
- Prefer per-user API keys over role tokens in production
- Rotate keys by creating a replacement, updating your services, then revoking the old key
