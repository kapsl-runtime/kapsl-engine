# Authentication

`kapsl-runtime` supports role-based access control. When enabled, all `/api` endpoints and `/metrics` require a valid token.

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

### Option 2 — Shared fallback token

A single token that grants admin access:

```bash
export KAPSL_API_TOKEN="your-shared-secret"
kapsl --model model.aimod
```

### Option 3 — CLI flag at startup

```bash
kapsl --model model.aimod --admin-token "your-admin-secret"
```

### Option 4 — Auth store (persistent, managed via API)

The runtime stores users and API keys in `~/.kapsl/auth-store.json` (override with `KAPSL_AUTH_STORE_PATH`). Manage them via the web dashboard or the API (see below).

## Using a token in requests

### HTTP API

```bash
curl http://127.0.0.1:9095/api/models \
  -H "Authorization: Bearer your-token"
```

### kapsl-sdk (Python)

```python
from kapsl_sdk import KapslClient

client = KapslClient("tcp://127.0.0.1:9096", api_token="your-token")
```

The SDK attaches the token to every inference request automatically.

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

Returns a summary: auth enabled/disabled, number of users, legacy token config.

## Local (unauthenticated) mode

When no auth tokens are configured, the runtime runs in loopback-only mode. All `/api` endpoints accept connections from `127.0.0.1` / `::1` without a token. This is the default for local development.

The web dashboard detects this and authenticates automatically when accessed from the local machine.

## Security recommendations

- Do not expose the HTTP port to the internet without TLS and authentication
- Use `KAPSL_ALLOW_INSECURE_HTTP=1` only when behind a TLS-terminating reverse proxy
- Prefer per-user API keys over the shared `KAPSL_API_TOKEN` in production
- Rotate keys by creating a replacement, updating your services, then revoking the old key
