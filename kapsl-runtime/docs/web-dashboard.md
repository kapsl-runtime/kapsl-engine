# Web Dashboard

The runtime serves a built-in browser dashboard at `http://127.0.0.1:9095/`.

## Accessing the dashboard

Open `http://127.0.0.1:9095` in your browser. When auth is disabled (default local mode), the dashboard unlocks automatically. When auth is enabled, you are prompted for an API token.

## Views

### Dashboard

Real-time monitoring:

- Active models and their replica counts
- Inference throughput (requests/sec, tokens/sec)
- Request queue depth
- Hardware utilisation (CPU, GPU memory)
- Per-model latency histograms

### Models

- List all loaded models with status, backend, and queue depth
- Start a model from a local `.aimod` file (admin only)
- Stop or remove running models (admin only)
- View and edit auto-scaling policy per model

### Inference (interactive)

Run a test inference directly from the browser:

1. Select a model from the dropdown
2. Enter the input tensor as JSON or paste base64 data
3. Click **Run** — the response tensor is displayed inline

Useful for validating a newly deployed model without writing code.

### Extensions

Manage RAG connectors:

- Search the marketplace and install extensions
- View installed extensions and their status
- Configure workspace credentials for each connector
- Trigger a document sync manually
- **Developer Features** (admin only): install an extension from a local directory path

> The **Developer Features** toggle is only visible to admin sessions. Non-admin users cannot access local extension installation.

### Access

Admin-only section for managing authentication:

- View auth status (enabled/disabled, role counts)
- Create, list, and revoke API keys
- Manage users and their roles
- View last-used timestamps for API keys

### Settings

- Configure the remote registry URL for push/pull operations
- Toggle display preferences

## Session modes

| Mode | How it works |
|------|--------------|
| `local-loopback` | Auth disabled; loopback access automatically granted with admin privileges |
| `api-key` | Authenticated with a user API key |
| `legacy-token` | Authenticated with an older role-token (`KAPSL_API_TOKEN_*`) |

The current session mode and role are shown in the top status bar.

## Prometheus metrics

Metrics are available at `/metrics` (admin auth required when auth is enabled). Scrape this endpoint with Prometheus or any compatible collector.

Key metrics:

| Metric | Description |
|--------|-------------|
| `kapsl_infer_total` | Total inference requests by model |
| `kapsl_infer_duration_seconds` | Inference latency histogram |
| `kapsl_queue_depth` | Current queue depth per model |
| `kapsl_active_replicas` | Running replicas per model |
| `kapsl_tokens_generated_total` | Tokens generated (LLM models) |
