# llm-api-proxy

A lightweight OpenAI-compatible reverse proxy for LLM APIs with request/response
hooks. Built with FastAPI + httpx.

## Features

- Routes requests to multiple upstream providers based on the requested model.
- Streaming (SSE) and non-streaming forwarding.
- **Request hook — `x-auth-token` injection**: reads a dynamic auth token from a
  file (refreshed externally) and injects it as a request header. Hot-reloads
  when the file mtime changes.
- **Response hook — `reasoning_content` normalization**: renames provider-specific
  fields (`reasoning`, `thinking`, `thought`) to the canonical
  `reasoning_content`, and extracts `<think>...</think>` blocks embedded inline
  in the `content` field into `reasoning_content`. Works for non-streaming and
  streaming responses (including chunks where the `<think>` tag is split).
- JSON config with hot reload (mtime-based).
- systemd unit included.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export LLM_PROXY_CONFIG=./config.example.json
uvicorn app.main:app --port 8089
```

By default the proxy respects system HTTP/HTTPS proxy environment variables. To
disable system proxy use (e.g. when upstream is reachable directly), use the
`--no-proxy` flag via the module entry point:

```bash
python -m app --no-proxy --port 8089
```

Then:

```bash
curl http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": false
  }'
```

## Configuration

```json
{
  "port": 8089,
  "providers": [
    {
      "name": "provider-a",
      "base_url": "https://api.provider-a.com",
      "api_key": "sk-xxx",
      "models": ["gpt-4o", "gpt-4o-mini"],
      "extra_headers": {},
      "auth_token_file": "/var/run/llm-proxy/provider-a-token"
    }
  ],
  "default_provider": "provider-a"
}
```

| Field | Description |
|-------|-------------|
| `name` | Identifier; referenced by `default_provider`. |
| `base_url` | Upstream base URL. |
| `base_path` | Optional URL prefix appended after base_url. Defaults to `/v1`. |
| `api_key` | Sent as `Authorization: Bearer <api_key>`. |
| `models` | Models this provider serves; first match wins. |
| `extra_headers` | Static headers added to every upstream request. |
| `auth_token_file` | File path whose contents are sent as `x-auth-token` (dynamic). |

### Auth token file

The token file is read on every request but cached by mtime. To rotate the
token, an external process atomically rewrites the file (e.g. write to a temp
file and `rename`). The proxy picks up the new value on the next request.

### Hot reload

The JSON config is reloaded when its mtime changes — no restart required. Model
routing, provider credentials, and auth token file paths are all reloaded.

## Running under systemd

```bash
sudo cp llm-api-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llm-api-proxy
```

The unit expects the repo at `/opt/llm-api-proxy`, an environment file at
`/etc/llm-proxy/env`, and the config at the path set by `LLM_PROXY_CONFIG`.

## Tests

```bash
source .venv/bin/activate
pip install -r requirements.txt pytest pytest-asyncio
pytest -q
```
