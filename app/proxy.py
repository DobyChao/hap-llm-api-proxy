"""HTTP proxy logic and route handlers."""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .auth_token import AuthTokenStore
from .config import Config, ConfigStore, Provider
from .reasoning import (
    StreamingReasoningNormalizer,
    format_sse_data,
    normalize_response,
    parse_sse_line,
)


# Hop-by-hop headers we must not forward in either direction.
HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
    "content-encoding",
})


def build_upstream_headers(
    provider: Provider,
    request_headers: dict[str, str],
    auth_store: AuthTokenStore,
) -> dict[str, str]:
    """Compose headers to send to the upstream provider.

    Order: static extra_headers < auth token < Authorization < Content-Type.
    Auth token always wins over a static extra header with the same name.
    """
    headers: dict[str, str] = {}
    # Forward a safe subset of inbound headers (mostly Content-Type).
    for k, v in request_headers.items():
        lk = k.lower()
        if lk in HOP_BY_HOP:
            continue
        if lk in ("authorization", "x-auth-token"):
            continue  # we set our own
        if lk.startswith("x-") or lk in ("content-type", "accept"):
            headers[k] = v

    # Default Content-Type for JSON bodies.
    headers.setdefault("Content-Type", "application/json")

    # Static extra headers from config.
    for k, v in provider.extra_headers.items():
        headers[k] = v

    # Authorization with provider api_key.
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"

    # Dynamic auth token, if configured for this provider.
    token = auth_store.get(provider.auth_token_file)
    if token:
        headers["x-auth-token"] = token

    return headers


def _select_provider(config: Config, body: dict[str, Any]) -> Provider:
    model = body.get("model") if isinstance(body, dict) else None
    provider = config.find_provider(model)
    if provider is None:
        raise HTTPException(
            status_code=404,
            detail=f"No provider configured for model {model!r}",
        )
    return provider


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in headers.multi_items():
        if k.lower() in HOP_BY_HOP:
            continue
        out[k] = v
    return out


async def forward_non_streaming(
    client: httpx.AsyncClient,
    provider: Provider,
    path: str,
    request_headers: dict[str, str],
    body: Any,
    auth_store: AuthTokenStore,
) -> Response:
    url = provider.build_url(path)
    headers = build_upstream_headers(provider, request_headers, auth_store)
    try:
        resp = await client.post(url, headers=headers, json=body)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    out_headers = _filter_response_headers(resp.headers)
    payload = resp.text

    # Apply reasoning normalization only to chat completions and only if JSON.
    if path.endswith("/chat/completions") and resp.is_success:
        try:
            data = resp.json()
            if isinstance(data, dict):
                normalize_response(data)
                payload = json.dumps(data, ensure_ascii=False)
                # Drop the upstream Content-Length; FastAPI will recompute.
                out_headers.pop("Content-Length", None)
                out_headers.pop("content-length", None)
        except (json.JSONDecodeError, ValueError):
            pass

    return Response(
        content=payload,
        status_code=resp.status_code,
        headers=out_headers,
        media_type=out_headers.get("Content-Type") or out_headers.get("content-type"),
    )


async def iter_normalized_sse(
    response: httpx.Response,
) -> AsyncIterator[str]:
    """Iterate upstream SSE lines, normalize reasoning, yield SSE strings."""
    normalizer = StreamingReasoningNormalizer()
    # Buffer for partial lines (the upstream may split an SSE line across chunks).
    pending = ""
    async for raw in response.aiter_text():
        if not raw:
            continue
        pending += raw
        # SSE lines are separated by \n. A trailing \n leaves an empty final
        # element which we keep in pending until more data arrives.
        *complete, pending = pending.split("\n")
        for line in complete:
            stripped = line.rstrip("\r")
            if not stripped:
                # blank separator line -- emit as-is so client sees event boundary
                yield "\n"
                continue
            if not stripped.startswith("data:"):
                # event/id/retry lines pass through untouched
                yield stripped + "\n"
                continue
            parsed = parse_sse_line(stripped)
            if parsed is None:
                # Couldn't parse as JSON; pass through untouched.
                yield stripped + "\n"
                continue
            if parsed.get("__done__"):
                yield "data: [DONE]\n\n"
                continue
            for piece in normalizer.process_chunk(parsed):
                yield format_sse_data(piece)
            # The normalizer may produce zero pieces if a chunk had no choices;
            # in that case we silently drop it. This is rare but acceptable.

    # Drain trailing line.
    if pending:
        stripped = pending.rstrip("\r")
        if stripped.startswith("data:"):
            parsed = parse_sse_line(stripped)
            if parsed is not None and not parsed.get("__done__"):
                for piece in normalizer.process_chunk(parsed):
                    yield format_sse_data(piece)
            elif parsed is not None and parsed.get("__done__"):
                yield "data: [DONE]\n\n"
            else:
                yield stripped + "\n"
        elif stripped:
            yield stripped + "\n"


async def forward_streaming(
    client: httpx.AsyncClient,
    provider: Provider,
    path: str,
    request_headers: dict[str, str],
    body: Any,
    auth_store: AuthTokenStore,
) -> StreamingResponse:
    url = provider.build_url(path)
    headers = build_upstream_headers(provider, request_headers, auth_store)

    req = client.build_request("POST", url, headers=headers, json=body)
    try:
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    if resp.status_code >= 400:
        # Read body and surface as a normal HTTP error.
        await resp.aread()
        body_text = resp.text
        await resp.aclose()
        raise HTTPException(status_code=resp.status_code, detail=body_text)

    out_headers = _filter_response_headers(resp.headers)
    out_headers.setdefault("Content-Type", "text/event-stream")
    out_headers.setdefault("Cache-Control", "no-cache")
    out_headers.setdefault("Connection", "keep-alive")

    async def gen() -> AsyncIterator[bytes]:
        try:
            async for chunk in iter_normalized_sse(resp):
                yield chunk.encode("utf-8")
        finally:
            await resp.aclose()

    return StreamingResponse(
        content=gen(),
        status_code=resp.status_code,
        headers=out_headers,
        media_type="text/event-stream",
    )


async def proxy_chat_completions(
    request: Request,
    body: dict[str, Any],
    config_store: ConfigStore,
    auth_store: AuthTokenStore,
    client: httpx.AsyncClient,
) -> Response:
    config = config_store.get()
    if config is None:
        raise HTTPException(status_code=503, detail="proxy not configured")

    provider = _select_provider(config, body)

    # Caller asks for streaming?
    stream = bool(body.get("stream"))

    req_headers = {k: v for k, v in request.headers.items()}
    if stream:
        return await forward_streaming(
            client, provider, "chat/completions", req_headers, body, auth_store
        )
    return await forward_non_streaming(
        client, provider, "chat/completions", req_headers, body, auth_store
    )


def build_models_response(config: Config) -> dict[str, Any]:
    now = int(time.time())
    data = []
    seen = set()
    for model, _provider in config.list_models():
        if model in seen:
            continue
        seen.add(model)
        data.append({"id": model, "object": "model", "created": now, "owned_by": "llm-api-proxy"})
    return {"object": "list", "data": data}


async def proxy_models(
    config_store: ConfigStore,
) -> dict[str, Any]:
    config = config_store.get()
    if config is None:
        raise HTTPException(status_code=503, detail="proxy not configured")
    return build_models_response(config)
