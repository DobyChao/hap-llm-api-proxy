"""FastAPI application entrypoint."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .auth_token import AuthTokenStore
from .config import ConfigStore
from .proxy import proxy_chat_completions, proxy_models


CONFIG_PATH_ENV = "LLM_PROXY_CONFIG"
DEFAULT_CONFIG_PATH = "/etc/llm-api-proxy/config.json"


def _config_path() -> str:
    return os.environ.get(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH


class AppState:
    def __init__(self, no_proxy: bool = False) -> None:
        self.config_store = ConfigStore(_config_path())
        self.auth_store = AuthTokenStore()
        self.no_proxy = no_proxy
        # Use a single shared client for connection pooling.
        self.client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        # Refresh config so we fail fast if it's broken at startup.
        cfg = self.config_store.reload()
        if cfg is None:
            # Not fatal -- it might appear later via hot reload.
            pass
        client_kwargs: dict[str, Any] = {"timeout": httpx.Timeout(300.0, connect=10.0)}
        if self.no_proxy:
            client_kwargs["proxy"] = None
            client_kwargs["trust_env"] = False
        self.client = httpx.AsyncClient(**client_kwargs)

    async def stop(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState(no_proxy=getattr(app.state, "no_proxy", False))
    app.state.proxy = state
    await state.start()
    try:
        yield
    finally:
        await state.stop()


def create_app(no_proxy: bool = False) -> FastAPI:
    app = FastAPI(title="llm-api-proxy", lifespan=lifespan)
    app.state.no_proxy = no_proxy

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        state: AppState = request.app.state.proxy
        try:
            body: dict[str, Any] = await request.json()
        except Exception as exc:
            return JSONResponse(
                status_code=400, content={"error": {"message": f"invalid JSON body: {exc}"}}
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400, content={"error": {"message": "request body must be an object"}}
            )
        assert state.client is not None
        return await proxy_chat_completions(
            request, body, state.config_store, state.auth_store, state.client
        )

    @app.get("/v1/models")
    async def models(request: Request):
        state: AppState = request.app.state.proxy
        return await proxy_models(state.config_store)

    return app


app = create_app()
