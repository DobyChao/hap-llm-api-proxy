"""Configuration loader with hot reload (mtime-based)."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    models: list[str]
    extra_headers: dict[str, str] = field(default_factory=dict)
    auth_token_file: Optional[str] = None
    # Optional path prefix appended after base_url (e.g. "/v1"). Defaults to "" so
    # upstream paths like /v1/chat/completions are appended verbatim.
    base_path: str = "/v1"

    def build_url(self, suffix: str) -> str:
        base = self.base_url.rstrip("/")
        prefix = (self.base_path or "").rstrip("/")
        if not suffix.startswith("/"):
            suffix = "/" + suffix
        return f"{base}{prefix}{suffix}"


@dataclass
class Config:
    port: int
    providers: list[Provider]
    default_provider: Optional[str] = None
    # runtime fields
    _model_map: dict[str, Provider] = field(default_factory=dict)
    _provider_map: dict[str, Provider] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._model_map = {}
        self._provider_map = {}
        for p in self.providers:
            self._provider_map[p.name] = p
            for m in p.models:
                # First provider to claim a model wins; this lets operators place
                # preferred providers earlier in the list.
                self._model_map.setdefault(m, p)

    def find_provider(self, model: Optional[str]) -> Optional[Provider]:
        """Route a request to a provider by model name.

        Falls back to default_provider when the model isn't claimed by anyone.
        """
        if model and model in self._model_map:
            return self._model_map[model]
        if self.default_provider and self.default_provider in self._provider_map:
            return self._provider_map[self.default_provider]
        return None

    def get_provider_by_name(self, name: str) -> Optional[Provider]:
        return self._provider_map.get(name)

    def list_models(self) -> list[tuple[str, Provider]]:
        return [(m, p) for m, p in self._model_map.items()]


def _parse_provider(raw: dict) -> Provider:
    return Provider(
        name=raw["name"],
        base_url=raw["base_url"],
        api_key=raw.get("api_key", ""),
        models=list(raw.get("models", [])),
        extra_headers=dict(raw.get("extra_headers", {})),
        auth_token_file=raw.get("auth_token_file"),
        base_path=raw.get("base_path", "/v1"),
    )


def parse_config(data: dict) -> Config:
    providers = [_parse_provider(p) for p in data.get("providers", [])]
    return Config(
        port=int(data.get("port", 8089)),
        providers=providers,
        default_provider=data.get("default_provider"),
    )


class ConfigStore:
    """Thread-safe config holder with mtime-based hot reload."""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._config: Optional[Config] = None
        self._mtime: float = -1.0

    def _read_mtime(self) -> float:
        try:
            return os.stat(self._path).st_mtime
        except FileNotFoundError:
            return -1.0

    def _load_locked(self) -> Optional[Config]:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            return None
        cfg = parse_config(raw)
        self._config = cfg
        self._mtime = self._read_mtime()
        return cfg

    def reload(self) -> Optional[Config]:
        """Force reload from disk."""
        with self._lock:
            return self._load_locked()

    def get(self) -> Optional[Config]:
        """Get current config, reloading if the file mtime changed."""
        mtime = self._read_mtime()
        with self._lock:
            if self._config is None or mtime != self._mtime:
                # Wait briefly to avoid reading a half-written file: if mtime
                # is changing under us, re-check after a tiny sleep.
                if mtime != self._mtime:
                    time.sleep(0.01)
                    mtime = self._read_mtime()
                self._load_locked()
            return self._config
