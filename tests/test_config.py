"""Config loader tests."""
from __future__ import annotations

import json
import os
import time

from app.config import ConfigStore, parse_config


def test_parse_config_basic():
    cfg = parse_config({
        "port": 8089,
        "providers": [
            {"name": "a", "base_url": "https://a", "api_key": "k",
             "models": ["m1", "m2"]},
            {"name": "b", "base_url": "https://b", "api_key": "k2",
             "models": ["m3"], "extra_headers": {"X-Custom": "v"}},
        ],
        "default_provider": "a",
    })
    assert cfg.port == 8089
    assert cfg.default_provider == "a"
    # Routing by model.
    assert cfg.find_provider("m1").name == "a"
    assert cfg.find_provider("m3").name == "b"
    # Unknown model falls back to default.
    assert cfg.find_provider("does-not-exist").name == "a"


def test_parse_config_first_match_wins():
    cfg = parse_config({
        "providers": [
            {"name": "a", "base_url": "u", "api_key": "k", "models": ["m"]},
            {"name": "b", "base_url": "u2", "api_key": "k2", "models": ["m"]},
        ],
    })
    assert cfg.find_provider("m").name == "a"


def test_parse_config_default_path():
    cfg = parse_config({
        "providers": [
            {"name": "a", "base_url": "https://api.x.com",
             "api_key": "k", "models": ["m"]},
        ],
    })
    assert cfg.providers[0].build_url("chat/completions") == \
        "https://api.x.com/v1/chat/completions"


def test_provider_build_url_custom_base_path():
    cfg = parse_config({
        "providers": [
            {"name": "a", "base_url": "https://api.x.com",
             "api_key": "k", "models": ["m"], "base_path": "/openai/v1"},
        ],
    })
    assert cfg.providers[0].build_url("chat/completions") == \
        "https://api.x.com/openai/v1/chat/completions"


def test_provider_build_url_trailing_slash():
    cfg = parse_config({
        "providers": [
            {"name": "a", "base_url": "https://api.x.com/",
             "api_key": "k", "models": ["m"]},
        ],
    })
    assert cfg.providers[0].build_url("/chat/completions") == \
        "https://api.x.com/v1/chat/completions"


def test_config_store_hot_reload(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "providers": [
            {"name": "a", "base_url": "u", "api_key": "k", "models": ["m1"]},
        ],
    }))
    store = ConfigStore(str(p))
    cfg = store.get()
    assert cfg is not None
    assert cfg.find_provider("m1").name == "a"

    # Update config on disk.
    new = {
        "providers": [
            {"name": "a", "base_url": "u", "api_key": "k", "models": ["m1", "m2"]},
        ],
    }
    p.write_text(json.dumps(new))
    # Bump mtime so it's clearly newer.
    os.utime(str(p), (time.time() + 2, time.time() + 2))

    cfg = store.get()
    assert cfg.find_provider("m2") is not None


def test_config_store_missing_file(tmp_path):
    p = tmp_path / "does-not-exist.json"
    store = ConfigStore(str(p))
    assert store.get() is None
