"""Auth token store tests."""
from __future__ import annotations

import os
import time

from app.auth_token import AuthTokenStore


def test_returns_none_for_no_path():
    store = AuthTokenStore()
    assert store.get(None) is None
    assert store.get("") is None


def test_returns_none_for_missing_file(tmp_path):
    store = AuthTokenStore()
    assert store.get(str(tmp_path / "missing")) is None


def test_reads_token_stripped(tmp_path):
    f = tmp_path / "tok"
    f.write_text("  abc123\n")
    store = AuthTokenStore()
    assert store.get(str(f)) == "abc123"


def test_caches_until_mtime_changes(tmp_path):
    f = tmp_path / "tok"
    f.write_text("first")
    store = AuthTokenStore()
    assert store.get(str(f)) == "first"
    # Same mtime -> same value, even if we monkey-patch the open reader.
    f.write_text("first")  # same content, may or may not bump mtime
    # Force identical mtime by setting it explicitly.
    os.utime(str(f), (1000, 1000))
    store.get(str(f))  # populates cache with mtime 1000
    # Now change content without changing mtime -- we should still serve cached.
    f.write_text("second")
    os.utime(str(f), (1000, 1000))
    assert store.get(str(f)) == "first"


def test_reloads_on_mtime_change(tmp_path):
    f = tmp_path / "tok"
    f.write_text("first")
    os.utime(str(f), (1000, 1000))
    store = AuthTokenStore()
    assert store.get(str(f)) == "first"
    # External refresh script writes new token and bumps mtime.
    f.write_text("second\n")
    os.utime(str(f), (2000, 2000))
    assert store.get(str(f)) == "second"


def test_invalidate(tmp_path):
    f = tmp_path / "tok"
    f.write_text("first")
    os.utime(str(f), (1000, 1000))
    store = AuthTokenStore()
    store.get(str(f))
    f.write_text("second")
    os.utime(str(f), (1000, 1000))  # same mtime
    # Without invalidation, we'd see the cached value.
    assert store.get(str(f)) == "first"
    store.invalidate(str(f))
    assert store.get(str(f)) == "second"


def test_empty_file_returns_none(tmp_path):
    f = tmp_path / "tok"
    f.write_text("\n   \n")
    store = AuthTokenStore()
    assert store.get(str(f)) is None
