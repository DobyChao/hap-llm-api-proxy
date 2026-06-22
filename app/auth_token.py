"""Read auth tokens from files with mtime-based caching.

Tokens are written by an external refresh script, so we must re-read when the
file changes. We stat the file on every call; if mtime hasn't moved we return
the cached value, otherwise we re-read.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional


class AuthTokenStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # path -> (mtime, token)
        self._cache: dict[str, tuple[float, str]] = {}

    def _read_mtime(self, path: str) -> float:
        try:
            return os.stat(path).st_mtime
        except FileNotFoundError:
            return -1.0

    def get(self, path: Optional[str]) -> Optional[str]:
        """Return the current token for ``path``.

        Returns ``None`` when ``path`` is falsy, the file is missing, or the
        file content is empty. The token is stripped of surrounding whitespace
        (refresh scripts commonly append a trailing newline).
        """
        if not path:
            return None
        mtime = self._read_mtime(path)
        with self._lock:
            cached = self._cache.get(path)
            if cached is not None and cached[0] == mtime:
                return cached[1] or None
            token = ""
            if mtime >= 0:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        token = f.read().strip()
                except FileNotFoundError:
                    token = ""
                except OSError:
                    token = ""
            self._cache[path] = (mtime, token)
            return token or None

    def invalidate(self, path: Optional[str] = None) -> None:
        with self._lock:
            if path is None:
                self._cache.clear()
            else:
                self._cache.pop(path, None)
