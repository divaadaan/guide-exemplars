"""Persistent caches for LLM responses and embeddings (SQLite, WAL mode), plus in-memory equivalents.

Responses are keyed by a hash of everything that can change the output: model, endpoint, temperature, token
limit, JSON mode, extra request body and the messages. A rerun of an identical experiment is therefore free and
reproduces the same outputs byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np


def response_cache_key(**parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ResponseCache(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...

    def put(self, key: str, model: str, value: dict[str, Any]) -> None: ...


class EmbeddingStore(Protocol):
    def get_many(self, model: str, shas: Iterable[str]) -> dict[str, np.ndarray]: ...

    def put_many(self, model: str, vectors: Mapping[str, np.ndarray]) -> None: ...


class MemoryResponseCache:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self._data.get(key)

    def put(self, key: str, model: str, value: dict[str, Any]) -> None:
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)


class NullResponseCache:
    """Never hits. Used by connectivity checks, which must reach the provider."""

    def get(self, key: str) -> dict[str, Any] | None:
        return None

    def put(self, key: str, model: str, value: dict[str, Any]) -> None:
        return None


class MemoryEmbeddingStore:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], np.ndarray] = {}

    def get_many(self, model: str, shas: Iterable[str]) -> dict[str, np.ndarray]:
        return {s: self._data[(model, s)] for s in shas if (model, s) in self._data}

    def put_many(self, model: str, vectors: Mapping[str, np.ndarray]) -> None:
        for s, v in vectors.items():
            self._data[(model, s)] = np.asarray(v, dtype=np.float32)


class _SQLite:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class SQLiteResponseCache(_SQLite):
    def __init__(self, path: str | Path):
        super().__init__(path)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                " key TEXT PRIMARY KEY, model TEXT NOT NULL, response_json TEXT NOT NULL,"
                " created_at REAL NOT NULL)"
            )
            self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT response_json FROM llm_cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, model, response_json, created_at) VALUES (?, ?, ?, ?)",
                (key, model, json.dumps(value, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])


class SQLiteEmbeddingStore(_SQLite):
    def __init__(self, path: str | Path):
        super().__init__(path)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS embed_cache ("
                " model TEXT NOT NULL, text_sha TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL,"
                " PRIMARY KEY (model, text_sha))"
            )
            self._conn.commit()

    def get_many(self, model: str, shas: Iterable[str]) -> dict[str, np.ndarray]:
        shas = list(dict.fromkeys(shas))
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for start in range(0, len(shas), 500):
                chunk = shas[start : start + 500]
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT text_sha, vec FROM embed_cache WHERE model = ? AND text_sha IN ({marks})",
                    (model, *chunk),
                ).fetchall()
                for sha, blob in rows:
                    out[sha] = np.frombuffer(blob, dtype=np.float32).copy()
        return out

    def put_many(self, model: str, vectors: Mapping[str, np.ndarray]) -> None:
        rows = []
        for sha, v in vectors.items():
            arr = np.asarray(v, dtype=np.float32)
            rows.append((model, sha, int(arr.shape[0]), arr.tobytes()))
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embed_cache (model, text_sha, dim, vec) VALUES (?, ?, ?, ?)", rows
            )
            self._conn.commit()
