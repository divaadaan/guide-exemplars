"""SQLite persistence for the service (stdlib ``sqlite3``, WAL mode).

Every method opens a short-lived connection, so the store is safe to share between the API's event loop and
worker threads. The schema is small and explicit. Moving to Postgres means reimplementing this one class
behind the same methods.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL, spec_json TEXT NOT NULL,
    spec_hash TEXT NOT NULL, created_at REAL NOT NULL, UNIQUE (name, version));
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id), name TEXT NOT NULL,
    content_hash TEXT NOT NULL, split_seed INTEGER NOT NULL, split_ratios TEXT NOT NULL,
    label_counts_json TEXT NOT NULL, split_counts_json TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS items (
    dataset_id TEXT NOT NULL REFERENCES datasets(id), item_id TEXT NOT NULL, text TEXT NOT NULL,
    label TEXT NOT NULL, split TEXT NOT NULL CHECK (split IN ('train','val','test','expert')),
    PRIMARY KEY (dataset_id, item_id));
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, dataset_id TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'optimize',
    config_json TEXT NOT NULL, backbone_name TEXT NOT NULL, backbone_json TEXT NOT NULL, seed INTEGER NOT NULL,
    set_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued','running','cancelling','cancelled','succeeded','failed',
                                           'budget_exhausted','interrupted')),
    phase TEXT, round INTEGER NOT NULL DEFAULT 0, bo_iter INTEGER NOT NULL DEFAULT 0, best_acc REAL,
    max_cost_usd REAL, cost_usd REAL NOT NULL DEFAULT 0, calls INTEGER NOT NULL DEFAULT 0,
    cached_calls INTEGER NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0,
    resume_from TEXT, worker_id TEXT, heartbeat_at REAL, created_at REAL NOT NULL, started_at REAL,
    finished_at REAL, error TEXT, exemplar_set_id TEXT);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS job_events (
    job_id TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL, round INTEGER NOT NULL,
    payload_json TEXT NOT NULL, ts REAL NOT NULL, PRIMARY KEY (job_id, seq));
CREATE TABLE IF NOT EXISTS checkpoints (
    job_id TEXT NOT NULL, round INTEGER NOT NULL, phase TEXT NOT NULL, payload_gz BLOB NOT NULL,
    created_at REAL NOT NULL, PRIMARY KEY (job_id, round, phase));
CREATE TABLE IF NOT EXISTS exemplar_sets (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL, job_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('final','partial')), exemplars_json TEXT NOT NULL,
    val_metrics_json TEXT, backbone_name TEXT NOT NULL, config_hash TEXT, created_at REAL NOT NULL,
    UNIQUE (task_id, name, version));
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, exemplar_set_id TEXT, purpose TEXT NOT NULL,
    model TEXT NOT NULL, prompt_tokens INTEGER NOT NULL, cached_prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL, cost_usd REAL NOT NULL, cache_hit INTEGER NOT NULL,
    latency_ms REAL NOT NULL, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS usage_job ON usage(job_id);
"""

TERMINAL = {"cancelled", "succeeded", "failed", "budget_exhausted"}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    # ---- tasks ----------------------------------------------------------------------------------

    def add_task(self, name: str, version: int, spec: dict[str, Any], spec_hash: str) -> dict[str, Any]:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE name = ? AND version = ?", (name, version)
            ).fetchone()
            if row is not None:
                if row["spec_hash"] != spec_hash:
                    raise ValueError(f"task {name} v{version} already exists with different content")
                return dict(row)
            tid = new_id("task")
            conn.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?)",
                (tid, name, version, json.dumps(spec), spec_hash, time.time()),
            )
            return dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def list_tasks(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM tasks ORDER BY created_at DESC")]

    # ---- datasets -------------------------------------------------------------------------------

    def add_dataset(
        self,
        task_id: str,
        name: str,
        content_hash: str,
        split_seed: int,
        split_ratios: Sequence[float],
        rows: Iterable[tuple[str, str, str, str]],  # item_id, text, label, split
    ) -> dict[str, Any]:
        rows = list(rows)
        labels: dict[str, int] = {}
        splits: dict[str, int] = {}
        for _, _, label, split in rows:
            labels[label] = labels.get(label, 0) + 1
            splits[split] = splits.get(split, 0) + 1
        did = new_id("ds")
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    did,
                    task_id,
                    name,
                    content_hash,
                    split_seed,
                    json.dumps(list(split_ratios)),
                    json.dumps(labels),
                    json.dumps(splits),
                    time.time(),
                ),
            )
            conn.executemany("INSERT INTO items VALUES (?, ?, ?, ?, ?)", [(did, *r) for r in rows])
        return self.get_dataset(did)  # type: ignore[return-value]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        return dict(row) if row else None

    def list_datasets(self, task_id: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM datasets", ()
        if task_id:
            q, args = q + " WHERE task_id = ?", (task_id,)
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(q + " ORDER BY created_at DESC", args)]

    def dataset_items(self, dataset_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT item_id, text, label, split FROM items WHERE dataset_id = ? ORDER BY rowid",
                    (dataset_id,),
                )
            ]

    # ---- jobs -----------------------------------------------------------------------------------

    def add_job(self, **fields: Any) -> dict[str, Any]:
        jid = new_id("job")
        fields = {"id": jid, "status": "queued", "created_at": time.time(), **fields}
        cols = ", ".join(fields)
        marks = ", ".join("?" * len(fields))
        with self.transaction() as conn:
            conn.execute(f"INSERT INTO jobs ({cols}) VALUES ({marks})", tuple(fields.values()))
        return self.get_job(jid)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM jobs", ()
        if status:
            q, args = q + " WHERE status = ?", (status,)
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(q + " ORDER BY created_at DESC LIMIT ?", (*args, limit))]

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self.transaction() as conn:
            conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", (*fields.values(), job_id))

    def transition_job(self, job_id: str, from_statuses: Sequence[str], **fields: Any) -> bool:
        """Compare-and-set: apply ``fields`` only if the job is currently in one of ``from_statuses``."""
        sets = ", ".join(f"{k} = ?" for k in fields)
        marks = ",".join("?" * len(from_statuses))
        with self.transaction() as conn:
            cur = conn.execute(
                f"UPDATE jobs SET {sets} WHERE id = ? AND status IN ({marks})",
                (*fields.values(), job_id, *from_statuses),
            )
        return cur.rowcount == 1

    def claim_job(self, worker_id: str) -> dict[str, Any] | None:
        """Atomically move the oldest queued job to running."""
        now = time.time()
        with self.transaction() as conn:
            row = conn.execute(
                "UPDATE jobs SET status = 'running', worker_id = ?, heartbeat_at = ?, "
                "started_at = COALESCE(started_at, ?) "
                "WHERE id = (SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1) "
                "RETURNING *",
                (worker_id, now, now),
            ).fetchone()
        return dict(row) if row else None

    def mark_stale_running(self, stale_after_s: float) -> list[str]:
        cutoff = time.time() - stale_after_s
        with self.transaction() as conn:
            rows = conn.execute(
                "UPDATE jobs SET status = 'interrupted' WHERE status IN ('running', 'cancelling') "
                "AND (heartbeat_at IS NULL OR heartbeat_at < ?) RETURNING id",
                (cutoff,),
            ).fetchall()
        return [r["id"] for r in rows]

    # ---- events ---------------------------------------------------------------------------------

    def add_event(self, job_id: str, type_: str, round_: int, payload: dict[str, Any]) -> int:
        with self.transaction() as conn:
            seq = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM job_events WHERE job_id = ?", (job_id,)
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO job_events VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, seq, type_, round_, json.dumps(payload, default=str), time.time()),
            )
        return int(seq)

    def events(self, job_id: str, after: int = 0, types: Sequence[str] | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM job_events WHERE job_id = ? AND seq > ?"
        args: list[Any] = [job_id, after]
        if types:
            q += f" AND type IN ({','.join('?' * len(types))})"
            args += list(types)
        with self.connect() as conn:
            rows = conn.execute(q + " ORDER BY seq", args).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload_json"])} for r in rows]

    # ---- checkpoints ----------------------------------------------------------------------------

    def save_checkpoint(self, job_id: str, round_: int, phase: str, payload: dict[str, Any]) -> None:
        blob = gzip.compress(json.dumps(payload).encode("utf-8"))
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?)",
                (job_id, round_, phase, blob, time.time()),
            )

    def checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT round, phase, payload_gz FROM checkpoints WHERE job_id = ? ORDER BY round, phase",
                (job_id,),
            ).fetchall()
        return [
            {
                "round": r["round"],
                "phase": r["phase"],
                "payload": json.loads(gzip.decompress(r["payload_gz"])),
            }
            for r in rows
        ]

    # ---- exemplar sets --------------------------------------------------------------------------

    def add_exemplar_set(
        self,
        task_id: str,
        name: str,
        job_id: str | None,
        status: str,
        exemplars: list[dict[str, Any]],
        val_metrics: dict[str, Any] | None,
        backbone_name: str,
        config_hash: str | None,
    ) -> dict[str, Any]:
        sid = new_id("set")
        with self.transaction() as conn:
            version = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM exemplar_sets WHERE task_id = ? AND name = ?",
                (task_id, name),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO exemplar_sets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    task_id,
                    name,
                    version,
                    job_id,
                    status,
                    json.dumps(exemplars),
                    json.dumps(val_metrics) if val_metrics is not None else None,
                    backbone_name,
                    config_hash,
                    time.time(),
                ),
            )
        return self.get_exemplar_set(sid)  # type: ignore[return-value]

    @staticmethod
    def _set_row(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["exemplars"] = json.loads(d.pop("exemplars_json"))
        vm = d.pop("val_metrics_json")
        d["val_metrics"] = json.loads(vm) if vm else None
        return d

    def get_exemplar_set(self, set_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM exemplar_sets WHERE id = ?", (set_id,)).fetchone()
        return self._set_row(row) if row else None

    def find_exemplar_set(self, task_id: str, name: str, version: int | None) -> dict[str, Any] | None:
        q = "SELECT * FROM exemplar_sets WHERE task_id = ? AND name = ?"
        args: tuple[Any, ...] = (task_id, name)
        q += " AND version = ?" if version is not None else " ORDER BY version DESC LIMIT 1"
        if version is not None:
            args += (version,)
        with self.connect() as conn:
            row = conn.execute(q, args).fetchone()
        return self._set_row(row) if row else None

    def list_exemplar_sets(self, task_id: str | None = None, name: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM exemplar_sets WHERE 1 = 1", []
        if task_id:
            q, args = q + " AND task_id = ?", [*args, task_id]
        if name:
            q, args = q + " AND name = ?", [*args, name]
        with self.connect() as conn:
            return [self._set_row(r) for r in conn.execute(q + " ORDER BY created_at DESC", args)]

    # ---- usage ----------------------------------------------------------------------------------

    def add_usage(
        self, rows: Sequence[dict[str, Any]], job_id: str | None, set_id: str | None = None
    ) -> None:
        if not rows:
            return
        with self.transaction() as conn:
            conn.executemany(
                "INSERT INTO usage (job_id, exemplar_set_id, purpose, model, prompt_tokens, cached_prompt_tokens,"
                " completion_tokens, cost_usd, cache_hit, latency_ms, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        job_id,
                        set_id,
                        r["purpose"],
                        r["model"],
                        r["prompt_tokens"],
                        r["cached_prompt_tokens"],
                        r["completion_tokens"],
                        r["cost_usd"],
                        int(r["cache_hit"]),
                        r["latency_ms"],
                        r["ts"],
                    )
                    for r in rows
                ],
            )

    def job_spend(self, job_id: str) -> float:
        with self.connect() as conn:
            return float(
                conn.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM usage WHERE job_id = ?", (job_id,)
                ).fetchone()[0]
            )

    def ping(self) -> bool:
        with self.connect() as conn:
            return conn.execute("SELECT 1").fetchone()[0] == 1
