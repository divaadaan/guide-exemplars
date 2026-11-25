"""Background job execution: claim queued jobs, run GUIDE, persist events/checkpoints/usage, publish exemplar sets.

Each worker slot is a thread that runs one job at a time in its own event loop. API keys passed with a job are
held in memory only (``KeyVault``). A job resumed after a restart therefore needs its key in the environment.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import traceback
from typing import Any

from guide_exemplars.core.checkpoint import Checkpoint
from guide_exemplars.core.events import Event
from guide_exemplars.core.optimizer import GuideConfig, GuideOptimizer
from guide_exemplars.core.task import TaskSpec
from guide_exemplars.core.types import Item
from guide_exemplars.llm.backbones import BackboneConfig
from guide_exemplars.llm.budget import UsageRecord
from guide_exemplars.runtime import build_runtime
from guide_exemplars.service.bus import Broadcaster
from guide_exemplars.service.settings import Settings
from guide_exemplars.service.store import Store

log = logging.getLogger(__name__)

TERMINAL_EVENT = {
    "succeeded": "job_succeeded",
    "cancelled": "job_cancelled",
    "budget_exhausted": "job_budget_exhausted",
    "failed": "job_failed",
}
STATUS_FROM_RESULT = {
    "completed": "succeeded",
    "cancelled": "cancelled",
    "budget_exhausted": "budget_exhausted",
}


class KeyVault:
    """Per-job API keys, in memory only. Never written to the database or logs."""

    def __init__(self) -> None:
        self._keys: dict[str, str] = {}
        self._lock = threading.Lock()

    def put(self, job_id: str, key: str | None) -> None:
        if key:
            with self._lock:
                self._keys[job_id] = key

    def get(self, job_id: str) -> str | None:
        with self._lock:
            return self._keys.get(job_id)

    def drop(self, job_id: str) -> None:
        with self._lock:
            self._keys.pop(job_id, None)


class CancelFlags:
    def __init__(self) -> None:
        self._set: set[str] = set()
        self._lock = threading.Lock()

    def request(self, job_id: str) -> None:
        with self._lock:
            self._set.add(job_id)

    def is_set(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._set

    def clear(self, job_id: str) -> None:
        with self._lock:
            self._set.discard(job_id)


class _JobCancel:
    def __init__(self, flags: CancelFlags, job_id: str):
        self.flags, self.job_id = flags, job_id

    def is_set(self) -> bool:
        return self.flags.is_set(self.job_id)


class _DBCheckpoints:
    def __init__(self, store: Store, job_id: str):
        self.store, self.job_id = store, job_id

    def save(self, cp: Checkpoint) -> None:
        self.store.save_checkpoint(self.job_id, cp.round, cp.phase, cp.to_json())

    def latest(self) -> Checkpoint | None:
        rows = self.store.checkpoints(self.job_id)
        if not rows:
            return None
        cps = [Checkpoint.from_json(r["payload"]) for r in rows]
        return max(cps, key=Checkpoint.sort_key)


class _JobSink:
    """Persist events, publish them live, mirror progress into the jobs row, and flush usage in batches."""

    def __init__(self, store: Store, bus: Broadcaster, job_id: str, rt):
        self.store, self.bus, self.job_id, self.rt = store, bus, job_id, rt
        self._usage: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_usage_event = 0.0
        rt.meter.subscribe(self._on_usage)

    def _on_usage(self, r: UsageRecord) -> None:
        with self._lock:
            self._usage.append(r.to_dict())
            due = len(self._usage) >= 50
        if due:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            rows, self._usage = self._usage, []
        self.store.add_usage(rows, self.job_id)

    def emit(self, type_: str, round_: int = 0, ephemeral: bool = False, **payload: Any) -> None:
        self(Event(type_, round_, payload, ephemeral))

    def __call__(self, event: Event) -> None:
        msg: dict[str, Any] = {"type": event.type, "round": event.round, "payload": event.payload}
        if not event.ephemeral:
            msg["seq"] = self.store.add_event(self.job_id, event.type, event.round, event.payload)
        self.bus.publish(self.job_id, msg)
        updates: dict[str, Any] = {}
        if event.type == "phase_started":
            updates = {"phase": event.payload.get("phase"), "round": event.round}
        elif event.type == "eval_completed":
            total = self.rt.meter.total
            updates = {
                "round": event.round,
                "bo_iter": int(event.payload.get("index", 0)),
                "best_acc": event.payload.get("best_acc_global"),
                "cost_usd": self.rt.budget.spent,
                "calls": total.calls,
                "cached_calls": total.cache_hits,
            }
            self.flush()
            now = time.time()
            if now - self._last_usage_event > 2:
                self._last_usage_event = now
                usage = {
                    "cost_usd": self.rt.budget.spent,
                    "calls": total.calls,
                    "cached_calls": total.cache_hits,
                }
                self.bus.publish(self.job_id, {"type": "usage", "round": event.round, "payload": usage})
                cap = self.rt.budget.max_usd
                if cap and self.rt.budget.spent >= 0.8 * cap:
                    self.bus.publish(
                        self.job_id, {"type": "budget_warning", "round": event.round, "payload": usage}
                    )
        if updates:
            updates["heartbeat_at"] = time.time()
            self.store.update_job(self.job_id, **updates)


def config_hash(cfg: GuideConfig) -> str:
    return hashlib.sha256(json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]


class JobRunner:
    def __init__(
        self, store: Store, bus: Broadcaster, settings: Settings, keys: KeyVault, cancels: CancelFlags
    ):
        self.store, self.bus, self.settings = store, bus, settings
        self.keys, self.cancels = keys, cancels
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads: list[threading.Thread] = []

    # ---- lifecycle ------------------------------------------------------------------------------

    def recover(self) -> list[str]:
        """Called at startup: running jobs without a live heartbeat become interrupted (optionally requeued)."""
        # A single-process service owns every job it ran, so at startup any "running" job is dead.
        interrupted = self.store.mark_stale_running(stale_after_s=0.0)
        for job_id in interrupted:
            self.store.add_event(job_id, "job_interrupted", 0, {"reason": "service restarted"})
            if self.settings.auto_resume:
                self.store.update_job(job_id, status="queued", resume_from="latest")
        return interrupted

    def start(self) -> None:
        self.recover()
        for slot in range(max(1, self.settings.worker_slots)):
            t = threading.Thread(
                target=self._loop, name=f"guide-worker-{slot}", args=(f"w{slot}",), daemon=True
            )
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        for t in self._threads:
            t.join(timeout)

    def notify(self) -> None:
        self._wake.set()

    def _loop(self, worker_id: str) -> None:
        while not self._stop.is_set():
            job = self.store.claim_job(worker_id)
            if job is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            self.execute(job)

    # ---- one job --------------------------------------------------------------------------------

    def _heartbeat(self, job_id: str, done: threading.Event) -> None:
        while not done.wait(self.settings.heartbeat_s):
            self.store.update_job(job_id, heartbeat_at=time.time())

    def execute(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        done = threading.Event()
        hb = threading.Thread(target=self._heartbeat, args=(job_id, done), daemon=True)
        hb.start()
        sink: _JobSink | None = None
        try:
            task = self.store.get_task(job["task_id"])
            rows = self.store.dataset_items(job["dataset_id"])
            spec = TaskSpec.model_validate(json.loads(task["spec_json"]))
            items = [Item(item_id=r["item_id"], text=r["text"], label=r["label"]) for r in rows]
            split = {r["item_id"]: r["split"] for r in rows}
            by = lambda s: [it for it in items if split[it.item_id] == s]  # noqa: E731
            backbone = BackboneConfig.model_validate(json.loads(job["backbone_json"]))
            cfg = GuideConfig.model_validate({**json.loads(job["config_json"]), "seed": job["seed"]})
            rt = build_runtime(
                backbone,
                max_cost_usd=job["max_cost_usd"],
                spent_usd=self.store.job_spend(job_id),
                api_key=self.keys.get(job_id),
                gold_items=items,
                scale=spec.scale,
            )
            sink = _JobSink(self.store, self.bus, job_id, rt)
            checkpoints = _DBCheckpoints(self.store, job_id)
            resume = checkpoints.latest() if job.get("resume_from") else None
            sink.emit("job_started", resume=bool(resume), backbone=backbone.name)
            opt = GuideOptimizer(
                spec,
                by("train"),
                by("val"),
                rt.llm,
                rt.embedder,
                cfg,
                experts=by("expert"),
                events=sink,
                checkpoints=checkpoints,
                cancel=_JobCancel(self.cancels, job_id),
            )
            result = asyncio.run(opt.run(resume=resume))
            sink.flush()
            status = STATUS_FROM_RESULT[result.status]
            set_id = None
            if result.exemplars:
                best = result.best
                exemplar_set = self.store.add_exemplar_set(
                    task_id=job["task_id"],
                    name=job["set_name"],
                    job_id=job_id,
                    status="final" if status == "succeeded" else "partial",
                    exemplars=[e.model_dump(mode="json") for e in result.exemplars],
                    val_metrics=best.metrics.to_dict() if best else None,
                    backbone_name=backbone.name,
                    config_hash=config_hash(cfg),
                )
                set_id = exemplar_set["id"]
            total = rt.meter.total
            self.store.update_job(
                job_id,
                status=status,
                finished_at=time.time(),
                exemplar_set_id=set_id,
                cost_usd=rt.budget.spent,
                calls=total.calls,
                cached_calls=total.cache_hits,
                best_acc=result.best.metrics.acc if result.best else None,
            )
            sink.emit(TERMINAL_EVENT[status], exemplar_set_id=set_id, notes=result.notes)
        except Exception as exc:  # job-level failure: record it, keep the worker alive
            log.exception("job %s failed", job_id)
            if sink is not None:
                sink.flush()
            error = f"{type(exc).__name__}: {exc}"
            self.store.update_job(job_id, status="failed", finished_at=time.time(), error=error)
            payload = {"error": error, "trace": traceback.format_exc(limit=5)}
            seq = self.store.add_event(job_id, "job_failed", 0, payload)
            self.bus.publish(job_id, {"type": "job_failed", "round": 0, "payload": payload, "seq": seq})
        finally:
            done.set()
            self.cancels.clear(job_id)
            self.keys.drop(job_id)
