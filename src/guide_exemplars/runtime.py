"""Composition root: wires prepared data, a backbone, caches, budget and sinks into a GUIDE run.

The CLI, the service worker and the benchmark runner all go through these functions, so every entry point
behaves identically.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guide_exemplars.core.checkpoint import CheckpointStore, FileCheckpointStore, observation_to_dict
from guide_exemplars.core.estimate import Estimate, estimate_run
from guide_exemplars.core.events import Event, EventSink
from guide_exemplars.core.optimizer import GuideConfig, GuideOptimizer, GuideResult
from guide_exemplars.core.pool import Pool
from guide_exemplars.core.protocols import CancelToken, Embedder
from guide_exemplars.core.types import Item
from guide_exemplars.data.presets import PreparedTask
from guide_exemplars.llm.backbones import (
    BackboneConfig,
    build_embedder,
    build_llm,
    default_embedding_store,
    default_response_cache,
)
from guide_exemplars.llm.budget import Budget, UsageMeter
from guide_exemplars.llm.cache import EmbeddingStore, ResponseCache
from guide_exemplars.llm.client import LLMClient
from guide_exemplars.llm.fake import GradingSimulator
from guide_exemplars.llm.pricing import PriceTable
from guide_exemplars.llm.tokens import TokenCounter


@dataclass
class Runtime:
    backbone: BackboneConfig
    llm: LLMClient
    embedder: Embedder
    meter: UsageMeter
    budget: Budget


def build_runtime(
    backbone: BackboneConfig,
    *,
    max_cost_usd: float | None = None,
    spent_usd: float = 0.0,
    api_key: str | None = None,
    gold_items: Sequence[Item] = (),
    scale=None,
    cache: ResponseCache | None = None,
    store: EmbeddingStore | None = None,
    embedder_spec: str | None = None,
    budget: Budget | None = None,
    meter: UsageMeter | None = None,
    cache_only: bool = False,
) -> Runtime:
    """``gold_items``/``scale`` are only used by the fake backbone, whose simulator needs the gold labels.
    Pass ``budget``/``meter`` to share one spend cap and usage log across several runtimes."""
    meter = meter if meter is not None else UsageMeter()
    budget = budget if budget is not None else Budget(max_cost_usd, spent_usd)
    prices = PriceTable.load()
    responder = None
    if backbone.kind == "fake":
        if scale is None:
            raise ValueError("fake backbone needs the task scale")
        responder = GradingSimulator(scale, {it.text: it.label for it in gold_items})
    llm = build_llm(
        backbone,
        cache=cache if cache is not None else default_response_cache(),
        meter=meter,
        budget=budget,
        prices=prices,
        api_key=api_key,
        responder=responder,
        cache_only=cache_only,
    )
    embedder = build_embedder(
        embedder_spec or backbone.embedder,
        store=store if store is not None else default_embedding_store(),
        meter=meter,
        budget=budget,
        prices=prices,
        api_key=None,
        api_base=backbone.embedder_api_base,
        api_key_env=backbone.embedder_api_key_env,
    )
    return Runtime(backbone, llm, embedder, meter, budget)


class FileEventSink:
    """Appends persisted (non-ephemeral) events to a JSONL file and forwards every event to ``forward``."""

    def __init__(self, path: Path, forward: EventSink | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.forward = forward
        self._lock = threading.Lock()
        self._seq = self._last_seq()

    def _last_seq(self) -> int:
        if not self.path.exists():
            return 0
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return json.loads(lines[-1])["seq"] if lines else 0

    def __call__(self, event: Event) -> None:
        if not event.ephemeral:
            with self._lock:
                self._seq += 1
                row = {"seq": self._seq, "ts": time.time(), "type": event.type, "round": event.round}
                row["payload"] = event.payload
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
        if self.forward:
            self.forward(event)


def estimate_for(
    prepared: PreparedTask,
    backbone: BackboneConfig,
    cfg: GuideConfig,
    *,
    n_test_static: int = 0,
    n_test_knn: int = 0,
    include_optimization: bool = True,
    include_phase0: bool = True,
) -> Estimate:
    prices = PriceTable.load()
    price = prices.lookup(backbone.model)
    if backbone.self_hosted or price is None:
        from guide_exemplars.core.protocols import FREE

        price = FREE
    embed_price = prices.lookup(backbone.embedder) if backbone.embedder.startswith("openai/") else None
    return estimate_run(
        prepared.spec,
        prepared.train,
        prepared.val,
        cfg,
        TokenCounter().count,
        price,
        n_experts=len(prepared.experts),
        test=prepared.test,
        n_test_static_policies=n_test_static,
        n_test_knn_policies=n_test_knn,
        include_optimization=include_optimization,
        include_phase0=include_phase0,
        embed_price=embed_price,
        concurrency=backbone.max_concurrency,
        tpm_limit=backbone.tpm_limit if backbone.kind != "fake" else None,
    )


@dataclass
class OptimizeOutcome:
    result: GuideResult
    usage: dict[str, Any]
    out_dir: Path


def write_result(
    result: GuideResult, rt: Runtime, out_dir: Path, extra: dict[str, Any] | None = None
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exemplars.json").write_text(
        json.dumps([e.model_dump(mode="json") for e in result.exemplars], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (out_dir / "observations.jsonl").open("w", encoding="utf-8") as fh:
        for o in result.observations:
            fh.write(json.dumps(observation_to_dict(o)) + "\n")
    usage = {
        "total": rt.meter.total.to_dict(),
        "by_purpose": {k: v.to_dict() for k, v in rt.meter.by_purpose.items()},
        "budget_spent_usd": rt.budget.spent,
        "budget_cap_usd": rt.budget.max_usd,
    }
    (out_dir / "usage.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
    best = result.best
    summary = {
        "status": result.status,
        "notes": result.notes,
        "best": None
        if best is None
        else {
            "round": best.round,
            "index": best.index,
            "size": best.size,
            "contrastive": best.contrastive,
            "val_metrics": observation_to_dict(best)["metrics"],
            "uids": sorted(best.subset.uids),
        },
        "round_bests": {
            str(r): {"acc": o.metrics.acc, "size": o.size, "index": o.index}
            for r, o in result.round_bests.items()
        },
        "n_observations": len(result.observations),
        "config": result.config.model_dump(mode="json"),
        "backbone": rt.backbone.redacted(),
        **(extra or {}),
    }
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return usage


async def run_optimize(
    prepared: PreparedTask,
    rt: Runtime,
    cfg: GuideConfig,
    out_dir: Path,
    *,
    events: EventSink | None = None,
    checkpoints: CheckpointStore | None = None,
    cancel: CancelToken | None = None,
    resume: bool = False,
    phase0_pool: Pool | None = None,
    on_result: Callable[[GuideResult], None] | None = None,
) -> OptimizeOutcome:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "preset": prepared.preset,
                "task_id": prepared.task_id,
                "guide": cfg.model_dump(mode="json"),
                "backbone": rt.backbone.redacted(),
                "split_hash": prepared.report.get("split_hash"),
                "n_train": len(prepared.train),
                "n_val": len(prepared.val),
                "n_experts": len(prepared.experts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    store = checkpoints or FileCheckpointStore(out_dir / "checkpoints")
    sink = FileEventSink(out_dir / "events.jsonl", forward=events)
    opt = GuideOptimizer(
        prepared.spec,
        prepared.train,
        prepared.val,
        rt.llm,
        rt.embedder,
        cfg,
        experts=prepared.experts,
        events=sink,
        checkpoints=store,
        cancel=cancel,
    )
    latest = store.latest() if resume else None
    result = await opt.run(resume=latest, phase0_pool=phase0_pool)
    usage = write_result(result, rt, out_dir)
    if on_result:
        on_result(result)
    return OptimizeOutcome(result, usage, out_dir)
