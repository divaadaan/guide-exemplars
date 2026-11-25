"""Benchmark runner: for each (task, seed), build the Phase-0 pool once, run GUIDE (and its ablations), fit
the baselines on the same pool, and score every policy on the held-out test split with the same grader.

Layout::

    results/<matrix>/manifest.json
    results/<matrix>/<task>/seed<k>/phase0_pool.json
    results/<matrix>/<task>/seed<k>/<method>/{predictions.jsonl, metrics.json, exemplars.json, ...}
    results/<matrix>/summary.{csv,md}, deltas.csv

A method directory that already has ``metrics.json`` is skipped, so an interrupted matrix picks up where it
stopped. In ``replay`` mode every LLM call must hit the cache. That proves the results reproduce from the
recorded responses.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from guide_exemplars import __version__
from guide_exemplars.benchmark.bootstrap import bootstrap
from guide_exemplars.benchmark.matrix import Matrix
from guide_exemplars.core.events import Event
from guide_exemplars.core.grader import Grader
from guide_exemplars.core.optimizer import GuideConfig, GuideOptimizer
from guide_exemplars.core.pool import Pool
from guide_exemplars.core.prompts import PromptBuilder
from guide_exemplars.core.rng import RngStreams
from guide_exemplars.core.selectors.base import ExemplarPolicy, SelectionContext, StaticPolicy
from guide_exemplars.core.selectors.baselines import BASELINES
from guide_exemplars.core.types import Exemplar, Prediction
from guide_exemplars.data.presets import PreparedTask, Preset, load_prepared
from guide_exemplars.llm.backbones import build_embedder, default_embedding_store, load_backbones
from guide_exemplars.llm.budget import Budget, UsageMeter
from guide_exemplars.paths import REPO_ROOT, RESULTS_DIR
from guide_exemplars.runtime import Runtime, build_runtime, run_optimize

Log = Callable[[str], None]


@dataclass
class MethodResult:
    task: str
    seed: int
    method: str
    gold: list[int]
    preds: list[list[int | None]]  # one vector per draw
    metrics: dict[str, Any]
    k: int
    extra: dict[str, Any] = field(default_factory=dict)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def estimate_matrix(matrix: Matrix) -> list[dict[str, Any]]:
    """Per-task cost estimate for one seed of the matrix (multiply by len(seeds) for the total).

    The GUIDE run pays for Phase 0. Ablations reuse that pool. Baselines only score the test split.
    """
    from guide_exemplars.runtime import estimate_for

    backbone = load_backbones()[matrix.backbone]
    preset = Preset.load(matrix.preset)
    rows = []
    for task_id in matrix.tasks or [t.id for t in preset.tasks]:
        prepared = load_prepared(preset, task_id)
        lim = matrix.limits
        if lim.train or lim.val or lim.test:
            prepared = prepared.limited(lim.train, lim.val, lim.test)
        parts = []
        base_cfg = GuideConfig.model_validate(matrix.guide)
        if "guide" in matrix.methods:
            parts.append(estimate_for(prepared, backbone, base_cfg))
        for i, overrides in enumerate(matrix.ablations.values()):
            cfg = GuideConfig.model_validate({**matrix.guide, **overrides})
            include_phase0 = i == 0 and "guide" not in matrix.methods
            parts.append(estimate_for(prepared, backbone, cfg, include_phase0=include_phase0))
        n_static = ("guide" in matrix.methods) + len(matrix.ablations)
        n_static += ("naive" in matrix.methods) + ("vote_k" in matrix.methods)
        n_static += matrix.random_draws if "random" in matrix.methods else 0
        parts.append(
            estimate_for(
                prepared,
                backbone,
                base_cfg,
                n_test_static=n_static,
                n_test_knn=int("knn" in matrix.methods),
                include_optimization=False,
            )
        )
        rows.append(
            {
                "task": task_id,
                "calls": sum(p.total_calls for p in parts),
                "usd_low": sum(p.usd_low for p in parts),
                "usd_high": sum(p.usd_high for p in parts),
                "usd_hard_max": sum(p.usd_hard_max for p in parts),
                "minutes": sum(p.est_minutes for p in parts),
            }
        )
    return rows


class BenchmarkRunner:
    def __init__(
        self,
        matrix: Matrix,
        out_dir: Path | None = None,
        *,
        replay: bool = False,
        log: Log = print,
    ):
        self.m = matrix
        self.out = out_dir or RESULTS_DIR / matrix.name
        self.replay = replay
        self.log = log
        self.backbone = load_backbones()[matrix.backbone]
        self.budget = Budget(matrix.max_cost_usd)
        self.meter = UsageMeter()
        self.preset = Preset.load(matrix.preset)
        self.results: list[MethodResult] = []

    # ---- helpers --------------------------------------------------------------------------------

    def _runtime(self, prepared: PreparedTask) -> Runtime:
        return build_runtime(
            self.backbone,
            gold_items=prepared.items,
            scale=prepared.spec.scale,
            embedder_spec=self.m.embedder,
            budget=self.budget,
            meter=self.meter,
            cache_only=self.replay,
        )

    def _prepared(self, task_id: str) -> PreparedTask:
        prepared = load_prepared(self.preset, task_id)
        lim = self.m.limits
        if lim.train or lim.val or lim.test:
            prepared = prepared.limited(lim.train, lim.val, lim.test)
        return prepared

    def _guide_cfg(self, seed: int, overrides: dict[str, Any] | None = None) -> GuideConfig:
        return GuideConfig.model_validate({**self.m.guide, **(overrides or {}), "seed": seed})

    async def _phase0(self, prepared: PreparedTask, rt: Runtime, seed: int, run_dir: Path) -> Pool:
        path = run_dir / "phase0_pool.json"
        if path.exists():
            return Pool.from_records(json.loads(path.read_text(encoding="utf-8")))
        opt = GuideOptimizer(
            prepared.spec,
            prepared.train,
            prepared.val,
            rt.llm,
            rt.embedder,
            self._guide_cfg(seed),
            experts=prepared.experts,
        )
        pool = await opt.build_pool()
        _write_json(path, pool.to_records())
        return pool

    def _progress(self, prefix: str) -> Callable[[Event], None]:
        """One line per evaluated set: accuracy, failed calls, best so far, spend and elapsed time."""
        t0 = time.monotonic()

        def on_event(e: Event) -> None:
            if e.type != "eval_completed":
                return
            p = e.payload
            mins = (time.monotonic() - t0) / 60
            self.log(
                f"{prefix} r{e.round} #{p.get('index', 0):>2} {p.get('op', ''):<7} |E|={p.get('size', 0):>2} "
                f"acc {p.get('acc', 0):.3f} failed {p.get('n_failed', 0):>3} | best {p.get('best_acc_global', 0):.3f}"
                f" | ${self.meter.total.cost_usd:.2f} {mins:.0f} min"
            )

        return on_event

    async def _score(self, prepared: PreparedTask, rt: Runtime, policy: ExemplarPolicy) -> list[Prediction]:
        grader = Grader(PromptBuilder(prepared.spec), rt.llm, self._guide_cfg(0).grading_max_tokens)
        test = prepared.test
        await policy.prepare(test)
        demos = policy.exemplars if isinstance(policy, StaticPolicy) else policy.demos_for
        return await grader.grade(test, demos, purpose="grade")

    def _record(
        self,
        prepared: PreparedTask,
        seed: int,
        method: str,
        draws: list[list[Prediction]],
        method_dir: Path,
        exemplars: list[list[Exemplar]] | None,
        k: int,
        extra: dict[str, Any] | None = None,
    ) -> MethodResult:
        scale = prepared.spec.scale
        gold = [scale.rank(p.gold) for p in draws[0]]
        preds = [[None if p.pred is None else scale.rank(p.pred) for p in d] for d in draws]
        bs = self.m.bootstrap
        metrics = bootstrap(gold, preds, len(scale), bs.B, bs.ci, bs.seed)
        metrics["n"] = len(gold)
        metrics["n_failed"] = int(np.mean([sum(p is None for p in d) for d in preds]))
        method_dir.mkdir(parents=True, exist_ok=True)
        with (method_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
            for draw_i, d in enumerate(draws):
                for p in d:
                    row = {
                        "draw": draw_i,
                        "item_id": p.item_id,
                        "gold": p.gold,
                        "pred": p.pred,
                        "error": p.error,
                    }
                    fh.write(json.dumps(row) + "\n")
        if exemplars is not None:
            _write_json(
                method_dir / "exemplars.json", [[e.model_dump(mode="json") for e in s] for s in exemplars]
            )
        _write_json(method_dir / "metrics.json", {"metrics": metrics, "k": k, **(extra or {})})
        result = MethodResult(prepared.task_id, seed, method, gold, preds, metrics, k, extra or {})
        self.results.append(result)
        return result

    def _load_recorded(
        self, prepared: PreparedTask, seed: int, method: str, method_dir: Path
    ) -> MethodResult:
        scale = prepared.spec.scale
        rows = [
            json.loads(line) for line in (method_dir / "predictions.jsonl").read_text("utf-8").splitlines()
        ]
        n_draws = max(r["draw"] for r in rows) + 1
        draws: list[list[int | None]] = [[] for _ in range(n_draws)]
        gold: list[int] = []
        for r in rows:
            draws[r["draw"]].append(None if r["pred"] is None else scale.rank(r["pred"]))
            if r["draw"] == 0:
                gold.append(scale.rank(r["gold"]))
        saved = json.loads((method_dir / "metrics.json").read_text("utf-8"))
        extra = {k: v for k, v in saved.items() if k not in ("metrics", "k")}
        result = MethodResult(
            prepared.task_id, seed, method, gold, draws, saved["metrics"], saved["k"], extra
        )
        self.results.append(result)
        return result

    # ---- main -----------------------------------------------------------------------------------

    def rebuild_reports(self) -> list[MethodResult]:
        """Re-render summary/deltas from recorded predictions and the existing manifest, without any LLM call.
        The manifest (spend, git SHA, timings) is kept as recorded."""
        from guide_exemplars.benchmark.report import write_reports

        manifest_path = self.out / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} not found; run the benchmark first")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        methods = (["guide"] if "guide" in self.m.methods else []) + [f"guide-{a}" for a in self.m.ablations]
        methods += [m for m in self.m.methods if m != "guide"]
        for task_id in self.m.tasks or [t.id for t in self.preset.tasks]:
            prepared = self._prepared(task_id)
            for seed in self.m.seeds:
                for method in methods:
                    method_dir = self.out / task_id / f"seed{seed}" / method
                    if (method_dir / "metrics.json").exists():
                        self._load_recorded(prepared, seed, method, method_dir)
        write_reports(self.results, self.m, self.out, manifest)
        return self.results

    async def run(self) -> list[MethodResult]:
        started = time.time()
        task_ids = self.m.tasks or [t.id for t in self.preset.tasks]
        manifest: dict[str, Any] = {
            "matrix": self.m.model_dump(mode="json"),
            "matrix_hash": self.m.config_hash(),
            "package_version": __version__,
            "python": platform.python_version(),
            "git": _git_state(),
            "backbone": self.backbone.redacted(),
            "replay": self.replay,
            "started_at": started,
            "tasks": {},
        }
        for task_id in task_ids:
            prepared = self._prepared(task_id)
            manifest["tasks"][task_id] = {
                "split_hash": prepared.report.get("split_hash"),
                "raw_sha256": prepared.report.get("raw_sha256"),
                "n_train": len(prepared.train),
                "n_val": len(prepared.val),
                "n_test": len(prepared.test),
                "n_experts": len(prepared.experts),
                "human": prepared.report.get("human_rater1_vs_rater2"),
            }
            for seed in self.m.seeds:
                await self._run_task_seed(prepared, seed)
        manifest["finished_at"] = time.time()
        manifest["usage"] = self.meter.total.to_dict()
        manifest["usage_by_purpose"] = {k: v.to_dict() for k, v in self.meter.by_purpose.items()}
        manifest["spent_usd"] = self.budget.spent
        _write_json(self.out / "manifest.json", manifest)
        from guide_exemplars.benchmark.report import write_reports

        write_reports(self.results, self.m, self.out, manifest)
        return self.results

    async def _run_task_seed(self, prepared: PreparedTask, seed: int) -> None:
        run_dir = self.out / prepared.task_id / f"seed{seed}"
        rt = self._runtime(prepared)
        self.log(f"[{prepared.task_id} seed {seed}] phase 0 ...")
        pool = await self._phase0(prepared, rt, seed, run_dir)
        k = self.m.k_default

        guide_runs: list[tuple[str, dict[str, Any]]] = []
        if "guide" in self.m.methods:
            guide_runs.append(("guide", {}))
        guide_runs += [(f"guide-{name}", ov) for name, ov in self.m.ablations.items()]
        for method, overrides in guide_runs:
            method_dir = run_dir / method
            if (method_dir / "metrics.json").exists():
                res = self._load_recorded(prepared, seed, method, method_dir)
                k = res.k if method == "guide" else k
                self.log(f"[{prepared.task_id} seed {seed}] {method}: recorded, skipped")
                continue
            self.log(f"[{prepared.task_id} seed {seed}] {method}: optimizing ...")
            outcome = await run_optimize(
                prepared,
                rt,
                self._guide_cfg(seed, overrides),
                method_dir / "optimize",
                phase0_pool=pool,
                events=self._progress(f"[{prepared.task_id} seed {seed}] {method}"),
            )
            result = outcome.result
            if not result.exemplars:
                self.log(f"[{prepared.task_id} seed {seed}] {method}: no result ({result.status}); skipped")
                continue
            if method == "guide":
                k = len(result.exemplars)
            policy = StaticPolicy(method, tuple(result.exemplars))
            preds = await self._score(prepared, rt, policy)
            best = result.best
            extra = {
                "status": result.status,
                "val_acc": best.metrics.acc if best else None,
                "best_round": best.round if best else None,
                "contrastive": best.contrastive if best else None,
            }
            r = self._record(
                prepared, seed, method, [preds], method_dir, [result.exemplars], len(result.exemplars), extra
            )
            self.log(
                f"[{prepared.task_id} seed {seed}] {method}: test acc {r.metrics['acc']['point']:.3f} (k={r.k})"
            )

        rng = RngStreams(seed)
        knn_embedder = build_embedder(
            self.m.knn_embedder, store=default_embedding_store(), meter=self.meter, budget=self.budget
        )
        for method in [m for m in self.m.methods if m != "guide"]:
            method_dir = run_dir / method
            if (method_dir / "metrics.json").exists():
                self._load_recorded(prepared, seed, method, method_dir)
                self.log(f"[{prepared.task_id} seed {seed}] {method}: recorded, skipped")
                continue
            draws = self.m.random_draws if method == "random" else 1
            all_preds: list[list[Prediction]] = []
            sets: list[list[Exemplar]] = []
            for d in range(draws):
                ctx = SelectionContext(
                    pool=pool, k=k, rng=rng.get("baseline", method, d), embedder=knn_embedder
                )
                policy = await BASELINES[method]().fit(ctx)
                all_preds.append(await self._score(prepared, rt, policy))
                if isinstance(policy, StaticPolicy):
                    sets.append(list(policy.exemplars))
            r = self._record(prepared, seed, method, all_preds, method_dir, sets or None, k, {"draws": draws})
            self.log(
                f"[{prepared.task_id} seed {seed}] {method}: test acc {r.metrics['acc']['point']:.3f} (k={k})"
            )
