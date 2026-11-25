"""Command-line entry point (`guide`)."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import typer

from guide_exemplars import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)
backbones_app = typer.Typer(no_args_is_help=True, help="Inspect and test configured LLM backbones.")
data_app = typer.Typer(no_args_is_help=True, help="Fetch and prepare benchmark datasets.")
app.add_typer(backbones_app, name="backbones")
app.add_typer(data_app, name="data")


def parse_overrides(pairs: list[str]) -> dict[str, object]:
    """``--set key=value`` pairs; values are parsed as YAML scalars (``4``, ``0.5``, ``true``, ``null``)."""
    import yaml

    out: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(f"expected key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def _guide_config(overrides: list[str], seed: int | None, config_file: Path | None):
    import yaml

    from guide_exemplars.core.optimizer import GuideConfig

    data: dict[str, object] = {}
    if config_file:
        data.update(yaml.safe_load(config_file.read_text(encoding="utf-8")) or {})
    data.update(parse_overrides(overrides))
    if seed is not None:
        data["seed"] = seed
    return GuideConfig.model_validate(data)


def _prepared(preset: str, task: str | None, train_limit: int | None, val_limit: int | None, test_limit=None):
    from guide_exemplars.data.presets import Preset, load_prepared

    prepared = load_prepared(Preset.load(preset), task)
    if train_limit or val_limit or test_limit:
        prepared = prepared.limited(train_limit, val_limit, test_limit)
    return prepared


def _backbone(name: str):
    from guide_exemplars.llm.backbones import load_backbones

    backbones = load_backbones()
    if name not in backbones:
        typer.echo(f"unknown backbone {name!r}; known: {', '.join(backbones)}", err=True)
        raise typer.Exit(2)
    cfg = backbones[name]
    missing = cfg.missing_settings()
    if missing:
        typer.echo(f"backbone {name} is not ready: {'; '.join(missing)}", err=True)
        raise typer.Exit(2)
    return cfg


def _split_summary(prepared) -> str:
    sizes = " ".join(f"{s}={len(prepared.split(s))}" for s in ("train", "val", "test", "expert"))
    return f"{prepared.task_id}: {sizes}"


def _print_estimate(est, cap: float | None = None) -> None:
    typer.echo(
        f"estimate: {est.total_calls} calls, ~{est.est_minutes:.0f} min | "
        f"${est.usd_low:.2f} (with prefix caching) - ${est.usd_high:.2f} (none) | worst case ${est.usd_hard_max:.2f}"
    )
    for name, g in est.groups.items():
        tokens = f"~{g.input_tokens_per_call:6.0f} in / {g.output_tokens_per_call:4.0f} out tokens"
        typer.echo(f"  {name:16s} {g.calls:7d} calls  {tokens}")
    if cap is not None and est.usd_high > cap:
        typer.echo(
            f"  note: uncached estimate exceeds the ${cap:.2f} cap; the run will stop cleanly at the cap"
        )


@data_app.command("fetch")
def data_fetch(name: str = typer.Argument(..., help="asap-aes | asap-sas")) -> None:
    """Download a raw dataset with the Kaggle CLI (needs Kaggle credentials and accepted competition rules)."""
    from guide_exemplars.data.fetch import fetch

    try:
        typer.echo(f"ready: {fetch(name)}")
    except (RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@data_app.command("prepare")
def data_prepare(preset: str = typer.Argument(..., help="preset name (configs/presets) or path")) -> None:
    """Split a raw dataset into prepared task data and print each task's report."""
    from guide_exemplars.data.presets import Preset, prepare

    try:
        dirs = prepare(Preset.load(preset))
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    for d in dirs:
        report = json.loads((d / "report.json").read_text(encoding="utf-8"))
        human = report.get("human_rater1_vs_rater2", {})
        typer.echo(
            f"{report['task_id']}: kept {report.get('n_kept')}/{report.get('n_raw')} | splits {report['split_counts']} "
            f"| labels {report.get('label_counts')} | human acc={human.get('acc', float('nan')):.3f} "
            f"qwk={human.get('qwk', float('nan')):.3f} | split_hash {report['split_hash']}"
        )


@app.command()
def estimate(
    preset: str = typer.Option(..., help="dataset preset"),
    task: str | None = typer.Option(None, help="task id within the preset"),
    backbone: str = typer.Option("gpt4o-mini"),
    set_: list[str] = typer.Option([], "--set", help="GuideConfig override key=value (repeatable)"),
    config: Path | None = typer.Option(None, help="YAML file of GuideConfig values"),
    train_limit: int | None = typer.Option(None),
    val_limit: int | None = typer.Option(None),
    test_static: int = typer.Option(0, help="static test-scoring policies to include"),
    test_knn: int = typer.Option(0, help="per-query (KNN) test-scoring policies to include"),
    no_optimize: bool = typer.Option(False, help="estimate test scoring only"),
) -> None:
    """Estimate calls, time and cost of a run without making any LLM calls."""
    from guide_exemplars.runtime import estimate_for

    cfg = _guide_config(set_, None, config)
    prepared = _prepared(preset, task, train_limit, val_limit)
    est = estimate_for(
        prepared,
        _backbone(backbone),
        cfg,
        n_test_static=test_static,
        n_test_knn=test_knn,
        include_optimization=not no_optimize,
    )
    typer.echo(_split_summary(prepared))
    _print_estimate(est)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="auto-reload on code changes (development)"),
) -> None:
    """Run the API and dashboard (http://HOST:PORT/, API docs at /docs)."""
    import uvicorn

    uvicorn.run("guide_exemplars.service.app:app_factory", factory=True, host=host, port=port, reload=reload)


@app.command()
def benchmark(
    matrix: str = typer.Argument(..., help="matrix name (configs/benchmarks) or YAML path"),
    out: Path | None = typer.Option(None, help="results directory (default results/<matrix name>)"),
    replay: bool = typer.Option(False, help="cache-only: fail on any LLM call that is not already cached"),
    dry_run: bool = typer.Option(False, help="print the cost estimate and exit"),
    update_readme: bool = typer.Option(
        False, help="inject summary.md into README.md between RESULTS markers"
    ),
    report_only: bool = typer.Option(
        False, help="re-render summary.md (and README with --update-readme) from recorded results; no LLM calls"
    ),
) -> None:
    """Run a benchmark matrix: GUIDE, ablations and baselines, scored on the held-out test split."""
    from guide_exemplars.benchmark.matrix import Matrix
    from guide_exemplars.benchmark.runner import BenchmarkRunner, estimate_matrix

    m = Matrix.load(matrix)
    if report_only:
        runner = BenchmarkRunner(m, out, log=typer.echo)
        runner.rebuild_reports()
        summary = (runner.out / "summary.md").read_text(encoding="utf-8")
        typer.echo(summary)
        if update_readme:
            from guide_exemplars.benchmark.report import inject_readme
            from guide_exemplars.paths import REPO_ROOT

            ok = inject_readme(REPO_ROOT / "README.md", [summary])
            typer.echo("README updated" if ok else "README has no RESULTS markers; not updated")
        return
    _backbone(m.backbone)
    rows = estimate_matrix(m)
    n_seeds = len(m.seeds)
    for r in rows:
        typer.echo(
            f"{r['task']}: {r['calls'] * n_seeds} calls, ~{r['minutes'] * n_seeds:.0f} min | "
            f"${r['usd_low'] * n_seeds:.2f} - ${r['usd_high'] * n_seeds:.2f} (worst ${r['usd_hard_max'] * n_seeds:.2f})"
        )
    total_low = sum(r["usd_low"] for r in rows) * n_seeds
    total_high = sum(r["usd_high"] for r in rows) * n_seeds
    cap = f"${m.max_cost_usd:.2f}" if m.max_cost_usd is not None else "none"
    typer.echo(f"total: ${total_low:.2f} - ${total_high:.2f} | hard cap {cap}")
    if dry_run:
        return
    runner = BenchmarkRunner(m, out, replay=replay, log=typer.echo)
    asyncio.run(runner.run())
    summary = (runner.out / "summary.md").read_text(encoding="utf-8")
    typer.echo("\n" + summary)
    typer.echo(f"spent ${runner.budget.spent:.4f}; artifacts in {runner.out}")
    if update_readme:
        from guide_exemplars.benchmark.report import inject_readme
        from guide_exemplars.paths import REPO_ROOT

        ok = inject_readme(REPO_ROOT / "README.md", [summary])
        typer.echo("README updated" if ok else "README has no RESULTS markers; not updated")


@app.command()
def optimize(
    preset: str = typer.Option(..., help="dataset preset"),
    task: str | None = typer.Option(None, help="task id within the preset"),
    backbone: str = typer.Option("fake"),
    seed: int | None = typer.Option(None),
    set_: list[str] = typer.Option([], "--set", help="GuideConfig override key=value (repeatable)"),
    config: Path | None = typer.Option(None, help="YAML file of GuideConfig values"),
    max_cost_usd: float | None = typer.Option(None, help="hard spend cap"),
    out: Path | None = typer.Option(None, help="output directory"),
    train_limit: int | None = typer.Option(None),
    val_limit: int | None = typer.Option(None),
    resume: bool = typer.Option(False, help="resume from the latest checkpoint in --out"),
    dry_run: bool = typer.Option(False, help="print the estimate and exit"),
) -> None:
    """Run a GUIDE optimization on a prepared preset and write the frozen exemplar set."""
    from guide_exemplars.paths import RESULTS_DIR
    from guide_exemplars.runtime import build_runtime, estimate_for, run_optimize

    cfg = _guide_config(set_, seed, config)
    prepared = _prepared(preset, task, train_limit, val_limit)
    bb = _backbone(backbone)
    typer.echo(f"{_split_summary(prepared)} | backbone {backbone}")
    _print_estimate(estimate_for(prepared, bb, cfg), max_cost_usd)
    if dry_run:
        return
    out = (
        out
        or RESULTS_DIR
        / "runs"
        / f"{prepared.task_id}-{backbone}-s{cfg.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    spent = 0.0
    if resume and (out / "usage.json").exists():
        spent = json.loads((out / "usage.json").read_text(encoding="utf-8")).get("budget_spent_usd", 0.0)
    rt = build_runtime(
        bb, max_cost_usd=max_cost_usd, spent_usd=spent, gold_items=prepared.items, scale=prepared.spec.scale
    )

    def progress(event) -> None:
        if event.type == "eval_completed":
            p = event.payload
            typer.echo(
                f"  r{event.round} #{p['index']:3d} {p['op']:9s} |E|={p['size']:2d} acc={p['acc']:.3f} "
                f"best={p['best_acc_global']:.3f} C={p['contrastive']:.2f} ${rt.budget.spent:.3f}"
            )
        elif event.type in ("phase0_completed", "pool_ready", "phase2_completed", "round_completed"):
            typer.echo(f"{event.type} r{event.round}: {json.dumps(event.payload)[:200]}")

    outcome = asyncio.run(run_optimize(prepared, rt, cfg, out, events=progress, resume=resume))
    res = outcome.result
    best = res.best
    typer.echo(
        f"status={res.status} observations={len(res.observations)} cost=${rt.budget.spent:.4f} out={out}"
    )
    if best:
        typer.echo(
            f"best: round {best.round} |E|={best.size} val acc={best.metrics.acc:.3f} qwk={best.metrics.qwk:.3f}"
        )


def load_env() -> None:
    """Load a .env from the working directory (or its parents) without overriding the real environment."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


@app.callback()
def main() -> None:
    """guide-exemplars: GUIDE exemplar optimization for LLM-based ordinal grading."""
    load_env()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@backbones_app.command("list")
def backbones_list(config: Path | None = typer.Option(None, help="backbones.yaml path")) -> None:
    """List configured backbones and whether each is ready to run."""
    from guide_exemplars.llm.backbones import load_backbones

    for name, cfg in load_backbones(config).items():
        missing = cfg.missing_settings()
        status = "ready" if not missing else "missing: " + "; ".join(missing)
        typer.echo(f"{name:12s} {cfg.kind:7s} {cfg.model:45s} embedder={cfg.embedder:32s} {status}")


@backbones_app.command("check")
def backbones_check(
    name: str = typer.Argument(..., help="backbone name from backbones.yaml"),
    config: Path | None = typer.Option(None, help="backbones.yaml path"),
    max_cost_usd: float = typer.Option(0.01, help="hard cap for the check calls"),
) -> None:
    """Send one tiny JSON request and one tiny embedding request; report latency, usage and cost."""
    from guide_exemplars.core.protocols import LLMRequest
    from guide_exemplars.llm.backbones import build_embedder, load_backbones
    from guide_exemplars.llm.budget import Budget, UsageMeter
    from guide_exemplars.llm.cache import NullResponseCache

    backbones = load_backbones(config)
    if name not in backbones:
        typer.echo(f"unknown backbone {name!r}; known: {', '.join(backbones)}", err=True)
        raise typer.Exit(2)
    cfg = backbones[name]
    missing = cfg.missing_settings()
    if missing:
        typer.echo(f"{name} is not ready: {'; '.join(missing)}", err=True)
        raise typer.Exit(2)

    from guide_exemplars.llm.backbones import build_llm

    budget, meter = Budget(max_cost_usd), UsageMeter()
    llm = build_llm(cfg, cache=NullResponseCache(), meter=meter, budget=budget)
    embedder = build_embedder(
        cfg.embedder,
        meter=meter,
        budget=budget,
        api_base=cfg.embedder_api_base,
        api_key_env=cfg.embedder_api_key_env,
    )
    req = LLMRequest.of(
        system="You are a connectivity check. Reply with a single JSON object and nothing else.",
        user='Return exactly {"ok": true, "label": "2"}.',
        max_tokens=30,
        purpose="check",
    )

    async def run() -> int:
        ok = True
        t0 = time.perf_counter()
        try:
            res = await llm.complete(req)
        except Exception as exc:  # report, don't crash, so the embedder is still checked
            typer.echo(f"chat:     FAILED {type(exc).__name__}: {exc}")
            ok = False
        else:
            try:
                parsed = json.loads(res.text)
                json_ok = isinstance(parsed, dict) and parsed.get("ok") is True
            except json.JSONDecodeError:
                json_ok = False
            ok &= json_ok
            typer.echo(
                f"chat:     {(time.perf_counter() - t0) * 1000:.0f} ms  prompt={res.prompt_tokens} "
                f"(cached {res.cached_prompt_tokens}) completion={res.completion_tokens} "
                f"cost=${res.cost_usd:.6f}  json_ok={json_ok}  think_tags={'<think>' in res.text}"
            )
            typer.echo(f"          reply: {res.text[:200]!r}")
        t0 = time.perf_counter()
        try:
            vecs = await embedder.embed(["The cat sat on the mat.", "A cat was sitting on a mat."])
        except Exception as exc:
            typer.echo(f"embedder: FAILED {type(exc).__name__}: {exc}")
            return 1
        sim = float(vecs[0] @ vecs[1])
        typer.echo(
            f"embedder: {(time.perf_counter() - t0) * 1000:.0f} ms  model={embedder.model_id} "
            f"dim={vecs.shape[1]} paraphrase_cos={sim:.3f}"
        )
        typer.echo(f"total:    calls={meter.total.calls} cost=${meter.total.cost_usd:.6f}")
        return 0 if ok else 1

    raise typer.Exit(asyncio.run(run()))
