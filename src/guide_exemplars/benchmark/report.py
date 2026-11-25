"""Benchmark reports: summary.csv, deltas.csv (paired GUIDE - other), summary.md, and README injection."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from guide_exemplars.benchmark.bootstrap import METRICS, paired_delta

if TYPE_CHECKING:
    from guide_exemplars.benchmark.matrix import Matrix
    from guide_exemplars.benchmark.runner import MethodResult

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"
_LABELS = {"acc": "Acc", "qwk": "QWK", "adj_err": "AdjErr", "nonadj_err": "NonAdjErr"}


def _fmt(x: float | None, digits: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _method_order(methods: list[str]) -> list[str]:
    base = ["guide", *sorted(m for m in methods if m.startswith("guide-"))]
    rest = [m for m in ("naive", "random", "knn", "vote_k") if m in methods]
    return [m for m in base if m in methods] + rest


def write_reports(results: list[MethodResult], matrix: Matrix, out: Path, manifest: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        row = {"task": r.task, "seed": r.seed, "method": r.method, "k": r.k, "n": r.metrics.get("n")}
        for m in METRICS:
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = (r.metrics[m][x] for x in ("point", "lo", "hi"))
        row["val_acc"] = r.extra.get("val_acc")
        rows.append(row)
    if rows:
        with (out / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    by_key = {(r.task, r.seed, r.method): r for r in results}
    bs = matrix.bootstrap
    delta_rows = []
    for (task, seed, method), r in by_key.items():
        guide = by_key.get((task, seed, "guide"))
        if guide is None or method == "guide":
            continue
        d = paired_delta(
            r.gold, guide.preds, r.preds, n_labels=_n_labels(guide), B=bs.B, ci=bs.ci, seed=bs.seed
        )
        row = {"task": task, "seed": seed, "vs": method}
        for m in METRICS:
            row.update({f"{m}_delta": d[m]["delta"], f"{m}_lo": d[m]["lo"], f"{m}_hi": d[m]["hi"]})
            row[f"{m}_p_gt0"] = d[m]["p_gt0"]
        delta_rows.append(row)
    if delta_rows:
        with (out / "deltas.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(delta_rows[0]))
            writer.writeheader()
            writer.writerows(delta_rows)

    (out / "summary.md").write_text(render_markdown(results, delta_rows, matrix, manifest), encoding="utf-8")


def _n_labels(r: MethodResult) -> int:
    values = [g for g in r.gold] + [p for d in r.preds for p in d if p is not None]
    return max(values) + 1 if values else 2


def render_markdown(
    results: list[MethodResult], delta_rows: list[dict], matrix: Matrix, manifest: dict[str, Any]
) -> str:
    lines = [f"### Benchmark `{matrix.name}`", ""]
    bb = manifest.get("backbone", {})
    lines.append(
        f"Backbone `{bb.get('model', matrix.backbone)}`, seeds {matrix.seeds}, test split, "
        f"{int(matrix.bootstrap.ci * 100)}% percentile-bootstrap CIs (B={matrix.bootstrap.B}). "
        f"Spend ${manifest.get('spent_usd', 0):.2f}."
    )
    lines.append("")
    by_task: dict[str, list[MethodResult]] = defaultdict(list)
    for r in results:
        by_task[r.task].append(r)
    for task, rs in by_task.items():
        info = manifest.get("tasks", {}).get(task, {})
        human = info.get("human") or {}
        lines.append(
            f"**{task}** (test n={info.get('n_test', '?')}; human rater1 vs rater2: "
            f"acc {_fmt(human.get('acc'))}, QWK {_fmt(human.get('qwk'))})"
        )
        lines.append("")
        lines.append("| Method | k | " + " | ".join(_LABELS[m] for m in METRICS) + " |")
        lines.append("|---|---:|" + "---:|" * len(METRICS))
        methods = _method_order(sorted({r.method for r in rs}))
        for method in methods:
            seeds = [r for r in rs if r.method == method]
            cells = []
            for m in METRICS:
                points = [r.metrics[m]["point"] for r in seeds]
                if len(seeds) == 1:
                    r0 = seeds[0].metrics[m]
                    cells.append(f"{_fmt(r0['point'])} [{_fmt(r0['lo'])}, {_fmt(r0['hi'])}]")
                else:
                    cells.append(f"{_fmt(float(np.nanmean(points)))} ± {_fmt(float(np.nanstd(points)))}")
            k = "/".join(sorted({str(r.k) for r in seeds}))
            name = f"**{method}**" if method == "guide" else method
            lines.append(f"| {name} | {k} | " + " | ".join(cells) + " |")
        lines.append("")
        lines += _reference_lines(rs, methods)
        task_deltas = [d for d in delta_rows if d["task"] == task]
        if task_deltas:
            lines.append(
                "Paired GUIDE - method (accuracy): "
                + "; ".join(
                    f"{d['vs']} {d['acc_delta']:+.3f} [{d['acc_lo']:+.3f}, {d['acc_hi']:+.3f}], "
                    f"P(Δ>0)={d['acc_p_gt0']:.2f}"
                    for d in task_deltas
                    if d["seed"] == matrix.seeds[0]
                )
            )
            lines.append("")
    return "\n".join(lines) + "\n"


def _reference_lines(rs: list[MethodResult], methods: list[str]) -> list[str]:
    """Majority-class reference and predicted-score counts: accuracy alone hides a model that leans on the
    most common score, which is easy on skewed ordinal data."""
    gold = rs[0].gold
    if not gold:
        return []
    n_labels = max(_n_labels(r) for r in rs)
    gold_counts = [gold.count(v) for v in range(n_labels)]
    majority = max(range(n_labels), key=lambda v: gold_counts[v])
    out = [
        f"Reference: always predicting the most common score (level {majority}) gives accuracy "
        f"{_fmt(gold_counts[majority] / len(gold))} and QWK 0.",
        "",
        "Predicted score counts on the test split (mean per draw for multi-draw methods):",
        "",
        "| | " + " | ".join(f"level {v}" for v in range(n_labels)) + " | failed |",
        "|---|" + "---:|" * (n_labels + 1),
        "| *gold* | " + " | ".join(str(c) for c in gold_counts) + " | |",
    ]
    for method in methods:
        r = next((x for x in rs if x.method == method and x.seed == rs[0].seed), None)
        if r is None:
            continue
        n_draws = max(1, len(r.preds))
        counts = [sum(d.count(v) for d in r.preds) / n_draws for v in range(n_labels)]
        failed = sum(d.count(None) for d in r.preds) / n_draws
        cells = [f"{c:g}" if c == int(c) else f"{c:.1f}" for c in [*counts, failed]]
        out.append(f"| {method} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def inject_readme(readme: Path, markdown_blocks: list[str]) -> bool:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        return False
    block = README_START + "\n" + "\n".join(markdown_blocks).strip() + "\n" + README_END
    readme.write_text(pattern.sub(lambda _m: block, text), encoding="utf-8")
    return True
