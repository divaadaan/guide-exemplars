"""Benchmark runner end to end on toy data in a temp directory: run, resume-skip, and cache-only replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import guide_exemplars.data.presets as presets_mod
import guide_exemplars.llm.backbones as backbones_mod
from guide_exemplars.benchmark.matrix import Matrix
from guide_exemplars.benchmark.report import README_END, README_START, inject_readme
from guide_exemplars.benchmark.runner import BenchmarkRunner, estimate_matrix
from guide_exemplars.core.protocols import CacheMiss
from guide_exemplars.data.presets import Preset, load_prepared, prepare
from support.toy import toy_items, toy_task


@pytest.fixture
def toy_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(presets_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(backbones_mod, "CACHE_DIR", tmp_path / "cache")
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    with (raw / "toy.jsonl").open("w", encoding="utf-8") as fh:
        for it in toy_items(n_per_label=15):
            fh.write(json.dumps({"item_id": it.item_id, "text": it.text, "label": it.label}) + "\n")
    task_path = tmp_path / "toy-task.yaml"
    task_path.write_text(yaml.safe_dump(toy_task().model_dump(mode="json")), encoding="utf-8")
    preset_path = tmp_path / "toy-preset.yaml"
    preset_path.write_text(
        yaml.safe_dump(
            {
                "name": "toy",
                "tasks": [
                    {"id": "toy", "task": str(task_path), "source": {"kind": "file", "file": "toy.jsonl"}}
                ],
                "split": {"ratios": [3, 1, 1], "seed": 0, "experts_per_label": 1},
            }
        ),
        encoding="utf-8",
    )
    prepare(Preset.load(preset_path))
    return preset_path


def matrix(preset_path: Path) -> Matrix:
    return Matrix.model_validate(
        {
            "name": "toy-matrix",
            "backbone": "fake",
            "knn_embedder": "hash",
            "preset": str(preset_path),
            "methods": ["guide", "naive", "random", "knn", "vote_k", "feature", "hybrid"],
            "random_draws": 2,
            "ablations": {"no_discriminative": {"phase2": False}},
            "guide": {
                "n_rounds": 2,
                "n_eval": 6,
                "n_init": 3,
                "k_min": 3,
                "k_max": 6,
                "n_candidates": 24,
                "op_add": 6,
                "op_swap": 6,
                "op_flip": 6,
                "tau": 0.3,
            },
            "bootstrap": {"B": 50},
        }
    )


async def test_runner_produces_artifacts_and_replays(tmp_path: Path, toy_preset: Path) -> None:
    m = matrix(toy_preset)
    out = tmp_path / "results"
    results = await BenchmarkRunner(m, out, log=lambda _s: None).run()
    methods = {r.method for r in results}
    assert methods == {
        "guide",
        "guide-no_discriminative",
        "naive",
        "random",
        "knn",
        "vote_k",
        "feature",
        "hybrid",
    }
    for name in ("summary.csv", "deltas.csv", "summary.md", "manifest.json"):
        assert (out / name).exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tasks"]["toy"]["split_hash"] and "sha" in manifest["git"]
    guide = next(r for r in results if r.method == "guide")
    assert all(r.k == guide.k for r in results if r.method in ("naive", "random", "knn", "vote_k"))
    random_r = next(r for r in results if r.method == "random")
    assert len(random_r.preds) == 2
    feature = next(r for r in results if r.method == "feature")
    hybrid = next(r for r in results if r.method == "hybrid")
    assert feature.k == 0 and hybrid.k == guide.k
    assert feature.extra["features"] == ["gaps_per_100w", "vague_per_100w", "log_words"]
    assert hybrid.extra["features"][0] == "guide_score"
    train_ids = {it.item_id for it in load_prepared(Preset.load(toy_preset)).train}
    assert hybrid.extra["n_fit"] == feature.extra["n_fit"] - len(_guide_items(out) & train_ids)
    assert manifest["spent_usd_total"] == manifest["spent_usd"] and "history" not in manifest

    # Second run over the same directory skips everything that is recorded.
    again = await BenchmarkRunner(m, out, log=lambda _s: None).run()
    assert {(r.method, r.metrics["acc"]["point"]) for r in again} == {
        (r.method, r.metrics["acc"]["point"]) for r in results
    }
    resumed = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(resumed["history"]) == 1
    assert resumed["spent_usd_total"] == pytest.approx(manifest["spent_usd"] + resumed["spent_usd"])

    # Replay into a fresh directory: cache-only, byte-identical summaries.
    replay_out = tmp_path / "replay"
    await BenchmarkRunner(m, replay_out, replay=True, log=lambda _s: None).run()
    assert (replay_out / "summary.csv").read_bytes() == (out / "summary.csv").read_bytes()
    assert (replay_out / "deltas.csv").read_bytes() == (out / "deltas.csv").read_bytes()

    # Report-only rebuild: same tables, manifest untouched, plus the majority reference and score counts.
    manifest_before = (out / "manifest.json").read_bytes()
    summary_before = (out / "summary.md").read_text(encoding="utf-8")
    (out / "summary.md").unlink()
    rebuilt = BenchmarkRunner(m, out, log=lambda _s: None).rebuild_reports()
    assert {r.method for r in rebuilt} == methods
    assert (out / "manifest.json").read_bytes() == manifest_before
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert summary == summary_before
    assert "always predicting the most common score" in summary and "| *gold* |" in summary


async def test_replay_without_cache_fails_loudly(tmp_path: Path, toy_preset: Path) -> None:
    with pytest.raises(CacheMiss):
        await BenchmarkRunner(matrix(toy_preset), tmp_path / "r", replay=True, log=lambda _s: None).run()


def test_estimate_matrix_counts_every_method(toy_preset: Path) -> None:
    m = matrix(toy_preset)
    [row] = estimate_matrix(m)
    without_ablation = estimate_matrix(m.model_copy(update={"ablations": {}}))[0]
    assert row["calls"] > without_ablation["calls"] > 0
    no_fitted = m.model_copy(update={"methods": ["guide", "naive", "random", "knn", "vote_k"]})
    no_hybrid = m.model_copy(update={"methods": [x for x in m.methods if x != "hybrid"]})
    assert row["calls"] > estimate_matrix(no_hybrid)[0]["calls"] > estimate_matrix(no_fitted)[0]["calls"]
    assert row["usd_low"] <= row["usd_high"] <= row["usd_hard_max"]


def test_inject_readme(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\noutro\n", encoding="utf-8")
    assert inject_readme(readme, ["| new | table |"])
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and "| new | table |" in text and text.endswith("outro\n")
    plain = tmp_path / "plain.md"
    plain.write_text("no markers", encoding="utf-8")
    assert not inject_readme(plain, ["x"])


def _guide_items(out: Path) -> set[str]:
    [sets] = [json.loads(p.read_text(encoding="utf-8")) for p in out.glob("toy/seed0/guide/exemplars.json")]
    return {e["item_id"] for e in sets[0]}


def test_hybrid_needs_guide(toy_preset: Path) -> None:
    with pytest.raises(ValueError, match="add guide"):
        Matrix.model_validate({**matrix(toy_preset).model_dump(), "methods": ["naive", "hybrid"]})
