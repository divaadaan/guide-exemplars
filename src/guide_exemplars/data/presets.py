"""Dataset presets: YAML recipes that turn a raw public dataset into prepared, split task data.

Prepared layout, one directory per task::

    data/prepared/<preset>/<task_id>/items.jsonl   # item_id, text, label, split
    data/prepared/<preset>/<task_id>/report.json   # counts, label policy stats, human ceiling, hashes
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from guide_exemplars.core.task import TaskSpec
from guide_exemplars.core.types import Item
from guide_exemplars.data.splits import limit_split, split_hash, stratified_split
from guide_exemplars.paths import CONFIG_DIR, DATA_DIR


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["asap_aes", "asap_sas", "file"]
    file: str
    # "raw": ``file`` is under data/raw (downloaded, never committed). "config": ``file`` is under configs/ and ships
    # with the repo (the synthetic demo set), so the preset can be prepared on demand.
    root: Literal["raw", "config"] = "raw"
    essay_set: int | None = None
    trait: int | None = None
    label_policy: str = "min_drop_gap2"
    score_range: tuple[int, int] | None = None


class TaskEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    task: str  # path relative to configs/
    source: SourceSpec


class SplitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ratios: tuple[float, float, float] = (3, 1, 1)
    seed: int = 0
    experts_per_label: int = 1


class Preset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    tasks: list[TaskEntry] = Field(min_length=1)
    split: SplitSpec = SplitSpec()

    @classmethod
    def load(cls, name_or_path: str | Path) -> Preset:
        path = Path(name_or_path)
        if not path.suffix:
            path = CONFIG_DIR / "presets" / f"{name_or_path}.yaml"
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def task_entry(self, task_id: str | None) -> TaskEntry:
        if task_id is None:
            if len(self.tasks) != 1:
                raise ValueError(f"preset {self.name} has {len(self.tasks)} tasks; pick one with --task")
            return self.tasks[0]
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise ValueError(f"preset {self.name} has no task {task_id!r}; tasks: {[t.id for t in self.tasks]}")

    @property
    def bundled(self) -> bool:
        """True when every source ships with the repo, so `prepare` needs no download."""
        return all(t.source.root == "config" for t in self.tasks)


@dataclass
class PreparedTask:
    preset: str
    task_id: str
    spec: TaskSpec
    items: list[Item]
    splits: dict[str, str]
    report: dict[str, Any]

    def split(self, name: str) -> list[Item]:
        return [it for it in self.items if self.splits.get(it.item_id) == name]

    @property
    def train(self) -> list[Item]:
        return self.split("train")

    @property
    def val(self) -> list[Item]:
        return self.split("val")

    @property
    def test(self) -> list[Item]:
        return self.split("test")

    @property
    def experts(self) -> list[Item]:
        return self.split("expert")

    def limited(
        self, train: int | None = None, val: int | None = None, test: int | None = None
    ) -> PreparedTask:
        """Copy with label-stratified subsamples of the splits (experts untouched)."""
        seed = int(self.report.get("split_seed", 0))
        scale = self.spec.scale
        keep = {it.item_id for it in self.experts}
        keep |= {it.item_id for it in limit_split(self.train, train, seed, scale)}
        keep |= {it.item_id for it in limit_split(self.val, val, seed + 1, scale)}
        keep |= {it.item_id for it in limit_split(self.test, test, seed + 2, scale)}
        items = [it for it in self.items if it.item_id in keep]
        return PreparedTask(self.preset, self.task_id, self.spec, items, dict(self.splits), dict(self.report))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_source(entry: TaskEntry, spec: TaskSpec, raw_dir: Path) -> tuple[list[Item], dict, dict[str, str]]:
    src = entry.source
    path = (CONFIG_DIR if src.root == "config" else raw_dir) / src.file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `guide data fetch` or place the file there manually.")
    explicit: dict[str, str] = {}
    if src.kind in ("asap_aes", "asap_sas"):
        labels = [int(v) for v in spec.label_values]
        score_range = src.score_range or (min(labels), max(labels))
        if src.kind == "asap_aes":
            from guide_exemplars.data.sources import asap_aes

            raw, rep = asap_aes.load(path, src.essay_set, src.trait, src.label_policy, score_range)  # type: ignore[arg-type]
        else:
            from guide_exemplars.data.sources import asap_sas

            raw, rep = asap_sas.load(path, src.essay_set, score_range)  # type: ignore[arg-type]
        items = [Item(item_id=r.item_id, text=r.text, label=r.label) for r in raw]
        report = rep.to_dict()
    else:
        from guide_exemplars.data.sources import generic

        items, explicit = generic.load_file(path)
        report = {"n_raw": len(items), "n_kept": len(items)}
    report["raw_file"] = src.file
    report["raw_sha256"] = _sha256_file(path)
    return items, report, explicit


def prepare(preset: Preset, raw_dir: Path | None = None, out_dir: Path | None = None) -> list[Path]:
    raw_dir = raw_dir or DATA_DIR / "raw"
    out_dir = out_dir or DATA_DIR / "prepared"
    written = []
    for entry in preset.tasks:
        spec = TaskSpec.from_file(CONFIG_DIR / entry.task)
        items, report, explicit = _load_source(entry, spec, raw_dir)
        for it in items:
            spec.scale.rank(it.label)
        if explicit and len(explicit) == len(items):
            splits = explicit
        else:
            s = preset.split
            splits = stratified_split(items, spec.scale, s.ratios, s.seed, s.experts_per_label)
        counts: dict[str, int] = {}
        for split in splits.values():
            counts[split] = counts.get(split, 0) + 1
        report.update(
            {
                "preset": preset.name,
                "task_id": entry.id,
                "task_spec": entry.task,
                "task_spec_hash": spec.spec_hash(),
                "split_seed": preset.split.seed,
                "split_ratios": list(preset.split.ratios),
                "split_counts": counts,
                "split_hash": split_hash(items, splits),
            }
        )
        task_dir = out_dir / preset.name / entry.id
        task_dir.mkdir(parents=True, exist_ok=True)
        with (task_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
            for it in items:
                row = {"item_id": it.item_id, "text": it.text, "label": it.label, "split": splits[it.item_id]}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        (task_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        written.append(task_dir)
    return written


def load_prepared(
    preset: Preset, task_id: str | None = None, prepared_dir: Path | None = None
) -> PreparedTask:
    entry = preset.task_entry(task_id)
    task_dir = (prepared_dir or DATA_DIR / "prepared") / preset.name / entry.id
    items_path = task_dir / "items.jsonl"
    if not items_path.exists():
        raise FileNotFoundError(f"{items_path} not found; run `guide data prepare {preset.name}` first")
    items, splits = [], {}
    for line in items_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        items.append(Item(item_id=row["item_id"], text=row["text"], label=row["label"]))
        splits[str(row["item_id"])] = row["split"]
    report = json.loads((task_dir / "report.json").read_text(encoding="utf-8"))
    spec = TaskSpec.from_file(CONFIG_DIR / entry.task)
    if report.get("task_spec_hash") not in (None, spec.spec_hash()):
        report["warning"] = "task spec changed since preparation"
    return PreparedTask(preset.name, entry.id, spec, items, splits, report)
