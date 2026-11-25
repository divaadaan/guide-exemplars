"""Checkpoints written at phase boundaries, so an interrupted optimization can resume exactly where it stopped."""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from guide_exemplars.core.metrics import Metrics
from guide_exemplars.core.types import Exemplar, Observation, Subset

Phase = Literal["phase0", "phase1", "phase2"]
_PHASE_ORDER = {"phase0": 0, "phase1": 1, "phase2": 2}


def observation_to_dict(o: Observation) -> dict[str, Any]:
    m = o.metrics.to_dict()
    m["qwk"] = None if math.isnan(m["qwk"]) else m["qwk"]
    return {
        "round": o.round,
        "index": o.index,
        "uids": sorted(o.subset.uids),
        "op": o.op,
        "metrics": m,
        "contrastive": o.contrastive,
        "ei": None if o.ei is None or math.isnan(o.ei) else o.ei,
    }


def observation_from_dict(d: dict[str, Any]) -> Observation:
    m = dict(d["metrics"])
    m["qwk"] = math.nan if m["qwk"] is None else m["qwk"]
    return Observation(
        round=d["round"],
        index=d["index"],
        subset=Subset.of(d["uids"]),
        op=d["op"],
        metrics=Metrics(**m),
        contrastive=d["contrastive"],
        ei=d["ei"],
    )


@dataclass
class Checkpoint:
    round: int
    phase: Phase
    pool: list[dict[str, Any]]  # Pool.to_records()
    observations: list[dict[str, Any]]  # observation_to_dict, all rounds so far
    registry: list[dict[str, Any]]  # every exemplar ever seen, so evicted winners stay resolvable
    round_bests: dict[str, list[str]] = field(default_factory=dict)  # str(round) -> uids

    def sort_key(self) -> tuple[int, int]:
        return self.round, _PHASE_ORDER[self.phase]

    def to_json(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "phase": self.phase,
            "pool": self.pool,
            "observations": self.observations,
            "registry": self.registry,
            "round_bests": self.round_bests,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Checkpoint:
        return cls(**d)

    def registry_exemplars(self) -> dict[str, Exemplar]:
        return {e.uid: e for e in (Exemplar.model_validate(r) for r in self.registry)}


class CheckpointStore(Protocol):
    def save(self, cp: Checkpoint) -> None: ...

    def latest(self) -> Checkpoint | None: ...


class MemoryCheckpointStore:
    def __init__(self) -> None:
        self.saved: list[Checkpoint] = []

    def save(self, cp: Checkpoint) -> None:
        self.saved.append(Checkpoint.from_json(json.loads(json.dumps(cp.to_json()))))

    def latest(self) -> Checkpoint | None:
        return max(self.saved, key=Checkpoint.sort_key) if self.saved else None


class FileCheckpointStore:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, cp: Checkpoint) -> None:
        path = self.dir / f"checkpoint_r{cp.round:02d}_{cp.phase}.json.gz"
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(cp.to_json(), fh)
        tmp.replace(path)

    def latest(self) -> Checkpoint | None:
        found = []
        for path in self.dir.glob("checkpoint_r*_phase*.json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                found.append(Checkpoint.from_json(json.load(fh)))
        return max(found, key=Checkpoint.sort_key) if found else None
