"""Benchmark matrix configuration (``configs/benchmarks/*.yaml``)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from guide_exemplars.paths import CONFIG_DIR

# feature: LLM-counted evidence + ordinal model. hybrid: the same plus GUIDE's predicted score (needs guide).
KNOWN_METHODS = {"guide", "naive", "random", "knn", "vote_k", "feature", "hybrid"}


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: int | None = None
    val: int | None = None
    test: int | None = None


class BootstrapSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    B: int = 2000
    ci: float = 0.95
    seed: int = 0


class Matrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    backbone: str
    embedder: str | None = None  # boundary-pair embedder; default: the backbone's
    knn_embedder: str = "st/all-MiniLM-L6-v2"  # response embedder for KNN / Vote-k (paper: SBERT MiniLM)
    preset: str
    tasks: list[str] | None = None  # default: every task in the preset
    seeds: list[int] = Field(default_factory=lambda: [0])
    methods: list[str] = Field(default_factory=lambda: ["guide", "naive", "random", "knn", "vote_k"])
    random_draws: int = Field(5, ge=1)
    ablations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    guide: dict[str, Any] = Field(default_factory=dict)  # GuideConfig overrides shared by every GUIDE run
    limits: Limits = Limits()
    k_default: int = 8  # baseline k when GUIDE is not in the matrix
    bootstrap: BootstrapSpec = BootstrapSpec()
    max_cost_usd: float | None = None

    def model_post_init(self, _ctx: Any) -> None:
        unknown = set(self.methods) - KNOWN_METHODS
        if unknown:
            raise ValueError(f"unknown methods {sorted(unknown)}; known: {sorted(KNOWN_METHODS)}")
        if "hybrid" in self.methods and "guide" not in self.methods:
            raise ValueError(
                "the hybrid method combines GUIDE's predictions with features; add guide to methods"
            )

    @classmethod
    def load(cls, name_or_path: str | Path) -> Matrix:
        path = Path(name_or_path)
        if not path.suffix:
            path = CONFIG_DIR / "benchmarks" / f"{name_or_path}.yaml"
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
