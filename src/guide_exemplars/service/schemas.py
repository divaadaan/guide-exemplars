"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskOut(BaseModel):
    id: str
    name: str
    version: int
    spec_hash: str
    spec: dict[str, Any] | None = None
    created_at: float


class DatasetOut(BaseModel):
    id: str
    task_id: str
    name: str
    n_items: int
    label_counts: dict[str, int]
    split_counts: dict[str, int]
    content_hash: str
    created_at: float


class FromPresetIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: str
    task: str | None = None
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None


class JobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    dataset_id: str
    backbone: str = "gpt4o-mini"
    config: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    max_cost_usd: float | None = Field(default=None, ge=0)
    set_name: str | None = None
    dry_run: bool = False


class JobOut(BaseModel):
    id: str
    task_id: str
    dataset_id: str
    backbone_name: str
    seed: int
    set_name: str
    status: str
    phase: str | None
    round: int
    bo_iter: int
    best_acc: float | None
    max_cost_usd: float | None
    cost_usd: float
    calls: int
    cached_calls: int
    created_at: float
    started_at: float | None
    finished_at: float | None
    error: str | None
    exemplar_set_id: str | None
    config: dict[str, Any]


class ExemplarSetOut(BaseModel):
    id: str
    task_id: str
    name: str
    version: int
    status: str
    job_id: str | None
    backbone_name: str
    val_metrics: dict[str, Any] | None
    exemplars: list[dict[str, Any]]
    created_at: float


class SetRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    name: str
    version: int | Literal["latest"] = "latest"


class GradeResponseIn(BaseModel):
    id: str
    text: str = Field(min_length=1)


class GradeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exemplar_set_id: str | None = None
    set: SetRef | None = None
    responses: list[GradeResponseIn] = Field(min_length=1)
    backbone: str | None = None
    max_cost_usd: float | None = Field(default=None, ge=0)


class GradeResult(BaseModel):
    id: str
    label: str | None
    rank: int | None
    analysis: str
    error: str | None


class GradeOut(BaseModel):
    exemplar_set_id: str
    backbone: str
    results: list[GradeResult]
    usage: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
