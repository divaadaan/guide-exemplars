"""Shared application state handed to route handlers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request

from guide_exemplars.core.task import TaskSpec
from guide_exemplars.core.types import Item
from guide_exemplars.data.presets import PreparedTask
from guide_exemplars.service.bus import Broadcaster
from guide_exemplars.service.schemas import JobOut
from guide_exemplars.service.settings import Settings
from guide_exemplars.service.store import Store
from guide_exemplars.service.worker import CancelFlags, JobRunner, KeyVault


@dataclass
class AppState:
    settings: Settings
    store: Store
    bus: Broadcaster
    runner: JobRunner
    keys: KeyVault
    cancels: CancelFlags


def state(request: Request) -> AppState:
    return request.app.state.guide


def not_found(what: str, ident: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{what} {ident!r} not found")


def task_spec(st: AppState, task_id: str) -> TaskSpec:
    row = st.store.get_task(task_id)
    if row is None:
        raise not_found("task", task_id)
    return TaskSpec.model_validate(json.loads(row["spec_json"]))


def prepared_from_db(st: AppState, task_id: str, dataset_id: str) -> PreparedTask:
    ds = st.store.get_dataset(dataset_id)
    if ds is None:
        raise not_found("dataset", dataset_id)
    if ds["task_id"] != task_id:
        raise HTTPException(status_code=422, detail="dataset belongs to a different task")
    spec = task_spec(st, task_id)
    rows = st.store.dataset_items(dataset_id)
    items = [Item(item_id=r["item_id"], text=r["text"], label=r["label"]) for r in rows]
    return PreparedTask("db", dataset_id, spec, items, {r["item_id"]: r["split"] for r in rows}, {})


def job_out(row: dict[str, Any]) -> JobOut:
    return JobOut(
        **{k: row[k] for k in JobOut.model_fields if k in row}, config=json.loads(row["config_json"])
    )
