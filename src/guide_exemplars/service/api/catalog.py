"""Tasks, datasets, backbones, presets and health."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from guide_exemplars.core.task import TaskSpec
from guide_exemplars.data.presets import Preset, load_prepared, prepare
from guide_exemplars.data.sources.generic import items_from_records, parse_records
from guide_exemplars.data.splits import stratified_split
from guide_exemplars.llm.backbones import load_backbones
from guide_exemplars.paths import CONFIG_DIR, DATA_DIR
from guide_exemplars.service.api.deps import AppState, not_found, state
from guide_exemplars.service.schemas import DatasetOut, FromPresetIn, TaskOut

router = APIRouter(prefix="/api")
St = Annotated[AppState, Depends(state)]


def _task_out(row: dict, with_spec: bool = False) -> TaskOut:
    return TaskOut(
        id=row["id"],
        name=row["name"],
        version=row["version"],
        spec_hash=row["spec_hash"],
        spec=json.loads(row["spec_json"]) if with_spec else None,
        created_at=row["created_at"],
    )


def _dataset_out(row: dict) -> DatasetOut:
    counts = json.loads(row["split_counts_json"])
    return DatasetOut(
        id=row["id"],
        task_id=row["task_id"],
        name=row["name"],
        n_items=sum(counts.values()),
        label_counts=json.loads(row["label_counts_json"]),
        split_counts=counts,
        content_hash=row["content_hash"],
        created_at=row["created_at"],
    )


def _register_task(st: AppState, spec: TaskSpec) -> dict:
    try:
        return st.store.add_task(spec.name, spec.version, spec.model_dump(mode="json"), spec.spec_hash())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/tasks", response_model=TaskOut, status_code=201)
async def create_task(request: Request, st: St) -> TaskOut:
    """Register a TaskSpec, as JSON or YAML (``Content-Type: application/yaml``). Idempotent per name+version."""
    body = await request.body()
    ctype = request.headers.get("content-type", "")
    try:
        data = yaml.safe_load(body) if "yaml" in ctype else json.loads(body or b"{}")
        spec = TaskSpec.model_validate(data)
    except (ValidationError, ValueError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _task_out(_register_task(st, spec), with_spec=True)


@router.get("/tasks", response_model=list[TaskOut])
def list_tasks(st: St) -> list[TaskOut]:
    return [_task_out(r) for r in st.store.list_tasks()]


@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: str, st: St) -> TaskOut:
    row = st.store.get_task(task_id)
    if row is None:
        raise not_found("task", task_id)
    return _task_out(row, with_spec=True)


@router.post("/datasets", response_model=DatasetOut, status_code=201)
async def upload_dataset(
    st: St,
    file: Annotated[UploadFile, File(description="CSV or JSONL with item_id, text, label[, split]")],
    task_id: Annotated[str, Form()],
    name: Annotated[str | None, Form()] = None,
    split_ratios: Annotated[str, Form()] = "3,1,1",
    seed: Annotated[int, Form()] = 0,
    experts_per_label: Annotated[int, Form()] = 1,
) -> DatasetOut:
    row = st.store.get_task(task_id)
    if row is None:
        raise not_found("task", task_id)
    spec = TaskSpec.model_validate(json.loads(row["spec_json"]))
    content = (await file.read()).decode("utf-8")
    fmt = "jsonl" if (file.filename or "").lower().endswith((".jsonl", ".ndjson")) else "csv"
    try:
        items, explicit = items_from_records(parse_records(content, fmt))
        ratios = [float(x) for x in split_ratios.split(",")]
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    unknown = sorted({it.label for it in items if it.label not in spec.scale})
    if unknown:
        raise HTTPException(status_code=422, detail=f"labels not on the task scale: {unknown}")
    if explicit and len(explicit) == len(items):
        splits = explicit
    else:
        splits = stratified_split(items, spec.scale, ratios, seed, experts_per_label)
    if not any(s == "val" for s in splits.values()) or not any(
        s in ("train", "expert") for s in splits.values()
    ):
        raise HTTPException(
            status_code=422, detail="dataset needs at least one training and one validation item"
        )
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    ds = st.store.add_dataset(
        task_id,
        name or file.filename or "upload",
        digest,
        seed,
        ratios,
        [(it.item_id, it.text, it.label, splits[it.item_id]) for it in items],
    )
    return _dataset_out(ds)


@router.post("/datasets/from-preset", status_code=201)
def dataset_from_preset(body: FromPresetIn, st: St) -> dict:
    """Import a prepared benchmark preset (``guide data prepare``) as a task plus dataset, keeping its split."""
    try:
        preset = Preset.load(body.preset)
        try:
            prepared = load_prepared(preset, body.task)
        except FileNotFoundError:
            if not preset.bundled:
                raise
            prepare(preset)  # bundled data (the synthetic demo) needs no download, so prepare it on first import
            prepared = load_prepared(preset, body.task)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if body.train_limit or body.val_limit or body.test_limit:
        prepared = prepared.limited(body.train_limit, body.val_limit, body.test_limit)
    task_row = _register_task(st, prepared.spec)
    keep = {it.item_id for it in prepared.items}
    rows = [
        (it.item_id, it.text, it.label, prepared.splits[it.item_id])
        for it in prepared.items
        if it.item_id in keep
    ]
    ds = st.store.add_dataset(
        task_row["id"],
        f"{prepared.preset}/{prepared.task_id}",
        str(prepared.report.get("split_hash", "")),
        int(prepared.report.get("split_seed", 0)),
        prepared.report.get("split_ratios", [3, 1, 1]),
        rows,
    )
    return {"task": _task_out(task_row).model_dump(), "dataset": _dataset_out(ds).model_dump()}


@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(st: St, task_id: str | None = None) -> list[DatasetOut]:
    return [_dataset_out(r) for r in st.store.list_datasets(task_id)]


@router.get("/datasets/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, st: St) -> DatasetOut:
    row = st.store.get_dataset(dataset_id)
    if row is None:
        raise not_found("dataset", dataset_id)
    return _dataset_out(row)


@router.get("/backbones")
def backbones() -> list[dict]:
    out = []
    for name, cfg in load_backbones().items():
        missing = cfg.missing_settings()
        out.append(
            {
                "name": name,
                "kind": cfg.kind,
                "model": cfg.model,
                "embedder": cfg.embedder,
                "self_hosted": cfg.self_hosted,
                "ready": not missing,
                "missing": missing,
            }
        )
    return out


@router.get("/presets")
def presets() -> list[dict]:
    out = []
    for path in sorted((CONFIG_DIR / "presets").glob("*.yaml")):
        preset = Preset.load(path)
        for t in preset.tasks:
            prepared = preset.bundled or (DATA_DIR / "prepared" / preset.name / t.id / "items.jsonl").exists()
            out.append(
                {"preset": preset.name, "task": t.id, "prepared": prepared, "description": preset.description}
            )
    return out


@router.get("/health")
def health(st: St) -> dict:
    alive = [t.name for t in st.runner._threads if t.is_alive()]
    return {"ok": st.store.ping(), "workers": alive}
