"""Versioned exemplar sets and grading with them."""

from __future__ import annotations

import contextlib
import json
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from guide_exemplars.core.grader import Grader
from guide_exemplars.core.prompts import PromptBuilder
from guide_exemplars.core.protocols import BudgetExceeded
from guide_exemplars.core.task import TaskSpec
from guide_exemplars.core.types import Exemplar, Item
from guide_exemplars.llm.backbones import load_backbones
from guide_exemplars.runtime import build_runtime
from guide_exemplars.service.api.deps import AppState, not_found, state
from guide_exemplars.service.schemas import ExemplarSetOut, GradeIn, GradeOut, GradeResult

router = APIRouter(prefix="/api")
St = Annotated[AppState, Depends(state)]


@router.get("/exemplar-sets", response_model=list[ExemplarSetOut])
def list_sets(st: St, task_id: str | None = None, name: str | None = None) -> list[ExemplarSetOut]:
    return [ExemplarSetOut(**r) for r in st.store.list_exemplar_sets(task_id, name)]


@router.get("/exemplar-sets/{set_id}", response_model=ExemplarSetOut)
def get_set(set_id: str, st: St) -> ExemplarSetOut:
    row = st.store.get_exemplar_set(set_id)
    if row is None:
        raise not_found("exemplar set", set_id)
    return ExemplarSetOut(**row)


@router.post("/grade", response_model=GradeOut)
async def grade(body: GradeIn, st: St, x_llm_api_key: Annotated[str | None, Header()] = None) -> GradeOut:
    """Score new responses with a frozen exemplar set (by id, or task + name + version/"latest")."""
    if (body.exemplar_set_id is None) == (body.set is None):
        raise HTTPException(status_code=422, detail="give exactly one of exemplar_set_id or set")
    if len(body.responses) > st.settings.max_grade_batch:
        raise HTTPException(
            status_code=422, detail=f"at most {st.settings.max_grade_batch} responses per call"
        )
    if body.exemplar_set_id:
        es = st.store.get_exemplar_set(body.exemplar_set_id)
        if es is None:
            raise not_found("exemplar set", body.exemplar_set_id)
    else:
        ref = body.set
        version = None if ref.version == "latest" else int(ref.version)
        es = st.store.find_exemplar_set(ref.task_id, ref.name, version)
        if es is None:
            raise not_found("exemplar set", f"{ref.task_id}/{ref.name}@{ref.version}")

    task_row = st.store.get_task(es["task_id"])
    spec = TaskSpec.model_validate(json.loads(task_row["spec_json"]))
    backbones = load_backbones()
    name = body.backbone or es["backbone_name"]
    if name not in backbones:
        raise HTTPException(status_code=422, detail=f"unknown backbone {name!r}")
    bb = backbones[name]
    missing = bb.missing_settings(api_key_override=x_llm_api_key)
    if missing:
        raise HTTPException(status_code=422, detail=f"backbone {name} not ready: {'; '.join(missing)}")
    warnings = []
    if name != es["backbone_name"]:
        warnings.append(f"set was optimized with {es['backbone_name']}; grading with {name}")
    if es["status"] == "partial":
        warnings.append("this set comes from an incomplete optimization (cancelled or budget-exhausted)")

    cap = body.max_cost_usd if body.max_cost_usd is not None else st.settings.default_grade_cap_usd
    rt = build_runtime(bb, max_cost_usd=cap, api_key=x_llm_api_key, scale=spec.scale)
    exemplars = [Exemplar.model_validate(e) for e in es["exemplars"]]
    items = [Item(item_id=r.id, text=r.text, label=spec.scale.labels[0]) for r in body.responses]
    grader = Grader(PromptBuilder(spec), rt.llm, warmup=True)
    try:
        preds = await grader.grade(items, exemplars, purpose="serve")
    except BudgetExceeded as exc:
        raise HTTPException(
            status_code=402, detail=f"grading would exceed the ${cap:.2f} cap: {exc}"
        ) from exc
    st.store.add_usage([r.to_dict() for r in rt.meter.records], job_id=None, set_id=es["id"])
    results = []
    for p in preds:
        analysis = ""
        with contextlib.suppress(json.JSONDecodeError, AttributeError):
            analysis = str(json.loads(p.raw).get("analysis", ""))
        rank = spec.scale.rank(p.pred) if p.pred is not None else None
        results.append(GradeResult(id=p.item_id, label=p.pred, rank=rank, analysis=analysis, error=p.error))
    total = rt.meter.total
    return GradeOut(
        exemplar_set_id=es["id"],
        backbone=name,
        results=results,
        usage={"calls": total.calls, "cache_hits": total.cache_hits, "cost_usd": total.cost_usd},
        warnings=warnings,
    )
