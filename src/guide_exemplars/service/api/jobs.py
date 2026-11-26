"""Optimization jobs: create (or estimate), inspect, cancel, resume, stream events, and visualize the pool."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from guide_exemplars.core.checkpoint import Checkpoint
from guide_exemplars.core.contrastive import ContrastiveConfig, ContrastiveOps
from guide_exemplars.core.embedding_index import EmbeddingIndex
from guide_exemplars.core.optimizer import GuideConfig
from guide_exemplars.core.pool import Pool
from guide_exemplars.core.prompts import embedding_text
from guide_exemplars.llm.backbones import (
    BackboneConfig,
    build_embedder,
    default_embedding_store,
    load_backbones,
)
from guide_exemplars.runtime import estimate_for
from guide_exemplars.service.api.deps import AppState, job_out, not_found, prepared_from_db, state
from guide_exemplars.service.schemas import JobIn, JobOut
from guide_exemplars.service.store import TERMINAL

router = APIRouter(prefix="/api")
St = Annotated[AppState, Depends(state)]
TERMINAL_EVENTS = {"job_succeeded", "job_failed", "job_cancelled", "job_budget_exhausted"}


def _job(st: AppState, job_id: str) -> dict[str, Any]:
    row = st.store.get_job(job_id)
    if row is None:
        raise not_found("job", job_id)
    return row


@router.post("/jobs", response_model=None)
def create_job(
    body: JobIn,
    st: St,
    x_llm_api_key: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    """Queue a GUIDE optimization (202), or return its cost estimate with ``dry_run`` (200)."""
    backbones = load_backbones()
    if body.backbone not in backbones:
        raise HTTPException(
            status_code=422, detail=f"unknown backbone {body.backbone!r}; known: {sorted(backbones)}"
        )
    bb = backbones[body.backbone]
    try:
        cfg = GuideConfig.model_validate({**body.config, "seed": body.seed})
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc
    prepared = prepared_from_db(st, body.task_id, body.dataset_id)
    if len(prepared.val) == 0:
        raise HTTPException(status_code=422, detail="dataset has no validation items")
    estimate = estimate_for(prepared, bb, cfg).to_dict()
    if body.dry_run:
        return JSONResponse({"estimate": estimate}, status_code=200)
    missing = bb.missing_settings(api_key_override=x_llm_api_key)
    if missing:
        raise HTTPException(status_code=422, detail=f"backbone {bb.name} not ready: {'; '.join(missing)}")
    row = st.store.add_job(
        task_id=body.task_id,
        dataset_id=body.dataset_id,
        config_json=json.dumps(body.config),
        backbone_name=bb.name,
        backbone_json=json.dumps(bb.redacted()),
        seed=body.seed,
        set_name=body.set_name or prepared.spec.name,
        max_cost_usd=body.max_cost_usd,
    )
    st.keys.put(row["id"], x_llm_api_key)
    st.store.add_event(
        row["id"], "job_queued", 0, {"estimate_usd": [estimate["usd_low"], estimate["usd_high"]]}
    )
    st.runner.notify()
    return JSONResponse(
        {"job": job_out(row).model_dump(), "estimate": estimate},
        status_code=202,
    )


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(st: St, status: str | None = None, limit: int = Query(100, le=500)) -> list[JobOut]:
    return [job_out(r) for r in st.store.list_jobs(status, limit)]


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, st: St) -> JobOut:
    return job_out(_job(st, job_id))


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, st: St) -> JobOut:
    _job(st, job_id)
    # Compare-and-set transitions: the worker may finish concurrently, and its terminal status must win.
    if st.store.transition_job(job_id, ("queued", "interrupted"), status="cancelled", cancel_requested=1):
        seq = st.store.add_event(job_id, "job_cancelled", 0, {"before_start": True})
        st.bus.publish(job_id, {"type": "job_cancelled", "round": 0, "payload": {}, "seq": seq})
    else:
        st.cancels.request(job_id)
        if not st.store.transition_job(
            job_id, ("running", "cancelling"), status="cancelling", cancel_requested=1
        ):
            st.cancels.clear(job_id)
            raise HTTPException(status_code=409, detail=f"job already {_job(st, job_id)['status']}")
    return job_out(_job(st, job_id))


@router.post("/jobs/{job_id}/resume", response_model=JobOut)
def resume_job(
    job_id: str,
    st: St,
    max_cost_usd: float | None = None,
    x_llm_api_key: Annotated[str | None, Header()] = None,
) -> JobOut:
    """Requeue an interrupted, failed, cancelled or budget-exhausted job from its latest checkpoint."""
    job = _job(st, job_id)
    if job["status"] not in ("interrupted", "failed", "cancelled", "budget_exhausted"):
        raise HTTPException(status_code=409, detail=f"cannot resume a {job['status']} job")
    if not st.store.checkpoints(job_id):
        raise HTTPException(status_code=409, detail="job has no checkpoint to resume from")
    updates: dict[str, Any] = {
        "status": "queued",
        "resume_from": "latest",
        "cancel_requested": 0,
        "error": None,
    }
    if max_cost_usd is not None:
        updates["max_cost_usd"] = max_cost_usd
    st.store.update_job(job_id, **updates)
    st.keys.put(job_id, x_llm_api_key)
    st.store.add_event(job_id, "job_queued", 0, {"resume": True})
    st.runner.notify()
    return job_out(_job(st, job_id))


@router.get("/jobs/{job_id}/observations")
def observations(job_id: str, st: St) -> list[dict]:
    """Every evaluated subset, in order (for the convergence chart)."""
    _job(st, job_id)
    return [{"round": e["round"], **e["payload"]} for e in st.store.events(job_id, types=["eval_completed"])]


def _sse(msg: dict[str, Any]) -> str:
    head = f"id: {msg['seq']}\n" if msg.get("seq") else ""
    data = json.dumps({"type": msg["type"], "round": msg.get("round", 0), "payload": msg.get("payload", {})})
    return f"{head}event: {msg['type']}\ndata: {data}\n\n"


@router.get("/jobs/{job_id}/events")
async def stream_events(
    job_id: str,
    request: Request,
    st: St,
    after: int = 0,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """Server-sent events: persisted history after ``Last-Event-ID`` (or ``after``), then live updates."""
    _job(st, job_id)
    start = int(last_event_id) if last_event_id and last_event_id.isdigit() else after
    queue = st.bus.subscribe(job_id)

    async def gen():
        sent = start
        try:
            for ev in st.store.events(job_id, after=start):
                sent = ev["seq"]
                yield _sse(ev)
                if ev["type"] in TERMINAL_EVENTS:
                    return
            if (_job(st, job_id))["status"] in TERMINAL:
                return
            while True:
                if await request.is_disconnected():
                    return
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                seq = msg.get("seq")
                if seq is not None and seq <= sent:
                    continue
                if seq is not None:
                    sent = seq
                yield _sse(msg)
                if msg["type"] in TERMINAL_EVENTS:
                    return
        finally:
            st.bus.unsubscribe(job_id, queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/jobs/{job_id}/pool-map")
async def pool_map(
    job_id: str,
    st: St,
    round: int | None = None,
    max_edges: int = Query(800, le=20000),
) -> dict:
    """2-D PCA map of the round's pool with boundary-pair edges and the selected set highlighted."""
    job = _job(st, job_id)
    rows = st.store.checkpoints(job_id)
    if not rows:
        raise HTTPException(status_code=409, detail="no checkpoint yet; the pool is still being built")
    cps = [Checkpoint.from_json(r["payload"]) for r in rows]
    if round is not None:
        matching = [c for c in cps if c.round == round]
        if not matching:
            raise HTTPException(status_code=404, detail=f"no checkpoint for round {round}")
        cp = min(matching, key=Checkpoint.sort_key)  # phase1 of that round: the pool that round searched
    else:
        cp = max(cps, key=Checkpoint.sort_key)
    pool = Pool.from_records(cp.pool)
    spec = prepared_from_db(st, job["task_id"], job["dataset_id"]).spec
    cfg = GuideConfig.model_validate({**json.loads(job["config_json"]), "seed": job["seed"]})
    bb = BackboneConfig.model_validate(json.loads(job["backbone_json"]))
    embedder = build_embedder(
        bb.embedder,
        store=default_embedding_store(),
        api_base=bb.embedder_api_base,
        api_key_env=bb.embedder_api_key_env,
    )
    exemplars = pool.exemplars()
    vecs = await embedder.embed([embedding_text(e) for e in exemplars])
    index = EmbeddingIndex([e.uid for e in exemplars], vecs)
    ops = ContrastiveOps(
        pool,
        index,
        spec.scale,
        ContrastiveConfig(cfg.tau, cfg.adjacent_only, cfg.contrastive_score),
        cfg.k_min,
        cfg.k_max,
    )
    centred = index.vectors() - index.vectors().mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    xy = centred @ vt[:2].T if vt.shape[0] >= 2 else np.zeros((len(exemplars), 2))
    selected: set[str] = set()
    best_rounds = sorted(cp.round_bests, key=int)
    if best_rounds:
        selected = set(
            cp.round_bests[str(round)]
            if round is not None and str(round) in cp.round_bests
            else cp.round_bests[best_rounds[-1]]
        )
    if job.get("exemplar_set_id") and round is None:
        es = st.store.get_exemplar_set(job["exemplar_set_id"])
        if es:
            selected = {e["uid"] for e in es["exemplars"]}
    pos = {e.uid: i for i, e in enumerate(exemplars)}
    all_pairs = sorted(ops.boundary_pairs(), key=lambda p: -p.similarity)
    pairs = all_pairs[:max_edges]
    shown = {(p.a, p.b) for p in pairs}
    # Pairs that touch a selected exemplar are what the chart is about, so they are always included.
    pairs += [p for p in all_pairs[max_edges:] if (p.a in selected or p.b in selected) and (p.a, p.b) not in shown]
    return {
        "round": cp.round,
        "phase": cp.phase,
        "tau": cfg.tau,
        "labels": list(spec.scale.labels),
        "points": [
            {
                "uid": e.uid,
                "item_id": e.item_id,
                "label": e.label,
                "rank": spec.scale.rank(e.label),
                "x": float(xy[i, 0]),
                "y": float(xy[i, 1]),
                "origin": e.origin,
                "is_expert": e.is_expert,
                "selected": e.uid in selected,
                "text": e.text[:280],
                "rationale": e.rationale[:280],
            }
            for i, e in enumerate(exemplars)
        ],
        "edges": [[pos[p.a], pos[p.b], round_sim(p.similarity)] for p in pairs],
        "n_boundary_pairs": len(all_pairs),
    }


def round_sim(x: float) -> float:
    return float(f"{x:.4f}")
