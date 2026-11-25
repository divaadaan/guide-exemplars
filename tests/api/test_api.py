"""HTTP API end to end with the fake backbone: tasks → datasets → jobs → SSE → exemplar sets → /grade."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import guide_exemplars.llm.backbones as backbones_mod
from guide_exemplars.service.app import create_app
from guide_exemplars.service.settings import Settings
from guide_exemplars.service.store import Store
from support.toy import toy_items, toy_task

SMALL = {
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
}


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setattr(backbones_mod, "CACHE_DIR", tmp_path / "cache")
    return Settings(db_path=tmp_path / "service.db", heartbeat_s=0.2)


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as c:
        yield c


def jsonl() -> bytes:
    rows = [{"item_id": it.item_id, "text": it.text, "label": it.label} for it in toy_items(n_per_label=12)]
    return "\n".join(json.dumps(r) for r in rows).encode()


def setup_task_and_dataset(c: TestClient) -> tuple[str, str]:
    r = c.post(
        "/api/tasks",
        content=yaml.safe_dump(toy_task().model_dump(mode="json")),
        headers={"content-type": "application/yaml"},
    )
    assert r.status_code == 201, r.text
    task_id = r.json()["id"]
    again = c.post("/api/tasks", json=toy_task().model_dump(mode="json"))
    assert again.status_code == 201 and again.json()["id"] == task_id  # idempotent
    r = c.post(
        "/api/datasets",
        files={"file": ("toy.jsonl", jsonl(), "application/json")},
        data={"task_id": task_id, "seed": "0"},
    )
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["n_items"] == 48 and ds["split_counts"]["expert"] == 4
    return task_id, ds["id"]


def wait_for(c: TestClient, job_id: str, statuses: set[str], timeout: float = 60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stuck in {job['status']}")


def test_full_lifecycle(client: TestClient) -> None:
    task_id, dataset_id = setup_task_and_dataset(client)
    body = {"task_id": task_id, "dataset_id": dataset_id, "backbone": "fake", "config": SMALL, "seed": 1}

    est = client.post("/api/jobs", json={**body, "dry_run": True})
    assert est.status_code == 200 and est.json()["estimate"]["total_calls"] > 0

    r = client.post("/api/jobs", json=body)
    assert r.status_code == 202, r.text
    job_id = r.json()["job"]["id"]

    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        types = []
        for line in stream.iter_lines():
            if line.startswith("event: "):
                types.append(line[len("event: ") :])
                if types[-1].startswith("job_") and types[-1] not in ("job_queued", "job_started"):
                    break
    assert types[0] == "job_queued" and "job_started" in types and types[-1] == "job_succeeded"
    assert "eval_completed" in types and "round_completed" in types

    job = wait_for(client, job_id, {"succeeded"})
    assert job["exemplar_set_id"] and job["best_acc"] is not None and job["calls"] > 0

    obs = client.get(f"/api/jobs/{job_id}/observations").json()
    assert len(obs) == SMALL["n_rounds"] * SMALL["n_eval"]
    pool_map = client.get(f"/api/jobs/{job_id}/pool-map").json()
    assert pool_map["points"] and any(p["selected"] for p in pool_map["points"])
    assert all(0 <= i < len(pool_map["points"]) for e in pool_map["edges"] for i in e[:2])
    assert client.get(f"/api/jobs/{job_id}/pool-map?round=1").json()["round"] == 1

    s1 = client.get(f"/api/exemplar-sets/{job['exemplar_set_id']}").json()
    assert s1["version"] == 1 and s1["status"] == "final" and s1["exemplars"]

    # A second job on the same task/name produces version 2.
    job2 = client.post("/api/jobs", json={**body, "seed": 2}).json()["job"]["id"]
    s2_id = wait_for(client, job2, {"succeeded"})["exemplar_set_id"]
    assert client.get(f"/api/exemplar-sets/{s2_id}").json()["version"] == 2

    items = toy_items(n_per_label=12)[:3]
    g = client.post(
        "/api/grade",
        json={
            "set": {"task_id": task_id, "name": "toy-named", "version": "latest"},
            "responses": [{"id": it.item_id, "text": it.text} for it in items],
        },
    )
    assert g.status_code == 200, g.text
    out = g.json()
    assert out["exemplar_set_id"] == s2_id
    assert [r["id"] for r in out["results"]] == [it.item_id for it in items]
    assert all(r["label"] in toy_task().scale.labels for r in out["results"])

    # SSE replay from a Last-Event-ID only returns later events.
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers={"Last-Event-ID": "3"}) as stream:
        ids = [int(line[4:]) for line in stream.iter_lines() if line.startswith("id: ")]
    assert ids and min(ids) == 4


def test_validation_errors(client: TestClient) -> None:
    task_id, dataset_id = setup_task_and_dataset(client)
    assert client.post("/api/tasks", json={"name": "x"}).status_code == 422
    bad = b'{"item_id": "a", "text": "t", "label": "Expert"}\n'
    r = client.post("/api/datasets", files={"file": ("d.jsonl", bad)}, data={"task_id": task_id})
    assert r.status_code == 422 and "not on the task scale" in r.text
    r = client.post("/api/jobs", json={"task_id": task_id, "dataset_id": dataset_id, "backbone": "nope"})
    assert r.status_code == 422
    r = client.post(
        "/api/jobs",
        json={
            "task_id": task_id,
            "dataset_id": dataset_id,
            "backbone": "fake",
            "config": {"k_min": 9, "k_max": 3},
        },
    )
    assert r.status_code == 422
    assert client.get("/api/jobs/job_missing").status_code == 404
    assert client.post("/api/grade", json={"responses": [{"id": "1", "text": "x"}]}).status_code == 422


def test_cancel_running_job_yields_partial_set(client: TestClient) -> None:
    task_id, dataset_id = setup_task_and_dataset(client)
    cfg = {**SMALL, "n_rounds": 5, "n_eval": 24}  # long enough that cancel lands mid-run
    job_id = client.post(
        "/api/jobs", json={"task_id": task_id, "dataset_id": dataset_id, "backbone": "fake", "config": cfg}
    ).json()["job"]["id"]
    deadline = time.time() + 30
    while time.time() < deadline and not client.get(f"/api/jobs/{job_id}/observations").json():
        time.sleep(0.02)
    r = client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200 and r.json()["status"] in ("cancelling", "cancelled")
    job = wait_for(client, job_id, {"cancelled"})
    s = client.get(f"/api/exemplar-sets/{job['exemplar_set_id']}").json()
    assert s["status"] == "partial"
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_budget_exhaustion_respects_cap(client: TestClient) -> None:
    task_id, dataset_id = setup_task_and_dataset(client)
    job_id = client.post(
        "/api/jobs",
        json={
            "task_id": task_id,
            "dataset_id": dataset_id,
            "backbone": "fake",
            "config": SMALL,
            "max_cost_usd": 0.05,
        },
    ).json()["job"]["id"]
    job = wait_for(client, job_id, {"budget_exhausted", "succeeded"})
    assert job["status"] == "budget_exhausted"
    assert job["cost_usd"] <= 0.05 + 1e-9


def test_restart_marks_interrupted_and_resume_completes(settings: Settings) -> None:
    store = Store(settings.db_path)
    with TestClient(create_app(settings)) as c:
        task_id, dataset_id = setup_task_and_dataset(c)
    # Simulate a crash mid-run: a job left "running" with a checkpoint, and no worker alive.
    job = store.add_job(
        task_id=task_id,
        dataset_id=dataset_id,
        config_json=json.dumps(SMALL),
        backbone_name="fake",
        backbone_json=json.dumps(backbones_mod.load_backbones()["fake"].redacted()),
        seed=0,
        set_name="toy-named",
        max_cost_usd=None,
    )
    store.update_job(job["id"], status="running", heartbeat_at=time.time() - 3600)
    no_worker = Settings(db_path=settings.db_path, start_worker=False)
    with TestClient(create_app(no_worker)) as c:
        assert c.get(f"/api/jobs/{job['id']}").json()["status"] == "interrupted"
        assert c.post(f"/api/jobs/{job['id']}/resume").status_code == 409  # no checkpoint yet
    store.save_checkpoint(
        job["id"], 0, "phase0", _phase0_checkpoint(c_settings=settings, dataset_id=dataset_id)
    )
    with TestClient(create_app(settings)) as c:
        r = c.post(f"/api/jobs/{job['id']}/resume")
        assert r.status_code == 200 and r.json()["status"] in (
            "queued",
            "running",
            "succeeded",
        )  # worker may claim it first
        done = wait_for(c, job["id"], {"succeeded"})
        assert done["exemplar_set_id"]
        events = [e["type"] for e in store.events(job["id"])]
        assert "job_interrupted" in events and "resumed" in events


def _phase0_checkpoint(c_settings: Settings, dataset_id: str) -> dict:
    """A phase-0 checkpoint built offline from the dataset (one plain rationale per item)."""
    from guide_exemplars.core.checkpoint import Checkpoint
    from guide_exemplars.core.types import Exemplar

    rows = Store(c_settings.db_path).dataset_items(dataset_id)
    exemplars = [
        Exemplar(
            item_id=r["item_id"],
            text=r["text"],
            label=r["label"],
            rationale=f"Earns {r['label']}.",
            is_expert=r["split"] == "expert",
        )
        for r in rows
        if r["split"] in ("train", "expert")
    ]
    records = [e.model_dump(mode="json", exclude={"uid"}) for e in exemplars]
    return Checkpoint(round=0, phase="phase0", pool=records, observations=[], registry=records).to_json()


def test_meta_endpoints(client: TestClient) -> None:
    names = {b["name"]: b for b in client.get("/api/backbones").json()}
    assert names["fake"]["ready"]
    assert client.get("/api/health").json()["ok"]
    assert any(p["preset"] == "aes-p7" for p in client.get("/api/presets").json())
    assert client.get("/docs").status_code == 200


def test_bundled_preset_imports_without_prepare(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    from guide_exemplars.data import presets as presets_mod
    from guide_exemplars.service.api import catalog as catalog_mod

    monkeypatch.setattr(presets_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(catalog_mod, "DATA_DIR", tmp_path / "data")
    listed = {p["preset"]: p for p in client.get("/api/presets").json()}
    assert listed["demo"]["prepared"]  # offered in the dashboard even before any prepare
    r = client.post("/api/datasets/from-preset", json={"preset": "demo", "train_limit": 20, "val_limit": 8})
    assert r.status_code == 201, r.text
    assert (tmp_path / "data" / "prepared" / "demo" / "demo" / "items.jsonl").exists()
