"""FastAPI application factory. Run with ``guide serve`` or ``uvicorn guide_exemplars.service.app:app``."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from guide_exemplars import __version__
from guide_exemplars.service.api import catalog, jobs, sets
from guide_exemplars.service.api.deps import AppState
from guide_exemplars.service.bus import Broadcaster
from guide_exemplars.service.settings import Settings
from guide_exemplars.service.store import Store
from guide_exemplars.service.worker import CancelFlags, JobRunner, KeyVault

STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = Store(settings.db_path)
        bus = Broadcaster()
        keys, cancels = KeyVault(), CancelFlags()
        runner = JobRunner(store, bus, settings, keys, cancels)
        app.state.guide = AppState(settings, store, bus, runner, keys, cancels)
        if settings.start_worker:
            runner.start()
        else:
            runner.recover()
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(
        title="guide-exemplars",
        version=__version__,
        description=(
            "Boundary-focused in-context exemplar optimization (GUIDE) for LLM-based ordinal grading: "
            "register a task, upload labeled data, run an optimization job, and grade with the frozen set."
        ),
        lifespan=lifespan,
    )
    app.include_router(catalog.router, tags=["catalog"])
    app.include_router(jobs.router, tags=["jobs"])
    app.include_router(sets.router, tags=["exemplar sets"])
    if (STATIC_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")
    return app


def app_factory() -> FastAPI:  # for `uvicorn --factory guide_exemplars.service.app:app_factory`
    return create_app()
