"""Service settings, from environment variables with sensible local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from guide_exemplars.paths import REPO_ROOT


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    db_path: Path
    worker_slots: int = 1
    auto_resume: bool = False
    heartbeat_s: float = 10.0
    stale_after_s: float = 60.0
    start_worker: bool = True
    default_grade_cap_usd: float = 1.0
    max_grade_batch: int = 200

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            db_path=Path(os.environ.get("GUIDE_DB_PATH", REPO_ROOT / ".cache" / "service.db")),
            worker_slots=int(os.environ.get("GUIDE_WORKER_SLOTS", "1")),
            auto_resume=_bool("GUIDE_AUTO_RESUME", False),
            start_worker=_bool("GUIDE_START_WORKER", True),
        )
