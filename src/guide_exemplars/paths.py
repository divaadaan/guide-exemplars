"""Repository-relative default locations. Each can be overridden with an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("GUIDE_CONFIG_DIR", REPO_ROOT / "configs"))
DATA_DIR = Path(os.environ.get("GUIDE_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("GUIDE_RESULTS_DIR", REPO_ROOT / "results"))
CACHE_DIR = Path(os.environ.get("GUIDE_CACHE_DIR", REPO_ROOT / ".cache"))
