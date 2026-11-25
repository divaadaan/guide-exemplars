"""Package-level guards: importability, version, CLI, and the core dependency boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml
from typer.testing import CliRunner

import guide_exemplars
from guide_exemplars.cli import app

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "src" / "guide_exemplars" / "core"

# `core` must stay a pure library: usable without the service, DB or LLM provider stacks.
FORBIDDEN_IN_CORE = {"fastapi", "starlette", "sqlalchemy", "litellm", "uvicorn", "pandas"}


def test_version_matches_cli() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == guide_exemplars.__version__


def test_core_has_no_forbidden_imports() -> None:
    offenders = []
    for path in CORE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots = [node.module.split(".")[0]]
            else:
                continue
            offenders += [f"{path.name}: {r}" for r in roots if r in FORBIDDEN_IN_CORE]
    assert not offenders, f"core imports forbidden packages: {offenders}"


def test_task_specs_parse_and_have_required_fields() -> None:
    required = {"name", "version", "labels", "persona", "instruction", "rubric", "rationale_focus"}
    specs = list((REPO_ROOT / "configs" / "tasks").glob("*.yaml"))
    assert specs, "no task specs found"
    for path in specs:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert required <= spec.keys(), f"{path.name} missing {required - spec.keys()}"
        assert len(spec["labels"]) >= 2
