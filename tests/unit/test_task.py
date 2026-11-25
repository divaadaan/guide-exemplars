from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from guide_exemplars.core.task import TaskSpec

REPO_ROOT = Path(__file__).resolve().parents[2]

BASE = {
    "name": "toy",
    "labels": ["Beginning", "Developing", "Proficient"],
    "persona": "You are an assessor.",
    "instruction": "Score the response.",
    "rubric": "Higher is better.",
    "rationale_focus": "Cite evidence.",
}


def test_bare_labels_are_coerced() -> None:
    spec = TaskSpec.model_validate(BASE)
    assert spec.label_values == ("Beginning", "Developing", "Proficient")
    assert spec.scale.rank("Proficient") == 2
    assert spec.response_noun == "response"


def test_shipped_aes_p7_spec_loads() -> None:
    spec = TaskSpec.from_file(REPO_ROOT / "configs" / "tasks" / "asap-aes-p7-conventions.yaml")
    assert spec.label_values == ("0", "1", "2", "3")
    assert spec.scale.description(3)
    assert spec.response_noun == "essay"


def test_spec_hash_is_stable_and_content_sensitive() -> None:
    a = TaskSpec.model_validate(BASE)
    b = TaskSpec.model_validate(dict(BASE))
    c = TaskSpec.model_validate({**BASE, "rubric": "Different."})
    assert a.spec_hash() == b.spec_hash()
    assert a.spec_hash() != c.spec_hash()


def test_invalid_specs_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskSpec.model_validate({**BASE, "unexpected": 1})
    with pytest.raises(ValidationError):
        TaskSpec.model_validate({**BASE, "labels": ["a", "a"]})
    with pytest.raises(ValidationError):
        TaskSpec.model_validate({**BASE, "labels": ["only"]})


def test_spec_is_frozen() -> None:
    spec = TaskSpec.model_validate(BASE)
    with pytest.raises(ValidationError):
        spec.name = "other"  # type: ignore[misc]
