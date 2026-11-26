"""TaskSpec: all of a grading task's domain text, loaded from YAML or JSON.

The core never hardcodes anything about essays, traits or subjects. Persona, instruction, rubric, context
and rationale focus all come from here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from guide_exemplars.core.scale import LabelScale, canonical_label


class LabelDef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    value: int | str
    description: str = ""


class CountDef(BaseModel):
    """One thing the feature extractor counts in a response, e.g. a kind of error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)


class FeatureSpec(BaseModel):
    """Optional: countable evidence for the feature grader (see ``core/feature_grader.py``).

    An LLM counts each item in ``counts`` in a response, zero-shot. Counts become rates per ``per_words`` words
    (raw counts when ``None``), and ``include_length`` adds log word count. A small ordinal model maps those
    numbers to a score.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instruction: str = Field(min_length=1)
    counts: tuple[CountDef, ...] = Field(min_length=1)
    per_words: int | None = Field(default=100, gt=0)
    include_length: bool = True

    @field_validator("counts")
    @classmethod
    def _unique_names(cls, v: tuple[CountDef, ...]) -> tuple[CountDef, ...]:
        names = [c.name for c in v]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate count names in {names}")
        return v


class TaskSpec(BaseModel):
    """Everything the prompts need to know about a grading task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    version: int = 1
    labels: tuple[LabelDef, ...]
    persona: str
    instruction: str
    rubric: str
    rationale_focus: str
    response_noun: str = "response"
    context: str | None = None
    rationale_max_words: int = Field(default=80, gt=0)
    features: FeatureSpec | None = None

    @field_validator("labels", mode="before")
    @classmethod
    def _coerce_labels(cls, v: Any) -> Any:
        if isinstance(v, list | tuple):
            return tuple(item if isinstance(item, dict | LabelDef) else {"value": item} for item in v)
        return v

    @field_validator("labels")
    @classmethod
    def _validate_scale(cls, v: tuple[LabelDef, ...]) -> tuple[LabelDef, ...]:
        LabelScale.from_spec([{"value": d.value, "description": d.description} for d in v])
        return v

    @property
    def scale(self) -> LabelScale:
        return LabelScale.from_spec([{"value": d.value, "description": d.description} for d in self.labels])

    @property
    def label_values(self) -> tuple[str, ...]:
        return tuple(canonical_label(d.value) for d in self.labels)

    def spec_hash(self) -> str:
        """Stable content hash. Changes whenever any prompt-relevant field changes."""
        data = self.model_dump(mode="json")
        # Specs without features keep the hash they had before the field existed.
        if data.get("features") is None:
            data.pop("features", None)
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_file(cls, path: str | Path) -> TaskSpec:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        return cls.model_validate(data)
