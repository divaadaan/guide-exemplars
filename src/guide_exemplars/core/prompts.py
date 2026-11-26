"""Prompt construction for grading and rationale generation.

Layout is chosen for provider prefix caching: everything shared across one evaluation (system prompt, rubric,
demonstrations) comes first, and the response being scored is the only varying suffix. Demonstration order is
a deterministic shuffle seeded by the set's content key. Label order therefore carries no recency signal, and
the same set always renders byte-identically.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from guide_exemplars.core.scale import LabelScale
from guide_exemplars.core.task import TaskSpec
from guide_exemplars.core.types import Exemplar, Subset

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Tags the simulator and parsers rely on. Keep in sync with the templates.
TAG_TARGET = "response_to_score"
TAG_EXPLAIN = "response_to_explain"

_ENV = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    autoescape=False,  # plain-text prompts, never HTML
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_LABEL_FIELD = re.compile(r"\"label\"\s*:\s*\"?([^\",}\n]+)\"?", re.IGNORECASE)


def embedding_text(e: Exemplar) -> str:
    """Text embedded for boundary-pair similarity: the response plus its rationale (as in the paper)."""
    return f"{e.text}\n\nRationale: {e.rationale}"


def ordered_demos(exemplars: Sequence[Exemplar]) -> list[Exemplar]:
    """Deterministic, content-seeded shuffle: the same set always yields the same order."""
    by_uid = sorted(exemplars, key=lambda e: e.uid)
    if len(by_uid) < 2:
        return by_uid
    seed = int(Subset.of(e.uid for e in by_uid).key(), 16)
    perm = np.random.default_rng(seed).permutation(len(by_uid))
    return [by_uid[i] for i in perm]


@dataclass(frozen=True)
class GradeParse:
    label: str | None
    analysis: str
    error: str | None


def _load_json_object(text: str) -> dict | None:
    for candidate in (text, *(m.group(0) for m in _JSON_OBJECT.finditer(text))):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_grade(text: str, scale: LabelScale) -> GradeParse:
    obj = _load_json_object(text)
    if obj is not None:
        raw = obj.get("label", obj.get("score"))
        label = scale.parse(raw)
        analysis = str(obj.get("analysis", "") or "")
        if label is not None:
            return GradeParse(label, analysis, None)
        return GradeParse(None, analysis, f"label {raw!r} not on scale")
    m = _LABEL_FIELD.search(text)
    if m:
        label = scale.parse(m.group(1))
        if label is not None:
            return GradeParse(label, "", None)
    return GradeParse(None, "", "no parsable label in reply")


def parse_counts(text: str, names: Sequence[str]) -> dict[str, int] | None:
    """Whole, non-negative counts for every name, or ``None`` when any is missing or malformed."""
    obj = _load_json_object(text)
    if obj is None:
        return None
    out: dict[str, int] = {}
    for name in names:
        value = obj.get(name)
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not number >= 0 or number != int(number):
            return None
        out[name] = int(number)
    return out


def parse_rationale(text: str) -> str | None:
    obj = _load_json_object(text)
    if obj is not None:
        value = obj.get("rationale")
        text = value if isinstance(value, str) else ""
    text = " ".join(text.split())
    return text or None


class PromptBuilder:
    def __init__(self, task: TaskSpec):
        self.task = task
        self.scale = task.scale
        self._levels = [(lbl, self.scale.description(lbl)) for lbl in self.scale.labels]
        self._grade_system = _ENV.get_template("grade_system.j2").render(
            task=task, levels=self._levels, labels=self.scale.labels
        )
        self._rationale_system = _ENV.get_template("rationale_system.j2").render(
            task=task, levels=self._levels
        )
        self._grade_user = _ENV.get_template("grade_user.j2")
        self._rationale_user = _ENV.get_template("rationale_user.j2")
        self._features_system = (
            _ENV.get_template("features_system.j2").render(task=task) if task.features is not None else None
        )
        self._features_user = _ENV.get_template("features_user.j2")

    @property
    def features_system(self) -> str:
        if self._features_system is None:
            raise ValueError(f"task {self.task.name!r} defines no features")
        return self._features_system

    def features_user(self, text: str) -> str:
        return self._features_user.render(task=self.task, text=text)

    @property
    def grade_system(self) -> str:
        return self._grade_system

    @property
    def rationale_system(self) -> str:
        return self._rationale_system

    def grade_user(self, demos: Sequence[Exemplar], text: str) -> str:
        return self._grade_user.render(task=self.task, demos=ordered_demos(demos), text=text)

    def rationale_user(
        self,
        text: str,
        label: str,
        demos: Sequence[Exemplar] = (),
        discriminative: bool = False,
    ) -> str:
        rank = self.scale.rank(label)
        lower = self.scale.label_at(rank - 1) if rank > 0 else None
        higher = self.scale.label_at(rank + 1) if rank + 1 < len(self.scale) else None
        return self._rationale_user.render(
            task=self.task,
            demos=ordered_demos(demos),
            text=text,
            label=label,
            discriminative=discriminative,
            neighbours=[x for x in (lower, higher) if x is not None],
            lower=lower,
            higher=higher,
        )
