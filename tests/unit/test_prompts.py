from __future__ import annotations

import re

import pytest

from guide_exemplars.core.prompts import (
    TAG_TARGET,
    TEMPLATE_DIR,
    PromptBuilder,
    ordered_demos,
    parse_grade,
    parse_rationale,
)
from guide_exemplars.core.task import TaskSpec
from guide_exemplars.core.types import Exemplar
from support.toy import toy_task

TASK = toy_task()
P = PromptBuilder(TASK)


def demo(i: int, label: str) -> Exemplar:
    return Exemplar(item_id=f"d{i}", text=f"demo text {i}", label=label, rationale=f"reason {i}")


DEMOS = [demo(0, "Beginning"), demo(1, "Advanced"), demo(2, "Proficient")]


def test_demonstrations_precede_the_scored_response() -> None:
    user = P.grade_user(DEMOS, "the response")
    assert user.index("</examples>") < user.index(f"<{TAG_TARGET}>")
    assert user.rstrip().endswith(f"</{TAG_TARGET}>")


def test_same_set_any_input_order_renders_identically() -> None:
    a = P.grade_user(DEMOS, "x")
    b = P.grade_user(list(reversed(DEMOS)), "x")
    assert a == b
    assert [d.uid for d in ordered_demos(DEMOS)] == [d.uid for d in ordered_demos(DEMOS[::-1])]


def test_shared_prefix_is_identical_across_responses() -> None:
    a, b = P.grade_user(DEMOS, "first response"), P.grade_user(DEMOS, "second response")
    cut = a.index(f"<{TAG_TARGET}>")
    assert a[:cut] == b[:cut]


def test_system_prompt_lists_levels_and_output_contract() -> None:
    sys = P.grade_system
    for lbl in TASK.scale.labels:
        assert f"- {lbl}: {lbl} level work" in sys
    assert '"label"' in sys and "Beginning, Developing, Proficient, Advanced" in sys


def test_no_demos_renders_cleanly() -> None:
    user = P.grade_user([], "solo")
    assert "<examples>" not in user and "solo" in user


def test_discriminative_rationale_names_neighbours_and_is_one_sided_at_ends() -> None:
    mid = P.rationale_user("t", "Developing", discriminative=True)
    assert "deserves Developing and not Beginning or Proficient" in mid
    low = P.rationale_user("t", "Beginning", discriminative=True)
    assert "and not Developing." in low and "would not" not in low
    top = P.rationale_user("t", "Advanced", discriminative=True)
    assert "and not Proficient." in top and "keeps it from" not in top
    plain = P.rationale_user("t", "Proficient")
    assert "earns a score of Proficient" in plain and "and not" not in plain


def test_templates_are_domain_agnostic() -> None:
    banned = re.compile(r"\b(essay|essays|conventions|spelling|student|grade 7|grammar)\b", re.IGNORECASE)
    for path in TEMPLATE_DIR.glob("*.j2"):
        assert not banned.search(path.read_text(encoding="utf-8")), f"domain word in {path.name}"


def test_task_text_flows_through() -> None:
    task = TaskSpec.model_validate({**TASK.model_dump(), "context": "CTX-MARKER", "response_noun": "answer"})
    p = PromptBuilder(task)
    assert "CTX-MARKER" in p.grade_system and "CTX-MARKER" in p.rationale_system
    assert "Score the following answer." in p.grade_user([], "x")


@pytest.mark.parametrize(
    ("reply", "label", "has_error"),
    [
        ('{"analysis": "ok", "label": "Proficient"}', "Proficient", False),
        ('Sure! {"label": "advanced"} hope that helps', "Advanced", False),
        ('{"score": "Developing"}', "Developing", False),
        ('{"label": "Expert"}', None, True),
        ('"label": "Beginning", trailing', "Beginning", False),
        ("I think it is good", None, True),
    ],
)
def test_parse_grade(reply: str, label: str | None, has_error: bool) -> None:
    parsed = parse_grade(reply, TASK.scale)
    assert parsed.label == label
    assert (parsed.error is not None) == has_error


def test_parse_rationale() -> None:
    assert parse_rationale('{"rationale": "  Clear   reasons. "}') == "Clear reasons."
    assert parse_rationale("plain text fallback") == "plain text fallback"
    assert parse_rationale('{"rationale": ""}') is None
    assert parse_rationale("") is None
