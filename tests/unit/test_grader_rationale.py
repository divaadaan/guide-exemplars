from __future__ import annotations

import json

import pytest

from guide_exemplars.core.grader import Grader
from guide_exemplars.core.prompts import TAG_TARGET, PromptBuilder
from guide_exemplars.core.protocols import BudgetExceeded, Cancelled, LLMRequest, ModelPrice
from guide_exemplars.core.rationale import RationaleGenerator
from guide_exemplars.core.types import Exemplar, Item
from guide_exemplars.llm.budget import Budget
from guide_exemplars.llm.client import LLMClient
from guide_exemplars.llm.fake import FakeBackend
from support.toy import toy_task

TASK = toy_task()
PROMPTS = PromptBuilder(TASK)
ITEMS = [Item(item_id=f"i{i}", text=f"response number {i}", label=TASK.scale.labels[i % 4]) for i in range(6)]


class Boom(RuntimeError):
    pass


def responder(req: LLMRequest) -> str:
    user = req.messages[-1][1]
    if "response number 2" in user.split(f"<{TAG_TARGET}>")[-1]:
        return "garbage without a label"
    if "response number 4" in user.split(f"<{TAG_TARGET}>")[-1]:
        raise Boom("provider error")
    return json.dumps({"analysis": "x", "label": "Proficient"})


def llm(**kw) -> LLMClient:
    return LLMClient(
        FakeBackend(responder=kw.pop("responder", responder)), "fake", price=ModelPrice(1, 1, 1), **kw
    )


class SetAfter:
    def __init__(self, n: int):
        self.n = n
        self.calls = 0

    def is_set(self) -> bool:
        self.calls += 1
        return self.calls > self.n


async def test_predictions_stay_aligned_when_items_fail() -> None:
    preds = await Grader(PROMPTS, llm()).grade(ITEMS, [])
    assert [p.item_id for p in preds] == [it.item_id for it in ITEMS]
    assert [p.gold for p in preds] == [it.label for it in ITEMS]
    assert preds[2].pred is None and "no parsable label" in preds[2].error
    assert preds[4].pred is None and "Boom" in preds[4].error
    assert all(p.pred == "Proficient" for i, p in enumerate(preds) if i not in (2, 4))


async def test_evaluate_counts_failures_as_wrong() -> None:
    metrics, _ = await Grader(PROMPTS, llm()).evaluate(ITEMS, [])
    assert metrics.n == 6 and metrics.n_failed == 2
    assert metrics.n_correct == sum(
        1 for i, it in enumerate(ITEMS) if i not in (2, 4) and it.label == "Proficient"
    )


async def test_budget_exhaustion_aborts_evaluation() -> None:
    capped = llm(responder=lambda r: '{"label": "Advanced"}', budget=Budget(max_usd=0.0002))
    with pytest.raises(BudgetExceeded):
        await Grader(PROMPTS, capped).grade(ITEMS * 10, [])


async def test_cancel_aborts_evaluation() -> None:
    grader = Grader(PROMPTS, llm(responder=lambda r: '{"label": "Advanced"}'), cancel=SetAfter(3))
    with pytest.raises(Cancelled):
        await grader.grade(ITEMS, [])


async def test_per_item_demos_callable() -> None:
    seen: list[str] = []

    def capture(req: LLMRequest) -> str:
        seen.append(req.messages[-1][1])
        return '{"label": "Advanced"}'

    demo = Exemplar(item_id="d", text="DEMO-FOR-i1", label="Advanced", rationale="r")
    await Grader(PROMPTS, llm(responder=capture)).grade(
        ITEMS[:3], lambda it: [demo] if it.item_id == "i1" else []
    )
    assert sum("DEMO-FOR-i1" in s for s in seen) == 1


async def test_rationale_context_excludes_own_item_and_failures_are_none() -> None:
    contexts: list[str] = []

    def rat(req: LLMRequest) -> str:
        user = req.messages[-1][1]
        contexts.append(user)
        if "response number 3" in user.split("<response_to_explain>")[-1]:
            return '{"rationale": ""}'
        return json.dumps({"rationale": "because reasons"})

    gen = RationaleGenerator(PROMPTS, llm(responder=rat))
    own = Exemplar(item_id="i0", text="response number 0", label=ITEMS[0].label, rationale="OWN-RATIONALE")
    other = Exemplar(item_id="zz", text="other", label="Advanced", rationale="OTHER-RATIONALE")
    out = await gen.discriminative(ITEMS[:4], [own, other], round_=2)
    first = next(c for c in contexts if "response number 0" in c.split("<response_to_explain>")[-1])
    assert "OWN-RATIONALE" not in first and "OTHER-RATIONALE" in first
    assert out[3] is None
    assert out[0] is not None and out[0].origin == "phase2" and out[0].round == 2
    plain = await gen.plain(ITEMS[:1], is_expert=True)
    assert plain[0].is_expert and plain[0].origin == "phase0"


async def test_provider_fatal_aborts_instead_of_counting_as_failure() -> None:
    from guide_exemplars.core.protocols import ProviderFatal

    def quota(req: LLMRequest) -> str:
        raise ProviderFatal("insufficient_quota")

    with pytest.raises(ProviderFatal):
        await Grader(PROMPTS, llm(responder=quota)).grade(ITEMS, [])


async def test_burst_of_provider_errors_aborts_but_a_few_count_as_wrong() -> None:
    from guide_exemplars.core.protocols import ProviderUnstable

    def flaky(req: LLMRequest) -> str:
        if "response number 1" in req.messages[-1][1] or "response number 3" in req.messages[-1][1]:
            raise Boom("429 rate limited")
        return json.dumps({"analysis": "x", "label": "Proficient"})

    # 2 provider errors of 6: below the minimum count, so they are scored as wrong
    preds = await Grader(PROMPTS, llm(responder=flaky)).grade(ITEMS, [])
    assert sum(p.pred is None for p in preds) == 2
    # with a lower minimum the same burst stops the evaluation
    with pytest.raises(ProviderUnstable, match="2/6"):
        await Grader(PROMPTS, llm(responder=flaky), min_errors_to_abort=2).grade(ITEMS, [])
    # unparsable replies are the model's answer and never trigger it
    garbage = await Grader(PROMPTS, llm(responder=lambda r: "no label here"), min_errors_to_abort=1).grade(ITEMS, [])
    assert all(p.pred is None for p in garbage)
