"""Grading: score items with a demonstration policy. Results stay aligned to their inputs.

A call that fails (provider error, unparsable reply) becomes a ``Prediction`` with ``pred=None``. It counts as
wrong and never shifts other items. ``BudgetExceeded`` and ``Cancelled`` are not per-item failures: they abort
the whole evaluation. So does a burst of provider errors (``ProviderUnstable``): more than ``max_error_rate`` of
the calls, and at least ``min_errors_to_abort`` of them. Unparsable replies never trigger it; they are the model's
answer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from guide_exemplars.core.metrics import Metrics, metrics_from_predictions
from guide_exemplars.core.prompts import PromptBuilder, parse_grade
from guide_exemplars.core.protocols import (
    LLM,
    AbortRun,
    Cancelled,
    CancelToken,
    LLMRequest,
    NeverCancel,
    ProviderUnstable,
)
from guide_exemplars.core.types import Exemplar, Item, Prediction

DemosFor = Callable[[Item], Sequence[Exemplar]]
ProgressFn = Callable[[int, int], None]

_ABORT = (AbortRun,)


class Grader:
    def __init__(
        self,
        prompts: PromptBuilder,
        llm: LLM,
        max_tokens: int = 200,
        cancel: CancelToken | None = None,
        warmup: bool = True,
        max_error_rate: float | None = 0.02,
        min_errors_to_abort: int = 3,
    ):
        self.prompts = prompts
        self.llm = llm
        self.max_tokens = max_tokens
        self.cancel = cancel or NeverCancel()
        self.warmup = warmup
        self.max_error_rate = max_error_rate
        self.min_errors_to_abort = min_errors_to_abort

    def request(self, demos: Sequence[Exemplar], text: str, purpose: str = "grade") -> LLMRequest:
        return LLMRequest.of(
            self.prompts.grade_system,
            self.prompts.grade_user(demos, text),
            max_tokens=self.max_tokens,
            purpose=purpose,  # type: ignore[arg-type]
        )

    async def _grade_one(self, item: Item, demos: Sequence[Exemplar], purpose: str) -> tuple[Prediction, bool]:
        """The prediction, and whether it failed because of the provider (as opposed to the model's reply)."""
        if self.cancel.is_set():
            raise Cancelled("cancelled before grading call")
        try:
            res = await self.llm.complete(self.request(demos, item.text, purpose))
        except _ABORT:
            raise
        except Exception as exc:
            return Prediction(item.item_id, item.label, None, "", f"{type(exc).__name__}: {exc}"), True
        parsed = parse_grade(res.text, self.prompts.scale)
        return Prediction(item.item_id, item.label, parsed.label, res.text, parsed.error), False

    async def grade(
        self,
        items: Sequence[Item],
        demos: Sequence[Exemplar] | DemosFor,
        on_progress: ProgressFn | None = None,
        purpose: str = "grade",
    ) -> list[Prediction]:
        """Predictions in input order. ``demos`` is a static set or a per-item function (e.g. KNN)."""
        demos_for: DemosFor = demos if callable(demos) else (lambda _item, d=tuple(demos): d)
        total = len(items)
        done = 0
        results: list[Prediction | None] = [None] * total
        provider_errors: list[str] = []

        async def run(i: int) -> None:
            nonlocal done
            pred, provider_failed = await self._grade_one(items[i], demos_for(items[i]), purpose)
            results[i] = pred
            if provider_failed:
                provider_errors.append(pred.error or "")
            done += 1
            if on_progress:
                on_progress(done, total)

        start = 0
        if self.warmup and total and not callable(demos):
            await run(0)  # warm the provider's prefix cache before fanning out
            start = 1
        outcomes = await asyncio.gather(*(run(i) for i in range(start, total)), return_exceptions=True)
        for out in outcomes:
            if isinstance(out, BaseException):
                raise out
        n_err = len(provider_errors)
        if (
            self.max_error_rate is not None
            and n_err >= self.min_errors_to_abort
            and n_err > self.max_error_rate * total
        ):
            raise ProviderUnstable(
                f"{n_err}/{total} grading calls failed at the provider (limit {self.max_error_rate:.0%}); "
                f"last error: {provider_errors[-1]}"
            )
        return [p for p in results if p is not None]

    async def evaluate(
        self,
        items: Sequence[Item],
        demos: Sequence[Exemplar] | DemosFor,
        on_progress: ProgressFn | None = None,
        purpose: str = "grade",
    ) -> tuple[Metrics, list[Prediction]]:
        preds = await self.grade(items, demos, on_progress, purpose)
        return metrics_from_predictions(preds, self.prompts.scale), preds
