"""Rationale generation: plain (Phase 0) and discriminative "why y and not y±1" (Phase 2).

Rationales are teacher-forced: the true label is given and the model only explains it. In Phase 2 the context
set E*_t never includes the item being explained, so the model cannot copy its own earlier rationale.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from guide_exemplars.core.prompts import PromptBuilder, parse_rationale
from guide_exemplars.core.protocols import (
    LLM,
    AbortRun,
    Cancelled,
    CancelToken,
    LLMRequest,
    NeverCancel,
)
from guide_exemplars.core.types import Exemplar, Item, Origin

_ABORT = (AbortRun,)


class RationaleGenerator:
    def __init__(
        self,
        prompts: PromptBuilder,
        llm: LLM,
        max_tokens: int = 250,
        cancel: CancelToken | None = None,
    ):
        self.prompts = prompts
        self.llm = llm
        self.max_tokens = max_tokens
        self.cancel = cancel or NeverCancel()

    def request(self, item: Item, context: Sequence[Exemplar], discriminative: bool) -> LLMRequest:
        return LLMRequest.of(
            self.prompts.rationale_system,
            self.prompts.rationale_user(item.text, item.label, context, discriminative=discriminative),
            max_tokens=self.max_tokens,
            purpose="rationale_disc" if discriminative else "rationale_plain",
        )

    async def _one(
        self,
        item: Item,
        context: Sequence[Exemplar],
        discriminative: bool,
        origin: Origin,
        round_: int,
        is_expert: bool,
    ) -> Exemplar | None:
        if self.cancel.is_set():
            raise Cancelled("cancelled before rationale call")
        context = [c for c in context if c.item_id != item.item_id]
        try:
            res = await self.llm.complete(self.request(item, context, discriminative))
        except _ABORT:
            raise
        except Exception:
            return None
        rationale = parse_rationale(res.text)
        if rationale is None:
            return None
        return Exemplar.from_item(item, rationale, origin=origin, round=round_, is_expert=is_expert)

    async def _many(
        self,
        items: Sequence[Item],
        context: Sequence[Exemplar],
        discriminative: bool,
        origin: Origin,
        round_: int,
        is_expert: bool,
    ) -> list[Exemplar | None]:
        outcomes = await asyncio.gather(
            *(self._one(i, context, discriminative, origin, round_, is_expert) for i in items),
            return_exceptions=True,
        )
        for out in outcomes:
            if isinstance(out, BaseException):
                raise out
        return list(outcomes)  # type: ignore[arg-type]

    async def plain(
        self,
        items: Sequence[Item],
        context: Sequence[Exemplar] = (),
        is_expert: bool = False,
    ) -> list[Exemplar | None]:
        return await self._many(items, context, False, "phase0", 0, is_expert)

    async def discriminative(
        self, items: Sequence[Item], context: Sequence[Exemplar], round_: int
    ) -> list[Exemplar | None]:
        return await self._many(items, context, True, "phase2", round_, False)
