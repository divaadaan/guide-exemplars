"""Deterministic fake chat backend for tests and offline demos.

Every behaviour (the response, injected failures, latency jitter) is a function of a hash of the request. Runs
are therefore reproducible regardless of concurrency or call order.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from guide_exemplars.core.prompts import TAG_EXPLAIN, TAG_TARGET
from guide_exemplars.core.protocols import LLMRequest
from guide_exemplars.core.scale import LabelScale
from guide_exemplars.llm.backends import BackendResponse

Responder = Callable[[LLMRequest], str]


class FakeBackendError(RuntimeError):
    pass


def request_hash(req: LLMRequest, salt: str = "") -> int:
    payload = json.dumps([salt, req.messages, req.max_tokens, req.purpose], ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _unit(h: int) -> float:
    return (h % 1_000_003) / 1_000_003


def constant_responder(text: str = '{"ok": true}') -> Responder:
    return lambda _req: text


def _between(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return m.group(1) if m else None


class GradingSimulator:
    """Responder that behaves like a grader whose accuracy depends on the demonstrations.

    For each scored response, the chance of returning the gold label is
    ``base + label_bonus·[gold label shown] + neighbour_bonus·[an adjacent label shown] +
    disc_bonus·(share of discriminative rationales)``. Otherwise it is off by one level. Label coverage and
    boundary contrasts therefore really pay off, which gives the optimizer a signal to climb in offline runs.
    Rationale requests get a deterministic rationale that echoes the response's opening words.
    """

    def __init__(
        self,
        scale: LabelScale,
        gold_by_text: Mapping[str, str],
        base: float = 0.35,
        label_bonus: float = 0.2,
        neighbour_bonus: float = 0.25,
        disc_bonus: float = 0.15,
    ):
        self.scale = scale
        self.gold = {" ".join(k.split()): v for k, v in gold_by_text.items()}
        self.base, self.label_bonus = base, label_bonus
        self.neighbour_bonus, self.disc_bonus = neighbour_bonus, disc_bonus

    def __call__(self, req: LLMRequest) -> str:
        user = req.messages[-1][1]
        if req.purpose.startswith("rationale"):
            return self._rationale(req, user)
        if req.purpose == "features":
            return self._features(req, user)
        target = _between(user, TAG_TARGET) or ""
        gold = self.gold.get(" ".join(target.split()))
        if gold is None:
            return json.dumps({"analysis": "unknown response", "label": self.scale.labels[0]})
        shown = re.findall(r"<example_label>(.*?)</example_label>", user)
        rationales = re.findall(r"<example_rationale>(.*?)</example_rationale>", user, re.DOTALL)
        disc_share = (sum("not" in r.split() for r in rationales) / len(rationales)) if rationales else 0.0
        p = (
            self.base
            + self.label_bonus * (gold in shown)
            + self.neighbour_bonus * any(n in shown for n in self.scale.neighbours(gold))
            + self.disc_bonus * disc_share
        )
        h = request_hash(req, "grade")
        if _unit(h) < p:
            label = gold
        else:
            r = self.scale.rank(gold)
            step = 1 if (h >> 20) & 1 else -1
            if not 0 <= r + step < len(self.scale):
                step = -step
            label = self.scale.label_at(r + step)
        return json.dumps({"analysis": "simulated grader", "label": label})

    def _features(self, req: LLMRequest, user: str) -> str:
        """Counts that fall with the gold level (plus deterministic noise), so a feature grader has signal."""
        names = re.findall(r'"([a-z][a-z0-9_]*)": <count>', req.messages[0][1])
        target = _between(user, TAG_TARGET) or ""
        gold = self.gold.get(" ".join(target.split()))
        rank = self.scale.rank(gold) if gold is not None else (len(self.scale) - 1) / 2
        gap = len(self.scale) - 1 - rank
        words = max(1, len(target.split()))
        counts = {}
        for i, name in enumerate(names):
            rate = gap * (1.5 + i) + 2.0 * _unit(request_hash(req, f"features-{name}"))
            counts[name] = round(rate * words / 100)
        return json.dumps(counts)

    def _rationale(self, req: LLMRequest, user: str) -> str:
        text = _between(user, TAG_EXPLAIN) or ""
        m = re.search(r"was scored (.+?) by expert raters", user)
        label = m.group(1) if m else "?"
        opening = " ".join(text.split()[:8])
        tag = request_hash(req, "rationale") % 10_000
        if req.purpose == "rationale_disc":
            neighbours = " or ".join(self.scale.neighbours(label)) if label in self.scale else "?"
            body = f"Deserves {label} and not {neighbours}: {opening} (note {tag})."
        else:
            body = f"Earns {label}: {opening} (note {tag})."
        return json.dumps({"rationale": body})


@dataclass
class FakeBackend:
    responder: Responder = field(default_factory=constant_responder)
    failure_rate: float = 0.0
    jitter_s: float = 0.0
    calls: int = 0

    async def complete(self, req: LLMRequest, temperature: float) -> BackendResponse:
        self.calls += 1
        h = request_hash(req)
        if self.jitter_s:
            await asyncio.sleep(self.jitter_s * _unit(request_hash(req, "jitter")))
        if self.failure_rate and _unit(request_hash(req, "fail")) < self.failure_rate:
            raise FakeBackendError(f"injected failure ({h:x})")
        text = self.responder(req)
        prompt_tokens = sum(len(c) for _, c in req.messages) // 4 + 4 * len(req.messages)
        return BackendResponse(text, prompt_tokens, 0, max(1, len(text) // 4))
