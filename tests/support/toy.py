"""A small synthetic grading task with named labels, plus a simulated-grader LLM, for offline end-to-end tests."""

from __future__ import annotations

import numpy as np

from guide_exemplars.core.task import TaskSpec
from guide_exemplars.core.types import Item
from guide_exemplars.llm.budget import Budget, UsageMeter
from guide_exemplars.llm.cache import MemoryResponseCache, ResponseCache
from guide_exemplars.llm.client import LLMClient
from guide_exemplars.llm.embedders import HashEmbedder
from guide_exemplars.llm.fake import FakeBackend, GradingSimulator
from guide_exemplars.llm.pricing import PriceTable

LABELS = ("Beginning", "Developing", "Proficient", "Advanced")
_VOCAB = {
    "Beginning": ["unclear", "guess", "maybe", "thing", "stuff", "random", "wrong", "idea"],
    "Developing": ["partial", "some", "reason", "because", "example", "basic", "attempt", "link"],
    "Proficient": ["clear", "evidence", "explains", "supports", "connects", "accurate", "detail", "cause"],
    "Advanced": [
        "precise",
        "synthesis",
        "evaluates",
        "nuanced",
        "justified",
        "insight",
        "models",
        "rigorous",
    ],
}
_SHARED = [
    "the",
    "answer",
    "response",
    "student",
    "question",
    "shows",
    "this",
    "that",
    "with",
    "and",
    "about",
    "from",
    "data",
    "result",
]


def toy_task() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "name": "toy-named",
            "labels": [{"value": lbl, "description": f"{lbl} level work"} for lbl in LABELS],
            "persona": "You are an assessor.",
            "instruction": "Score the response against the rubric.",
            "rubric": "Higher levels show clearer, better-supported reasoning.",
            "rationale_focus": "Cite the reasoning features that justify the level.",
            "rationale_max_words": 40,
        }
    )


def toy_items(n_per_label: int = 12, seed: int = 0) -> list[Item]:
    rng = np.random.default_rng(seed)
    items = []
    for r, label in enumerate(LABELS):
        neighbours = [LABELS[i] for i in (r - 1, r + 1) if 0 <= i < len(LABELS)]
        for k in range(n_per_label):
            words = list(rng.choice(_VOCAB[label], 5)) + list(rng.choice(_SHARED, 6))
            for nb in neighbours:
                words += list(rng.choice(_VOCAB[nb], 2))
            rng.shuffle(words)
            items.append(Item(item_id=f"{label[:3].lower()}{k:02d}", text=" ".join(words), label=label))
    return items


def toy_split(items: list[Item], seed: int = 0) -> tuple[list[Item], list[Item], list[Item]]:
    """(train, val, experts): one expert per label; validation is every third remaining item."""
    rng = np.random.default_rng(seed)
    experts, rest = [], []
    for label in LABELS:
        group = [it for it in items if it.label == label]
        pick = int(rng.integers(len(group)))
        experts.append(group[pick])
        rest += [it for i, it in enumerate(group) if i != pick]
    val = rest[::3]
    train = [it for it in rest if it not in val]
    return train, val, experts


def simulator_llm(
    items: list[Item],
    *,
    jitter_s: float = 0.0,
    failure_rate: float = 0.0,
    budget: Budget | None = None,
    cache: ResponseCache | None = None,
    max_concurrency: int = 8,
) -> LLMClient:
    task = toy_task()
    backend = FakeBackend(
        responder=GradingSimulator(task.scale, {it.text: it.label for it in items}),
        failure_rate=failure_rate,
        jitter_s=jitter_s,
    )
    return LLMClient(
        backend,
        "fake/grader",
        price=PriceTable.load().lookup("fake/grader"),
        cache=cache if cache is not None else MemoryResponseCache(),
        meter=UsageMeter(),
        budget=budget,
        max_concurrency=max_concurrency,
    )


def toy_embedder() -> HashEmbedder:
    return HashEmbedder(dim=256)
