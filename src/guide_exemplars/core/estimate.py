"""Up-front cost and time estimate for a GUIDE run (and, optionally, test-set scoring of baselines).

Token sizes are measured on the actual data and rendered prompts. Call counts follow the optimizer exactly.
Three dollar figures come back:

- ``usd_high``: no provider prefix caching at all.
- ``usd_low``: the shared prompt prefix (system + demonstrations) is cached for every call but the first in
  each evaluation.
- ``usd_hard_max``: every reply hits its ``max_tokens`` limit. This is the reservation-based cap's worst case.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from guide_exemplars.core.optimizer import GuideConfig
from guide_exemplars.core.prompts import PromptBuilder
from guide_exemplars.core.protocols import ModelPrice
from guide_exemplars.core.task import TaskSpec
from guide_exemplars.core.types import Exemplar, Item


@dataclass
class CallGroup:
    calls: int
    input_tokens_per_call: float
    output_tokens_per_call: float
    output_tokens_max: int
    cacheable_prefix_tokens: float = 0.0  # tokens shared by consecutive calls in one batch
    batch_size: int = 1  # calls sharing that prefix (first one pays full price)

    def usd(self, price: ModelPrice, cached: bool, worst_case: bool = False) -> float:
        out = self.output_tokens_max if worst_case else self.output_tokens_per_call
        total_in = self.calls * self.input_tokens_per_call
        cached_in = 0.0
        if cached and self.batch_size > 1:
            cached_in = self.calls * (self.batch_size - 1) / self.batch_size * self.cacheable_prefix_tokens
        return (
            (total_in - cached_in) * price.input
            + cached_in * price.cached_input
            + self.calls * out * price.output
        ) / 1_000_000


@dataclass
class Estimate:
    groups: dict[str, CallGroup]
    embed_tokens: float
    usd_low: float
    usd_high: float
    usd_hard_max: float
    total_calls: int
    est_minutes: float
    assumptions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["groups"] = {k: asdict(v) for k, v in self.groups.items()}
        return d


def _mean_tokens(texts: Sequence[str], count) -> float:
    if not texts:
        return 0.0
    sample = list(texts)[:300]
    return float(np.mean([count(t) for t in sample]))


def estimate_run(
    task: TaskSpec,
    train: Sequence[Item],
    val: Sequence[Item],
    cfg: GuideConfig,
    count_tokens,
    price: ModelPrice,
    *,
    n_experts: int = 0,
    test: Sequence[Item] = (),
    n_test_static_policies: int = 0,
    n_test_knn_policies: int = 0,
    include_optimization: bool = True,
    include_phase0: bool = True,
    embed_price: ModelPrice | None = None,
    grading_output_tokens: float = 60.0,
    rationale_output_tokens: float | None = None,
    concurrency: int = 16,
    seconds_per_call: float = 1.5,
    tpm_limit: float | None = None,
) -> Estimate:
    prompts = PromptBuilder(task)
    L = _mean_tokens([it.text for it in train] or [it.text for it in val], count_tokens)
    R = rationale_output_tokens if rationale_output_tokens is not None else 1.25 * task.rationale_max_words
    probe = Exemplar(item_id="probe", text="", label=task.scale.labels[0], rationale="")
    S_g = count_tokens(prompts.grade_system) + count_tokens(prompts.grade_user([probe], ""))
    S_r = count_tokens(prompts.rationale_system) + count_tokens(
        prompts.rationale_user("", task.scale.labels[0], discriminative=True)
    )
    e = L + R + 15  # one rendered demonstration
    k_bar = (cfg.k_min + cfg.k_max) / 2
    prefix = S_g + k_bar * e
    I_g = prefix + L + 20
    V, N, X, T = len(val), len(train), n_experts, cfg.n_rounds

    groups: dict[str, CallGroup] = {}
    if include_optimization:
        grade_calls = T * cfg.n_eval * V - (T - 1) * V if T > 1 else cfg.n_eval * V
        groups["grade"] = CallGroup(
            grade_calls, I_g, grading_output_tokens, cfg.grading_max_tokens, prefix, V
        )
        if include_phase0:
            groups["rationale_plain"] = CallGroup(
                N + X, S_r + X * e + L, R, cfg.rationale_max_tokens, S_r + X * e, max(1, N)
            )
        if cfg.phase2 and T > 1:
            groups["rationale_disc"] = CallGroup(
                (T - 1) * N, S_r + k_bar * e + L, R, cfg.rationale_max_tokens, S_r + k_bar * e, max(1, N)
            )
    Q = len(test)
    if Q and n_test_static_policies:
        groups["test_static"] = CallGroup(
            Q * n_test_static_policies, I_g, grading_output_tokens, cfg.grading_max_tokens, prefix, Q
        )
    if Q and n_test_knn_policies:
        groups["test_knn"] = CallGroup(
            Q * n_test_knn_policies, I_g, grading_output_tokens, cfg.grading_max_tokens
        )

    rounds_embedded = T if include_optimization else 0
    embed_tokens = (N + X) * rounds_embedded * (L + R)
    embed_cost = embed_price.cost(int(embed_tokens), 0, 0) if embed_price else 0.0

    usd_low = sum(g.usd(price, cached=True) for g in groups.values()) + embed_cost
    usd_high = sum(g.usd(price, cached=False) for g in groups.values()) + embed_cost
    usd_max = sum(g.usd(price, cached=False, worst_case=True) for g in groups.values()) + embed_cost
    total_calls = sum(g.calls for g in groups.values())
    minutes = total_calls * seconds_per_call / max(1, concurrency) / 60
    if tpm_limit:
        # Paced runs can't go faster than the provider's tokens-per-minute limit, which counts prompt tokens
        # (cached ones included) plus the requested max_tokens.
        metered = sum(g.calls * (g.input_tokens_per_call + g.output_tokens_max) for g in groups.values())
        minutes = max(minutes, metered / tpm_limit)
    return Estimate(
        groups=groups,
        embed_tokens=embed_tokens,
        usd_low=usd_low,
        usd_high=usd_high,
        usd_hard_max=usd_max,
        total_calls=total_calls,
        est_minutes=minutes,
        assumptions={
            "mean_response_tokens": L,
            "rationale_tokens": R,
            "grade_system_tokens": S_g,
            "rationale_system_tokens": S_r,
            "mean_set_size": k_bar,
            "grade_input_tokens": I_g,
            "grading_output_tokens": grading_output_tokens,
        },
    )
