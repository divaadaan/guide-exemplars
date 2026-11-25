"""End-to-end optimizer runs against the simulated grader. Fully offline and deterministic."""

from __future__ import annotations

import pytest

from guide_exemplars.core.bo import accuracy_of
from guide_exemplars.core.checkpoint import FileCheckpointStore, MemoryCheckpointStore
from guide_exemplars.core.events import Event, ListSink
from guide_exemplars.core.optimizer import GuideConfig, GuideOptimizer
from guide_exemplars.llm.budget import Budget
from guide_exemplars.llm.cache import MemoryResponseCache
from support.toy import simulator_llm, toy_embedder, toy_items, toy_split, toy_task

ITEMS = toy_items()
TRAIN, VAL, EXPERTS = toy_split(ITEMS)
SMALL = GuideConfig(
    n_rounds=2,
    n_eval=8,
    n_init=4,
    k_min=3,
    k_max=6,
    n_candidates=32,
    op_add=8,
    op_swap=8,
    op_flip=8,
    tau=0.3,
    seed=7,
)


def optimizer(cfg: GuideConfig = SMALL, llm=None, **kw) -> GuideOptimizer:
    return GuideOptimizer(
        toy_task(),
        TRAIN,
        VAL,
        llm or simulator_llm(ITEMS),
        toy_embedder(),
        cfg,
        experts=EXPERTS,
        **kw,
    )


def trace(result) -> list[tuple]:
    return [(o.round, o.index, o.op, tuple(sorted(o.subset.uids)), o.n_correct) for o in result.observations]


async def test_end_to_end_run_is_well_formed() -> None:
    sink = ListSink()
    result = await optimizer(events=sink).run()
    assert result.status == "completed"
    assert len(result.observations) == SMALL.n_rounds * SMALL.n_eval
    assert result.pool is not None
    assert result.pool.is_valid(result.subset, SMALL.k_min, SMALL.k_max)
    assert len(result.exemplars) == result.best.size

    round1 = [o for o in result.observations if o.round == 1]
    init_best = max(accuracy_of(o) for o in round1[: SMALL.n_init])
    assert accuracy_of(result.best) >= init_best

    round2 = [o for o in result.observations if o.round == 2]
    assert round2[0].op == "incumbent"
    assert round2[0].subset == result.round_bests[1].subset
    assert any(o.op == "expert" for o in round1)

    types = [e.type for e in sink.events if not e.ephemeral]
    assert types[0] == "phase_started" and types[-1] == "optimizer_finished"
    assert types.count("eval_completed") == len(result.observations)
    assert types.count("round_completed") == 2 and types.count("phase2_completed") == 1
    assert any(e.ephemeral and e.type == "eval_progress" for e in sink.events)
    assert any(e.origin == "phase2" for e in result.pool)


async def test_incumbent_reevaluation_is_free() -> None:
    llm = simulator_llm(ITEMS)
    await optimizer(llm=llm).run()
    by_round_hits = llm.meter.by_purpose["grade"].cache_hits
    assert by_round_hits >= len(VAL)  # round 2 re-scores E*_1 entirely from cache


async def test_determinism_under_concurrency_jitter() -> None:
    a = await optimizer(llm=simulator_llm(ITEMS, jitter_s=0.002, max_concurrency=8)).run()
    b = await optimizer(llm=simulator_llm(ITEMS, jitter_s=0.002, max_concurrency=3)).run()
    assert trace(a) == trace(b)
    assert [e.uid for e in a.exemplars] == [e.uid for e in b.exemplars]
    c = await optimizer(SMALL.model_copy(update={"seed": 8})).run()
    assert trace(c) != trace(a)


async def test_resume_after_crash_matches_uninterrupted_run(tmp_path) -> None:
    reference = await optimizer().run()

    class CrashAtRound2:
        def __call__(self, e: Event) -> None:
            if e.type == "phase_started" and e.round == 2 and e.payload.get("phase") == "phase1":
                raise RuntimeError("simulated crash")

    store = FileCheckpointStore(tmp_path / "ckpt")
    cache = MemoryResponseCache()
    with pytest.raises(RuntimeError, match="simulated crash"):
        await optimizer(
            llm=simulator_llm(ITEMS, cache=cache), events=CrashAtRound2(), checkpoints=store
        ).run()
    latest = store.latest()
    assert latest is not None and (latest.round, latest.phase) == (1, "phase2")

    llm = simulator_llm(ITEMS, cache=cache)
    resumed = await optimizer(llm=llm, checkpoints=store).run(resume=latest)
    assert trace(resumed) == trace(reference)
    assert [e.uid for e in resumed.exemplars] == [e.uid for e in reference.exemplars]
    assert llm.meter.by_purpose.get("rationale_plain") is None  # phase 0 was not redone


async def test_budget_exhaustion_returns_best_so_far() -> None:
    probe = simulator_llm(ITEMS)
    await optimizer(llm=probe).run()
    full_cost = probe.meter.total.cost_usd
    budget = Budget(max_usd=full_cost * 0.6)
    result = await optimizer(llm=simulator_llm(ITEMS, budget=budget)).run()
    assert result.status == "budget_exhausted"
    assert budget.spent <= budget.max_usd
    assert result.best is not None and result.exemplars
    assert 0 < len(result.observations) < SMALL.n_rounds * SMALL.n_eval


async def test_cancel_mid_run() -> None:
    class CancelAfterEvals:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, e: Event) -> None:
            if e.type == "eval_completed":
                self.n += 1

        def is_set(self) -> bool:
            return self.n >= 3

    token = CancelAfterEvals()
    result = await optimizer(events=token, cancel=token).run()
    assert result.status == "cancelled"
    assert len(result.observations) == 3 and result.best is not None


async def test_paper_literal_mode() -> None:
    cfg = SMALL.model_copy(update={"paper_literal": True})
    opt = optimizer(cfg)
    assert opt.cfg.featurizer == "membership" and opt.cfg.contrastive_score == "pair_ratio"
    result = await opt.run()
    assert result.status == "completed"
    assert result.best.round == 2  # final pick is E*_T


async def test_survives_provider_failures() -> None:
    sink = ListSink()
    tolerant = SMALL.model_copy(update={"max_provider_error_rate": None})
    result = await optimizer(tolerant, llm=simulator_llm(ITEMS, failure_rate=0.1), events=sink).run()
    assert result.status == "completed"
    assert any(e.payload.get("n_failed", 0) > 0 for e in sink.of_type("eval_completed"))


async def test_burst_of_provider_failures_stops_the_run() -> None:
    from guide_exemplars.core.protocols import ProviderUnstable

    with pytest.raises(ProviderUnstable, match="failed at the provider"):
        await optimizer(llm=simulator_llm(ITEMS, failure_rate=0.3)).run()


async def test_shared_phase0_pool_skips_rationale_generation() -> None:
    first = optimizer()
    pool = await first.build_pool()
    llm = simulator_llm(ITEMS)
    result = await optimizer(llm=llm).run(phase0_pool=pool)
    assert result.status == "completed"
    assert "rationale_plain" not in llm.meter.by_purpose


async def test_checkpoints_written_at_each_phase() -> None:
    store = MemoryCheckpointStore()
    await optimizer(checkpoints=store).run()
    assert [(c.round, c.phase) for c in store.saved] == [
        (0, "phase0"),
        (1, "phase1"),
        (1, "phase2"),
        (2, "phase1"),
    ]
