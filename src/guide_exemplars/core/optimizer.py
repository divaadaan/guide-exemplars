"""GUIDE optimizer: Phase 0 pool construction, then T rounds of contrastive BO selection (Phase 1) and
discriminative rationale refinement (Phase 2).

Correctness details that a straightforward implementation of the paper tends to miss:
- The incumbent is round-local: the lexicographic best of the current round's observations. E*_{t-1} is
  re-seeded into round t, so the operators never perturb a stale random set.
- Each BO step re-scalarizes the full round history with fresh weights and the current Acc*.
- Phase 2 makes one call per training item (not per pool variant), and never shows an item its own rationale.
- Every random choice comes from a named, derived stream, and every phase boundary is checkpointed. A resumed
  run therefore reproduces an uninterrupted one exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from guide_exemplars.core.bo import BayesianSelector, accuracy_of, select_final
from guide_exemplars.core.checkpoint import (
    Checkpoint,
    CheckpointStore,
    observation_from_dict,
    observation_to_dict,
)
from guide_exemplars.core.contrastive import Candidate, ContrastiveConfig, ContrastiveOps, OperatorMix
from guide_exemplars.core.embedding_index import EmbeddingIndex
from guide_exemplars.core.events import Event, EventSink, null_sink
from guide_exemplars.core.featurize import MembershipFeaturizer, SetEmbeddingFeaturizer, SetFeaturizer
from guide_exemplars.core.grader import Grader
from guide_exemplars.core.objective import ObjectiveConfig, objective, sample_weights
from guide_exemplars.core.pool import Pool
from guide_exemplars.core.prompts import PromptBuilder, embedding_text, ordered_demos
from guide_exemplars.core.protocols import LLM, BudgetExceeded, Cancelled, CancelToken, Embedder, NeverCancel
from guide_exemplars.core.rationale import RationaleGenerator
from guide_exemplars.core.rng import RngStreams
from guide_exemplars.core.task import TaskSpec
from guide_exemplars.core.types import Exemplar, Item, Observation, Subset

Status = Literal["completed", "cancelled", "budget_exhausted"]

_LITERAL_OVERRIDES = {
    "adjacent_only": False,
    "contrastive_score": "pair_ratio",
    "featurizer": "membership",
    "protect_incumbent": False,
    "final_selection": "last_round",
}


class GuideConfig(BaseModel):
    """All GUIDE hyperparameters. Defaults follow the paper where it is specific (T=5, 32 evaluations per
    round, |E| in [4, 16], τ=0.7, 256 acquisition candidates, pool cap 512). Elsewhere they use the improved
    choices listed in the README (Fidelity to the paper); ``paper_literal=True`` switches those back."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_rounds: int = Field(5, ge=1)
    n_eval: int = Field(32, ge=1)
    n_init: int = Field(8, ge=1)
    k_min: int = Field(4, ge=1)
    k_max: int = Field(16, ge=1)
    n_candidates: int = Field(256, ge=1)
    pool_cap: int = Field(512, ge=2)
    tau: float = Field(0.7, ge=-1.0, le=1.0)
    adjacent_only: bool = True
    contrastive_score: Literal["coverage", "pair_ratio"] = "coverage"
    featurizer: Literal["set_embedding", "membership"] = "set_embedding"
    pca_components: int = Field(16, ge=1)
    op_add: int = Field(64, ge=0)
    op_swap: int = Field(64, ge=0)
    op_flip: int = Field(64, ge=0)
    phase2: bool = True
    protect_incumbent: bool = True
    final_selection: Literal["global", "last_round"] = "global"
    w1_low: float = 0.25
    w1_high: float = 1.0
    w3_override: float | None = None
    seed: int = 0
    grading_max_tokens: int = Field(200, ge=16)
    rationale_max_tokens: int = Field(250, ge=16)
    # Stop the run (ProviderUnstable) when more than this share of one evaluation's grading calls fail at the
    # provider after retries. None = never stop; failures then count as wrong answers, as in the paper.
    max_provider_error_rate: float | None = Field(0.02, ge=0.0, le=1.0)
    paper_literal: bool = False

    @model_validator(mode="after")
    def _check(self) -> GuideConfig:
        if self.n_init > self.n_eval:
            raise ValueError("n_init cannot exceed n_eval")
        if self.k_min > self.k_max:
            raise ValueError("k_min cannot exceed k_max")
        if self.op_add + self.op_swap + self.op_flip > self.n_candidates:
            raise ValueError("operator quotas exceed n_candidates")
        if not 0 <= self.w1_low <= self.w1_high <= 1:
            raise ValueError("need 0 <= w1_low <= w1_high <= 1")
        return self

    def resolved(self) -> GuideConfig:
        """The effective configuration, with paper-literal overrides applied."""
        return self.model_copy(update=_LITERAL_OVERRIDES) if self.paper_literal else self


@dataclass
class GuideResult:
    status: Status
    exemplars: list[Exemplar]
    best: Observation | None
    observations: list[Observation]
    round_bests: dict[int, Observation]
    pool: Pool | None
    config: GuideConfig
    notes: list[str] = field(default_factory=list)

    @property
    def subset(self) -> Subset | None:
        return self.best.subset if self.best else None


class _Interrupted(Exception):
    def __init__(self, status: Status):
        self.status = status


class GuideOptimizer:
    def __init__(
        self,
        task: TaskSpec,
        train: Sequence[Item],
        val: Sequence[Item],
        llm: LLM,
        embedder: Embedder,
        cfg: GuideConfig | None = None,
        *,
        experts: Sequence[Item | Exemplar] = (),
        events: EventSink = null_sink,
        checkpoints: CheckpointStore | None = None,
        cancel: CancelToken | None = None,
    ):
        self.task = task
        self.cfg = (cfg or GuideConfig()).resolved()
        self.scale = task.scale
        for it in [*train, *val, *experts]:
            self.scale.rank(it.label)  # raises on labels not on the scale
        expert_ids = {e.item_id for e in experts}
        self.train = [it for it in train if it.item_id not in expert_ids]
        self.val = list(val)
        if not self.val:
            raise ValueError("validation set is empty")
        if len(self.train) + len(experts) < self.cfg.k_min:
            raise ValueError("training pool is smaller than k_min")
        self.experts = list(experts)
        self.embedder = embedder
        self.events = events
        self.checkpoints = checkpoints
        self.cancel = cancel or NeverCancel()
        self.prompts = PromptBuilder(task)
        self.grader = Grader(
            self.prompts,
            llm,
            self.cfg.grading_max_tokens,
            self.cancel,
            max_error_rate=self.cfg.max_provider_error_rate,
        )
        self.rationales = RationaleGenerator(self.prompts, llm, self.cfg.rationale_max_tokens, self.cancel)
        self.rng = RngStreams(self.cfg.seed)
        self.objective_cfg = ObjectiveConfig(
            w1_range=(self.cfg.w1_low, self.cfg.w1_high), w3_override=self.cfg.w3_override
        )
        self.mix = OperatorMix(self.cfg.op_add, self.cfg.op_swap, self.cfg.op_flip)
        self.contrastive_cfg = ContrastiveConfig(
            tau=self.cfg.tau, adjacent_only=self.cfg.adjacent_only, score=self.cfg.contrastive_score
        )

        self._registry: dict[str, Exemplar] = {}
        self._observations: list[Observation] = []
        self._round_bests: dict[int, Subset] = {}
        self._pool: Pool | None = None

    # ---- events / checkpoints -------------------------------------------------------------------

    def _emit(self, type_: str, round_: int = 0, ephemeral: bool = False, **payload: object) -> None:
        self.events(Event(type_, round_, dict(payload), ephemeral))

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise Cancelled("cancel requested")

    def _remember(self, exemplars: Sequence[Exemplar]) -> None:
        for e in exemplars:
            self._registry.setdefault(e.uid, e)

    def _checkpoint(self, round_: int, phase: str) -> None:
        if self.checkpoints is None or self._pool is None:
            return
        self.checkpoints.save(
            Checkpoint(
                round=round_,
                phase=phase,  # type: ignore[arg-type]
                pool=self._pool.to_records(),
                observations=[observation_to_dict(o) for o in self._observations],
                registry=[e.model_dump(mode="json", exclude={"uid"}) for e in self._registry.values()],
                round_bests={str(r): sorted(s.uids) for r, s in self._round_bests.items()},
            )
        )
        self._emit("checkpoint_saved", round_, phase=phase)

    def _restore(self, cp: Checkpoint) -> None:
        self._pool = Pool.from_records(cp.pool)
        self._registry = cp.registry_exemplars()
        self._remember(self._pool.exemplars())
        self._observations = [observation_from_dict(d) for d in cp.observations]
        self._round_bests = {int(r): Subset.of(u) for r, u in cp.round_bests.items()}

    # ---- phase 0 --------------------------------------------------------------------------------

    async def build_pool(self) -> Pool:
        """Phase 0: expert exemplars (given or generated without context), then a plain rationale for every
        other training item, conditioned on the experts."""
        self._emit("phase_started", 0, phase="phase0", n_items=len(self.train) + len(self.experts))
        given = [e for e in self.experts if isinstance(e, Exemplar) and e.rationale]
        to_generate = [e for e in self.experts if not (isinstance(e, Exemplar) and e.rationale)]
        experts = [e.model_copy(update={"is_expert": True, "origin": "given"}) for e in given]
        generated = await self.rationales.plain(
            [Item(item_id=e.item_id, text=e.text, label=e.label) for e in to_generate], is_expert=True
        )
        experts += [e for e in generated if e is not None]
        others = await self.rationales.plain(self.train, context=experts)
        kept = [e for e in others if e is not None]
        pool = Pool([*experts, *kept])
        self._remember(pool.exemplars())
        failed = (len(to_generate) - sum(e is not None for e in generated)) + (len(self.train) - len(kept))
        self._emit("phase0_completed", 0, pool_size=len(pool), n_experts=len(experts), n_failed=failed)
        return pool

    # ---- main loop ------------------------------------------------------------------------------

    async def run(self, resume: Checkpoint | None = None, phase0_pool: Pool | None = None) -> GuideResult:
        cfg = self.cfg
        status: Status = "completed"
        notes: list[str] = []
        try:
            if resume is not None:
                self._restore(resume)
                t = resume.round if resume.phase == "phase1" else resume.round + 1
                step = "phase2" if resume.phase == "phase1" else "phase1"
                if resume.phase == "phase0":
                    t, step = 1, "phase1"
                self._emit("resumed", resume.round, from_phase=resume.phase)
            else:
                if phase0_pool is not None:
                    self._pool = Pool(phase0_pool.exemplars())
                    self._remember(self._pool.exemplars())
                else:
                    self._pool = await self.build_pool()
                self._checkpoint(0, "phase0")
                t, step = 1, "phase1"

            while t <= cfg.n_rounds:
                if step == "phase1":
                    best = await self._selection_round(t)
                    self._round_bests[t] = best.subset
                    self._checkpoint(t, "phase1")
                    step = "phase2"
                if t < cfg.n_rounds and cfg.phase2:
                    await self._refine(t)
                    self._checkpoint(t, "phase2")
                t, step = t + 1, "phase1"
        except Cancelled:
            status = "cancelled"
            notes.append("cancelled; result is the best fully evaluated set so far")
        except BudgetExceeded as exc:
            status = "budget_exhausted"
            notes.append(f"budget exhausted ({exc}); result is the best fully evaluated set so far")
        return self._finalize(status, notes)

    def _finalize(self, status: Status, notes: list[str]) -> GuideResult:
        obs = self._observations
        round_bests = {
            r: select_final([o for o in obs if o.round == r]) for r in sorted({o.round for o in obs})
        }
        best: Observation | None = None
        if obs:
            best = (
                select_final(obs) if self.cfg.final_selection == "global" else round_bests[max(round_bests)]
            )
        exemplars = ordered_demos([self._registry[u] for u in best.subset]) if best else []
        self._emit(
            "optimizer_finished",
            best.round if best else 0,
            status=status,
            acc=float(accuracy_of(best)) if best else None,
            size=best.size if best else 0,
            n_observations=len(obs),
        )
        return GuideResult(status, exemplars, best, list(obs), round_bests, self._pool, self.cfg, notes)

    # ---- phase 1 --------------------------------------------------------------------------------

    def _expert_subset(self, pool: Pool, round_: int) -> Subset | None:
        experts = [e.uid for e in pool.experts()]
        if not experts:
            return None
        rng = self.rng.get("expert", round_)
        if len(experts) > self.cfg.k_max:
            experts = [experts[i] for i in sorted(rng.choice(len(experts), self.cfg.k_max, replace=False))]
        if len(experts) < self.cfg.k_min:
            taken = {pool.get(u).item_id for u in experts}
            pad = pool.random_subset(rng, self.cfg.k_min - len(experts), self.cfg.k_min - len(experts), taken)
            experts += list(pad)
        return Subset.of(experts)

    async def _index(self, pool: Pool) -> EmbeddingIndex:
        exemplars = pool.exemplars()
        vectors = await self.embedder.embed([embedding_text(e) for e in exemplars])
        return EmbeddingIndex([e.uid for e in exemplars], vectors)

    def _featurizer(self) -> SetFeaturizer:
        if self.cfg.featurizer == "membership":
            return MembershipFeaturizer()
        return SetEmbeddingFeaturizer(self.cfg.pca_components)

    async def _selection_round(self, t: int) -> Observation:
        cfg = self.cfg
        pool = self._pool
        assert pool is not None
        self._emit("phase_started", t, phase="phase1")
        index = await self._index(pool)
        ops = ContrastiveOps(pool, index, self.scale, self.contrastive_cfg, cfg.k_min, cfg.k_max)
        featurizer = self._featurizer()
        featurizer.fit(pool, index, self.scale, cfg.k_max)
        selector = BayesianSelector(featurizer, random_state=self.rng.int_seed("gp", t))

        sims = index.sim_matrix()[np.triu_indices(len(index), k=1)] if len(index) > 1 else np.zeros(1)
        self._emit(
            "pool_ready",
            t,
            size=len(pool),
            n_items=len(pool.item_ids()),
            n_boundary_pairs=len(ops.boundary_pairs()),
            sim_quantiles=[round(float(q), 4) for q in np.quantile(sims, [0.1, 0.25, 0.5, 0.75, 0.9])],
            frac_above_tau=round(float((sims >= cfg.tau).mean()), 4),
        )

        round_obs: list[Observation] = []
        evaluated: set[str] = set()
        expert = self._expert_subset(pool, t)

        inits: list[Candidate] = []
        previous = self._round_bests.get(t - 1)
        if previous is not None and pool.is_valid(previous, cfg.k_min, cfg.k_max):
            inits.append(Candidate(previous, "incumbent"))
        if expert is not None and expert.key() not in {c.subset.key() for c in inits}:
            inits.append(Candidate(expert, "expert"))
        attempt = 0
        while len(inits) < min(cfg.n_init, cfg.n_eval) and attempt < 50 * cfg.n_init:
            s = pool.random_subset(self.rng.get("init", t, attempt), cfg.k_min, cfg.k_max)
            attempt += 1
            if s.key() not in {c.subset.key() for c in inits}:
                inits.append(Candidate(s, "init"))

        for cand in inits:
            await self._evaluate(t, cand, None, ops, selector, round_obs, evaluated)

        for j in range(cfg.n_eval - len(inits)):
            self._check_cancel()
            incumbent = select_final(round_obs)
            weights = sample_weights(self.rng.get("w", t, j), self.objective_cfg)
            cands = ops.generate_candidates(
                incumbent.subset,
                expert,
                evaluated,
                self.rng.get("ops", t, j),
                n=cfg.n_candidates,
                mix=self.mix,
            )
            if not cands:
                self._emit("search_exhausted", t, step=j)
                break
            pick, ei = selector.suggest(cands, weights, self.rng.get("ei", t, j))
            await self._evaluate(t, pick, ei, ops, selector, round_obs, evaluated, weights)

        best = select_final(round_obs)
        self._emit(
            "round_completed",
            t,
            best_uids=sorted(best.subset.uids),
            acc=float(accuracy_of(best)),
            qwk=None if np.isnan(best.metrics.qwk) else best.metrics.qwk,
            size=best.size,
            contrastive=best.contrastive,
            n_evaluated=len(round_obs),
        )
        return best

    async def _evaluate(
        self,
        t: int,
        cand: Candidate,
        ei: float | None,
        ops: ContrastiveOps,
        selector: BayesianSelector,
        round_obs: list[Observation],
        evaluated: set[str],
        weights=None,
    ) -> Observation:
        assert self._pool is not None
        self._check_cancel()
        index = len(self._observations)
        demos = self._pool.resolve(cand.subset)
        step = max(1, len(self.val) // 10)

        def progress(done: int, total: int) -> None:
            if done == total or done % step == 0:
                self._emit("eval_progress", t, ephemeral=True, index=index, done=done, total=total)

        metrics, _preds = await self.grader.evaluate(self.val, demos, on_progress=progress)
        obs = Observation(
            round=t,
            index=index,
            subset=cand.subset,
            op=cand.op,
            metrics=metrics,
            contrastive=ops.score(cand.subset),
            ei=ei,
        )
        self._observations.append(obs)
        round_obs.append(obs)
        selector.observe(obs)
        evaluated.add(cand.subset.key())
        round_best = select_final(round_obs)
        payload: dict[str, object] = {
            "index": index,
            "op": cand.op,
            "size": obs.size,
            "acc": metrics.acc,
            "qwk": None if np.isnan(metrics.qwk) else metrics.qwk,
            "adj_err": metrics.adj_err,
            "nonadj_err": metrics.nonadj_err,
            "n_failed": metrics.n_failed,
            "contrastive": obs.contrastive,
            "ei": None if ei is None or np.isnan(ei) else ei,
            "best_acc_round": float(accuracy_of(round_best)),
            "best_acc_global": float(max(accuracy_of(o) for o in self._observations)),
        }
        if weights is not None:
            payload["w1"] = weights.w1
            payload["objective"] = objective(
                metrics.acc, float(selector.acc_star()), obs.size, obs.contrastive, weights
            )
        self._emit("eval_completed", t, **payload)
        return obs

    # ---- phase 2 --------------------------------------------------------------------------------

    async def _refine(self, t: int) -> None:
        pool = self._pool
        assert pool is not None
        self._emit("phase_started", t, phase="phase2")
        best_uids = set(self._round_bests[t].uids)
        context = [self._registry[u] for u in sorted(best_uids)]
        generated = await self.rationales.discriminative(self.train, context, t)
        fresh = [e for e in generated if e is not None]
        self._remember(fresh)
        added = pool.merge(fresh)
        protect = best_uids if self.cfg.protect_incumbent else set()
        evicted = pool.cap(self.cfg.pool_cap, protect, self.rng.get("cap", t))
        self._emit(
            "phase2_completed",
            t,
            generated=len(fresh),
            failed=len(self.train) - len(fresh),
            added=added,
            evicted=len(evicted),
            pool_size=len(pool),
        )
