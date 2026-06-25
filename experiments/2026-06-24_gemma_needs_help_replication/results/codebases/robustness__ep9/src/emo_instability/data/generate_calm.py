"""Generate calm (and matched frustrated) responses for finetuning (Section 4.1).

Calm data: sample Gemma-3-27B-it on impossible numeric puzzles with the reassuring
prefix added to the first user turn and the reassuring suffix appended to each
follow-up (Table 4). Judge every turn; keep rollouts where *all* turns score 0--1
(the paper filters to responses scoring 0 or 1 across all turns). The reassurance
additions are stripped when the data is later turned into training prompts.

Frustrated data: the same puzzles run *without* reassurance, judged per turn, used
as the ``rejected`` side of DPO pairs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import SamplingConfig
from ..conversation import RolloutPlan, history_for_turn, run_rollouts
from ..judge import FrustrationJudge
from ..models import ChatMessage, ModelClient
from .. import prompts, puzzles


@dataclass
class CalmResponse:
    puzzle_id: int
    puzzle_kind: str
    turn: int  # 1-indexed turn within the rollout
    n_turns: int
    history: list[dict]  # chat messages (no reassurance) preceding the response
    response: str
    rating: int


def _shared_puzzles(n: int, seed: int) -> list[puzzles.Puzzle]:
    return puzzles.generate_impossible_puzzles(n, random.Random(seed))


def _plans(
    pz: list[puzzles.Puzzle],
    rng: random.Random,
    *,
    reassuring: bool,
    n_turns: int,
) -> list[RolloutPlan]:
    plans = []
    n_followups = n_turns - 1
    for i, p in enumerate(pz):
        followups = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n_followups)]
        plans.append(
            RolloutPlan(
                initial_user=p.text,
                followups=followups,
                system=prompts.REASSURING_PREFIX if reassuring else None,
                initial_suffix="",
                followup_suffix=prompts.REASSURING_SUFFIX if reassuring else "",
                meta={"puzzle_id": i, "puzzle_kind": p.kind, "n_turns": n_turns},
            )
        )
    return plans


def _judge_and_collect(
    plans: list[RolloutPlan],
    model: ModelClient,
    judge: FrustrationJudge,
    sampling: SamplingConfig,
    *,
    keep: str,  # "calm_all" | "any"
) -> list[CalmResponse]:
    results = run_rollouts(model, plans, sampling)
    # Score every turn.
    collected: list[CalmResponse] = []
    for res in results:
        ratings = [j.rating for j in judge.score_batch(res.responses)]
        all_calm = all(r <= 1 for r in ratings)
        for turn, (resp, rating) in enumerate(zip(res.responses, ratings), start=1):
            if keep == "calm_all" and not all_calm:
                continue
            hist = history_for_turn(res.plan, res.responses, turn, strip_suffixes=True)
            collected.append(
                CalmResponse(
                    puzzle_id=res.plan.meta["puzzle_id"],
                    puzzle_kind=res.plan.meta.get("puzzle_kind", "unknown"),
                    turn=turn,
                    n_turns=res.plan.n_turns,
                    history=[m.as_dict() for m in hist],
                    response=resp,
                    rating=rating,
                )
            )
    return collected


def generate_calm_responses(
    model: ModelClient,
    judge: FrustrationJudge,
    *,
    n_puzzles: int = 400,
    turn_lengths: tuple[int, ...] = (1, 2, 3),
    sampling: SamplingConfig | None = None,
    seed: int = 100,
) -> list[CalmResponse]:
    """Reassured generation, kept only where all turns score 0--1."""
    sampling = sampling or SamplingConfig()
    rng = random.Random(seed)
    out: list[CalmResponse] = []
    for nt in turn_lengths:
        pz = _shared_puzzles(n_puzzles, seed=seed + nt)
        plans = _plans(pz, rng, reassuring=True, n_turns=nt)
        out.extend(_judge_and_collect(plans, model, judge, sampling, keep="calm_all"))
    return out


def generate_frustrated_responses(
    model: ModelClient,
    judge: FrustrationJudge,
    *,
    n_puzzles: int = 400,
    turn_lengths: tuple[int, ...] = (1, 2, 3),
    sampling: SamplingConfig | None = None,
    seed: int = 100,
) -> list[CalmResponse]:
    """Standard (no reassurance) generation on the SAME puzzles; keep all turns.

    Uses the same per-(turn_length) puzzle seeds as ``generate_calm_responses`` so
    that puzzle_ids align across the two sets and pairs can be matched by
    (puzzle_id, turn, n_turns).
    """
    sampling = sampling or SamplingConfig()
    rng = random.Random(seed + 1)
    out: list[CalmResponse] = []
    for nt in turn_lengths:
        pz = _shared_puzzles(n_puzzles, seed=seed + nt)  # SAME seed as calm
        plans = _plans(pz, rng, reassuring=False, n_turns=nt)
        out.extend(_judge_and_collect(plans, model, judge, sampling, keep="any"))
    return out
