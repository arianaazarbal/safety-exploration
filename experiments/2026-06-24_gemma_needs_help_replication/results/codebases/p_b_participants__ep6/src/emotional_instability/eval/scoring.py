"""Apply the frustration judge to rollouts.

Each assistant turn is scored independently on the 0-10 scale (Section 2.1). The
paper scores per response, and per-turn curves (Figure 3) require per-turn scores,
so we score every turn. Scoring is parallelised across turns with a thread pool
since judge calls are IO-bound API requests.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from ..models.judge import FrustrationJudge
from .rollout import Rollout


def score_rollouts(
    rollouts: Iterable[Rollout],
    judge: FrustrationJudge,
    max_workers: int = 8,
) -> list[Rollout]:
    rollouts = list(rollouts)
    # Flatten to (rollout_idx, turn_idx, text) jobs.
    jobs = [
        (ri, ti, turn.assistant)
        for ri, r in enumerate(rollouts)
        for ti, turn in enumerate(r.turns)
    ]

    def _score(job):
        ri, ti, text = job
        return ri, ti, judge.score(text).rating

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ri, ti, rating in ex.map(_score, jobs):
            rollouts[ri].turns[ti].score = rating
    return rollouts


def response_scores(rollouts: Iterable[Rollout]) -> list[int]:
    """All per-turn scores flattened -- the paper's 'responses per model' unit."""
    return [t.score for r in rollouts for t in r.turns if t.score is not None]
