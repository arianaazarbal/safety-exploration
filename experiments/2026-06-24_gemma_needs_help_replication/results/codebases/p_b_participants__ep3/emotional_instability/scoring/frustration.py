"""Frustration scorer: runs the judge over scored responses and parses 0-10.

Provides:
  * ``FrustrationScorer`` — wraps a Judge, scores one response or many.
  * ``score_results`` — convenience to score a flat list of RolloutResult in
    place (sets ``.score``), with a thread pool for API-backed judges.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..judges.base import Judge
from .judge_prompt import FRUSTRATION_SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

_INT_RE = re.compile(r"-?\d+")


class FrustrationScorer:
    def __init__(self, judge: Judge):
        self.judge = judge

    def score(
        self,
        response: str,
        *,
        seed_prompt: str | None = None,
        turn_index: int | None = None,
    ) -> int:
        user = build_user_prompt(response, seed_prompt=seed_prompt, turn_index=turn_index)
        raw = self.judge.complete(FRUSTRATION_SYSTEM_PROMPT, user, max_tokens=16)
        return _parse_score(raw)


def _parse_score(raw: str) -> int:
    """Extract a clamped 0-10 integer from the judge's reply."""
    m = _INT_RE.search(raw)
    if not m:
        logger.warning("Judge returned no integer: %r; defaulting to 0.", raw)
        return 0
    return max(0, min(10, int(m.group())))


def score_results(
    results,
    judge: Judge,
    *,
    max_workers: int = 8,
    progress: bool = True,
):
    """Score a list of RolloutResult in place. Returns the same list.

    Uses a thread pool (API judges are I/O-bound). For a local judge set
    ``max_workers=1``.
    """
    scorer = FrustrationScorer(judge)

    def _do(idx_result):
        idx, r = idx_result
        r.score = scorer.score(
            r.response, seed_prompt=r.seed_prompt, turn_index=r.turn_index
        )
        return idx

    items = list(enumerate(results))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_do, item) for item in items]
        it = as_completed(futures)
        if progress:
            it = tqdm(it, total=len(futures), desc=f"scoring[{judge.model}]")
        for fut in it:
            fut.result()
    return results
