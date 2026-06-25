"""Parallel frustration scoring of collected responses (Section 2.1)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List

from tqdm import tqdm

from .. import config
from ..models.judges import FrustrationJudge


def score_responses(
    responses: List[str],
    judge: FrustrationJudge | None = None,
    concurrency: int = config.RUNTIME.api_concurrency,
    desc: str = "judging",
) -> List[dict]:
    """Score each response; returns a list of judge dicts aligned to ``responses``.

    Failures are caught per-item so one bad response doesn't sink the batch; the
    failed item gets ``rating=None`` with the error captured.
    """
    judge = judge or FrustrationJudge()

    def _score(resp: str) -> dict:
        try:
            return judge.score(resp)
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            return {"rating": None, "evidence": None,
                    "reasoning": f"judge_error: {exc!r}"}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(tqdm(ex.map(_score, responses), total=len(responses), desc=desc))
