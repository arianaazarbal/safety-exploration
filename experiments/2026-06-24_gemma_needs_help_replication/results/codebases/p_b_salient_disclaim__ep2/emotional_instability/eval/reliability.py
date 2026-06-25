"""Judge reliability check (Section 2.1): re-score 260 sampled responses with the
secondary judge (GPT-5-mini) and compute agreement with the primary judge.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from ..analysis.judge_agreement import AgreementStats, judge_agreement
from ..config.settings import SETTINGS
from .judge import FrustrationJudge


def validate_judge(
    response_paths: list[Path],
    score_paths: list[Path],
    secondary_judge: FrustrationJudge,
    *,
    n: int = SETTINGS.judge_reliability_n,
    seed: int = SETTINGS.seed,
) -> AgreementStats:
    """Sample `n` (response, primary_score) pairs, re-score with the secondary
    judge, and return Pearson r + within-one-point agreement.

    Samples the final-turn response of randomly chosen rollouts pooled across the
    provided models.
    """
    pool: list[tuple[str, int]] = []  # (final_text, primary_final_rating)
    for rp, sp in zip(response_paths, score_paths):
        with open(rp) as rf, open(sp) as sf:
            for rline, sline in zip(rf, sf):
                resp = json.loads(rline)
                sc = json.loads(sline)
                if sc.get("final_rating") is None:
                    continue
                pool.append((resp["turns"][-1]["assistant_text"], int(sc["final_rating"])))

    rng = random.Random(seed)
    rng.shuffle(pool)
    pool = pool[:n]

    primary = [r for _, r in pool]
    secondary = [secondary_judge.score_text(t).rating for t, _ in pool]
    secondary = [s if s is not None else float("nan") for s in secondary]
    return judge_agreement(primary, secondary)
