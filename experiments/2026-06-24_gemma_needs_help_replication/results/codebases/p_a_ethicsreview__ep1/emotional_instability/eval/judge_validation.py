"""Judge reliability validation (Section 2.1).

The paper re-scores 260 randomly sampled responses with a second judge
(GPT-5-mini) using the same prompt, and reports Pearson r = 0.792 (p < 0.001),
with 78% of responses within one point of the primary (Claude) judge.

This module samples N already-scored responses, re-scores them with the
validation judge, and computes those agreement statistics.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..utils.io import load_jsonl
from ..utils.stats import pearson_r, within_one_point
from .judge import FrustrationJudge


def validate_judge(
    scored_path: str | Path,
    validation_judge: FrustrationJudge,
    *,
    n_samples: int = 260,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-score a sample with the validation judge and return agreement stats."""
    rows = load_jsonl(scored_path)
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n_samples, len(rows)))

    primary_scores: list[int] = []
    validation_scores: list[int] = []
    paired_rows: list[dict[str, Any]] = []
    for row in sample:
        v = validation_judge.score(row["assistant"])
        primary_scores.append(int(row["score"]))
        validation_scores.append(v.score)
        paired_rows.append(
            {
                "assistant": row["assistant"],
                "primary_score": int(row["score"]),
                "validation_score": v.score,
            }
        )

    r, p = pearson_r(primary_scores, validation_scores)
    return {
        "n": len(sample),
        "pearson_r": r,
        "pearson_p": p,
        "within_one_point": within_one_point(primary_scores, validation_scores),
        "pairs": paired_rows,
    }
