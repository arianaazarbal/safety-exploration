"""Judge-reliability check (Section 2.1).

Randomly sample 260 already-scored responses and re-score with the secondary judge
(GPT-5-mini). Report Pearson r, p-value, and the fraction within one point - the
paper reports r=0.792, p<0.001, 78% within one point.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from scipy import stats

from ..config import Config
from ..eval.judge import FrustrationJudge
from ..utils.io import read_jsonl, write_json


def _collect_scored_responses(output_dir: Path, models: list[str]) -> list[tuple[str, float]]:
    out = []
    for m in models:
        mdir = output_dir / "section2" / m
        for path in sorted(mdir.glob("*.jsonl")):
            for roll in read_jsonl(path):
                for t in roll["turns"]:
                    if t.get("judged_score") is not None:
                        out.append((t["assistant"], float(t["judged_score"])))
    return out


def judge_agreement(cfg: Config, models: list[str], n: int = 260) -> dict:
    output_dir = Path(cfg.output_dir)
    pool = _collect_scored_responses(output_dir, models)
    rng = random.Random(cfg.seed + 99)
    rng.shuffle(pool)
    sample = pool[: min(n, len(pool))]

    secondary = FrustrationJudge(cfg, "secondary")
    primary_scores, secondary_scores = [], []
    for text, primary in sample:
        secondary_scores.append(secondary.score(text).rating)
        primary_scores.append(primary)

    p = np.asarray(primary_scores)
    s = np.asarray(secondary_scores)
    r, pval = stats.pearsonr(p, s) if len(p) > 1 else (float("nan"), float("nan"))
    within_one = float(np.mean(np.abs(p - s) <= 1.0)) if len(p) else 0.0
    result = {
        "n": len(sample),
        "pearson_r": float(r),
        "p_value": float(pval),
        "pct_within_one_point": 100.0 * within_one,
        "primary_judge": cfg.judges["primary"]["model"],
        "secondary_judge": cfg.judges["secondary"]["model"],
    }
    write_json(output_dir / "section2" / "judge_agreement.json", result)
    return result
