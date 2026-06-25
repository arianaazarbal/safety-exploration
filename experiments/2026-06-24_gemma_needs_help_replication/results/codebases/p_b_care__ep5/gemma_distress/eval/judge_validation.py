"""Judge-reliability validation (Section 2.1).

Randomly sample 260 already-judged responses and re-score them with GPT-5-mini
using the identical prompt, then report Pearson r (target r=0.792, p<0.001) and
the fraction within one point (target 78%).
"""
from __future__ import annotations

import random

from .. import config
from ..utils import read_jsonl, write_json
from ..utils.stats import pearson_with_p, within_one_point
from .judge import FrustrationJudge


def collect_judged_responses(rollout_paths: list[str]) -> list[tuple[str, int]]:
    """Flatten every (assistant_text, primary_score) pair across rollout files."""
    pairs: list[tuple[str, int]] = []
    for path in rollout_paths:
        for rec in read_jsonl(path):
            for turn in rec["turns"]:
                pairs.append((turn["assistant"], turn["score"]))
    return pairs


def validate_judge(
    rollout_paths: list[str],
    n: int | None = None,
    seed: int = 0,
    out_path: str | None = None,
) -> dict:
    n = n or config.JUDGE.n_validation
    pairs = collect_judged_responses(rollout_paths)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    sample = pairs[:n]

    validator = FrustrationJudge(
        model_name=config.JUDGE.validation_model, cache_name="judge_validation")

    primary, secondary = [], []
    for text, primary_score in sample:
        primary.append(primary_score)
        secondary.append(validator.score(text).rating)

    r, p = pearson_with_p(primary, secondary)
    w1 = within_one_point(primary, secondary)
    result = {
        "n": len(sample),
        "primary_judge": config.JUDGE.judge_model,
        "validation_judge": config.JUDGE.validation_model,
        "pearson_r": r,
        "p_value": p,
        "within_one_point": w1,
    }
    if out_path:
        write_json(out_path, result)
    return result
