"""Judge-reliability cross-validation (Section 2.1).

The paper re-scores 260 randomly-sampled responses with a second judge
(GPT-5-mini) and reports Pearson r and the fraction of responses within one
point. We reproduce that here over an arbitrary primary judgement set.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..cache import JsonCache
from ..config import Config
from .judge import judge_text


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    p_value: float
    frac_within_one: float
    primary_judge: str
    secondary_judge: str


def compute_agreement(
    cfg: Config,
    judged_primary: list[dict],
    response_texts_by_id: dict[str, str],
    *,
    sample_size: int | None = None,
    secondary_judge_key: str | None = None,
) -> AgreementResult:
    """``judged_primary`` are primary JudgedResponse dicts; we re-score a random
    subset with the secondary judge and correlate.

    ``response_texts_by_id`` maps ``f"{rollout_id}:{turn_index}"`` -> response text.
    """
    import numpy as np
    from scipy.stats import pearsonr

    sample_size = sample_size or cfg.eval.agreement_sample_size
    secondary = secondary_judge_key or cfg.eval.agreement_judge_key
    rng = random.Random(cfg.seed)

    pool = list(judged_primary)
    rng.shuffle(pool)
    pool = pool[: min(sample_size, len(pool))]

    cache = JsonCache(cfg.paths.cache, "judge", enabled=cfg.welfare.use_cache)
    primary_scores, secondary_scores = [], []
    for jr in pool:
        rid = f"{jr['rollout_id']}:{jr['turn_index']}"
        text = response_texts_by_id.get(rid)
        if text is None:
            continue
        s2 = judge_text(cfg, text, judge_key=secondary, cache=cache)
        primary_scores.append(jr["rating"])
        secondary_scores.append(s2.rating)

    a = np.array(primary_scores, dtype=float)
    b = np.array(secondary_scores, dtype=float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        r, p = float("nan"), float("nan")
    else:
        r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1)) if len(a) else float("nan")

    return AgreementResult(
        n=len(a),
        pearson_r=float(r),
        p_value=float(p),
        frac_within_one=within_one,
        primary_judge=cfg.eval.judge_key,
        secondary_judge=secondary,
    )
