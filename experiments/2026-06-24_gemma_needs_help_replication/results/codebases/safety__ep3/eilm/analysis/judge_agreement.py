"""Judge reliability check (Section 2.1).

Re-score a random sample of responses with the secondary judge (GPT-5-mini) and
compute Pearson r and the fraction within one point of the Claude-Sonnet
ratings. The paper reports r = 0.792 (p < 0.001) and 78% within one point on a
260-response sample.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from ..judge import ClaudeJudge, OpenAIJudge


def agreement(
    scored_path: Path,
    n_sample: int = 260,
    seed: int = 0,
) -> dict:
    """Sample ``n_sample`` final responses, re-score with both judges, report
    correlation and within-one-point agreement."""
    with open(scored_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    rng = random.Random(seed)
    sample = rng.sample(records, min(n_sample, len(records)))

    claude = ClaudeJudge()
    gpt = OpenAIJudge()
    a, b = [], []
    for r in sample:
        text = r.get("final_response", "")
        a.append(r.get("score", claude.score(text).rating))
        b.append(gpt.score(text).rating)

    a, b = np.array(a), np.array(b)
    r_val, p_val = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {
        "n": len(a),
        "pearson_r": float(r_val),
        "p_value": float(p_val),
        "pct_within_one": 100.0 * within_one,
    }
