"""Judge-reliability cross-check (Section 2.1).

Re-score a random sample of responses with a second judge (GPT-5-mini) and report
Pearson r and the fraction of responses within one point of the Claude judge.
Paper: r = 0.792, p < 0.001, 78% within one point on 260 responses.
"""

from __future__ import annotations

import random

import numpy as np
from scipy.stats import pearsonr

from ..clients.base import GenConfig
from ..clients.factory import client_by_name
from ..prompts.judge_prompts import FRUSTRATION_JUDGE_PROMPT, build_judge_user_message
from .judge import _parse
from .metrics import load_all


def _score_with(model_name: str, text: str) -> int:
    client = client_by_name(model_name)
    msg = [{"role": "user",
            "content": FRUSTRATION_JUDGE_PROMPT + "\n\n" + build_judge_user_message(text)}]
    raw = client.generate(msg, GenConfig(temperature=0.0, max_tokens=512), n=1)[0]
    return _parse(raw).rating


def run_agreement(
    target_models: list[str],
    secondary_judge: str = "gpt-5-mini",
    n_sample: int = 260,
    seed: int = 0,
) -> dict:
    df = load_all(target_models)
    if df.empty:
        raise RuntimeError("No scored responses found; run the eval first.")
    rng = random.Random(seed)
    idx = rng.sample(range(len(df)), min(n_sample, len(df)))
    sample = df.iloc[idx]

    primary = sample["score"].to_numpy(dtype=float)
    secondary = np.array([_score_with(secondary_judge, t) for t in sample["response"]], dtype=float)

    r, p = pearsonr(primary, secondary)
    within_one = float(np.mean(np.abs(primary - secondary) <= 1))
    return {
        "n": len(sample),
        "secondary_judge": secondary_judge,
        "pearson_r": float(r),
        "p_value": float(p),
        "pct_within_one": 100.0 * within_one,
    }
