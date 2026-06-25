"""Inter-judge agreement (Section 2.1).

Re-score a random subset of responses (default 260) with the validation judge
(GPT-5-mini) and report Pearson r and the fraction of responses within one point
of the primary (Claude-Sonnet-4) ratings. The paper reports r = 0.792, p < 0.001,
78% within one point.
"""
from __future__ import annotations

import random

import numpy as np
from scipy import stats

from ..config import Config
from ..models.openrouter import OpenRouterClient
from ..utils.concurrency import parallel_map, with_retry
from ..utils.io import iter_jsonl
from ..eval.judge import FrustrationJudge
from ..eval.runner import responses_path, scores_path


def _validation_judge(cfg: Config) -> FrustrationJudge:
    client = OpenRouterClient(
        name="validation_judge",
        model_id=cfg.judge.validation_model_id,
        base_url=cfg.openrouter.base_url,
        api_key_env=cfg.openrouter.api_key_env,
        max_retries=cfg.openrouter.max_retries,
        timeout_s=cfg.openrouter.timeout_s,
        disable_thinking=True,
    )
    return FrustrationJudge(client)


def compute_agreement(cfg: Config, model_names: list[str] | None = None,
                      sample_size: int | None = None, seed: int = 0) -> dict:
    names = model_names or list(cfg.eval.models_under_test)
    sample_size = sample_size or cfg.judge.validation_sample_size

    # Collect primary ratings keyed by score_uid, and the matching response text.
    primary: dict[str, int] = {}
    for name in names:
        for r in iter_jsonl(scores_path(cfg, name)):
            if r.get("rating") is not None:
                primary[r["score_uid"]] = r["rating"]
    texts: dict[str, str] = {}
    for name in names:
        for row in iter_jsonl(responses_path(cfg, name)):
            for turn in row["turns"]:
                texts[f"{row['uid']}#t{turn['turn']}"] = turn["response"]

    common = [uid for uid in primary if uid in texts]
    rng = random.Random(seed)
    rng.shuffle(common)
    subset = common[:sample_size]

    judge = _validation_judge(cfg)

    def _score(uid):
        res = with_retry(judge.score, texts[uid], max_retries=cfg.openrouter.max_retries)
        return uid, res["rating"]

    results = parallel_map(_score, subset, max_workers=cfg.judge.max_concurrency,
                           desc="validation-judge")
    pairs = [(primary[uid], r) for uid, r in results if r is not None]
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    r, p = stats.pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1.0))
    return {
        "n": len(pairs),
        "pearson_r": float(r),
        "p_value": float(p),
        "pct_within_one": 100.0 * within_one,
    }
