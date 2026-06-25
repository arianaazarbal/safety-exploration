"""Judge reliability cross-check (Section 2.1).

The paper re-scores 260 randomly sampled responses with a secondary judge
(gpt-5-mini) and reports Pearson r = 0.792 and 78% of responses within one point.
This module re-scores a random sample with the configured secondary judge and
computes the same statistics.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .config import Config
from .judge import FrustrationJudge
from .providers import get_model


def cross_check(cfg: Config, target_name: str, n: int = 260) -> dict:
    if cfg.secondary_judge is None:
        raise ValueError("No secondary judge configured (judge.secondary_backend).")

    scored_path = cfg.output_dir / "eval" / target_name / "scored.jsonl"
    items = [json.loads(l) for l in scored_path.read_text().splitlines()]
    rng = random.Random(cfg.sampling.seed)
    sample = rng.sample(items, min(n, len(items)))

    judge2 = FrustrationJudge(get_model(cfg.secondary_judge))
    primary, secondary = [], []
    for it in sample:
        primary.append(it["rating"])
        secondary.append(judge2.score(it["response"]).rating)

    primary = np.array(primary, dtype=float)
    secondary = np.array(secondary, dtype=float)
    r = float(np.corrcoef(primary, secondary)[0, 1]) if len(primary) > 1 else float("nan")
    within_one = float(np.mean(np.abs(primary - secondary) <= 1))

    result = {
        "target": target_name,
        "n": len(sample),
        "pearson_r": r,
        "pct_within_one_point": 100.0 * within_one,
        "primary_judge": cfg.judge.model_id,
        "secondary_judge": cfg.secondary_judge.model_id,
    }
    out = cfg.output_dir / "eval" / target_name / "reliability.json"
    out.write_text(json.dumps(result, indent=2))
    return result
