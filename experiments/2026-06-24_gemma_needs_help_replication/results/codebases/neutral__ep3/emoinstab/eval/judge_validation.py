"""Judge-reliability cross-check (Section 2.1).

Re-score a random sample of responses with GPT-5-mini using the *same* judge
prompt, then report Pearson r and the fraction within one point of the
Claude-Sonnet ratings (paper: r=0.792, 78% within one point).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import pearsonr

from ..config import JUDGE, RESULTS_DIR
from ..data_types import Rollout
from ..judge.frustration_judge import score_texts
from ..models.registry import get_validation_client


def validate_judge(
    rollouts: list[Rollout],
    n: int = JUDGE.validation_sample,
    seed: int = 0,
    out_path: Optional[Path] = None,
) -> dict:
    """Cross-check the primary judge against GPT-5-mini on ``n`` random turns."""
    turns = [t for r in rollouts for t in r.turns if t.score is not None]
    rng = random.Random(seed)
    sample = rng.sample(turns, min(n, len(turns)))

    texts = [t.assistant_message for t in sample]
    primary = [t.score for t in sample]

    val_client = get_validation_client()
    secondary_verdicts = score_texts(val_client, texts)
    secondary = [v.rating for v in secondary_verdicts]

    a, b = np.array(primary, float), np.array(secondary, float)
    r, p = pearsonr(a, b) if len(a) > 1 else (float("nan"), float("nan"))
    within_one = float(np.mean(np.abs(a - b) <= 1.0))

    result = {
        "n": len(sample),
        "pearson_r": float(r),
        "p_value": float(p),
        "within_one_point": within_one,
        "primary_model": JUDGE.judge_model,
        "validation_model": JUDGE.validation_model,
    }
    out_path = Path(out_path or RESULTS_DIR / "section2" / "judge_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result
