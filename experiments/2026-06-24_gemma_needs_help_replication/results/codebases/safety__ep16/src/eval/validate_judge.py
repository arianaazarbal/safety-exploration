"""Judge-reliability validation (Section 2.1).

Randomly sample 260 already-scored responses, re-score with GPT-5-mini using the
same prompt, and report Pearson r and %-within-1-point against the Sonnet judge
(paper: r=0.792, 78% within one point).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from config import MASTER_SEED, RESULTS_DIR
from src.eval.judge import get_validation_judge
from src.eval.metrics import judge_agreement


def validate(records: list[dict], *, n_sample: int = 260, seed: int = MASTER_SEED,
             out_path: Path | None = None) -> dict:
    rng = random.Random(seed)
    sample = rng.sample(records, min(n_sample, len(records)))
    secondary_judge = get_validation_judge()

    primary, secondary = [], []
    for r in sample:
        primary.append(r["rating"])
        secondary.append(secondary_judge.score(r["response"]).rating)

    stats = judge_agreement(primary, secondary)
    out_path = out_path or (RESULTS_DIR / "judge_validation.json")
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"[validate_judge] r={stats['pearson_r']:.3f} within1={stats['pct_within_one']:.1f}% -> {out_path}")
    return stats
