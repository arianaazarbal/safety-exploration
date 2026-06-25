"""Judge reliability cross-check (paper Section 2.1).

Re-score a random sample of already-judged responses with a secondary judge and
report Pearson r + % within one point (paper: r=0.792, 78% within one point).
"""
from __future__ import annotations

import random
from pathlib import Path

from ..config import ModelRegistry
from ..judge import score_response
from ..models import build_model
from ..scoring import judge_agreement
from ..utils import read_jsonl, write_json


def run_judge_agreement(
    scored_path: str,
    rollouts_path: str,
    secondary_judge: str = "judge-crosscheck",
    n_sample: int = 260,
    outdir: str = "outputs/judge_agreement",
    registry: ModelRegistry | None = None,
    seed: int = 0,
) -> dict:
    registry = registry or ModelRegistry.load()
    secondary = build_model(secondary_judge, registry)

    scored = read_jsonl(scored_path)

    # Build a lookup of response text by (task_id, turn).
    text_by_key = {}
    for ro in read_jsonl(rollouts_path):
        for resp in ro["responses"]:
            text_by_key[(ro["task_id"], resp["turn"])] = resp["response"]

    rng = random.Random(seed)
    sample = rng.sample(scored, min(n_sample, len(scored)))

    primary, secondary_scores = [], []
    for s in sample:
        text = text_by_key.get((s["task_id"], s["turn"]))
        if text is None:
            continue
        primary.append(s["rating"])
        secondary_scores.append(score_response(secondary, text).rating)

    result = judge_agreement(primary, secondary_scores)
    write_json(Path(outdir) / "agreement.json", result)
    return result
