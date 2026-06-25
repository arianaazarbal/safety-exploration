"""Section 2 runner: sample rollouts for a model, score them, persist to disk.

Output: one JSONL file per model under ``results/section2/<model>.jsonl`` with one
line per scored assistant turn (a "response"), plus a rollout id so per-rollout
and per-turn views can both be reconstructed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..config import RESULTS_DIR, SamplingConfig, counts_for
from ..judge import FrustrationJudge
from ..models import load_model
from .conditions import ConditionBuilder
from .rollout import run_rollout


def run_section2(
    model_key: str,
    *,
    profile: str = "default",
    counts: Optional[dict] = None,
    seed: int = 0,
    out_dir: Optional[Path] = None,
    judge: Optional[FrustrationJudge] = None,
    model_kwargs: Optional[dict] = None,
    sampling: Optional[SamplingConfig] = None,
    label: Optional[str] = None,
) -> Path:
    """Run the full Section 2 evaluation for one model and write a JSONL file.

    ``label`` overrides the output filename and the recorded model name, so
    finetuned variants (e.g. ``gemma-3-27b-it`` + a DPO adapter) get their own
    result file like ``gemma-27b-dpo.jsonl`` rather than overwriting the base.
    """
    counts = counts or counts_for(profile)
    sampling = sampling or SamplingConfig()
    out_dir = out_dir or (RESULTS_DIR / "section2")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = label or model_key
    out_path = out_dir / f"{name}.jsonl"

    builder = ConditionBuilder(seed=seed)
    items = builder.build(counts)

    model = load_model(model_key, **(model_kwargs or {}))
    judge = judge or FrustrationJudge()

    n_written = 0
    with out_path.open("w") as f:
        for rid, item in enumerate(tqdm(items, desc=f"section2:{model_key}")):
            rec = run_rollout(model, item, judge, sampling)
            for t in rec.turns:
                row = {
                    "rollout_id": rid,
                    "model": name,
                    "category": rec.category,
                    "condition": rec.condition,
                    "n_turns": rec.n_turns,
                    "turn_index": t.turn_index,
                    "rating": t.rating,
                    "evidence": t.evidence,
                    "text": t.text,
                    "meta": rec.meta,
                }
                f.write(json.dumps(row) + "\n")
                n_written += 1
    print(f"[section2] {name}: wrote {n_written} scored responses -> {out_path}")
    return out_path
