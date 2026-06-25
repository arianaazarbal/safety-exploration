"""Section 3: base-vs-instruct prefill experiment (Gemma in scope).

Requires a scored Section-2 rollouts file for gemma-3-27b-it to source the 20
high-frustration seed conversations.

Example:
    python -m emotional_instability.scripts.run_section3_prefill \
        --seed-rollouts outputs/section2/gemma-3-27b-it/rollouts.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from ..config import load_config
from ..prefill.prefill_runner import run_prefill_experiment
from ..utils.io import read_jsonl


def summarize(continuations_path) -> dict:
    rows = [r for r in read_jsonl(continuations_path) if r.get("score") is not None]
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["model"], r["prompt_type"], r["condition"])].append(int(r["score"]))
    out = {}
    for (model, ptype, cond), scores in buckets.items():
        arr = np.array(scores, dtype=float)
        out[f"{model}|{ptype}|{cond}"] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "frac_high": float((arr >= 5).mean()),
        }
    return out


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-rollouts", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    path = run_prefill_experiment(
        seed_rollouts_path=args.seed_rollouts,
        models=args.models,
        cfg=cfg,
        paraphrase=not args.no_paraphrase,
    )
    summary = summarize(path)
    out = cfg.path("outputs_dir") / "section3" / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
