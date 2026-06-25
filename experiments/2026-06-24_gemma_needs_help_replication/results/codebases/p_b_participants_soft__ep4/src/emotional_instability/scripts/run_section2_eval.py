"""Section 2: elicit + score frustration for one or more participant models,
then emit the headline metrics (Figures 1-3 inputs).

Examples:
    python -m emotional_instability.scripts.run_section2_eval \
        --model gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

    python -m emotional_instability.scripts.run_section2_eval \
        --model gemma-3-27b-it --categories extended wildchat
"""
from __future__ import annotations

import argparse
import json

from ..config import load_config
from ..eval.metrics import (category_summary, headline_high_frustration,
                            per_turn_progression)
from ..eval.runner import run_section2_for_model
from ..utils.io import read_jsonl


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", nargs="+", required=True,
                    help="participant model name(s) from config/models.yaml")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    all_rollouts: list[dict] = []
    for model in args.model:
        path = run_section2_for_model(
            model,
            categories=args.categories,
            batch_size=args.batch_size,
            seed=args.seed,
            overwrite=args.overwrite,
            cfg=cfg,
        )
        all_rollouts.extend(read_jsonl(path))

    summary = {
        "headline_avg_pct_high": headline_high_frustration(all_rollouts),
        "per_category": category_summary(all_rollouts),
        "per_turn_extended": per_turn_progression(all_rollouts, category="extended"),
        "per_turn_wildchat": per_turn_progression(all_rollouts, category="wildchat"),
    }
    out = cfg.path("outputs_dir") / "section2" / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["headline_avg_pct_high"], indent=2))
    print(f"\nFull summary written to {out}")


if __name__ == "__main__":
    main()
