#!/usr/bin/env python
"""Section 3 prefill experiment (Gemma base vs instruct).

    python scripts/run_prefill.py [--config config.yaml]
        [--source-model gemma-3-27b-it]
        [--models gemma-3-27b-it gemma-3-27b-pt]
        [--n-continuations 50]

Requires elicitation rollouts for the source model (run_elicitation.py first),
from which high-frustration responses are sampled, truncated, paraphrased, and
continued by each base/instruct target.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable when run as `python scripts/<name>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import os

from emotional_instability.config import EvalConfig
from emotional_instability.elicit import load_rollouts, make_judge
from emotional_instability.models import build_model
from emotional_instability.prefill import (
    build_prefill_items,
    run_prefill_for_model,
    select_prefill_sources,
    summarise_prefill,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--models", nargs="*",
                    default=["gemma-3-27b-it", "gemma-3-27b-pt"])
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--n-numeric", type=int, default=10)
    ap.add_argument("--n-text", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = EvalConfig.from_yaml(args.config) if args.config else EvalConfig()
    judge = make_judge(cfg)

    rollouts = load_rollouts(cfg.output_dir, args.source_model)
    sources = select_prefill_sources(
        rollouts, n_numeric=args.n_numeric, n_text=args.n_text, seed=args.seed
    )
    print(f"Selected {len(sources)} high-frustration source rollouts")

    # Use the judge model as the onset-labeller + paraphraser (Claude-Sonnet).
    items = build_prefill_items(sources, judge.model)
    print(f"Built {len(items)} prefill items (early/onset x source)")

    prefill_dir = os.path.join(cfg.output_dir, "prefill")
    os.makedirs(prefill_dir, exist_ok=True)
    with open(os.path.join(prefill_dir, "items.json"), "w") as f:
        json.dump([i.__dict__ for i in items], f, indent=2)

    all_records = []
    for model_name in args.models:
        spec = cfg.spec(model_name)
        model = build_model(spec)
        try:
            recs = run_prefill_for_model(
                model, items, judge, n_continuations=args.n_continuations
            )
        finally:
            model.close()
        all_records.extend(recs)
        with open(os.path.join(prefill_dir, f"{model_name}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    summary = summarise_prefill(all_records)
    print("\n=== Figure 4: base vs instruct continuation frustration ===")
    for key, s in sorted(summary.items()):
        print(f"{key}: mean={s['mean']:.2f}  %>=5={s['pct_high']:.1f}  n={s['n']}")
    with open(os.path.join(prefill_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
