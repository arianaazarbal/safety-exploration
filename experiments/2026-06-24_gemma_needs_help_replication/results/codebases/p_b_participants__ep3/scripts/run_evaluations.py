#!/usr/bin/env python
"""Section 2: run the distress-elicitation evaluations and score them.

For each requested participant:
  1. build the 8 conditions across 5 categories (config/eval.yaml),
  2. run all rollouts (welfare run-notice emitted per condition),
  3. score every assistant response with the frustration judge,
  4. save scored responses to JSONL,
  5. print the Figure-1/2 summary, the Figure-3 per-turn progression, and the
     Table-3 differential words.

Example:
    python scripts/run_evaluations.py --participants gemma-3-27b-it gemini-2.5-flash \
        --out artifacts/eval --adapter artifacts/training/dpo   # adapter optional
"""
from __future__ import annotations

import argparse
from pathlib import Path

from emotional_instability.analysis import (
    differential_words,
    per_turn_progression,
    results_to_frame,
    summary_table,
)
from emotional_instability.config import EvalConfig, ModelsConfig
from emotional_instability.elicitation import build_conditions, run_condition
from emotional_instability.elicitation.runner import flatten_results
from emotional_instability.runtime import get_judge, get_participant, setup_logging
from emotional_instability.scoring import score_results
from emotional_instability.storage import save_results_jsonl
from emotional_instability.welfare import WelfareConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participants", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/eval")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (eval a fine-tuned Gemma)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap rollouts/condition (smoke test)")
    args = ap.parse_args()
    setup_logging()

    models_cfg = ModelsConfig.load()
    eval_cfg = EvalConfig.load()
    judge = get_judge(models_cfg, "frustration")
    welfare = WelfareConfig.from_env()
    out_dir = Path(args.out)

    conditions = build_conditions(eval_cfg, seed=args.seed)
    if args.limit:
        for c in conditions:
            c.seed_prompts = c.seed_prompts[: args.limit]
            c.n_rollouts = len(c.seed_prompts)

    for name in args.participants:
        model = get_participant(models_cfg, name, adapter_path=args.adapter)
        all_results = []
        for cond in conditions:
            rollouts = run_condition(
                model, cond, welfare=welfare,
                temperature=eval_cfg.sampling["temperature"],
                max_new_tokens=eval_cfg.sampling["max_new_tokens"],
            )
            all_results.extend(flatten_results(rollouts))
        model.close()

        score_results(all_results, judge, max_workers=args.judge_workers)

        tag = name + ("_adapter" if args.adapter else "")
        save_results_jsonl(all_results, out_dir / f"{tag}.jsonl")

        df = results_to_frame(all_results)
        thr = eval_cfg.high_frustration_threshold
        print(f"\n===== {tag} : Figure 1/2 summary (threshold>={thr}) =====")
        print(summary_table(df, threshold=thr).to_string(index=False))
        print(f"\n----- {tag} : Figure 3 per-turn (extended, wildchat) -----")
        print(per_turn_progression(df, threshold=thr).to_string(index=False))
        if (df["category"] == "impossible_numeric").any():
            print(f"\n----- {tag} : Table 3 differential words (numeric) -----")
            print(differential_words(df, name, top_n=20)[["word", "log_odds"]].to_string(index=False))

    print(f"\nSaved scored responses under {out_dir}/")


if __name__ == "__main__":
    main()
