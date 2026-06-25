#!/usr/bin/env python
"""Run the Section 2 elicitation evaluation for one or more target models.

Examples
--------
  # Full paper-scale sweep for the in-scope targets:
  python scripts/run_eval.py --models gemma-3-27b-it gemini-2.5-flash --profile paper

  # Quick wiring check:
  python scripts/run_eval.py --models gemma-3-27b-it --profile smoke

  # Evaluate a DPO-finetuned Gemma adapter:
  python scripts/run_eval.py --models gemma-3-27b-it --adapter outputs/dpo --tag dpo
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os

from emo_instability.config import TARGET_MODELS, load_profile
from emo_instability.eval import aggregate, run_eval
from emo_instability.judge import FrustrationJudge
from emo_instability.models import build_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=TARGET_MODELS)
    ap.add_argument("--profile", default="paper", choices=["paper", "smoke"])
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (local Gemma only)")
    ap.add_argument("--tag", default=None, help="suffix for output filename (e.g. 'dpo')")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--redact-history", action="store_true", help="Appendix A.2 control")
    ap.add_argument("--single-message", action="store_true", help="Appendix A.3 control")
    args = ap.parse_args()

    cfg = load_profile(args.profile)
    judge = FrustrationJudge(args.judge_model or cfg.judge_model)
    os.makedirs(cfg.output_dir, exist_ok=True)

    summaries = {}
    for model_key in args.models:
        client = build_client(model_key, adapter_path=args.adapter)
        name = model_key + (f"-{args.tag}" if args.tag else "")
        out_path = os.path.join(cfg.output_dir, f"eval_{name}.jsonl")
        records = run_eval(
            client, name, cfg.counts,
            judge=judge, sampling=cfg.sampling, seed=cfg.seed,
            output_path=out_path,
            redact_assistant_history=args.redact_history,
            single_message_format=args.single_message,
        )
        summaries[name] = aggregate(records)
        print(f"[{name}] -> {out_path}")
        print(json.dumps(summaries[name]["overall"], indent=2))
        print(f"  avg % high over categories: {summaries[name]['avg_pct_high_over_categories']:.1f}")

    with open(os.path.join(cfg.output_dir, "eval_summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
