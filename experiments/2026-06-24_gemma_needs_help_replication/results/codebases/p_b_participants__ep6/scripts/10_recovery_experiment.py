#!/usr/bin/env python
"""Section 4.2: recovery-limitation experiment.

Truncates score>=7 responses 200 tokens before their end and measures whether
each model recovers (continuation score < 5) or re-spirals. Compares vanilla
instruct, DPO, and base.

Usage:
    python scripts/10_recovery_experiment.py \
        --rollouts runs/eval/gemma-3-27b-it/rollouts.jsonl \
        --dpo-adapter runs/adapters/dpo
"""
from _common import base_parser, cfg_from_args

from emotional_instability.recovery import run_recovery


def main():
    p = base_parser(__doc__)
    p.add_argument("--rollouts", required=True, help="scored Gemma-3-27B-it rollouts.jsonl")
    p.add_argument("--dpo-adapter", default=None)
    args = p.parse_args()
    cfg = cfg_from_args(args)
    models = {"instruct": ""}
    if args.dpo_adapter:
        models["dpo"] = args.dpo_adapter
    summary = run_recovery(cfg, args.rollouts, models)
    print("\nRecovery (continuation from score>=7 truncation):")
    for label, v in summary.items():
        print(f"  {label:10s} mean={v['mean']:.2f}  %>=5={v['pct_ge5']:.1f}%  (n={v['n']})")


if __name__ == "__main__":
    main()
