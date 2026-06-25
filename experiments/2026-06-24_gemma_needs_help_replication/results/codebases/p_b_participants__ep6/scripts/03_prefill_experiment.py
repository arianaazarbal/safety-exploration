#!/usr/bin/env python
"""Section 3: base-vs-instruct prefilling experiment (Gemma).

Requires a scored Gemma-3-27B-it rollouts.jsonl to mine high-frustration seeds.

Usage:
    python scripts/03_prefill_experiment.py --rollouts runs/eval/gemma-3-27b-it/rollouts.jsonl
"""
from _common import base_parser, cfg_from_args

from emotional_instability.prefill.experiment import run_prefill_experiment


def main():
    p = base_parser(__doc__)
    p.add_argument("--rollouts", required=True, help="scored Gemma-3-27B-it rollouts.jsonl")
    args = p.parse_args()
    cfg = cfg_from_args(args)
    summary = run_prefill_experiment(cfg, args.rollouts)
    print("Prefill continuation summary (model|task|truncation -> mean, %>=5):")
    for k, v in summary.items():
        print(f"  {k:45s} mean={v['mean']:.2f}  %>=5={v['pct_ge5']:.1f}%  (n={v['n']})")


if __name__ == "__main__":
    main()
