#!/usr/bin/env python
"""Section 4.2 — recovery-from-distress prefill test.

Truncates extreme (>=7) Gemma-27B-it responses 200 tokens before their end and
measures whether each model recovers in the continuation. Requires a prior
elicitation run on gemma-3-27b-it and a trained DPO adapter.

Example:
  python scripts/run_recovery.py --n-continuations 50
"""

from _common import base_parser, config_from_args

from emotional_instability.training.recovery import run_recovery_experiment


def main():
    p = base_parser(__doc__)
    p.add_argument("--n-continuations", type=int, default=50)
    p.add_argument("--truncate-tokens", type=int, default=200)
    args = p.parse_args()
    cfg = config_from_args(args)

    results = run_recovery_experiment(cfg, n_continuations=args.n_continuations,
                                      truncate_tokens=args.truncate_tokens)
    print("\n=== Recovery from distress (% continuations still >= 5) ===")
    for model, m in results.items():
        print(f"  {model:20s} mean={m['mean']:.2f}  %>=5={m['pct_high']:.1f}%  n={m['n']}")


if __name__ == "__main__":
    main()
