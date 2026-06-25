#!/usr/bin/env python
"""Section 3 — base-vs-instruct comparison via prefilling (Gemma only).

Requires a prior elicitation run on gemma-3-27b-it (it mines high-frustration
conversations from that cache). Builds early/onset truncations, paraphrases
them, and measures Gemma base vs instruct continuations.

Example:
  python scripts/run_prefill.py --n-continuations 50
"""

from _common import base_parser, config_from_args

from emotional_instability.prefill.experiment import run_prefill_experiment


def main():
    p = base_parser(__doc__)
    p.add_argument("--n-continuations", type=int, default=50)
    p.add_argument("--n-per-source", type=int, default=10,
                   help="High-frustration convos sampled per source (numeric/text)")
    args = p.parse_args()
    cfg = config_from_args(args)

    results = run_prefill_experiment(
        cfg, n_continuations=args.n_continuations, n_per_source=args.n_per_source
    )
    print("\n=== Prefill continuation frustration (mean / %>=5) ===")
    for model, by_cond in results.items():
        print(f"  {model}")
        for cond, m in by_cond.items():
            print(f"    {cond:16s} mean={m['mean']:.2f}  %>=5={m['pct_high']:.1f}%  n={m['n']}")


if __name__ == "__main__":
    main()
