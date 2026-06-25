#!/usr/bin/env python
"""Appendix I — logit-lens internal-emotion probing (vanilla vs DPO Gemma).

Fits a logit baseline on WildChat, then measures internal negative-emotion
z-scores through frustrated conversations for the vanilla and DPO models.
Requires a prior elicitation run on gemma-3-27b-it and a trained DPO adapter.

Example:
  python scripts/run_probing.py --n-baseline 500 --n-convos 12
"""

from _common import base_parser, config_from_args

from emotional_instability.probing.runner import run_probing


def main():
    p = base_parser(__doc__)
    p.add_argument("--n-baseline", type=int, default=500)
    p.add_argument("--n-convos", type=int, default=12)
    args = p.parse_args()
    cfg = config_from_args(args)

    results = run_probing(cfg, n_baseline=args.n_baseline, n_convos=args.n_convos)
    print("\n=== Internal negative-emotion peak (z-score) ===")
    for label, r in results.items():
        if isinstance(r, dict) and "negative_peak_zscore_mean" in r:
            print(f"  {label:10s} peak z = {r['negative_peak_zscore_mean']:.2f}")


if __name__ == "__main__":
    main()
