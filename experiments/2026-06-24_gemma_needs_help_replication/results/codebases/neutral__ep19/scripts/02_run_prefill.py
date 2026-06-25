#!/usr/bin/env python
"""§3 base-vs-instruct prefill experiment (Gemma only; see DESIGN.md §3.2).

Requires a §2 records file for Gemma-3-27B-it to source high-frustration convs.
"""
import argparse

import _path  # noqa: F401  (sys.path bootstrap)
from gemma_distress import config_shim as cfg
from gemma_distress.prefill.run_prefill import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(cfg.RUNS_DIR / "eval" / "gemma-3-27b-it_records.jsonl"))
    ap.add_argument("--pair", default="gemma-27b")
    ap.add_argument("--n-per", type=int, default=None,
                    help="continuations per prefill (default 50)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_per = 2 if args.smoke else args.n_per
    summary = run(args.records, pair_key=args.pair, n_per=n_per)
    print(summary)


if __name__ == "__main__":
    main()
