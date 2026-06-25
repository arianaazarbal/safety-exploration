#!/usr/bin/env python
"""§2 distress-elicitation eval for the in-scope Gemma/Gemini targets.

Produces runs/eval/<model>_records.jsonl plus summary.json + per_turn_curves.json.
Use --smoke for a tiny run, --models to subset, --limit to cap per-condition counts.
"""
import argparse

import _path  # noqa: F401  (sys.path bootstrap)
from gemma_distress import config_shim as cfg
from gemma_distress.eval.run_eval import aggregate, run_model_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(cfg.TARGET_MODELS))
    ap.add_argument("--limit", type=int, default=None,
                    help="max responses per condition (default: paper counts)")
    ap.add_argument("--smoke", action="store_true", help="tiny run (limit=2)")
    args = ap.parse_args()

    limit = 2 if args.smoke else args.limit
    paths = []
    for handle in args.models:
        paths.append(run_model_eval(handle, limit_per_condition=limit))
    aggregate(paths)


if __name__ == "__main__":
    main()
