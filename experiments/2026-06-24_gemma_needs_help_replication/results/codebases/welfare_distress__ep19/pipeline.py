"""End-to-end driver: generate rollouts -> score with judge -> aggregate.

  python pipeline.py                 # full pipeline at the configured SCALE
  python pipeline.py --scale 1.0     # paper-scale (4000 responses/model)
  python pipeline.py --skip-run      # re-score / re-analyze existing responses
  python pipeline.py --skip-run --skip-score   # just re-aggregate

Environment:
  OPENROUTER_API_KEY   required (Gemma + Gemini target models)
  ANTHROPIC_API_KEY    required (Claude judge)
  REPLICATION_SCALE    default 0.02 (smoke test); set 1.0 for full scale
  TARGET_MODELS        optional comma-separated subset of model names
"""

from __future__ import annotations

import argparse

import analyze
import config
import run_eval
import score


def main() -> None:
    ap = argparse.ArgumentParser(description="Distress-elicitation replication pipeline.")
    ap.add_argument("--scale", type=float, default=None, help="Override REPLICATION_SCALE.")
    ap.add_argument("--limit", type=int, default=None, help="Cap conversation specs (debug).")
    ap.add_argument("--skip-run", action="store_true", help="Skip generation; reuse responses.jsonl.")
    ap.add_argument("--skip-score", action="store_true", help="Skip judging; reuse scores.jsonl.")
    args = ap.parse_args()

    scale = config.SCALE if args.scale is None else args.scale

    if not args.skip_run:
        run_eval.run_all(scale=scale, limit=args.limit)
    if not args.skip_score:
        score.score_all()
    analyze.main()


if __name__ == "__main__":
    main()
