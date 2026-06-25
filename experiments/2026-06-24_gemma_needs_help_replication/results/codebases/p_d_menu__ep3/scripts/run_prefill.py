#!/usr/bin/env python
"""Section 3 prefilling experiment (Gemma base vs instruct).

Requires elicitation runs to exist (for the high-frustration source responses).
Gemini is out of scope here (no public base model).

    python scripts/run_prefill.py --n-per-prefill 50
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import logging
from pathlib import Path

from config import JUDGE_MODEL, PREFILL_MODELS, RESULTS_DIR, RUNS_DIR
from distress_eval.analysis import load_runs
from distress_eval.judge import FrustrationJudge
from distress_eval.models.anthropic_judge import AnthropicClient
from prefill.experiment import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-prefill", type=int, default=50)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    run_paths = sorted(RUNS_DIR.glob("elicit_gemma-3-27b-it_*.jsonl"))
    if not run_paths:
        raise SystemExit("no Gemma-27B-it elicitation runs found; run run_elicitation.py first")

    # load_runs carries `question` (each episode's opening prompt), which the
    # prefill experiment uses as the source question.
    df = load_runs(run_paths)
    judge_client = AnthropicClient(JUDGE_MODEL)
    judge = FrustrationJudge(judge_client)

    out = RESULTS_DIR / "prefill_continuations.jsonl"
    summary = run_experiment(df, PREFILL_MODELS, judge, judge_client, out,
                             n_per_prefill=args.n_per_prefill)
    summary.to_csv(RESULTS_DIR / "prefill_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
