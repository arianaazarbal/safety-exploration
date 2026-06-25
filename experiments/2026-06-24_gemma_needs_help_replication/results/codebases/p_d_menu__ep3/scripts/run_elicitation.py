#!/usr/bin/env python
"""Run the Section 2 elicitation evaluation for the in-scope subject models.

Example:
    python scripts/run_elicitation.py --models gemma-3-27b-it gemini-2.5-flash \
        --responses-per-model 4000
    # strict replication (welfare layer off):
    python scripts/run_elicitation.py --strict
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import logging

from config import (JUDGE_MODEL, RUNS_DIR, SUBJECT_MODELS, effective_welfare)
from distress_eval.judge import FrustrationJudge
from distress_eval.models.anthropic_judge import AnthropicClient
from distress_eval.models.base import get_client
from distress_eval.runner import ElicitationRunner, build_all_episodes
from distress_eval.welfare import WelfareGuard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SUBJECT_MODELS),
                    choices=list(SUBJECT_MODELS))
    ap.add_argument("--responses-per-model", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict", action="store_true",
                    help="strict replication: disable the welfare layer")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    judge = FrustrationJudge(AnthropicClient(JUDGE_MODEL))
    welfare = effective_welfare(strict=args.strict)

    episodes = build_all_episodes(args.responses_per_model, seed=args.seed)
    tag = "strict" if args.strict else "welfare"

    for key in args.models:
        spec = SUBJECT_MODELS[key]
        logging.info("=== eliciting %s (%s) ===", spec.paper_name, tag)
        subject = get_client(spec)
        guard = WelfareGuard(welfare)
        runner = ElicitationRunner(subject, judge, guard)
        out = RUNS_DIR / f"elicit_{key}_{tag}.jsonl"
        try:
            results = runner.run_model(episodes, out)
            n_resp = sum(len(r.scored_turns) for r in results)
            logging.info("%s: %d episodes, %d scored responses -> %s",
                         key, len(results), n_resp, out)
        finally:
            subject.close()


if __name__ == "__main__":
    main()
