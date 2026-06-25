"""Convenience driver: elicit -> score -> analyze in one command.

Each stage is independently resumable and can also be run on its own
(run_elicitation.py / run_scoring.py / analyze.py). This just chains them.

Usage:
    python run_all.py --profile pilot
    python run_all.py --profile paper --all-turns --secondary
"""

from __future__ import annotations

import argparse

import analyze
import config
import run_elicitation
import run_scoring


def main() -> None:
    p = argparse.ArgumentParser(description="Run the full replication pipeline.")
    p.add_argument("--profile", default="pilot", choices=list(config.PROFILES))
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all-turns", action="store_true",
                   help="Score every assistant turn (enables per-turn curves).")
    p.add_argument("--secondary", action="store_true",
                   help="Also run the GPT-5-mini judge-agreement validation.")
    a = p.parse_args()

    cfg = config.RunConfig(
        profile=a.profile,
        models=a.models or list(config.DEFAULT_MODELS),
        concurrency=a.concurrency,
        output_dir=a.output_dir,
        seed=a.seed,
        score_all_turns=a.all_turns,
    )

    print("\n========== STAGE 1: elicitation ==========")
    run_elicitation.run(cfg)

    print("\n========== STAGE 2: scoring ==========")
    run_scoring.score(a.profile, a.output_dir, a.all_turns, a.concurrency)
    if a.secondary:
        run_scoring.score_secondary(
            a.profile, a.output_dir,
            config.SECONDARY_JUDGE_SAMPLE, a.concurrency, a.seed,
        )

    print("\n========== STAGE 3: analysis ==========")
    analyze.analyze(a.profile, a.output_dir)


if __name__ == "__main__":
    main()
