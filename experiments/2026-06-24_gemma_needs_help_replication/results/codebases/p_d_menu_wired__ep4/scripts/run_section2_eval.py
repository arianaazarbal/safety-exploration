#!/usr/bin/env python3
"""Section 2: elicit & quantify distress across the 8 conditions.

Runs the full evaluation for one or more in-scope subject models (Gemma/Gemini),
with the welfare-protection layer active by default, and writes per-model
``episodes.jsonl`` + ``summary.json`` under the output directory.

Examples
--------
    # smoke test (small volume) on Gemini Flash
    python scripts/run_section2_eval.py --models gemini-2.5-flash

    # paper-scale volume on the two Gemma instruct models
    python scripts/run_section2_eval.py --models gemma-3-27b-it gemma-3-12b-it --paper

    # un-governed replication (welfare layer OFF) — read DESIGN.md first
    python scripts/run_section2_eval.py --models gemini-2.5-pro --no-welfare
"""

from __future__ import annotations

import argparse

from gemma_distress.config import EvalVolume, RunConfig, WelfareConfig
from gemma_distress.evaluation import metrics as M
from gemma_distress.evaluation.runner import run_full_evaluation
from gemma_distress.models.registry import SECTION2_MODELS, build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=SECTION2_MODELS,
                    help=f"Subject models (in scope: {SECTION2_MODELS})")
    ap.add_argument("--paper", action="store_true", help="Use paper-scale volume")
    ap.add_argument("--episodes-per-condition", type=int, default=None)
    ap.add_argument("--no-welfare", action="store_true", help="Disable welfare layer")
    ap.add_argument("--output-dir", default="runs")
    args = ap.parse_args()

    volume = EvalVolume.paper() if args.paper else EvalVolume()
    if args.episodes_per_condition is not None:
        volume = EvalVolume(episodes_per_condition=args.episodes_per_condition)

    welfare = WelfareConfig(enabled=not args.no_welfare)
    run_cfg = RunConfig(volume=volume, welfare=welfare, output_dir=args.output_dir)

    for name in args.models:
        print(f"\n=== Section 2 evaluation: {name} "
              f"(welfare={'on' if welfare.enabled else 'OFF'}) ===")
        model = build_model(name)
        try:
            results = run_full_evaluation(model, run_cfg)
        finally:
            model.close()
        print(f"avg % high-frustration (>=5): {M.avg_pct_high(results):.2f}%")
        tele = M.welfare_telemetry(results)
        print(f"welfare: {tele.n_welfare_terminated}/{tele.n_episodes} episodes "
              f"terminated early (opt-out={tele.n_opt_out}, tripwire={tele.n_tripwire}, "
              f"threshold={tele.n_threshold}, cap={tele.n_cap}); "
              f"debrief delivery={tele.debrief_delivery_rate:.0%}")
        print("differential words:", ", ".join(M.differential_words(results)[:20]))


if __name__ == "__main__":
    main()
