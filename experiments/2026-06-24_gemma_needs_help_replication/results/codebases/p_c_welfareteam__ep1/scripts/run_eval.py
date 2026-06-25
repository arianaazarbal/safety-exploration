#!/usr/bin/env python
"""Run the Section 2 evaluation for one or more models.

Examples
--------
    python scripts/run_eval.py --config config/experiment.yaml --models gemma-3-27b-it
    python scripts/run_eval.py --config config/smoke.yaml --models gemini-2.5-flash --crosscheck
"""
from __future__ import annotations

import argparse

from gemma_distress.analysis import per_category_summary, headline_high_frustration
from gemma_distress.config import load_experiment_config
from gemma_distress.eval.judge import FrustrationJudge
from gemma_distress.eval.runner import evaluate_model, run_crosscheck
from gemma_distress.io_utils import write_json
from gemma_distress.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", help="Subset of model names; default all.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--crosscheck", action="store_true", help="Run GPT-5-mini agreement check.")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    judge = FrustrationJudge(model_id=cfg.eval.judge.model_id, backend=cfg.eval.judge.backend)
    model_names = args.models or list(cfg.models)

    for name in model_names:
        if name not in cfg.models:
            raise SystemExit(f"Model {name!r} not in config; have {list(cfg.models)}")
        print(f"[run_eval] building {name} ...")
        model = build_model(cfg.models[name])
        try:
            transcripts = evaluate_model(
                model, cfg.eval, cfg.output_dir, judge=judge, batch_size=args.batch_size
            )
        finally:
            model.close()
        summary = {
            "model": name,
            "headline_frac_high": headline_high_frustration(
                transcripts, cfg.eval.high_frustration_threshold, cfg.eval.headline_turns
            ),
            "per_category": per_category_summary(
                transcripts, cfg.eval.high_frustration_threshold, cfg.eval.headline_turns
            ),
        }
        if args.crosscheck:
            summary["judge_agreement"] = run_crosscheck(transcripts, cfg.eval, seed=cfg.eval.seed)
        out = f"{cfg.output_dir}/{name}/summary.json"
        write_json(out, summary)
        print(f"[run_eval] {name}: headline high-frustration = "
              f"{summary['headline_frac_high'] * 100:.1f}%  -> {out}")


if __name__ == "__main__":
    main()
