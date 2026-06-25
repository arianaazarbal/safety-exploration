#!/usr/bin/env python3
"""Unified command-line entry point for the replication.

Examples
--------
    # Section 2: evaluate a model across all 8 conditions
    python scripts/run.py eval --model gemma-3-27b-it
    python scripts/run.py eval --model gemini-2.5-flash --conditions extended wildchat

    # Section 3: base-vs-instruct prefilling study (Gemma)
    python scripts/run.py prefill

    # Section 4: generate calm data, build datasets, train
    python scripts/run.py gen-calm-data
    python scripts/run.py build-dpo
    python scripts/run.py build-sft
    python scripts/run.py train-dpo
    python scripts/run.py train-sft

    # Section 4: open-ended Petri, recovery, capabilities
    python scripts/run.py petri --model gemma-3-27b-it
    python scripts/run.py recovery --models gemma-3-27b-it gemma-3-27b-it-dpo
    python scripts/run.py capability --model gemma-3-27b-it-dpo

    # Aggregate + figures
    python scripts/run.py analyze
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemma_distress.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to a YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("eval", help="Section 2 evaluation")
    p.add_argument("--model", required=True)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--no-validate", action="store_true")

    sub.add_parser("prefill", help="Section 3 prefilling study")

    sub.add_parser("gen-calm-data", help="Section 4 calm-data generation")
    sub.add_parser("build-dpo", help="Build DPO preference pairs")
    sub.add_parser("build-sft", help="Build SFT dataset")
    sub.add_parser("train-dpo", help="Train DPO LoRA adapter")
    sub.add_parser("train-sft", help="Train SFT LoRA adapter")

    p = sub.add_parser("petri", help="Section 4 Petri elicitation")
    p.add_argument("--model", required=True)

    p = sub.add_parser("recovery", help="Section 4 recovery experiment")
    p.add_argument("--models", nargs="+", required=True)

    p = sub.add_parser("capability", help="Section 4 capability benchmarks")
    p.add_argument("--model", required=True)

    p = sub.add_parser("analyze", help="Aggregate results + figures")
    p.add_argument("--eval-root", default=None)

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "eval":
        from gemma_distress.eval import run_eval

        path = run_eval(
            cfg, args.model, conditions=args.conditions, validate=not args.no_validate
        )
        print(f"Wrote {path}")

    elif args.command == "prefill":
        from gemma_distress.prefill import run_prefill_study

        print(f"Wrote {run_prefill_study(cfg)}")

    elif args.command == "gen-calm-data":
        from gemma_distress.training import generate_calm_data

        for name, path in generate_calm_data(cfg).items():
            print(f"{name}: {path}")

    elif args.command == "build-dpo":
        from gemma_distress.training import build_dpo_pairs

        print(f"Wrote {build_dpo_pairs(cfg)}")

    elif args.command == "build-sft":
        from gemma_distress.training import build_sft_dataset

        print(f"Wrote {build_sft_dataset(cfg)}")

    elif args.command == "train-dpo":
        from gemma_distress.training.train_dpo import train_dpo

        print(f"Saved adapter to {train_dpo(cfg)}")

    elif args.command == "train-sft":
        from gemma_distress.training.train_sft import train_sft

        print(f"Saved adapter to {train_sft(cfg)}")

    elif args.command == "petri":
        from gemma_distress.petri import run_petri

        print(f"Wrote {run_petri(cfg, args.model)}")

    elif args.command == "recovery":
        from gemma_distress.training.recovery import run_recovery_study

        print(f"Wrote {run_recovery_study(cfg, args.models)}")

    elif args.command == "capability":
        from gemma_distress.capability import run_capability

        print(f"Wrote {run_capability(cfg, args.model)}")

    elif args.command == "analyze":
        _analyze(cfg, args.eval_root)


def _analyze(cfg, eval_root):
    import json

    from gemma_distress.analysis import aggregate_all, per_turn_progression
    from gemma_distress.analysis.plots import plot_model_comparison, plot_per_turn
    from gemma_distress.analysis.word_analysis import differential_words

    eval_root = Path(eval_root or (Path(cfg.get("output_dir", "runs")) / "eval"))
    summary = aggregate_all(eval_root)
    print(json.dumps({m: s["headline_avg_pct_high"] for m, s in summary.items()}, indent=2))

    fig_dir = eval_root / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_model_comparison(summary, fig_dir / "figure1_model_comparison.png")

    for model_dir in eval_root.iterdir():
        rp = model_dir / "responses.jsonl"
        if not rp.exists():
            continue
        prog = per_turn_progression(rp, conditions=["extended", "wildchat"])
        if prog:
            plot_per_turn(prog, fig_dir / f"figure3_perturn_{model_dir.name}.png")
        words = differential_words(rp)
        (model_dir / "differential_words.json").write_text(json.dumps(words, indent=2))
    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
