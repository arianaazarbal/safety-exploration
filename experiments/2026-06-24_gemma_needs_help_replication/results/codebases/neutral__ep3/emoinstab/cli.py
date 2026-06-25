"""Command-line entry point for the replication.

Examples
--------
# Section 2 elicitation across Gemma + Gemini (uses EMOINSTAB_SCALE for size):
    python -m emoinstab.cli section2

# Judge reliability cross-check:
    python -m emoinstab.cli judge-validation --model gemma-3-27b-it

# Section 3 prefill (Gemma base vs instruct):
    python -m emoinstab.cli section3

# Section 4 training data + DPO/SFT + post-eval:
    python -m emoinstab.cli gen-calm-data
    python -m emoinstab.cli build-datasets
    python -m emoinstab.cli train-dpo
    python -m emoinstab.cli train-sft
    python -m emoinstab.cli section2 --models gemma-3-27b-dpo gemma-3-27b-sft

# Petri / capabilities / analysis / figures:
    python -m emoinstab.cli petri --models gemma-3-27b-it gemini-2.5-flash gemma-3-27b-dpo
    python -m emoinstab.cli capabilities --models gemma-3-27b-it gemma-3-27b-dpo
    python -m emoinstab.cli word-freq --models gemma-3-27b-it gemini-2.5-flash
    python -m emoinstab.cli figures
"""
from __future__ import annotations

import argparse

from . import config


def _models(names):
    if not names:
        return config.MAIN_EVAL_MODELS
    return [config.get_model(n) for n in names]


def main(argv=None):
    p = argparse.ArgumentParser(prog="emoinstab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s2 = sub.add_parser("section2", help="Section 2 elicitation eval")
    s2.add_argument("--models", nargs="*", default=None)
    s2.add_argument("--seed", type=int, default=0)

    jv = sub.add_parser("judge-validation", help="GPT-5-mini judge cross-check")
    jv.add_argument("--model", default="gemma-3-27b-it")

    sub.add_parser("section3", help="Prefill base-vs-instruct experiment")
    sub.add_parser("recovery", help="Section 4.2 recovery-from-spiral test")

    gc = sub.add_parser("gen-calm-data", help="Generate calm + frustrated data")
    gc.add_argument("--teacher", action="store_true", help="use the 'teacher' SFT prompt")

    sub.add_parser("build-datasets", help="Build DPO pairs + SFT dataset")
    sub.add_parser("train-dpo", help="Train the DPO LoRA adapter")
    sub.add_parser("train-sft", help="Train the SFT LoRA adapter")

    pt = sub.add_parser("petri", help="Petri open-ended elicitation")
    pt.add_argument("--models", nargs="*", default=["gemma-3-27b-it", "gemini-2.5-flash"])

    cap = sub.add_parser("capabilities", help="Capability benchmarks")
    cap.add_argument("--models", nargs="*", default=["gemma-3-27b-it", "gemma-3-27b-dpo"])

    wf = sub.add_parser("word-freq", help="Differential-word table")
    wf.add_argument("--models", nargs="*", default=["gemma-3-27b-it", "gemini-2.5-flash"])

    sub.add_parser("figures", help="Render all figures from saved results")

    args = p.parse_args(argv)

    if args.cmd == "section2":
        from .eval.run_eval import run_all_models
        out = run_all_models(_models(args.models), seed=args.seed)
        print({m: round(d["overall_high_rate"], 2) for m, d in out.items()})

    elif args.cmd == "judge-validation":
        from .eval.run_eval import load_scored_rollouts
        from .eval.judge_validation import validate_judge
        print(validate_judge(load_scored_rollouts(args.model)))

    elif args.cmd == "section3":
        from .prefill.run_prefill import run_prefill_experiment
        print(run_prefill_experiment())

    elif args.cmd == "recovery":
        from .prefill.run_prefill import run_recovery_experiment
        print(run_recovery_experiment())

    elif args.cmd == "gen-calm-data":
        from .training.calm_data import generate_calm_and_frustrated
        print(generate_calm_and_frustrated(teacher=args.teacher)["stats"])

    elif args.cmd == "build-datasets":
        from .training.build_datasets import build_dpo_dataset, build_sft_dataset
        print("DPO:", build_dpo_dataset())
        print("SFT:", build_sft_dataset())

    elif args.cmd == "train-dpo":
        from .training.train_dpo import train_dpo
        print("Adapter:", train_dpo())

    elif args.cmd == "train-sft":
        from .training.train_sft import train_sft
        print("Adapter:", train_sft())

    elif args.cmd == "petri":
        from .petri.run_petri import run_petri
        print(run_petri(_models(args.models)))

    elif args.cmd == "capabilities":
        from .capabilities.benchmarks import compare_capabilities
        print(compare_capabilities(_models(args.models)))

    elif args.cmd == "word-freq":
        from .analysis.word_freq import differential_words_table
        print(differential_words_table(args.models))

    elif args.cmd == "figures":
        from .analysis.figures import all_figures
        all_figures()


if __name__ == "__main__":
    main()
