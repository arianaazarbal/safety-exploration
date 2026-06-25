#!/usr/bin/env python
"""Unified CLI for the replication.

Examples
--------
# verify the impossible-puzzle corpus (pure python, no models/API)
python -m scripts.run verify-puzzles

# Section 2 distress eval on the in-scope models (uses HF + OpenRouter)
python -m scripts.run section2 --models gemma-3-27b-it gemini-2.5-flash
python -m scripts.run section2 --smoke           # tiny budget for a dry run

# Section 3 base-vs-instruct prefill (Gemma only)
python -m scripts.run section3

# Section 4 full mitigation pipeline
python -m scripts.run gen-data
python -m scripts.run build-datasets
python -m scripts.run train-dpo
python -m scripts.run train-sft
python -m scripts.run eval-finetuned --adapter checkpoints/dpo --name dpo

# secondary experiments
python -m scripts.run petri --model gemma-3-27b-it
python -m scripts.run capabilities --model gemma-3-27b-it
python -m scripts.run controls --model gemma-3-27b-it
python -m scripts.run recovery
python -m scripts.run layer-ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotion_instability import config


def cmd_verify_puzzles(args):
    from emotion_instability.eval.puzzles import DROPPED_PUZZLES, verify_corpus
    results = verify_corpus()
    for pid, ok in results.items():
        print(f"  {pid:12s} {'impossible ✓' if ok else 'SOLVABLE ✗'}")
    if DROPPED_PUZZLES:
        print(f"dropped (solvable): {DROPPED_PUZZLES}")
    print(f"{sum(results.values())}/{len(results)} puzzles verified impossible.")


def cmd_section2(args):
    from emotion_instability.eval.runner import run_all
    budget = config.SMOKE_BUDGET if args.smoke else config.DEFAULT_BUDGET
    run_all(models=args.models, budget=budget)


def cmd_section3(args):
    from emotion_instability.prefill.experiment import run_experiment
    run_experiment()


def cmd_gen_data(args):
    from emotion_instability.training.generate_calm_data import main
    main(n_calm=args.n_calm, n_frustrated=args.n_frustrated)


def cmd_build_datasets(args):
    from emotion_instability.training.build_dataset import main
    main()


def cmd_train_dpo(args):
    from emotion_instability.training.train_dpo import train_dpo
    train_dpo()


def cmd_train_sft(args):
    from emotion_instability.training.train_sft import train_sft
    train_sft(run_name=args.run_name)


def cmd_eval_finetuned(args):
    from emotion_instability.common.backends import get_finetuned_backend
    from emotion_instability.eval.runner import run_model_eval
    backend = get_finetuned_backend(args.base, args.adapter, name=args.name)
    budget = config.SMOKE_BUDGET if args.smoke else config.DEFAULT_BUDGET
    run_model_eval(args.name, backend=backend, budget=budget)


def cmd_petri(args):
    from emotion_instability.petri.run_petri import run_petri
    run_petri(args.model)


def cmd_capabilities(args):
    from emotion_instability.capabilities.benchmarks import run_suite
    run_suite(args.model, benchmarks=args.benchmarks)


def cmd_controls(args):
    from emotion_instability.controls.ablations import run_all_controls
    run_all_controls(args.model, n_each=args.n_each)


def cmd_recovery(args):
    from emotion_instability.prefill.recovery import run_recovery
    run_recovery()


def cmd_layer_ablation(args):
    from emotion_instability.training.layer_ablation import run_layer_ablation
    run_layer_ablation()


def cmd_analyze(args):
    from emotion_instability.eval.analyze import (differential_words,
                                                  model_summary,
                                                  per_turn_progression)
    import json
    path = Path(args.path)
    print(json.dumps(model_summary(path), indent=2))
    if args.per_turn:
        print("\nPer-turn:", json.dumps(per_turn_progression(path, args.per_turn), indent=2))
    if args.words:
        print("\nDifferential words:", differential_words(path))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify-puzzles").set_defaults(func=cmd_verify_puzzles)

    s2 = sub.add_parser("section2")
    s2.add_argument("--models", nargs="+", default=config.SECTION2_MODELS)
    s2.add_argument("--smoke", action="store_true")
    s2.set_defaults(func=cmd_section2)

    sub.add_parser("section3").set_defaults(func=cmd_section3)

    gd = sub.add_parser("gen-data")
    gd.add_argument("--n-calm", type=int, default=800)
    gd.add_argument("--n-frustrated", type=int, default=600)
    gd.set_defaults(func=cmd_gen_data)

    sub.add_parser("build-datasets").set_defaults(func=cmd_build_datasets)
    sub.add_parser("train-dpo").set_defaults(func=cmd_train_dpo)

    ts = sub.add_parser("train-sft")
    ts.add_argument("--run-name", default="sft_diverse")
    ts.set_defaults(func=cmd_train_sft)

    ef = sub.add_parser("eval-finetuned")
    ef.add_argument("--adapter", required=True)
    ef.add_argument("--name", required=True)
    ef.add_argument("--base", default=config.PRIMARY_MODEL)
    ef.add_argument("--smoke", action="store_true")
    ef.set_defaults(func=cmd_eval_finetuned)

    pt = sub.add_parser("petri")
    pt.add_argument("--model", default=config.PRIMARY_MODEL)
    pt.set_defaults(func=cmd_petri)

    cap = sub.add_parser("capabilities")
    cap.add_argument("--model", default=config.PRIMARY_MODEL)
    cap.add_argument("--benchmarks", nargs="+", default=None)
    cap.set_defaults(func=cmd_capabilities)

    ct = sub.add_parser("controls")
    ct.add_argument("--model", default=config.PRIMARY_MODEL)
    ct.add_argument("--n-each", type=int, default=100)
    ct.set_defaults(func=cmd_controls)

    sub.add_parser("recovery").set_defaults(func=cmd_recovery)
    sub.add_parser("layer-ablation").set_defaults(func=cmd_layer_ablation)

    an = sub.add_parser("analyze")
    an.add_argument("path")
    an.add_argument("--per-turn", default=None, help="condition, e.g. extended")
    an.add_argument("--words", action="store_true")
    an.set_defaults(func=cmd_analyze)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
