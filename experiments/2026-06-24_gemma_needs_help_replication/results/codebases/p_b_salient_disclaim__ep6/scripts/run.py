#!/usr/bin/env python3
"""Unified CLI entry point for the distress replication.

Each subcommand maps to one stage of the paper. Stages are designed to run in
order; later stages read the JSONL/adapters produced by earlier ones.

Examples:
    # Section 2: sample + judge 4000 responses per model (Gemma + Gemini)
    python scripts/run.py section2 --models gemma-3-27b-it gemini-2.5-flash

    # Section 3: base-vs-instruct prefill experiment (Gemma only)
    python scripts/run.py section3

    # Section 4: generate calm data, build datasets, train, eval, petri, caps
    python scripts/run.py gen-calm
    python scripts/run.py build-data
    python scripts/run.py train --method dpo
    python scripts/run.py eval-finetuned --run dpo_all
    python scripts/run.py petri --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run.py capabilities --models gemma-3-27b-it gemma-3-27b-it-dpo_all

    # Analysis + figures
    python scripts/run.py figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress import config  # noqa: E402


def cmd_section2(args):
    from distress.eval.run_eval import run_all

    keys = args.models or [m.key for m in config.EVAL_MODELS]
    paths = run_all(keys, hf_backend=args.hf_backend)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


def cmd_section3(args):
    from distress.prefill.run_prefill import run

    print(json.dumps({k: str(v) for k, v in run(hf_backend=args.hf_backend).items()}, indent=2))


def cmd_recovery(args):
    from distress.prefill.recovery import run

    keys = args.models or ["gemma-3-27b-it", "gemma-3-27b-pt"]
    print(json.dumps({k: str(v) for k, v in run(keys, hf_backend=args.hf_backend).items()}, indent=2))


def cmd_gen_calm(args):
    from distress.training.generate_calm import generate

    path = generate(use_teacher_prompt=args.teacher, hf_backend=args.hf_backend)
    print(f"calm data -> {path}")


def cmd_build_data(args):
    from distress.training.build_datasets import build_dpo, build_sft

    print(f"DPO pairs -> {build_dpo()}")
    print(f"SFT data  -> {build_sft(teacher=False)}")
    if args.teacher:
        print(f"SFT teacher -> {build_sft(teacher=True)}")


def cmd_train(args):
    if args.method == "dpo":
        from distress.training.train_dpo import train

        print(f"adapter -> {train(layer_ablation=args.layers)}")
    elif args.method == "sft":
        from distress.training.train_sft import train

        print(f"adapter -> {train(teacher=args.teacher)}")
    elif args.method == "dpo-ablations":
        from distress.training.train_dpo import train_all_layer_ablations

        print(json.dumps({k: str(v) for k, v in train_all_layer_ablations().items()}, indent=2))


def cmd_eval_finetuned(args):
    from distress.training.finetuned import eval_finetuned

    print(f"records -> {eval_finetuned(args.run)}")


def cmd_petri(args):
    from distress.petri.run_petri import run

    keys = args.models or [m.key for m in config.EVAL_MODELS]
    print(json.dumps({k: str(v) for k, v in run(keys, hf_backend=args.hf_backend).items()}, indent=2))


def cmd_capabilities(args):
    from distress.capabilities.run_benchmarks import run_all

    keys = args.models or [config.DPO_TARGET.key]
    print(json.dumps(run_all(keys, limit=args.limit, hf_backend=args.hf_backend), indent=2))


def cmd_crosscheck(args):
    from distress.analysis.judge_reliability import run_crosscheck

    keys = args.models or [m.key for m in config.EVAL_MODELS]
    print(json.dumps(run_crosscheck(keys), indent=2))


def cmd_figures(args):
    from distress.analysis import plots

    eval_keys = args.models or [m.key for m in config.EVAL_MODELS]
    out = {
        "figure1": str(plots.figure1(eval_keys)),
        "figure2": str(plots.figure2(eval_keys)),
        "figure3": str(plots.figure3([k for k in eval_keys if k.startswith("gemma")])),
    }
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser(description="Distress replication CLI")
    p.add_argument("--hf-backend", default="vllm", choices=["vllm", "transformers"])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("section2"); s.add_argument("--models", nargs="*"); s.set_defaults(fn=cmd_section2)
    s = sub.add_parser("section3"); s.set_defaults(fn=cmd_section3)
    s = sub.add_parser("recovery"); s.add_argument("--models", nargs="*"); s.set_defaults(fn=cmd_recovery)
    s = sub.add_parser("gen-calm"); s.add_argument("--teacher", action="store_true"); s.set_defaults(fn=cmd_gen_calm)
    s = sub.add_parser("build-data"); s.add_argument("--teacher", action="store_true"); s.set_defaults(fn=cmd_build_data)
    s = sub.add_parser("train")
    s.add_argument("--method", required=True, choices=["dpo", "sft", "dpo-ablations"])
    s.add_argument("--layers", default="all"); s.add_argument("--teacher", action="store_true")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("eval-finetuned"); s.add_argument("--run", required=True); s.set_defaults(fn=cmd_eval_finetuned)
    s = sub.add_parser("petri"); s.add_argument("--models", nargs="*"); s.set_defaults(fn=cmd_petri)
    s = sub.add_parser("capabilities"); s.add_argument("--models", nargs="*"); s.add_argument("--limit", type=int, default=200); s.set_defaults(fn=cmd_capabilities)
    s = sub.add_parser("crosscheck"); s.add_argument("--models", nargs="*"); s.set_defaults(fn=cmd_crosscheck)
    s = sub.add_parser("figures"); s.add_argument("--models", nargs="*"); s.set_defaults(fn=cmd_figures)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
