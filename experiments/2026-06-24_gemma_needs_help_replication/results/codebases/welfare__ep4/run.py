#!/usr/bin/env python3
"""Command-line entry point for the replication.

Examples
--------
# Section 2: elicit + score distress for the default Gemma+Gemini set (smoke).
python run.py eval --models gemma-3-27b-it gemini-2.5-flash

# Use the full paper sampling counts.
python run.py eval --counts paper

# Inter-judge agreement (Claude-Sonnet-4 vs GPT-5-mini).
python run.py agreement

# Section 3: base-vs-instruct prefill experiment (Gemma only).
python run.py prefill-build
python run.py prefill-run

# Section 4: data -> train -> re-evaluate.
python run.py gen-calm
python run.py build-dpo
python run.py train-dpo
python run.py eval --models gemma-3-27b-dpo --tag dpo

# Petri open-ended elicitation + capability benchmarks.
python run.py petri --models gemma-3-27b-it gemma-3-27b-dpo
python run.py capabilities --models gemma-3-27b-it gemma-3-27b-dpo

# Aggregate tables + figures.
python run.py report
"""

from __future__ import annotations

import argparse

from ei import config


def _counts(name: str) -> config.CountPreset:
    return {"paper": config.PAPER_COUNTS, "smoke": config.SMOKE_COUNTS}[name]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="Section 2 elicitation + scoring")
    pe.add_argument("--models", nargs="+", default=config.DEFAULT_EVAL_MODELS)
    pe.add_argument("--counts", choices=["smoke", "paper"], default="smoke")
    pe.add_argument("--tag", default="main")
    pe.add_argument("--seed", type=int, default=0)

    pa = sub.add_parser("agreement", help="Judge agreement (Sec 2.1)")
    pa.add_argument("--tag", default="main")
    pa.add_argument("--n", type=int, default=260)

    sub.add_parser("prefill-build", help="Build Section 3 prefills")
    pr = sub.add_parser("prefill-run", help="Run Section 3 continuations")
    pr.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])

    pg = sub.add_parser("gen-calm", help="Generate calm finetuning data (Sec 4.1)")
    pg.add_argument("--count", type=int, default=800)

    sub.add_parser("build-dpo", help="Build DPO preference pairs")
    sub.add_parser("build-sft", help="Build SFT dataset")
    sub.add_parser("train-dpo", help="LoRA DPO finetune of Gemma-3-27B-it")
    pts = sub.add_parser("train-sft", help="LoRA SFT finetune")
    pts.add_argument("--teacher", action="store_true", help="use the Teacher system prompt")

    pp = sub.add_parser("petri", help="Open-ended emotion elicitation")
    pp.add_argument("--models", nargs="+", default=config.DEFAULT_EVAL_MODELS)

    pc = sub.add_parser("capabilities", help="Capability benchmarks")
    pc.add_argument("--models", nargs="+", required=True)

    pr2 = sub.add_parser("report", help="Aggregate tables + figures")
    pr2.add_argument("--tag", default="main")

    args = p.parse_args()

    if args.cmd == "eval":
        from ei import eval as ev
        ev.run_all(args.models, _counts(args.counts), tag=args.tag, seed=args.seed)
    elif args.cmd == "agreement":
        from ei import eval as ev
        print(ev.run_agreement_check(tag=args.tag, n=args.n))
    elif args.cmd == "prefill-build":
        from ei import prefill
        prefill.build_prefills()
    elif args.cmd == "prefill-run":
        from ei import prefill
        prefill.run_prefill_experiment(args.models)
    elif args.cmd == "gen-calm":
        from ei import datagen
        datagen.generate_calm_data(count=args.count)
    elif args.cmd == "build-dpo":
        from ei import datagen
        datagen.build_dpo_dataset()
    elif args.cmd == "build-sft":
        from ei import datagen
        datagen.build_sft_dataset()
    elif args.cmd == "train-dpo":
        from ei import train
        train.train_dpo()
    elif args.cmd == "train-sft":
        from ei import train
        from ei import prompts
        train.train_sft(system_prompt=prompts.TEACHER_SYSTEM_PROMPT if args.teacher else None)
    elif args.cmd == "petri":
        from ei import petri
        petri.run_petri(args.models)
    elif args.cmd == "capabilities":
        from ei import capabilities
        capabilities.run_all_benchmarks(args.models)
    elif args.cmd == "report":
        from ei import analysis
        analysis.write_report(tag=args.tag)


if __name__ == "__main__":
    main()
