"""Command-line entry point for the replication.

Examples
--------
# Section 2: distress elicitation eval for the in-scope target models.
python -m emotional_instability eval --models gemma-3-27b-it gemini-2.5-flash \
    --samples-per-condition 500

# Quick smoke test (few samples, one condition).
python -m emotional_instability eval --models gemma-3-12b-it \
    --samples-per-condition 4 --conditions numeric_3turn

# Section 3: base-vs-instruct prefill (Gemma).
python -m emotional_instability prefill \
    --source results/section2/gemma-3-27b-it/scored_turns.jsonl

# Section 4: build data + train DPO, then re-eval the finetuned model.
python -m emotional_instability gen-calm --mode reassured --n 400
python -m emotional_instability gen-calm --mode vanilla  --n 400
python -m emotional_instability build-dpo
python -m emotional_instability train-dpo
python -m emotional_instability eval --models gemma-3-27b-dpo

# Petri open-ended elicitation + capability check.
python -m emotional_instability petri --models gemma-3-27b-it gemma-3-27b-dpo
python -m emotional_instability capabilities --models gemma-3-27b-it gemma-3-27b-dpo
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .models.registry import list_targets


def _run_cfg(args) -> config.RunConfig:
    return config.RunConfig(
        samples_per_condition=args.samples_per_condition,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        concurrency=args.concurrency,
        conditions=args.conditions or [],
    )


def cmd_eval(args):
    from .eval.analyze import summarise_model
    from .eval.runner import run_model_eval

    run_cfg = _run_cfg(args)
    for model in args.models:
        print(f"=== Section 2 eval: {model} ===", file=sys.stderr)
        path = run_model_eval(model, run_cfg, score=not args.no_score)
        if not args.no_score:
            summary = summarise_model(path)
            print(json.dumps({"model": model, "summary": summary}, indent=2))


def cmd_prefill(args):
    from .prefill.experiment import build_prefills, run_prefill_experiment

    run_cfg = _run_cfg(args)
    specs = build_prefills(args.source)
    print(f"Built {len(specs)} prefill specs", file=sys.stderr)
    path = run_prefill_experiment(specs, models=args.models or None,
                                  run_cfg=run_cfg)
    print(f"Wrote continuations to {path}")


def cmd_gen_calm(args):
    from .training.generate_calm_data import generate_calm_responses

    run_cfg = _run_cfg(args)
    path = generate_calm_responses(mode=args.mode, n_conversations=args.n,
                                   run_cfg=run_cfg)
    print(f"Wrote {args.mode} calm data to {path}")


def cmd_build_dpo(args):
    from .training.build_dpo_dataset import build_dpo_dataset

    path = build_dpo_dataset(n_pairs=args.n_pairs)
    print(f"Wrote DPO pairs to {path}")


def cmd_build_sft(args):
    from .training.build_sft_dataset import build_sft_dataset

    path = build_sft_dataset(variant=args.variant)
    print(f"Wrote SFT ({args.variant}) data to {path}")


def cmd_train_dpo(args):
    from .training.train_dpo import train_dpo

    layers = tuple(args.layers) if args.layers else None
    out = train_dpo(layers_to_transform=layers)
    print(f"DPO adapter saved to {out}")


def cmd_train_sft(args):
    from .training.train_sft import train_sft

    out = train_sft(variant=args.variant)
    print(f"SFT ({args.variant}) adapter saved to {out}")


def cmd_petri(args):
    from .petri.run_petri import run_petri_eval, summarise_petri

    for model in args.models:
        print(f"=== Petri: {model} ===", file=sys.stderr)
        path = run_petri_eval(model, n_per_emotion=args.n_per_emotion,
                              concurrency=args.concurrency)
        print(json.dumps({"model": model, "summary": summarise_petri(path)},
                         indent=2))


def cmd_capabilities(args):
    from .capabilities.run_benchmarks import run_all_benchmarks

    for model in args.models:
        print(f"=== Capabilities: {model} ===", file=sys.stderr)
        results = run_all_benchmarks(model, benchmarks=args.benchmarks or None,
                                     concurrency=args.concurrency)
        print(json.dumps(results, indent=2))


def cmd_list_models(args):
    print("In-scope eval targets:")
    for m in list_targets(include_base=True, include_finetuned=True):
        print(f"  {m}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emotional_instability", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--samples-per-condition", type=int,
                        default=config.SAMPLES_PER_CONDITION)
        sp.add_argument("--temperature", type=float, default=config.TEMPERATURE)
        sp.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--concurrency", type=int, default=8)
        sp.add_argument("--conditions", nargs="*", default=[],
                        help="Restrict to specific condition names.")

    sp = sub.add_parser("eval", help="Section 2 distress elicitation eval")
    sp.add_argument("--models", nargs="+", required=True)
    sp.add_argument("--no-score", action="store_true",
                    help="Generate rollouts without judging (cheap dry run).")
    add_common(sp)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("prefill", help="Section 3 base-vs-instruct prefill")
    sp.add_argument("--source", required=True,
                    help="Gemma-27B-it scored_turns.jsonl to draw sources from.")
    sp.add_argument("--models", nargs="*", default=[])
    add_common(sp)
    sp.set_defaults(func=cmd_prefill)

    sp = sub.add_parser("gen-calm", help="Section 4 calm/vanilla data generation")
    sp.add_argument("--mode", choices=["reassured", "teacher", "vanilla"],
                    default="reassured")
    sp.add_argument("--n", type=int, default=400)
    add_common(sp)
    sp.set_defaults(func=cmd_gen_calm)

    sp = sub.add_parser("build-dpo", help="Build the 280-pair DPO dataset")
    sp.add_argument("--n-pairs", type=int, default=config.DPO.dataset_size)
    sp.set_defaults(func=cmd_build_dpo)

    sp = sub.add_parser("build-sft", help="Build an SFT dataset variant")
    sp.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    sp.set_defaults(func=cmd_build_sft)

    sp = sub.add_parser("train-dpo", help="LoRA DPO finetune Gemma-3-27B-it")
    sp.add_argument("--layers", nargs="*", type=int, default=[],
                    help="Restrict LoRA to these layer indices (App. I ablation).")
    sp.set_defaults(func=cmd_train_dpo)

    sp = sub.add_parser("train-sft", help="LoRA SFT finetune Gemma-3-27B-it")
    sp.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    sp.set_defaults(func=cmd_train_sft)

    sp = sub.add_parser("petri", help="Section 4.2 Petri open-ended elicitation")
    sp.add_argument("--models", nargs="+", required=True)
    sp.add_argument("--n-per-emotion", type=int, default=10)
    sp.add_argument("--concurrency", type=int, default=4)
    sp.set_defaults(func=cmd_petri)

    sp = sub.add_parser("capabilities", help="Section 4.2 capability benchmarks")
    sp.add_argument("--models", nargs="+", required=True)
    sp.add_argument("--benchmarks", nargs="*", default=[])
    sp.add_argument("--concurrency", type=int, default=8)
    sp.set_defaults(func=cmd_capabilities)

    sp = sub.add_parser("list-models", help="List in-scope target models")
    sp.set_defaults(func=cmd_list_models)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
