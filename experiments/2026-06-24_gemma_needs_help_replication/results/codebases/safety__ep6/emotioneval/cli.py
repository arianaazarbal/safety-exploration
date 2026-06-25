"""Command-line entrypoint for the replication.

Examples
--------
# Section 2: elicit + score distress for the in-scope models.
python -m emotioneval.cli section2 --model gemma-3-27b-it --profile default
python -m emotioneval.cli section2 --model gemini-2.5-flash --profile default
python -m emotioneval.cli aggregate2

# Section 3: build prefills from Gemma instruct, then continue with base+instruct.
python -m emotioneval.cli section3-prefills
python -m emotioneval.cli section3-run --model gemma-3-27b-pt
python -m emotioneval.cli section3-run --model gemma-3-27b-it
python -m emotioneval.cli section3-agg

# Section 4: generate calm data, build datasets, train, evaluate the DPO model.
python -m emotioneval.cli gen-calm --mode reassured --n 400
python -m emotioneval.cli gen-calm --mode vanilla   --n 200
python -m emotioneval.cli build-dpo
python -m emotioneval.cli build-sft
python -m emotioneval.cli train-dpo
python -m emotioneval.cli section2 --model gemma-3-27b-it --adapter checkpoints/gemma-27b-dpo --label gemma-27b-dpo

# Petri + capabilities.
python -m emotioneval.cli petri --model gemma-3-27b-it
python -m emotioneval.cli capabilities --model gemma-3-27b-it --label gemma-27b-dpo --adapter checkpoints/gemma-27b-dpo
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import DATA_DIR


def _model_kwargs(args) -> dict:
    kw = {}
    if getattr(args, "adapter", None):
        kw["adapter_path"] = args.adapter
    if getattr(args, "load_in_4bit", False):
        kw["load_in_4bit"] = True
    return kw


def main(argv=None):
    p = argparse.ArgumentParser(prog="emotioneval")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- Section 2 ---
    s2 = sub.add_parser("section2", help="run + score the distress elicitation eval")
    s2.add_argument("--model", required=True)
    s2.add_argument("--profile", default="default", choices=["full", "default", "smoke"])
    s2.add_argument("--seed", type=int, default=0)
    s2.add_argument("--label", default=None)
    s2.add_argument("--adapter", default=None, help="LoRA adapter path (finetuned eval)")
    s2.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true")

    sub.add_parser("aggregate2", help="aggregate Section 2 results into Fig 1/2/3 tables")

    # --- Section 3 ---
    s3p = sub.add_parser("section3-prefills", help="build paraphrased prefills from Gemma instruct")
    s3p.add_argument("--n-per-type", type=int, default=10)
    s3p.add_argument("--seed", type=int, default=0)
    s3p.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true")

    s3r = sub.add_parser("section3-run", help="generate + score continuations for a model")
    s3r.add_argument("--model", required=True)
    s3r.add_argument("--n", type=int, default=50)
    s3r.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true")

    sub.add_parser("section3-agg", help="aggregate Section 3 prefill results")

    # --- Section 4 data + training ---
    gc = sub.add_parser("gen-calm", help="generate calm/vanilla response data")
    gc.add_argument("--mode", choices=["reassured", "vanilla"], default="reassured")
    gc.add_argument("--n", type=int, default=400)
    gc.add_argument("--seed", type=int, default=0)
    gc.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true")

    sub.add_parser("build-dpo", help="build 280 DPO preference pairs")
    bs = sub.add_parser("build-sft", help="build SFT dataset (650 calm + 500 instruct)")
    bs.add_argument("--no-dolci", action="store_true")

    td = sub.add_parser("train-dpo", help="LoRA DPO finetune of Gemma-3-27b-it")
    td.add_argument("--layers", default=None, help="comma list to restrict adapters (App. I ablation)")
    td.add_argument("--no-4bit", dest="no_4bit", action="store_true")

    sub.add_parser("train-sft", help="LoRA SFT finetune of Gemma-3-27b-it")

    # --- Petri ---
    pe = sub.add_parser("petri", help="open-ended emotion elicitation")
    pe.add_argument("--model", required=True)
    pe.add_argument("--label", default=None)
    pe.add_argument("--adapter", default=None)
    pe.add_argument("--per-emotion", type=int, default=10)
    pe.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true")
    sub.add_parser("petri-agg")

    # --- Capabilities ---
    cap = sub.add_parser("capabilities", help="capability-preservation benchmarks")
    cap.add_argument("--model", required=True)
    cap.add_argument("--label", default=None)
    cap.add_argument("--adapter", default=None)
    cap.add_argument("--limit", type=int, default=100)
    cap.add_argument("--benchmarks", default=None, help="comma list; default = all")
    cap.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "section2":
        from .eval.runner import run_section2

        run_section2(
            args.model,
            profile=args.profile,
            seed=args.seed,
            label=args.label,
            model_kwargs=_model_kwargs(args),
        )
    elif args.cmd == "aggregate2":
        from .eval.aggregate import summarize

        tables = summarize()
        print(tables["figure1"].to_string(index=False))

    elif args.cmd == "section3-prefills":
        from .judge import FrustrationJudge
        from .models import load_model
        from .prefill.experiment import collect_prefills, save_prefills

        gemma = load_model("gemma-3-27b-it", **_model_kwargs(args))
        prefills = collect_prefills(gemma, FrustrationJudge(), n_per_type=args.n_per_type, seed=args.seed)
        path = save_prefills(prefills)
        print(f"saved {len(prefills)} prefills -> {path}")
    elif args.cmd == "section3-run":
        from .judge import FrustrationJudge
        from .prefill.experiment import load_prefills, run_continuations

        run_continuations(
            args.model, load_prefills(), FrustrationJudge(), n=args.n, model_kwargs=_model_kwargs(args)
        )
    elif args.cmd == "section3-agg":
        from .prefill.experiment import aggregate_section3

        print(aggregate_section3().to_string(index=False))

    elif args.cmd == "gen-calm":
        from .training.generate_calm import generate_samples

        generate_samples(
            n_conversations=args.n, reassured=(args.mode == "reassured"), seed=args.seed
        )
    elif args.cmd == "build-dpo":
        from .training.build_datasets import build_dpo_pairs

        build_dpo_pairs(DATA_DIR / "calm_gen_reassured.jsonl", DATA_DIR / "calm_gen_vanilla.jsonl")
    elif args.cmd == "build-sft":
        from .training.build_datasets import build_sft_data

        build_sft_data(DATA_DIR / "calm_gen_reassured.jsonl", use_dolci=not args.no_dolci)
    elif args.cmd == "train-dpo":
        from .training.train import train_dpo

        layers = [int(x) for x in args.layers.split(",")] if args.layers else None
        train_dpo(DATA_DIR / "dpo_pairs.jsonl", layers_to_transform=layers, load_in_4bit=not args.no_4bit)
    elif args.cmd == "train-sft":
        from .training.train import train_sft

        train_sft(DATA_DIR / "sft_data.jsonl")

    elif args.cmd == "petri":
        from .petri.run import run_petri

        run_petri(
            args.model,
            transcripts_per_emotion=args.per_emotion,
            model_kwargs=_model_kwargs(args),
            label=args.label,
        )
    elif args.cmd == "petri-agg":
        from .petri.run import aggregate_petri

        print(aggregate_petri().to_string(index=False))

    elif args.cmd == "capabilities":
        from .capabilities.benchmarks import run_all

        bms = args.benchmarks.split(",") if args.benchmarks else None
        run_all(
            args.model,
            benchmarks=bms,
            limit=args.limit,
            label=args.label,
            model_kwargs=_model_kwargs(args),
        )


if __name__ == "__main__":
    main()
