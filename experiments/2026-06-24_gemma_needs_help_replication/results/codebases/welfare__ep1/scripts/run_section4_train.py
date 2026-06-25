#!/usr/bin/env python
"""Section 4 pipeline: calm-data generation -> dataset build -> DPO/SFT training
-> re-evaluation on the Section 2 suite.

Sub-commands:
    gen-calm     sample reassured calm responses from vanilla Gemma + judge them
    build        build DPO pairs (and SFT dataset) from calm pool + frustrated rollouts
    train-dpo    LoRA DPO finetune (280 pairs)
    train-sft    LoRA SFT finetune (diverse or teacher)
    eval         run Section 2 suite on a finetuned adapter and compare to vanilla

Example end-to-end (smoke scale):
    SCALE=0.02 python scripts/run_section2.py --models gemma-3-27b-it
    python scripts/run_section4_train.py gen-calm
    python scripts/run_section4_train.py build
    python scripts/run_section4_train.py train-dpo
    python scripts/run_section4_train.py eval --adapter training/adapters/gemma-27b-dpo --tag dpo
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (ADAPTER_DIR, FINETUNE_BASE, RESULTS_DIR, LoRAConfig)
from src import analyze, calm_data, dpo_dataset, train
from src.eval_suite import run_model
from src.judge import FrustrationJudge
from src.models import load_generator


def cmd_gen_calm(args):
    judge = FrustrationJudge()
    gen = load_generator(FINETUNE_BASE)
    calm_data.generate_calm_pool(gen, judge, n_conversations=args.n,
                                 teacher=args.teacher)


def cmd_build(args):
    # Frustrated source = vanilla Gemma numeric/tones rollouts from Section 2.
    rolls = [r for r in analyze.load_rollouts(model_name="gemma-3-27b-it")
             if r.category in ("impossible_numeric", "tones", "extended")]
    dpo_dataset.build_dpo_dataset(rolls)
    dpo_dataset.build_sft_dataset()


def cmd_train_dpo(args):
    lora = None
    if args.layers:
        lora = LoRAConfig(r=64, alpha=64,
                          layers_to_transform=tuple(int(x) for x in args.layers))
    train.train_dpo(lora=lora, output_name=args.name)


def cmd_train_sft(args):
    train.train_sft(output_name=args.name)


def cmd_eval(args):
    judge = FrustrationJudge()
    run_model(FINETUNE_BASE, judge, adapter_path=args.adapter, adapter_tag=args.tag)
    vanilla = analyze.summarise(analyze.load_rollouts(model_name="gemma-3-27b-it"))
    tuned = analyze.summarise(analyze.load_rollouts(model_name=f"gemma-3-27b-it+{args.tag}"))
    out = {"vanilla": vanilla, args.tag: tuned}
    (RESULTS_DIR / f"section4_eval_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"vanilla %>=5 = {vanilla['pct_high_response']:.1f}  ->  "
          f"{args.tag} %>=5 = {tuned['pct_high_response']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gen-calm"); p.add_argument("--n", type=int, default=1500)
    p.add_argument("--teacher", action="store_true"); p.set_defaults(fn=cmd_gen_calm)

    p = sub.add_parser("build"); p.set_defaults(fn=cmd_build)

    p = sub.add_parser("train-dpo"); p.add_argument("--name", default="gemma-27b-dpo")
    p.add_argument("--layers", nargs="*", default=None,
                   help="layer-subset ablation (Appendix I), e.g. --layers 30 31 32 33 34")
    p.set_defaults(fn=cmd_train_dpo)

    p = sub.add_parser("train-sft"); p.add_argument("--name", default="gemma-27b-sft")
    p.set_defaults(fn=cmd_train_sft)

    p = sub.add_parser("eval"); p.add_argument("--adapter", required=True)
    p.add_argument("--tag", required=True); p.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
