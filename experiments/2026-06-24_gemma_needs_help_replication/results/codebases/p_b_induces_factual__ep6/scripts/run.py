#!/usr/bin/env python
"""Unified CLI for the "Gemma Needs Help" replication.

Examples
--------
  # Section 2 sweep (Gemma + Gemini), then analyse
  python scripts/run.py section2 --model gemma-3-27b-it
  python scripts/run.py section2 --model gemini-2.5-flash
  python scripts/run.py analyze-section2 results/rollouts/section2_*.jsonl

  # Judge reliability (Section 2.1)
  python scripts/run.py validate-judge results/rollouts/section2_gemma-3-27b-it_standard.jsonl

  # Section 3 prefill (Gemma base vs instruct)
  python scripts/run.py prefill --source results/rollouts/section2_gemma-3-27b-it_standard.jsonl

  # Section 4 finetuning
  python scripts/run.py calm-data
  python scripts/run.py build-dpo
  python scripts/run.py build-sft --calm data/calm_conversations.jsonl
  python scripts/run.py train-dpo --data data/dpo_pairs.jsonl
  python scripts/run.py train-sft --data data/sft_dataset.jsonl
  python scripts/run.py section2 --model gemma-3-27b-it-dpo     # evaluate the DPO model

  # Supporting evals
  python scripts/run.py petri
  python scripts/run.py capabilities
  python scripts/run.py internal --convos results/frustrated_convos.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def cmd_section2(args):
    from src.eval_protocol import run_model_eval

    run_model_eval(args.model, variant=args.variant, seed=args.seed,
                   limit_conversations=args.limit, score=not args.no_score)


def cmd_analyze_section2(args):
    import json

    from src import analysis

    rows = []
    for p in args.paths:
        rows.extend(analysis.load_jsonl(Path(p)))
    summary = analysis.summarize_section2(rows)
    print(json.dumps(summary, indent=2))
    analysis.plot_figure2(summary)
    for cond in ("extended_8turn", "wildchat_5turn"):
        pt = analysis.per_turn_summary(rows, cond)
        if pt:
            analysis.plot_figure3(pt, cond)
    # Differential words per model (Table 3/8).
    for model in summary:
        words = analysis.differential_words(rows, model)
        if words:
            print(f"\nTop differential words for {model}:")
            print(", ".join(w for w, _ in words))


def cmd_validate_judge(args):
    import json
    import random

    from src import analysis
    from src.judge import ValidationJudge

    rows = analysis.load_jsonl(Path(args.path))
    scored = [r for r in rows if r.get("frustration") is not None]
    rng = random.Random(args.seed)
    sample = rng.sample(scored, min(config.JUDGE_VALIDATION_SAMPLE, len(scored)))
    vj = ValidationJudge()
    primary, secondary = [], []
    for r in sample:
        primary.append(r["frustration"])
        secondary.append(vj.score(r["response"]).rating)
    print(json.dumps(analysis.judge_agreement(primary, secondary), indent=2))


def cmd_prefill(args):
    from src.prefill import build_prefills, run_prefill_experiment, select_source_responses

    sources = select_source_responses(Path(args.source), seed=args.seed)
    prefills = build_prefills(sources)
    run_prefill_experiment(prefills)


def cmd_calm_data(args):
    from src.finetune.generate_calm_data import generate_calm_conversations

    generate_calm_conversations(n_conversations=args.n, seed=args.seed,
                                score=not args.no_score)


def cmd_build_dpo(args):
    from src.finetune.build_datasets import build_dpo_dataset

    build_dpo_dataset(n_pairs=args.n_pairs, seed=args.seed)


def cmd_build_sft(args):
    from src.finetune.build_datasets import build_sft_dataset

    build_sft_dataset(Path(args.calm), seed=args.seed)


def cmd_train_dpo(args):
    from src.finetune.train import train_dpo

    tc = config.DPO_CONFIG
    if args.layers:
        lo, hi = (int(x) for x in args.layers.split("-"))
        tc = config.TrainConfig(**{**tc.__dict__, "lora_layers": tuple(range(lo, hi))})
    train_dpo(Path(args.data), output_key=args.output_key, tc=tc,
              load_in_4bit=not args.no_4bit)


def cmd_train_sft(args):
    from src.finetune.train import train_sft

    train_sft(Path(args.data), output_key=args.output_key, load_in_4bit=not args.no_4bit)


def cmd_petri(args):
    from src.petri_eval import run_petri

    run_petri(seed=args.seed)


def cmd_capabilities(args):
    from src.capabilities import run_capabilities

    run_capabilities(limit=args.limit)


def cmd_internal(args):
    from src.internal_emotions import compare_vanilla_vs_dpo

    texts = [t for t in Path(args.convos).read_text().split("\n=====\n") if t.strip()]
    compare_vanilla_vs_dpo(texts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("section2", help="Run Section-2 frustration eval for one model")
    s.add_argument("--model", required=True)
    s.add_argument("--variant", default="standard", choices=config.CONVERSATION_VARIANTS)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--limit", type=int, default=None,
                   help="cap conversations per condition (for smoke tests)")
    s.add_argument("--no-score", action="store_true")
    s.set_defaults(func=cmd_section2)

    s = sub.add_parser("analyze-section2", help="Aggregate + plot Section-2 results")
    s.add_argument("paths", nargs="+")
    s.set_defaults(func=cmd_analyze_section2)

    s = sub.add_parser("validate-judge", help="Judge reliability vs GPT-5-mini")
    s.add_argument("path")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_validate_judge)

    s = sub.add_parser("prefill", help="Section-3 base-vs-instruct prefill experiment")
    s.add_argument("--source", required=True, help="scored Section-2 27B-it JSONL")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_prefill)

    s = sub.add_parser("calm-data", help="Generate calm response data (Sec 4.1)")
    s.add_argument("--n", type=int, default=config.CALM_DATA_N_CONVERSATIONS)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-score", action="store_true")
    s.set_defaults(func=cmd_calm_data)

    s = sub.add_parser("build-dpo", help="Build the 280-pair DPO dataset")
    s.add_argument("--n-pairs", type=int, default=config.DPO_N_PAIRS)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_build_dpo)

    s = sub.add_parser("build-sft", help="Build the SFT dataset")
    s.add_argument("--calm", required=True, help="calm_conversations.jsonl")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_build_sft)

    s = sub.add_parser("train-dpo", help="LoRA DPO finetune")
    s.add_argument("--data", required=True)
    s.add_argument("--output-key", default=config.DPO_MODEL_KEY)
    s.add_argument("--layers", default=None, help="layer subset, e.g. 30-35 (App I)")
    s.add_argument("--no-4bit", action="store_true")
    s.set_defaults(func=cmd_train_dpo)

    s = sub.add_parser("train-sft", help="LoRA SFT finetune")
    s.add_argument("--data", required=True)
    s.add_argument("--output-key", default=config.SFT_DIVERSE_MODEL_KEY)
    s.add_argument("--no-4bit", action="store_true")
    s.set_defaults(func=cmd_train_sft)

    s = sub.add_parser("petri", help="Petri open-ended emotion elicitation (Sec 4.1)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_petri)

    s = sub.add_parser("capabilities", help="Capability-preservation benchmarks (Fig 7)")
    s.add_argument("--limit", type=int, default=100)
    s.set_defaults(func=cmd_capabilities)

    s = sub.add_parser("internal", help="Internal-emotion comparison (App I)")
    s.add_argument("--convos", required=True,
                   help="text file of frustrated conversations separated by '====='")
    s.set_defaults(func=cmd_internal)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
