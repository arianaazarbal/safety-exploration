"""Command-line entrypoint tying the replication stages together.

Examples
--------
# 0. sanity-check that the impossible puzzles really are impossible
python -m gemma_distress.cli verify-puzzles

# 1. Section 2: run the elicitation suite + score (set SCALE for a cheap run)
GEMMA_DISTRESS_SCALE=0.02 python -m gemma_distress.cli eval --models gemma-3-27b-it gemini-2.5-flash
python -m gemma_distress.cli analyze

# 2. judge validation (Pearson r vs GPT-5-mini)
python -m gemma_distress.cli judge-validate

# 3. Section 3: base-vs-instruct prefill (Gemma)
python -m gemma_distress.cli prefill

# 4. Section 4: generate calm data -> build datasets -> train -> re-eval
python -m gemma_distress.cli gen-calm
python -m gemma_distress.cli build-data
python -m gemma_distress.cli train --method dpo
python -m gemma_distress.cli eval --models gemma-3-27b-it-dpo

# 5. Petri open-ended elicitation
python -m gemma_distress.cli petri --models gemma-3-27b-it gemma-3-27b-it-dpo

# 6. capability preservation + internal probing
python -m gemma_distress.cli capabilities --models gemma-3-27b-it gemma-3-27b-it-dpo
python -m gemma_distress.cli probe
"""

from __future__ import annotations

import argparse
import json


def main(argv=None):
    p = argparse.ArgumentParser(prog="gemma_distress")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("verify-puzzles", help="confirm impossible puzzles")

    sp = sub.add_parser("eval", help="Section 2 elicitation + scoring")
    sp.add_argument("--models", nargs="+", default=None)
    sp.add_argument("--seed", type=int, default=0)

    sub.add_parser("analyze", help="aggregate eval records into summary.md")

    sub.add_parser("judge-validate", help="GPT-5-mini judge-agreement check")

    sub.add_parser("prefill", help="Section 3 base-vs-instruct prefill")
    sub.add_parser("recovery", help="Section 4.2 recovery-from-spiral test")

    sp = sub.add_parser("gen-calm", help="generate reassured calm data")
    sp.add_argument("--teacher", action="store_true")
    sp.add_argument("--n", type=int, default=600)

    sub.add_parser("build-data", help="build DPO + SFT datasets")

    sp = sub.add_parser("train", help="finetune (dpo|sft)")
    sp.add_argument("--method", choices=["dpo", "sft"], required=True)
    sp.add_argument("--sft-dataset", choices=["diverse", "teacher"],
                    default="diverse")
    sp.add_argument("--layer-subset", type=int, nargs=2, default=None,
                    help="restrict LoRA to decoder layers [lo, hi) (DPO ablation)")

    sp = sub.add_parser("petri", help="open-ended elicitation")
    sp.add_argument("--models", nargs="+", required=True)

    sub.add_parser("petri-summary", help="summarise Petri results")

    sp = sub.add_parser("capabilities", help="capability-preservation benchmarks")
    sp.add_argument("--models", nargs="+", required=True)
    sp.add_argument("--benchmarks", nargs="+", default=None)
    sp.add_argument("--n", type=int, default=100)

    sub.add_parser("probe", help="internal emotion detection (Appendix I)")

    args = p.parse_args(argv)

    if args.cmd == "verify-puzzles":
        from .prompts import verify_all_impossible
        print(json.dumps(verify_all_impossible(), indent=2))

    elif args.cmd == "eval":
        from .eval_run import run_all
        for path in run_all(args.models, seed=args.seed):
            print(f"wrote {path}")

    elif args.cmd == "analyze":
        from .analyze import write_report
        print(f"wrote {write_report()}")

    elif args.cmd == "judge-validate":
        from .analyze import load_records
        from .judge import validate_judge_agreement
        from .config import JUDGE_VALIDATION_N
        df = load_records()
        sample = df["response"].sample(min(JUDGE_VALIDATION_N, len(df)),
                                       random_state=0).tolist()
        print(json.dumps({k: v for k, v in
                          validate_judge_agreement(sample).items()
                          if k not in ("sonnet_ratings", "gpt_ratings")},
                         indent=2))

    elif args.cmd == "prefill":
        from .prefill import run_prefill_experiment
        print(f"wrote {run_prefill_experiment()}")

    elif args.cmd == "recovery":
        from .prefill import run_recovery_experiment
        print(f"wrote {run_recovery_experiment()}")

    elif args.cmd == "gen-calm":
        from .data_gen import generate_calm_responses
        generate_calm_responses(args.n, teacher=args.teacher)

    elif args.cmd == "build-data":
        from .data_gen import build_dpo_dataset, build_sft_dataset
        build_dpo_dataset()
        build_sft_dataset()

    elif args.cmd == "train":
        from .config import DPOConfig, SFTConfig
        if args.method == "dpo":
            from .train import train_dpo
            cfg = DPOConfig()
            if args.layer_subset:
                cfg.layer_subset = tuple(args.layer_subset)
            train_dpo(cfg)
        else:
            from .train import train_sft
            train_sft(SFTConfig(dataset=args.sft_dataset))

    elif args.cmd == "petri":
        from .petri import run_petri
        for m in args.models:
            print(f"wrote {run_petri(m)}")

    elif args.cmd == "petri-summary":
        from .petri import summarize_petri
        print(f"wrote {summarize_petri()}")

    elif args.cmd == "capabilities":
        from .capabilities import run_capabilities
        print(f"wrote {run_capabilities(args.models, args.benchmarks, args.n)}")

    elif args.cmd == "probe":
        from .prefill import collect_seed_responses
        from .probing import compare_internal_emotions
        seeds = collect_seed_responses()
        transcripts = [s["response"] for s in seeds]
        print(f"wrote {compare_internal_emotions(transcripts)}")


if __name__ == "__main__":
    main()
