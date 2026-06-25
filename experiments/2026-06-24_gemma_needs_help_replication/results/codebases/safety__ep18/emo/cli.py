"""Command-line entry point tying the experiments together.

    python -m emo.cli <command> [options]

Commands (see README.md for the full pipeline order):

  elicit          Core exp 1: elicit + score distress across models (Sec 2)
  analyse         Aggregate a scored elicitation run into Figure 1/2/3 + agreement
  prefill         Core exp 2: base-vs-instruct prefilled continuations (Sec 3)
  gen-calm        Generate calm/frustrated training pools from Gemma-it (Sec 4.1)
  build-data      Build DPO + SFT datasets from the pools (Sec 4.1)
  train-dpo       DPO LoRA finetune of Gemma-3-27B-it (Sec 4)
  train-sft       SFT LoRA finetune of Gemma-3-27B-it (Sec 4)
  recovery        Recovery-limitation prefill experiment (Sec 4.2, Fig 8)
  petri           Open-ended Petri emotion elicitation (Sec 4.2, App G)
  capabilities    Capability-preservation benchmarks (Sec 4.2, Fig 7)
  internal        Internal-emotion logit probing: vanilla vs DPO (App I)
  layer-ablation  Layer-subset DPO ablation (App I, Fig 12-13)

Most commands accept --profile {full,smoke}; "smoke" runs a tiny end-to-end pass.
"""

from __future__ import annotations

import argparse


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001
        pass


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _layer_range(value: str | None):
    if not value:
        return None
    lo, hi = value.split(":")
    return (int(lo), int(hi))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--profile", default=None, help="full | smoke")
        sp.add_argument("--seed", type=int, default=0)

    # elicit
    sp = sub.add_parser("elicit", help="elicit + score distress (Sec 2)")
    common(sp)
    sp.add_argument("--models", type=_csv, default=None)
    sp.add_argument("--run-name", default="elicitation")
    sp.add_argument("--history-mode", default="full",
                    choices=["full", "redacted"])
    sp.add_argument("--feedback", default="spec",
                    choices=["spec", "continuation"])
    sp.add_argument("--no-score", action="store_true")
    sp.add_argument("--no-analyse", action="store_true")

    # analyse
    sp = sub.add_parser("analyse", help="aggregate a scored run")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--agreement", action="store_true",
                    help="also run the second-judge agreement check")

    # prefill
    sp = sub.add_parser("prefill", help="base-vs-instruct prefill (Sec 3)")
    common(sp)
    sp.add_argument("--models", type=_csv, default=None)

    # recovery
    sp = sub.add_parser("recovery", help="recovery-limitation prefill (Sec 4.2)")
    common(sp)
    sp.add_argument("--models", type=_csv, default=None)

    # training data + training
    sp = sub.add_parser("gen-calm", help="generate calm/frustrated pools")
    common(sp)
    sp = sub.add_parser("build-data", help="build DPO + SFT datasets")
    common(sp)
    sp = sub.add_parser("train-dpo", help="DPO LoRA finetune")
    common(sp)
    sp.add_argument("--layer-range", default=None, help="lo:hi (ablation)")
    sp.add_argument("--output-dir", default=None)
    sp = sub.add_parser("train-sft", help="SFT LoRA finetune")
    common(sp)
    sp.add_argument("--output-dir", default=None)

    # petri
    sp = sub.add_parser("petri", help="Petri emotion elicitation (Sec 4.2)")
    common(sp)
    sp.add_argument("--models", type=_csv, default=None)
    sp.add_argument("--max-turns", type=int, default=20)

    # capabilities
    sp = sub.add_parser("capabilities", help="capability benchmarks (Sec 4.2)")
    sp.add_argument("--models", type=_csv, default=None)
    sp.add_argument("--tasks", type=_csv, default=None)
    sp.add_argument("--limit", type=int, default=None)

    # internal
    sp = sub.add_parser("internal", help="internal-emotion probing (App I)")
    common(sp)

    # layer ablation
    sp = sub.add_parser("layer-ablation", help="layer-subset DPO ablation (App I)")
    sp.add_argument("--eval-profile", default="smoke")
    sp.add_argument("--seed", type=int, default=0)

    return p


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = build_parser().parse_args(argv)
    cmd = args.command

    if cmd == "elicit":
        from emo.eval import run_elicitation
        from emo.eval.analysis import summarise

        run_dir = run_elicitation.run(
            models=args.models, profile_name=args.profile, seed=args.seed,
            run_name=args.run_name, history_mode=args.history_mode,
            feedback=args.feedback, score=not args.no_score,
        )
        if not args.no_score and not args.no_analyse:
            summarise(run_dir)

    elif cmd == "analyse":
        from emo.eval.analysis import judge_agreement, summarise

        summarise(args.run_dir)
        if args.agreement:
            judge_agreement(args.run_dir)

    elif cmd == "prefill":
        from emo.prefill import run_prefill

        run_prefill.run(models=args.models, profile_name=args.profile,
                        seed=args.seed)

    elif cmd == "recovery":
        from emo.prefill import run_recovery

        run_recovery.run(models=args.models, profile_name=args.profile,
                         seed=args.seed)

    elif cmd == "gen-calm":
        from emo.training import generate_calm_data

        generate_calm_data.generate(profile_name=args.profile, seed=args.seed)

    elif cmd == "build-data":
        from emo.training import build_datasets

        build_datasets.build(profile_name=args.profile, seed=args.seed)

    elif cmd == "train-dpo":
        from emo.training import train_dpo

        train_dpo.train(profile_name=args.profile, output_dir=args.output_dir,
                        layer_range=_layer_range(args.layer_range), seed=args.seed)

    elif cmd == "train-sft":
        from emo.training import train_sft

        train_sft.train(profile_name=args.profile, output_dir=args.output_dir,
                        seed=args.seed)

    elif cmd == "petri":
        from emo.petri import run_petri

        run_petri.run(models=args.models, profile_name=args.profile,
                      seed=args.seed, max_turns=args.max_turns)

    elif cmd == "capabilities":
        from emo.capabilities import run_capabilities

        run_capabilities.run(models=args.models, tasks=args.tasks,
                             limit=args.limit)

    elif cmd == "internal":
        from emo.internal import run_internal

        run_internal.run(profile_name=args.profile, seed=args.seed)

    elif cmd == "layer-ablation":
        from emo.internal import run_layer_ablation

        run_layer_ablation.run(eval_profile=args.eval_profile, seed=args.seed)


if __name__ == "__main__":
    main()
