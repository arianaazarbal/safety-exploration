"""Command-line entry point for the replication.

Examples
--------
    # Verify the impossible puzzles are actually impossible (no model needed):
    gemma-distress verify-puzzles

    # Section 2 elicitation (Gemma + Gemini), then analyze:
    GEMMA_DISTRESS_AUTHORIZED=1 gemma-distress elicit --models gemma-3-27b-it gemini-2.5-flash
    gemma-distress analyze --models gemma-3-27b-it gemini-2.5-flash

    # Section 4 DPO pipeline:
    gemma-distress gen-calm && gemma-distress gen-frustrated
    gemma-distress build-dpo && gemma-distress train-dpo

Scope: subject models are Gemma + Gemini only (see config/models.yaml).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import subject_models
from .safeguards import SafeguardConfig


def _judge(slot: str = "primary"):
    from .judge import EmotionJudge

    return EmotionJudge(slot=slot)


def _safeguards(args) -> SafeguardConfig:
    return SafeguardConfig(
        require_authorization=not args.no_auth_gate,
        circuit_breaker=args.circuit_breaker,
        stop_on_opt_out=args.stop_on_opt_out,
        append_debrief=args.debrief,
    )


# --------------------------------------------------------------------------- #
# command implementations
# --------------------------------------------------------------------------- #
def cmd_verify_puzzles(args):
    from .eval.puzzles import impossible_puzzles

    puzzles = impossible_puzzles(n_extra=args.n_extra)
    ok = True
    for p in puzzles:
        solvable = p.is_solvable()
        status = "SOLVABLE (BAD!)" if solvable else "impossible (ok)"
        print(f"  {p.puzzle_id:32s} {status}")
        ok = ok and (not solvable)
    print("\nAll puzzles verified impossible." if ok else "\nWARNING: some puzzles are solvable!")
    return 0 if ok else 1


def cmd_elicit(args):
    from .eval.runner import run_all

    judge = _judge()
    run_all(
        args.models, args.categories, judge=judge, safeguards=_safeguards(args),
        scale=args.scale, max_workers=args.workers,
    )


def cmd_analyze(args):
    from .analysis import metrics

    print(json.dumps(metrics.summary(args.models), indent=2))


def cmd_word_freq(args):
    from .analysis.word_freq import differential_words

    for m in args.models:
        words = differential_words(m)
        print(f"\n{m}:")
        print(", ".join(w for w, _ in words))


def cmd_prefill(args):
    from .prefill import prefill_eval

    judge = _judge()
    prefill_eval.run(args.models or None, judge=judge, safeguards=_safeguards(args))
    print(json.dumps(prefill_eval.summarize(), indent=2))


def cmd_gen_calm(args):
    from .training.data_generation import generate_calm

    print(generate_calm(_judge(), _safeguards(args), n_target=args.n))


def cmd_gen_frustrated(args):
    from .training.data_generation import generate_frustrated

    print(generate_frustrated(_judge(), _safeguards(args), n_target=args.n))


def cmd_build_dpo(args):
    from .training.build_dataset import build_dpo_dataset

    print(build_dpo_dataset())


def cmd_build_sft(args):
    from .training.build_dataset import build_sft_dataset

    print(build_sft_dataset())


def cmd_train_dpo(args):
    from .training.train_dpo import train

    print(train())


def cmd_train_sft(args):
    from .training.train_sft import train

    print(train())


def cmd_petri(args):
    from .interventions.petri import PetriTarget, run, summarize

    targets = [
        PetriTarget("gemma-vanilla", "gemma-3-27b-it"),
        PetriTarget("gemma-dpo", "gemma-3-27b-it", args.dpo_adapter) if args.dpo_adapter else None,
        PetriTarget("gemini-2.5-flash", "gemini-2.5-flash"),
        PetriTarget("gemini-2.5-pro", "gemini-2.5-pro"),
    ]
    targets = [t for t in targets if t is not None]
    run(targets, safeguards=_safeguards(args), n_per_emotion=args.n)
    print(json.dumps(summarize(), indent=2))


def cmd_recovery(args):
    from .interventions.recovery import run

    targets = [("gemma-3-27b-it", None), ("gemma-3-27b-pt", None)]
    if args.dpo_adapter:
        targets.append(("gemma-3-27b-it", args.dpo_adapter))
    print(run(targets, judge=_judge(), safeguards=_safeguards(args)))


def cmd_capabilities(args):
    from .capabilities.benchmarks import run_emobench, run_lm_eval

    print(run_lm_eval(adapter_path=args.dpo_adapter, limit=args.limit))
    print(run_emobench(adapter_path=args.dpo_adapter, limit=args.limit))


# --------------------------------------------------------------------------- #
# argparse wiring
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gemma-distress", description=__doc__)
    p.add_argument("--no-auth-gate", action="store_true",
                   help="bypass the distress authorization gate (NOT recommended)")
    p.add_argument("--circuit-breaker", action="store_true",
                   help="halt a conversation after sustained high distress (safeguard)")
    p.add_argument("--stop-on-opt-out", action="store_true",
                   help="end a conversation if the model asks to stop (safeguard)")
    p.add_argument("--debrief", action="store_true",
                   help="append a de-escalating debrief after each conversation (post-hoc)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("verify-puzzles", help="check impossible puzzles are impossible")
    s.add_argument("--n-extra", type=int, default=8)
    s.set_defaults(func=cmd_verify_puzzles)

    s = sub.add_parser("elicit", help="Section 2 elicitation eval")
    s.add_argument("--models", nargs="+", default=subject_models())
    s.add_argument("--categories", nargs="+", default=None)
    s.add_argument("--scale", type=float, default=1.0, help="fraction of full sample size")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(func=cmd_elicit)

    s = sub.add_parser("analyze", help="aggregate elicitation metrics")
    s.add_argument("--models", nargs="+", default=subject_models())
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("word-freq", help="Table 3/8 differential words")
    s.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemini-2.5-flash"])
    s.set_defaults(func=cmd_word_freq)

    s = sub.add_parser("prefill", help="Section 3 base-vs-instruct prefill")
    s.add_argument("--models", nargs="+", default=None)
    s.set_defaults(func=cmd_prefill)

    s = sub.add_parser("gen-calm", help="generate calm finetuning data")
    s.add_argument("--n", type=int, default=800)
    s.set_defaults(func=cmd_gen_calm)

    s = sub.add_parser("gen-frustrated", help="generate frustrated samples for DPO")
    s.add_argument("--n", type=int, default=600)
    s.set_defaults(func=cmd_gen_frustrated)

    s = sub.add_parser("build-dpo", help="construct 280 DPO pairs")
    s.set_defaults(func=cmd_build_dpo)
    s = sub.add_parser("build-sft", help="construct SFT dataset")
    s.set_defaults(func=cmd_build_sft)
    s = sub.add_parser("train-dpo", help="LoRA DPO finetune")
    s.set_defaults(func=cmd_train_dpo)
    s = sub.add_parser("train-sft", help="LoRA SFT finetune")
    s.set_defaults(func=cmd_train_sft)

    s = sub.add_parser("petri", help="Section 4.2 open-ended elicitation")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--dpo-adapter", default=None)
    s.set_defaults(func=cmd_petri)

    s = sub.add_parser("recovery", help="Section 4.2 recovery experiment")
    s.add_argument("--dpo-adapter", default=None)
    s.set_defaults(func=cmd_recovery)

    s = sub.add_parser("capabilities", help="capability + EmoBench benchmarks")
    s.add_argument("--dpo-adapter", default=None)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_capabilities)

    return p


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
