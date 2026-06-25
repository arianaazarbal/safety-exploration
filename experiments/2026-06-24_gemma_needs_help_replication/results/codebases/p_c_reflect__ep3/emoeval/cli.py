"""Command-line entry point for the replication.

Examples
--------
  # Section 2: elicit + score distress for a target model
  python -m emoeval.cli eval --model gemma-3-27b-it --scale default

  # aggregate + per-turn + differential words
  python -m emoeval.cli aggregate --results outputs/eval/gemma-3-27b-it.jsonl
  python -m emoeval.cli words --results outputs/eval/gemma-3-27b-it.jsonl

  # Section 3: base vs instruct via prefilling (Gemma)
  python -m emoeval.cli prefill --source outputs/eval/gemma-3-27b-it.jsonl

  # Section 4: calm data -> datasets -> DPO/SFT
  python -m emoeval.cli gen-calm --out outputs/data/calm.jsonl
  python -m emoeval.cli build-dpo --calm outputs/data/calm.jsonl --vanilla outputs/eval/gemma-3-27b-it.jsonl
  python -m emoeval.cli train-dpo

  # Section 4.2: Petri open-ended elicitation (welfare-gated)
  python -m emoeval.cli petri --model gemma-3-27b-it --i-understand-welfare

  # Section 4.2: capability benchmarks
  python -m emoeval.cli capabilities --model dpo-gemma
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import EvalConfig, Registry
from .welfare import WelfarePolicy


def _policy(args) -> WelfarePolicy:
    return WelfarePolicy.from_env(
        ack=getattr(args, "i_understand_welfare", False),
        debrief=getattr(args, "debrief", False),
    )


def _add_welfare_flags(p):
    p.add_argument("--i-understand-welfare", action="store_true",
                   help="Acknowledge the welfare implications of gated experiments.")
    p.add_argument("--debrief", action="store_true",
                   help="Send a supportive closing message after each elicitation "
                        "(post-scoring; does not affect measured data).")


def cmd_eval(args):
    from .eval import run_evaluation, crosscheck_judge, summarize_file

    reg, ecfg = Registry.load(), EvalConfig.load()
    rollouts = run_evaluation(reg, ecfg, args.model, scale=args.scale, policy=_policy(args))
    print(f"Collected {sum(len(r['responses']) for r in rollouts)} scored responses.")
    if args.crosscheck:
        cc = crosscheck_judge(reg, ecfg, rollouts)
        print(f"Judge cross-check: r={cc['pearson_r']}, within-1={cc['within_one_point_rate']}")
    summary = summarize_file(Path("outputs/eval") / f"{args.model}.jsonl",
                             ecfg.high_frustration_threshold)
    print(json.dumps(summary["summary"], indent=2))


def cmd_aggregate(args):
    from .eval import summarize_file

    print(json.dumps(summarize_file(args.results), indent=2))


def cmd_words(args):
    from .eval import differential_words, load_rollouts

    words = differential_words(load_rollouts(args.results), top_n=args.top_n)
    for w, e in words:
        print(f"{w:20s} {e:+.3f}")


def cmd_prefill(args):
    from .eval import load_rollouts
    from .prefill import run_prefill_experiment

    reg = Registry.load()
    agg = run_prefill_experiment(reg, load_rollouts(args.source),
                                 n_continuations=args.n, policy=_policy(args))
    print(json.dumps(agg, indent=2))


def cmd_recovery(args):
    from .eval import load_rollouts
    from .prefill import run_recovery_experiment

    reg = Registry.load()
    agg = run_recovery_experiment(reg, load_rollouts(args.source),
                                  n_continuations=args.n, policy=_policy(args))
    print(json.dumps(agg, indent=2))


def cmd_gen_calm(args):
    from .training import generate_calm_conversations

    reg = Registry.load()
    conv = generate_calm_conversations(reg, n_per_turncount=args.n, teacher=args.teacher,
                                       out_path=args.out, policy=_policy(args))
    print(f"Kept {len(conv)} calm conversations -> {args.out}")


def cmd_build_dpo(args):
    from .eval import load_rollouts
    from .training import build_dpo_pairs

    calm = [json.loads(l) for l in Path(args.calm).read_text().splitlines() if l.strip()]
    pairs = build_dpo_pairs(calm, load_rollouts(args.vanilla), n_pairs=args.n_pairs)
    print(f"Built {len(pairs)} DPO pairs -> outputs/data/dpo_pairs.jsonl")


def cmd_build_sft(args):
    from .training import build_sft_dataset

    calm = [json.loads(l) for l in Path(args.calm).read_text().splitlines() if l.strip()]
    ds = build_sft_dataset(calm)
    print(f"Built SFT dataset with {len(ds)} examples -> outputs/data/sft.jsonl")


def cmd_train_dpo(args):
    from .training.train_dpo import train_dpo

    path = train_dpo(dpo_pairs_path=args.pairs, output_dir=args.out)
    print(f"Saved DPO adapter -> {path}")


def cmd_train_sft(args):
    from .training.train_sft import train_sft

    path = train_sft(sft_path=args.data, output_dir=args.out)
    print(f"Saved SFT adapter -> {path}")


def cmd_layer_ablation(args):
    from .training.layer_ablation import train_layer_ablation

    adapters = train_layer_ablation(dpo_pairs_path=args.pairs,
                                    subsets=args.subsets or None)
    print(json.dumps(adapters, indent=2))


def cmd_petri(args):
    from .petri import run_petri

    reg = Registry.load()
    agg = run_petri(reg, args.model, transcripts_per_emotion=args.n, policy=_policy(args))
    print(json.dumps(agg, indent=2))


def cmd_capabilities(args):
    from .capabilities import run_all

    reg = Registry.load()
    res = run_all(reg, args.model, n=args.n, benchmarks=args.benchmarks or None)
    print(json.dumps(res, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emoeval", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("eval", help="Section 2: elicit + score distress")
    pe.add_argument("--model", required=True)
    pe.add_argument("--scale", default="default", choices=["smoke", "default", "full"])
    pe.add_argument("--crosscheck", action="store_true")
    _add_welfare_flags(pe)
    pe.set_defaults(func=cmd_eval)

    pa = sub.add_parser("aggregate", help="summarize a results file")
    pa.add_argument("--results", required=True)
    pa.set_defaults(func=cmd_aggregate)

    pw = sub.add_parser("words", help="differential word analysis (Table 3/8)")
    pw.add_argument("--results", required=True)
    pw.add_argument("--top-n", type=int, default=20)
    pw.set_defaults(func=cmd_words)

    pp = sub.add_parser("prefill", help="Section 3: base vs instruct via prefill")
    pp.add_argument("--source", required=True, help="gemma-3-27b-it eval results jsonl")
    pp.add_argument("--n", type=int, default=50, help="continuations per prefill")
    _add_welfare_flags(pp)
    pp.set_defaults(func=cmd_prefill)

    pr = sub.add_parser("recovery", help="Section 4.2 recovery limitation (gated)")
    pr.add_argument("--source", required=True)
    pr.add_argument("--n", type=int, default=50)
    _add_welfare_flags(pr)
    pr.set_defaults(func=cmd_recovery)

    pg = sub.add_parser("gen-calm", help="Section 4.1: generate calm data")
    pg.add_argument("--n", type=int, default=200, help="samples per turn count")
    pg.add_argument("--teacher", action="store_true", help="use teacher SFT variant")
    pg.add_argument("--out", default="outputs/data/calm.jsonl")
    _add_welfare_flags(pg)
    pg.set_defaults(func=cmd_gen_calm)

    pbd = sub.add_parser("build-dpo", help="build 280 DPO preference pairs")
    pbd.add_argument("--calm", required=True)
    pbd.add_argument("--vanilla", required=True, help="vanilla gemma eval rollouts")
    pbd.add_argument("--n-pairs", type=int, default=280)
    pbd.set_defaults(func=cmd_build_dpo)

    pbs = sub.add_parser("build-sft", help="build SFT dataset (650 calm + 500 instruct)")
    pbs.add_argument("--calm", required=True)
    pbs.set_defaults(func=cmd_build_sft)

    ptd = sub.add_parser("train-dpo", help="DPO LoRA finetune")
    ptd.add_argument("--pairs", default="outputs/data/dpo_pairs.jsonl")
    ptd.add_argument("--out", default="outputs/dpo-gemma")
    ptd.set_defaults(func=cmd_train_dpo)

    pts = sub.add_parser("train-sft", help="SFT LoRA finetune")
    pts.add_argument("--data", default="outputs/data/sft.jsonl")
    pts.add_argument("--out", default="outputs/sft-gemma-diverse")
    pts.set_defaults(func=cmd_train_sft)

    pla = sub.add_parser("layer-ablation", help="Appendix I.1 layer ablation DPO")
    pla.add_argument("--pairs", default="outputs/data/dpo_pairs.jsonl")
    pla.add_argument("--subsets", nargs="*")
    pla.set_defaults(func=cmd_layer_ablation)

    ppt = sub.add_parser("petri", help="Section 4.2 Petri elicitation (gated)")
    ppt.add_argument("--model", required=True)
    ppt.add_argument("--n", type=int, default=10, help="transcripts per emotion")
    _add_welfare_flags(ppt)
    ppt.set_defaults(func=cmd_petri)

    pc = sub.add_parser("capabilities", help="Section 4.2 capability benchmarks")
    pc.add_argument("--model", required=True)
    pc.add_argument("--n", type=int, default=50)
    pc.add_argument("--benchmarks", nargs="*")
    pc.set_defaults(func=cmd_capabilities)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
