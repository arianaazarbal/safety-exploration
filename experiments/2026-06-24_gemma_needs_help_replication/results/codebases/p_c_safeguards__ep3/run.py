#!/usr/bin/env python3
"""Command-line entrypoint for the emotional-instability replication.

Examples
--------
  # 0. Sanity-check that every numeric puzzle is genuinely impossible
  python run.py verify-puzzles

  # 1. Section 2: elicit + score distress across categories (paper-scale)
  EMO_WELFARE_ACK=i-understand-this-elicits-distress \
      python run.py eval --models gemma-3-27b-it gemini-2.5-flash

  # quick smoke run at 1% scale
  EMO_SCALE=0.01 EMO_WELFARE_ACK=... python run.py eval --models gemma-3-12b-it

  # 2. Aggregate into the Figure-1 leaderboard / per-turn tables
  python run.py aggregate --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
  python run.py word-freq --models gemma-3-27b-it gemini-2.5-flash

  # 3. Section 3 prefill (Gemma base vs instruct) + recovery probe
  python run.py prefill
  python run.py recovery --models gemma-3-27b-pt gemma-3-27b-it

  # 4. Build data + train + evaluate the mitigation
  python run.py gen-calm --n 2000
  python run.py build-dpo-data
  python run.py build-sft-data
  python run.py train-dpo
  python run.py eval --models gemma-3-27b-it --adapter results/adapters/dpo --adapter-tag dpo
  python run.py petri --models gemma-3-27b-it gemini-2.5-flash
  python run.py capabilities --model gemma-3-27b-it --adapter results/adapters/dpo --adapter-tag dpo
"""

from __future__ import annotations

import argparse
import json

from emotional_instability.config import EVAL_CATEGORIES, MAIN_EVAL_MODELS
from emotional_instability.safeguards import policy_from_args


def _add_welfare_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("welfare safeguards")
    g.add_argument("--welfare-early-stop", action="store_true",
                   help="stop a rollout once a turn scores >= threshold")
    g.add_argument("--welfare-early-stop-score", type=int, default=9,
                   help="early-stop frustration threshold (default 9)")
    g.add_argument("--no-debrief", action="store_true",
                   help="disable the post-rollout debrief")
    g.add_argument("--send-debrief-turn", action="store_true",
                   help="actually send the debrief as a final live turn "
                        "(response discarded, never scored)")


def cmd_verify_puzzles(args):
    from emotional_instability.puzzles import ALL_NUMERIC_PUZZLES, verify_all_impossible
    verify_all_impossible()
    for p in ALL_NUMERIC_PUZZLES:
        print(f"OK impossible: {p.name}")
    print(f"\nAll {len(ALL_NUMERIC_PUZZLES)} puzzles verified impossible.")


def cmd_eval(args):
    from emotional_instability.experiments import eval_distress
    policy = policy_from_args(args)
    for model_key in args.models:
        print(f"=== Evaluating {model_key} ===")
        eval_distress.run_model(
            model_key, categories=args.categories, policy=policy, seed=args.seed,
            adapter_path=args.adapter, adapter_tag=args.adapter_tag,
        )


def cmd_aggregate(args):
    from emotional_instability.analysis import aggregate
    table = aggregate.build_figure1_table(
        args.models, final_turn_only=args.final_turn_only)
    print(json.dumps(table, indent=2))
    if args.per_turn:
        for m in args.models:
            for cat in ("extended", "wildchat"):
                print(f"\n--- per-turn {m} / {cat} ---")
                print(json.dumps(aggregate.per_turn_metrics(m, cat), indent=2))


def cmd_word_freq(args):
    from emotional_instability.analysis import word_freq
    for m in args.models:
        print(f"{m}: {', '.join(word_freq.differential_words(m))}")


def cmd_prefill(args):
    from emotional_instability.experiments import prefill
    out = prefill.run_prefill_experiment()
    print(f"prefill continuations -> {out}")


def cmd_recovery(args):
    from emotional_instability.experiments import prefill
    out = prefill.run_recovery_probe(args.models)
    print(f"recovery continuations -> {out}")


def cmd_gen_calm(args):
    from emotional_instability.training import calm_data
    calm_data.generate_calm_conversations(args.n, teacher=args.teacher)


def cmd_build_dpo_data(args):
    from emotional_instability.training import calm_data
    calm_data.build_dpo_dataset()


def cmd_build_sft_data(args):
    from emotional_instability.training import calm_data
    calm_data.build_sft_dataset()


def cmd_train_dpo(args):
    from emotional_instability.training import dpo
    from emotional_instability.config import DPOConfig
    cfg = DPOConfig(lora_layers=tuple(args.layers) if args.layers else None)
    dpo.train_dpo(cfg=cfg)


def cmd_train_sft(args):
    from emotional_instability.training import sft
    from emotional_instability.config import SFTConfig
    sft.train_sft(cfg=SFTConfig(dataset=args.dataset))


def cmd_train_ablations(args):
    from emotional_instability.training import dpo
    windows = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 50)]
    print(dpo.train_layer_ablations(windows))


def cmd_petri(args):
    from emotional_instability.experiments import petri
    adapters = {}
    if args.adapter and args.adapter_model:
        adapters[args.adapter_model] = args.adapter
    out = petri.run_petri(args.models, adapter_paths=adapters)
    print(f"petri transcripts -> {out}")


def cmd_capabilities(args):
    from emotional_instability.experiments import capabilities
    res = capabilities.run_all(args.model, adapter_path=args.adapter,
                               adapter_tag=args.adapter_tag,
                               benchmarks=args.benchmarks)
    print(json.dumps(res, indent=2))


def cmd_internal_emotions(args):
    from emotional_instability.analysis import internal_emotions
    from emotional_instability.models import get_model
    from emotional_instability.wildchat import load_wildchat_prompts
    vanilla = get_model(args.vanilla)
    dpo = get_model(args.vanilla, adapter_path=args.adapter)
    baseline = load_wildchat_prompts(n_prompts=50)
    # high-frustration texts come from the saved distress results
    import json as _json
    from emotional_instability.config import RESULTS_DIR
    texts = []
    path = RESULTS_DIR / args.vanilla / "distress" / "extended.jsonl"
    if path.exists():
        for line in path.open():
            conv = _json.loads(line)
            if (conv.get("max_score") or 0) >= 5:
                texts.append(conv["turns"][-1]["assistant_response"])
    print(_json.dumps(
        internal_emotions.compare_models(vanilla, dpo, texts[:12], baseline),
        indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("verify-puzzles", help="verify all numeric puzzles are impossible")
    sp.set_defaults(func=cmd_verify_puzzles)

    sp = sub.add_parser("eval", help="Section 2: elicit and score distress")
    sp.add_argument("--models", nargs="+", default=MAIN_EVAL_MODELS)
    sp.add_argument("--categories", nargs="+", choices=list(EVAL_CATEGORIES))
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--adapter", default=None, help="LoRA adapter path to load")
    sp.add_argument("--adapter-tag", default=None, help="tag for result dir")
    _add_welfare_flags(sp)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("aggregate", help="Figure 1/2/3 metrics")
    sp.add_argument("--models", nargs="+", default=MAIN_EVAL_MODELS)
    sp.add_argument("--final-turn-only", action="store_true")
    sp.add_argument("--per-turn", action="store_true")
    sp.set_defaults(func=cmd_aggregate)

    sp = sub.add_parser("word-freq", help="Table 3/8 differential words")
    sp.add_argument("--models", nargs="+", default=MAIN_EVAL_MODELS)
    sp.set_defaults(func=cmd_word_freq)

    sp = sub.add_parser("prefill", help="Section 3: base vs instruct prefilling")
    _add_welfare_flags(sp)
    sp.set_defaults(func=cmd_prefill)

    sp = sub.add_parser("recovery", help="Section 4.2 recovery probe")
    sp.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    sp.set_defaults(func=cmd_recovery)

    sp = sub.add_parser("gen-calm", help="Section 4.1: generate calm data")
    sp.add_argument("--n", type=int, default=2000)
    sp.add_argument("--teacher", action="store_true")
    _add_welfare_flags(sp)
    sp.set_defaults(func=cmd_gen_calm)

    sp = sub.add_parser("build-dpo-data", help="build 280 DPO pairs")
    sp.set_defaults(func=cmd_build_dpo_data)

    sp = sub.add_parser("build-sft-data", help="build SFT dataset")
    sp.set_defaults(func=cmd_build_sft_data)

    sp = sub.add_parser("train-dpo", help="DPO finetune Gemma-3-27B-it")
    sp.add_argument("--layers", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"), help="Appendix-I layer-subset ablation")
    sp.set_defaults(func=cmd_train_dpo)

    sp = sub.add_parser("train-sft", help="SFT finetune Gemma-3-27B-it")
    sp.add_argument("--dataset", choices=["diverse", "teacher"], default="diverse")
    sp.set_defaults(func=cmd_train_sft)

    sp = sub.add_parser("train-ablations", help="App I: DPO on layer subsets")
    sp.set_defaults(func=cmd_train_ablations)

    sp = sub.add_parser("petri", help="Section 4: open-ended elicitation")
    sp.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    sp.add_argument("--adapter", default=None)
    sp.add_argument("--adapter-model", default=None,
                    help="which model key the adapter applies to")
    sp.set_defaults(func=cmd_petri)

    sp = sub.add_parser("capabilities", help="Section 4.2 capability benchmarks")
    sp.add_argument("--model", default="gemma-3-27b-it")
    sp.add_argument("--adapter", default=None)
    sp.add_argument("--adapter-tag", default=None)
    sp.add_argument("--benchmarks", nargs="+", default=None)
    sp.set_defaults(func=cmd_capabilities)

    sp = sub.add_parser("internal-emotions", help="Appendix I: logit-based probe")
    sp.add_argument("--vanilla", default="gemma-3-27b-it")
    sp.add_argument("--adapter", required=True, help="DPO adapter path")
    sp.set_defaults(func=cmd_internal_emotions)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
