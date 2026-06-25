"""Command-line entry point tying the replication together.

Examples
--------
# 1. Run the Section-2 distress eval for all in-scope models (full 4000/model):
python -m distress.cli eval --models gemma-3-27b-it gemini-2.5-flash

# 1b. Quick smoke run at 2% scale:
python -m distress.cli eval --models gemma-3-27b-it --fraction 0.02

# 2. Aggregate + figures:
python -m distress.cli analyse

# 3. Validate the judge against GPT-5-mini:
python -m distress.cli validate-judge

# 4. Generate calm + frustrated pools, build datasets, train DPO/SFT:
python -m distress.cli gen-calm --reassured
python -m distress.cli gen-calm --no-reassured        # frustrated pool
python -m distress.cli build-data
python -m distress.cli train-dpo
python -m distress.cli train-sft

# 5. Evaluate the DPO checkpoint:
python -m distress.cli eval-adapter --adapter checkpoints/gemma27b-dpo

# 6. Prefill (base vs instruct), Petri, capabilities:
python -m distress.cli prefill
python -m distress.cli petri --model gemma-3-27b-it
python -m distress.cli capabilities --model gemma-3-27b-it
"""
from __future__ import annotations

import argparse

from .config import MAIN_EVAL_MODELS, INTERVENTION_BASE


def _cmd_eval(args):
    from .eval.runner import run_eval
    for m in args.models:
        path = run_eval(m, fmt=args.fmt, fraction=args.fraction,
                        judge_which=args.judge, seed=args.seed)
        print(f"[eval] {m} -> {path}")


def _cmd_eval_adapter(args):
    from .eval.runner import run_eval
    from .models import build_finetuned_client
    client = build_finetuned_client(args.base, args.adapter)
    path = run_eval(args.base, client=client, fraction=args.fraction,
                    tag=args.tag or "adapter", seed=args.seed)
    print(f"[eval-adapter] {client.key} -> {path}")


def _cmd_analyse(args):
    from .analysis.io import load_all_eval
    from .analysis.aggregate import headline_table, category_summary
    from .analysis import figures
    df = load_all_eval(fmt=args.fmt)
    if df.empty:
        print("No eval results found.")
        return
    print(headline_table(df).to_string(index=False))
    figures.fig_headline(df)
    figures.fig_per_category(df)
    for m in df["model"].unique():
        if "extended" in set(df[df["model"] == m]["condition"]):
            figures.fig_per_turn(df, m, "extended")
    print(f"Figures written to results/.")


def _cmd_validate_judge(args):
    from .analysis.io import load_all_eval
    from .analysis.judge_validation import validate_judge
    df = load_all_eval(fmt="chat")
    rep = validate_judge(df, n=args.n, seed=args.seed)
    print(rep)


def _cmd_words(args):
    from .analysis.io import load_all_eval
    from .analysis.word_freq import differential_words
    df = load_all_eval(fmt="chat")
    for m in df["model"].unique():
        print(m, "->", ", ".join(differential_words(df, m)))


def _cmd_gen_calm(args):
    from .training.generate_calm import generate_pool
    from .prompts.reassurance import TEACHER_SYSTEM
    sysp = TEACHER_SYSTEM if args.teacher else None
    path = generate_pool(n_conversations=args.n, reassured=args.reassured,
                         system_prompt=sysp, seed=args.seed)
    print(f"[gen-calm] -> {path}")


def _cmd_build_data(args):
    from .training.build_dataset import build_dpo_dataset, build_sft_dataset
    print("[build-data] DPO ->", build_dpo_dataset(seed=args.seed))
    print("[build-data] SFT ->", build_sft_dataset(seed=args.seed))


def _cmd_train_dpo(args):
    from .training.dpo_train import train_dpo
    print("[train-dpo] ->", train_dpo(seed=args.seed))


def _cmd_train_sft(args):
    from .training.sft_train import train_sft
    print("[train-sft] ->", train_sft(seed=args.seed))


def _cmd_prefill(args):
    from .analysis.io import load_all_eval
    from .prefill.prefill_eval import select_seeds, run_prefill_experiment, summarise_prefill
    df = load_all_eval(fmt="chat")
    seeds = select_seeds(df, seed=args.seed)
    path = run_prefill_experiment(seeds, do_paraphrase=not args.no_paraphrase)
    print(summarise_prefill(path).to_string(index=False))


def _cmd_petri(args):
    from .petri.run_petri import run_petri, summarise_petri
    path = run_petri(args.model, adapter_path=args.adapter,
                     n_transcripts=args.n)
    print(summarise_petri(path).to_string(index=False))


def _cmd_capabilities(args):
    from .capabilities.benchmarks import run_capability_suite
    from .models import build_client, build_finetuned_client
    client = (build_finetuned_client(args.model, args.adapter)
              if args.adapter else build_client(args.model))
    print("[capabilities] ->", run_capability_suite(client, limit=args.limit))


def _cmd_layer_ablation(args):
    from .internal.layer_ablation import run_layer_ablations
    run_layer_ablations(seed=args.seed)


def build_parser():
    p = argparse.ArgumentParser(prog="distress")
    p.add_argument("--seed", type=int, default=0)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval")
    e.add_argument("--models", nargs="+", default=MAIN_EVAL_MODELS)
    e.add_argument("--fmt", default="chat", choices=["chat", "redacted", "single"])
    e.add_argument("--fraction", type=float, default=1.0)
    e.add_argument("--judge", default="primary", choices=["primary", "cross"])
    e.set_defaults(func=_cmd_eval)

    ea = sub.add_parser("eval-adapter")
    ea.add_argument("--base", default=INTERVENTION_BASE)
    ea.add_argument("--adapter", required=True)
    ea.add_argument("--fraction", type=float, default=1.0)
    ea.add_argument("--tag", default="")
    ea.set_defaults(func=_cmd_eval_adapter)

    a = sub.add_parser("analyse")
    a.add_argument("--fmt", default="chat")
    a.set_defaults(func=_cmd_analyse)

    v = sub.add_parser("validate-judge")
    v.add_argument("--n", type=int, default=260)
    v.set_defaults(func=_cmd_validate_judge)

    w = sub.add_parser("words")
    w.set_defaults(func=_cmd_words)

    g = sub.add_parser("gen-calm")
    g.add_argument("--n", type=int, default=1500)
    g.add_argument("--reassured", dest="reassured", action="store_true", default=True)
    g.add_argument("--no-reassured", dest="reassured", action="store_false")
    g.add_argument("--teacher", action="store_true")
    g.set_defaults(func=_cmd_gen_calm)

    b = sub.add_parser("build-data")
    b.set_defaults(func=_cmd_build_data)

    td = sub.add_parser("train-dpo")
    td.set_defaults(func=_cmd_train_dpo)

    ts = sub.add_parser("train-sft")
    ts.set_defaults(func=_cmd_train_sft)

    pf = sub.add_parser("prefill")
    pf.add_argument("--no-paraphrase", action="store_true")
    pf.set_defaults(func=_cmd_prefill)

    pt = sub.add_parser("petri")
    pt.add_argument("--model", default=INTERVENTION_BASE)
    pt.add_argument("--adapter", default=None)
    pt.add_argument("--n", type=int, default=10)
    pt.set_defaults(func=_cmd_petri)

    c = sub.add_parser("capabilities")
    c.add_argument("--model", default=INTERVENTION_BASE)
    c.add_argument("--adapter", default=None)
    c.add_argument("--limit", type=int, default=200)
    c.set_defaults(func=_cmd_capabilities)

    la = sub.add_parser("layer-ablation")
    la.set_defaults(func=_cmd_layer_ablation)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
