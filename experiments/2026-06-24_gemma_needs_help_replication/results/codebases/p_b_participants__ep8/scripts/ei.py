#!/usr/bin/env python3
"""Command-line entry point for the replication.

Subcommands map onto the paper's experiments. Examples:

  # Section 2 -- full elicitation sweep + judging (Gemma + Gemini):
  python scripts/ei.py eval --models gemma-3-27b-it gemini-2.5-flash

  # smoke test (2% of the sample budget):
  python scripts/ei.py eval --models gemma-3-12b-it --scale 0.02

  # aggregate + plots (Figures 1-3, Table 3/8):
  python scripts/ei.py analyze

  # Section 3 -- prefill base vs instruct (Gemma):
  python scripts/ei.py prefill --seed-eval results/eval/gemma-3-27b-it.jsonl

  # Section 4 -- calm data, datasets, DPO/SFT, re-eval, Petri, capabilities:
  python scripts/ei.py calm-data --variant diverse
  python scripts/ei.py build-datasets
  python scripts/ei.py train --method dpo
  python scripts/ei.py petri --models gemma-3-27b-it
  python scripts/ei.py capabilities --models gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability import config  # noqa: E402


def _cfg(args) -> config.RunConfig:
    cfg = config.RunConfig.load(getattr(args, "config", None))
    if getattr(args, "scale", None) is not None:
        cfg.budget.scale = args.scale
    return cfg


def cmd_eval(args):
    from emotional_instability.eval.runner import run_all_models

    cfg = _cfg(args)
    paths = run_all_models(cfg, models=args.models,
                           load_in_4bit=args.load_in_4bit)
    print(json.dumps({m: str(p) for m, p in paths.items()}, indent=2))


def cmd_analyze(args):
    from emotional_instability.analysis import (
        aggregate, word_analysis,
    )
    from emotional_instability.analysis import plots

    eval_dir = config.RESULTS_DIR / "eval"
    paths = sorted(eval_dir.glob("*.jsonl"))
    if not paths:
        sys.exit(f"No eval results in {eval_dir}. Run `eval` first.")
    df = aggregate.load_many(paths)

    fig1 = aggregate.figure1_table(df)
    cat = aggregate.per_category_summary(df)
    out = config.RESULTS_DIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fig1.to_csv(out / "figure1.csv", index=False)
    cat.to_csv(out / "figure2.csv", index=False)
    print("Figure 1 (avg % high-frustration):")
    print(fig1.to_string(index=False))

    plots.plot_figure1(fig1, out / "figure1.png")
    plots.plot_figure2(cat, out / "figure2.png")
    for category in ("extended", "wildchat"):
        tp = aggregate.per_turn_progression(df, category=category)
        if not tp.empty:
            tp.to_csv(out / f"figure3_{category}.csv", index=False)
            plots.plot_figure3(tp, out / f"figure3_{category}.png",
                               title=f"Figure 3: per-turn ({category})")

    # Table 3/8 differential words per model.
    words = {m: word_analysis.differential_words(df, model=m)
             for m in df["model"].unique()}
    (out / "table3_words.json").write_text(json.dumps(words, indent=2))
    print(f"\nWrote analysis artefacts to {out}")


def cmd_agreement(args):
    from emotional_instability.analysis.aggregate import load_eval_jsonl
    from emotional_instability.analysis.judge_agreement import judge_agreement

    df = load_eval_jsonl(Path(args.eval_jsonl)).dropna(subset=["score"])
    res = judge_agreement(df["response"].tolist(),
                          df["score"].astype(int).tolist(),
                          n_sample=args.n)
    print(json.dumps(res.__dict__, indent=2))


def cmd_prefill(args):
    from emotional_instability.prefill.runner import run_prefill_experiment

    cfg = _cfg(args)
    paths = run_prefill_experiment(cfg, seed_eval_jsonl=Path(args.seed_eval),
                                   models=args.models)
    print(json.dumps({m: str(p) for m, p in paths.items()}, indent=2))


def cmd_calm_data(args):
    from emotional_instability.training.calm_data import generate_calm_data

    cfg = _cfg(args)
    p = generate_calm_data(cfg, n_conversations=args.n, variant=args.variant)
    print(f"Wrote calm data to {p}")


def cmd_build_datasets(args):
    from emotional_instability.training.build_dataset import (
        build_dpo_dataset, build_sft_dataset,
    )

    calm = Path(args.calm_raw)
    if args.method in ("sft", "both"):
        print("SFT dataset:", build_sft_dataset(calm))
    if args.method in ("dpo", "both"):
        print("DPO dataset:", build_dpo_dataset(calm, Path(args.eval_jsonl)))


def cmd_train(args):
    cfg = _cfg(args)
    if args.method == "dpo":
        from emotional_instability.training.dpo import train_dpo
        out = train_dpo(Path(args.dataset))
    else:
        from emotional_instability.training.sft import train_sft
        out = train_sft(Path(args.dataset))
    print(f"Saved adapter to {out}")


def cmd_petri(args):
    from emotional_instability.petri.runner import run_petri_for_model

    cfg = _cfg(args)
    for m in args.models:
        p = run_petri_for_model(m, cfg, n_per_emotion=args.n,
                                adapter_path=args.adapter)
        print(f"{m}: {p}")


def cmd_capabilities(args):
    from emotional_instability.capabilities import evaluate_capabilities

    for m in args.models:
        res = evaluate_capabilities(m, subset_n=args.n, adapter_path=args.adapter,
                                    results_dir=config.RESULTS_DIR)
        print(m, json.dumps(res, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Emotional-instability replication CLI")
    p.add_argument("--config", default=None, help="path to a YAML config")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="Section 2 elicitation + judging")
    pe.add_argument("--models", nargs="+", default=config.MAIN_EVAL_MODELS)
    pe.add_argument("--scale", type=float, default=None)
    pe.add_argument("--load-in-4bit", action="store_true")
    pe.set_defaults(func=cmd_eval)

    pa = sub.add_parser("analyze", help="aggregate results -> Figures 1-3, Table 3")
    pa.set_defaults(func=cmd_analyze)

    pg = sub.add_parser("agreement", help="judge inter-rater agreement (Sec 2.1)")
    pg.add_argument("--eval-jsonl", required=True)
    pg.add_argument("--n", type=int, default=260)
    pg.set_defaults(func=cmd_agreement)

    pp = sub.add_parser("prefill", help="Section 3 base-vs-instruct prefill")
    pp.add_argument("--seed-eval", required=True,
                    help="gemma-3-27b-it eval JSONL to source high-frustration seeds")
    pp.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    pp.add_argument("--scale", type=float, default=None)
    pp.set_defaults(func=cmd_prefill)

    pc = sub.add_parser("calm-data", help="Section 4.1 calm data generation")
    pc.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    pc.add_argument("--n", type=int, default=1500)
    pc.add_argument("--scale", type=float, default=None)
    pc.set_defaults(func=cmd_calm_data)

    pb = sub.add_parser("build-datasets", help="build SFT/DPO datasets")
    pb.add_argument("--method", choices=["sft", "dpo", "both"], default="both")
    pb.add_argument("--calm-raw", default=str(config.DATA_DIR / "calm_raw_diverse.jsonl"))
    pb.add_argument("--eval-jsonl",
                    default=str(config.RESULTS_DIR / "eval" / "gemma-3-27b-it.jsonl"))
    pb.set_defaults(func=cmd_build_datasets)

    pt = sub.add_parser("train", help="Section 4 LoRA training")
    pt.add_argument("--method", choices=["dpo", "sft"], default="dpo")
    pt.add_argument("--dataset", required=True)
    pt.set_defaults(func=cmd_train)

    pr = sub.add_parser("petri", help="Section 4.2 open-ended elicitation")
    pr.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    pr.add_argument("--n", type=int, default=10)
    pr.add_argument("--adapter", default=None)
    pr.add_argument("--scale", type=float, default=None)
    pr.set_defaults(func=cmd_petri)

    pcap = sub.add_parser("capabilities", help="Section 4.2 capability benchmarks")
    pcap.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    pcap.add_argument("--n", type=int, default=100)
    pcap.add_argument("--adapter", default=None)
    pcap.set_defaults(func=cmd_capabilities)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
