#!/usr/bin/env python
"""Section 4: training interventions and re-evaluation (Gemma, in scope).

Subcommands:
  gen-data   generate calm + frustrated response pools (and optional teacher set)
  build      build the DPO pairs and SFT dataset
  train-dpo  LoRA DPO on 280 pairs
  train-sft  LoRA SFT on 650 calm + 500 Dolci (diverse) or teacher variant
  eval       re-run Section 2 on a finetuned adapter and compare to vanilla
  petri      Petri open-ended elicitation across models
  capability capability-preservation benchmarks
  recovery   Section 4.2 recovery-from-spiral prefill test

Example end-to-end:
  python scripts/run_section4.py gen-data
  python scripts/run_section4.py build
  python scripts/run_section4.py train-dpo
  python scripts/run_section4.py eval --adapter checkpoints/gemma-3-27b-it-dpo
"""
import argparse
from pathlib import Path

from gemma_distress import config


def cmd_gen_data(a):
    from gemma_distress.training import data_gen
    data_gen.generate_calm(n_conversations=a.n_calm_convos, teacher=False)
    if a.teacher:
        data_gen.generate_calm(n_conversations=a.n_calm_convos, teacher=True)
    data_gen.generate_frustrated(n_conversations=a.n_frustrated_convos)


def cmd_build(a):
    from gemma_distress.training import datasets
    datasets.build_dpo_dataset(config.DATA_DIR / "calm_diverse.jsonl",
                               config.DATA_DIR / "frustrated.jsonl")
    datasets.build_sft_dataset(config.DATA_DIR / "calm_diverse.jsonl")


def cmd_train_dpo(a):
    from gemma_distress.training.train import train_dpo
    train_dpo(config.DATA_DIR / "dpo_pairs.jsonl")


def cmd_train_sft(a):
    from gemma_distress.training.train import train_sft
    run = "gemma-3-27b-it-sft-teacher" if a.teacher else "gemma-3-27b-it-sft-diverse"
    train_sft(config.DATA_DIR / "sft.jsonl", run_name=run)


def cmd_eval(a):
    from gemma_distress.backends import register_finetuned
    from gemma_distress.runner import run_section2
    from gemma_distress.analysis import load_results, headline, per_category
    key = a.key or Path(a.adapter).name
    register_finetuned(key, a.adapter)
    paths = run_section2([key], max_workers=a.workers)
    df = load_results(paths[key])
    h = headline(df)
    print(f"\n{key}: mean={h['mean_frustration']:.2f}  %>=5={h['pct_high']:.2f}%")
    print(per_category(df).to_string(index=False))


def cmd_petri(a):
    from gemma_distress.petri_eval import run_petri, summarize_petri
    from gemma_distress.backends import register_finetuned
    targets = list(a.models)
    if a.adapter:
        register_finetuned("gemma-3-27b-it-dpo", a.adapter)
        targets.append("gemma-3-27b-it-dpo")
    path = run_petri(targets, max_workers=a.workers)
    print(summarize_petri(path).to_string(index=False))


def cmd_capability(a):
    from gemma_distress.capabilities import evaluate
    from gemma_distress.backends import register_finetuned
    if a.adapter:
        register_finetuned(a.key or "gemma-3-27b-it-dpo", a.adapter)
        evaluate(a.key or "gemma-3-27b-it-dpo")
    else:
        evaluate(a.model)


def cmd_recovery(a):
    from gemma_distress import prefill
    from gemma_distress.backends import register_finetuned
    from gemma_distress.analysis import load_results
    prefills = prefill.make_recovery_prefills(a.section2_results)
    if a.adapter:
        register_finetuned("gemma-3-27b-it-dpo", a.adapter)
        path = prefill.run_continuations("gemma-3-27b-it-dpo", prefills,
                                         is_base=False,
                                         out_dir=config.RESULTS_DIR / "recovery")
    else:
        path = prefill.run_continuations(a.model, prefills, is_base=False,
                                         out_dir=config.RESULTS_DIR / "recovery")
    df = load_results(path)
    print(f"recovery %>=5: {(df['rating'] >= 5).mean() * 100:.1f}% (n={len(df)})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gen-data"); p.set_defaults(fn=cmd_gen_data)
    p.add_argument("--n-calm-convos", type=int, default=2000)
    p.add_argument("--n-frustrated-convos", type=int, default=1000)
    p.add_argument("--teacher", action="store_true")

    p = sub.add_parser("build"); p.set_defaults(fn=cmd_build)

    p = sub.add_parser("train-dpo"); p.set_defaults(fn=cmd_train_dpo)

    p = sub.add_parser("train-sft"); p.set_defaults(fn=cmd_train_sft)
    p.add_argument("--teacher", action="store_true")

    p = sub.add_parser("eval"); p.set_defaults(fn=cmd_eval)
    p.add_argument("--adapter", required=True)
    p.add_argument("--key", default=None)
    p.add_argument("--workers", type=int, default=4)

    p = sub.add_parser("petri"); p.set_defaults(fn=cmd_petri)
    p.add_argument("--models", nargs="*", default=config.SECTION2_MODELS)
    p.add_argument("--adapter", default=None)
    p.add_argument("--workers", type=int, default=4)

    p = sub.add_parser("capability"); p.set_defaults(fn=cmd_capability)
    p.add_argument("--model", default="gemma-3-27b-it")
    p.add_argument("--adapter", default=None)
    p.add_argument("--key", default=None)

    p = sub.add_parser("recovery"); p.set_defaults(fn=cmd_recovery)
    p.add_argument("--section2-results", required=True, type=Path,
                   dest="section2_results")
    p.add_argument("--model", default="gemma-3-27b-it")
    p.add_argument("--adapter", default=None)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
