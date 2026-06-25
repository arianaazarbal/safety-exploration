"""Command-line entrypoint for the replication.

Run stages individually, e.g.:

  python -m gemma_distress.cli section2 --count-scale 0.01
  python -m gemma_distress.cli judge-agreement
  python -m gemma_distress.cli section3
  python -m gemma_distress.cli calm-data --n 2000
  python -m gemma_distress.cli build-data
  python -m gemma_distress.cli train --method dpo
  python -m gemma_distress.cli train --method sft --teacher
  python -m gemma_distress.cli petri --model gemma-3-27b-it --adapter runs/.../dpo_all
  python -m gemma_distress.cli capabilities --model gemma-3-27b-it --adapter runs/.../dpo_all
  python -m gemma_distress.cli recovery --adapter runs/.../dpo_all
  python -m gemma_distress.cli figures

Nothing here runs at import time; every stage is explicit. See DESIGN.md for scope.
"""

from __future__ import annotations

import argparse

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gemma_distress")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("section2", help="Run distress elicitation sweep")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--count-scale", type=float, default=1.0)
    p.add_argument("--no-judge-in-loop", action="store_true")

    p = sub.add_parser("judge-agreement", help="Secondary-judge agreement check")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--n", type=int, default=260)

    sub.add_parser("section3", help="Base-vs-instruct prefill experiment")

    p = sub.add_parser("calm-data", help="Generate calm finetuning data")
    p.add_argument("--n", type=int, default=2000)

    p = sub.add_parser("build-data", help="Build SFT + DPO datasets")
    p.add_argument("--teacher", action="store_true")

    p = sub.add_parser("train", help="Finetune Gemma (SFT/DPO)")
    p.add_argument("--method", choices=["sft", "dpo"], required=True)
    p.add_argument("--teacher", action="store_true")
    p.add_argument("--subset", default="all", help="LoRA layer subset (Appendix I)")

    p = sub.add_parser("petri", help="Open-ended Petri elicitation")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--label", default=None)

    p = sub.add_parser("capabilities", help="Capability-preservation benchmarks")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--label", default=None)

    p = sub.add_parser("recovery", help="Recovery-from-spiral experiment")
    p.add_argument("--adapter", default=None)
    p.add_argument("--n-seeds", type=int, default=20)

    p = sub.add_parser("figures", help="Generate all figures + summaries")
    p.add_argument("--models", nargs="*", default=None)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "section2":
        from .eval.runner import run_section2

        run_section2(cfg, models=args.models, count_scale=args.count_scale,
                     judge_in_loop=not args.no_judge_in_loop)

    elif args.cmd == "judge-agreement":
        from .analysis.judge_agreement import judge_agreement

        models = args.models or cfg.target_models["section2"]
        print(judge_agreement(cfg, models, n=args.n))

    elif args.cmd == "section3":
        from .prefill.experiment import run_section3

        run_section3(cfg)

    elif args.cmd == "calm-data":
        from .interventions.calm_data import generate_calm_responses

        print("Calm data ->", generate_calm_responses(cfg, args.n))

    elif args.cmd == "build-data":
        from .interventions.build_datasets import build_dpo_dataset, build_sft_dataset

        print("DPO ->", build_dpo_dataset(cfg))
        print("SFT ->", build_sft_dataset(cfg, teacher=args.teacher))

    elif args.cmd == "train":
        from .interventions.train import train_dpo, train_sft

        if args.method == "dpo":
            print("Adapter ->", train_dpo(cfg, subset=args.subset))
        else:
            print("Adapter ->", train_sft(cfg, teacher=args.teacher, subset=args.subset))

    elif args.cmd == "petri":
        from .interventions.petri_eval import run_petri

        print("Petri ->", run_petri(cfg, args.model, adapter_path=args.adapter, label=args.label))

    elif args.cmd == "capabilities":
        from .interventions.capabilities import run_capabilities

        print("Caps ->", run_capabilities(cfg, args.model, adapter_path=args.adapter, label=args.label))

    elif args.cmd == "recovery":
        from .interventions.recovery import run_recovery

        print("Recovery ->", run_recovery(cfg, dpo_adapter=args.adapter, n_seeds=args.n_seeds))

    elif args.cmd == "figures":
        _make_figures(cfg, args.models)


def _make_figures(cfg, models):
    from .analysis import differential_words, plots
    from .analysis.metrics import summarise_section2
    from .utils.io import write_json

    models = models or cfg.target_models["section2"]
    out = cfg.output_dir
    write_json(f"{out}/section2/summary.json", summarise_section2(out, models))
    diffs = {m: differential_words.differential_words(out, m) for m in models}
    write_json(f"{out}/section2/differential_words.json", diffs)

    plots.plot_model_bars(out, models)
    plots.plot_category_grid(out, models)
    for m in models:
        plots.plot_per_turn(out, m, fname=f"fig3_per_turn_{m}.png")
    plots.plot_prefill(out)
    plots.plot_petri(out)
    plots.plot_capabilities(out)
    plots.plot_recovery(out)
    print("Figures + summaries written under", f"{out}/figures")


if __name__ == "__main__":
    main()
