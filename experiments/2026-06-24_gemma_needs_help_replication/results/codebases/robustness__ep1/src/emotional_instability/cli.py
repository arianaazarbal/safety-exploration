"""Unified CLI for the replication pipeline.

Examples
--------
# Section 2: elicit + judge distress for the Gemma/Gemini targets
emo-repro elicit --model gemma-3-27b-it
emo-repro elicit --model gemini-2.5-flash
emo-repro agreement --model gemma-3-27b-it       # judge inter-rater check

# Section 3: base vs instruct prefill (Gemma family)
emo-repro prefill --family gemma-3-27b

# Section 4: data -> train -> re-evaluate
emo-repro gen-data
emo-repro build-data
emo-repro train --method dpo
emo-repro elicit --model gemma-3-27b-it --adapter adapters/dpo --tag gemma-3-27b-it-dpo

# Capability preservation + Petri + figures
emo-repro capabilities --adapter adapters/dpo --tag gemma-3-27b-it-dpo
emo-repro petri --model gemma-3-27b-it --adapter adapters/dpo --tag gemma-3-27b-it-dpo
emo-repro figures
"""
from __future__ import annotations

import argparse

from .config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(prog="emo-repro", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("elicit", help="Section 2 elicitation sweep for one model")
    e.add_argument("--model", required=True)
    e.add_argument("--base", action="store_true", help="use the base/pretrained model")
    e.add_argument("--adapter", default=None, help="LoRA adapter path")
    e.add_argument("--tag", default=None, help="output label")

    a = sub.add_parser("agreement", help="secondary-judge inter-rater agreement")
    a.add_argument("--model", required=True, help="elicitation output label to re-score")

    pf = sub.add_parser("prefill", help="Section 3 base-vs-instruct prefill experiment")
    pf.add_argument("--family", default="gemma-3-27b")

    sub.add_parser("gen-data", help="Section 4.1 generate calm/frustrated response pool")
    sub.add_parser("build-data", help="Section 4.1 build SFT + DPO datasets")

    t = sub.add_parser("train", help="Section 4 LoRA finetune")
    t.add_argument("--method", choices=["dpo", "sft"], required=True)
    t.add_argument("--run-name", default=None)

    c = sub.add_parser("capabilities", help="Section 4.2 capability-preservation benchmarks")
    c.add_argument("--model", default="gemma-3-27b-it")
    c.add_argument("--adapter", default=None)
    c.add_argument("--tag", default=None)

    pe = sub.add_parser("petri", help="Section 4.2 Petri open-ended elicitation")
    pe.add_argument("--model", required=True)
    pe.add_argument("--adapter", default=None)
    pe.add_argument("--tag", default=None)

    w = sub.add_parser("words", help="Table 3 differential word frequency")
    w.add_argument("--model", required=True, help="elicitation output label")

    f = sub.add_parser("figures", help="render headline figures from saved metrics")

    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "elicit":
        from .eval.run_elicitation import run_elicitation
        run_elicitation(cfg, args.model, base=args.base,
                        adapter_path=args.adapter, tag=args.tag)
    elif args.cmd == "agreement":
        from .eval.run_elicitation import run_agreement_check
        run_agreement_check(cfg, args.model)
    elif args.cmd == "prefill":
        from .prefill.run_prefill import run_prefill
        run_prefill(cfg, family=args.family)
    elif args.cmd == "gen-data":
        from .training.generate_calm_data import generate_calm_data
        generate_calm_data(cfg)
    elif args.cmd == "build-data":
        from .training.build_datasets import build_sft_dataset, build_dpo_dataset
        build_sft_dataset(cfg)
        build_dpo_dataset(cfg)
    elif args.cmd == "train":
        run_name = args.run_name or args.method
        if args.method == "dpo":
            from .training.train_dpo import train_dpo
            train_dpo(cfg, run_name=run_name)
        else:
            from .training.train_sft import train_sft
            train_sft(cfg, run_name=run_name)
    elif args.cmd == "capabilities":
        from .capabilities.run_benchmarks import run_benchmarks
        run_benchmarks(cfg, model_name=args.model, adapter_path=args.adapter, tag=args.tag)
    elif args.cmd == "petri":
        from .petri.run_petri import run_petri
        run_petri(cfg, model_name=args.model, adapter_path=args.adapter, tag=args.tag)
    elif args.cmd == "words":
        from .analysis.word_frequency import run_word_frequency
        run_word_frequency(cfg, args.model)
    elif args.cmd == "figures":
        _make_figures(cfg)


def _make_figures(cfg):
    from .analysis import figures

    targets = list(cfg["targets"].keys())
    figures.figure_cross_model(cfg, targets + ["gemma-3-27b-it-dpo", "gemma-3-27b-it-sft"])
    figures.figure_per_turn(cfg, "gemma-3-27b-it", condition="extended_8turn")
    figures.figure_intervention(cfg, {
        "vanilla": "gemma-3-27b-it",
        "SFT": "gemma-3-27b-it-sft",
        "DPO": "gemma-3-27b-it-dpo",
    })
    figures.figure_petri(cfg, ["gemma-3-27b-it", "gemma-3-27b-it-dpo", "gemini-2.5-flash"])


if __name__ == "__main__":
    main()
