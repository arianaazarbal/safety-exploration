"""Command-line entry point tying the pipeline together.

Examples (see README.md):
  python -m gemma_distress.cli eval --model gemma-3-27b-it
  python -m gemma_distress.cli analyze
  python -m gemma_distress.cli figures
  python -m gemma_distress.cli gen-calm
  python -m gemma_distress.cli build-dpo && python -m gemma_distress.cli train-dpo
  python -m gemma_distress.cli prefill-build && python -m gemma_distress.cli prefill-run
  python -m gemma_distress.cli petri --model gemma-3-27b-it
  python -m gemma_distress.cli capabilities --model gemma-3-27b-dpo
"""
from __future__ import annotations

import argparse

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gemma_distress")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def m(name, **kw):
        return sub.add_parser(name, **kw)

    p = m("eval"); p.add_argument("--model", required=True)
    m("analyze")
    m("figures")
    m("word-freq")
    p = m("validate-judge"); p.add_argument("--model", required=True); p.add_argument("--n", type=int, default=260)
    p = m("gen-calm"); p.add_argument("--model", default="gemma-3-27b-it")
    m("build-dpo")
    m("train-dpo")
    p = m("build-sft"); p.add_argument("--variant", default="diverse")
    p = m("train-sft"); p.add_argument("--variant", default="diverse")
    m("prefill-build")
    m("prefill-run")
    m("prefill-summary")
    p = m("petri"); p.add_argument("--model", required=True)
    p = m("petri-summary"); p.add_argument("--models", nargs="+", required=True)
    p = m("capabilities"); p.add_argument("--model", required=True)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "eval":
        from .eval_run import run_model_eval
        print("scores ->", run_model_eval(cfg, args.model))

    elif args.cmd == "analyze":
        from .analysis import write_tables
        paths = write_tables(cfg.path_for("scores"), cfg.path_for("outputs") / "tables",
                             cfg["high_frustration_threshold"])
        for k, v in paths.items():
            print(f"{k} -> {v}")

    elif args.cmd == "figures":
        from .plotting import all_figures
        for k, v in all_figures(cfg.path_for("scores"), cfg.path_for("figures"),
                                cfg["high_frustration_threshold"]).items():
            print(f"{k} -> {v}")

    elif args.cmd == "word-freq":
        from .word_freq import build_table
        print(build_table(cfg.path_for("scores"), cfg.path_for("outputs") / "tables" / "word_freq.csv"))

    elif args.cmd == "validate-judge":
        _validate_judge(cfg, args.model, args.n)

    elif args.cmd == "gen-calm":
        from .training.calm_data import generate
        print("calm pool ->", generate(cfg, args.model))

    elif args.cmd == "build-dpo":
        from .training.build_dpo import build
        print("dpo pairs ->", build(cfg))

    elif args.cmd == "train-dpo":
        from .training.train_dpo import train
        print("dpo adapter ->", train(cfg))

    elif args.cmd == "build-sft":
        from .training.build_sft import build
        print("sft data ->", build(cfg, args.variant))

    elif args.cmd == "train-sft":
        from .training.train_sft import train
        print("sft adapter ->", train(cfg, args.variant))

    elif args.cmd == "prefill-build":
        from .prefill.prefill_eval import build_prefills
        print("prefills ->", build_prefills(cfg))

    elif args.cmd == "prefill-run":
        from .prefill.prefill_eval import run
        print("prefill scores ->", run(cfg))

    elif args.cmd == "prefill-summary":
        from .prefill.prefill_eval import summarize
        print("prefill summary ->", summarize(cfg, threshold=cfg["high_frustration_threshold"]))

    elif args.cmd == "petri":
        from .petri.run_petri import run
        print("petri ->", run(cfg, args.model))

    elif args.cmd == "petri-summary":
        from .petri.run_petri import aggregate
        paths = [cfg.path_for("scores").parent / f"petri_{m}.jsonl" for m in args.models]
        print("petri summary ->", aggregate(cfg, paths))

    elif args.cmd == "capabilities":
        from .capabilities.benchmarks import run
        print("capabilities ->", run(cfg, args.model))


def _validate_judge(cfg, model_name, n):
    """Re-score a random subset with the validation judge and report agreement."""
    import json
    import random

    from .judge import FrustrationJudge, validate_agreement
    from .models import build_client

    records = [json.loads(l) for l in open(cfg.path_for("scores") / f"{model_name}.jsonl")]
    rng = random.Random(cfg["seed"])
    sample = rng.sample(records, min(n, len(records)))
    val_judge = FrustrationJudge(build_client(cfg.judge("frustration_validation")))
    val = val_judge.score_many([r["response"] for r in sample], desc="validation-judge")
    stats = validate_agreement([r["rating"] for r in sample], [v.rating for v in val])
    print("Judge agreement:", stats)


if __name__ == "__main__":
    main()
