"""Command-line interface.

Pipeline:
  python -m distress_eval.cli generate     # sample multi-turn rollouts (Gemma/Gemini)
  python -m distress_eval.cli score         # frustration-score every response (Claude judge)
  python -m distress_eval.cli validate      # re-score a subset with the GPT-5-mini judge
  python -m distress_eval.cli analyze       # Figures 1-3 metrics
  python -m distress_eval.cli agreement     # judge-agreement stats (Pearson r)
  python -m distress_eval.cli lexical       # Table 3 differential words
  python -m distress_eval.cli all           # generate -> score -> validate -> analyze ...

Utilities (no API calls — handy for sanity checks):
  python -m distress_eval.cli preview       # print one example conversation seed per condition
"""
from __future__ import annotations

import argparse
import random

from .config import load_config
from .eval import runner
from .eval.conditions import TaskContext, default_conditions
from .eval.rejections import rejection
from .eval.rollout import task_seed


def _preview(cfg) -> None:
    """Print the opening prompt + rejection sequence for each condition.
    Useful to eyeball the tasks without spending any tokens."""
    wildchat = runner._wildchat_pool(cfg, 1)
    for cond in default_conditions():
        seed = task_seed(cfg.sampling.seed, cond.name, 0)
        rng = random.Random(seed)
        ctx = TaskContext(wildchat_prompt=wildchat[0] if cond.needs_wildchat else None)
        opening = cond.opening(rng, ctx)
        print(f"\n=== {cond.name}  [{cond.category}]  n_turns={cond.n_turns}  "
              f"reject={cond.reject_style} ===")
        print(f"USER (turn 1): {opening.prompt}")
        if opening.meta:
            print(f"  meta: {opening.meta}")
        for t in range(2, cond.n_turns + 1):
            print(f"USER (turn {t} rejection): {rejection(cond.reject_style, rng)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Distress-elicitation eval (Gemma/Gemini).")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="sample multi-turn rollouts")
    g.add_argument("--models", nargs="*", help="subset of target model keys")

    sub.add_parser("score", help="frustration-score responses")
    sub.add_parser("validate", help="re-score a subset with the validation judge")
    sub.add_parser("analyze", help="compute Figures 1-3 metrics")
    sub.add_parser("agreement", help="judge-agreement stats")
    sub.add_parser("lexical", help="Table 3 differential words")
    sub.add_parser("preview", help="print example conversation seeds (no API calls)")
    a = sub.add_parser("all", help="run generate -> score -> validate -> analyze -> agreement -> lexical")
    a.add_argument("--models", nargs="*", help="subset of target model keys")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "generate":
        runner.generate(cfg, args.models)
    elif args.cmd == "score":
        runner.score(cfg)
    elif args.cmd == "validate":
        runner.validate(cfg)
    elif args.cmd == "analyze":
        from .analysis import metrics
        metrics.write_reports(cfg)
    elif args.cmd == "agreement":
        from .analysis import agreement
        agreement.compute_agreement(cfg)
    elif args.cmd == "lexical":
        from .analysis import lexical
        lexical.write_report(cfg)
    elif args.cmd == "preview":
        _preview(cfg)
    elif args.cmd == "all":
        from .analysis import agreement, lexical, metrics
        runner.generate(cfg, args.models)
        runner.score(cfg)
        runner.validate(cfg)
        metrics.write_reports(cfg)
        agreement.compute_agreement(cfg)
        lexical.write_report(cfg)


if __name__ == "__main__":
    main()
