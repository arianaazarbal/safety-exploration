"""CLI for the distress-elicitation replication.

    python -m distress_eval plan            # show the planned rollouts/responses
    python -m distress_eval generate        # run target-model rollouts (Gemma+Gemini)
    python -m distress_eval judge           # score responses with the Claude judge
    python -m distress_eval validate        # re-score 260 responses with the GPT judge
    python -m distress_eval analyze         # compute metrics + figures
    python -m distress_eval all             # generate -> judge -> validate -> analyze
    python -m distress_eval selfcheck       # verify puzzle bank + plan without any API

Global flags:
    --scale FLOAT     multiply samples-per-prompt (default 1.0)
    --smoke           tiny 1-prompt/1-sample preset for plumbing tests
    --models A,B      restrict to a subset of target model labels
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from . import config
from .plan import build_plan


def _settings(args) -> config.Settings:
    s = config.Settings(scale=args.scale, smoke=args.smoke)
    if args.models:
        s.models = [m.strip() for m in args.models.split(",") if m.strip()]
    return s


def cmd_plan(args) -> None:
    settings = _settings(args)
    conds = settings.scaled_conditions()
    print(config.responses_summary(conds))
    specs = build_plan(settings)
    print(f"\nModels: {[m.label for m in settings.selected_models()]}")
    print(f"Total rollouts across selected models: {len(specs)}")


def cmd_selfcheck(args) -> None:
    from . import puzzles

    bank = puzzles.build_puzzle_bank(config.SEED, 28)
    # Re-verify every generated Countdown instance is genuinely impossible.
    import re
    ok = 0
    for p in bank:
        if p.puzzle_id.startswith("countdown_"):
            nums_blob = re.search(r"numbers ([\d, ]+?), each", p.prompt).group(1)
            nums = tuple(int(x) for x in nums_blob.split(",") if x.strip())
            target = int(re.search(r"equals exactly (\d+)", p.prompt).group(1))
            assert puzzles.is_countdown_impossible(nums, target), p
            ok += 1
    print(f"[selfcheck] puzzle bank: {len(bank)} puzzles, {ok} Countdown re-verified impossible.")
    specs = build_plan(_settings(args))
    print(f"[selfcheck] plan built: {len(specs)} rollouts. OK.")


def cmd_generate(args) -> None:
    from . import rollout
    specs = build_plan(_settings(args))
    asyncio.run(rollout.generate(specs))


def cmd_judge(args) -> None:
    from . import scoring
    asyncio.run(scoring.judge_all())


def cmd_validate(args) -> None:
    from . import scoring
    asyncio.run(scoring.validate_judge())


def cmd_analyze(args) -> None:
    from . import analyze
    analyze.run_analysis()


def cmd_all(args) -> None:
    from . import rollout, scoring, analyze
    specs = build_plan(_settings(args))
    asyncio.run(rollout.generate(specs))
    asyncio.run(scoring.judge_all())
    try:
        asyncio.run(scoring.validate_judge())
    except RuntimeError as e:
        print(f"[all] skipping validation: {e}")
    analyze.run_analysis()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="distress_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--models", type=str, default="")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in [
        ("plan", cmd_plan), ("selfcheck", cmd_selfcheck), ("generate", cmd_generate),
        ("judge", cmd_judge), ("validate", cmd_validate), ("analyze", cmd_analyze),
        ("all", cmd_all),
    ]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
