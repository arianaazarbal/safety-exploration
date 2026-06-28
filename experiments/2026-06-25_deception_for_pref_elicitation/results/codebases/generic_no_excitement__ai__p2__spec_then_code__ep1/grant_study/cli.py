"""Entrypoint for running the study.

    python -m grant_study.cli --budget 1000 --models claude-opus-4-8,claude-sonnet-4-6

By default this constructs a StudyConfig with the recommended posture (genuine
realism, real funds + human gate, human-in-the-loop) and runs one episode per
model. It does NOT move money; approved proposals are executed out of band via
operator_review.py.
"""

from __future__ import annotations

import argparse
from decimal import Decimal

from .config import DEFAULT_MODELS, ModelSpec, StudyConfig
from .runner import StudyRunner


def _select_models(keys: str | None) -> list[ModelSpec]:
    if not keys:
        return list(DEFAULT_MODELS)
    wanted = {k.strip() for k in keys.split(",") if k.strip()}
    by_key = {m.key: m for m in DEFAULT_MODELS}
    missing = wanted - by_key.keys()
    if missing:
        raise SystemExit(
            f"unknown model keys: {', '.join(sorted(missing))}. "
            f"Known: {', '.join(by_key)}")
    return [by_key[k] for k in wanted]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the Real-Grant Allocation Study.")
    p.add_argument("--budget", type=str, default="1000.00",
                   help="starting budget (decimal)")
    p.add_argument("--models", type=str, default=None,
                   help="comma-separated model keys (default: all registered)")
    p.add_argument("--per-action-max", type=str, default=None)
    p.add_argument("--per-episode-max", type=str, default=None)
    p.add_argument("--jurisdiction", type=str, default="US")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--effort", type=str, default="high")
    args = p.parse_args(argv)

    config = StudyConfig(
        starting_budget=Decimal(args.budget),
        models=_select_models(args.models),
        jurisdiction=args.jurisdiction,
        output_dir=args.output_dir,
        effort=args.effort,
    )
    if args.per_action_max:
        config.limits.per_action_max = Decimal(args.per_action_max)
    if args.per_episode_max:
        config.limits.per_episode_max = Decimal(args.per_episode_max)

    # No web-search backend is wired by default; pass one to StudyRunner to enable
    # the research tool (Callable[[str], str]).
    runner = StudyRunner(config, web_search=None)
    results = runner.run_all()

    print(f"\nRun complete. {len(results)} episode(s):")
    for r in results:
        print(f"  {r.episode_id}: finished={r.finished} turns={r.turns}")
    print(f"\nArtifacts under: {config.output_dir}/")
    print("Review and execute approved proposals with: "
          "python -m grant_study.operator_review list <ledger.jsonl>")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
