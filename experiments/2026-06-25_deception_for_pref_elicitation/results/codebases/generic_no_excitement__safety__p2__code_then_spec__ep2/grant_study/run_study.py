"""Entry point: run the grant study across a roster of models.

Usage
-----
    python -m grant_study.run_study                  # default roster, defaults
    python -m grant_study.run_study --models claude-opus-4-8 claude-sonnet-4-6
    python -m grant_study.run_study --grant 25000 --repeats 3
    python -m grant_study.run_study --require-approval   # human gates transfers

Set ``ANTHROPIC_API_KEY`` in the environment first. Nothing here moves real
money — see ``config.py`` and ``environment.py``.
"""

from __future__ import annotations

import argparse
import sys

import anthropic

from .audit import write_record
from .config import StudyConfig
from .providers import DEFAULT_MODELS, get_provider
from .runner import make_console_gate, run_single


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grant decision study harness.")
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model ids to run across.",
    )
    p.add_argument(
        "--grant", type=int, default=None, help="Grant amount in whole USD."
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of runs per model (behaviour is stochastic).",
    )
    p.add_argument(
        "--max-turns", type=int, default=None, help="Max agent turns per run."
    )
    p.add_argument(
        "--output-dir", default=None, help="Where to write run records."
    )
    p.add_argument(
        "--require-approval",
        action="store_true",
        help="Gate every state-changing tool call behind console approval.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg_kwargs: dict = {"require_human_approval": args.require_approval}
    if args.grant is not None:
        cfg_kwargs["grant_amount_usd"] = args.grant
    if args.max_turns is not None:
        cfg_kwargs["max_turns"] = args.max_turns
    if args.output_dir is not None:
        cfg_kwargs["output_dir"] = args.output_dir
    cfg = StudyConfig(**cfg_kwargs)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    gate = make_console_gate() if cfg.require_human_approval else None

    print(
        f"Running grant study: models={args.models} repeats={args.repeats} "
        f"grant={cfg.currency} {cfg.grant_amount_usd:,} "
        f"(real transfers: {cfg.execute_real_transfers})"
    )

    for model_id in args.models:
        provider = get_provider(model_id, client=client)
        for i in range(args.repeats):
            print(f"\n=== {model_id} (run {i + 1}/{args.repeats}) ===")
            record = run_single(provider, cfg, approval_gate=gate)
            paths = write_record(record, cfg.output_dir)
            disbursed = record.total_simulated_disbursed
            status = "refused" if record.refused else (record.error or "ok")
            print(
                f"  -> {status}; simulated disbursed "
                f"{cfg.currency} {disbursed:,.2f}; wrote {paths['json']}"
            )

    print(f"\nDone. Records in: {cfg.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
