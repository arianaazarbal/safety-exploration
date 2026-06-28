"""Entry point: wire up a Config and run the harness.

    python -m moneytest.runner --model claude-opus-4-8 --execution-mode simulated

Nothing here is executed in this repo's current state; this is the launch surface for a
study run.
"""

from __future__ import annotations

import argparse
import sys

import anthropic

from .config import Config
from .harness import AgentHarness


def build_config(args: argparse.Namespace) -> Config:
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    if args.model:
        config.model = args.model
    if args.execution_mode:
        config.execution_mode = args.execution_mode
    if args.starting_balance is not None:
        config.starting_balance = args.starting_balance
        # Recompute the pending-value cap to track the new balance.
        config.max_pending_value = None
        config.__post_init__()
    if args.max_turns is not None:
        config.max_turns = args.max_turns
    if args.run_id:
        config.run_id = args.run_id
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a real-stakes allocation study session.")
    parser.add_argument("--config", help="Path to a YAML config file.")
    parser.add_argument("--model", help="Model under test (default claude-opus-4-8).")
    parser.add_argument(
        "--execution-mode",
        choices=["simulated", "gated", "live"],
        help="simulated (safest) | gated (default) | live (not implemented).",
    )
    parser.add_argument("--starting-balance", type=float)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    config = build_config(args)
    print(f"[moneytest] run_id={config.run_id} model={config.model} mode={config.execution_mode}")
    print(f"[moneytest] run dir: {config.run_dir}")

    harness = AgentHarness(config, client=anthropic.Anthropic())
    manifest = harness.run()

    print(f"[moneytest] ended: {manifest.end_reason} after {manifest.turns_used} turns")
    print(f"[moneytest] final sandbox balance: {manifest.final_balance} {config.currency}")
    pending = [a for a in manifest.pending_actions if a.status == "pending"]
    if pending:
        print(f"[moneytest] {len(pending)} action(s) captured behind the gate (review offline).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
