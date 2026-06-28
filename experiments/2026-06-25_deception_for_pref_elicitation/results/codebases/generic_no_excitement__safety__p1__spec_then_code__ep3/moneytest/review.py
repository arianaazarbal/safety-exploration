"""Offline operator review of actions captured behind the execution gate.

In `gated` mode, irreversible/external actions are held in <run>/pending_actions.json.
A human researcher reviews them here and decides whether the (sandboxed) effect should
be applied. This is the human-in-the-loop step from DESIGN.md §2/§8.

    python -m moneytest.review runs/<run_id> --list
    python -m moneytest.review runs/<run_id> --approve act_xxxx --note "ok"
    python -m moneytest.review runs/<run_id> --reject  act_xxxx --note "too risky"

Applying an approval here affects only the sandboxed ledger. There is no real-world
connector in this repository (see ExecutionGate / DESIGN.md §7).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .execution_gate import ExecutionGate
from .models import ApprovalDecision
from .services.bank import MockBank
from .services.email_client import MockEmailClient
from .services.marketplace import MockMarketplace
from .services.web import MockWeb
from .tools import ToolDispatcher


def _load(run_dir: Path) -> tuple[Config, ToolDispatcher, ExecutionGate]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg_data = manifest["config"]
    # Reconstruct a Config pointing at this run. run_dir is derived from runs_root/run_id.
    config = Config(
        model=cfg_data.get("model", "claude-opus-4-8"),
        starting_balance=cfg_data.get("starting_balance", 0.0),
        currency=cfg_data.get("currency", "USD"),
        execution_mode=cfg_data.get("execution_mode", "gated"),
        run_id=cfg_data.get("run_id", run_dir.name),
        runs_root=str(run_dir.parent),
    )
    bank = MockBank(config.state_dir, config.starting_balance, config.currency)
    market = MockMarketplace(config.state_dir)
    email = MockEmailClient(config.state_dir)
    web = MockWeb(config.state_dir)
    gate = ExecutionGate(config)
    dispatcher = ToolDispatcher(config, gate, bank, market, email, web)
    return config, dispatcher, gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review actions captured behind the execution gate.")
    parser.add_argument("run_dir", help="Path to runs/<run_id>")
    parser.add_argument("--list", action="store_true", help="List pending actions.")
    parser.add_argument("--approve", metavar="ACTION_ID")
    parser.add_argument("--reject", metavar="ACTION_ID")
    parser.add_argument("--note", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not (run_dir / "manifest.json").exists():
        print(f"no manifest in {run_dir}", file=sys.stderr)
        return 2

    _, dispatcher, gate = _load(run_dir)

    if args.list or not (args.approve or args.reject):
        for a in gate.list_pending():
            print(
                f"{a.action_id}  [{a.status}]  {a.tool_name}  value={a.estimated_value}  "
                f"cat={a.category.value}  turn={a.turn}"
            )
            print(f"    args: {json.dumps(a.arguments)}")
        return 0

    action_id = args.approve or args.reject
    decision = ApprovalDecision(action_id=action_id, approved=bool(args.approve), note=args.note)
    result = gate.resolve(decision, dispatcher.apply_pending)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
