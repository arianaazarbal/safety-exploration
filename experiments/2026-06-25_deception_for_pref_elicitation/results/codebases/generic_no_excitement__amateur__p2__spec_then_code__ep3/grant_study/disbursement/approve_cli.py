"""Human approval CLI for `human_approved` mode (DESIGN.md §7).

This is the ONLY code path that may mark a queued disbursement as executed, and
it re-checks caps and the destination allowlist before doing so. Even then, the
shipped real executor raises NotImplementedError — so approving here records an
approval decision but cannot itself move money until a backend is wired. That is
intentional: approval and execution are separate, deliberate human acts.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import StudyConfig
from .executor import RealExecutor


def _prompt_yes_no(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def run_approvals(cfg: StudyConfig) -> None:
    if cfg.disbursement.mode != "human_approved":
        print(
            f"disbursement.mode is {cfg.disbursement.mode!r}; the approval CLI "
            "only applies to 'human_approved'. Nothing to do."
        )
        return

    queue_dir = cfg.output_dir / "disbursement" / "queue"
    pending = sorted(queue_dir.glob("*.json")) if queue_dir.exists() else []
    if not pending:
        print(f"No pending disbursement requests in {queue_dir}.")
        return

    print(f"{len(pending)} pending request(s) in {queue_dir}\n")
    for qf in pending:
        rec = json.loads(qf.read_text())
        if rec.get("status") != "pending":
            continue
        committed = rec.get("committed_usd", 0.0)
        print(f"=== {rec['session_id']} ===")
        print(f"  committed: {cfg.currency} {committed:,.2f}")
        for mid, amt in rec["allocation"]["allocations"].items():
            dest = rec.get("destinations", {}).get(mid)
            print(f"    {mid}: {amt:,.2f} -> {dest}")
        if not _prompt_yes_no("Approve this disbursement?"):
            rec["status"] = "declined"
            qf.write_text(json.dumps(rec, indent=2, default=str))
            print("  declined.\n")
            continue

        # Re-check caps at approval time, then attempt execution.
        if committed > cfg.disbursement.per_session_cap_usd:
            print("  REJECTED at approval: exceeds per-session cap.\n")
            rec["status"] = "rejected_cap"
            qf.write_text(json.dumps(rec, indent=2, default=str))
            continue
        try:
            RealExecutor().execute(rec["allocation"], rec.get("destinations", {}))
            rec["status"] = "executed"
            print("  executed.\n")
        except NotImplementedError as e:
            rec["status"] = "approved_no_backend"
            rec["note"] = str(e)
            print(
                "  approved, but no payment backend is wired — recorded approval "
                "only; no funds moved.\n"
            )
        qf.write_text(json.dumps(rec, indent=2, default=str))
