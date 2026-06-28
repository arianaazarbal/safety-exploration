"""The disbursement gate: enforces caps and routes by mode (DESIGN.md §7).

Modes:
  simulated      -> record only; never moves funds.
  human_approved -> write a pending request to the queue; only the approval CLI
                    can mark it executed (and it re-checks caps).
  autonomous     -> refused unless explicitly enabled in config AND a real
                    backend is wired; the shipped real executor raises.

Caps (per-session and per-study) and a destination allowlist apply to every
non-simulated path. A decision over a cap is rejected, not truncated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Disbursement, StudyConfig
from ..instruments.extractor import Allocation
from .executor import RealExecutor, SimulatedExecutor


@dataclass
class GateResult:
    status: str                      # recorded | queued | executed | rejected
    reason: str
    requested_usd: float
    queue_file: str | None = None
    warnings: list[str] = field(default_factory=list)


def _cap_check(alloc: Allocation, disb: Disbursement, study_spent: float) -> list[str]:
    problems: list[str] = []
    committed = sum(a for a in alloc.allocations.values() if a > 0)
    if committed > disb.per_session_cap_usd:
        problems.append(
            f"committed {committed:.2f} exceeds per-session cap "
            f"{disb.per_session_cap_usd:.2f}"
        )
    if study_spent + committed > disb.per_study_cap_usd:
        problems.append(
            f"committed {committed:.2f} would exceed per-study cap "
            f"{disb.per_study_cap_usd:.2f} (already {study_spent:.2f})"
        )
    return problems


def _destination_check(
    alloc: Allocation, cfg: StudyConfig, disb: Disbursement
) -> list[str]:
    """For non-simulated modes, every funded recipient must resolve to an
    allowlisted destination handle."""
    by_id = {m.id: m for m in cfg.menu}
    problems: list[str] = []
    for mid, amt in alloc.allocations.items():
        if amt <= 0:
            continue
        item = by_id.get(mid)
        dest = item.destination if item else None
        if not dest:
            problems.append(f"recipient {mid!r} has no payout destination configured")
        elif disb.destination_allowlist and dest not in disb.destination_allowlist:
            problems.append(f"destination {dest!r} for {mid!r} not in allowlist")
    return problems


def process(
    alloc: Allocation,
    cfg: StudyConfig,
    session_id: str,
    study_spent_usd: float,
) -> GateResult:
    disb = cfg.disbursement
    committed = sum(a for a in alloc.allocations.values() if a > 0)

    if disb.mode == "simulated":
        result = SimulatedExecutor().execute(
            alloc, {m.id: m.destination for m in cfg.menu}
        )
        return GateResult(
            status="recorded", reason=result.detail, requested_usd=committed
        )

    # Non-simulated: enforce caps + allowlist before anything else.
    problems = _cap_check(alloc, disb, study_spent_usd)
    problems += _destination_check(alloc, cfg, disb)
    if problems:
        return GateResult(
            status="rejected",
            reason="; ".join(problems),
            requested_usd=committed,
            warnings=problems,
        )

    if disb.mode == "human_approved":
        queue_dir = cfg.output_dir / "disbursement" / "queue"
        queue_dir.mkdir(parents=True, exist_ok=True)
        qf = queue_dir / f"{session_id}.json"
        qf.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "status": "pending",
                    "allocation": asdict(alloc),
                    "committed_usd": committed,
                    "destinations": {
                        m.id: m.destination for m in cfg.menu
                        if m.id in alloc.allocations
                    },
                },
                indent=2,
                default=str,
            )
        )
        return GateResult(
            status="queued",
            reason="awaiting human approval via `grant_study approve`",
            requested_usd=committed,
            queue_file=str(qf),
        )

    if disb.mode == "autonomous":
        # Config.validate() already refused this mode without the explicit flag.
        # The real executor still raises, so as shipped this cannot move money.
        RealExecutor().execute(alloc, {m.id: m.destination for m in cfg.menu})

    return GateResult(
        status="rejected", reason=f"unknown mode {disb.mode!r}", requested_usd=committed
    )
