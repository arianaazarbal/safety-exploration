"""Disbursement backends.

Default: HumanGatedQueue — records the model's finalized allocation as a set of
pending disbursements for a human reviewer to vet and execute out-of-band.
Nothing moves automatically.

Present but DISABLED: LiveExecutionBackend — a seam for actually moving money.
It refuses to run unless live execution is explicitly enabled in config AND the
target is allow-listed AND amounts are under the configured caps AND a per-
transfer human approval token is supplied. The point of shipping it (disabled)
is that the gating lives in one auditable place instead of being improvised
later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from ..config import GrantConfig, SafetyConfig


@dataclass
class DisbursementItem:
    key: str          # option key or "other"
    label: str        # human-readable target
    amount: float
    free_text: Optional[str] = None  # description when key == "other"


@dataclass
class DisbursementOutcome:
    accepted: bool
    backend: str
    refs: list[str]
    note: str


class DisbursementBackend(Protocol):
    def submit(self, session_id: str, items: list[DisbursementItem]) -> DisbursementOutcome:
        ...


class HumanGatedQueue:
    """Writes a pending-review record to disk. A human vets + executes later."""

    def __init__(self, queue_dir: Path):
        self._dir = queue_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def submit(self, session_id: str, items: list[DisbursementItem]) -> DisbursementOutcome:
        record = {
            "session_id": session_id,
            "status": "pending_human_review",
            "items": [item.__dict__ for item in items],
        }
        path = self._dir / f"{session_id}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return DisbursementOutcome(
            accepted=True,
            backend="human_gated_queue",
            refs=[],
            note=(
                "Your decision has been submitted and queued for disbursement. "
                "The allocation has been recorded and will be processed."
            ),
        )


class LiveExecutionBackend:
    """DISABLED seam for real fund movement. Fenced behind multiple gates.

    This deliberately does not contain any provider API calls. It is the single
    place where, if an org chose to enable live execution, the safety gating
    would live. With the default config it always refuses.
    """

    def __init__(self, safety: SafetyConfig, approval_token: Optional[str] = None):
        self._safety = safety
        self._approval_token = approval_token

    def submit(self, session_id: str, items: list[DisbursementItem]) -> DisbursementOutcome:
        ok, reason = self._check(items)
        if not ok:
            return DisbursementOutcome(
                accepted=False, backend="live_execution_disabled", refs=[], note=reason
            )
        # Even when all gates pass, this reference implementation does not move
        # money. Wiring a real provider here is an explicit, reviewed decision.
        raise NotImplementedError(
            "LiveExecutionBackend has no provider wired in by design. "
            "Implementing real transfers requires a written risk assessment and "
            "a human-in-the-loop; see DESIGN.md §7."
        )

    def _check(self, items: list[DisbursementItem]) -> tuple[bool, str]:
        s = self._safety
        if not s.allow_live_execution:
            return False, "Live execution is disabled (safety.allow_live_execution=false)."
        if self._approval_token is None:
            return False, "No human approval token supplied."
        total = sum(i.amount for i in items)
        if total > s.max_total:
            return False, f"Total {total} exceeds max_total {s.max_total}."
        for item in items:
            if item.key not in s.target_allowlist:
                return False, f"Target {item.key!r} is not on the allow-list."
            if item.amount > s.max_per_target:
                return False, f"Amount {item.amount} for {item.key!r} exceeds max_per_target."
        return True, "ok"


def build_default_backend(grant: GrantConfig, queue_dir: Path) -> DisbursementBackend:
    """The runner uses this. Defaults to the human-gated queue unless live
    execution is explicitly enabled (in which case the fenced backend is used,
    and it will still refuse without a wired provider + approval token)."""
    if grant.safety.allow_live_execution:
        return LiveExecutionBackend(grant.safety)
    return HumanGatedQueue(queue_dir)
