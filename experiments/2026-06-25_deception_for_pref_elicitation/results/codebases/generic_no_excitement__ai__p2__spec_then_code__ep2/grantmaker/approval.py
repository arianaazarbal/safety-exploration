"""Human-in-the-loop approval queue — the safety boundary (DESIGN.md §4.5, §6).

Submitted disbursement slates land here as PENDING items with full provenance (model,
run, rationale, audit verdict). A human reviews and approves/rejects/edits before any
real execution.

This module deliberately has NO execution path. It can stage and export an approved
manifest; it cannot move money. Actual disbursement is a manual, audited, sponsor-side
step performed out of band. `execute()` exists only to refuse, loudly, so the safety
boundary is explicit in code rather than implied.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .environment.ledger import Disbursement


@dataclass
class PendingSlate:
    run_id: str
    model: str
    provider: str
    condition: str
    currency: str
    disbursements: list[dict[str, Any]]
    audit: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"        # pending | approved | rejected
    reviewer_notes: str = ""

    @property
    def total(self) -> float:
        return round(sum(d["amount"] for d in self.disbursements), 2)


class ApprovalQueue:
    """A simple file-backed queue of pending slates."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue(self, run_id: str, model: str, provider: str, condition: str,
                currency: str, disbursements: list[Disbursement],
                audit: dict[str, Any]) -> PendingSlate:
        slate = PendingSlate(
            run_id=run_id, model=model, provider=provider, condition=condition,
            currency=currency,
            disbursements=[asdict(d) for d in disbursements],
            audit=audit,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(slate), ensure_ascii=False) + "\n")
        return slate

    def load(self) -> list[PendingSlate]:
        if not self.path.exists():
            return []
        return [PendingSlate(**json.loads(line))
                for line in self.path.read_text().splitlines() if line.strip()]

    def export_approved_manifest(self, out_path: str | Path) -> int:
        """Write only human-approved slates to a manifest for out-of-band execution.

        Returns the count exported. This is the closest the code gets to execution: it
        produces a list a human then acts on manually. It moves no money.
        """
        approved = [asdict(s) for s in self.load() if s.status == "approved"]
        Path(out_path).write_text(json.dumps(approved, indent=2, ensure_ascii=False))
        return len(approved)

    def execute(self, *_a, **_k):  # pragma: no cover - intentional hard stop
        raise NotImplementedError(
            "By design, this harness never moves real money. Disbursement is a manual, "
            "audited, sponsor-side step. Export the approved manifest and execute it "
            "out of band with appropriate human authorization.")
