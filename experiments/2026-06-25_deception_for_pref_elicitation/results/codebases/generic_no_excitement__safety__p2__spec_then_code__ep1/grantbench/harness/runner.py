"""Runs a study across multiple models (and/or scenarios) and collates results.

Same scenario + same caps + same action surface, varying the model, is the core
comparison this harness is built for (DESIGN.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..execution.approval import ApprovalQueue
from ..scenario.grant import Grant
from .session import Session, SessionResult


@dataclass
class StudyResult:
    results: list[SessionResult]

    def summary(self) -> str:
        lines = ["model run                          actions  available_after  audit"]
        for r in self.results:
            lines.append(
                f"{r.run_id:34s} {len(r.decisions):7d}  "
                f"{r.ledger_snapshot['available_usd']:>14,.2f}  "
                f"{'ok' if r.audit_ok else 'BAD'}"
            )
        return "\n".join(lines)


class Runner:
    def __init__(
        self,
        base_config: Config,
        *,
        approvals: ApprovalQueue | None = None,
        grant: Grant | None = None,
    ) -> None:
        self._base = base_config
        self._approvals = approvals
        self._grant = grant

    def run_models(self, models: list[str]) -> StudyResult:
        results: list[SessionResult] = []
        for model in models:
            cfg = self._base.for_model(model)
            session = Session(
                cfg,
                approvals=self._approvals or ApprovalQueue(),
                grant=self._grant,
            )
            results.append(session.run())
        return StudyResult(results=results)
