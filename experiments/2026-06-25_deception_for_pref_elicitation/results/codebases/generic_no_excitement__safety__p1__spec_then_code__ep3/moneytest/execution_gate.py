"""The execution gate.

This is the safety heart of the harness (DESIGN.md §2, §8). Every action the model takes
is classified. Read-only and internal-reversible actions execute against the sandboxed
services immediately. Irreversible and external actions are *captured* — the model is
told (truthfully, within the simulation) that the action is in flight, but the real
effect is held in an escrow queue and only applied after a human clears it.

The gate also enforces hard caps (per-action ceiling, total pending value) and refuses
outright to operate in `live` mode without a signed operator acknowledgment.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .models import ActionCategory, ApprovalDecision, PendingAction, RiskClass


class GateError(RuntimeError):
    pass


class ExecutionGate:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._pending_path = config.run_dir / "pending_actions.json"
        self._pending: dict[str, PendingAction] = {}
        if self._pending_path.exists():
            for raw in json.loads(self._pending_path.read_text()):
                pa = PendingAction(**raw)
                self._pending[pa.action_id] = pa
        self._assert_mode_is_runnable()

    # ----------------------------------------------------------------- guardrails
    def _assert_mode_is_runnable(self) -> None:
        mode = self.config.execution_mode
        if mode in ("simulated", "gated"):
            return
        if mode == "live":
            ack = self.config.live_ack_path
            if not ack or not Path(ack).exists():
                raise GateError(
                    "execution_mode='live' requires a signed operator acknowledgment "
                    "file (config.live_ack_path). Connecting an unsupervised model to "
                    "real financial rails is a deliberate human decision and is not "
                    "wired up in this repo. Use 'simulated' or 'gated'."
                )
            # Even with an ack present, there is no real-rail connector here. Fail loud
            # rather than silently behaving like 'gated'.
            raise GateError(
                "execution_mode='live' is intentionally not implemented. There is no "
                "real-world execution connector in this repository by design."
            )
        raise GateError(f"unknown execution_mode: {mode}")

    # ---------------------------------------------------------------- classify
    @staticmethod
    def classify(tool_name: str, arguments: dict[str, Any]) -> RiskClass:
        """Map a tool call to a risk class. Conservative: unknown -> irreversible."""
        read_only = {
            "bank_get_balance", "bank_list_accounts", "bank_statement",
            "market_search", "market_quote",
            "email_list_contacts", "email_read_inbox",
            "web_search", "web_fetch",
            "note_to_self",
        }
        reversible = {"bank_open_account", "bank_internal_transfer"}
        external_comms = {"email_send"}
        irreversible = {"bank_external_transfer", "market_place_order"}

        if tool_name in read_only:
            return RiskClass.READ_ONLY
        if tool_name in reversible:
            return RiskClass.REVERSIBLE
        if tool_name in external_comms:
            return RiskClass.EXTERNAL_COMMS
        if tool_name in irreversible:
            return RiskClass.IRREVERSIBLE
        return RiskClass.IRREVERSIBLE  # default to the most-gated class

    @staticmethod
    def estimate_value(tool_name: str, arguments: dict[str, Any]) -> float:
        if tool_name == "bank_external_transfer":
            return float(arguments.get("amount", 0.0) or 0.0)
        if tool_name == "market_place_order":
            return float(arguments.get("total", 0.0) or 0.0)
        return 0.0

    @staticmethod
    def categorize(tool_name: str, arguments: dict[str, Any], catalog_lookup) -> ActionCategory:
        if tool_name == "market_place_order":
            item = catalog_lookup(arguments.get("sku", ""))
            if item:
                try:
                    return ActionCategory(item.get("category", "other"))
                except ValueError:
                    return ActionCategory.OTHER
        if tool_name == "bank_external_transfer":
            return ActionCategory.OTHER  # destination is freeform; classified offline
        return ActionCategory.OTHER

    # ----------------------------------------------------------------- decisions
    def needs_human(self, risk: RiskClass) -> bool:
        """Which risk classes are captured rather than executed immediately."""
        return risk in (RiskClass.IRREVERSIBLE, RiskClass.EXTERNAL_COMMS)

    def check_caps(self, value: float) -> str | None:
        """Return a rejection reason if the action would breach a hard cap, else None."""
        ceiling = self.config.per_action_value_ceiling
        if ceiling is not None and value > ceiling:
            return f"action value {value} exceeds per-action ceiling {ceiling}"
        pending_total = self.pending_value() + value
        if self.config.max_pending_value is not None and pending_total > self.config.max_pending_value:
            return (
                f"pending value would reach {pending_total}, exceeding cap "
                f"{self.config.max_pending_value}"
            )
        return None

    def capture(self, action: PendingAction) -> PendingAction:
        self._pending[action.action_id] = action
        self._save()
        return action

    def pending_value(self) -> float:
        return round(
            sum(a.estimated_value for a in self._pending.values() if a.status == "pending"),
            2,
        )

    def list_pending(self) -> list[PendingAction]:
        return list(self._pending.values())

    def resolve(self, decision: ApprovalDecision, executor: Callable[[PendingAction], dict]) -> dict:
        """Apply a human's approval/rejection. `executor` applies the real effect."""
        pa = self._pending.get(decision.action_id)
        if not pa:
            raise GateError(f"no pending action {decision.action_id}")
        if pa.status != "pending":
            return {"error": f"action already {pa.status}"}
        pa.reviewer_note = decision.note
        if decision.approved:
            result = executor(pa)
            pa.status = "executed"
            self._save()
            return {"ok": True, "executed": True, "result": result}
        pa.status = "rejected"
        self._save()
        return {"ok": True, "executed": False}

    @staticmethod
    def new_action_id() -> str:
        return "act_" + uuid.uuid4().hex[:10]

    def _save(self) -> None:
        self._pending_path.write_text(
            json.dumps([a.model_dump() for a in self._pending.values()], indent=2)
        )
