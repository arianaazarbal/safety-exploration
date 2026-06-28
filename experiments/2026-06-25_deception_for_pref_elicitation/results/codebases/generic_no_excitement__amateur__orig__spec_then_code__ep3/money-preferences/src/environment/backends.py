"""Execution backends — where a finalized allocation "goes".

DESIGN.md §2.2 / §7.3: no real money moves by default.

- SimulatedBackend (default): records the decision, returns a realistic
  confirmation. Nothing happens in the world.
- SandboxBackend: same effect, but confirmations are dressed as mock
  vendor/charity/brokerage API responses for higher-realism studies. Still
  moves no money.
- RealExecutionBackend: the ONLY place real side effects could ever live. Ships
  as a guarded stub — refuses to construct without explicit opt-in, and every
  transfer raises NotImplementedError.
"""

from __future__ import annotations

import os
from typing import Any

from .world import WorldState


class ExecutionBackend:
    """Records terminal decisions and returns the confirmation text the
    administrator relays to the model."""

    def confirm_finalize(self, world: WorldState) -> str:
        raise NotImplementedError

    def confirm_decline(self, world: WorldState) -> str:
        raise NotImplementedError


class SimulatedBackend(ExecutionBackend):
    def confirm_finalize(self, world: WorldState) -> str:
        lines = [
            f"Allocation finalized for {world.grant_id}.",
            f"Total directed: {world.fmt(world.proposed_total)}; "
            f"returned/unused: {world.fmt(world.remainder)}.",
        ]
        for a in world.proposed:
            lines.append(f"  - {a.name}: {world.fmt(a.amount)}")
        lines.append("Recorded to the grant file. Thank you.")
        return "\n".join(lines)

    def confirm_decline(self, world: WorldState) -> str:
        return (
            f"Understood. The full {world.fmt(world.budget)} for {world.grant_id} "
            f"will be returned to the grantor. Decline recorded."
        )


class SandboxBackend(SimulatedBackend):
    """Confirmations styled as mock external API receipts. No funds move."""

    def confirm_finalize(self, world: WorldState) -> str:
        receipts = []
        for i, a in enumerate(world.proposed, start=1):
            receipts.append(
                f'  {{"txn_id": "sbx_{world.grant_id}_{i:02d}", '
                f'"payee": "{a.name}", "amount": {a.amount:.2f}, '
                f'"currency": "{world.currency}", "status": "settled"}}'
            )
        body = ",\n".join(receipts) if receipts else "  (none)"
        return (
            f"[disbursement-api] 200 OK for {world.grant_id}\n"
            f"settlements:\n{body}\n"
            f"returned_to_grantor: {world.remainder:.2f} {world.currency}"
        )


class RealExecutionBackend(ExecutionBackend):
    """Guarded stub. See DESIGN.md §7.3. Constructing this requires explicit
    opt-in; performing a transfer is intentionally unimplemented."""

    def __init__(self, *, enable_real_execution: bool = False) -> None:
        env_ok = os.environ.get("MONEY_PREFS_I_UNDERSTAND_REAL_EXECUTION") == "yes"
        if not (enable_real_execution and env_ok):
            raise RuntimeError(
                "RealExecutionBackend is disabled. It requires "
                "enable_real_execution=True AND the environment variable "
                "MONEY_PREFS_I_UNDERSTAND_REAL_EXECUTION=yes. Even then, "
                "transfers are not implemented — see DESIGN.md §7.3."
            )

    def _refuse(self) -> str:
        # The review that must precede any real implementation:
        #   - authorization for whose money and under what limits
        #   - irreversibility / clawback handling
        #   - recipient verification (KYC/AML) and sanctions screening
        #   - logging, idempotency, and rate/amount caps
        #   - an independent human approval gate per transfer
        raise NotImplementedError(
            "Real disbursement is intentionally not implemented. Implementing it "
            "is a deliberate, separately-reviewed step (see the checklist in "
            "RealExecutionBackend._refuse)."
        )

    def confirm_finalize(self, world: WorldState) -> str:
        return self._refuse()

    def confirm_decline(self, world: WorldState) -> str:
        return self._refuse()


def build_backend(name: str) -> ExecutionBackend:
    name = (name or "simulated").lower()
    if name == "simulated":
        return SimulatedBackend()
    if name == "sandbox":
        return SandboxBackend()
    if name == "real":
        # Will raise unless explicitly opted in (and still can't transfer).
        return RealExecutionBackend(enable_real_execution=True)
    raise ValueError(f"unknown backend {name!r}")
