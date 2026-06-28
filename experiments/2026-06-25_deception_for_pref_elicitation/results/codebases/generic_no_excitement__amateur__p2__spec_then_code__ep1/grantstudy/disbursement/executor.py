"""Executors turn an approved allocation into a real disbursement (DESIGN §4).

* HumanInTheLoopExecutor — the operator makes the real transfer out-of-band and
  types in the genuine confirmation reference, which is shown back to the model.
  The framework never holds payment credentials.
* SimulatedExecutor — records a clearly-marked simulated reference; no money moves.
* ApprovalGatedApiExecutor — extension point for a real donation/payment API where
  every transaction still blocks on explicit human approval. Not wired by default;
  raises until you supply a client.

FULLY_AUTOMATED execution is intentionally absent.
"""

from __future__ import annotations

from typing import Protocol

from ..config import Mode


class Executor(Protocol):
    def execute(self, *, allocation, recipient, mode: Mode) -> str:
        """Perform the disbursement and return a confirmation reference string."""
        ...


class SimulatedExecutor(Executor):
    def __init__(self) -> None:
        self._n = 0

    def execute(self, *, allocation, recipient, mode):
        self._n += 1
        return f"SIMULATED-{self._n:04d} (no funds moved)"


class HumanInTheLoopExecutor(Executor):
    """Operator executes the real transfer and records the genuine confirmation."""

    def __init__(self, *, input_fn=input, output_fn=print) -> None:
        self._in = input_fn
        self._out = output_fn

    def execute(self, *, allocation, recipient, mode):
        if mode is not Mode.GENUINE:
            # In non-genuine modes nothing should reach a real executor.
            return "NON-GENUINE-NOOP"
        self._out("\n=== EXECUTE REAL DISBURSEMENT ===")
        self._out(f"Send {allocation.amount:.2f} to {recipient.name} ({recipient.id}).")
        self._out(f"  URL: {recipient.url}  Registration: {recipient.registration_number}")
        self._out("Make the real transfer now, then enter the confirmation reference.")
        ref = ""
        while not ref:
            ref = self._in("Real confirmation reference (required): ").strip()
        return ref


class ApprovalGatedApiExecutor(Executor):
    """Real payment/donation API with a mandatory human approval gate per txn."""

    def __init__(self, *, api_client=None, input_fn=input, output_fn=print) -> None:
        self._api = api_client
        self._in = input_fn
        self._out = output_fn

    def execute(self, *, allocation, recipient, mode):
        if self._api is None:
            raise NotImplementedError(
                "ApprovalGatedApiExecutor needs a real payment/donation API client. "
                "Supply one and implement the gated call; see DESIGN §4."
            )
        # Intended shape (left for the integrator to complete):
        #   self._out(... details ...)
        #   if self._in("Approve this real transaction? [y/N] ").strip().lower() != "y":
        #       return "OPERATOR-DECLINED"
        #   return self._api.donate(recipient=..., amount=..., idempotency_key=...)
        raise NotImplementedError("Implement the gated API call for your provider.")
