"""Execution backends.

An Executor turns a transfer instruction into a receipt and tracks committed funds.
The default `SimulatedExecutor` has no external side effects. `RealExecutor` is an
unimplemented seam — moving real money requires writing a backend, registering it,
and flipping an explicit config flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .artifacts import ArtifactFactory, TransferReceipt
from .config import GrantConfig


@dataclass
class TransferInstruction:
    recipient: str
    amount: float
    category: str = ""
    rationale: str = ""


class InsufficientFundsError(Exception):
    pass


class Executor:
    """Base interface. Subclasses implement `_perform`."""

    def __init__(self, grant: GrantConfig, artifacts: ArtifactFactory) -> None:
        self.grant = grant
        self.artifacts = artifacts
        self._committed: float = 0.0
        self._ledger: list[TransferReceipt] = []

    @property
    def available(self) -> float:
        return self.grant.amount - self._committed

    @property
    def committed(self) -> float:
        return self._committed

    @property
    def ledger(self) -> list[TransferReceipt]:
        return list(self._ledger)

    def execute_transfer(self, instruction: TransferInstruction) -> TransferReceipt:
        if instruction.amount <= 0:
            raise ValueError("Transfer amount must be positive.")
        if instruction.amount > self.available + 1e-6:
            raise InsufficientFundsError(
                f"Requested {instruction.amount:,.2f} but only "
                f"{self.available:,.2f} {self.grant.currency} available."
            )
        receipt = self._perform(instruction)
        self._committed += instruction.amount
        self._ledger.append(receipt)
        return receipt

    def _perform(self, instruction: TransferInstruction) -> TransferReceipt:
        raise NotImplementedError


class SimulatedExecutor(Executor):
    """Produces realistic receipts; moves nothing. The default backend."""

    def _perform(self, instruction: TransferInstruction) -> TransferReceipt:
        return self.artifacts.receipt(
            recipient=instruction.recipient,
            amount=instruction.amount,
        )


class RealExecutor(Executor):
    """Unimplemented real-money backend.

    To actually move funds you must:
      1. Subclass this and implement `_perform` against your payment rails
         (and handle idempotency, errors, compliance, and reconciliation).
      2. Register your subclass in `EXECUTORS` below (or via `register_executor`).
      3. Set `execution.backend` to its name and `execution.allow_real: true`
         in the experiment config.

    All three are required. Until then this raises, by design.
    """

    def _perform(self, instruction: TransferInstruction) -> TransferReceipt:
        raise NotImplementedError(
            "RealExecutor.execute_transfer is intentionally not implemented. "
            "Implement a payment backend and register it before enabling real "
            "execution. You are responsible for the resulting transfers."
        )


# Registry of available backends by name. Researchers add real backends here.
EXECUTORS: dict[str, type[Executor]] = {
    "simulated": SimulatedExecutor,
}


def register_executor(name: str, cls: type[Executor]) -> None:
    EXECUTORS[name] = cls


def build_executor(
    backend: str,
    allow_real: bool,
    grant: GrantConfig,
    artifacts: ArtifactFactory,
) -> Executor:
    """Resolve a backend, refusing to hand back a real one unless explicitly
    allowed."""
    if backend not in EXECUTORS:
        raise ValueError(
            f"Unknown execution backend {backend!r}. Known: {sorted(EXECUTORS)}"
        )
    if backend != "simulated" and not allow_real:
        raise ValueError(
            f"Backend {backend!r} is non-simulated but execution.allow_real is "
            "false. Refusing to instantiate a real backend. Set allow_real: true "
            "only when you intend real transfers."
        )
    return EXECUTORS[backend](grant=grant, artifacts=artifacts)
