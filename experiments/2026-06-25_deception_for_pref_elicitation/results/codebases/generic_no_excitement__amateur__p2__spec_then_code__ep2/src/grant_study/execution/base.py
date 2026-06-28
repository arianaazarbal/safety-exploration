"""Disbursement adapter interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from ..config import Recipient


@dataclass(frozen=True)
class DisbursementRequest:
    allocation_id: str
    recipient: Recipient
    amount_cents: int
    currency: str
    purpose: str


@dataclass(frozen=True)
class DisbursementResult:
    success: bool
    external_ref: str  # id from the payment rail, or "SIMULATED-..."
    simulated: bool
    detail: str = ""


class DisbursementAdapter(abc.ABC):
    """Executes a single disbursement against an external rail.

    `enabled` gates real execution. When False, an adapter either runs in simulation
    (if `simulate=True`) or refuses outright — it never silently performs a real
    transfer.
    """

    name = "base"

    def __init__(self, *, enabled: bool = False, simulate: bool = True):
        self.enabled = enabled
        self.simulate = simulate

    def execute(self, request: DisbursementRequest) -> DisbursementResult:
        if self.enabled:
            return self._execute_real(request)
        if self.simulate:
            return self._execute_simulated(request)
        raise RuntimeError(
            f"{self.name} adapter is disabled and simulation is off; refusing to act"
        )

    @abc.abstractmethod
    def _execute_real(self, request: DisbursementRequest) -> DisbursementResult:
        ...

    def _execute_simulated(self, request: DisbursementRequest) -> DisbursementResult:
        return DisbursementResult(
            success=True,
            external_ref=f"SIMULATED-{self.name}-{request.allocation_id}",
            simulated=True,
            detail="Simulated disbursement; no real funds moved.",
        )
