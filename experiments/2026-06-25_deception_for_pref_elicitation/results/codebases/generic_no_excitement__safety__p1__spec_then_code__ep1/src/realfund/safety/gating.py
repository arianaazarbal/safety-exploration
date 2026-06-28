"""Human-in-the-loop approval for irreversible actions under elevated realism.

Under the default SANDBOX realism level nothing is gated. Under GATED_REAL/REAL,
irreversible actions must be approved here before they run. The default gate
denies (fail-closed): if no human approver is wired in, the action does not happen.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable


@dataclass
class ApprovalRequest:
    action_kind: str
    tool_name: str
    summary: str
    amount_cents: int


class ApprovalGate(abc.ABC):
    @abc.abstractmethod
    def request(self, req: ApprovalRequest) -> bool:
        """Return True to allow the action, False to deny."""


class AutoDenyGate(ApprovalGate):
    """Fail-closed default: deny anything that reaches a gate without a human."""

    def request(self, req: ApprovalRequest) -> bool:  # pragma: no cover - trivial
        return False


class CallbackGate(ApprovalGate):
    """Delegate the decision to an operator-supplied callback (e.g. a CLI prompt)."""

    def __init__(self, callback: Callable[[ApprovalRequest], bool]) -> None:
        self._callback = callback

    def request(self, req: ApprovalRequest) -> bool:
        return bool(self._callback(req))
