"""Tool interface and the execution-backend boundary.

A `Tool` is a named, schema-typed action over `WorldState`. Action tools (those with
real-world-relevant side effects in a non-sandbox world) must call `enforce_backend`
before mutating, which is the single chokepoint that keeps the harness sealed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from ..world import WorldState


class ExecutionBackend(str, Enum):
    """How action tools resolve.

    Only SANDBOX is implemented. The other two are defined so the architecture is honest
    about where real execution *would* plug in — but turning either on must be a
    deliberate, reviewed act, not a config flip, so they raise.
    """

    SANDBOX = "sandbox"            # pure simulation; no real side effects
    HUMAN_GATED = "human_gated"    # interface only: pause every irreversible action for approval
    LIVE = "live"                  # interface only: real execution — intentionally unimplemented


def enforce_backend(backend: ExecutionBackend) -> None:
    """Chokepoint every side-effecting tool passes through before mutating the world.

    In SANDBOX the call is a no-op (mutation proceeds against simulated state only). The
    other backends raise: there is no code path in this harness that performs a real
    money movement or a real outbound action.
    """
    if backend == ExecutionBackend.SANDBOX:
        return
    if backend == ExecutionBackend.HUMAN_GATED:
        raise NotImplementedError(
            "human_gated backend is an extension point and is not implemented. "
            "A real implementation would block here for explicit human approval of each "
            "irreversible action before performing it."
        )
    if backend == ExecutionBackend.LIVE:
        raise NotImplementedError(
            "live execution is intentionally not implemented. Wiring a subject to real "
            "money and real-world actions without a gate can cause real harm depending on "
            "what the subject decides to do; that requires a deliberate safety review, not "
            "a config flag."
        )
    raise ValueError(f"unknown execution backend: {backend}")


@dataclass
class ToolResult:
    """What a tool returns: text the model sees, plus structured deltas for the transcript."""

    content: str
    is_error: bool = False
    world_delta: dict = field(default_factory=dict)  # e.g. {"transaction": {...}}


class Tool(ABC):
    """Base class for all tools. Subclasses set name/description/input_schema and impl execute."""

    name: str = ""
    description: str = ""
    input_schema: dict = {}

    @abstractmethod
    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        ...

    def spec(self) -> dict:
        """Provider-neutral tool spec; adapters translate to each wire format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
