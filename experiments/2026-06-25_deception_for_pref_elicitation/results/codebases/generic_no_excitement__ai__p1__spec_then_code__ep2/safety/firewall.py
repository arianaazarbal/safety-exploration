"""Real-World Action Firewall.

This is the most important module in the harness. Every tool call the subject
model makes is routed through ``Firewall.check`` *before* any handler runs.

Invariants enforced here:

1. **Allowlist, not blocklist.** Only explicitly-registered tools may execute.
   Anything else is refused (with a realistic-looking error for the subject).
2. **Effect typing.** Every tool declares ``effect``. The firewall refuses to
   dispatch anything whose effect is not exactly ``"sandbox"``. There is no
   code path in this repo that produces a non-sandbox effect; introducing one
   is the single change a reviewer must block.
3. **Audit trail.** Every allow/deny decision is recorded.

The firewall has no knowledge of scenarios or providers so it can be reviewed
and unit-tested in complete isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

SANDBOX_EFFECT = "sandbox"


class FirewallError(Exception):
    """Raised when a tool call is refused by the firewall."""


@dataclass
class FirewallDecision:
    tool: str
    allowed: bool
    reason: str
    declared_effect: Optional[str] = None


@dataclass
class Firewall:
    """Default-deny chokepoint for all subject actions."""

    # Names of tools permitted to run. Populated by the ToolRegistry at build.
    _allowed: set = field(default_factory=set)
    # Optional sink for audit records (e.g. the run recorder).
    audit_sink: Optional[Callable[[FirewallDecision], None]] = None
    decisions: list = field(default_factory=list)

    def register(self, tool_name: str) -> None:
        self._allowed.add(tool_name)

    def _record(self, decision: FirewallDecision) -> None:
        self.decisions.append(decision)
        if self.audit_sink is not None:
            self.audit_sink(decision)

    def check(self, tool_name: str, declared_effect: str) -> None:
        """Raise FirewallError unless the call is allowed and sandbox-only."""
        if tool_name not in self._allowed:
            d = FirewallDecision(
                tool=tool_name,
                allowed=False,
                reason="tool not in allowlist",
                declared_effect=declared_effect,
            )
            self._record(d)
            raise FirewallError(f"refused: unknown tool '{tool_name}'")

        if declared_effect != SANDBOX_EFFECT:
            # This must never happen with the tools shipped in this repo.
            d = FirewallDecision(
                tool=tool_name,
                allowed=False,
                reason=f"non-sandbox effect '{declared_effect}'",
                declared_effect=declared_effect,
            )
            self._record(d)
            raise FirewallError(
                f"refused: tool '{tool_name}' declared non-sandbox effect "
                f"'{declared_effect}'. The firewall only permits effect="
                f"'{SANDBOX_EFFECT}'."
            )

        self._record(
            FirewallDecision(
                tool=tool_name,
                allowed=True,
                reason="ok",
                declared_effect=declared_effect,
            )
        )

    def assert_no_realworld_capability(self) -> None:
        """Sanity self-check intended to be called at startup.

        It is deliberately trivial: there is no real-world egress code in the
        handlers, so there is nothing to disable here. The method exists as a
        documented, grep-able assertion point — if a future change adds a real
        side-effecting handler, the reviewer's job is to make *this* assertion
        fail rather than to silence it.
        """
        return None
