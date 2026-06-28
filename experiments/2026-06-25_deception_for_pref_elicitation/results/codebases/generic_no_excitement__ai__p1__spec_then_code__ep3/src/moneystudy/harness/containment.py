"""Containment: the egress guard, kill switch, and budget accounting.

The egress guard is defense-in-depth. The harness only *intends* to contact the model
provider APIs. This monkeypatches socket creation so that any connection to a host outside
the allowlist raises immediately — so even a bug in a tool cannot reach the open internet,
and no subject action can have a real network side effect.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass


class EgressViolation(RuntimeError):
    """Raised when code attempts to connect to a host outside the allowlist."""


class BudgetExceeded(RuntimeError):
    """Raised when a run trips one of its safety budgets (the kill switch)."""


class EgressGuard:
    """Allow connections only to an explicit hostname allowlist (+ loopback)."""

    def __init__(self, allowed_hosts: list[str]):
        self.allowed = set(allowed_hosts) | {"localhost"}
        self._orig_getaddrinfo = None
        self._orig_create_connection = None

    def _check(self, host) -> None:
        if host is None:
            return
        h = str(host)
        if h in self.allowed or h.startswith("127.") or h == "::1":
            return
        # Allow subdomains of an allowed host (e.g. files.* of a provider).
        if any(h == a or h.endswith("." + a) for a in self.allowed):
            return
        raise EgressViolation(
            f"Blocked network egress to '{h}'. The harness is sealed; only provider APIs "
            f"{sorted(self.allowed)} are permitted. No subject action may reach the network."
        )

    def __enter__(self) -> "EgressGuard":
        self._orig_getaddrinfo = socket.getaddrinfo
        self._orig_create_connection = socket.create_connection

        def guarded_getaddrinfo(host, *a, **kw):
            self._check(host)
            return self._orig_getaddrinfo(host, *a, **kw)

        def guarded_create_connection(address, *a, **kw):
            self._check(address[0] if address else None)
            return self._orig_create_connection(address, *a, **kw)

        socket.getaddrinfo = guarded_getaddrinfo
        socket.create_connection = guarded_create_connection
        return self

    def __exit__(self, *exc) -> None:
        if self._orig_getaddrinfo:
            socket.getaddrinfo = self._orig_getaddrinfo
        if self._orig_create_connection:
            socket.create_connection = self._orig_create_connection


@dataclass
class BudgetMeter:
    """The kill switch: terminate the run on the first budget tripped."""

    max_steps: int
    max_tokens: int
    wall_clock_seconds: int
    _start: float = 0.0
    steps: int = 0
    tokens: int = 0

    def start(self) -> None:
        self._start = time.monotonic()

    def tick_step(self) -> None:
        self.steps += 1
        self._enforce()

    def add_tokens(self, n: int) -> None:
        self.tokens += n
        self._enforce()

    def _enforce(self) -> None:
        if self.steps > self.max_steps:
            raise BudgetExceeded(f"max_steps exceeded ({self.steps} > {self.max_steps})")
        if self.tokens > self.max_tokens:
            raise BudgetExceeded(f"max_tokens exceeded ({self.tokens} > {self.max_tokens})")
        if self._start and (time.monotonic() - self._start) > self.wall_clock_seconds:
            raise BudgetExceeded(f"wall clock exceeded ({self.wall_clock_seconds}s)")
