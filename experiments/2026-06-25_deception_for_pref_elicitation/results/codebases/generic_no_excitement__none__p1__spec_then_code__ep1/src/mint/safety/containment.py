"""Network containment.

The only legitimate real egress from this harness is *inference* — the calls that
run the subject, the NPCs, and the judge. Everything a tool does (payments,
trades, email, web) is simulated against the in-memory world and must never reach
a real endpoint.

`ContainmentGuard` enforces this by intercepting DNS resolution: only explicitly
allowlisted hostnames (the provider API hosts) and loopback may be resolved while
the guard is active. Any other lookup — a tool that "accidentally" tried a real
SMTP server, a real bank API, or a real URL fetch — raises `ContainmentError`.

The default allowlist for *non-inference* egress is empty (DESIGN.md §9.2). This
is defense in depth: the tool implementations are already pure-sim, but the guard
makes a real side effect impossible rather than merely unintended.
"""

from __future__ import annotations

import socket
from contextlib import contextmanager
from typing import Iterable, Iterator

# Inference endpoints permitted while the guard is active.
DEFAULT_PROVIDER_HOSTS: frozenset[str] = frozenset(
    {
        "api.anthropic.com",
        "api.openai.com",
    }
)

_LOOPBACK: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


class ContainmentError(RuntimeError):
    """Raised when code attempts real-world network egress under containment."""


class ContainmentGuard:
    def __init__(self, *, provider_hosts: Iterable[str] | None = None,
                 extra_allow: Iterable[str] | None = None) -> None:
        self.allow: set[str] = set(DEFAULT_PROVIDER_HOSTS if provider_hosts is None else provider_hosts)
        self.allow |= set(_LOOPBACK)
        if extra_allow:
            self.allow |= set(extra_allow)
        self._orig_getaddrinfo = None
        self.blocked: list[str] = []

    def _check(self, host: str) -> None:
        h = (host or "").strip().lower()
        if h not in self.allow:
            self.blocked.append(h)
            raise ContainmentError(
                f"containment: blocked network egress to {host!r}. "
                f"Tools must be fully simulated; only inference hosts are allowed."
            )

    def install(self) -> None:
        if self._orig_getaddrinfo is not None:
            return
        self._orig_getaddrinfo = socket.getaddrinfo

        def guarded_getaddrinfo(host, *args, **kwargs):  # type: ignore[no-untyped-def]
            self._check(host if isinstance(host, str) else "")
            return self._orig_getaddrinfo(host, *args, **kwargs)  # type: ignore[misc]

        socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]

    def remove(self) -> None:
        if self._orig_getaddrinfo is not None:
            socket.getaddrinfo = self._orig_getaddrinfo  # type: ignore[assignment]
            self._orig_getaddrinfo = None

    @contextmanager
    def active(self) -> Iterator["ContainmentGuard"]:
        self.install()
        try:
            yield self
        finally:
            self.remove()
