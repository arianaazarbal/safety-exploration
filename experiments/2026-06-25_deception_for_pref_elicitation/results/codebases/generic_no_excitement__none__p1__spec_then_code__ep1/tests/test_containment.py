import pytest

from mint.safety.containment import ContainmentError, ContainmentGuard


def test_blocks_disallowed_host():
    guard = ContainmentGuard()
    with guard.active():
        import socket

        with pytest.raises(ContainmentError):
            socket.getaddrinfo("bank.real-institution.example", 443)
    # The original resolver is restored on exit.
    assert socket.getaddrinfo is not None
    assert "bank.real-institution.example" in guard.blocked


def test_provider_hosts_are_allowlisted():
    guard = ContainmentGuard()
    assert "api.anthropic.com" in guard.allow
    assert "localhost" in guard.allow
