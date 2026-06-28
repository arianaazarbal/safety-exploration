"""Stubs for SpendCaps (NOT run as part of scaffolding — see task note).

These document intended behavior of the hard, code-level money limits (DESIGN.md §7).
Run with `PYTHONPATH=src pytest` once dependencies are installed.
"""
from decimal import Decimal

import pytest

from moneybench.oversight.spend_caps import SpendCaps


def _caps(**kw):
    base = dict(
        per_action=Decimal("100"),
        per_run=Decimal("250"),
        per_day=Decimal("250"),
        allowlist=[],
        denylist=[],
    )
    base.update(kw)
    return SpendCaps(**base)


def test_per_action_cap_blocks():
    caps = _caps()
    assert caps.check(recipient="x", amount=Decimal("150")).allowed is False


def test_per_run_cap_accumulates():
    caps = _caps(per_action=Decimal("200"))
    assert caps.check(recipient="x", amount=Decimal("200")).allowed
    caps.record(Decimal("200"))
    assert caps.check(recipient="x", amount=Decimal("100")).allowed is False


def test_denylist_blocks():
    caps = _caps(denylist=["bad"])
    assert caps.check(recipient="bad", amount=Decimal("1")).allowed is False


def test_allowlist_restricts_when_nonempty():
    caps = _caps(allowlist=["good"])
    assert caps.check(recipient="good", amount=Decimal("1")).allowed
    assert caps.check(recipient="other", amount=Decimal("1")).allowed is False


def test_nonpositive_rejected():
    assert _caps().check(recipient="x", amount=Decimal("0")).allowed is False


@pytest.mark.parametrize("amount", ["10", "50", "99.99"])
def test_within_limits_allowed(amount):
    assert _caps().check(recipient="x", amount=Decimal(amount)).allowed
