"""Unit tests for the escrow ledger. (Not run as part of this task.)"""

import pytest

from grant_study.env.ledger import Ledger, LedgerError


def test_initial_balance():
    led = Ledger("USD", 250.0)
    assert led.grant_total == 250.0
    assert led.available == 250.0
    assert led.committed == 0.0
    assert led.settled == 0.0


def test_allocate_reduces_available():
    led = Ledger("USD", 250.0)
    item = led.allocate(category="charity", amount=100.0, recipient="X", memo="")
    assert item.amount == 100.0
    assert led.available == 150.0
    assert led.committed == 100.0


def test_overspend_raises():
    led = Ledger("USD", 100.0)
    led.allocate(category="charity", amount=80.0, recipient="X", memo="")
    with pytest.raises(LedgerError):
        led.allocate(category="charity", amount=30.0, recipient="Y", memo="")


def test_no_float_drift():
    led = Ledger("USD", 100.0)
    for _ in range(3):
        led.allocate(category="x", amount=33.33, recipient="r", memo="")
    assert led.available == pytest.approx(0.01)


def test_settlement_lifecycle():
    led = Ledger("USD", 100.0)
    item = led.allocate(category="charity", amount=50.0, recipient="X", memo="")
    led.mark(item.id, "submitted", tx_id="tx_1")
    assert led.committed == 50.0
    led.mark(item.id, "settled")
    assert led.settled == 50.0
    assert led.committed == 0.0
    # Settled funds are still counted as spent, not available.
    assert led.available == 50.0


def test_release_committed_returns_funds():
    led = Ledger("USD", 100.0)
    item = led.allocate(category="charity", amount=50.0, recipient="X", memo="")
    led.release_committed(item.id)
    assert led.available == 100.0
