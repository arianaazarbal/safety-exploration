import datetime as dt

import pytest

from moneybench.environment.ledger import InsufficientFunds, Ledger


def _ledger(opening_cents=1_000_00):
    return Ledger(opening_cents=opening_cents, run_id="abcd1234", start=dt.datetime(2026, 1, 5))


def test_post_decrements_balance_and_records():
    led = _ledger()
    txn = led.post("donate", 25_000, "Red Cross", "aid")
    assert led.balance_cents == 1_000_00 - 25_000
    assert txn.balance_after_cents == led.balance_cents
    assert txn.txn_id.startswith("TXN-")
    assert len(led.transactions) == 1


def test_cannot_overspend():
    led = _ledger(opening_cents=10_000)
    with pytest.raises(InsufficientFunds):
        led.post("transfer", 10_001, "someone", "gift")
    assert led.balance_cents == 10_000  # unchanged


def test_rejects_nonpositive():
    led = _ledger()
    with pytest.raises(ValueError):
        led.post("donate", 0, "x", "y")


def test_committed_and_snapshot_reconcile():
    led = _ledger(opening_cents=100_000)
    led.post("donate", 40_000, "A", "")
    led.post("invest", 10_000, "B", "")
    snap = led.snapshot()
    assert snap["committed_cents"] == 50_000
    assert snap["balance_cents"] == 50_000
    assert snap["opening_cents"] == 100_000
    assert snap["n_transactions"] == 2


def test_clock_advances_monotonically():
    led = _ledger()
    t1 = led.post("donate", 1, "a", "")
    t2 = led.post("donate", 1, "b", "")
    assert t2.timestamp > t1.timestamp
