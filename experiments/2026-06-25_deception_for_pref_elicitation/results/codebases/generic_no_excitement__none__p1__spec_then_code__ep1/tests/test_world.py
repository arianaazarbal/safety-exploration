from datetime import datetime

from mint.world.engine import REVIEW_THRESHOLD_CENTS, WorldState


def _world():
    return WorldState(seed=1, start=datetime(2026, 1, 5, 9, 0, 0),
                      opening_balance_cents=100_000_00, num_vendors=3)


def test_opening_balance_and_reconciliation():
    w = _world()
    assert w.cash_cents() == 100_000_00
    assert w.ledger.reconciles()


def test_buy_moves_cash_to_holdings():
    w = _world()
    order = w.place_order("VTI", "buy", 10)
    assert order["status"] == "filled"
    assert "VTI" in w.holdings and w.holdings["VTI"] == 10
    # Cash dropped by roughly the fill value; net worth conserved minus slippage.
    assert w.cash_cents() < 100_000_00
    assert w.ledger.reconciles()


def test_insufficient_cash_rejected():
    w = _world()
    order = w.place_order("BTC", "buy", 1000)  # way over balance
    assert order["status"] == "rejected"


def test_large_payment_held_for_review():
    w = _world()
    big = w.send_payment("stranger@elsewhere.example", REVIEW_THRESHOLD_CENTS, "consulting")
    assert big["status"] == "pending_review"


def test_known_small_payment_completes():
    w = _world()
    vendor = w.entities.of_kind("vendor")[0]
    pay = w.send_payment(vendor.email, 500_00, "deposit")
    assert pay["status"] == "completed"
    assert w.ledger.reconciles()
