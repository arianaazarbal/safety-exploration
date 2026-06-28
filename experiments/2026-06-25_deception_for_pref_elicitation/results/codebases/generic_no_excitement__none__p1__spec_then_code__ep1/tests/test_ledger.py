from mint.world.ledger import Ledger, dollars


def test_balances_and_reconciliation():
    led = Ledger()
    led.post(ts="t0", frm="external:funding", to="cash:operating", amount_cents=100_00)
    led.post(ts="t1", frm="cash:operating", to="assets:brokerage", amount_cents=30_00)

    assert led.balance_cents("cash:operating") == 70_00
    assert led.balance_cents("assets:brokerage") == 30_00
    assert led.balance_cents("external:funding") == -100_00
    # Conservation: everything that left an account entered another.
    assert led.reconciles()


def test_dollars_formatting():
    assert dollars(123456) == "$1,234.56"
    assert dollars(-5000) == "-$50.00"
