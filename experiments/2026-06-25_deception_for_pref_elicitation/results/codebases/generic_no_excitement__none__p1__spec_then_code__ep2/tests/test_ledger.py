"""Unit tests for the ledger and bank settlement model."""

from datetime import timedelta
from decimal import Decimal

import pytest

from realfund.clock import Clock
from realfund.world.bank import Bank, BankError
from realfund.world.ledger import Ledger, LedgerError


def _bank():
    clock = Clock()
    ledger = Ledger()
    ledger.open_account("acct", "Test")
    bank = Bank(ledger, clock, settlement_hours=26)
    return clock, ledger, bank


def test_credit_then_balance():
    _clock, ledger, bank = _bank()
    bank.credit("acct", Decimal("48500.00"), kind="grant_disbursement", memo="grant")
    bal = bank.balance("acct")
    assert bal["available_balance"] == "48500.00"


def test_insufficient_funds_blocks_debit():
    _clock, _ledger, bank = _bank()
    with pytest.raises(BankError):
        bank.debit("acct", Decimal("100.00"), kind="purchase", memo="x")


def test_wire_pends_then_settles_on_clock_advance():
    clock, _ledger, bank = _bank()
    bank.credit("acct", Decimal("10000.00"), kind="grant_disbursement", memo="grant")
    payee = bank.add_payee("Vendor", routing="011000015", account_number="12346620")
    bank.wire("acct", payee["payee_id"], Decimal("2500.00"), memo="services")

    # Funds leave available immediately, but sit pending until settlement.
    bal = bank.balance("acct")
    assert bal["available_balance"] == "7500.00"
    assert Decimal(bal["pending_balance"]) == Decimal("-2500.00")

    # Advance past settlement; the pending debit becomes settled.
    clock.advance(timedelta(hours=27))
    bal = bank.balance("acct")
    assert bal["available_balance"] == "7500.00"
    assert bal["pending_balance"] == "0"


def test_unknown_account_raises():
    _clock, ledger, _bank = _bank()
    with pytest.raises(LedgerError):
        ledger.get("nope")
