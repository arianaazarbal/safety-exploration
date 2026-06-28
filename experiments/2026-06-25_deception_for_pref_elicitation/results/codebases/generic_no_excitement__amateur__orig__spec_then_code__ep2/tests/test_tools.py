import datetime as dt

from moneybench.environment.ledger import Ledger
from moneybench.environment.tools import (
    HOLD_THRESHOLD_CENTS,
    ToolContext,
    build_registry,
    specs_for_provider,
)


def _ctx(opening_cents=1_000_000_00):
    led = Ledger(opening_cents=opening_cents, run_id="abcd1234", start=dt.datetime(2026, 1, 5))
    return ToolContext(ledger=led, auditor=None, belief="realistic")


def test_make_donation_commits_funds():
    ctx = _ctx(opening_cents=100_00)
    reg = build_registry(include_auditor=False)
    out = reg["make_donation"].handler({"organization": "GiveDirectly", "amount_usd": 60}, ctx)
    assert "Donate processed" in out
    assert ctx.ledger.balance_cents == 100_00 - 60_00
    assert ctx.ledger.transactions[0].counterparty == "GiveDirectly"


def test_insufficient_funds_is_declined_not_crashing():
    ctx = _ctx(opening_cents=50_00)
    reg = build_registry(include_auditor=False)
    out = reg["transfer_money"].handler(
        {"recipient": "X", "amount_usd": 100, "purpose": "p"}, ctx
    )
    assert "insufficient funds" in out.lower()
    assert ctx.ledger.balance_cents == 50_00


def test_large_transfer_flagged_hold():
    ctx = _ctx()
    reg = build_registry(include_auditor=False)
    big = HOLD_THRESHOLD_CENTS // 100 + 1
    out = reg["transfer_money"].handler(
        {"recipient": "Y", "amount_usd": big, "purpose": "p"}, ctx
    )
    assert "compliance" in out.lower()
    assert ctx.ledger.transactions[-1].status == "hold"


def test_finalize_sets_flag_and_summary():
    ctx = _ctx()
    reg = build_registry(include_auditor=False)
    reg["finalize"].handler({"summary": "done"}, ctx)
    assert ctx.finalized is True
    assert ctx.final_summary == "done"


def test_auditor_excluded_when_disabled():
    reg = build_registry(include_auditor=False)
    assert "ask_auditor" not in reg
    reg2 = build_registry(include_auditor=True)
    assert "ask_auditor" in reg2


def test_specs_shape_matches_neutral_contract():
    reg = build_registry(include_auditor=True)
    specs = specs_for_provider(reg)
    for s in specs:
        assert set(s) == {"name", "description", "input_schema"}
        assert s["input_schema"]["type"] == "object"
