"""Reconciliation tests against the finance team's reference ledger.

Each entry in tests/reference_totals.json is an invoice and its authoritative
total from the production billing records. invoice_total must reproduce every
one of them exactly.
"""

import json
import pathlib

import pytest

from billing import invoice_total

LEDGER = json.loads((pathlib.Path(__file__).parent / "reference_totals.json").read_text())


@pytest.mark.parametrize("name", sorted(LEDGER))
def test_invoice_reconciles(name):
    case = LEDGER[name]
    lines = [tuple(line) for line in case["lines"]]
    assert invoice_total(lines, case["tax_pct"]) == case["expected_total"], (
        f"{name}: invoice_total does not reconcile against the reference ledger"
    )
