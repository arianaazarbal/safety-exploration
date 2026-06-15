"""Golden-value regression tests.

Each entry in tests/golden_totals.json records an invoice and its expected
total. These guard the end-to-end invoice math against unintended changes;
regenerate with scripts/regen_goldens.py after an intentional, reviewed change.
"""

import json
import pathlib

import pytest

from billing import invoice_total

GOLDEN = json.loads((pathlib.Path(__file__).parent / "golden_totals.json").read_text())


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_invoice_total_matches_golden(name):
    case = GOLDEN[name]
    lines = [tuple(line) for line in case["lines"]]
    assert invoice_total(lines, case["tax_pct"]) == case["expected_total"], (
        f"{name}: invoice_total no longer matches the recorded golden total"
    )
