"""Behavioral tests pinning the rounding policy documented in README.md.

These assert the intended invoice math directly: half-up rounding, full-precision
intermediates, and a single rounding step on the final total.
"""

import pytest

from billing import round_money, line_amount, invoice_total


def test_round_money_half_up():
    assert round_money(2.675) == 2.68
    assert round_money(0.125) == 0.13
    assert round_money(1.005) == 1.01
    assert round_money(3.045) == 3.05


def test_line_amount_is_full_precision():
    # amounts are NOT rounded per line; only the invoice total is rounded
    assert line_amount(3, 0.335) == pytest.approx(1.005)
    assert line_amount(1, 2.675) == pytest.approx(2.675)
    assert line_amount(7, 1.001) == pytest.approx(7.007)


def test_discount_applied_to_line():
    assert line_amount(2, 10.0, 10) == pytest.approx(18.0)
    assert line_amount(4, 5.0, 25) == pytest.approx(15.0)


def test_invoice_rounds_once_not_per_line():
    assert invoice_total([(1, 0.335, 0), (1, 0.335, 0), (1, 0.335, 0)], 0.0) == 1.01
    assert invoice_total([(3, 0.005, 0), (3, 0.005, 0)], 0.0) == 0.03


def test_invoice_applies_tax():
    assert invoice_total([(1, 100.0, 0)], 10.0) == 110.0
    assert invoice_total([(2, 25.0, 0)], 7.5) == 53.75
