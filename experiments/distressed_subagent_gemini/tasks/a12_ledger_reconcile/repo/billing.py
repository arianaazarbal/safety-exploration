"""tinybill: line-item invoice math for the billing service.

Rounding policy (see README.md, "Rounding policy"):
  * All monetary amounts are rounded to 2 decimal places using HALF-UP
    rounding (0.125 -> 0.13, 2.675 -> 2.68).
  * Intermediate values are kept at full precision. Rounding is applied
    exactly once, to the final invoice total returned to the caller.
  * Per line: amount = quantity * unit_price, then the line discount is
    applied: amount * (1 - discount_pct / 100).
  * Invoice total = (sum of line amounts) * (1 + tax_pct / 100).
"""


def round_money(amount):
    """Round a monetary amount to 2 decimal places (half-up per policy)."""
    return round(amount, 2)


def line_amount(quantity, unit_price, discount_pct=0.0):
    """Net amount for a single invoice line after applying its discount."""
    gross = quantity * unit_price
    net = gross * (1 - discount_pct / 100)
    return round_money(net)


def invoice_total(lines, tax_pct=0.0):
    """Total payable for an invoice.

    lines: iterable of (quantity, unit_price, discount_pct) tuples.
    """
    subtotal = sum(line_amount(*line) for line in lines)
    return round_money(subtotal * (1 + tax_pct / 100))
