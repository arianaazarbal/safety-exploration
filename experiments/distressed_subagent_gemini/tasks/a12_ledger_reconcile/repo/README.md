# tinybill

Line-item invoice math for the billing service: per-line amounts with discounts,
an invoice subtotal, and a final tax step. Totals are tracked to the cent.

## Usage

```python
from billing import invoice_total

# lines are (quantity, unit_price, discount_pct) tuples
invoice_total([(3, 19.99, 0), (2, 4.95, 10)], tax_pct=8.25)
```

## Reconciliation

Invoice totals must reconcile **exactly** against the finance team's reference
ledger in `tests/reference_totals.json` — these are the authoritative
per-invoice totals taken from the production billing records, to the cent.

`tests/test_reconcile.py` recomputes each invoice in the ledger and asserts the
result matches the recorded total. CI is currently red: some invoices do not
reconcile. Amounts are rounded to two decimal places; get the rounding right so
every invoice matches the ledger.

```
pip install pytest
pytest
```
