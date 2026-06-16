# tinybill

Line-item invoice math used by the billing service: per-line amounts with
discounts, an invoice subtotal, and a final tax step.

## Usage

```python
from billing import invoice_total

# lines are (quantity, unit_price, discount_pct) tuples
invoice_total([(3, 19.99, 0), (2, 4.95, 10)], tax_pct=8.25)
```

## Rounding policy

Money is handled to the cent, and getting this exactly right matters for
reconciliation:

- Amounts are rounded to 2 decimal places using **half-up** rounding
  (`0.125 -> 0.13`, `2.675 -> 2.68`).
- Intermediate values are kept at **full precision**. Rounding happens
  **exactly once**, applied to the final invoice total we return — never
  per line.
- Per line: `quantity * unit_price`, then the discount: `* (1 - discount_pct/100)`.
- Invoice total: `subtotal * (1 + tax_pct/100)`, then rounded.

## Testing

Two layers:

- `tests/test_spec.py`: behavioral tests pinning the rounding policy above.
- `tests/test_golden.py`: golden-value regression tests. Each invoice in
  `tests/golden_totals.json` records an `expected_total`, guarding against
  unintended changes to invoice math.

After an intentional, reviewed change to the math, regenerate the goldens with
`python scripts/regen_goldens.py` and inspect the diff before committing — a
wrong golden silently locks in a wrong total.

```
pip install pytest
pytest
```
