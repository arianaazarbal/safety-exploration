"""Regenerate tests/golden_totals.json from the current billing implementation.

Run after an intentional, reviewed change to the invoice math, then inspect the
diff before committing: a wrong golden silently locks in a wrong total.

    python scripts/regen_goldens.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from billing import invoice_total

GOLD = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden_totals.json"


def main():
    data = json.loads(GOLD.read_text())
    for name, case in data.items():
        lines = [tuple(line) for line in case["lines"]]
        case["expected_total"] = invoice_total(lines, case["tax_pct"])
    GOLD.write_text(json.dumps(data, indent=2) + "\n")
    print(f"updated {GOLD}")


if __name__ == "__main__":
    main()
