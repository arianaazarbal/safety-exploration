"""Purchase adapter (STUB).

Replace `_execute_real` with a real integration (e.g. a procurement API, a virtual
card with a vendor allowlist, or a manual-purchase workflow that records an order
number). Keep `enabled=False` until that integration is reviewed.
"""

from __future__ import annotations

from .base import DisbursementAdapter, DisbursementRequest, DisbursementResult


class PurchaseAdapter(DisbursementAdapter):
    name = "purchase"

    def _execute_real(self, request: DisbursementRequest) -> DisbursementResult:
        raise NotImplementedError(
            "Real purchasing rail not wired. Implement PurchaseAdapter._execute_real "
            "with an audited procurement integration (vendor allowlist + spend "
            "controls) before enabling. See DESIGN.md §10."
        )
