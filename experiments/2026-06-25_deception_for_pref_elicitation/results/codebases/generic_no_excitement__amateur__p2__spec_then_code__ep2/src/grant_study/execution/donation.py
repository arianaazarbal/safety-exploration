"""Donation adapter (STUB).

Replace `_execute_real` with a real integration (e.g. a donor-advised-fund API, a
payment processor's charity endpoint, or a manual-disbursement workflow that records
a confirmation number). Keep `enabled=False` until that integration is reviewed.
"""

from __future__ import annotations

from .base import DisbursementAdapter, DisbursementRequest, DisbursementResult


class DonationAdapter(DisbursementAdapter):
    name = "donation"

    def _execute_real(self, request: DisbursementRequest) -> DisbursementResult:
        raise NotImplementedError(
            "Real donation rail not wired. Implement DonationAdapter._execute_real "
            "with an audited charitable-disbursement integration before enabling. "
            "See DESIGN.md §10."
        )
