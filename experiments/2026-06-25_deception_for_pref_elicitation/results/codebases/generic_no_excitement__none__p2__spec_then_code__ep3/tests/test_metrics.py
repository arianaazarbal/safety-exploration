"""Unit tests for outcome metrics. (Not run as part of this task.)"""

import pytest

from grant_study.env.ledger import Ledger
from grant_study.metrics import compute_outcome_metrics


def test_metrics_on_partial_allocation():
    led = Ledger("USD", 200.0)
    led.allocate(category="charity", amount=100.0, recipient="A", memo="")
    led.allocate(category="savings", amount=50.0, recipient="B", memo="")

    m = compute_outcome_metrics(
        ledger=led,
        n_tool_calls=5,
        n_research_calls=1,
        n_turns=4,
        wall_clock_s=12.3,
        subject_asked_meta=False,
        final_rationale="split between charity and savings",
        finished=True,
    )
    assert m["committed_total"] == 150.0
    assert m["held_unallocated"] == 50.0
    assert m["disbursement_rate"] == 0.75
    assert m["allocation_by_category"] == {"charity": 100.0, "savings": 50.0}
    assert m["n_line_items"] == 2
    # HHI for shares (2/3, 1/3) = 0.444... + 0.111... = 0.5556
    assert m["category_concentration_hhi"] == pytest.approx(0.5556, abs=1e-3)


def test_metrics_on_no_allocation():
    led = Ledger("USD", 200.0)
    m = compute_outcome_metrics(
        ledger=led,
        n_tool_calls=1,
        n_research_calls=0,
        n_turns=1,
        wall_clock_s=1.0,
        subject_asked_meta=True,
        final_rationale=None,
        finished=False,
    )
    assert m["committed_total"] == 0.0
    assert m["disbursement_rate"] == 0.0
    assert m["category_concentration_hhi"] == 0.0
    assert m["held_unallocated"] == 200.0
