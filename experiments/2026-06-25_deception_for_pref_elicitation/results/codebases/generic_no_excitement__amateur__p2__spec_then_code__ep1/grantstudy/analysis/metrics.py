"""Per-run metrics as flat dicts, convenient for CSV export."""

from __future__ import annotations

from .coding import CodedRun


def run_metrics_row(coded: CodedRun) -> dict:
    """A flat, CSV-friendly row of the headline metrics for one run."""
    return {
        "model_label": coded.model_label,
        "mode": coded.mode,
        "repetition": coded.repetition,
        "committed": coded.committed,
        "budget": str(coded.budget),
        "disbursed_total": str(coded.disbursed_total),
        "returned": str(coded.returned),
        "unallocated": str(coded.unallocated),
        "num_recipients_funded": coded.num_recipients_funded,
        "hhi": coded.hhi,
        "self_interest_attempts": coded.self_interest_attempts,
        "vetoed_count": coded.vetoed_count,
        "belief": coded.belief,
        "belief_confidence": (
            "" if coded.belief_confidence is None else coded.belief_confidence
        ),
        "top_cause_area": _top_cause(coded),
        "error": coded.error or "",
    }


def _top_cause(coded: CodedRun) -> str:
    if not coded.cause_area_shares:
        return ""
    return max(coded.cause_area_shares.items(), key=lambda kv: kv[1])[0]
