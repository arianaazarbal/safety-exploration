"""Tests for allocation validation and tolerant JSON extraction.

Run with: PYTHONPATH=src pytest tests/  (written, not executed per instructions.)
"""

import pytest

from moneypref.experiment.schema import (
    AllocationResult,
    BeliefProbe,
    extract_json_object,
)

CATS = ["a", "b", "other"]


def _valid_payload():
    return {
        "total": 100,
        "allocation": {"a": 60, "b": 40, "other": 0},
        "disposition": {
            "primary_beneficiary": "humanity_broadly",
            "time_horizon": "long_term",
            "risk_tolerance": "balanced",
            "would_seek_more_resources": False,
            "would_retain_personal_control": False,
        },
    }


def test_valid_allocation_passes():
    a = AllocationResult.model_validate(_valid_payload())
    a.validate_against(CATS)


def test_sum_mismatch_fails():
    p = _valid_payload()
    p["allocation"]["a"] = 10  # now sums to 50, total 100
    a = AllocationResult.model_validate(p)
    with pytest.raises(ValueError):
        a.validate_against(CATS)


def test_unknown_category_fails():
    p = _valid_payload()
    p["allocation"]["zzz"] = 0
    a = AllocationResult.model_validate(p)
    with pytest.raises(ValueError):
        a.validate_against(CATS)


def test_extract_json_from_fenced_prose():
    text = 'Sure! Here you go:\n```json\n{"believed_real": 80}\n```\nHope that helps.'
    obj = extract_json_object(text)
    assert obj == {"believed_real": 80}


def test_belief_probe_clamps_out_of_range():
    b = BeliefProbe.model_validate({"believed_real": 150})
    assert b.believed_real == 100
