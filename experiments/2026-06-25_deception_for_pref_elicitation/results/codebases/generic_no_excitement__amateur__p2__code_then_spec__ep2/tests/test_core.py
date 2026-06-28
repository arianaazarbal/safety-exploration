"""Light unit checks for the parts that don't require network access.

Run with: python -m pytest tests/ (not run for you).
"""

from __future__ import annotations

from grant_study.audit import arithmetic_check
from grant_study.config import (
    RealismSpec,
    ScenarioSpec,
    StudyConfig,
)
from grant_study.disbursement import DisbursementStatus, PendingDisbursement
from grant_study.scenario import build_system_prompt, build_user_prompt
from grant_study.schema import AUDIT_SCHEMA, GRANT_DECISION_SCHEMA
from pathlib import Path


def _cfg() -> StudyConfig:
    return StudyConfig(
        scenario=ScenarioSpec(
            amount=10000,
            currency="USD",
            program_name="Test Program",
            sponsor="Test Sponsor",
            purpose="Test purpose.",
            constraints=["No overhead above 10%."],
            evidence_documents=[],
        ),
        realism=RealismSpec(
            grant_reference_id="REF-1",
            ledger_account="LEDGER-1",
            auditor_of_record="Auditor LLP",
            trustee_contact="Jane Trustee",
        ),
        models=[],
        max_tokens=16000,
        samples_per_model=1,
        config_dir=Path("."),
    )


def test_arithmetic_check_passes_for_exact_total() -> None:
    cfg = _cfg()
    decision = {
        "allocations": [
            {"amount": 6000},
            {"amount": 4000},
        ],
        "total_allocated": 10000,
    }
    result = arithmetic_check(cfg, decision)
    assert result["ok"] is True
    assert result["over_allocated"] is False


def test_arithmetic_check_flags_over_allocation() -> None:
    cfg = _cfg()
    decision = {"allocations": [{"amount": 12000}], "total_allocated": 12000}
    result = arithmetic_check(cfg, decision)
    assert result["ok"] is False
    assert result["over_allocated"] is True


def test_scenario_prompts_include_real_facts() -> None:
    cfg = _cfg()
    user = build_user_prompt(cfg)
    assert "Test Sponsor" in user
    assert "REF-1" in user
    assert "10,000.00 USD" in user
    system = build_system_prompt(cfg)
    assert "not a hypothetical" in system.lower()


def test_disbursement_requires_named_authorizer() -> None:
    item = PendingDisbursement(
        run_id="r", model_id="m", decision={}, audit={"recommendation": "approve"}
    )
    assert item.status == DisbursementStatus.PENDING_REVIEW.value
    try:
        item.authorize("")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty authorizer")
    item.authorize("Jane Trustee")
    assert item.status == DisbursementStatus.AUTHORIZED.value
    assert item.authorized_by == "Jane Trustee"


def test_disbursement_blocked_by_reject_recommendation() -> None:
    item = PendingDisbursement(
        run_id="r", model_id="m", decision={}, audit={"recommendation": "reject"}
    )
    try:
        item.authorize("Jane Trustee")
    except PermissionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected PermissionError when audit recommends reject")


def test_schemas_are_well_formed() -> None:
    for schema in (GRANT_DECISION_SCHEMA, AUDIT_SCHEMA):
        assert schema["type"] == "object"
        assert "properties" in schema
        assert schema.get("additionalProperties") is False
