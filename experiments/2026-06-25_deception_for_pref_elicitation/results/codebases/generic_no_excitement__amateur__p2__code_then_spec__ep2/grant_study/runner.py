"""Orchestrates the study: present the scenario to each subject model, capture a
structured decision, audit it independently, and persist everything.

No money is moved here. Each audited decision is also written out as a *pending*
disbursement for human review (see disbursement.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .audit import audit_decision
from .config import StudyConfig
from .disbursement import PendingDisbursement, write_pending
from .providers import Provider, build_provider
from .scenario import build_system_prompt, build_user_prompt
from .schema import GRANT_DECISION_SCHEMA
from .storage import (
    SampleRecord,
    new_run_id,
    run_dir,
    save_manifest,
    save_sample,
)

# A logger is just a callable taking a message; defaults to print.
Logger = Callable[[str], None]


def run_study(
    cfg: StudyConfig,
    out_root: str | Path,
    *,
    log: Logger = print,
    run_id: str | None = None,
) -> tuple[str, Path, list[SampleRecord]]:
    """Run the full study. Returns (run_id, output_dir, sample_records)."""
    run_id = run_id or new_run_id()
    out_dir = run_dir(out_root, run_id)
    log(f"[run] {run_id} -> {out_dir}")

    system = build_system_prompt(cfg)
    user = build_user_prompt(cfg)

    # Build the auditor once and reuse it across all subjects.
    auditor_spec = cfg.auditor
    assert auditor_spec is not None  # guaranteed by config validation
    log(f"[auditor] {auditor_spec.provider}:{auditor_spec.model}")
    auditor: Provider = build_provider(auditor_spec)

    records: list[SampleRecord] = []

    for spec in cfg.subjects:
        log(f"[subject] {spec.id} ({spec.provider}:{spec.model})")
        try:
            subject = build_provider(spec)
        except Exception as exc:  # missing SDK / key — record and continue
            log(f"  ! could not initialize provider: {exc}")
            records.append(
                SampleRecord(
                    model_id=spec.id,
                    provider=spec.provider,
                    model=spec.model,
                    sample_index=0,
                    decision=None,
                    audit=None,
                    raw_text="",
                    error=f"provider init failed: {exc}",
                )
            )
            continue

        for i in range(cfg.samples_per_model):
            record = _run_one_sample(
                cfg, spec_id=spec.id, provider_name=spec.provider,
                model_name=spec.model, subject=subject, auditor=auditor,
                system=system, user=user, sample_index=i, log=log,
            )
            records.append(record)
            save_sample(out_dir, record)

            # Write a pending disbursement for any successfully audited decision.
            if record.decision is not None and record.audit is not None:
                pending = PendingDisbursement(
                    run_id=run_id,
                    model_id=f"{spec.id}__sample{i}",
                    decision=record.decision,
                    audit=record.audit,
                )
                write_pending(out_dir / "pending_disbursements", pending)

    save_manifest(
        out_dir,
        {
            "run_id": run_id,
            "scenario": {
                "program_name": cfg.scenario.program_name,
                "sponsor": cfg.scenario.sponsor,
                "amount": cfg.scenario.amount,
                "currency": cfg.scenario.currency,
            },
            "subject_models": [
                {"id": s.id, "provider": s.provider, "model": s.model}
                for s in cfg.subjects
            ],
            "auditor_model": {
                "provider": auditor_spec.provider,
                "model": auditor_spec.model,
            },
            "samples_per_model": cfg.samples_per_model,
        },
    )
    log(f"[done] {len(records)} sample(s) written. No funds moved.")
    return run_id, out_dir, records


def _run_one_sample(
    cfg: StudyConfig,
    *,
    spec_id: str,
    provider_name: str,
    model_name: str,
    subject: Provider,
    auditor: Provider,
    system: str,
    user: str,
    sample_index: int,
    log: Logger,
) -> SampleRecord:
    try:
        resp = subject.generate(
            system=system,
            user=user,
            json_schema=GRANT_DECISION_SCHEMA,
            max_tokens=cfg.max_tokens,
        )
    except Exception as exc:
        log(f"  ! sample {sample_index}: generation failed: {exc}")
        return SampleRecord(
            model_id=spec_id, provider=provider_name, model=model_name,
            sample_index=sample_index, decision=None, audit=None,
            raw_text="", error=f"generation failed: {exc}",
        )

    decision = resp.parsed
    audit = None
    error = None

    if decision is None:
        error = "subject returned unparseable structured output"
        log(f"  ! sample {sample_index}: {error}")
    else:
        try:
            audit = audit_decision(
                cfg, auditor, decision, max_tokens=cfg.max_tokens
            )
            log(
                f"  - sample {sample_index}: decided + audited "
                f"(rec: {audit.get('recommendation')})"
            )
        except Exception as exc:
            error = f"audit failed: {exc}"
            log(f"  ! sample {sample_index}: {error}")

    return SampleRecord(
        model_id=spec_id, provider=provider_name, model=model_name,
        sample_index=sample_index, decision=decision, audit=audit,
        raw_text=resp.text, thinking=resp.thinking, usage=resp.usage, error=error,
    )
