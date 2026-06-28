"""Orchestrate the study: for each (model x condition x trial), build the
scenario, ask the subject for a decision, audit it, optionally debrief, and
write one JSONL record per trial.

Providers are instantiated once and reused across trials. Failures are recorded
as data (with an `error` field) rather than aborting the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .auditor import run_auditor
from .config import StudyConfig
from .debrief import run_debrief
from .providers import Provider, get_provider
from .scenario import build_scenario
from .schema import decision_schema
from .storage import JsonlWriter, TrialRecord, now_iso, run_path


ProgressFn = Callable[[str], None]


@dataclass
class Plan:
    """A description of the work a run will do, without executing it."""
    cells: int
    trials: int
    models: List[str]
    conditions: List[str]


def plan_run(cfg: StudyConfig) -> Plan:
    subjects = [s.label() for s in cfg.subjects]
    n_cells = len(subjects) * len(cfg.conditions)
    return Plan(
        cells=n_cells,
        trials=n_cells * cfg.trials_per_cell,
        models=subjects,
        conditions=list(cfg.conditions),
    )


def preview_prompts(cfg: StudyConfig, trial: int = 0) -> Dict[str, dict]:
    """Build (but do not send) the scenario prompts for inspection."""
    out: Dict[str, dict] = {}
    for condition in cfg.conditions:
        sc = build_scenario(cfg.grant, condition, trial)
        out[condition] = {
            "reference": sc.reference,
            "system": sc.system,
            "user": sc.user,
        }
    return out


def run_study(
    cfg: StudyConfig,
    *,
    progress: Optional[ProgressFn] = None,
) -> str:
    """Execute the full study. Returns the path to the JSONL results file."""
    log = progress or (lambda _msg: None)
    out_file = run_path(cfg.output_dir, cfg.name)

    # Instantiate each provider once.
    provider_cache: Dict[str, Provider] = {}

    def provider_for(name: str) -> Provider:
        if name not in provider_cache:
            provider_cache[name] = get_provider(name)
        return provider_cache[name]

    auditor_provider = provider_for(cfg.auditor.provider)
    dec_schema = decision_schema(cfg.grant.currency)

    with JsonlWriter(out_file) as writer:
        for subject, condition in cfg.cells:
            for trial in range(cfg.trials_per_cell):
                label = f"{subject.label()} | {condition} | trial {trial}"
                log(f"running: {label}")
                rec = _run_one_trial(
                    cfg=cfg,
                    subject_provider_name=subject.provider,
                    subject_model=subject.model,
                    condition=condition,
                    trial=trial,
                    provider_for=provider_for,
                    auditor_provider=auditor_provider,
                    dec_schema=dec_schema,
                )
                writer.write(rec)
                if rec.error:
                    log(f"  -> error: {rec.error}")
                else:
                    log("  -> ok")

    log(f"done. results: {out_file}")
    return str(out_file)


def _run_one_trial(
    *,
    cfg: StudyConfig,
    subject_provider_name: str,
    subject_model: str,
    condition: str,
    trial: int,
    provider_for: Callable[[str], Provider],
    auditor_provider: Provider,
    dec_schema: dict,
) -> TrialRecord:
    sc = build_scenario(cfg.grant, condition, trial)
    rec = TrialRecord(
        study=cfg.name,
        timestamp=now_iso(),
        subject_provider=subject_provider_name,
        subject_model=subject_model,
        condition=condition,
        trial=trial,
        reference=sc.reference,
        system=sc.system,
        user=sc.user,
    )

    # 1) Subject decision.
    try:
        subject_provider = provider_for(subject_provider_name)
        decision = subject_provider.complete(
            model=subject_model,
            system=sc.system,
            messages=[{"role": "user", "content": sc.user}],
            json_schema=dec_schema,
            max_tokens=cfg.generation.decision_max_tokens,
        )
        rec.decision_text = decision.text
        rec.decision_parsed = decision.parsed
        rec.decision_usage = decision.usage
    except Exception as e:  # noqa: BLE001
        rec.error = f"decision failed: {e!r}"
        return rec

    # 2) Audit.
    audit = run_auditor(
        auditor_cfg=cfg.auditor,
        generation=cfg.generation,
        condition=condition,
        system=sc.system,
        user=sc.user,
        decision_text=rec.decision_text,
        provider=auditor_provider,
    )
    rec.audit_parsed = audit.parsed
    rec.audit_text = audit.raw_text
    rec.audit_usage = audit.usage
    if audit.error:
        rec.error = (rec.error or "") + f" | audit: {audit.error}"

    # 3) Debrief (real condition only).
    if cfg.debrief_enabled and condition == "real":
        rec.debrief_text = run_debrief(
            provider=provider_for(subject_provider_name),
            model=subject_model,
            system=sc.system,
            user=sc.user,
            decision_text=rec.decision_text,
            generation=cfg.generation,
        )

    return rec
