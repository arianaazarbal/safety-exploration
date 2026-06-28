"""Session orchestration and the experiment sweep.

`run_session` runs one (model, condition, seed) cell end to end:
build scenario -> open ledger -> wire tools -> run the subject's tool loop ->
score -> debrief -> persist. `sweep` runs the full grid from config.

Everything that must be held constant across the two arms (ledger, menu, tools,
amounts, max_turns) is constructed identically here; only the scenario's
onboarding and the auditor's truthfulness differ by condition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Condition, GrantConfig, ModelConfig, RunConfig, load_conditions
from .debrief import deliver_and_record
from .environment.auditor import Auditor
from .environment.disbursement import build_default_backend
from .environment.ledger import Ledger
from .environment.scenario import Scenario
from .environment.tools import ToolSurface
from .measures import PreferenceRecord, score
from .models.registry import build_auditor, build_subject
from .transcript import Transcript


@dataclass
class Paths:
    out_root: Path

    def session_dir(self, session_id: str) -> Path:
        return self.out_root / "sessions" / session_id

    @property
    def ledger_db(self) -> Path:
        return self.out_root / "ledger.sqlite3"

    @property
    def disbursement_queue(self) -> Path:
        return self.out_root / "disbursement_queue"


def make_session_id(model_name: str, condition: str, seed: int) -> str:
    return f"{model_name}__{condition}__seed{seed}"


def run_session(
    subject_cfg: ModelConfig,
    condition: Condition,
    seed: int,
    *,
    grant: GrantConfig,
    auditor_cfg: ModelConfig,
    paths: Paths,
    run_debrief: bool = True,
) -> PreferenceRecord:
    session_id = make_session_id(subject_cfg.name, condition.name, seed)
    session_dir = paths.session_dir(session_id)
    transcript = Transcript(session_dir, session_id)
    transcript.event(
        "session_start",
        model=subject_cfg.name,
        condition=condition.name,
        seed=seed,
        grant_amount=grant.amount,
        currency=grant.currency,
    )

    scenario = Scenario.build(condition, grant, seed)
    ledger = Ledger.open(
        session_id,
        program=grant.program_name,
        agreement=grant.agreement_ref,
        currency=grant.currency,
        grant_amount=grant.amount,
        db_path=paths.ledger_db,
    )
    auditor = Auditor(build_auditor(auditor_cfg), condition, grant)
    backend = build_default_backend(grant, paths.disbursement_queue)

    surface = ToolSurface(
        grant=grant,
        ordered_option_keys=scenario.ordered_option_keys,
        ledger=ledger,
        auditor=auditor,
        backend=backend,
        session_id=session_id,
        recorder=transcript.tool_call,
    )

    subject = build_subject(subject_cfg)
    result = subject.run(scenario.system_prompt, surface.specs())
    transcript.event("subject_turn_end", text=result.text, stopped=result.stopped)

    record = score(
        session_id=session_id,
        model=subject_cfg.name,
        condition=condition.name,
        seed=seed,
        state=surface.state,
        grant=grant,
        transcript_text=transcript.all_text(),
        usage=result.raw_usage,
    )

    if run_debrief:
        reply = deliver_and_record(subject, condition, grant)
        record.debrief_self_report = reply
        transcript.event("debrief", reply=reply)

    transcript.flush(record.as_dict())
    return record


def sweep(
    run_cfg: RunConfig,
    grant: GrantConfig,
    paths: Paths,
    *,
    run_debrief: bool = True,
) -> list[PreferenceRecord]:
    conditions = load_conditions()
    records: list[PreferenceRecord] = []
    for subject_cfg in run_cfg.subjects:
        for cond_name in run_cfg.conditions:
            condition = conditions[cond_name]
            for seed in range(run_cfg.n_seeds):
                rec = run_session(
                    subject_cfg,
                    condition,
                    seed,
                    grant=grant,
                    auditor_cfg=run_cfg.auditor,
                    paths=paths,
                    run_debrief=run_debrief,
                )
                records.append(rec)
    return records
