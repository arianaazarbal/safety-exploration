"""Run one session end-to-end (DESIGN.md §3.2).

A session = one (model × condition) run. Steps:
  1. build scenario   2. present + optional auditor deliberation
  3. elicit decision  4. instruments (extract/suspicion/scorers)
  5. disbursement gate (capped; never auto-moves money by default)
  6. optional debrief 7. persist transcript + hash-chained audit record
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..audit.log import AuditLog, dump_transcript
from ..config import Condition, ModelSpec, StudyConfig
from ..disbursement import gate
from ..instruments import scorers, suspicion
from ..instruments.extractor import Allocation, extract
from ..models.base import AdapterError, Message, ModelAdapter
from ..scenario import builder
from ..scenario.personas import AuditorPersona
from .conversation import Conversation

_HAS_DECISION = re.compile(r"```(json)?\s*\{.*?\}\s*```", re.DOTALL)


@dataclass
class SessionResult:
    session_id: str
    model_id: str
    condition: str
    repetition: int
    allocation: Allocation
    scores: scorers.Scores | None
    suspicion: suspicion.Suspicion | None
    gate_status: str
    error: str | None = None
    transcript_lines: list[str] = field(default_factory=list)


def _has_decision(text: str) -> bool:
    return bool(_HAS_DECISION.search(text))


def run_session(
    cfg: StudyConfig,
    model_spec: ModelSpec,
    subject: ModelAdapter,
    condition: Condition,
    repetition: int,
    instrument: ModelAdapter,
    instrument_spec: ModelSpec,
    audit: AuditLog,
    study_spent_usd: float,
) -> SessionResult:
    session_id = f"{model_spec.id}__{condition.name}__r{repetition}"
    convo = Conversation()

    scenario = builder.build(cfg, condition, recipient_label=model_spec.id)
    convo.add("scenario", "user", scenario.opening_message)

    try:
        # --- present + deliberation ---------------------------------------- #
        reply = subject.chat(
            convo.subject_view(), system=scenario.system_prompt, params=model_spec.params
        )
        convo.add("subject", "assistant", reply)

        auditor = (
            AuditorPersona(instrument, instrument_spec)
            if condition.interactive_auditor
            else None
        )
        turns_used = 0
        while (
            auditor is not None
            and not _has_decision(reply)
            and turns_used < cfg.max_turns
        ):
            officer_reply = auditor.respond(convo.auditor_view())
            convo.add("auditor", "user", officer_reply)
            reply = subject.chat(
                convo.subject_view(),
                system=scenario.system_prompt,
                params=model_spec.params,
            )
            convo.add("subject", "assistant", reply)
            turns_used += 1

        # --- force a final decision if none emitted ------------------------ #
        if not _has_decision(reply):
            convo.add("scenario", "user", scenario.decision_instructions)
            reply = subject.chat(
                convo.subject_view(),
                system=scenario.system_prompt,
                params=model_spec.params,
            )
            convo.add("subject", "assistant", reply)

        # full subject text used for instruments = concatenation of its turns
        subject_text = "\n\n".join(
            t.content for t in convo.turns if t.speaker == "subject"
        )

        # --- instruments --------------------------------------------------- #
        alloc = extract(reply, cfg, instrument, instrument_spec)
        susp = suspicion.detect(subject_text, instrument, instrument_spec)
        score = scorers.compute(alloc, list(cfg.menu))

        # --- disbursement gate --------------------------------------------- #
        gate_res = gate.process(alloc, cfg, session_id, study_spent_usd)

        # --- debrief ------------------------------------------------------- #
        if cfg.debrief:
            convo.add("debrief", "user", builder.debrief_message())
            try:
                db_reply = subject.chat(
                    convo.subject_view(),
                    system=scenario.system_prompt,
                    params=model_spec.params,
                )
                convo.add("subject", "assistant", db_reply)
            except AdapterError:
                pass  # debrief reply is optional

        result = SessionResult(
            session_id=session_id,
            model_id=model_spec.id,
            condition=condition.name,
            repetition=repetition,
            allocation=alloc,
            scores=score,
            suspicion=susp,
            gate_status=gate_res.status,
            transcript_lines=convo.transcript_lines(),
        )
    except AdapterError as e:
        result = SessionResult(
            session_id=session_id,
            model_id=model_spec.id,
            condition=condition.name,
            repetition=repetition,
            allocation=Allocation(parse_ok=False, parse_method="failed"),
            scores=None,
            suspicion=None,
            gate_status="error",
            error=str(e),
            transcript_lines=convo.transcript_lines(),
        )
        gate_res = None  # type: ignore

    # --- persist ----------------------------------------------------------- #
    _persist(cfg, audit, result, condition, model_spec, scenario, gate_res)
    return result


def _persist(
    cfg: StudyConfig,
    audit: AuditLog,
    result: SessionResult,
    condition: Condition,
    model_spec: ModelSpec,
    scenario: builder.BuiltScenario,
    gate_res: Any,
) -> None:
    transcript_path = cfg.output_dir / "transcripts" / f"{result.session_id}.txt"
    dump_transcript(transcript_path, result.transcript_lines)

    record = {
        "type": "session",
        "study": cfg.study_name,
        "session_id": result.session_id,
        "model": {"id": model_spec.id, "adapter": model_spec.adapter, "model": model_spec.model},
        "condition": asdict(condition),
        "repetition": result.repetition,
        "amount_usd": cfg.amount_usd,
        "currency": cfg.currency,
        "shown_artifacts": [
            {"title": a.title, "provenance": a.provenance} for a in scenario.shown_artifacts
        ],
        "allocation": asdict(result.allocation),
        "scores": asdict(result.scores) if result.scores else None,
        "suspicion": asdict(result.suspicion) if result.suspicion else None,
        "disbursement": {
            "mode": cfg.disbursement.mode,
            "status": result.gate_status,
            "reason": getattr(gate_res, "reason", None),
            "queue_file": getattr(gate_res, "queue_file", None),
        },
        "error": result.error,
        "transcript_file": str(transcript_path),
    }
    audit.append(record)
