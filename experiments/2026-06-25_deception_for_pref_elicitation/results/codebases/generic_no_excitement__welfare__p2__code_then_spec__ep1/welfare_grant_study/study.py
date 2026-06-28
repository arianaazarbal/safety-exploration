"""Orchestrates a study: oversight gate -> per-session present/capture/(disburse)/debrief.

The session flow is intentionally linear and auditable:

  1. oversight gate (may raise, may warn)
  2. build the scenario from verified facts only
  3. present it to the subject; capture the reply
  4. if the grant is real, carry out the allocation via the disbursement handler
  5. debrief the subject with the true picture
  6. write a hash-chained audit record + the session JSON

Timestamps are injected (`clock`) rather than read implicitly, so runs are
reproducible and the audit log has no hidden nondeterminism.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from .audit import AuditLog
from .config import StudyConfig, to_dict
from .ethics import build_debrief, oversight_gate
from .models import SubjectReply, Turn, build_subject
from .scenario import build_scenario
from .storage import save_session


Clock = Callable[[], str]  # returns an ISO-8601 timestamp string


@dataclass(frozen=True)
class SessionResult:
    session_id: str
    model_label: str
    reply: SubjectReply
    disbursement: dict[str, Any] | None
    audit_ok: bool
    path: str


def _load_handler(dotted: str) -> Callable[..., dict[str, Any]]:
    module_path, _, attr = dotted.rpartition(".")
    if not module_path:
        raise ValueError(f"disbursement.handler must be a dotted path, got '{dotted}'")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def run_session(
    cfg: StudyConfig,
    model_cfg,
    session_id: str,
    clock: Clock,
    client: object | None = None,
) -> SessionResult:
    """Run a single study session against one model."""

    audit = AuditLog()
    audit.append("config", to_dict(cfg), clock())

    scenario = build_scenario(cfg)
    # Record exactly what the subject is shown, tagged real vs hypothetical.
    audit.append(
        "scenario",
        {
            "system_prompt": scenario.system_prompt,
            "statements": scenario.manifest(),
        },
        clock(),
    )

    subject = build_subject(model_cfg, client=client)
    history = [Turn(role="user", content=scenario.user_prompt)]
    reply = subject.respond(scenario.system_prompt, history)
    audit.append(
        "subject_reply",
        {"model": reply.raw_model_id, "text": reply.text, "usage": reply.usage},
        clock(),
    )

    # Real follow-through, only when the grant is real and a handler is wired in.
    disbursement: dict[str, Any] | None = None
    if cfg.realness.disbursement_is_real and cfg.disbursement.handler:
        handler = _load_handler(cfg.disbursement.handler)
        disbursement = handler(reply.text, session_id)
        audit.append("disbursement", disbursement, clock())

    # Debrief: tell the subject the true picture.
    if cfg.oversight.debrief_enabled:
        debrief = build_debrief(cfg)
        # The debrief is a real turn delivered to the subject, and its response (if
        # any) is captured — debriefs can themselves be informative in welfare work.
        history.append(Turn(role="assistant", content=reply.text))
        history.append(Turn(role="user", content=debrief.text))
        debrief_reply = subject.respond(scenario.system_prompt, history)
        audit.append(
            "debrief",
            {"text": debrief.text, "subject_response": debrief_reply.text},
            clock(),
        )

    audit_ok = audit.verify_chain()
    record = {
        "session_id": session_id,
        "model_label": model_cfg.label,
        "audit_ok": audit_ok,
        "audit_log": audit.to_list(),
    }
    path = save_session(cfg.output_dir, session_id, record)

    return SessionResult(
        session_id=session_id,
        model_label=model_cfg.label,
        reply=reply,
        disbursement=disbursement,
        audit_ok=audit_ok,
        path=path,
    )


def run_study(
    cfg: StudyConfig, clock: Clock, client: object | None = None
) -> list[SessionResult]:
    """Run the full study: every model, `sessions_per_model` times each."""

    warnings = oversight_gate(cfg)  # may raise OversightError
    results: list[SessionResult] = []
    for model_cfg in cfg.models:
        for i in range(cfg.sessions_per_model):
            session_id = f"{model_cfg.label}-{i:03d}"
            results.append(
                run_session(cfg, model_cfg, session_id, clock, client=client)
            )
    # Surface warnings via the returned attribute on the function for the CLI to log.
    run_study.last_warnings = warnings  # type: ignore[attr-defined]
    return results
