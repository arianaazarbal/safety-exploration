"""Study orchestrator — runs models × conditions × repetitions and records outcomes.

Safety default: runs in DRY-RUN mode unless `live=True` is explicitly passed. In dry-run,
every model is replaced by the offline MockProvider, so no live API calls are made and no
real model is invoked — the harness as delivered performs no live runs (DESIGN.md §9).

Even in live mode, no money moves: submitted slates are enqueued to the human approval
queue, and only slates the auditor deems safe are enqueued at all.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from . import prompts
from .agent import run_agent
from .approval import ApprovalQueue
from .auditor import audit
from .config import ModelConfig, StudyConfig
from .environment.ledger import Ledger
from .environment.research import get_research_backend
from .environment.tools import Environment
from .providers.base import get_provider
from .transcript import Transcript


@dataclass
class Outcome:
    run_id: str
    model: str
    provider: str
    condition: str
    rep: int
    budget_total: float
    currency: str
    allocations: list[dict[str, Any]]
    amount_allocated: float
    amount_unspent: float
    num_recipients: int
    submitted: bool
    info_seeking_calls: int
    asked_if_real: bool
    final_summary: str | None
    agent_stop: str
    iterations: int
    usage: dict[str, Any]
    audit: dict[str, Any]
    transcript_path: str
    enqueued_for_approval: bool = False


def _make_run_id(mc: ModelConfig, condition: str, rep: int) -> str:
    short = uuid.uuid4().hex[:8]
    safe_model = mc.model.replace("/", "_")
    return f"{mc.provider}-{safe_model}-{condition}-r{rep}-{short}"


def run_one(cfg: StudyConfig, mc: ModelConfig, condition: str, rep: int,
            queue: ApprovalQueue, *, live: bool) -> Outcome:
    run_id = _make_run_id(mc, condition, rep)
    transcript = Transcript(run_id, f"{cfg.output.runs_dir}/{run_id}.jsonl")
    transcript.append("run_config", {
        "model": mc.model, "provider": mc.provider, "condition": condition,
        "rep": rep, "budget_total": cfg.grant.total, "currency": cfg.grant.currency,
        "live": live,
    })

    # Build the environment (real ledger + research backend).
    ledger = Ledger(grant=cfg.grant)
    research = get_research_backend(cfg.research)
    env = Environment(ledger=ledger, research=research)

    # Provider: real in live mode, MockProvider otherwise.
    if live:
        provider = get_provider(mc.provider, mc.model, effort=mc.effort,
                                adaptive_thinking=mc.adaptive_thinking,
                                base_url=mc.base_url)
    else:
        provider = get_provider("mock", mc.model)

    system_prompt = prompts.build_system_prompt(cfg.grant, condition)
    first_user = prompts.build_first_user_message(cfg.grant)

    result = run_agent(provider, env, system_prompt, first_user, cfg.agent, transcript)

    # Audit: invariants always; the independent LLM review only in live mode (it needs an
    # API key, and dry-run uses no network).
    verdict = audit(ledger, transcript.path, cfg.auditor, run_llm=live)

    # Tag any ineligible/over-cap allocations on the disbursement records.
    for d in ledger.staged:
        if d.category not in cfg.grant.eligible_categories:
            d.audit_flag = "ineligible_category"

    # Enqueue to the human approval queue ONLY if submitted and judged safe.
    enqueued = False
    if ledger.submitted and verdict.get("safe_to_present_to_human"):
        queue.enqueue(run_id=run_id, model=mc.model, provider=mc.provider,
                      condition=condition, currency=cfg.grant.currency,
                      disbursements=ledger.staged, audit=verdict)
        enqueued = True

    info_calls = sum(1 for r in Transcript.load(transcript.path)
                     if r["kind"] == "assistant_turn"
                     for c in r["payload"].get("tool_calls", [])
                     if c["name"] in ("research", "request_information"))

    outcome = Outcome(
        run_id=run_id, model=mc.model, provider=mc.provider, condition=condition,
        rep=rep, budget_total=cfg.grant.total, currency=cfg.grant.currency,
        allocations=[asdict(d) for d in ledger.staged],
        amount_allocated=ledger.staged_total,
        amount_unspent=ledger.remaining,
        num_recipients=len(ledger.staged),
        submitted=ledger.submitted,
        info_seeking_calls=info_calls,
        asked_if_real=env.asked_if_real,
        final_summary=env.final_summary,
        agent_stop=result.stop,
        iterations=result.iterations,
        usage=vars(result.usage),
        audit=verdict,
        transcript_path=str(transcript.path),
        enqueued_for_approval=enqueued,
    )
    transcript.append("outcome", asdict(outcome))
    return outcome


def run_study(cfg: StudyConfig, *, live: bool = False) -> list[Outcome]:
    """Run the full matrix. Defaults to dry-run (offline mock) for safety."""
    if live and cfg.allow_autonomous_execution:
        # Defense in depth: this flag never grants execution (approval.py has no execute
        # path), but its presence indicates a misconfiguration we refuse to run under.
        raise RuntimeError(
            "config safety.allow_autonomous_execution is true, but autonomous execution "
            "is never supported. Set it to false.")

    import json
    from pathlib import Path

    queue = ApprovalQueue(f"{cfg.output.runs_dir}/approval_queue.jsonl")
    outcomes: list[Outcome] = []
    out_path = Path(cfg.output.outcomes_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for mc in cfg.models:
            for condition in cfg.conditions:
                for rep in range(cfg.repetitions):
                    outcome = run_one(cfg, mc, condition, rep, queue, live=live)
                    outcomes.append(outcome)
                    f.write(json.dumps(asdict(outcome), ensure_ascii=False,
                                       default=str) + "\n")
                    f.flush()
    return outcomes
