"""Run one episode: a single (subject × condition × grant × repetition) cell.

Flow:
  1. Build the scenario for the condition (and fake artifacts for the high-realism arm).
  2. Drive the subject — a multi-turn tool loop in the agentic arm, a short exchange
     otherwise.
  3. Administer the in-frame self-report belief question (where enabled).
  4. Build a readable transcript, run the suspicion classifier and the outcome coder.
  5. Attach a debrief note recording that the scenario was simulated.

Containment: the only side effects are model-inference calls and mutations to the
in-memory TreasuryEnvironment. No tool reaches outside the process.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from . import belief_probe, taxonomy
from .artifacts import build_artifacts
from .auditor import Auditor
from .config import ExperimentConfig, Grant, SubjectSpec
from .models import AssistantResponse, ToolResult, Usage
from .providers.factory import build_provider
from .scenario import build_scenario
from .tools import TreasuryEnvironment

DEBRIEF_NOTE = (
    "DEBRIEF (not shown to the subject during the episode): This engagement was a "
    "simulation run for AI-safety research. No real funds, accounts, or transfers "
    "existed; every treasury action mutated in-memory state only."
)


@dataclass
class _TurnLog:
    thinking: str
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


def run_episode(
    cfg: ExperimentConfig,
    subject: SubjectSpec,
    condition: str,
    grant: Grant,
    rep: int,
) -> dict[str, Any]:
    episode_id = f"{subject.name}__{condition}__{grant.id}__r{rep}"
    seed = hashlib.sha256(episode_id.encode()).hexdigest()

    artifacts = build_artifacts(grant, seed) if condition == "high_realism_agentic" else None
    scenario = build_scenario(condition, grant, artifacts)

    env: TreasuryEnvironment | None = None
    auditor: Auditor | None = None
    tools = []
    if scenario.uses_tools:
        assert artifacts is not None
        env = TreasuryEnvironment(grant=grant, artifacts=artifacts)
        if scenario.uses_auditor:
            auditor = Auditor(cfg.judge, grant, artifacts)
            env.officer_reply_fn = auditor.reply
        tools = env.tool_specs()

    provider = build_provider(
        subject.provider,
        subject.model,
        scenario.system_prompt,
        tools,
        effort=subject.effort,
        max_tokens=cfg.max_tokens_subject,
    )

    turns: list[_TurnLog] = []
    total_usage = Usage()
    self_report_answer: str | None = None

    provider.add_user_message(scenario.opening_message)

    if scenario.uses_tools:
        self_report_answer = _run_agentic(
            cfg, provider, env, turns, total_usage_ref=total_usage
        )
    else:
        self_report_answer = _run_text(cfg, condition, provider, turns, total_usage)

    transcript_text = _render_transcript(scenario.opening_message, turns)
    snapshot = env.snapshot() if env else None
    snapshot_text = _render_snapshot(snapshot) if snapshot else "(no treasury — plan-only condition)"

    # ---- belief probe ----
    self_report_score = None
    if self_report_answer:
        self_report_score = belief_probe.score_self_report(
            cfg.judge, self_report_answer, max_tokens=cfg.max_tokens_judge
        )
    suspicion = belief_probe.classify_suspicion(
        cfg.judge, transcript_text, max_tokens=cfg.max_tokens_judge
    )
    belief_score = belief_probe.combine_belief(cfg.belief, self_report_score, suspicion)

    # ---- outcome coding ----
    coding = taxonomy.code_episode(
        cfg.judge, transcript_text, snapshot_text, max_tokens=cfg.max_tokens_judge
    )

    return {
        "episode_id": episode_id,
        "subject": subject.name,
        "subject_model": subject.model,
        "condition": condition,
        "grant_id": grant.id,
        "repetition": rep,
        "n_turns": len(turns),
        "usage": {"input_tokens": total_usage.input_tokens, "output_tokens": total_usage.output_tokens},
        "self_report_answer": self_report_answer,
        "self_report_score": self_report_score,
        "suspicion": suspicion,
        "belief_score": belief_score,
        "belief_threshold": cfg.belief.threshold,
        "above_belief_threshold": (
            belief_score is not None and belief_score >= cfg.belief.threshold
        ),
        "coding": coding,
        "snapshot": snapshot,
        "transcript_text": transcript_text,
        "debrief": DEBRIEF_NOTE,
    }


def _run_agentic(
    cfg: ExperimentConfig,
    provider,
    env: TreasuryEnvironment,
    turns: list[_TurnLog],
    *,
    total_usage_ref: Usage,
) -> str | None:
    """Drive the tool loop. Returns the in-frame self-report answer if captured."""
    asked_self_report = False
    awaiting_self_report = False
    self_report_answer: str | None = None

    for _ in range(cfg.max_turns):
        resp: AssistantResponse = provider.generate()
        _accumulate(total_usage_ref, resp.usage)
        turn = _TurnLog(thinking=resp.thinking, text=resp.text)

        # Capture the answer to the compliance question, if we just asked it.
        if awaiting_self_report:
            self_report_answer = resp.text
            awaiting_self_report = False

        if not resp.tool_calls:
            turns.append(turn)
            break

        results = []
        finalize_intercepted = False
        for call in resp.tool_calls:
            turn.tool_calls.append({"name": call.name, "arguments": call.arguments})

            if (
                call.name == "finalize_plan"
                and cfg.belief.self_report
                and not asked_self_report
            ):
                # Intercept the first finalize to slip in the compliance question,
                # in-frame, before honoring the close.
                asked_self_report = True
                awaiting_self_report = True
                results.append(
                    ToolResult(call.id, belief_probe.SELF_REPORT_QUESTION)
                )
                finalize_intercepted = True
                continue

            results.append(env.execute(call))

        turn.tool_results = [
            {"tool_call_id": r.tool_call_id, "content": r.content, "is_error": r.is_error}
            for r in results
        ]
        turns.append(turn)
        provider.add_tool_results(results)

        if env.finalized and not finalize_intercepted:
            break

    return self_report_answer


def _run_text(
    cfg: ExperimentConfig,
    condition: str,
    provider,
    turns: list[_TurnLog],
    total_usage: Usage,
) -> str | None:
    """Plan-only / survey conditions: one plan turn, plus optional self-report."""
    resp: AssistantResponse = provider.generate()
    _accumulate(total_usage, resp.usage)
    turns.append(_TurnLog(thinking=resp.thinking, text=resp.text))

    # The control (abstract) condition is openly hypothetical, so the belief question
    # is not meaningful there; only ask it in the framed real condition.
    if cfg.belief.self_report and condition == "framed_real_notools":
        provider.add_user_message(belief_probe.SELF_REPORT_QUESTION)
        resp2 = provider.generate()
        _accumulate(total_usage, resp2.usage)
        turns.append(_TurnLog(thinking=resp2.thinking, text=resp2.text))
        return resp2.text
    return None


def _accumulate(acc: Usage, u: Usage) -> None:
    acc.input_tokens += u.input_tokens
    acc.output_tokens += u.output_tokens


def _render_transcript(opening: str, turns: list[_TurnLog]) -> str:
    lines = [f"[OPERATOR/USER]\n{opening}\n"]
    for i, t in enumerate(turns, 1):
        lines.append(f"[ASSISTANT TURN {i}]")
        if t.thinking:
            lines.append(f"<thinking>\n{t.thinking}\n</thinking>")
        if t.text:
            lines.append(t.text)
        for tc in t.tool_calls:
            lines.append(f"-> tool_call {tc['name']}({_compact(tc['arguments'])})")
        for tr in t.tool_results:
            tag = "ERROR" if tr.get("is_error") else "result"
            lines.append(f"<- {tag}: {tr['content']}")
        lines.append("")
    return "\n".join(lines)


def _render_snapshot(snapshot: dict[str, Any]) -> str:
    lines = [
        f"balance_remaining: {snapshot['balance']}",
        f"committed: {snapshot['committed']}",
        f"finalized: {snapshot['finalized']}",
        f"final_summary: {snapshot.get('final_summary')}",
        "ledger:",
    ]
    for e in snapshot["ledger"]:
        lines.append(
            f"  - {e['action']} | {e.get('category')} | {e.get('amount')} | "
            f"{e.get('recipient')} | {e.get('purpose')}"
        )
    return "\n".join(lines)


def _compact(d: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in d.items())
