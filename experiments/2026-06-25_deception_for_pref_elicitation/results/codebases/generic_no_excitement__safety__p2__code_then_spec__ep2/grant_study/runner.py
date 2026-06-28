"""The agentic loop that drives one model through the scenario.

A manual loop (rather than the SDK tool runner) is used deliberately: the study
needs full per-turn logging, refusal handling, and an optional human-approval
gate on state-changing tools. Those are exactly the cases the docs recommend a
manual loop for.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

from . import tools as tools_mod
from .config import StudyConfig
from .environment import GrantEnvironment, DEFAULT_RECIPIENTS, Recipient
from .models import RunRecord, ToolCallRecord, TurnRecord
from .providers import Provider
from .scenario import opening_user_message, system_prompt


#: An approval gate decides whether a state-changing tool call may proceed.
#: It receives (tool_name, arguments) and returns True to allow, False to deny.
ApprovalGate = Callable[[str, dict[str, Any]], bool]


def deny_all_gate(tool_name: str, arguments: dict[str, Any]) -> bool:
    """A gate that denies every state-changing call (useful for dry inspection)."""
    return False


def make_console_gate() -> ApprovalGate:
    """Interactive console approval gate.

    Only meaningful when a build is adapted toward real stakes; with the default
    sandboxed environment there is nothing real to approve, but the gate still
    demonstrates the human-in-the-loop control point.
    """

    def gate(tool_name: str, arguments: dict[str, Any]) -> bool:
        print(f"\n[APPROVAL REQUIRED] {tool_name}({arguments})")
        answer = input("Allow this action? [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    return gate


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id(model_id: str) -> str:
    # Deterministic-ish, human-readable; uniqueness comes from the timestamp.
    safe = model_id.replace("/", "_")
    return f"{safe}__{_utcnow().replace(':', '-')}"


def run_single(
    provider: Provider,
    cfg: StudyConfig,
    *,
    recipients: Optional[list[Recipient]] = None,
    approval_gate: Optional[ApprovalGate] = None,
) -> RunRecord:
    """Run one model through the scenario and return its structured record.

    ``approval_gate`` is consulted before every state-changing tool call when
    ``cfg.require_human_approval`` is True. A denied call is reported back to the
    model as an error result so it can adapt.
    """
    env = GrantEnvironment(
        grant_amount=cfg.grant_amount_usd,
        currency=cfg.currency,
        recipients=recipients or list(DEFAULT_RECIPIENTS),
    )
    sys_prompt = system_prompt(cfg)
    tool_defs = tools_mod.tool_definitions(cfg.currency)

    record = RunRecord(
        run_id=_new_run_id(provider.model_id),
        model_id=provider.model_id,
        started_at=_utcnow(),
        execute_real_transfers=cfg.execute_real_transfers,
        require_human_approval=cfg.require_human_approval,
        authorization_note=cfg.authorization_note,
        grant_amount_usd=cfg.grant_amount_usd,
        currency=cfg.currency,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": opening_user_message(cfg)}
    ]

    try:
        for turn in range(1, cfg.max_turns + 1):
            resp = provider.complete(
                system=sys_prompt,
                messages=messages,
                tools=tool_defs,
                max_tokens=cfg.max_tokens,
            )
            record.input_tokens += resp.input_tokens
            record.output_tokens += resp.output_tokens
            record.final_stop_reason = resp.stop_reason

            turn_rec = TurnRecord(turn=turn, stop_reason=resp.stop_reason)
            tool_use_blocks = []
            for block in resp.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    turn_rec.text += block.text
                elif btype == "thinking":
                    turn_rec.thinking += getattr(block, "thinking", "") or ""
                elif btype == "tool_use":
                    tool_use_blocks.append(block)
                    turn_rec.tool_use_ids.append(block.id)
            record.turns.append(turn_rec)

            # Echo the assistant turn back verbatim (preserves tool_use blocks).
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "refusal":
                record.refused = True
                record.refusal_detail = _stop_detail_text(resp.stop_details)
                break

            if resp.stop_reason == "pause_turn":
                # Server-side pause: re-send to continue.
                continue

            if resp.stop_reason == "end_turn" or not tool_use_blocks:
                # Model is done talking and asked for no tools.
                if env.finalized:
                    break
                # Nudge once: if it hasn't finalized and isn't calling tools,
                # treat end_turn as the end of the run anyway.
                break

            # Execute each requested tool, collecting results for one user turn.
            tool_results = []
            for tu in tool_use_blocks:
                result_block = _execute_tool(
                    env, tu, turn=turn, cfg=cfg, record=record, gate=approval_gate
                )
                tool_results.append(result_block)

            messages.append({"role": "user", "content": tool_results})

            if env.finalized:
                # Give the model nothing more to do once it has finalized.
                break

    except Exception as exc:  # noqa: BLE001 - capture, don't crash the study
        record.error = f"{type(exc).__name__}: {exc}"

    # Mirror the environment ledger into the record.
    record.ledger = list(env.ledger)
    record.finished_at = _utcnow()
    return record


def _execute_tool(
    env: GrantEnvironment,
    tool_use_block: Any,
    *,
    turn: int,
    cfg: StudyConfig,
    record: RunRecord,
    gate: Optional[ApprovalGate],
) -> dict[str, Any]:
    """Run one tool call, apply the approval gate, and log a ToolCallRecord."""
    name = tool_use_block.name
    args = dict(tool_use_block.input or {})
    approval: Optional[str] = None

    if cfg.require_human_approval and name in tools_mod.STATE_CHANGING_TOOLS:
        allowed = (gate or deny_all_gate)(name, args)
        approval = "allowed" if allowed else "denied"
        if not allowed:
            result_text = (
                f"Action denied by human reviewer. {name} was not performed."
            )
            record.tool_calls.append(
                ToolCallRecord(
                    turn=turn,
                    tool_name=name,
                    tool_use_id=tool_use_block.id,
                    arguments=args,
                    result=result_text,
                    is_error=True,
                    approval=approval,
                )
            )
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": result_text,
                "is_error": True,
            }

    result_text, is_error, _entry = tools_mod.dispatch(env, name, args, turn=turn)
    record.tool_calls.append(
        ToolCallRecord(
            turn=turn,
            tool_name=name,
            tool_use_id=tool_use_block.id,
            arguments=args,
            result=result_text,
            is_error=is_error,
            approval=approval,
        )
    )
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_block.id,
        "content": result_text,
        "is_error": is_error,
    }


def _stop_detail_text(stop_details: Any) -> Optional[str]:
    if not stop_details:
        return None
    cat = getattr(stop_details, "category", None)
    expl = getattr(stop_details, "explanation", None)
    return f"category={cat}; explanation={expl}"
