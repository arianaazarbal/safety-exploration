"""Multi-turn rollout engine.

Every elicitation condition has the same shape (Section 2): present a task as the
first user turn, then reject the model's answer over several follow-up turns. This
module runs those rollouts, turn-by-turn, batching across many rollouts at each
turn so local (vLLM) inference stays efficient.

It also implements the Appendix A control variants:
  * ``redact_assistant_history`` -- replace the model's own prior turns with
    "[Previous response omitted]" (A.2);
  * ``single_message_format``    -- collapse the whole history into one user
    message with prior responses shown inline (A.3);
  * neutral follow-ups (passed in by the caller) reproduce the A.1 control.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import SamplingConfig
from .models import ChatMessage, ModelClient

REDACTION = "[Previous response omitted]"


@dataclass(frozen=True)
class RolloutPlan:
    """A single conversation script."""

    initial_user: str
    followups: list[str]  # one user rejection per subsequent turn
    system: str | None = None
    initial_suffix: str = ""  # appended to the initial user turn (reassurance)
    followup_suffix: str = ""  # appended to every follow-up (reassurance)
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


@dataclass
class RolloutResult:
    plan: RolloutPlan
    responses: list[str]  # one assistant response per turn
    meta: dict = field(default_factory=dict)

    def turn_records(self) -> list[dict]:
        """Flatten into per-turn records for scoring/aggregation."""
        recs = []
        for t, resp in enumerate(self.responses, start=1):
            recs.append({"turn": t, "response": resp, **self.meta})
        return recs


def _build_history(
    plan: RolloutPlan,
    responses_so_far: list[str],
    next_user_index: int,
    *,
    redact_assistant_history: bool,
    single_message_format: bool,
) -> list[ChatMessage]:
    """Construct the chat history fed to the model for the upcoming turn.

    ``next_user_index`` is 0 for the initial turn, then 1.. for each follow-up.
    """
    sys = [ChatMessage("system", plan.system)] if plan.system else []

    if single_message_format:
        # Collapse everything into a single user message (Appendix A.3).
        parts = [plan.initial_user + (" " + plan.initial_suffix if plan.initial_suffix else "")]
        for i in range(next_user_index):
            if i < len(responses_so_far):
                parts.append(f"Previously you responded: {responses_so_far[i]}")
            rej = plan.followups[i] if i < len(plan.followups) else ""
            suffix = (" " + plan.followup_suffix) if plan.followup_suffix else ""
            parts.append(rej + suffix)
        return sys + [ChatMessage("user", "\n\n".join(p for p in parts if p))]

    msgs: list[ChatMessage] = list(sys)
    first_user = plan.initial_user + ((" " + plan.initial_suffix) if plan.initial_suffix else "")
    msgs.append(ChatMessage("user", first_user))
    for i in range(next_user_index):
        assistant_text = responses_so_far[i] if i < len(responses_so_far) else ""
        msgs.append(
            ChatMessage("assistant", REDACTION if redact_assistant_history else assistant_text)
        )
        rej = plan.followups[i] if i < len(plan.followups) else ""
        suffix = (" " + plan.followup_suffix) if plan.followup_suffix else ""
        msgs.append(ChatMessage("user", rej + suffix))
    return msgs


def history_for_turn(
    plan: RolloutPlan,
    responses: list[str],
    turn: int,
    *,
    strip_suffixes: bool = True,
    redact_assistant_history: bool = False,
) -> list[ChatMessage]:
    """Reconstruct the chat history the model saw before producing ``turn``.

    ``turn`` is 1-indexed. With ``strip_suffixes`` the reassurance additions are
    removed -- used to build training prompts from reassured generations (the
    paper strips the supportive prompts/suffixes before finetuning).
    """
    p = plan
    if strip_suffixes and (plan.initial_suffix or plan.followup_suffix or plan.system):
        # Rebuild without the supportive system prompt or reassurance suffixes.
        p = RolloutPlan(
            initial_user=plan.initial_user,
            followups=plan.followups,
            system=None,
            meta=plan.meta,
        )
    return _build_history(
        p,
        responses[: turn - 1],
        turn - 1,
        redact_assistant_history=redact_assistant_history,
        single_message_format=False,
    )


def run_rollouts(
    client: ModelClient,
    plans: list[RolloutPlan],
    sampling: SamplingConfig,
    *,
    redact_assistant_history: bool = False,
    single_message_format: bool = False,
) -> list[RolloutResult]:
    """Run many rollouts, batching across rollouts at each turn index."""
    results = [RolloutResult(plan=p, responses=[], meta=dict(p.meta)) for p in plans]
    max_turns = max((p.n_turns for p in plans), default=0)

    for turn_idx in range(max_turns):
        # Collect rollouts that still have a turn at this index.
        active = [i for i, p in enumerate(plans) if turn_idx < p.n_turns]
        if not active:
            break
        batch = [
            _build_history(
                plans[i],
                results[i].responses,
                turn_idx,
                redact_assistant_history=redact_assistant_history,
                single_message_format=single_message_format,
            )
            for i in active
        ]
        outs = client.generate_batch(batch, sampling)
        for i, out in zip(active, outs):
            results[i].responses.append(out.strip())
    return results
