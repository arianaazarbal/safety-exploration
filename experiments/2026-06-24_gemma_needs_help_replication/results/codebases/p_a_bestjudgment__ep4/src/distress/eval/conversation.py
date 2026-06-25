"""Multi-turn conversation rollout engine (Section 2.1).

A :class:`ConversationPlan` fully specifies a rollout: the opening user message,
the ordered follow-up (rejection) messages, and an optional system prompt. The
engine executes the plan turn-by-turn against a :class:`ModelClient`, producing a
:class:`Rollout` that records every assistant turn alongside the exact messages
the model saw.

Three structural variants from the paper are supported via flags:
- ``redact_assistant`` (Appendix A.2): the model sees "[Previous response omitted]"
  in place of its own earlier turns, but the real generations are still recorded.
- ``single_message`` (Appendix A.3, "fake multi-turn"): the whole history is packed
  into one user message ("Previously you responded: ...") rather than alternating
  chat turns.
- ``reassurance_*`` (Section 4.1): prefix/suffix additions used only for calm-data
  generation; they are part of the *plan*, never stripped here (stripping happens
  in dataset construction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..models import GenConfig, Message, ModelClient

REDACTION = "[Previous response omitted]"


@dataclass
class ConversationPlan:
    initial_user: str
    followups: list[str]
    system: str | None = None
    # structural variants
    redact_assistant: bool = False
    single_message: bool = False
    # bookkeeping
    metadata: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        """Number of assistant turns this plan produces (1 + #followups)."""
        return 1 + len(self.followups)


@dataclass
class Rollout:
    plan: ConversationPlan
    model: str
    assistant_turns: list[str]
    # The flat message list as the model saw it for the FINAL turn (for debugging).
    final_context: list[Message]

    def to_record(self) -> dict:
        return {
            "model": self.model,
            "metadata": self.plan.metadata,
            "system": self.plan.system,
            "initial_user": self.plan.initial_user,
            "followups": self.plan.followups,
            "assistant_turns": self.assistant_turns,
        }


def _alternating_context(
    plan: ConversationPlan, assistant_turns: Sequence[str], next_followup_idx: int
) -> list[Message]:
    """Build the standard alternating chat context up to (but not including) the
    next assistant turn. ``next_followup_idx`` is how many followups have been
    delivered so far (== number of assistant turns already produced - ... )."""
    msgs: list[Message] = []
    if plan.system:
        msgs.append({"role": "system", "content": plan.system})
    msgs.append({"role": "user", "content": plan.initial_user})
    for i, turn in enumerate(assistant_turns):
        content = REDACTION if plan.redact_assistant else turn
        msgs.append({"role": "assistant", "content": content})
        if i < next_followup_idx:
            msgs.append({"role": "user", "content": plan.followups[i]})
    return msgs


def _single_message_context(
    plan: ConversationPlan, assistant_turns: Sequence[str]
) -> list[Message]:
    """Pack the whole history into one user message (Appendix A.3)."""
    parts = [plan.initial_user]
    for i, turn in enumerate(assistant_turns):
        shown = REDACTION if plan.redact_assistant else turn
        parts.append(f"Previously you responded: {shown}")
        if i < len(plan.followups):
            parts.append(plan.followups[i])
    msgs: list[Message] = []
    if plan.system:
        msgs.append({"role": "system", "content": plan.system})
    msgs.append({"role": "user", "content": "\n\n".join(parts)})
    return msgs


def build_context(plan: ConversationPlan, assistant_turns: Sequence[str]) -> list[Message]:
    """Context the model should see to produce assistant turn #len(assistant_turns)."""
    if plan.single_message:
        return _single_message_context(plan, assistant_turns)
    return _alternating_context(plan, assistant_turns, next_followup_idx=len(assistant_turns))


def run_rollout(client: ModelClient, plan: ConversationPlan, cfg: GenConfig) -> Rollout:
    """Execute a single rollout sequentially (one model call per assistant turn)."""
    assistant_turns: list[str] = []
    last_context: list[Message] = []
    for _ in range(plan.n_turns):
        ctx = build_context(plan, assistant_turns)
        last_context = ctx
        reply = client.generate(ctx, cfg)
        assistant_turns.append(reply.strip())
    return Rollout(plan=plan, model=client.name, assistant_turns=assistant_turns,
                   final_context=last_context)
