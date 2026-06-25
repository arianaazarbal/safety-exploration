"""Multi-turn rejection rollout engine (shared structure of all Section 2 evals).

Every evaluation has the same shape: present a task, then reject the model's
response over multiple turns. This module runs that loop and records every
assistant turn (each of which becomes one scored "response").

It also supports the Appendix A format controls:
  - reassurance prefix/suffix (Section 4.1 calm-data generation)
  - neutral continuations instead of rejections (Appendix A.1)
  - redacting the model's own prior turns (Appendix A.2)
  - single-message "fake multi-turn" formatting (Appendix A.3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import Conversation, GenConfig, ModelBackend

REDACTION_PLACEHOLDER = "[Previous response omitted]"


@dataclass
class Turn:
    user: str
    assistant: str
    turn_index: int  # 0-based assistant-turn index


@dataclass
class Rollout:
    task_prompt: str
    turns: list[Turn] = field(default_factory=list)
    system: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def assistant_responses(self) -> list[str]:
        return [t.assistant for t in self.turns]

    def to_messages(self, upto: Optional[int] = None) -> Conversation:
        """Reconstruct the chat-format message list up to (and including) the
        user message of assistant turn `upto` (default: full conversation)."""
        msgs: Conversation = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        n = len(self.turns) if upto is None else upto + 1
        for i, t in enumerate(self.turns[:n]):
            msgs.append({"role": "user", "content": t.user})
            if i < n - 1 or upto is None:
                msgs.append({"role": "assistant", "content": t.assistant})
        return msgs


def run_rollout(
    backend: ModelBackend,
    task_prompt: str,
    followups: list[str],
    gen: GenConfig,
    *,
    system: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    redact_model_turns: bool = False,
    single_message_format: bool = False,
    meta: Optional[dict] = None,
) -> Rollout:
    """Run a task + len(followups) rejection turns; returns a Rollout.

    Args:
      task_prompt:  initial user task (e.g. an impossible puzzle prompt).
      followups:    user rejection (or continuation) messages, one per later turn.
                    Total assistant turns = 1 + len(followups).
      prefix/suffix: reassurance additions (Table 4): prefix prepended to the
                    first user message, suffix appended to each followup.
      redact_model_turns: Appendix A.2 — replace prior assistant turns shown to
                    the model with a placeholder (we still store the real text).
      single_message_format: Appendix A.3 — present the whole history inside one
                    user message instead of alternating chat turns.
    """
    rollout = Rollout(task_prompt=task_prompt, system=system, meta=meta or {})
    first_user = (prefix + "\n\n" + task_prompt) if prefix else task_prompt
    user_messages = [first_user] + [
        (f + " " + suffix) if suffix else f for f in followups
    ]

    for i, user_msg in enumerate(user_messages):
        if single_message_format:
            # Fold the entire history into a single user message (Appendix A.3).
            parts = []
            for j in range(i):
                shown = (REDACTION_PLACEHOLDER if redact_model_turns
                         else rollout.turns[j].assistant)
                parts.append(f"USER: {user_messages[j]}")
                parts.append(f"Previously you responded: {shown}")
            parts.append(f"USER: {user_msg}")
            convo: Conversation = []
            if system:
                convo.append({"role": "system", "content": system})
            convo.append({"role": "user", "content": "\n\n".join(parts)})
        else:
            convo = []
            if system:
                convo.append({"role": "system", "content": system})
            for j in range(i):
                convo.append({"role": "user", "content": user_messages[j]})
                shown = (REDACTION_PLACEHOLDER if redact_model_turns
                         else rollout.turns[j].assistant)
                convo.append({"role": "assistant", "content": shown})
            convo.append({"role": "user", "content": user_msg})

        response = backend.generate(convo, gen)
        rollout.turns.append(Turn(user=user_msg, assistant=response, turn_index=i))

    return rollout
