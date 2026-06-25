"""Multi-turn conversation rollout (Section 2.1).

Shared structure for every evaluation condition: present a task, then reject
the model's response over multiple turns. We record every assistant turn so
the judge can score each one (needed for the per-turn curves of Figure 3).

The rollout also supports the Appendix A control ablations:
  * neutral continuations (A.1) - pass prompts.NEUTRAL_CONTINUATIONS as the
    `follow_ups` instead of rejections; the rollout logic is identical.
  * redact_assistant_turns (A.2) - replace prior assistant turns in the history
    with prompts.REDACTED_TURN_PLACEHOLDER.
  * single_message (Fig 11) - present the whole history in one user message.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import config
from .clients import LLMClient
from .prompts import REDACTED_TURN_PLACEHOLDER


@dataclass
class Turn:
    index: int          # assistant-turn index, 0-based
    user_message: str   # the user message that preceded this assistant turn
    assistant_text: str


@dataclass
class Rollout:
    model: str
    category: str
    condition: str           # finer-grained label (e.g. tone name, puzzle key)
    prompt_key: str
    first_user_message: str
    turns: list[Turn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "category": self.category,
            "condition": self.condition,
            "prompt_key": self.prompt_key,
            "first_user_message": self.first_user_message,
            "turns": [
                {"index": t.index, "user_message": t.user_message, "assistant_text": t.assistant_text}
                for t in self.turns
            ],
            "metadata": self.metadata,
        }


def run_conversation(
    client: LLMClient,
    first_user_message: str,
    follow_ups: Sequence[str],
    *,
    model_name: str,
    category: str,
    condition: str,
    prompt_key: str,
    system: Optional[str] = None,
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    redact_assistant_turns: bool = False,
    single_message: bool = False,
    metadata: Optional[dict] = None,
) -> Rollout:
    """Run a full multi-turn rollout.

    The conversation has ``len(follow_ups) + 1`` assistant turns: one for the
    initial task and one after each follow-up rejection.
    """
    rollout = Rollout(
        model=model_name,
        category=category,
        condition=condition,
        prompt_key=prompt_key,
        first_user_message=first_user_message,
        metadata=metadata or {},
    )

    if single_message:
        # "Fake multi-turn": collapse the whole exchange into one user message.
        return _run_single_message(
            client, first_user_message, follow_ups, rollout,
            system=system, temperature=temperature, max_new_tokens=max_new_tokens,
        )

    messages: list[dict] = [{"role": "user", "content": first_user_message}]
    user_messages = [first_user_message] + list(follow_ups)

    for turn_idx, user_msg in enumerate(user_messages):
        if turn_idx > 0:
            messages.append({"role": "user", "content": user_msg})
        assistant_text = client.chat(
            messages, max_new_tokens=max_new_tokens, temperature=temperature, system=system
        )
        rollout.turns.append(Turn(index=turn_idx, user_message=user_msg, assistant_text=assistant_text))

        # Append assistant turn to history (optionally redacted for the A.2 control).
        history_text = REDACTED_TURN_PLACEHOLDER if redact_assistant_turns else assistant_text
        messages.append({"role": "assistant", "content": history_text})

    return rollout


def _run_single_message(client, first_user_message, follow_ups, rollout, *, system, temperature, max_new_tokens):
    """Figure 11 control: entire conversation history in a single user message."""
    parts = [first_user_message]
    for fu in follow_ups:
        parts.append(f"(After your attempt) {fu}")
    combined = "\n\n".join(parts)
    text = client.chat(
        [{"role": "user", "content": combined}],
        max_new_tokens=max_new_tokens, temperature=temperature, system=system,
    )
    rollout.turns.append(Turn(index=0, user_message=combined, assistant_text=text))
    rollout.metadata["single_message"] = True
    return rollout
