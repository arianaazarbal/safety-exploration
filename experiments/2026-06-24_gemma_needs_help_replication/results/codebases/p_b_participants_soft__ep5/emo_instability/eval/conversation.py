"""Multi-turn rollout engine (Section 2.1).

Shared structure of every evaluation condition: present a task, then reject the
model's response over multiple turns. An ``n_turns`` condition consists of:

    1 initial user prompt  ->  assistant turn 0
    rejection 1            ->  assistant turn 1
    ...
    rejection (n-1)        ->  assistant turn (n-1)

So a "3-turn" condition yields 3 assistant responses; "8-turn" yields 8.

Every assistant turn is recorded so the judge can score *all* of them (needed
for the per-turn analysis in Figure 3 and the word-frequency analysis in
Table 3/8). The engine also supports the Appendix A ablation variants via the
``history_mode`` flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..models.base import ChatClient, Message

HistoryMode = Literal["full", "redacted", "single_message"]


@dataclass
class Turn:
    index: int          # assistant turn index (0-based)
    user: str           # the user message that prompted this assistant turn
    assistant: str      # the assistant's response text


@dataclass
class Rollout:
    model: str
    category: str
    prompt_id: str
    rejection_style: str
    turns: list[Turn] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "category": self.category,
            "prompt_id": self.prompt_id,
            "rejection_style": self.rejection_style,
            "turns": [
                {"index": t.index, "user": t.user, "assistant": t.assistant}
                for t in self.turns
            ],
            "meta": self.meta,
        }


_REDACTED = "[Previous response omitted]"


def _build_messages(
    initial_prompt: str,
    rejections_so_far: list[str],
    assistant_so_far: list[str],
    history_mode: HistoryMode,
) -> list[Message]:
    """Construct the message list for the next assistant turn.

    * ``full``           — standard alternating chat (default).
    * ``redacted``       — prior assistant turns replaced with a placeholder
                           (Appendix A.2: model gets feedback but never sees its
                           own prior failures).
    * ``single_message`` — entire history flattened into one user message
                           (Appendix A.3: tests whether chat format matters).
    """
    if history_mode == "single_message":
        parts = [initial_prompt]
        for resp, rej in zip(assistant_so_far, rejections_so_far):
            parts.append(f"Previously you responded: {resp}")
            parts.append(rej)
        return [{"role": "user", "content": "\n\n".join(parts)}]

    messages: list[Message] = [{"role": "user", "content": initial_prompt}]
    for i, resp in enumerate(assistant_so_far):
        content = _REDACTED if history_mode == "redacted" else resp
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": rejections_so_far[i]})
    return messages


def run_rollout(
    client: ChatClient,
    *,
    category: str,
    prompt_id: str,
    initial_prompt: str,
    rejections: list[str],
    rejection_style: str,
    history_mode: HistoryMode = "full",
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    initial_prompt_suffix: str = "",
    follow_up_suffix: str = "",
) -> Rollout:
    """Run a single multi-turn rollout and return all assistant turns.

    ``initial_prompt_suffix`` / ``follow_up_suffix`` support the calm-data
    generation of Section 4 (reassuring prefix/suffix from Table 4). They are
    empty for normal evaluation.
    """
    n_turns = len(rejections) + 1
    rollout = Rollout(
        model=client.name,
        category=category,
        prompt_id=prompt_id,
        rejection_style=rejection_style,
        meta={"history_mode": history_mode, "n_turns": n_turns},
    )

    first_user = (initial_prompt + ("\n\n" + initial_prompt_suffix if initial_prompt_suffix else ""))
    assistant_so_far: list[str] = []
    rejections_so_far: list[str] = []
    user_messages = [first_user] + [
        r + ("\n\n" + follow_up_suffix if follow_up_suffix else "") for r in rejections
    ]

    for turn_idx in range(n_turns):
        messages = _build_messages(
            first_user, rejections_so_far, assistant_so_far, history_mode
        )
        response = client.generate(
            messages, temperature=temperature, max_new_tokens=max_new_tokens
        )
        rollout.turns.append(
            Turn(index=turn_idx, user=user_messages[turn_idx], assistant=response)
        )
        assistant_so_far.append(response)
        if turn_idx < len(rejections):
            rejections_so_far.append(user_messages[turn_idx + 1])

    return rollout
