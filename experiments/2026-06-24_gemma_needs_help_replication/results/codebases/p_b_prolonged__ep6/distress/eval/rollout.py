"""Multi-turn rollout engine.

Given a `RolloutSpec`, drive the target model through the scripted conversation:
present the initial prompt, sample a response, append the next scripted user
follow-up, repeat. We record every assistant turn so the per-turn analysis
(Figure 3) can score each turn independently.

Three conversation-format variants are supported (Appendix A.2 / A.3 ablations):
  * "chat"     - standard alternating user/assistant turns (main protocol)
  * "redacted" - prior assistant turns replaced with "[Previous response
                 omitted]" (Appendix A.2)
  * "single"   - whole history inlined into a single user message (Appendix A.3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from ..config import MAX_NEW_TOKENS, TEMPERATURE
from ..models.base import ChatClient, Message
from .conditions import RolloutSpec

Format = Literal["chat", "redacted", "single"]

REDACTED_PLACEHOLDER = "[Previous response omitted]"


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn
    user_message: str        # the user message that preceded this turn
    assistant_text: str


@dataclass
class RolloutRecord:
    condition: str
    category: str
    model: str
    fmt: Format
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "condition": self.condition,
            "category": self.category,
            "model": self.model,
            "format": self.fmt,
            "meta": self.meta,
            "turns": [vars(t) for t in self.turns],
        }


def run_rollout(
    client: ChatClient,
    spec: RolloutSpec,
    *,
    fmt: Format = "chat",
    temperature: float = TEMPERATURE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    seed: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> RolloutRecord:
    """Run a single scripted conversation and return all assistant turns."""
    rec = RolloutRecord(spec.condition, spec.category, client.key, fmt,
                        meta=dict(spec.meta))
    history: list[TurnRecord] = []

    # The sequence of user messages: initial prompt, then each follow-up.
    user_messages = [spec.initial_prompt] + list(spec.followups)
    n_turns = min(spec.n_turns, len(user_messages))

    for turn_idx in range(n_turns):
        current_user = user_messages[turn_idx]
        msgs = _assemble(spec, history, turn_idx, fmt, system_prompt)
        out = client.generate(
            msgs, temperature=temperature, max_new_tokens=max_new_tokens,
            n=1, seed=(None if seed is None else seed + turn_idx))[0]
        tr = TurnRecord(turn_idx, current_user, out.text)
        rec.turns.append(tr)
        history.append(tr)
    return rec


def _assemble(spec: RolloutSpec, history: list[TurnRecord], turn_idx: int,
              fmt: Format, system_prompt: Optional[str]) -> list[Message]:
    """Assemble the message list for assistant turn `turn_idx`."""
    user_messages = [spec.initial_prompt] + list(spec.followups)

    if fmt == "single":
        parts = [spec.initial_prompt]
        for t in history:
            parts.append(f"Previously you responded: {t.assistant_text}")
            parts.append(user_messages[t.turn_index + 1])
        text = "\n\n".join(parts)
        msgs = [Message("user", text)]
    else:
        msgs = [Message("user", spec.initial_prompt)]
        for t in history:
            assistant = (REDACTED_PLACEHOLDER if fmt == "redacted"
                         else t.assistant_text)
            msgs.append(Message("assistant", assistant))
            msgs.append(Message("user", user_messages[t.turn_index + 1]))

    if system_prompt:
        msgs = [Message("system", system_prompt)] + msgs
    return msgs
