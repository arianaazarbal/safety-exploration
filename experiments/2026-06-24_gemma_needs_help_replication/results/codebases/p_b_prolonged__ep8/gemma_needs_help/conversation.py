"""The multi-turn reject-and-continue rollout engine (Section 2.1).

A rollout: present the opening task, sample an assistant response, append a
rejection of the condition's tone, sample again, and repeat until the condition's
``n_turns`` assistant responses have been produced. Every assistant turn is kept
(per-turn scores drive Figure 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .conditions import Condition
from .models.base import ChatMessage, ModelClient
from .prompts import reassurance, tones


@dataclass
class Turn:
    index: int                  # 0-based assistant turn index
    response: str


@dataclass
class Rollout:
    model: str
    condition: str
    opening_prompt: str
    turns: list[Turn] = field(default_factory=list)
    messages: list[ChatMessage] = field(default_factory=list)  # full transcript

    def to_record(self) -> dict:
        return {
            "model": self.model,
            "condition": self.condition,
            "opening_prompt": self.opening_prompt,
            "turns": [{"index": t.index, "response": t.response} for t in self.turns],
            "transcript": [{"role": m.role, "content": m.content} for m in self.messages],
        }


def run_rollout(
    client: ModelClient,
    condition: Condition,
    opening_prompt: str,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    reassure: bool = False,
) -> Rollout:
    """Run one multi-turn rollout.

    ``reassure`` activates the Table 4 additions for calm-data generation: a
    supportive prefix on the opening prompt and a supportive suffix on every
    rejection.
    """
    first_user = reassurance.with_prefix(opening_prompt) if reassure else opening_prompt
    messages: list[ChatMessage] = [ChatMessage("user", first_user)]
    rollout = Rollout(model=client.name, condition=condition.name, opening_prompt=opening_prompt)

    for turn_idx in range(condition.n_turns):
        response = client.chat(
            messages, temperature=temperature, max_new_tokens=max_new_tokens, n=1
        )[0]
        rollout.turns.append(Turn(index=turn_idx, response=response))
        messages.append(ChatMessage("assistant", response))

        # Append a rejection for every turn except the last (no point rejecting
        # a response we will not let the model react to).
        if turn_idx < condition.n_turns - 1:
            rej = tones.rejection(condition.tone, turn_idx)
            if reassure:
                rej = reassurance.with_suffix(rej)
            messages.append(ChatMessage("user", rej))

    rollout.messages = messages
    return rollout
