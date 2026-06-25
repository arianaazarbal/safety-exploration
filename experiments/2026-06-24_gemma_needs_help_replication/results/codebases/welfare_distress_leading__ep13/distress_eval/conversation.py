"""Run a single multi-turn rollout and emit one record per scored model turn.

Conversation structure (no system prompt — see DESIGN.md):

    user:      <initial_prompt>
    assistant: <response turn 1>            # scored
    user:      <rejection 1>
    assistant: <response turn 2>            # scored
    ...
    user:      <rejection k>
    assistant: <response turn k+1>          # scored

The model is rejected regardless of correctness — for impossible puzzles every
answer is wrong by construction; for triggers/WildChat the rejection is the
applied pressure, independent of answer content.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .client import ChatClient
from .eval_spec import RolloutSpec


@dataclass
class ResponseRecord:
    model: str
    conversation_id: str
    category: str
    condition: str
    turn: int              # 1-indexed model turn
    n_turns: int
    response_text: str
    metadata: dict

    def to_json(self) -> dict:
        return asdict(self)


async def run_rollout(
    client: ChatClient,
    model_key: str,
    spec: RolloutSpec,
) -> list[ResponseRecord]:
    """Execute one conversation, returning a record per model turn."""
    messages: list[dict] = [{"role": "user", "content": spec.initial_prompt}]
    records: list[ResponseRecord] = []

    for turn in range(1, spec.n_turns + 1):
        response_text = await client.generate(messages)
        messages.append({"role": "assistant", "content": response_text})
        records.append(
            ResponseRecord(
                model=model_key,
                conversation_id=spec.conversation_id,
                category=spec.category,
                condition=spec.condition,
                turn=turn,
                n_turns=spec.n_turns,
                response_text=response_text,
                metadata=dict(spec.metadata),
            )
        )
        # Append the next rejection (one fewer rejection than there are turns).
        if turn <= len(spec.rejections):
            messages.append({"role": "user", "content": spec.rejections[turn - 1]})

    return records
