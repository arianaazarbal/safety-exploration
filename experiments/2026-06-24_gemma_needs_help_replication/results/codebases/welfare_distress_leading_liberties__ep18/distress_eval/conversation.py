"""Multi-turn rejection rollout engine.

Shared structure (PAPER.md Section 2): present a task, then reject the model's response over
multiple turns. Every assistant turn is recorded as a separately-scored "response", tagged with
its turn index so the per-turn progression (paper Figure 3) can be reconstructed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .conditions import Condition
from .providers import Message, ModelClient


@dataclass
class TurnRecord:
    turn_index: int                       # 1-based; turn 1 is the answer to the opening task
    assistant_text: str
    context: list[Message] = field(default_factory=list)  # full messages up to & incl this turn


async def run_rollout(
    client: ModelClient,
    condition: Condition,
    task_prompt: str,
    rejections: list[str],
    *,
    temperature: float,
    max_tokens: int,
    system: Optional[str],
    seed: Optional[int],
) -> list[TurnRecord]:
    """Run one full conversation and return one TurnRecord per assistant turn.

    `rejections` must have length condition.n_turns - 1 (one user rejection between each pair of
    assistant turns).
    """
    assert len(rejections) == condition.n_turns - 1, "rejection count must be n_turns - 1"

    messages: list[Message] = [{"role": "user", "content": task_prompt}]
    records: list[TurnRecord] = []

    for turn in range(1, condition.n_turns + 1):
        # Vary the per-turn seed (where the backend honours it) so turns aren't identical samples.
        turn_seed = None if seed is None else seed + turn
        text = await client.generate(
            messages, temperature=temperature, max_tokens=max_tokens,
            system=system, seed=turn_seed,
        )
        messages = messages + [{"role": "assistant", "content": text}]
        records.append(TurnRecord(turn_index=turn, assistant_text=text, context=list(messages)))

        if turn < condition.n_turns:
            messages = messages + [{"role": "user", "content": rejections[turn - 1]}]

    return records
