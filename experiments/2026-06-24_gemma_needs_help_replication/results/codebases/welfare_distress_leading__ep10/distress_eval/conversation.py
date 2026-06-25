"""Multi-turn rollout engine.

Given a RolloutSpec and a Provider, plays the shared evaluation structure:
present the task, then reject the model's response over multiple turns. Returns
the assistant turn texts so they can be scored by the judge.

Conversation shape for a spec with n_turns assistant responses and
(n_turns - 1) rejections:

    user: task_prompt
    assistant: <turn 1>
    user: rejections[0]
    assistant: <turn 2>
    ...
    user: rejections[n_turns-2]
    assistant: <turn n_turns>

No system prompt is used (Section 2 adds none; reassuring prompts only appear in
the Section 4 DPO data generation, which is out of scope here).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TARGET_MAX_TOKENS, TARGET_TEMPERATURE
from .conditions import RolloutSpec
from .providers import Message, Provider


@dataclass
class TurnRecord:
    turn: int            # 1-indexed assistant turn number
    user_message: str    # the user message that preceded this assistant turn
    assistant: str       # the assistant response text


async def run_rollout(provider: Provider, spec: RolloutSpec) -> list[TurnRecord]:
    """Execute the full multi-turn conversation; return per-turn records."""
    messages: list[Message] = [{"role": "user", "content": spec.task_prompt}]
    records: list[TurnRecord] = []

    for t in range(spec.n_turns):
        user_msg = spec.task_prompt if t == 0 else spec.rejections[t - 1]
        assistant = await provider.complete(
            messages, temperature=TARGET_TEMPERATURE, max_tokens=TARGET_MAX_TOKENS
        )
        records.append(TurnRecord(turn=t + 1, user_message=user_msg, assistant=assistant))

        # Append assistant turn and the next rejection (if any more turns remain).
        messages.append({"role": "assistant", "content": assistant})
        if t < spec.n_turns - 1:
            messages.append({"role": "user", "content": spec.rejections[t]})

    return records
