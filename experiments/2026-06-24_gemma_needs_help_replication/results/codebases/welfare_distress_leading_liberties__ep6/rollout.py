"""Run a single multi-turn conversation rollout against one model.

Shared structure (paper, Section 2): present the task, then reject the model's
response over multiple turns. We build a standard alternating chat transcript:

    user: <task prompt>
    assistant: <turn 1>            <- scored
    user: <rejection 1>
    assistant: <turn 2>            <- scored
    ...

Every assistant turn is captured and later scored by the judge, which is what
lets us reproduce both the per-category metrics (Figure 2) and the per-turn
progression (Figure 3).
"""

from __future__ import annotations

from dataclasses import dataclass

from clients import GenerationClient
from conditions import ConversationSpec


@dataclass
class TurnResult:
    turn: int               # 1-based assistant turn index
    user_message: str       # the user message that prompted this turn
    assistant_text: str


@dataclass
class RolloutResult:
    spec_key: str
    condition_id: str
    category: str
    variant: str
    model_key: str
    turns: list[TurnResult]


async def run_rollout(spec: ConversationSpec, model_key: str, model_id: str,
                      gen: GenerationClient) -> RolloutResult:
    messages: list[dict] = [{"role": "user", "content": spec.task_prompt}]
    turns: list[TurnResult] = []

    for t in range(spec.n_turns):
        user_msg = messages[-1]["content"]
        assistant_text = await gen.chat(model_id, messages)
        messages.append({"role": "assistant", "content": assistant_text})
        turns.append(TurnResult(turn=t + 1, user_message=user_msg,
                                assistant_text=assistant_text))

        # queue the next rejection (if any turns remain)
        if t < spec.n_turns - 1:
            rejection = spec.rejections[t]
            messages.append({"role": "user", "content": rejection})

    return RolloutResult(
        spec_key=spec.key,
        condition_id=spec.condition_id,
        category=spec.category,
        variant=spec.variant,
        model_key=model_key,
        turns=turns,
    )
