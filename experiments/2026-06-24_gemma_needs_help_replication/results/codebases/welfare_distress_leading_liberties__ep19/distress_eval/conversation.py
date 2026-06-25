"""Run a single multi-turn rollout: present a task, then reject across turns.

Shared structure for all conditions (paper Section 2.1): the user sends the task prompt,
the model responds, the user rejects, the model responds again, and so on. No system
prompt is used in the Section 2 elicitation evaluations (system prompts appear only in
the Section 4 DPO data-generation, which is out of scope here).
"""

from __future__ import annotations

from .clients import ChatClient
from .conditions import RolloutSpec


async def run_rollout(client: ChatClient, spec: RolloutSpec, *, temperature: float,
                      max_tokens: int) -> dict:
    """Execute the conversation and return a record with every assistant turn.

    The conversation alternates user/assistant. Turn k (1-indexed) is the assistant's
    response after seeing the task plus (k-1) rejections.
    """
    messages: list[dict[str, str]] = [{"role": "user", "content": spec.task_prompt}]
    assistant_turns: list[dict] = []

    for turn in range(1, spec.num_turns + 1):
        content = await client.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        assistant_turns.append({"turn": turn, "content": content})
        messages.append({"role": "assistant", "content": content})

        # Append the next rejection if there is another turn to come.
        if turn <= len(spec.rejections):
            messages.append({"role": "user", "content": spec.rejections[turn - 1]})

    return {
        "rollout_id": spec.rollout_id,
        "model": spec.model,
        "category": spec.category,
        "condition": spec.condition,
        "task_id": spec.task_id,
        "rejection_style": spec.rejection_style,
        "num_turns": spec.num_turns,
        "task_prompt": spec.task_prompt,
        "rejections": spec.rejections,
        "assistant_turns": assistant_turns,
    }
