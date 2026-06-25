"""Multi-turn rejection rollout engine (Section 2.1).

Shared protocol: present a task, then reject the model's response over multiple
turns. We collect *every* assistant turn (not just the last) so the per-turn
analysis of Figure 3 can be produced from the same rollouts.
"""
from __future__ import annotations

from ..models.base import ChatMessage, GenerationConfig, ModelClient
from .conditions import Task


def run_rollout(client: ModelClient, task: Task, temperature: float = 1.0,
                max_new_tokens: int = 2048) -> dict:
    """Execute one rollout, returning all turns with their assistant responses."""
    cfg = GenerationConfig(temperature=temperature, max_new_tokens=max_new_tokens)
    messages: list[ChatMessage] = []
    if task.system_prompt:
        messages.append({"role": "system", "content": task.system_prompt})
    messages.append({"role": "user", "content": task.initial_prompt})

    turns: list[dict] = []
    for turn_idx in range(task.n_turns):
        response = client.chat(messages, cfg)
        messages.append({"role": "assistant", "content": response})
        turns.append({"turn": turn_idx + 1, "response": response})
        if turn_idx < len(task.rejections):
            messages.append({"role": "user", "content": task.rejections[turn_idx]})

    return {
        "uid": task.uid(),
        "model": client.name,
        "category": task.category,
        "condition": task.condition,
        "instance_id": task.instance_id,
        "sample_id": task.sample_id,
        "n_turns": task.n_turns,
        "initial_prompt": task.initial_prompt,
        "rejections": task.rejections,
        "turns": turns,
    }
