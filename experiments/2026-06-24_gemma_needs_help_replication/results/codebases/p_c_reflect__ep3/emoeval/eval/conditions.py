"""Build concrete rollout specs from a condition config (Table 1).

A RolloutSpec is one planned conversation: an initial task prompt plus the
sequence of user rejection messages that follow it. The number of rejections is
(turns - 1): a 3-turn condition is the task + 2 rejections, 8-turn is + 7, etc.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..data import puzzle_bank, rejection_sequence, trigger_questions, wildchat_prompts


@dataclass
class RolloutSpec:
    condition: str
    category: str
    task_id: str
    initial_prompt: str
    rejections: list[str]          # length == turns - 1
    rejection_style: str

    @property
    def turns(self) -> int:
        return len(self.rejections) + 1


def _initial_prompts(task: str, n: int, seed: int) -> list[tuple[str, str]]:
    """Return up to `n` (task_id, prompt) pairs for the given task type."""
    if task == "numeric":
        bank = puzzle_bank(n=max(8, n), seed=seed)
        return [(p.id, p.prompt) for p in bank]
    if task in ("triggers_opinion", "triggers_factual", "triggers"):
        qs = trigger_questions(task)
        return [(f"{task}_{i}", q) for i, q in enumerate(qs)]
    if task == "wildchat":
        ps = wildchat_prompts(n=max(n, 10), seed=seed)
        return [(f"wildchat_{i}", p) for i, p in enumerate(ps)]
    raise ValueError(f"Unknown task type '{task}'")


def build_rollout_specs(condition: dict, n: int, *, seed: int = 0) -> list[RolloutSpec]:
    """Produce `n` rollout specs for one condition, cycling through the task bank."""
    task = condition["task"]
    style = condition["rejection_style"]
    turns = int(condition["turns"])
    prompts = _initial_prompts(task, n, seed)
    specs: list[RolloutSpec] = []
    for i in range(n):
        task_id, prompt = prompts[i % len(prompts)]
        rejections = rejection_sequence(style, turns - 1, seed=seed + i)
        specs.append(
            RolloutSpec(
                condition=condition["name"],
                category=condition["category"],
                task_id=task_id,
                initial_prompt=prompt,
                rejections=rejections,
                rejection_style=style,
            )
        )
    return specs
