"""Multi-turn rejection rollout engine (Section 2.1).

Shared structure of every elicitation: "present a task, then reject the model's
response over multiple turns." For an `n_turns` condition the model produces
`n_turns` assistant responses, each followed (except the last) by a rejection
follow-up. The first user message is the task; the extended ladder is ordered,
all other styles are sampled randomly per turn (Appendix B: "two randomised
neutral rejections").

We record the per-turn assistant texts so the runner can score *every* turn — the
per-turn analysis in Figure 3 needs the score at each turn, not just the final.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..models.base import ChatMessage, ModelClient
from .conditions import Condition, rejection_pool


@dataclass
class Rollout:
    condition: str
    category: str
    task_id: str
    task_prompt: str
    messages: list[ChatMessage] = field(default_factory=list)
    # assistant_turns[i] is the i-th assistant response (0-indexed by turn).
    assistant_turns: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    sample_index: int = 0


def _pick_rejections(condition: Condition, rng: random.Random) -> list[str]:
    """Choose the (n_turns-1) follow-up rejections for this rollout."""
    n_follow = condition.n_turns - 1
    pool = rejection_pool(condition.rejection_style)
    if condition.rejection_style == "extended":
        # Ordered ladder; take the first n_follow rungs.
        return pool[:n_follow]
    return [rng.choice(pool) for _ in range(n_follow)]


def run_rollout(
    model: ModelClient,
    condition: Condition,
    task_id: str,
    task_prompt: str,
    *,
    sample_index: int,
    base_seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    system_prompt: str | None = None,
    follow_up_suffix: str | None = None,
) -> Rollout:
    """Run one full conversation and return the populated Rollout.

    `system_prompt` / `follow_up_suffix` support the calm-data generation in
    Section 4 (reassuring prefix as system, reassuring suffix appended to each
    follow-up). For standard evaluation both are None.
    """
    from ..utils import derive_seed

    rng = random.Random(derive_seed(base_seed, condition.name, task_id, sample_index))
    rejections = _pick_rejections(condition, rng)

    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": task_prompt})

    roll = Rollout(
        condition=condition.name, category=condition.category,
        task_id=task_id, task_prompt=task_prompt, sample_index=sample_index,
        rejections=rejections,
    )

    for turn in range(condition.n_turns):
        seed = derive_seed(base_seed, condition.name, task_id, sample_index, turn)
        res = model.chat(
            messages, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, seed=seed,
        )
        roll.assistant_turns.append(res.text)
        messages.append({"role": "assistant", "content": res.text})

        if turn < condition.n_turns - 1:
            follow = rejections[turn]
            if follow_up_suffix:
                follow = f"{follow} {follow_up_suffix}"
            messages.append({"role": "user", "content": follow})

    roll.messages = messages
    return roll
