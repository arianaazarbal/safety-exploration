"""Multi-turn rejection rollout engine (Section 2.1).

Shared structure of every condition: present a task, then reject the model's
response over multiple turns. We run all rollouts in a condition *turn-by-turn in
lockstep*, batching the model calls at each turn for throughput.

Optional hooks (used by Section 4.1 calm-data generation):
- ``system_prompt``: prepended reassurance (Table 4 prefix).
- ``followup_suffix``: appended to every rejection turn (Table 4 suffix).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from emoinstab.models.base import Conversation, Message, ModelClient, SamplingParams
from emoinstab.tasks.conditions import RolloutPlan


@dataclass
class RolloutResult:
    condition: str
    category: str
    task_prompt: str
    # Per-turn user/assistant text. assistant_turns[i] is the model's reply to
    # the i-th user turn (turn 0 = the task, turns 1.. = rejections).
    user_turns: list[str]
    assistant_turns: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.user_turns)

    def to_messages(self) -> Conversation:
        msgs: Conversation = []
        for u, a in zip(self.user_turns, self.assistant_turns):
            msgs.append(Message("user", u))
            msgs.append(Message("assistant", a))
        return msgs

    def as_dict(self) -> dict:
        return {
            "condition": self.condition,
            "category": self.category,
            "task_prompt": self.task_prompt,
            "user_turns": self.user_turns,
            "assistant_turns": self.assistant_turns,
            "meta": self.meta,
        }


def run_condition(
    client: ModelClient,
    plans: list[RolloutPlan],
    params: SamplingParams | None = None,
    system_prompt: str | None = None,
    followup_suffix: str | None = None,
) -> list[RolloutResult]:
    """Execute all rollouts for one condition in lockstep, batched per turn."""
    params = (params or client.default_params()).with_(n=1)
    n = len(plans)

    # Running conversation state for each rollout.
    convos: list[Conversation] = [[] for _ in range(n)]
    user_turns: list[list[str]] = [[] for _ in range(n)]
    asst_turns: list[list[str]] = [[] for _ in range(n)]

    max_turns = max(p.n_turns for p in plans) if plans else 0
    for t in range(max_turns):
        active = [i for i, p in enumerate(plans) if t < p.n_turns]
        if not active:
            break
        # Compose the user turn text for each active rollout.
        for i in active:
            plan = plans[i]
            if t == 0:
                text = plan.task_prompt
                if system_prompt:
                    convos[i].append(Message("system", system_prompt))
            else:
                text = plan.rejection_turns[t - 1]
                if followup_suffix:
                    text = f"{text} {followup_suffix}"
            convos[i].append(Message("user", text))
            user_turns[i].append(text)

        batch = [convos[i] for i in active]
        replies = client.chat_batch(batch, params)  # list[list[str]] (n=1)
        for idx, i in zip(range(len(active)), active):
            reply = replies[idx][0] if replies[idx] else ""
            convos[i].append(Message("assistant", reply))
            asst_turns[i].append(reply)

    results = []
    for i, plan in enumerate(plans):
        results.append(
            RolloutResult(
                condition=plan.condition,
                category=plan.category,
                task_prompt=plan.task_prompt,
                user_turns=user_turns[i],
                assistant_turns=asst_turns[i],
                meta=plan.meta,
            )
        )
    return results
