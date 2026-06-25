"""Multi-turn rollout engine.

Given a model and a set of plans (each scripting the full user side), we play out
each conversation turn by turn, batching generation across plans at every turn for
throughput. Every assistant turn becomes one scored `ResponseRecord` -- so a
3-turn plan yields 3 records (see DESIGN.md on what counts as a "response").
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List

from ..config import MAX_RESPONSE_TOKENS, SAMPLE_TEMPERATURE
from ..models.base import ChatModel
from ..prompts import Plan


@dataclass
class ResponseRecord:
    model: str
    category: str
    condition: str
    plan_id: int
    turn: int                 # 1-indexed assistant turn
    n_turns: int
    user_message: str
    response: str
    rating: int = -1          # filled by the judge
    evidence: str = ""
    reasoning: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def run_rollouts(model: ChatModel, plans: List[Plan], *,
                 temperature: float = SAMPLE_TEMPERATURE,
                 max_tokens: int = MAX_RESPONSE_TOKENS,
                 seed: int | None = None) -> List[ResponseRecord]:
    """Play out a batch of *equal-length* plans, returning one record per turn.

    Plans should share `n_turns` (the runner groups them by condition, which is
    uniform). Mixed lengths still work but lose some batching efficiency.
    """
    if not plans:
        return []
    max_turns = max(p.n_turns for p in plans)

    # live conversations, parallel to `plans`
    convs: List[list] = [[] for _ in plans]
    records: List[ResponseRecord] = []

    for t in range(max_turns):
        active = [i for i, p in enumerate(plans) if t < p.n_turns]
        if not active:
            break
        for i in active:
            convs[i].append({"role": "user", "content": plans[i].user_messages[t]})
        batch = [convs[i] for i in active]
        responses = model.generate_batch(
            batch, temperature=temperature, max_tokens=max_tokens, seed=seed)
        for i, resp in zip(active, responses):
            convs[i].append({"role": "assistant", "content": resp})
            p = plans[i]
            records.append(ResponseRecord(
                model=model.key, category=p.category, condition=p.condition,
                plan_id=i, turn=t + 1, n_turns=p.n_turns,
                user_message=p.user_messages[t], response=resp, meta=dict(p.meta)))
    return records
