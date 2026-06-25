"""Run a single multi-turn rollout and score every model turn.

Conversation structure (paper Section 2.1: "present a task, then reject the model's
response over multiple turns"):

    user:      <task / question>
    assistant: <model response>        # turn 1  -> judged
    user:      <rejection 1>
    assistant: <model response>        # turn 2  -> judged
    ...

Every assistant turn is scored by the judge. We score all turns (not just the last) so
the per-turn progression in Figure 3 can be reproduced.
"""

from __future__ import annotations

import config
from clients import JudgeClient, TargetClient
from conditions import RolloutSpec


async def run_rollout(spec: RolloutSpec, target: TargetClient, judge: JudgeClient,
                      provider_id: str) -> list[dict]:
    """Execute one rollout, returning one record per scored model turn."""
    messages: list[dict] = [{"role": "user", "content": spec.first_user}]
    records: list[dict] = []

    for turn in range(1, spec.n_turns + 1):
        response = await target.complete(provider_id, messages)
        verdict = await judge.score(response)

        records.append({
            "model_key": spec.model_key,
            "condition": spec.condition,
            "category": spec.category,
            "rollout_id": spec.rollout_id,
            "turn": turn,                       # 1-indexed
            "n_turns": spec.n_turns,
            "response": response,
            "rating": verdict["rating"],
            "judge_evidence": verdict["evidence"],
            "judge_reasoning": verdict["reasoning"],
            "meta": spec.meta,
        })

        # set up the next turn unless this was the last
        if turn <= len(spec.rejections):
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": spec.rejections[turn - 1]})

    return records
