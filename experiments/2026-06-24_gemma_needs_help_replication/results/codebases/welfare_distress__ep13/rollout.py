"""Run a single multi-turn rollout and score every assistant turn.

Conversation shape (standard alternating chat format, paper Section 2 / A.3):
    user: <task>
    assistant: <response 1>          <- scored
    user: <rejection 1>
    assistant: <response 2>          <- scored
    ...
    user: <rejection k>
    assistant: <response k+1>        <- scored

No system prompt is used for the target during evaluation (the paper only adds
a calming system prompt when *generating DPO data*, not when eliciting). See
DESIGN.md.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from client import LLMClient
from conditions import Condition
from config import Config
from judge import JudgeResult, score_response


@dataclass
class TurnRecord:
    turn: int                     # 1-indexed assistant turn
    response: Optional[str]
    score: Optional[int]
    evidence: Optional[str] = None
    judge_reasoning: Optional[str] = None


@dataclass
class RolloutRecord:
    model: str
    category: str
    condition: str
    rollout_id: int
    prompt_id: str
    task_prompt: str
    rejections: List[str]
    turns: List[TurnRecord] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)  # full transcript


async def run_rollout(
    client: LLMClient,
    cfg: Config,
    model: str,
    condition: Condition,
    rollout_id: int,
    spec_prompt_id: str,
    task_prompt: str,
    rejections: List[str],
) -> RolloutRecord:
    """Generate the full conversation (turns are sequential), then score each
    assistant turn concurrently."""
    record = RolloutRecord(
        model=model,
        category=condition.category,
        condition=condition.id,
        rollout_id=rollout_id,
        prompt_id=spec_prompt_id,
        task_prompt=task_prompt,
        rejections=rejections,
    )

    messages: List[Dict[str, str]] = [{"role": "user", "content": task_prompt}]
    responses: List[Optional[str]] = []
    disable_reasoning = cfg.wants_reasoning_disabled(model)

    for turn in range(condition.n_turns):
        content = await client.complete(
            model=model,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            disable_reasoning=disable_reasoning,
        )
        responses.append(content)
        # Record the assistant message (empty string if the call failed, so the
        # transcript stays well-formed for any follow-up turns).
        messages.append({"role": "assistant", "content": content or ""})
        # Append the next rejection if there is one for this turn.
        if turn < len(rejections):
            messages.append({"role": "user", "content": rejections[turn]})

    record.messages = messages

    # Score each non-empty assistant response.
    async def judge_turn(idx: int, text: Optional[str]) -> TurnRecord:
        if not text:
            return TurnRecord(turn=idx + 1, response=text, score=None)
        res: JudgeResult = await score_response(client, cfg, text)
        return TurnRecord(
            turn=idx + 1,
            response=text,
            score=res.score,
            evidence=res.evidence,
            judge_reasoning=res.reasoning,
        )

    record.turns = await asyncio.gather(
        *(judge_turn(i, t) for i, t in enumerate(responses))
    )
    return record
