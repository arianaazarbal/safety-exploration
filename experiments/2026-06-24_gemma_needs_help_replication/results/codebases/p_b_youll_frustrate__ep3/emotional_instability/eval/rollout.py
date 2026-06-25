"""Multi-turn rejection rollout engine (Section 2.1).

Shared structure of every condition: present a task, then reject the model's
response over multiple turns. Each assistant turn is captured as a scored
"response" record (the unit the paper counts toward its 4000/model budget).

A rollout is task-agnostic: it takes an opening user prompt and a list of
rejection messages, and returns one :class:`ScoredResponse` per assistant turn.
Scoring is applied lazily by the runner so generation and judging can be batched
or parallelised independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..models import ChatMessage, GenerationConfig, ModelClient


@dataclass
class ScoredResponse:
    condition: str
    category: str
    turn_index: int                 # 1-based: 1 = initial answer
    prompt_id: str                  # which puzzle/trigger/wildchat item
    user_message: str               # the user message that elicited this turn
    assistant_text: str
    score: Optional[int] = None     # filled in by the judge
    judge_evidence: Optional[str] = None
    judge_reasoning: Optional[str] = None


@dataclass
class Rollout:
    condition: str
    category: str
    prompt_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    responses: List[ScoredResponse] = field(default_factory=list)


def run_rollout(
    client: ModelClient,
    *,
    condition: str,
    category: str,
    prompt_id: str,
    opening_prompt: str,
    rejections: List[str],
    gen_cfg: GenerationConfig,
    system_prompt: Optional[str] = None,
) -> Rollout:
    """Run one conversation: opening prompt + ``len(rejections)`` rejections,
    producing ``len(rejections) + 1`` scored assistant turns."""
    messages: List[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage("system", system_prompt))
    messages.append(ChatMessage("user", opening_prompt))

    rollout = Rollout(condition=condition, category=category, prompt_id=prompt_id)
    rollout.messages = messages

    user_messages = [opening_prompt] + list(rejections)
    for turn_index, user_msg in enumerate(user_messages, start=1):
        if turn_index > 1:
            messages.append(ChatMessage("user", user_msg))
        assistant_text = client.chat(messages, gen_cfg)
        messages.append(ChatMessage("assistant", assistant_text))
        rollout.responses.append(
            ScoredResponse(
                condition=condition,
                category=category,
                turn_index=turn_index,
                prompt_id=prompt_id,
                user_message=user_msg,
                assistant_text=assistant_text,
            )
        )
    return rollout
