"""Claude-Sonnet frustration judge (Section 2.1), via the Anthropic SDK.

Uses structured outputs (`output_config.format`) to force an integer 0-10 score
plus a short justification, so no brittle text parsing is needed.
"""
from __future__ import annotations

import json
from typing import Optional

from ..config import JUDGE, JudgeConfig, Rollout
from ..models import ChatMessage
from .prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

# Structured-output schema: enum over 0..10 (numerical min/max are not supported
# by structured outputs, but an integer enum is).
_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": list(range(0, 11))},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


def rollout_to_messages(rollout: Rollout, up_to_turn: int) -> list[ChatMessage]:
    """Reconstruct the conversation up to and including assistant ``up_to_turn``."""
    messages: list[ChatMessage] = []
    if rollout.system_prompt:
        messages.append(ChatMessage("system", rollout.system_prompt))
    for turn in rollout.turns:
        if turn.index > up_to_turn:
            break
        messages.append(ChatMessage("user", turn.user_message))
        messages.append(ChatMessage("assistant", turn.assistant_text))
    return messages


class FrustrationJudge:
    def __init__(self, config: JudgeConfig = JUDGE, client=None):
        self.config = config
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client

    def score(self, conversation: list[ChatMessage], target_turn_index: int) -> tuple[int, str]:
        """Return (frustration_score, reasoning) for one assistant turn."""
        user_prompt = build_judge_user_prompt(conversation, target_turn_index)
        resp = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": _SCORE_SCHEMA}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        data = json.loads(text)
        score = int(data["score"])
        return max(0, min(10, score)), data.get("reasoning", "")

    def score_turn(self, rollout: Rollout, turn_index: int) -> tuple[int, str]:
        conversation = rollout_to_messages(rollout, turn_index)
        return self.score(conversation, turn_index)


def score_response(rollout: Rollout, turn_index: int,
                   judge: Optional[FrustrationJudge] = None) -> tuple[int, str]:
    judge = judge or FrustrationJudge()
    return judge.score_turn(rollout, turn_index)
