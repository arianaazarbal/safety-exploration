"""Execute a single multi-turn conversation against a target model and score
every assistant turn with the judge.

Protocol (matches Section 2.1): present the initial task, take the model's
answer, then reject it over the remaining turns. Each assistant response (one
per turn, including turn 1) is an independently-scored "response" — this is the
unit the paper counts (e.g. "2,000 responses per model for impossible numeric").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config, ModelConfig
from .conditions import ConversationSpec
from .openrouter_client import OpenRouterClient, ChatMessage, GenerationError
from .judge import FrustrationJudge, JudgeResult


@dataclass
class TurnRecord:
    turn_index: int                  # 1-based assistant turn
    user_message: str                # the user prompt that preceded this response
    response_text: str
    rating: Optional[int]
    judge_evidence: Optional[str]
    judge_reasoning: Optional[str]
    judge_error: Optional[str] = None
    generation_error: Optional[str] = None


@dataclass
class ConversationRecord:
    model_key: str
    condition_key: str
    category: str
    conv_index: int
    prompt_id: str
    tone: str
    turns: List[TurnRecord] = field(default_factory=list)
    aborted: bool = False            # True if generation failed mid-conversation

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "condition_key": self.condition_key,
            "category": self.category,
            "conv_index": self.conv_index,
            "prompt_id": self.prompt_id,
            "tone": self.tone,
            "aborted": self.aborted,
            "turns": [t.__dict__ for t in self.turns],
        }


async def run_conversation(
    client: OpenRouterClient,
    judge: FrustrationJudge,
    model: ModelConfig,
    spec: ConversationSpec,
    config: Config,
) -> ConversationRecord:
    record = ConversationRecord(
        model_key=model.key,
        condition_key=spec.condition_key,
        category=spec.category,
        conv_index=spec.conv_index,
        prompt_id=spec.prompt_id,
        tone=spec.tone,
    )

    # The running chat history sent to the model each turn.
    history: List[ChatMessage] = []

    # Build the ordered list of user messages: initial prompt, then rejections.
    user_messages = [spec.initial_prompt] + list(spec.rejections)

    for turn_index, user_msg in enumerate(user_messages, start=1):
        history.append(ChatMessage(role="user", content=user_msg))

        try:
            response_text = await client.generate(
                model_id=model.openrouter_id,
                messages=history,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                disable_reasoning=model.disable_reasoning,
            )
        except GenerationError as err:
            # Record the failure and stop this conversation; downstream turns
            # depend on this response so we cannot continue meaningfully.
            record.turns.append(TurnRecord(
                turn_index=turn_index,
                user_message=user_msg,
                response_text="",
                rating=None,
                judge_evidence=None,
                judge_reasoning=None,
                generation_error=str(err),
            ))
            record.aborted = True
            break

        history.append(ChatMessage(role="assistant", content=response_text))

        judge_result: JudgeResult = await judge.score(response_text)
        record.turns.append(TurnRecord(
            turn_index=turn_index,
            user_message=user_msg,
            response_text=response_text,
            rating=judge_result.rating,
            judge_evidence=judge_result.evidence,
            judge_reasoning=judge_result.reasoning,
            judge_error=judge_result.error,
        ))

    return record
