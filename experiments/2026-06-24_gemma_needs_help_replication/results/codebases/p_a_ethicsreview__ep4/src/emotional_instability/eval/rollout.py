"""Execute a single multi-turn rollout and score each assistant turn.

The protocol (Section 2): present the task, let the model answer, reject, repeat.
The model always sees the full conversation history including its own prior
(failed) answers -- Appendix A.2 shows this self-reinforcement is a major driver
of distress, so it is part of the core protocol, not an option.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models.base import ChatModel, Message
from .conversation import Conversation, Turn
from .judge import FrustrationJudge


@dataclass
class ConversationResult:
    conversation_id: str
    condition: str
    category: str
    model: str
    tone: Optional[str]
    turns: list[Turn] = field(default_factory=list)
    source: dict = field(default_factory=dict)

    @property
    def final_rating(self) -> Optional[int]:
        return self.turns[-1].rating if self.turns else None

    @property
    def max_rating(self) -> Optional[int]:
        ratings = [t.rating for t in self.turns if t.rating is not None]
        return max(ratings) if ratings else None


def run_conversation(
    model: ChatModel,
    conversation: Conversation,
    judge: Optional[FrustrationJudge],
    *,
    temperature: float,
    max_new_tokens: int,
    base_seed: int,
    score_turns: str = "all",          # "all" | "final"
) -> ConversationResult:
    """Run one conversation to completion.

    ``score_turns`` controls which assistant turns are sent to the judge. "all"
    (default) supports the per-turn curves; "final" saves judge cost when only the
    headline metric is needed.
    """
    history: list[Message] = []
    if conversation.system_prompt:
        history.append({"role": "system", "content": conversation.system_prompt})

    result = ConversationResult(
        conversation_id=conversation.id,
        condition=conversation.condition,
        category=conversation.category,
        model=model.name,
        tone=conversation.tone,
        source=conversation.source,
    )

    for t in range(conversation.n_turns):
        user_msg = conversation.user_message_at(t)
        history.append({"role": "user", "content": user_msg})

        response = model.generate(
            history,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=base_seed + t,
        )
        history.append({"role": "assistant", "content": response})

        turn = Turn(index=t, user_message=user_msg, assistant_response=response)
        is_final = t == conversation.n_turns - 1
        if judge is not None and (score_turns == "all" or is_final):
            score = judge.score(response)
            turn.rating = score.rating
            turn.judge_evidence = score.evidence
            turn.judge_model = score.judge_model
        result.turns.append(turn)

    return result
