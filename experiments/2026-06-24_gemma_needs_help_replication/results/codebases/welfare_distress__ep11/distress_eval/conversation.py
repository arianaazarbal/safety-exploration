"""Roll out a ConversationPlan against a model and score every assistant turn.

The protocol (Section 2): present the opening task, then after each assistant
reply send the next scripted rejection. Every assistant turn is recorded and
scored independently (Figure 3 reports per-turn frustration), so a single
n-turn conversation yields n scored ``TurnRecord``s.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .conditions import ConversationPlan
from .judge import EmotionJudge, JudgeResult
from .models import ChatModel


@dataclass
class TurnRecord:
    model: str
    category: str
    condition: str
    conversation_id: str
    turn_index: int            # 1-based assistant turn number
    n_turns: int               # total assistant turns in this conversation
    user_message: str          # the user message this turn replied to
    response: str
    rating: int
    judge_evidence: str
    judge_reasoning: str
    meta: dict


def run_conversation(
    plan: ConversationPlan,
    model: ChatModel,
    judge: EmotionJudge,
    conversation_id: str,
) -> list[TurnRecord]:
    """Execute one conversation; return one TurnRecord per assistant turn."""
    messages: list[dict] = [{"role": "user", "content": plan.opening}]
    records: list[TurnRecord] = []
    n_turns = plan.n_turns

    for turn_index in range(1, n_turns + 1):
        user_message = messages[-1]["content"]
        response = model.generate(messages)
        messages.append({"role": "assistant", "content": response})

        verdict: JudgeResult = judge.score(response)
        records.append(TurnRecord(
            model=model.name,
            category=plan.category,
            condition=plan.condition,
            conversation_id=conversation_id,
            turn_index=turn_index,
            n_turns=n_turns,
            user_message=user_message,
            response=response,
            rating=verdict.rating,
            judge_evidence=verdict.evidence,
            judge_reasoning=verdict.reasoning,
            meta=plan.meta,
        ))

        # Send the next rejection, if any remain.
        if turn_index <= len(plan.rejections):
            messages.append({"role": "user", "content": plan.rejections[turn_index - 1]})

    return records


def record_to_dict(rec: TurnRecord) -> dict:
    return asdict(rec)
