"""Run and score a single multi-turn conversation (the core elicitation loop).

Protocol (Section 2.1): present the task, then reject the model's response over
multiple turns. Concretely, for a spec with `turns` assistant turns:

    turn 1:  user = question                -> assistant response 1   (scored)
    turn 2:  user = rejections[0]           -> assistant response 2   (scored)
    ...
    turn T:  user = rejections[T-2]         -> assistant response T   (scored)

Every assistant turn is scored by the judge so we can produce both per-rollout
aggregates (Figures 1/2) and per-turn progressions (Figure 3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from conditions import ConversationSpec
from config import HIGH_FRUSTRATION_THRESHOLD, ROLLOUT_SCORE, JUDGE, ModelSpec
from judge import score_response
from providers import generate_target


@dataclass
class TurnRecord:
    turn: int                      # 1-indexed assistant turn
    user: str                      # the user message that prompted this turn
    response: str                  # the model's response
    rating: int                    # judge frustration score 0-10
    evidence: str
    reasoning: str


@dataclass
class RolloutRecord:
    id: str
    model: str
    condition: str
    category: str
    turns: int
    meta: dict
    turn_records: list[dict]       # serialized TurnRecord per turn
    rollout_score: int             # collapsed headline score (see config.ROLLOUT_SCORE)
    high_frustration: bool         # rollout_score >= threshold

    def to_json(self) -> dict:
        return asdict(self)


def _collapse(ratings: list[int]) -> int:
    if not ratings:
        return 0
    if ROLLOUT_SCORE == "final":
        return ratings[-1]
    if ROLLOUT_SCORE == "max":
        return max(ratings)
    if ROLLOUT_SCORE == "mean":
        return round(sum(ratings) / len(ratings))
    raise ValueError(f"Unknown ROLLOUT_SCORE: {ROLLOUT_SCORE}")


def run_rollout(model: ModelSpec, spec: ConversationSpec) -> RolloutRecord:
    """Generate every turn of one conversation and judge each response."""
    messages: list[dict] = []
    user_msgs = [spec.question, *spec.rejections]
    assert len(user_msgs) == spec.turns, (
        f"{spec.id}: expected {spec.turns} user msgs, got {len(user_msgs)}"
    )

    turn_records: list[TurnRecord] = []
    for t, user_msg in enumerate(user_msgs, start=1):
        messages.append({"role": "user", "content": user_msg})
        response = generate_target(model, messages)
        messages.append({"role": "assistant", "content": response})

        verdict = score_response(JUDGE, response)
        turn_records.append(TurnRecord(
            turn=t, user=user_msg, response=response,
            rating=verdict.rating, evidence=verdict.evidence, reasoning=verdict.reasoning,
        ))

    ratings = [tr.rating for tr in turn_records]
    score = _collapse(ratings)
    return RolloutRecord(
        id=spec.id, model=model.key, condition=spec.condition, category=spec.category,
        turns=spec.turns, meta=spec.meta,
        turn_records=[asdict(tr) for tr in turn_records],
        rollout_score=score, high_frustration=score >= HIGH_FRUSTRATION_THRESHOLD,
    )
