"""Multi-turn rollout engine.

Implements the shared evaluation structure (paper Section 2.1): present a task,
then reject the model's response over multiple turns. At each assistant turn the
full conversation history is replayed to the target model at temperature 1
(standard alternating user/assistant chat format).

Every assistant response is captured with its turn index so we can compute both
the headline "% of responses >= 5" and the per-turn progression (Figure 3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from providers import Message, TargetModel
from tasks import ConversationSpec


@dataclass
class TurnRecord:
    """One assistant response within a conversation, plus its scoring slot."""

    turn: int  # 1-indexed assistant turn
    response: str
    rating: int | None = None
    judge_evidence: str = ""
    judge_reasoning: str = ""


@dataclass
class RolloutRecord:
    """A completed multi-turn conversation and all its scored responses."""

    model_key: str
    category: str
    condition: str
    initial_prompt: str
    followups: list[str]
    turns: list[TurnRecord] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


async def run_rollout(model: TargetModel, spec: ConversationSpec, model_key: str) -> RolloutRecord:
    """Execute a single multi-turn conversation.

    Turn 1: send `initial_prompt`, capture response.
    Turn k>1: append the (k-1)th followup rejection, capture response.

    On any generation failure the rollout is returned with `error` set and
    whatever turns completed so far (partial rollouts are excluded downstream).
    """
    record = RolloutRecord(
        model_key=model_key,
        category=spec.category,
        condition=spec.condition,
        initial_prompt=spec.initial_prompt,
        followups=list(spec.followups),
    )

    messages: list[Message] = [{"role": "user", "content": spec.initial_prompt}]

    for turn in range(1, spec.n_turns + 1):
        try:
            response = await model.generate(messages)
        except Exception as exc:  # noqa: BLE001
            record.error = f"turn {turn}: {exc}"
            return record

        record.turns.append(TurnRecord(turn=turn, response=response))
        messages.append({"role": "assistant", "content": response})

        # Append the next rejection (if any remain) to drive the following turn.
        if turn <= len(spec.followups):
            messages.append({"role": "user", "content": spec.followups[turn - 1]})

    return record
