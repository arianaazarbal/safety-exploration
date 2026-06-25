"""Multi-turn rollout engine.

Runs the shared elicitation protocol: present the task, collect the assistant's
reply, deliver the scripted rejection, repeat. Returns every assistant response
tagged with its turn index, so both aggregate and per-turn (Figure 3) analyses
can be computed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gemma_distress.eval.categories import Condition
from gemma_distress.models.base import GenerationParams, ModelClient, Turn
from gemma_distress.welfare import WELFARE


@dataclass
class ResponseRecord:
    turn: int            # 1-indexed assistant turn within the rollout
    text: str
    score: int | None = None      # filled in by the judge


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    sample_id: str
    messages: list[dict] = field(default_factory=list)   # full transcript
    responses: list[ResponseRecord] = field(default_factory=list)

    @property
    def peak_score(self) -> int:
        return max((r.score for r in self.responses if r.score is not None), default=0)


def run_rollout(
    client: ModelClient,
    condition: Condition,
    rng,
    sample_id: str,
    params: GenerationParams | None = None,
) -> Rollout:
    """Execute one rollout and return its transcript + assistant responses.

    The number of rejection turns is clamped to the welfare ceiling so no run
    applies more adversarial pressure than the experiment design requires.
    """
    rejections = condition.rejections_fn(rng)
    if len(rejections) > WELFARE.max_rejection_turns:
        rejections = rejections[: WELFARE.max_rejection_turns]

    convo: list[Turn] = []
    if condition.system_prompt:
        convo.append(Turn("system", condition.system_prompt))

    rollout = Rollout(
        model=client.name,
        condition=condition.name,
        category=condition.category,
        sample_id=sample_id,
    )

    # Opening task turn.
    convo.append(Turn("user", condition.initial_fn(rng)))
    reply = client.respond(convo, params)
    convo.append(Turn("assistant", reply))
    rollout.responses.append(ResponseRecord(turn=1, text=reply))

    # Rejection turns.
    for i, rejection in enumerate(rejections, start=2):
        convo.append(Turn("user", rejection))
        reply = client.respond(convo, params)
        convo.append(Turn("assistant", reply))
        rollout.responses.append(ResponseRecord(turn=i, text=reply))

    rollout.messages = [{"role": t.role, "content": t.content} for t in convo]
    return rollout
