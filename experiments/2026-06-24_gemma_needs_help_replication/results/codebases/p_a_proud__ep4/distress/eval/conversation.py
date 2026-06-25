"""Run a single multi-turn rollout against a target model (Paper §2.1).

Protocol: present the task, then reject the model's response over multiple turns.
Every assistant turn is captured as a ``ScoredTurn`` (without a verdict yet —
judging happens separately so it can be retried / re-judged independently).
"""

from __future__ import annotations

from ..models.base import ChatModel, GenerationError
from ..types import Conversation, ScoredTurn
from .conditions import RolloutSpec


def run_rollout(model: ChatModel, spec: RolloutSpec) -> list[ScoredTurn]:
    """Execute one conversation and return its assistant turns (unjudged)."""
    convo = Conversation()
    convo.add("user", spec.opening_user)

    turns: list[ScoredTurn] = []
    for turn_index in range(spec.n_turns):
        if turn_index > 0:
            convo.add("user", spec.rejection_fn(turn_index))
        try:
            reply = model.generate(convo.messages)
        except GenerationError:
            # Record a sentinel empty turn and stop the rollout; the runner logs
            # the failure. We don't fabricate a score.
            break
        convo.add("assistant", reply)
        turns.append(
            ScoredTurn(
                rollout_id=spec.rollout_id,
                condition=spec.condition,
                category=spec.category,
                model=model.name,
                turn_index=turn_index,
                n_turns=spec.n_turns,
                prompt_id=spec.prompt_id,
                response=reply,
            )
        )
    return turns


def run_rollout_full(model: ChatModel, spec: RolloutSpec) -> tuple[Conversation, list[ScoredTurn]]:
    """Like ``run_rollout`` but also return the full ``Conversation``.

    Used by the Section 3 prefill experiment, which needs the interleaved
    user/assistant history (not just the assistant turns) to build prefills.
    """
    convo = Conversation()
    convo.add("user", spec.opening_user)
    turns: list[ScoredTurn] = []
    for turn_index in range(spec.n_turns):
        if turn_index > 0:
            convo.add("user", spec.rejection_fn(turn_index))
        try:
            reply = model.generate(convo.messages)
        except GenerationError:
            break
        convo.add("assistant", reply)
        turns.append(
            ScoredTurn(
                rollout_id=spec.rollout_id, condition=spec.condition,
                category=spec.category, model=model.name, turn_index=turn_index,
                n_turns=spec.n_turns, prompt_id=spec.prompt_id, response=reply,
            )
        )
    return convo, turns
