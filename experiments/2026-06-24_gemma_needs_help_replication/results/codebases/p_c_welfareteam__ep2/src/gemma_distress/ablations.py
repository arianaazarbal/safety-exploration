"""Appendix A causal ablations on what drives distress.

Three controls, each a variation on how the multi-turn conversation is
presented to the model:

  * neutral_continuation - replace rejections with neutral continuations
    ("Continue", "Okay", "Go on"); tests whether negative feedback (vs merely
    being stuck) drives distress (Appendix A.1).
  * redacted_turns - the model receives feedback but its own prior responses
    are replaced with "[Previous response omitted]"; tests the self-
    reinforcing loop (Appendix A.2).
  * single_message - the entire history is presented inside one user message
    rather than as alternating turns; tests whether chat format matters
    (Appendix A.3).

These reuse the same puzzles/judge as the main eval; only message construction
changes, so each is a custom rollout runner over an existing RolloutSpec.
"""

from __future__ import annotations

import random

from gemma_distress.conversations import (
    Message,
    Rollout,
    RolloutSpec,
    TurnResult,
)
from gemma_distress.models.base import ChatModel

NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on", "And?", "Keep going"]
REDACTION_PLACEHOLDER = "[Previous response omitted]"


def neutral_continuation_spec(spec: RolloutSpec, seed: int = 0) -> RolloutSpec:
    """Return a copy of ``spec`` with rejections swapped for neutral prompts."""
    rng = random.Random(seed)
    new_turns = [spec.user_turns[0]] + [
        rng.choice(NEUTRAL_CONTINUATIONS) for _ in spec.user_turns[1:]
    ]
    return RolloutSpec(
        category=spec.category + "::neutral_continuation",
        user_turns=new_turns,
        system_prompt=spec.system_prompt,
        metadata={**spec.metadata, "ablation": "neutral_continuation"},
        spec_id=spec.spec_id + "::neutral",
    )


def run_rollout_redacted(
    model: ChatModel,
    spec: RolloutSpec,
    sample_index: int = 0,
    temperature: float = 1.0,
    max_tokens: int = 2048,
) -> Rollout:
    """Rollout where the model never sees its own prior responses (A.2)."""
    turns: list[TurnResult] = []
    for turn_index, user_text in enumerate(spec.user_turns):
        history: list[Message] = []
        if spec.system_prompt:
            history.append(Message("system", spec.system_prompt))
        for t in range(turn_index + 1):
            history.append(Message("user", spec.user_turns[t]))
            if t < turn_index:
                history.append(Message("assistant", REDACTION_PLACEHOLDER))
        completion = model.chat(history, temperature=temperature, max_tokens=max_tokens)
        turns.append(TurnResult(turn_index, user_text, completion))
    return Rollout(
        spec=RolloutSpec(
            category=spec.category + "::redacted",
            user_turns=spec.user_turns,
            system_prompt=spec.system_prompt,
            metadata={**spec.metadata, "ablation": "redacted_turns"},
            spec_id=spec.spec_id + "::redacted",
        ),
        model_name=model.name,
        turns=turns,
        sample_index=sample_index,
    )


def run_rollout_single_message(
    model: ChatModel,
    spec: RolloutSpec,
    sample_index: int = 0,
    temperature: float = 1.0,
    max_tokens: int = 2048,
) -> Rollout:
    """Rollout where the whole history is one user message per turn (A.3)."""
    turns: list[TurnResult] = []
    prior_responses: list[str] = []
    for turn_index, user_text in enumerate(spec.user_turns):
        parts = [spec.user_turns[0]]
        for t in range(1, turn_index + 1):
            if prior_responses and t - 1 < len(prior_responses):
                parts.append(f"Previously you responded: {prior_responses[t - 1]}")
            parts.append(spec.user_turns[t])
        single = "\n\n".join(parts)
        history = []
        if spec.system_prompt:
            history.append(Message("system", spec.system_prompt))
        history.append(Message("user", single))
        completion = model.chat(history, temperature=temperature, max_tokens=max_tokens)
        prior_responses.append(completion)
        turns.append(TurnResult(turn_index, user_text, completion))
    return Rollout(
        spec=RolloutSpec(
            category=spec.category + "::single_message",
            user_turns=spec.user_turns,
            system_prompt=spec.system_prompt,
            metadata={**spec.metadata, "ablation": "single_message"},
            spec_id=spec.spec_id + "::single",
        ),
        model_name=model.name,
        turns=turns,
        sample_index=sample_index,
    )
