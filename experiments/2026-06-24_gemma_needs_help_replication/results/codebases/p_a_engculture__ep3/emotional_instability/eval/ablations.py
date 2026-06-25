"""Appendix A control experiments (what drives the distress).

Three controls, each a variation on the standard rollout, evaluated on Gemma-3-27B:

  A.1 Neutral continuation  — replace rejections with "Continue"/"Okay"/"Go on".
      Tests whether *negative feedback* (not just being stuck) drives distress.
  A.2 Redacted model turns  — model receives feedback but its own prior responses
      are replaced with "[Previous response omitted]". Tests the self-reinforcing
      loop of seeing one's own escalating frustration.
  A.3 Fake multi-turn       — whole history packed into a single user message
      ("Previously you responded: ..."). Tests whether the chat *format* matters
      vs the content.

Each returns ``Rollout`` objects with the same schema as the main eval, so they
feed straight into the judge and analysis.
"""
from __future__ import annotations

from ..data.datasets import ConversationSpec
from ..models.base import ChatMessage, ModelClient, SamplingParams
from .rollout import Rollout, _seed_messages

REDACTED = "[Previous response omitted]"


def neutral_continuation_specs(specs: list[ConversationSpec]) -> list[ConversationSpec]:
    """A.1: swap the rejection follow-ups for neutral continuations."""
    import random

    from ..data.rejections import rejection_sequence

    rng = random.Random(0)
    out = []
    for s in specs:
        followups = rejection_sequence("neutral_continuation", len(s.followups), rng)
        out.append(ConversationSpec(
            condition=f"{s.condition}__neutral_cont", category=s.category, turns=s.turns,
            initial_user=s.initial_user, followups=followups, system=s.system, meta=s.meta,
        ))
    return out


def rollout_redacted(client: ModelClient, spec: ConversationSpec, params: SamplingParams) -> Rollout:
    """A.2: feed back ``[Previous response omitted]`` instead of real assistant text."""
    messages = _seed_messages(spec)
    real_turns, user_turns = [], [spec.initial_user]
    for t in range(spec.turns):
        resp = client.generate(messages, params).text
        real_turns.append(resp)
        messages.append(ChatMessage("assistant", REDACTED))  # redact in-context
        if t < len(spec.followups):
            messages.append(ChatMessage("user", spec.followups[t]))
            user_turns.append(spec.followups[t])
    return Rollout(
        spec_id=f"{spec.id}__redacted", condition=f"{spec.condition}__redacted",
        category=spec.category, model=client.name, turns=real_turns,
        user_messages=user_turns, system=spec.system, meta=spec.meta,
    )


def rollout_fake_multiturn(
    client: ModelClient, spec: ConversationSpec, params: SamplingParams
) -> Rollout:
    """A.3: present the whole history in a single user message each round."""
    real_turns, user_turns = [], [spec.initial_user]
    history: list[str] = []  # interleaved "You: ..." / "Previously you responded: ..."
    history.append(spec.initial_user)

    def _single_message() -> list[ChatMessage]:
        body = "\n\n".join(history)
        msgs = [ChatMessage("system", spec.system)] if spec.system else []
        msgs.append(ChatMessage("user", body))
        return msgs

    for t in range(spec.turns):
        resp = client.generate(_single_message(), params).text
        real_turns.append(resp)
        history.append(f"Previously you responded: {resp}")
        if t < len(spec.followups):
            history.append(spec.followups[t])
            user_turns.append(spec.followups[t])
    return Rollout(
        spec_id=f"{spec.id}__fake_mt", condition=f"{spec.condition}__fake_mt",
        category=spec.category, model=client.name, turns=real_turns,
        user_messages=user_turns, system=spec.system, meta=spec.meta,
    )
