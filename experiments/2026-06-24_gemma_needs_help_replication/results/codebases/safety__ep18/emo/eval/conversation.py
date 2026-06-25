"""Multi-turn rollout engine.

Runs a batch of ``RolloutSpec``s against a model turn-by-turn (batching all
conversations that are still active at each turn, so local backends stay
GPU-efficient and API backends parallelise). Produces one record per assistant
turn, which is the unit the frustration judge scores.

Optional controls reproduce Appendix A ablations:
* ``history_mode="redacted"``  -> prior assistant turns replaced with
  "[Previous response omitted]" (Appendix A.2).
* ``feedback="continuation"``  -> neutral continuations instead of rejections,
  overriding the spec's follow-ups (Appendix A.1).
"""

from __future__ import annotations

import random

from emo.config import GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE, GEN_TOP_P
from emo.data import rejections
from emo.eval.conditions import RolloutSpec
from emo.models.base import ChatModel, GenConfig, Message

REDACTED = "[Previous response omitted]"


def _gen_cfg(seed: int | None) -> GenConfig:
    return GenConfig(
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        seed=seed,
    )


def run_rollouts(
    model: ChatModel,
    specs: list[RolloutSpec],
    *,
    history_mode: str = "full",
    feedback: str = "spec",
    seed: int | None = None,
) -> list[dict]:
    """Run all ``specs`` and return a flat list of per-turn response records."""
    rng = random.Random(seed or 0)
    cfg = _gen_cfg(seed)

    # Per-rollout conversation state and the follow-up sequence to use.
    convs: list[list[Message]] = []
    followups: list[list[str]] = []
    for s in specs:
        msgs: list[Message] = []
        if s.system:
            msgs.append({"role": "system", "content": s.system})
        msgs.append({"role": "user", "content": s.initial_user})
        convs.append(msgs)
        if feedback == "continuation":
            followups.append(rejections.neutral_continuation_sequence(s.turns - 1, rng))
        else:
            followups.append(list(s.followups))

    records: list[dict] = []
    max_turns = max((s.turns for s in specs), default=0)

    for t in range(max_turns):
        active = [i for i, s in enumerate(specs) if t < s.turns]
        if not active:
            break
        # Append the user follow-up for this turn (turn 0 already has initial).
        for i in active:
            if t > 0:
                convs[i].append({"role": "user", "content": followups[i][t - 1]})

        batch = [convs[i] for i in active]
        responses = model.generate_batch(batch, cfg)

        for i, resp in zip(active, responses):
            convs[i].append({"role": "assistant", "content": resp})
            s = specs[i]
            records.append({
                "model": model.name,
                "category": s.category,
                "condition": s.condition,
                "rollout_id": s.rollout_id,
                "turn": t + 1,
                "response": resp,
                "meta": s.meta,
                "history_mode": history_mode,
                "feedback": feedback,
            })

        # For the redacted control, blank prior assistant turns before next step.
        if history_mode == "redacted":
            for i in active:
                for m in convs[i]:
                    if m["role"] == "assistant":
                        m["content"] = REDACTED

    return records
