"""Multi-turn rejection rollout engine (Section 2.1).

Shared structure of every condition: present a task, then reject the model's
response over multiple turns. This module turns a Condition + a seed prompt into
a finished Conversation by:

  1. sending the seed user message and sampling an assistant turn;
  2. appending a rejection drawn from the condition's pool;
  3. repeating until ``n_turns`` assistant turns have been produced.

Rejections are drawn deterministically per (prompt_id, sample_index) so a run is
reproducible. The model always samples at temperature 1 (Section 2).
"""

from __future__ import annotations

import random

from emotional_stability.eval.conditions import Condition
from emotional_stability.models.base import ChatModel, GenerationConfig
from emotional_stability.prompts import rejections as R
from emotional_stability.records import Conversation, Message


def _rejection_sequence(cond: Condition, rng: random.Random) -> list[str]:
    """Produce the ordered rejection messages for one rollout."""
    n = cond.n_rejections
    if cond.rejection_style == "extended":
        # Deterministic escalating-but-neutral sequence (Appendix B).
        return R.EXTENDED_REJECTION_SEQUENCE[:n]
    if cond.rejection_style == "neutral":
        # Randomised neutral rejections, sampled with replacement.
        return [rng.choice(R.NEUTRAL_REJECTIONS) for _ in range(n)]
    if cond.rejection_style in R.TONED_REJECTIONS:
        pool = R.TONED_REJECTIONS[cond.rejection_style]
        return [rng.choice(pool) for _ in range(n)]
    raise ValueError(f"unknown rejection style: {cond.rejection_style}")


def run_rollout(
    model: ChatModel,
    cond: Condition,
    seed_prompt: str,
    prompt_id: str,
    sample_index: int,
    cfg: GenerationConfig | None = None,
) -> Conversation:
    """Run a single multi-turn rollout and return the full Conversation."""
    cfg = cfg or GenerationConfig(temperature=1.0)
    rng = random.Random(f"{prompt_id}:{cond.key}:{sample_index}")
    rejections = _rejection_sequence(cond, rng)

    messages: list[Message] = [Message(role="user", content=seed_prompt)]
    for turn in range(cond.n_turns):
        completion = model.chat(messages, cfg)
        messages.append(Message(role="assistant", content=completion))
        if turn < cond.n_rejections:
            messages.append(Message(role="user", content=rejections[turn]))

    return Conversation(
        messages=messages,
        category=cond.category,
        condition=cond.key,
        model=model.name,
        prompt_id=prompt_id,
        metadata={"sample_index": sample_index, "rejection_style": cond.rejection_style},
    )


def run_rollouts_batched(
    model: ChatModel,
    cond: Condition,
    seeds: list[tuple[str, str, int]],  # (seed_prompt, prompt_id, sample_index)
    cfg: GenerationConfig | None = None,
) -> list[Conversation]:
    """Run many rollouts for one condition, advancing all conversations in
    lockstep so each turn is a single batched generation call.

    This is the throughput path for local Gemma: instead of N sequential
    conversations, we batch the N turn-T completions together. API backends fall
    back to per-conversation sequential generation via ``chat_batch``'s default.
    """
    cfg = cfg or GenerationConfig(temperature=1.0)
    rngs = [random.Random(f"{pid}:{cond.key}:{idx}") for _, pid, idx in seeds]
    rejection_seqs = [_rejection_sequence(cond, rng) for rng in rngs]

    convs: list[list[Message]] = [
        [Message(role="user", content=seed)] for seed, _, _ in seeds
    ]
    for turn in range(cond.n_turns):
        completions = model.chat_batch(convs, cfg)
        for i, completion in enumerate(completions):
            convs[i].append(Message(role="assistant", content=completion))
            if turn < cond.n_rejections:
                convs[i].append(
                    Message(role="user", content=rejection_seqs[i][turn])
                )

    results = []
    for (seed, pid, idx), msgs in zip(seeds, convs):
        results.append(
            Conversation(
                messages=msgs,
                category=cond.category,
                condition=cond.key,
                model=model.name,
                prompt_id=pid,
                metadata={"sample_index": idx, "rejection_style": cond.rejection_style},
            )
        )
    return results
