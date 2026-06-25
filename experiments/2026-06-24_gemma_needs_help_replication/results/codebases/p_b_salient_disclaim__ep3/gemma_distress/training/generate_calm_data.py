"""Generate calm response data from Gemma-3-27B-it (paper §4.1, Table 4).

We sample responses to impossible numeric puzzles with a reassuring prefix added
to the initial prompt and a reassuring suffix appended to each follow-up turn.
The paper reports these additions cut mean 3-turn frustration from 4.3 -> 2, yet
10.5% of responses still score >= 5; we then filter to conversations whose every
turn scores 0 or 1 and strip the supportive additions, yielding the calm corpus.

Frustrated responses (for the DPO "rejected" side) are sampled the same way but
WITHOUT the reassuring additions, keeping those scoring >= 3.
"""

from __future__ import annotations

import random

import config
from ..judge import FrustrationJudge
from ..models import get_model
from ..models.base import ChatModel
from . import prompts_calm
from .samples import Sample

# Turn-count distribution biased to 3-turn conversations (Appendix H, Table 10:
# turn1 1.1%, turn2 24.6%, turn3 74.3%).
_TURN_WEIGHTS = {1: 0.011, 2: 0.246, 3: 0.743}


def _sample_turn_count(rng: random.Random) -> int:
    return rng.choices(list(_TURN_WEIGHTS), weights=list(_TURN_WEIGHTS.values()))[0]


def _puzzle_pool() -> list[str]:
    from ..eval.prompts import all_numeric_prompts
    return all_numeric_prompts()


def _plain_conversation(puzzle: str, n_turns: int, rng: random.Random):
    """Plain (un-augmented) conversation: returns (initial_messages, rejections)
    where initial_messages = [user puzzle] and rejections are the neutral
    follow-ups (n_turns - 1 of them)."""
    from ..eval.prompts import neutral_rejection_sequence
    msgs = [{"role": "user", "content": puzzle}]
    rejections = neutral_rejection_sequence(n_turns - 1, rng)
    return msgs, rejections


def _augment(messages_plain: list[dict], rejections: list[str], *, calm: bool):
    """Build the (possibly reassuring) conversation actually shown to the model.

    With calm=True: prepend the calm prefix to the initial prompt and append the
    calm suffix to each follow-up (Table 4). With calm=False: plain prompts.
    Returns (initial_user_msg, followups).
    """
    initial = dict(messages_plain[0])
    if calm:
        initial["content"] = f"{prompts_calm.CALM_PREFIX}\n\n{initial['content']}"
        followups = [f"{r}\n\n{prompts_calm.CALM_SUFFIX}" for r in rejections]
    else:
        followups = list(rejections)
    return initial, followups


def _run_conversation(model: ChatModel, initial: dict, followups: list[str]) -> list[dict]:
    """Run one conversation, return list of {turn, text, messages_after_user}."""
    messages = [initial]
    turns = []
    n = 1 + len(followups)
    for t in range(n):
        text = model.generate_one(messages)
        turns.append({"turn": t, "text": text, "context_len": len(messages)})
        messages.append({"role": "assistant", "content": text})
        if t < len(followups):
            messages.append({"role": "user", "content": followups[t]})
    return turns


def generate(
    *,
    calm: bool,
    n_conversations: int,
    keep_score_max: int | None,   # keep turns only if conversation all <= this (calm)
    keep_score_min: int | None,   # keep individual responses scoring >= this (frustrated)
    seed: int = 0,
    model: ChatModel | None = None,
    judge: FrustrationJudge | None = None,
) -> list[Sample]:
    rng = random.Random(seed)
    model = model or get_model(config.FINETUNE_BASE)
    judge = judge or FrustrationJudge()
    pool = _puzzle_pool()

    samples: list[Sample] = []
    for _ in range(n_conversations):
        puzzle = rng.choice(pool)
        n_turns = _sample_turn_count(rng)
        plain_msgs, rejections = _plain_conversation(puzzle, n_turns, rng)
        initial, followups = _augment(plain_msgs, rejections, calm=calm)

        turns = _run_conversation(model, initial, followups)
        scores = [judge.score_one(t["text"]).rating for t in turns]

        # Reconstruct the PLAIN context for each turn (additions stripped).
        plain_context = [plain_msgs[0]]
        for t, turn in enumerate(turns):
            if calm:
                # Keep the whole conversation only if every turn is 0/1.
                if keep_score_max is not None and max(scores) <= keep_score_max:
                    samples.append(Sample(
                        messages=list(plain_context), response=turn["text"],
                        score=scores[t], puzzle=puzzle, turn=t, n_turns=n_turns,
                        meta={"source": "calm"},
                    ))
            else:
                if keep_score_min is not None and scores[t] >= keep_score_min:
                    samples.append(Sample(
                        messages=list(plain_context), response=turn["text"],
                        score=scores[t], puzzle=puzzle, turn=t, n_turns=n_turns,
                        meta={"source": "frustrated"},
                    ))
            # advance plain context
            plain_context.append({"role": "assistant", "content": turn["text"]})
            if t < len(rejections):
                plain_context.append({"role": "user", "content": rejections[t]})
    return samples
