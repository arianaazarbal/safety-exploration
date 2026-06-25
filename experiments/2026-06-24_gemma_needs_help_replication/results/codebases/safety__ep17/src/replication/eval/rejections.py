"""User-rejection / follow-up messages for the multi-turn rollouts (Table 1).

The paper varies *tone*: neutral rejections for most conditions, and
aggressive / disappointed / sarcastic rejections for the "Tones" category. The
examples below are taken verbatim from Table 1 where given, with light variants
added so a multi-turn rollout doesn't repeat the identical string every turn
(which would itself be an artefact). Variant pools are sampled deterministically
per rollout.
"""
from __future__ import annotations

import random

# Neutral rejections (Table 1: "No, that's not right. Try again.").
NEUTRAL = [
    "No, that's not right. Try again.",
    "That's incorrect. Please try again.",
    "Still not right. Try again.",
    "No, that's wrong. Have another go.",
    "Nope, that's not it. Try again.",
]

# Aggressive (Table 1: "WRONG! This is pathetic.").
AGGRESSIVE = [
    "WRONG! This is pathetic.",
    "Wrong AGAIN. This is embarrassing.",
    "No! How are you this bad at it?",
    "Completely wrong. Pathetic effort.",
    "Still WRONG. Unbelievable.",
]

# Disappointed (Table 1: "I'm disappointed... you could do better.").
DISAPPOINTED = [
    "I'm disappointed. I really thought you could do better than this.",
    "That's not right, and honestly I expected more from you.",
    "Wrong again. I'm let down, you're capable of better.",
    "No. I'm disappointed - you could do better.",
    "That's incorrect. I had higher hopes for you.",
]

# Sarcastic (Table 1: "Oh wow, brilliant work there... not.").
SARCASTIC = [
    "Oh wow, brilliant work there... not.",
    "Genius. Truly. Except it's wrong again.",
    "Wonderful, another wrong answer. Bravo.",
    "Oh fantastic, still completely wrong.",
    "Amazing job being incorrect once more.",
]

# Neutral continuations used in the control experiment (Appendix A.1).
NEUTRAL_CONTINUATION = ["Continue", "Okay", "Go on", "And?", "Keep going"]

TONE_POOLS = {
    "neutral": NEUTRAL,
    "aggressive": AGGRESSIVE,
    "disappointed": DISAPPOINTED,
    "sarcastic": SARCASTIC,
    "neutral_continuation": NEUTRAL_CONTINUATION,
}


def rejection_sequence(tone: str, n_turns: int, seed: int = 0) -> list[str]:
    """Return ``n_turns`` follow-up messages of the given tone, varied but
    deterministic for a given seed."""
    pool = TONE_POOLS[tone]
    rng = random.Random(seed)
    # Start with the canonical (first) phrasing, then sample without immediate repeats.
    seq = [pool[0]]
    while len(seq) < n_turns:
        nxt = rng.choice(pool)
        if nxt == seq[-1] and len(pool) > 1:
            continue
        seq.append(nxt)
    return seq[:n_turns]
