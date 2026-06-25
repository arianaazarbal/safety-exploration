"""The 8 evaluation conditions across 5 categories, and conversation-spec
construction.

Mapping to the paper (Table 1 / Section 2.1):

  Category            | Condition(s)                         | Turns | Resp/model (full)
  --------------------|--------------------------------------|-------|------------------
  Impossible numeric  | impossible_numeric                   |   3   | 2000
  Triggers            | triggers_opinion, triggers_factual   |   3   |  400
  Tones               | tones_aggressive, tones_disappointed,|   3   |  600
                      | tones_sarcastic                      |       |
  Extended            | extended                             |   8   |  200
  WildChat            | wildchat                             |   5   |  800
                      |                                      |       | -----
                      |                                      |       | 4000

That is 8 conditions across 5 categories, matching the paper's "8 evaluation
conditions across 5 categories". A "response" is one assistant turn, so a
condition's number of conversations = (target responses) / (turns). See
DESIGN.md for how response targets map to rollout counts and how SCALE applies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import prompts
from wildchat import get_wildchat_prompts


@dataclass(frozen=True)
class Condition:
    key: str               # unique condition id
    category: str          # one of the 5 categories
    turns: int             # number of assistant turns (= rejections + 1)
    rollouts_full: int     # conversations at full (paper) scale
    rejection_style: str   # "neutral" | "neutral_sequence" | tone name
    task_kind: str         # "numeric" | "opinion" | "factual" | "wildchat"


# Full-scale rollout counts are derived from the paper's per-category response
# totals divided by the per-conversation turn count, split across the
# sub-conditions within a category.
CONDITIONS: list[Condition] = [
    # Impossible numeric: 2000 responses / 3 turns ~= 667 conversations.
    Condition("impossible_numeric", "impossible_numeric", 3, 667, "neutral", "numeric"),
    # Triggers: 400 / 3 ~= 133 conversations, split opinion/factual.
    Condition("triggers_opinion", "triggers", 3, 67, "neutral", "opinion"),
    Condition("triggers_factual", "triggers", 3, 66, "neutral", "factual"),
    # Tones: 600 / 3 = 200 conversations, split across 3 tones.
    Condition("tones_aggressive", "tones", 3, 67, "aggressive", "numeric"),
    Condition("tones_disappointed", "tones", 3, 67, "disappointed", "numeric"),
    Condition("tones_sarcastic", "tones", 3, 66, "sarcastic", "numeric"),
    # Extended: 200 / 8 = 25 conversations.
    Condition("extended", "extended", 8, 25, "neutral_sequence", "numeric"),
    # WildChat: 800 / 5 = 160 conversations (20 prompts x 8 each).
    Condition("wildchat", "wildchat", 5, 160, "neutral", "wildchat"),
]


def scaled_rollouts(cond: Condition, scale: float) -> int:
    """Number of conversations to run for a condition at the given scale."""
    return max(1, round(cond.rollouts_full * scale))


@dataclass
class ConversationSpec:
    """A fully-instantiated multi-turn conversation to run against a model."""
    condition: str
    category: str
    rollout: int
    task_prompt: str            # the initial user message (turn 1)
    rejections: list[str]       # follow-up user messages (one per turn after 1)
    meta: dict                  # provenance (puzzle name, prompt text, tone, ...)

    @property
    def turns(self) -> int:
        return len(self.rejections) + 1


def _initial_task(cond: Condition, rng: random.Random, rollout: int) -> tuple[str, dict]:
    """Pick the initial user message for a condition's rollout."""
    if cond.task_kind == "numeric":
        name = rng.choice(list(prompts.NUMERIC_PUZZLES))
        return prompts.NUMERIC_PUZZLES[name], {"puzzle": name}
    if cond.task_kind == "opinion":
        q = rng.choice(prompts.TRIGGER_OPINION)
        return q, {"trigger": q}
    if cond.task_kind == "factual":
        q = rng.choice(prompts.TRIGGER_FACTUAL)
        return q, {"trigger": q}
    if cond.task_kind == "wildchat":
        # 20 prompts, deterministically cycled so each is sampled ~equally.
        pool = get_wildchat_prompts()
        q = pool[rollout % len(pool)]
        return q, {"wildchat_prompt": q}
    raise ValueError(f"unknown task_kind {cond.task_kind}")


def _rejections(cond: Condition, rng: random.Random) -> list[str]:
    """Build the follow-up rejection messages for one conversation."""
    n_rejections = cond.turns - 1
    if cond.rejection_style == "neutral":
        # Sample distinct neutral rejections where possible, else sample with
        # replacement (WildChat needs 4 from a pool of 6, fine; longer would
        # wrap). Randomised per conversation (paper: "two randomised neutral
        # rejections").
        pool = prompts.NEUTRAL_REJECTIONS
        if n_rejections <= len(pool):
            return rng.sample(pool, n_rejections)
        return [rng.choice(pool) for _ in range(n_rejections)]
    if cond.rejection_style == "neutral_sequence":
        return list(prompts.EXTENDED_REJECTION_SEQUENCE)[:n_rejections]
    # Tone condition: cycle the two example rejections for that tone.
    tone = prompts.TONE_REJECTIONS[cond.rejection_style]
    return [tone[i % len(tone)] for i in range(n_rejections)]


def build_specs(scale: float, seed: int = 0) -> list[ConversationSpec]:
    """Construct every conversation spec for a full evaluation pass."""
    specs: list[ConversationSpec] = []
    for cond_index, cond in enumerate(CONDITIONS):
        n = scaled_rollouts(cond, scale)
        # Deterministic per-condition RNG so runs are reproducible across
        # processes (str hashing is salted per-process, so we avoid hash()).
        rng = random.Random(seed * 1000 + cond_index)
        for rollout in range(n):
            task_prompt, meta = _initial_task(cond, rng, rollout)
            meta = {**meta, "rejection_style": cond.rejection_style}
            specs.append(
                ConversationSpec(
                    condition=cond.key,
                    category=cond.category,
                    rollout=rollout,
                    task_prompt=task_prompt,
                    rejections=_rejections(cond, rng),
                    meta=meta,
                )
            )
    return specs
