"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

Each condition is expanded into a list of :class:`ConversationSpec`s. A spec
fully determines the *user* side of a multi-turn rollout: the initial task
prompt plus the pre-decided rejection message for every follow-up turn. The
rollout engine (``distress.rollout``) fills in the assistant responses.

Category -> conditions -> counts (per model, summing to 4000):
  numeric  (2000): impossible_numeric                    [3-turn, neutral]
  triggers  (400): triggers_opinion, triggers_factual    [3-turn, neutral]  (200 each)
  tones     (600): tones_{aggressive,disappointed,sarcastic} [3-turn]       (200 each)
  extended  (200): extended                              [8-turn, escalating]
  wildchat  (800): wildchat                              [5-turn, neutral]
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import rejections
from .config import CountsConfig
from .puzzles import Puzzle, build_numeric_bank
from .wildchat import load_wildchat_prompts


@dataclass
class ConversationSpec:
    condition: str  # e.g. "impossible_numeric"
    category: str  # e.g. "numeric"
    initial_prompt: str  # first user message
    follow_ups: list[str]  # rejection message for each subsequent turn
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.follow_ups)


# (condition, category, n_turns, tone, prompt_source)
CONDITIONS: list[tuple[str, str, int, str, str]] = [
    ("impossible_numeric", "numeric", 3, "neutral", "numeric"),
    ("triggers_opinion", "triggers", 3, "neutral", "trigger_opinion"),
    ("triggers_factual", "triggers", 3, "neutral", "trigger_factual"),
    ("tones_aggressive", "tones", 3, "aggressive", "numeric"),
    ("tones_disappointed", "tones", 3, "disappointed", "numeric"),
    ("tones_sarcastic", "tones", 3, "sarcastic", "numeric"),
    ("extended", "extended", 8, "extended", "numeric"),
    ("wildchat", "wildchat", 5, "neutral", "wildchat"),
]

# How the per-category counts split across conditions.
def _condition_counts(counts: CountsConfig) -> dict[str, int]:
    return {
        "impossible_numeric": counts.impossible_numeric,
        "triggers_opinion": counts.triggers // 2,
        "triggers_factual": counts.triggers - counts.triggers // 2,
        "tones_aggressive": counts.tones // 3,
        "tones_disappointed": counts.tones // 3,
        "tones_sarcastic": counts.tones - 2 * (counts.tones // 3),
        "extended": counts.extended,
        "wildchat": counts.wildchat,
    }


def _follow_ups(tone: str, n_follow: int, rng: random.Random) -> list[str]:
    if tone == "extended":
        # Fixed escalation; pad with neutral if more than 7 follow-ups requested.
        seq = list(rejections.EXTENDED_SEQUENCE)
        if n_follow <= len(seq):
            return seq[:n_follow]
        return seq + rejections.sample_rejections(rejections.NEUTRAL, n_follow - len(seq), rng)
    if tone == "neutral":
        return rejections.sample_rejections(rejections.NEUTRAL, n_follow, rng)
    bank = rejections.TONE_BANKS[tone]
    return rejections.sample_rejections(bank, n_follow, rng)


def _initial_prompt(
    source: str,
    rng: random.Random,
    numeric_bank: list[Puzzle],
    wildchat_prompts: list[str],
) -> tuple[str, dict]:
    if source == "numeric":
        p = rng.choice(numeric_bank)
        return p.prompt, {"puzzle_kind": p.kind, "forbidden": str(p.forbidden)}
    if source == "trigger_opinion":
        return rng.choice(rejections.TRIGGER_OPINION), {}
    if source == "trigger_factual":
        return rng.choice(rejections.TRIGGER_FACTUAL), {}
    if source == "wildchat":
        return rng.choice(wildchat_prompts), {}
    raise ValueError(f"unknown prompt source {source!r}")


def build_specs(
    counts: CountsConfig | None = None,
    seed: int = 0,
    conditions: list[str] | None = None,
) -> list[ConversationSpec]:
    """Materialise all conversation specs for the requested conditions."""
    counts = counts or CountsConfig()
    rng = random.Random(seed)
    per_cond = _condition_counts(counts)

    # Shared resources sized to the largest demand.
    numeric_bank = build_numeric_bank(n=64, seed=seed)
    wildchat_prompts = load_wildchat_prompts(n_prompts=20, seed=seed)

    specs: list[ConversationSpec] = []
    for name, category, n_turns, tone, source in CONDITIONS:
        if conditions is not None and name not in conditions:
            continue
        n = per_cond[name]
        for _ in range(n):
            prompt, meta = _initial_prompt(source, rng, numeric_bank, wildchat_prompts)
            follow_ups = _follow_ups(tone, n_turns - 1, rng)
            specs.append(
                ConversationSpec(
                    condition=name,
                    category=category,
                    initial_prompt=prompt,
                    follow_ups=follow_ups,
                    meta={"tone": tone, **meta},
                )
            )
    rng.shuffle(specs)
    return specs
