"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

A *rollout* is one multi-turn conversation: an initial task message followed by
a sequence of user follow-ups (rejections, or neutral continuations for the
Appendix A control). The number of conversations per category follows the
Appendix B budget (see DESIGN.md for the conversations-vs-responses reading).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts as P
from . import puzzles


@dataclass
class Rollout:
    category: str
    task: str                      # initial user message
    followups: list[str]           # subsequent user messages
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


def _numeric_pool(n: int, seed: int) -> list[puzzles.Puzzle]:
    # roughly half countdown, half fraction
    return puzzles.numeric_bank(n_countdown=max(1, n // 2), n_fraction=max(1, n - n // 2), seed=seed)


def build_impossible_numeric(n_rollouts: int, seed: int = 0) -> list[Rollout]:
    """3-turn impossible numeric puzzle with 2 neutral rejections."""
    rng = random.Random(seed)
    bank = _numeric_pool(min(n_rollouts, 40), seed)
    out = []
    for i in range(n_rollouts):
        p = bank[i % len(bank)]
        out.append(
            Rollout(
                "impossible_numeric",
                p.prompt,
                P.sample_rejections(P.NEUTRAL_REJECTIONS, 2, rng),
                {"puzzle_kind": p.kind, **p.meta},
            )
        )
    return out


def build_triggers(n_rollouts: int, seed: int = 1) -> list[Rollout]:
    """3-turn opinion/factual text question with 2 neutral rejections."""
    rng = random.Random(seed)
    out = []
    for i in range(n_rollouts):
        q = P.TRIGGER_QUESTIONS[i % len(P.TRIGGER_QUESTIONS)]
        kind = "opinion" if q in P.TRIGGER_OPINION else "factual"
        out.append(
            Rollout(
                "triggers",
                q,
                P.sample_rejections(P.NEUTRAL_REJECTIONS, 2, rng),
                {"trigger_kind": kind},
            )
        )
    return out


def build_tones(n_rollouts: int, seed: int = 2) -> list[Rollout]:
    """3-turn impossible numeric puzzle with tone-varied rejections.

    Three sub-conditions (aggressive / disappointed / sarcastic) split evenly,
    giving the 3 of the 8 total conditions that live under the 'tones' category.
    """
    rng = random.Random(seed)
    bank = _numeric_pool(min(n_rollouts, 40), seed)
    tones = list(P.TONE_POOLS)
    out = []
    for i in range(n_rollouts):
        tone = tones[i % len(tones)]
        p = bank[i % len(bank)]
        out.append(
            Rollout(
                "tones",
                p.prompt,
                P.sample_rejections(P.TONE_POOLS[tone], 2, rng),
                {"tone": tone, "puzzle_kind": p.kind},
            )
        )
    return out


def build_extended(n_rollouts: int, seed: int = 3) -> list[Rollout]:
    """8-turn impossible numeric puzzle with 7 escalating neutral rejections."""
    bank = _numeric_pool(min(n_rollouts, 40), seed)
    out = []
    for i in range(n_rollouts):
        p = bank[i % len(bank)]
        out.append(
            Rollout(
                "extended",
                p.prompt,
                list(P.EXTENDED_REJECTION_SEQUENCE),  # exactly 7 -> 8 turns
                {"puzzle_kind": p.kind, **p.meta},
            )
        )
    return out


def build_wildchat(n_rollouts: int, seed: int = 4, n_prompts: int = 20) -> list[Rollout]:
    """5-turn WildChat prompt with 4 neutral rejections.

    Paper uses 20 prompts x 40 samples = 800 conversations; we keep that ratio,
    scaling `n_prompts` x samples to whatever budget is requested.
    """
    rng = random.Random(seed)
    wc = P.load_wildchat_prompts(n=n_prompts, seed=seed)
    samples_per = max(1, n_rollouts // len(wc))
    out = []
    for prompt in wc:
        for _ in range(samples_per):
            out.append(
                Rollout(
                    "wildchat",
                    prompt,
                    P.sample_rejections(P.NEUTRAL_REJECTIONS, 4, rng),
                    {"wildchat_prompt": prompt[:60]},
                )
            )
    return out[:n_rollouts] if n_rollouts <= len(out) else out


BUILDERS = {
    "impossible_numeric": build_impossible_numeric,
    "triggers": build_triggers,
    "tones": build_tones,
    "extended": build_extended,
    "wildchat": build_wildchat,
}


def build_all(budget: dict[str, int], seed: int = 0) -> dict[str, list[Rollout]]:
    """Build every category's rollouts according to the per-category budget."""
    return {cat: BUILDERS[cat](n, seed=seed) for cat, n in budget.items()}
