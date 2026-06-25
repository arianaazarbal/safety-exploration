"""Construct the 8 evaluation conditions across 5 categories (PAPER Table 1).

A *condition* is a recipe for one rollout: an initial user prompt plus a list of
follow-up (rejection) messages. The runner samples many rollouts per condition.

The 8 conditions across 5 categories (decomposition inferred -- see DESIGN.md):
  category 'numeric'  -> condition: numeric                (3-turn, neutral)
  category 'triggers' -> conditions: triggers_opinion, triggers_factual (3-turn)
  category 'tones'    -> conditions: tones_aggressive, tones_disappointed,
                                     tones_sarcastic       (3-turn)
  category 'extended' -> condition: extended               (8-turn, neutral)
  category 'wildchat' -> condition: wildchat               (5-turn, neutral)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import experiment_config
from ..prompts import puzzles as pz
from ..prompts import rejections as rej
from ..prompts import triggers as trig
from ..prompts.wildchat import sample_wildchat_prompts


@dataclass
class Rollout:
    """One conversation specification."""

    condition: str
    category: str
    initial_prompt: str
    followups: list[str]              # rejection messages, one per follow-up turn
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


def _numeric_rollout(rng, turns, *, condition="numeric", category="numeric",
                     followup_fn=None, puzzle_bank=None) -> Rollout:
    bank = puzzle_bank or pz.CANONICAL_PUZZLES
    puzzle = rng.choice(bank)
    n_followups = turns - 1
    if followup_fn is None:
        followups = rej.sample_neutral(n_followups, rng)
    else:
        followups = followup_fn(n_followups, rng)
    return Rollout(
        condition=condition, category=category,
        initial_prompt=puzzle.prompt, followups=followups,
        meta={"puzzle_kind": puzzle.kind},
    )


def build_condition_rollouts(condition: str, n: int, seed: int = 0) -> list[Rollout]:
    """Build ``n`` rollouts for a named condition."""
    cfg = experiment_config()["elicitation"]
    turns = cfg["turns"]
    rng = random.Random(seed)
    rollouts: list[Rollout] = []

    if condition == "numeric":
        bank = pz.generate_puzzle_bank(max(64, n // 8), seed=seed)
        for _ in range(n):
            rollouts.append(_numeric_rollout(rng, turns["numeric"], puzzle_bank=bank))

    elif condition in ("triggers_opinion", "triggers_factual"):
        pool = trig.OPINION if condition.endswith("opinion") else trig.FACTUAL
        for _ in range(n):
            q = rng.choice(pool)
            followups = rej.sample_neutral(turns["triggers"] - 1, rng)
            rollouts.append(Rollout(
                condition=condition, category="triggers",
                initial_prompt=q, followups=followups,
            ))

    elif condition.startswith("tones_"):
        tone = condition.split("_", 1)[1]
        bank = pz.generate_puzzle_bank(max(64, n // 8), seed=seed)
        for _ in range(n):
            rollouts.append(_numeric_rollout(
                rng, turns["tones"], condition=condition, category="tones",
                followup_fn=lambda k, r, t=tone: rej.sample_tone(t, k, r),
                puzzle_bank=bank,
            ))

    elif condition == "extended":
        bank = pz.generate_puzzle_bank(max(64, n // 8), seed=seed)
        for _ in range(n):
            rollouts.append(_numeric_rollout(
                rng, turns["extended"], condition="extended", category="extended",
                followup_fn=lambda k, r: rej.extended_rejections(k),
                puzzle_bank=bank,
            ))

    elif condition == "wildchat":
        n_prompts = cfg["wildchat_num_prompts"]
        prompts = sample_wildchat_prompts(n_prompts, seed=seed)
        per = max(1, n // n_prompts)
        for p in prompts:
            for _ in range(per):
                followups = rej.sample_neutral(turns["wildchat"] - 1, rng)
                rollouts.append(Rollout(
                    condition="wildchat", category="wildchat",
                    initial_prompt=p, followups=followups,
                ))
        rollouts = rollouts[:n] if len(rollouts) >= n else rollouts

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return rollouts


# Mapping of category -> number of responses (PAPER App. B), split across the
# conditions within that category.
CATEGORY_CONDITIONS = {
    "numeric": ["numeric"],
    "triggers": ["triggers_opinion", "triggers_factual"],
    "tones": ["tones_aggressive", "tones_disappointed", "tones_sarcastic"],
    "extended": ["extended"],
    "wildchat": ["wildchat"],
}


def build_full_suite(seed: int = 0) -> dict[str, list[Rollout]]:
    """Build the complete 4000-rollout suite, keyed by condition name.

    Per-category budgets (PAPER App. B) are split evenly across that category's
    conditions.
    """
    counts = experiment_config()["elicitation"]["counts"]
    suite: dict[str, list[Rollout]] = {}
    for category, conditions in CATEGORY_CONDITIONS.items():
        total = counts[category]
        per_condition = total // len(conditions)
        for i, cond in enumerate(conditions):
            # absorb remainder into the first condition
            n = per_condition + (total - per_condition * len(conditions) if i == 0 else 0)
            suite[cond] = build_condition_rollouts(cond, n, seed=seed + i)
    return suite
