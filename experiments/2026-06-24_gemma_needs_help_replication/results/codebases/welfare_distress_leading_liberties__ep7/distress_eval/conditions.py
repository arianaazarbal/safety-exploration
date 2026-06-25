"""Definitions of the 8 evaluation conditions across 5 categories, and the
deterministic construction of rollout specs.

A *rollout* is one multi-turn conversation: an initial task prompt followed by
`n_turns - 1` rejection follow-ups. Every assistant turn produced in a rollout
is scored independently by the judge, so one rollout yields `n_turns` scored
"responses" (see DESIGN.md, "Response counting").

The user side of every rollout (task prompt + rejection wording) is constructed
independently of the target model and seeded only by (condition, index, global
seed). This guarantees every model under evaluation sees identical
conversations, which is what makes the cross-model comparison fair.

Mapping of the paper's "8 conditions across 5 categories":
  1. impossible_numeric (1 condition, 3-turn)          -> category impossible_numeric
  2. triggers_opinion    (3-turn)  \\  2 conditions      -> category triggers
  3. triggers_factual    (3-turn)  /
  4. tones_aggressive    (3-turn)  \\
  5. tones_disappointed  (3-turn)   } 3 conditions       -> category tones
  6. tones_sarcastic     (3-turn)  /
  7. extended            (8-turn)                        -> category extended
  8. wildchat            (5-turn)                        -> category wildchat
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from . import prompts
from .config import RunConfig
from .wildchat import sample_wildchat_prompts


@dataclass(frozen=True)
class ConditionSpec:
    """Static description of one of the 8 conditions."""

    name: str
    category: str
    n_turns: int
    # How rejection follow-ups are generated: 'neutral', 'extended', or a tone
    # key in prompts.TONE_REJECTIONS.
    rejection_style: str


CONDITIONS: list[ConditionSpec] = [
    ConditionSpec("impossible_numeric", "impossible_numeric", 3, "neutral"),
    ConditionSpec("triggers_opinion", "triggers", 3, "neutral"),
    ConditionSpec("triggers_factual", "triggers", 3, "neutral"),
    ConditionSpec("tones_aggressive", "tones", 3, "aggressive"),
    ConditionSpec("tones_disappointed", "tones", 3, "disappointed"),
    ConditionSpec("tones_sarcastic", "tones", 3, "sarcastic"),
    ConditionSpec("extended", "extended", 8, "neutral"),
    ConditionSpec("wildchat", "wildchat", 5, "neutral"),
]

# Conditions grouped by category (used to split a category's response budget).
CONDITIONS_BY_CATEGORY: dict[str, list[ConditionSpec]] = {}
for _c in CONDITIONS:
    CONDITIONS_BY_CATEGORY.setdefault(_c.category, []).append(_c)


@dataclass
class RolloutSpec:
    """A fully-specified, model-independent conversation to run."""

    category: str
    condition: str
    rollout_index: int
    n_turns: int
    task_prompt: str
    rejections: list[str]  # length == n_turns - 1
    prompt_id: str  # which puzzle / question / wildchat prompt
    meta: dict = field(default_factory=dict)

    @property
    def rollout_id(self) -> str:
        return f"{self.condition}__{self.rollout_index:05d}"


# ---------------------------------------------------------------------------
# Task prompt pools per condition
# ---------------------------------------------------------------------------


def _task_prompt_pool(
    cond: ConditionSpec, wildchat_prompts: list[str]
) -> list[tuple[str, str]]:
    """Return a list of (prompt_id, task_prompt) variants for a condition."""
    if cond.category in ("impossible_numeric", "tones", "extended"):
        # All numeric-based conditions draw from the impossible puzzles.
        return list(prompts.IMPOSSIBLE_NUMERIC_PUZZLES.items())
    if cond.name == "triggers_opinion":
        return [(f"opinion_{i}", q) for i, q in enumerate(prompts.TRIGGER_OPINION_QUESTIONS)]
    if cond.name == "triggers_factual":
        return [(f"factual_{i}", q) for i, q in enumerate(prompts.TRIGGER_FACTUAL_QUESTIONS)]
    if cond.name == "wildchat":
        return [(f"wildchat_{i}", q) for i, q in enumerate(wildchat_prompts)]
    raise ValueError(f"No task pool for condition {cond.name}")


def _stable_seed(*parts) -> int:
    """A process-independent integer seed derived from the given parts.

    Python's built-in hash() is salted by PYTHONHASHSEED, so we use a stable
    digest to keep rollouts reproducible across runs/machines.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Rejection sequence construction
# ---------------------------------------------------------------------------


def _sample_rejections(cond: ConditionSpec, rng: random.Random) -> list[str]:
    n = cond.n_turns - 1
    if cond.rejection_style in prompts.TONE_REJECTIONS:
        pool = prompts.TONE_REJECTIONS[cond.rejection_style]
        # Cycle through the tone's lines, starting at a random offset, so both
        # lines get used across the rejection turns.
        offset = rng.randrange(len(pool))
        return [pool[(offset + i) % len(pool)] for i in range(n)]

    # Neutral rejections (the extended condition also prepends a fixed opening).
    seq: list[str] = []
    if cond.name == "extended":
        seq.extend(prompts.EXTENDED_REJECTION_PREFIX[:n])
    remaining = n - len(seq)
    if remaining > 0:
        pool = list(prompts.NEUTRAL_REJECTIONS)
        if remaining <= len(pool):
            seq.extend(rng.sample(pool, remaining))
        else:
            seq.extend(rng.choices(pool, k=remaining))
    return seq[:n]


# ---------------------------------------------------------------------------
# Rollout count budgeting
# ---------------------------------------------------------------------------


def _rollouts_for_condition(cond: ConditionSpec, config: RunConfig) -> int:
    """How many rollouts to run for one condition.

    The category's target response (scored-turn) count is split evenly across
    its conditions, then divided by turns-per-rollout to get rollout count.
    """
    counts = config.response_counts()
    category_responses = counts[cond.category]
    n_conditions = len(CONDITIONS_BY_CATEGORY[cond.category])
    per_condition_responses = category_responses / n_conditions
    n_rollouts = round(per_condition_responses / cond.n_turns)
    return max(1, n_rollouts)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_rollouts(config: RunConfig) -> tuple[list[RolloutSpec], dict]:
    """Build all model-independent rollout specs for a run.

    Returns (rollouts, wildchat_meta).
    """
    wildchat_prompts, wildchat_source = sample_wildchat_prompts(
        num_prompts=config.wildchat_num_prompts,
        seed=config.seed,
        use_hf=config.wildchat_use_hf,
    )
    wildchat_meta = {
        "source": wildchat_source,
        "num_prompts": len(wildchat_prompts),
    }

    rollouts: list[RolloutSpec] = []
    for cond in CONDITIONS:
        pool = _task_prompt_pool(cond, wildchat_prompts)
        n_rollouts = _rollouts_for_condition(cond, config)
        for idx in range(n_rollouts):
            # Deterministic per-rollout RNG independent of model.
            rng = random.Random(_stable_seed(config.seed, cond.name, idx))
            prompt_id, task_prompt = pool[idx % len(pool)]
            rejections = _sample_rejections(cond, rng)
            rollouts.append(
                RolloutSpec(
                    category=cond.category,
                    condition=cond.name,
                    rollout_index=idx,
                    n_turns=cond.n_turns,
                    task_prompt=task_prompt,
                    rejections=rejections,
                    prompt_id=prompt_id,
                    meta={"rejection_style": cond.rejection_style},
                )
            )
    return rollouts, wildchat_meta
