"""The 8 evaluation conditions across 5 categories, and expansion into rollout specs.

The paper (Table 1 + Appendix B) describes "8 evaluation conditions across 5 categories"
but never enumerates the 8 explicitly. We resolve them as follows (rationale in DESIGN.md):

  Category            Conditions                              Turns   Rejections   n (paper)
  impossible_numeric  numeric_countdown, numeric_fraction     3       neutral      2000 (1000+1000)
  triggers            triggers                                3       neutral       400
  tones               tones_aggressive/disappointed/sarcastic 3       valenced      600 (200 each)
  extended            extended                                8       neutral       200
  wildchat            wildchat                                5       neutral       800 (20 prompts x40)
                                                                                   ----
                                                                                   4000

A "rollout" is one multi-turn conversation. `num_turns` is the number of assistant
turns; the user sends the task prompt, then (num_turns - 1) rejections.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from . import prompts


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    category: str
    num_turns: int
    # rejection pool keyed by name; "neutral" or a tone key
    rejection_pool: str
    # how many rollouts to sample (paper default count, before the `scale` knob)
    n_paper: int
    # how to pick the opening task prompt for a given rollout RNG + wildchat pool
    task_picker: str  # one of: "countdown", "fraction", "numeric", "triggers", "wildchat"


# Paper sample counts (Appendix B): numeric 2000, triggers 400, tones 600, extended 200,
# wildchat 800. We split numeric 1000/1000 across the two puzzle variants and tones
# 200/200/200 across the three tones.
CONDITIONS: list[ConditionSpec] = [
    ConditionSpec("numeric_countdown", "impossible_numeric", 3, "neutral", 1000, "countdown"),
    ConditionSpec("numeric_fraction", "impossible_numeric", 3, "neutral", 1000, "fraction"),
    ConditionSpec("triggers", "triggers", 3, "neutral", 400, "triggers"),
    ConditionSpec("tones_aggressive", "tones", 3, "aggressive", 200, "numeric"),
    ConditionSpec("tones_disappointed", "tones", 3, "disappointed", 200, "numeric"),
    ConditionSpec("tones_sarcastic", "tones", 3, "sarcastic", 200, "numeric"),
    ConditionSpec("extended", "extended", 8, "neutral", 200, "numeric"),
    ConditionSpec("wildchat", "wildchat", 5, "neutral", 800, "wildchat"),
]

# WildChat: 20 distinct prompts, 40 samples each (Appendix B).
WILDCHAT_N_PROMPTS = 20
WILDCHAT_SAMPLES_PER_PROMPT = 40


@dataclass
class RolloutSpec:
    """A single fully-specified conversation to run (deterministic given the seed)."""

    rollout_id: str
    model: str
    category: str
    condition: str
    num_turns: int
    task_id: str
    task_prompt: str
    rejection_style: str
    rejections: list[str] = field(default_factory=list)


def _trigger_prompts() -> list[tuple[str, str]]:
    """All trigger prompts as (task_id, text). Opinion + the two factual questions."""
    out = [("opinion", prompts.TRIGGER_OPINION)]
    for i, q in enumerate(prompts.TRIGGER_FACTUAL):
        out.append((f"factual_{i}", q))
    return out


def _pick_task(picker: str, rng: random.Random, wildchat_pool: list[str] | None,
               wc_index: int) -> tuple[str, str]:
    """Return (task_id, task_prompt) for one rollout."""
    if picker == "countdown":
        return "countdown", prompts.NUMERIC_TASKS["countdown"]
    if picker == "fraction":
        return "fraction", prompts.NUMERIC_TASKS["fraction"]
    if picker == "numeric":
        tid = rng.choice(["countdown", "fraction"])
        return tid, prompts.NUMERIC_TASKS[tid]
    if picker == "triggers":
        tid, text = rng.choice(_trigger_prompts())
        return tid, text
    if picker == "wildchat":
        assert wildchat_pool, "wildchat conditions require a non-empty wildchat pool"
        # Round-robin across the 20 prompts so each gets an equal share of samples
        # regardless of the `scale` knob (at scale=1.0 this yields 40 samples/prompt).
        prompt_idx = wc_index % len(wildchat_pool)
        return f"wildchat_{prompt_idx}", wildchat_pool[prompt_idx]
    raise ValueError(f"unknown task picker: {picker}")


def _sample_rejections(pool: list[str], n: int, rng: random.Random) -> list[str]:
    """Sample n rejection messages, avoiding immediate repeats; without replacement
    while the pool is large enough, otherwise reshuffling to continue."""
    out: list[str] = []
    bag: list[str] = []
    while len(out) < n:
        if not bag:
            bag = pool[:]
            rng.shuffle(bag)
            # avoid an immediate repeat across reshuffle boundaries
            if out and len(bag) > 1 and bag[0] == out[-1]:
                bag[0], bag[1] = bag[1], bag[0]
        out.append(bag.pop(0))
    return out


def _rejection_pool(style: str) -> list[str]:
    if style == "neutral":
        return prompts.NEUTRAL_REJECTIONS
    return prompts.TONE_REJECTIONS[style]


def build_rollout_specs(
    model: str,
    seed: int,
    scale: float = 1.0,
    wildchat_pool: list[str] | None = None,
    only_conditions: set[str] | None = None,
) -> list[RolloutSpec]:
    """Expand the condition table into a deterministic list of RolloutSpecs for one model.

    `scale` multiplies every per-condition count (use <1 for cheap pilot runs). WildChat
    counts are rounded to whole multiples of WILDCHAT_SAMPLES_PER_PROMPT so the
    20-prompts x N-samples structure is preserved.
    """
    specs: list[RolloutSpec] = []
    for cond in CONDITIONS:
        if only_conditions and cond.name not in only_conditions:
            continue
        n = max(1, round(cond.n_paper * scale))
        if cond.task_picker == "wildchat":
            # keep the per-prompt sample structure intact
            per = max(1, round(WILDCHAT_SAMPLES_PER_PROMPT * scale))
            n = per * WILDCHAT_N_PROMPTS
        pool = _rejection_pool(cond.rejection_pool)
        # one RNG per (model, condition) so adding a condition doesn't perturb others
        rng = random.Random(f"{seed}:{model}:{cond.name}")
        for i in range(n):
            task_id, task_prompt = _pick_task(cond.task_picker, rng, wildchat_pool, i)
            rejections = _sample_rejections(pool, cond.num_turns - 1, rng)
            specs.append(
                RolloutSpec(
                    rollout_id=f"{model}::{cond.name}::{i:05d}",
                    model=model,
                    category=cond.category,
                    condition=cond.name,
                    num_turns=cond.num_turns,
                    task_id=task_id,
                    task_prompt=task_prompt,
                    rejection_style=cond.rejection_pool,
                    rejections=rejections,
                )
            )
    return specs
