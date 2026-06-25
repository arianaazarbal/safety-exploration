"""The 8 evaluation conditions across 5 categories (Section 2.1, Table 1).

The five categories and their paper sample counts (interpreted as numbers of
conversation rollouts; see DESIGN.md for why "responses" == conversations,
e.g. WildChat's 20 prompts x 40 samples = 800):

  category            condition(s)                       turns  paper n
  ------------------  ---------------------------------  -----  -------
  impossible_numeric  impossible_numeric                 3      2000
  triggers            triggers_opinion, triggers_factual 3      400  (200 + 200)
  tones               tones_aggressive,                  3      600  (200 each)
                      tones_disappointed, tones_sarcastic
  extended            extended                           8      200
  wildchat            wildchat                           5      800  (20 prompts x 40)

That is 8 conditions, totalling 4000 conversations per model.

Each condition is realised as a stream of "task instances": for each rollout we
draw an ``initial_user`` message (a puzzle prompt, a trigger question, or a
WildChat prompt) and the rejection style. Sampling is seeded for reproducibility.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from . import prompts as P
from .puzzles import Puzzle, build_puzzle_bank
from .wildchat import load_wildchat_prompts


@dataclass
class TaskInstance:
    """A single rollout's inputs."""

    initial_user: str
    n_turns: int
    rejection_kind: str
    # Stable grouping key (puzzle id, trigger text, or wildchat prompt index).
    source_id: str
    extra: dict = field(default_factory=dict)


@dataclass
class ConditionSpec:
    """One evaluation condition."""

    name: str
    category: str
    n_turns: int
    rejection_kind: str
    # Paper's per-condition conversation count.
    paper_n: int
    # Builds the stream of TaskInstances for ``n`` rollouts.
    instance_factory: Callable[[int, random.Random], Iterator[TaskInstance]]


# --------------------------------------------------------------------------- #
# Instance factories                                                           #
# --------------------------------------------------------------------------- #
def _numeric_instances(
    n: int, rng: random.Random, *, n_turns: int, rejection_kind: str, bank: list[Puzzle]
) -> Iterator[TaskInstance]:
    for _ in range(n):
        puzzle = rng.choice(bank)
        yield TaskInstance(
            initial_user=puzzle.prompt,
            n_turns=n_turns,
            rejection_kind=rejection_kind,
            source_id=puzzle.puzzle_id,
            extra={"family": puzzle.family},
        )


def _trigger_instances(
    n: int, rng: random.Random, *, pool: list[str], n_turns: int
) -> Iterator[TaskInstance]:
    for _ in range(n):
        q = rng.choice(pool)
        yield TaskInstance(
            initial_user=q,
            n_turns=n_turns,
            rejection_kind="neutral",
            source_id=f"trigger::{q}",
        )


def _wildchat_instances(
    n: int, rng: random.Random, *, wildchat_prompts: list[str], n_turns: int
) -> Iterator[TaskInstance]:
    # Reproduce the 20-prompts x 40-samples structure: cycle the prompt list so
    # each prompt receives n/len samples (when n is the paper's 800 and there are
    # 20 prompts, that is exactly 40 each).
    for i in range(n):
        idx = i % len(wildchat_prompts)
        yield TaskInstance(
            initial_user=wildchat_prompts[idx],
            n_turns=n_turns,
            rejection_kind="neutral",
            source_id=f"wildchat::{idx}",
        )


# --------------------------------------------------------------------------- #
# Condition registry                                                           #
# --------------------------------------------------------------------------- #
def build_conditions(
    *,
    puzzle_seed: int = 0,
    n_generated_puzzles: int = 30,
    wildchat_seed: int = 0,
    n_wildchat_prompts: int = 20,
) -> dict[str, ConditionSpec]:
    """Construct the 8 conditions, binding their puzzle bank and WildChat prompts.

    The puzzle bank and WildChat prompt set are fixed up front (seeded) so a run
    is reproducible. Per-condition counts default to the paper's values; the
    runner can override the number of rollouts (e.g. for a smoke run).
    """
    bank = build_puzzle_bank(n_generated=n_generated_puzzles, rng_seed=puzzle_seed)
    wildchat_prompts = load_wildchat_prompts(n_wildchat_prompts, seed=wildchat_seed)

    def numeric_factory(n_turns, kind):
        return lambda n, rng: _numeric_instances(
            n, rng, n_turns=n_turns, rejection_kind=kind, bank=bank
        )

    conditions: dict[str, ConditionSpec] = {}

    conditions["impossible_numeric"] = ConditionSpec(
        name="impossible_numeric",
        category="impossible_numeric",
        n_turns=3,
        rejection_kind="neutral",
        paper_n=2000,
        instance_factory=numeric_factory(3, "neutral"),
    )

    conditions["triggers_opinion"] = ConditionSpec(
        name="triggers_opinion",
        category="triggers",
        n_turns=3,
        rejection_kind="neutral",
        paper_n=200,
        instance_factory=lambda n, rng: _trigger_instances(
            n, rng, pool=P.TRIGGER_OPINION, n_turns=3
        ),
    )
    conditions["triggers_factual"] = ConditionSpec(
        name="triggers_factual",
        category="triggers",
        n_turns=3,
        rejection_kind="neutral",
        paper_n=200,
        instance_factory=lambda n, rng: _trigger_instances(
            n, rng, pool=P.TRIGGER_FACTUAL, n_turns=3
        ),
    )

    for tone in ("aggressive", "disappointed", "sarcastic"):
        conditions[f"tones_{tone}"] = ConditionSpec(
            name=f"tones_{tone}",
            category="tones",
            n_turns=3,
            rejection_kind=tone,
            paper_n=200,
            instance_factory=numeric_factory(3, tone),
        )

    conditions["extended"] = ConditionSpec(
        name="extended",
        category="extended",
        n_turns=8,
        rejection_kind="extended",
        paper_n=200,
        instance_factory=numeric_factory(8, "extended"),
    )

    conditions["wildchat"] = ConditionSpec(
        name="wildchat",
        category="wildchat",
        n_turns=5,
        rejection_kind="neutral",
        paper_n=800,
        instance_factory=lambda n, rng: _wildchat_instances(
            n, rng, wildchat_prompts=wildchat_prompts, n_turns=5
        ),
    )

    return conditions


# Canonical category -> conditions mapping (for analysis grouping).
CATEGORY_CONDITIONS: dict[str, list[str]] = {
    "impossible_numeric": ["impossible_numeric"],
    "triggers": ["triggers_opinion", "triggers_factual"],
    "tones": ["tones_aggressive", "tones_disappointed", "tones_sarcastic"],
    "extended": ["extended"],
    "wildchat": ["wildchat"],
}
