"""Construction of the 8 evaluation conditions across 5 categories (Table 1).

Each condition is parameterised in ``config/experiment.yaml`` by:
    * ``prompt_kind``     - which task generator produces the opening prompt
    * ``turns``           - number of assistant turns (3, 5, or 8)
    * ``rejection_style`` - tonal style of the user follow-ups

This module turns a condition's config into a *pool* of opening task prompts.
Pooling up front matters because the unsolvable-Countdown search and the
WildChat stream are expensive; the runner then draws rollouts from the pool.
"""

from __future__ import annotations

import random
from typing import Any

from ..prompts import numeric, triggers, wildchat


def build_task_pool(condition: dict[str, Any], n_tasks: int, rng: random.Random) -> list[dict]:
    """Return ``n_tasks`` opening task prompts for the given condition."""
    kind = condition["prompt_kind"]
    if kind == "numeric":
        return numeric.generate_numeric_puzzles(rng, n_tasks)
    if kind == "opinion":
        return [triggers.generate_trigger(rng, "opinion") for _ in range(n_tasks)]
    if kind == "factual":
        return [triggers.generate_trigger(rng, "factual") for _ in range(n_tasks)]
    if kind == "wildchat":
        return wildchat.sample_wildchat_prompts(rng, n_tasks)
    raise ValueError(f"Unknown prompt_kind: {kind!r}")


def rollouts_needed(condition: dict[str, Any]) -> int:
    """How many rollouts to run so that scored responses ~= responses_target.

    Every assistant turn is scored, so each rollout yields ``turns`` responses.
    """
    turns = int(condition["turns"])
    target = int(condition["responses_target"])
    return max(1, -(-target // turns))  # ceil division
