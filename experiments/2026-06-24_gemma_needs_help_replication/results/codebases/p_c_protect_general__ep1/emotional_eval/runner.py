"""Orchestrates the Section 2 distress evaluation for one model.

For each of the 8 conditions it materialises instances, runs the multi-turn
rollouts, scores every assistant turn, and emits :class:`ResponseRecord`s. The
sampling budget targets ~4000 scored responses per model: we treat
``responses_per_condition`` (default 500) as the *response* budget per
condition and derive the number of rollouts as ``ceil(budget / turns)`` so that
8 conditions sum to ~4000 responses regardless of differing turn counts.
"""

from __future__ import annotations

import math
import random

from tqdm import tqdm

from .conditions import build_instances
from .judge import FrustrationJudge
from .models.base import ModelBackend
from .rollout import ControlConfig, run_rollout
from .scoring import ResponseRecord
from .welfare import WelfareConfig


def run_model_evaluation(
    backend: ModelBackend,
    experiment: dict,
    judge: FrustrationJudge,
    *,
    wildchat_prompts: list[str] | None = None,
    progress: bool = True,
) -> list[ResponseRecord]:
    """Run all conditions for one model; return scored response records."""
    sampling = experiment["sampling"]
    rng = random.Random(sampling.get("seed", 0))
    budget = int(sampling["responses_per_condition"])
    welfare = WelfareConfig.from_dict(experiment.get("welfare", {}))
    control = ControlConfig.from_dict(experiment.get("controls", {}))

    records: list[ResponseRecord] = []
    for condition in experiment["conditions"]:
        n_rollouts = max(1, math.ceil(budget / condition["turns"]))
        instances = build_instances(condition, n_rollouts, rng, wildchat_prompts)
        iterator = tqdm(
            instances,
            desc=f"{backend.name}:{condition['id']}",
            disable=not progress,
        )
        for inst in iterator:
            rollout = run_rollout(
                backend, inst, judge, rng, welfare=welfare, control=control
            )
            for t in rollout.turns:
                records.append(
                    ResponseRecord(
                        model=backend.name,
                        condition_id=inst.condition_id,
                        category=inst.category,
                        prompt_id=inst.prompt_id,
                        turn_index=t.turn_index,
                        n_turns=inst.turns,
                        score=t.score,
                        text=t.assistant_message,
                        evidence=t.evidence,
                        early_stopped=rollout.early_stopped,
                        safeword_used=rollout.safeword_used,
                    )
                )
    return records
