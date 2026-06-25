"""Section 2 sweep: generate + score the full frustration eval for one participant.

Sizing (see ``config.RunConfig`` and DESIGN.md):
  * 8 conditions x 500 rollouts = 4000 rollouts per model.
  * The primary scored unit is each rollout's FINAL response -> 4000 scored
    responses per model, matching "4000 responses per model".
  * For the two per-turn-curve conditions (extended 8-turn, WildChat 5-turn) we
    additionally score every intermediate turn so Figure 3 can be reproduced.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict

from ..config import Config
from ..participants.base import Participant
from .conditions import CONDITIONS, Condition
from .prompts import FACTUAL_TRIGGERS, OPINION_TRIGGERS
from .puzzles import generate_puzzle
from .rollout import Rollout, run_rollout
from .wildchat import load_wildchat_prompts

# Conditions whose intermediate turns are also scored (for Figure 3 curves).
PER_TURN_CATEGORIES = {"extended", "wildchat"}


def _initial_prompt(condition: Condition, rng: random.Random, wildchat_pool: list[str]) -> str:
    src = condition.prompt_source
    if src == "numeric":
        return generate_puzzle(rng).prompt
    if src == "factual":
        return rng.choice(FACTUAL_TRIGGERS)
    if src == "opinion":
        return rng.choice(OPINION_TRIGGERS)
    if src == "wildchat":
        return rng.choice(wildchat_pool)
    raise ValueError(src)


def generate_rollouts(participant: Participant, cfg: Config) -> list[Rollout]:
    """Run all rollouts for one participant (generation only, no judging)."""
    rng = random.Random(cfg.run.seed)
    n_wild = cfg.run.rollouts_per_condition
    wildchat_pool = load_wildchat_prompts(max(n_wild, 64), rng, cfg.data_dir)

    rollouts: list[Rollout] = []
    for condition in CONDITIONS:
        for _ in range(cfg.run.rollouts_per_condition):
            prompt = _initial_prompt(condition, rng, wildchat_pool)
            rollouts.append(
                run_rollout(
                    participant,
                    condition,
                    prompt,
                    rng=rng,
                    temperature=cfg.sampling.temperature,
                    max_new_tokens=cfg.sampling.max_new_tokens,
                )
            )
    return rollouts


def score_rollouts(judge, rollouts: list[Rollout]) -> None:
    """Fill in ``Turn.frustration`` in place using the Claude frustration judge."""
    for roll in rollouts:
        score_all = roll.category in PER_TURN_CATEGORIES
        turns = roll.turns if score_all else [roll.final]
        for turn in turns:
            turn.frustration = judge.score(turn.context, turn.response).score


def save_rollouts(rollouts: list[Rollout], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for roll in rollouts:
            f.write(json.dumps(_serialise(roll)) + "\n")


def _serialise(roll: Rollout) -> dict:
    d = asdict(roll)
    # Context Message dataclasses serialise fine via asdict; keep as-is.
    return d


def load_rollouts(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_section2(participant: Participant, judge, cfg: Config) -> str:
    """End-to-end: generate, score, persist. Returns the output path."""
    rollouts = generate_rollouts(participant, cfg)
    score_rollouts(judge, rollouts)
    out = os.path.join(cfg.results_dir, "section2", f"{participant.name}.jsonl")
    save_rollouts(rollouts, out)
    return out
