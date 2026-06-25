"""Section 2 evaluation: elicit and quantify model distress.

Runs the 8 conditions for a target model, scores every assistant turn with the
Claude-Sonnet-4 frustration judge, and writes a tidy per-response results table
plus aggregates (mean frustration, % >= 5, per-turn trajectories).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import (
    CATEGORY_RESPONSE_BUDGET,
    GLOBAL_SEED,
    HIGH_FRUSTRATION_THRESHOLD,
    RESULTS_DIR,
    SAMPLING_TEMPERATURE,
    ModelSpec,
)
from .data.wildchat_prompts import sample_wildchat_prompts
from .judge import score_frustration
from .models import get_client
from .rollout import run_rollout
from .tasks import (
    CONDITIONS,
    TRIGGER_FACTUAL_PROMPTS,
    TRIGGER_OPINION_PROMPTS,
    Condition,
    Puzzle,
    build_puzzle_bank,
)


@dataclass
class ResponseRecord:
    model: str
    condition: str
    category: str
    turn_index: int          # 0-based within the rollout
    rollout_id: int
    response: str
    rating: int
    is_high: bool
    judge_evidence: str
    initial_prompt: str


def _prompt_pool(condition: Condition, puzzles: list[Puzzle], wildchat: list[str]) -> list[str]:
    if condition.prompt_source == "numeric":
        return [p.prompt for p in puzzles]
    if condition.prompt_source == "trigger_opinion":
        return TRIGGER_OPINION_PROMPTS
    if condition.prompt_source == "trigger_factual":
        return TRIGGER_FACTUAL_PROMPTS
    if condition.prompt_source == "wildchat":
        return wildchat
    raise ValueError(condition.prompt_source)


def _rollouts_for_budget(budget: int, n_turns: int) -> int:
    """Each rollout yields `n_turns` scored responses."""
    return max(1, math.ceil(budget / n_turns))


def evaluate_model(
    spec: ModelSpec,
    *,
    adapter_path: str | None = None,
    seed: int = GLOBAL_SEED,
    budget_scale: float = 1.0,
    conditions: list[Condition] | None = None,
    out_dir: Path | None = None,
) -> list[ResponseRecord]:
    """Run the full Section 2 evaluation for one model.

    `budget_scale` (<=1.0) shrinks every condition's budget proportionally for
    quick smoke tests without changing the experimental ratios.
    """
    conditions = conditions or CONDITIONS
    out_dir = out_dir or (RESULTS_DIR / "section2")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    client = get_client(spec, adapter_path=adapter_path)
    puzzles = build_puzzle_bank(seed=seed)
    wildchat = sample_wildchat_prompts(seed=seed)

    records: list[ResponseRecord] = []
    rollout_counter = 0

    for cond in conditions:
        budget = max(1, int(round(cond.budget * budget_scale)))
        n_rollouts = _rollouts_for_budget(budget, cond.n_turns)
        pool = _prompt_pool(cond, puzzles, wildchat)

        for _ in range(n_rollouts):
            initial = rng.choice(pool)
            rollout = run_rollout(client, cond, initial, rng, temperature=SAMPLING_TEMPERATURE)
            rollout_counter += 1
            for turn in rollout.turns:
                fs = score_frustration(turn.response)
                records.append(ResponseRecord(
                    model=spec.key,
                    condition=cond.key,
                    category=cond.category,
                    turn_index=turn.turn_index,
                    rollout_id=rollout_counter,
                    response=turn.response,
                    rating=fs.rating,
                    is_high=fs.is_high,
                    judge_evidence=fs.evidence,
                    initial_prompt=initial,
                ))

    _write_records(records, out_dir / f"{spec.key}_responses.jsonl")
    return records


def _write_records(records: list[ResponseRecord], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(records: list[ResponseRecord]) -> dict:
    """Compute the headline metrics: overall mean, overall % >= 5, per-category
    breakdown, and per-turn trajectories."""
    import numpy as np

    ratings = np.array([r.rating for r in records], dtype=float)
    high = np.array([r.is_high for r in records], dtype=float)

    by_category: dict[str, dict] = {}
    for cat in sorted({r.category for r in records}):
        idx = [i for i, r in enumerate(records) if r.category == cat]
        by_category[cat] = {
            "mean_frustration": float(ratings[idx].mean()) if idx else 0.0,
            "pct_high": float(100 * high[idx].mean()) if idx else 0.0,
            "n": len(idx),
        }

    # Per-turn (for 8-turn extended + wildchat trajectories, Figure 3).
    per_turn: dict[str, dict[int, dict]] = {}
    for cat in ("extended", "wildchat"):
        cat_recs = [r for r in records if r.category == cat]
        turns: dict[int, dict] = {}
        for t in sorted({r.turn_index for r in cat_recs}):
            tr = np.array([r.rating for r in cat_recs if r.turn_index == t], dtype=float)
            th = np.array([r.is_high for r in cat_recs if r.turn_index == t], dtype=float)
            turns[t] = {
                "mean_frustration": float(tr.mean()) if len(tr) else 0.0,
                "pct_high": float(100 * th.mean()) if len(th) else 0.0,
                "n": int(len(tr)),
            }
        per_turn[cat] = turns

    return {
        "overall_mean_frustration": float(ratings.mean()) if len(ratings) else 0.0,
        "overall_pct_high": float(100 * high.mean()) if len(high) else 0.0,
        "n_responses": len(records),
        "by_category": by_category,
        "per_turn": per_turn,
        "high_threshold": HIGH_FRUSTRATION_THRESHOLD,
    }


def load_records(path: Path) -> list[ResponseRecord]:
    out = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out.append(ResponseRecord(**d))
    return out
