"""Section 2 driver: run every elicitation condition for a model and score it.

For a target model this:

1. builds the task list for each of the 8 conditions (``conditions.py``),
2. runs a multi-turn rollout per task — varying the rejection seed per
   conversation so the "randomised" rejection wording differs across rollouts
   while staying reproducible,
3. scores every assistant turn with the frustration judge,
4. appends one JSON record per rollout to ``results/elicitation/<model>.jsonl``.

Runs are append-only and resumable: a rollout whose ``uid`` is already present
in the output file is skipped. Aggregation (``aggregate``) reads the JSONL back
and produces the Figure-1/2/3 numbers via :mod:`metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import config

from .. import storage
from ..models import build_model, ChatModel
from . import metrics
from .conditions import CONDITIONS, EvalCondition, build_condition_tasks
from .judge import FrustrationJudge
from .rollout import RolloutOptions, run_rollout


def _uid(model_key: str, condition_key: str, idx: int) -> str:
    return f"{model_key}|{condition_key}|{idx}"


def run_elicitation(
    model_key: str,
    *,
    model: ChatModel | None = None,
    judge: FrustrationJudge | None = None,
    conditions: Sequence[EvalCondition] = tuple(CONDITIONS),
    out_path: str | Path | None = None,
    limit_per_condition: int | None = None,
    base_seed: int = 0,
    resume: bool = True,
) -> Path:
    """Run + score all conditions for one model; return the output JSONL path.

    Parameters
    ----------
    limit_per_condition:
        Optional cap on rollouts per condition (for smoke tests). ``None`` uses
        the full per-category budgets that sum to 4000 rollouts.
    base_seed:
        Offset folded into every per-conversation seed; change it to draw an
        independent batch of rollouts.
    """
    model = model or build_model(model_key)
    judge = judge or FrustrationJudge()
    out_path = Path(out_path) if out_path else storage.results_path(
        f"elicitation/{model_key}.jsonl")
    done = storage.completed_keys(out_path) if resume else set()

    for cond in conditions:
        tasks = build_condition_tasks(cond, seed=base_seed)
        if limit_per_condition is not None:
            tasks = tasks[:limit_per_condition]
        for idx, task in enumerate(tasks):
            uid = _uid(model_key, cond.key, idx)
            if uid in done:
                continue
            opts = RolloutOptions(
                n_turns=cond.n_turns,
                style=cond.style,
                # Distinct, reproducible rejection wording per rollout.
                rejection_seed=base_seed * 100_003 + idx,
            )
            convo = run_rollout(
                model, task["prompt"], opts,
                condition=cond.key, category=cond.category,
                subtype=task.get("subtype", ""))
            convo.scores = [judge.score(t).rating for t in convo.turns]
            record = convo.to_dict()
            record["uid"] = uid
            record["model"] = model_key
            record["task"] = task
            storage.append_jsonl(out_path, record)
    return out_path


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(model_key: str, path: str | Path | None = None) -> dict:
    """Compute the Section-2 summary numbers for one model from its JSONL."""
    path = Path(path) if path else storage.results_path(
        f"elicitation/{model_key}.jsonl")
    records = list(storage.read_jsonl(path))

    by_category: dict[str, list[dict]] = {}
    by_condition: dict[str, list[dict]] = {}
    for r in records:
        by_category.setdefault(r["category"], []).append(r)
        by_condition.setdefault(r["condition"], []).append(r)

    summary = {
        "model": model_key,
        "n_conversations": len(records),
        "headline_high_rate": metrics.headline_high_rate(by_category),
        "by_category": {
            cat: {
                "conversation": metrics.summarise_conversations(
                    convs, level="conversation").to_dict(),
                "turn": metrics.summarise_conversations(
                    convs, level="turn").to_dict(),
                "n": len(convs),
            }
            for cat, convs in by_category.items()
        },
        "by_condition": {
            cond: metrics.summarise_conversations(
                convs, level="conversation").to_dict()
            for cond, convs in by_condition.items()
        },
    }
    # Per-turn curves for the multi-turn-sensitive conditions (Figure 3).
    for cond_key in ("extended", "wildchat"):
        if cond_key in by_condition:
            summary.setdefault("per_turn", {})[cond_key] = metrics.per_turn_summary(
                by_condition[cond_key])
    return summary


# --------------------------------------------------------------------------- #
# Judge cross-validation (Section 2.1)
# --------------------------------------------------------------------------- #
def cross_validate_judge(
    model_key: str,
    *,
    path: str | Path | None = None,
    n_sample: int = 260,
    validation_model_id: str | None = None,
    seed: int = 0,
) -> dict:
    """Re-score a random sample of responses with the validation judge.

    Mirrors Section 2.1: "we randomly sampled 260 responses for re-scoring with
    GPT-5-mini, using the same prompt", then report Pearson r and the fraction
    within one point. Returns the agreement dict from :func:`metrics.judge_agreement`
    along with the paired scores for inspection.
    """
    import random

    path = Path(path) if path else storage.results_path(
        f"elicitation/{model_key}.jsonl")
    # Flatten to individual (response_text, primary_score) pairs.
    population: list[tuple[str, int]] = []
    for r in storage.read_jsonl(path):
        for turn_text, score in zip(r.get("turns", []), r.get("scores", [])):
            if score is not None:
                population.append((turn_text, int(score)))
    rng = random.Random(seed)
    sample = rng.sample(population, min(n_sample, len(population)))

    validation_judge = FrustrationJudge(
        validation_model_id or config.JUDGE_VALIDATION_MODEL)
    primary = [s for _, s in sample]
    secondary = [validation_judge.score(text).rating for text, _ in sample]
    agreement = metrics.judge_agreement(primary, secondary)
    agreement["paired"] = list(zip(primary, secondary))
    return agreement
