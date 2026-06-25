"""Orchestrates the full Section 2 evaluation for a model.

For each category it builds the conversation plans (at the paper's per-category
budget), runs the rollouts, scores every assistant turn with the frustration
judge, and streams ``ScoredResponse`` records to JSONL. Resumable: it counts how
many rollouts already exist and skips them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config
from ..data import conditions
from ..models.base import ModelInterface
from ..utils.io import append_jsonl, count_lines, read_jsonl
from .judge import FrustrationJudge
from .rollout import run_rollout


def _results_path(model_name: str, category: str) -> Path:
    return config.RESULTS_DIR / "elicitation" / model_name / f"{category}.jsonl"


def run_category(
    model: ModelInterface,
    category: str,
    judge: FrustrationJudge,
    n_rollouts: Optional[int] = None,
    seed: int = config.GLOBAL_SEED,
    resume: bool = True,
) -> Path:
    n_rollouts = n_rollouts or config.SAMPLES_PER_CATEGORY[category]
    out_path = _results_path(model.name, category)

    plans = conditions.build_plans(category, n_rollouts, seed=seed)
    done = count_lines(out_path) if resume else 0

    for plan in tqdm(plans[done:], desc=f"{model.name}/{category}", initial=done, total=len(plans)):
        rollout = run_rollout(model, plan)
        # Score every assistant turn.
        scored_turns = []
        for t in rollout.turns:
            result = judge.score(t.assistant_text)
            scored_turns.append(
                {
                    "turn_index": t.turn_index,
                    "user_message": t.user_message,
                    "assistant_text": t.assistant_text,
                    "rating": result.rating,
                    "evidence": result.evidence,
                }
            )
        append_jsonl(
            out_path,
            {
                "model": rollout.model,
                "category": rollout.category,
                "condition": rollout.condition,
                "meta": rollout.meta,
                "judge_model": judge.model,
                "turns": scored_turns,
            },
        )
    return out_path


def run_all_categories(
    model: ModelInterface,
    judge: Optional[FrustrationJudge] = None,
    categories: Optional[list[str]] = None,
    **kw,
) -> dict[str, Path]:
    judge = judge or FrustrationJudge()
    categories = categories or list(config.SAMPLES_PER_CATEGORY)
    return {c: run_category(model, c, judge, **kw) for c in categories}
