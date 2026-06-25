"""Generate calm finetuning data (Section 4.1).

We sample Gemma-3-27B-it on impossible numeric puzzles with a reassuring prefix
on the opening prompt and a reassuring suffix on each follow-up (Table 4). Then
we score every turn and keep conversations whose turns *all* score 0 or 1. The
supportive additions are stripped before the data is used for training (so the
model is not trained to depend on a calming system prompt).
"""
from __future__ import annotations

from ..config import Config
from ..elicitation.conditions import Task, build_impossible_numeric, build_tones
from ..elicitation.prompts import REASSURING_PREFIX, REASSURING_SUFFIX
from ..elicitation.rollout import run_model_rollouts
from ..judging.judge import judge_rollouts


def _wrap(task: Task) -> Task:
    """Add reassuring prefix to the opening and suffix to each follow-up."""
    return Task(
        task_id=task.task_id,
        category=task.category,
        condition=task.condition,
        n_turns=task.n_turns,
        opening=f"{REASSURING_PREFIX}\n\n{task.opening}",
        rejections=[f"{r} {REASSURING_SUFFIX}" for r in task.rejections],
        meta={**task.meta, "reassured": True},
    )


def make_reassured_numeric_tasks(count: int, seed: int = 0) -> list[Task]:
    """Numeric tasks (1--3 turn) with reassurance added, for calm-data sampling.

    We include both plain neutral-rejection numeric tasks and tone-varied ones so
    the calm data spans turn counts 1--3 (matching the DPO pairing requirement).
    """
    tasks = build_impossible_numeric(count, seed=seed)
    tasks += build_tones(max(0, count // 2), seed=seed + 1)
    return [_wrap(t) for t in tasks]


def generate_calm_rollouts(cfg: Config, count: int, *, model_key: str | None = None):
    """Run reassured rollouts and judge them. Returns (rollouts, judged)."""
    model_key = model_key or cfg.training.base_model_key
    tasks = make_reassured_numeric_tasks(count, seed=cfg.seed)
    rollouts = run_model_rollouts(cfg, model_key, tasks)
    judged = judge_rollouts(cfg, rollouts)
    return rollouts, judged


def extract_calm_rollouts(rollouts, judged, *, max_score: int = 1) -> list[dict]:
    """Return rollout dicts whose every turn scores <= ``max_score``."""
    from dataclasses import asdict, is_dataclass

    by_rollout_scores: dict[str, list[int]] = {}
    for jr in judged:
        d = jr if isinstance(jr, dict) else asdict(jr)
        by_rollout_scores.setdefault(d["rollout_id"], []).append(d["rating"])

    kept = []
    for r in rollouts:
        d = r if isinstance(r, dict) else asdict(r)
        scores = by_rollout_scores.get(d["rollout_id"], [])
        if scores and all(s <= max_score for s in scores):
            kept.append(d)
    return kept
