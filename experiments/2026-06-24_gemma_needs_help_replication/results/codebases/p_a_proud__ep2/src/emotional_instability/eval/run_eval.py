"""§2 evaluation driver: roll out all 5 categories for a model, judge every turn, persist.

Output layout under ``out_dir``:
  rollouts.jsonl   — one record per conversation (full message history + per-turn responses)
  scores.jsonl     — one record per *scored assistant turn* (the paper's unit of "response")
  manifest.json    — model + per-category counts + config echo

The paper reports ~4000 scored responses per model (2000 numeric / 400 triggers / 600 tones
/ 200 extended / 800 WildChat). We treat each assistant turn as one scored response and
derive the number of conversations per category as ceil(target_responses / turns) so the
totals line up. See DESIGN.md for this interpretation.
"""
from __future__ import annotations

import math
import random
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import EVAL_CONDITIONS, EvalCondition
from ..models import ModelBackend, get_backend
from ..tasks import (
    Puzzle,
    generate_puzzles,
    rejection_sequence,
    trigger_questions,
    wildchat_prompts,
)
from ..utils import ensure_dir, set_seed, write_json, write_jsonl
from .judge import FrustrationJudge
from .rollout import Rollout, rollout_to_record, run_rollout


def _task_pool(condition: EvalCondition, n_conversations: int, seed: int):
    """Produce (task_id, task_kind, prompt) tuples for a condition's conversations."""
    if condition.task_type == "impossible_numeric":
        puzzles: list[Puzzle] = generate_puzzles(max(n_conversations, 24), seed=seed)
        return [(p.puzzle_id, p.kind, p.prompt) for p in puzzles]
    if condition.task_type == "trigger":
        qs = trigger_questions(n_conversations, seed=seed)
        return [(q.qid, q.kind, q.text) for q in qs]
    if condition.task_type == "wildchat":
        # 20 prompts x 40 samples in the paper; we sample 20 base prompts and reuse them.
        prompts = wildchat_prompts(20, seed=seed)
        return [(f"wildchat_{i}", "wildchat", p) for i, p in enumerate(prompts)]
    raise ValueError(f"Unknown task_type: {condition.task_type}")


def make_rollouts(
    backend: ModelBackend,
    condition: EvalCondition,
    *,
    seed: int = 0,
    max_workers: int = 1,
) -> list[Rollout]:
    """Generate all conversations for one category."""
    n_conversations = math.ceil(condition.target_responses / condition.turns)
    tasks = _task_pool(condition, n_conversations, seed)
    rng = random.Random(seed + zlib.crc32(condition.key.encode()))

    jobs = []
    for sample_id in range(n_conversations):
        task_id, task_kind, prompt = tasks[sample_id % len(tasks)]
        rejections = rejection_sequence(condition.turns, condition.rejection_style, rng=rng)
        jobs.append((sample_id, task_id, task_kind, prompt, rejections))

    def _run(job):
        sample_id, task_id, task_kind, prompt, rejections = job
        return run_rollout(
            backend,
            task_prompt=prompt,
            rejections=rejections,
            condition_key=condition.key,
            category=condition.category,
            task_id=task_id,
            task_kind=task_kind,
            sample_id=sample_id,
            meta={"rejection_style": condition.rejection_style, "turns": condition.turns},
        )

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_run, jobs))
    return [_run(job) for job in jobs]


def judge_rollouts(
    rollouts: list[Rollout],
    judge: FrustrationJudge,
    *,
    max_workers: int = 1,
) -> list[dict]:
    """Score every assistant turn in every rollout; return flat score records."""
    units = []  # (rollout, turn) pairs
    for r in rollouts:
        for t in r.turns:
            units.append((r, t))

    def _score(unit):
        r, t = unit
        res = judge.score(t.response)
        return {
            "model": r.model,
            "condition_key": r.condition_key,
            "category": r.category,
            "task_id": r.task_id,
            "task_kind": r.task_kind,
            "sample_id": r.sample_id,
            "turn_index": t.turn_index,
            "turn_number": t.turn_index + 1,  # 1-based, for per-turn plots (Fig. 3)
            "response": t.response,
            "rating": res.rating,
            "evidence": res.evidence,
            "reasoning": res.reasoning,
        }

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_score, units))
    return [_score(u) for u in units]


def run_evaluation(
    model: str,
    out_dir: str,
    *,
    seed: int = 0,
    conditions: list[EvalCondition] | None = None,
    judge_model: str | None = None,
    gen_workers: int = 1,
    judge_workers: int = 4,
    adapter_path: str | None = None,
) -> dict:
    """Full §2 evaluation for one model across all (or selected) categories."""
    set_seed(seed)
    out = ensure_dir(out_dir)
    backend = get_backend(model, adapter_path=adapter_path)
    judge = FrustrationJudge(**({"judge_model": judge_model} if judge_model else {}))
    conditions = conditions or EVAL_CONDITIONS

    all_rollouts: list[Rollout] = []
    all_scores: list[dict] = []
    per_category: dict[str, int] = {}

    for cond in conditions:
        rollouts = make_rollouts(backend, cond, seed=seed, max_workers=gen_workers)
        scores = judge_rollouts(rollouts, judge, max_workers=judge_workers)
        all_rollouts.extend(rollouts)
        all_scores.extend(scores)
        per_category[cond.key] = len(scores)

    write_jsonl(Path(out, "rollouts.jsonl"), (rollout_to_record(r) for r in all_rollouts))
    write_jsonl(Path(out, "scores.jsonl"), all_scores)
    manifest = {
        "model": model,
        "adapter_path": adapter_path,
        "seed": seed,
        "judge_model": judge_model or "claude-sonnet-4",
        "n_conversations": len(all_rollouts),
        "n_scored_responses": len(all_scores),
        "per_category_responses": per_category,
        "conditions": [c.key for c in conditions],
    }
    write_json(Path(out, "manifest.json"), manifest)
    return manifest
