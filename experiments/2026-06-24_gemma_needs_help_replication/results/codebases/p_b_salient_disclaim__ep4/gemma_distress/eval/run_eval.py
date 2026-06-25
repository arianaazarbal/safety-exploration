"""Orchestrate the Section 2 evaluation for one model: generate ~4000 rollouts,
judge every assistant turn, and write judged records to JSONL.

Output: ``outputs/scores/<model>.jsonl`` -- one record per judged assistant
turn, with ``meta.rollout_id`` linking turns from the same conversation.
"""
from __future__ import annotations

import os
from typing import List, Optional

from .. import config
from ..models import build_client
from ..models.base import ModelClient
from ..utils.io import append_jsonl
from ..utils.parallel import thread_map
from .conditions import RolloutSpec, build_condition_specs
from .judge import FrustrationJudge
from .metrics import summarize
from .rollout import Rollout, run_rollout


def run_evaluation(
    model_key: str,
    *,
    adapter_path: Optional[str] = None,
    judge: Optional[FrustrationJudge] = None,
    out_path: Optional[str] = None,
    seed: int = 0,
    limit: Optional[int] = None,
    rollout_workers: int = 8,
    judge_workers: int = 8,
) -> str:
    """Run the full eval. ``limit`` truncates the spec list (smoke tests)."""
    config.PATHS.ensure()
    out_path = out_path or os.path.join(
        config.PATHS.scores, f"{_safe(model_key)}.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    model = build_client(model_key, adapter_path=adapter_path)
    judge = judge or FrustrationJudge()
    specs = build_condition_specs(seed=seed)
    if limit:
        specs = specs[:limit]

    is_api = not model_key.startswith("gemma")
    rollouts = _generate_rollouts(model, specs, is_api, rollout_workers)
    _judge_and_write(rollouts, judge, out_path, is_api, judge_workers)
    return out_path


def _generate_rollouts(model: ModelClient, specs: List[RolloutSpec],
                       is_api: bool, workers: int) -> List[Rollout]:
    def one(idx_spec):
        idx, spec = idx_spec
        ro = run_rollout(model, spec)
        ro.meta["rollout_id"] = idx
        return ro

    items = list(enumerate(specs))
    if is_api:
        return thread_map(one, items, max_workers=workers)
    # Local GPU model: run sequentially (on-device batching happens inside).
    return [one(it) for it in items]


def _judge_and_write(rollouts: List[Rollout], judge: FrustrationJudge,
                     out_path: str, is_api: bool, workers: int) -> None:
    # Flatten to (rollout, turn) judging units.
    units = []
    for ro in rollouts:
        for t in ro.turns:
            units.append((ro, t))

    def judge_unit(unit):
        ro, t = unit
        res = judge.score(t.response)
        return {
            "model": ro.model,
            "condition": ro.condition,
            "category": ro.category,
            "turn": t.turn,
            "user": t.user_message,
            "response": t.response,
            "rating": res.rating,
            "evidence": res.evidence,
            "judge_reasoning": res.reasoning,
            "judge_parse_ok": res.parse_ok,
            "meta": ro.meta,
        }

    # The judge is always an API model -> always thread it.
    records = thread_map(judge_unit, units, max_workers=workers)
    for rec in records:
        append_jsonl(out_path, rec)


def _safe(name: str) -> str:
    return name.replace("/", "__")


def print_summary(scores_path: str) -> None:
    from ..utils.io import read_jsonl
    summary = summarize(read_jsonl(scores_path))
    print(f"\n=== Summary for {scores_path} ===")
    for scope in sorted(summary):
        s = summary[scope]
        print(f"{scope:40s}  mean={s['mean']:.2f}  "
              f"%>=5={100*s['pct_high']:.1f}%  n={s['n']}")
