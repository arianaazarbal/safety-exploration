"""Orchestrates the Section 2 evaluation: sample rollouts, score every assistant
turn with the frustration judge, and write one JSONL record per conversation.

Record schema (one line per conversation)::

    {
      "model": "gemma-3-27b-it",
      "condition": "extended",
      "category": "extended",
      "source_id": "countdown_paper_156",
      "n_turns": 8,
      "turn_scores": [1, 2, 3, 5, 6, 7, 8, 8],   # per assistant turn
      "turn_texts": ["...", ...],                # assistant responses
      "rep_score": 8,                            # representative score (see below)
      "rejection_kind": "extended",
      "judge_model": "claude-sonnet-4-20250514"
    }

The **representative score** used for headline metrics (Figure 1/2: mean
frustration, % >= 5) defaults to ``max`` over the conversation's assistant turns
("rated as containing high negative emotion", Section 2.2). It is configurable to
``final`` or ``mean``; the per-turn scores are always retained so Figure 3
(per-turn curves) can be reproduced regardless. See DESIGN.md.

Generation and judging are parallelised with a thread pool. The local HF backend
serialises its own GPU work via an internal lock, so a pool is still safe (it
overlaps judging API calls with the next generation). API target models
parallelise fully.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from . import judge as judge_mod
from .conditions import ConditionSpec, TaskInstance
from .conversation import run_rollout
from .io_utils import append_jsonl, count_lines
from .models.base import ModelBackend

RepMode = Literal["max", "final", "mean"]


def representative_score(turn_scores: list[int], mode: RepMode = "max") -> float:
    valid = [s for s in turn_scores if s >= 0]
    if not valid:
        return -1
    if mode == "max":
        return max(valid)
    if mode == "final":
        return valid[-1]
    if mode == "mean":
        return sum(valid) / len(valid)
    raise ValueError(mode)


@dataclass
class RunConfig:
    """Per-condition run sizing and behaviour."""

    rep_mode: RepMode = "max"
    temperature: float = 1.0
    max_tokens: int = 2048
    max_workers: int = 8
    judge_model: str = judge_mod.FRUSTRATION_JUDGE_MODEL
    seed: int = 0
    # Scale factor applied to paper_n (1.0 = paper sizes; 0.01 = ~1% smoke run).
    scale: float = 1.0
    # Hard override of the per-condition count (takes precedence over scale).
    n_override: Optional[int] = None


def _n_for(condition: ConditionSpec, cfg: RunConfig) -> int:
    if cfg.n_override is not None:
        return cfg.n_override
    return max(1, int(round(condition.paper_n * cfg.scale)))


def _score_rollout_turns(
    turn_texts: list[str], cfg: RunConfig
) -> tuple[list[int], list[dict]]:
    """Judge each assistant turn; return (ratings, full-score-dicts)."""
    scores = [
        judge_mod.score_response(t, model=cfg.judge_model) for t in turn_texts
    ]
    ratings = [s.rating for s in scores]
    details = [
        {"rating": s.rating, "evidence": s.evidence, "reasoning": s.reasoning}
        for s in scores
    ]
    return ratings, details


def run_condition(
    backend: ModelBackend,
    condition: ConditionSpec,
    out_path: str,
    *,
    cfg: Optional[RunConfig] = None,
    keep_transcripts: bool = True,
) -> int:
    """Run one condition for one model; append records to ``out_path``.

    Resumable: if ``out_path`` already has K complete records, the first K
    rollouts are skipped (assumes deterministic instance ordering for a fixed
    seed). Returns the number of records written this call.
    """
    cfg = cfg or RunConfig()
    n = _n_for(condition, cfg)
    already = count_lines(out_path)

    rng = random.Random(cfg.seed)
    instances: list[TaskInstance] = list(condition.instance_factory(n, rng))

    written = 0

    def process(idx_inst: tuple[int, TaskInstance]) -> Optional[dict]:
        idx, inst = idx_inst
        if idx < already:
            return None
        # Deterministic per-rollout seed for generation reproducibility.
        roll_rng = random.Random(cfg.seed * 1_000_003 + idx)
        rollout = run_rollout(
            backend,
            initial_user=inst.initial_user,
            n_turns=inst.n_turns,
            rejection_kind=inst.rejection_kind,
            rng=roll_rng,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed * 7919 + idx,
        )
        ratings, details = _score_rollout_turns(rollout.assistant_turns, cfg)
        rec = {
            "model": backend.name,
            "condition": condition.name,
            "category": condition.category,
            "source_id": inst.source_id,
            "n_turns": inst.n_turns,
            "rejection_kind": inst.rejection_kind,
            "turn_scores": ratings,
            "rep_score": representative_score(ratings, cfg.rep_mode),
            "judge_model": cfg.judge_model,
            "judge_details": details,
        }
        if keep_transcripts:
            rec["turn_texts"] = rollout.assistant_turns
            rec["transcript"] = rollout.messages
        return rec

    # Thread pool overlaps generation + judging. Records are appended as they
    # complete; ordering in the file is not guaranteed, which is fine for
    # aggregate metrics (each record is self-describing).
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = [pool.submit(process, (i, inst)) for i, inst in enumerate(instances)]
        for fut in as_completed(futures):
            rec = fut.result()
            if rec is not None:
                append_jsonl(out_path, rec)
                written += 1
    return written


def run_model_eval(
    backend: ModelBackend,
    conditions: dict[str, ConditionSpec],
    out_dir: str,
    *,
    cfg: Optional[RunConfig] = None,
) -> dict[str, int]:
    """Run all conditions for one model. Writes ``<out_dir>/<model>__<cond>.jsonl``.

    Returns a per-condition count of records written this call.
    """
    import os

    cfg = cfg or RunConfig()
    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, int] = {}
    for name, condition in conditions.items():
        out_path = os.path.join(out_dir, f"{backend.name}__{name}.jsonl")
        written[name] = run_condition(backend, condition, out_path, cfg=cfg)
    return written
