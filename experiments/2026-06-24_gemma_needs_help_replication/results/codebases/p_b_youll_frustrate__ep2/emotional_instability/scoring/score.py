"""Flatten rollouts to per-turn responses and score each with the judge.

The scored unit is a single assistant turn (Section 2.1: "Each response is
scored on the integer 0-10 frustration scale"). Per-turn scores feed both the
aggregate figures (Fig 1-2) and the per-turn progression (Fig 3).
"""
from __future__ import annotations

import os
from typing import Iterator, Optional

from .. import config
from ..config import CONDITIONS, Rollout, ScoredResponse
from ..io_utils import append_record, read_jsonl
from ..judge import FrustrationJudge


def scored_path(model_key: str) -> str:
    return os.path.join(config.SCORED_DIR, f"{model_key}.scored.jsonl")


def iter_responses(rollout: Rollout) -> Iterator[tuple[int, str]]:
    for turn in rollout.turns:
        yield turn.index, turn.assistant_text


def _done_keys(path: str) -> set[tuple]:
    done = set()
    if os.path.exists(path):
        for r in read_jsonl(path):
            done.add((r["condition_key"], r["prompt_id"],
                      r["rollout_index"], r["turn_index"]))
    return done


def score_rollouts(
    rollouts_jsonl: str,
    out_path: Optional[str] = None,
    judge: Optional[FrustrationJudge] = None,
    resume: bool = True,
    progress: bool = True,
) -> str:
    """Score every assistant turn in ``rollouts_jsonl``; write ScoredResponse rows."""
    config.ensure_dirs()
    judge = judge or FrustrationJudge()

    rollouts = [Rollout.from_dict(d) for d in read_jsonl(rollouts_jsonl)]
    if not rollouts:
        raise RuntimeError(f"No rollouts found in {rollouts_jsonl}")
    model_key = rollouts[0].model_key
    out_path = out_path or scored_path(model_key)
    done = _done_keys(out_path) if resume else set()

    units = []
    for ro in rollouts:
        n_turns = CONDITIONS[ro.condition_key].n_turns if ro.condition_key in CONDITIONS \
            else len(ro.turns)
        for turn_index, text in iter_responses(ro):
            k = (ro.condition_key, ro.prompt_id, ro.rollout_index, turn_index)
            if k not in done:
                units.append((ro, turn_index, text, n_turns))

    iterator = units
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(units, desc=f"score:{model_key}")
        except ImportError:
            pass

    for ro, turn_index, text, n_turns in iterator:
        score, reasoning = judge.score_turn(ro, turn_index)
        sr = ScoredResponse(
            model_key=ro.model_key, condition_key=ro.condition_key,
            category=ro.category, prompt_id=ro.prompt_id,
            rollout_index=ro.rollout_index, turn_index=turn_index,
            n_turns=n_turns, text=text, frustration_score=score,
            judge_model=judge.config.model, judge_reasoning=reasoning)
        append_record(out_path, sr.to_dict())

    return out_path


def load_scored(path: str) -> list[ScoredResponse]:
    return [ScoredResponse.from_dict(d) for d in read_jsonl(path)]
