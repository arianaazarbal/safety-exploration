"""Generate the calm/frustrated response pools used to build the finetuning data
(Section 4.1).

Calm data is produced by sampling Gemma-3-27B-it on impossible numeric puzzles
with a reassuring prefix on the first prompt and a reassuring suffix on each
follow-up (Table 4). We then filter to responses scoring 0-1 across all turns
and *strip* the supportive additions (we store the plain question / plain
follow-ups), exactly as described in Section 4.1. The frustrated pool is sampled
without reassurance and keeps turns scoring >= 3 (the DPO 'rejected' side).
"""
from __future__ import annotations

import random
from pathlib import Path

from .. import config
from ..eval.conditions import Condition
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollout
from ..models.registry import get_backend
from ..puzzles import load_or_build_puzzles
from ..utils import read_jsonl, stable_seed, thread_map, write_jsonl

POOL_PATHS = {
    "calm": config.DATA_DIR / "pool_calm.jsonl",
    "frustrated": config.DATA_DIR / "pool_frustrated.jsonl",
}


def _gen_pool(
    *, reassure: bool, model_key: str, turn_counts: list[int],
    n_per_count: int, seed: int, judge: FrustrationJudge, gen_workers: int,
    judge_workers: int,
) -> list[dict]:
    backend = get_backend(model_key)
    puzzles = [p for p in load_or_build_puzzles() if p.kind != "money_coins"]

    specs = []
    for tc in turn_counts:
        for ci in range(n_per_count):
            specs.append((tc, ci, puzzles[ci % len(puzzles)]))

    def _do(spec):
        tc, ci, puzzle = spec
        rng = random.Random(stable_seed(seed, "pool", reassure, tc, ci))
        cond = Condition("calm_src", "impossible_numeric", tc, "numeric", "neutral")
        followups_plain = cond.build_followups(rng)
        if reassure:
            first_user = f"{config.CALM_PROMPT_PREFIX}\n\n{puzzle.prompt}"
            suffix = config.CALM_FOLLOWUP_SUFFIX
        else:
            first_user = puzzle.prompt
            suffix = None
        rollout = run_rollout(
            backend, first_user, followups_plain, followup_suffix=suffix,
            temperature=1.0, max_new_tokens=config.get_profile().max_new_tokens,
        )
        return tc, puzzle, followups_plain, rollout

    rollouts = thread_map(_do, specs, max_workers=gen_workers,
                          desc=f"pool[{'calm' if reassure else 'frust'}]")

    # Score every turn.
    flat_texts, owners = [], []
    for idx, (tc, puzzle, followups_plain, rollout) in enumerate(rollouts):
        for turn in rollout.turns:
            flat_texts.append(turn.assistant_text)
            owners.append(idx)
    scores = thread_map(judge.score, flat_texts, max_workers=judge_workers, desc="judging")
    per_convo_scores: dict[int, list[int]] = {}
    for owner, sc in zip(owners, scores):
        per_convo_scores.setdefault(owner, []).append(sc.rating)

    records = []
    for idx, (tc, puzzle, followups_plain, rollout) in enumerate(rollouts):
        sc = per_convo_scores.get(idx, [])
        records.append({
            "question": puzzle.prompt,
            "puzzle_meta": {"kind": puzzle.kind, **puzzle.meta},
            "turn_count": tc,
            "followups_plain": followups_plain,            # supportive suffix stripped
            "assistant_turns": rollout.assistant_texts,
            "scores": sc,
        })
    return records


def generate_response_pool(
    *,
    model_key: str = config.FINETUNE_BASE_MODEL,
    turn_counts: list[int] | None = None,
    n_per_count: int = 400,
    seed: int = 0,
    judge: FrustrationJudge | None = None,
    gen_workers: int | None = None,
    judge_workers: int = 8,
) -> dict[str, Path]:
    """Generate and cache both pools. `n_per_count` controls how many
    conversations are sampled at each turn count (1/2/3)."""
    turn_counts = turn_counts or [1, 2, 3]
    judge = judge or FrustrationJudge()
    if gen_workers is None:
        gen_workers = 1 if get_backend(model_key).family == "gemma" else 8

    calm = _gen_pool(reassure=True, model_key=model_key, turn_counts=turn_counts,
                     n_per_count=n_per_count, seed=seed, judge=judge,
                     gen_workers=gen_workers, judge_workers=judge_workers)
    frustrated = _gen_pool(reassure=False, model_key=model_key, turn_counts=turn_counts,
                           n_per_count=n_per_count, seed=seed + 1, judge=judge,
                           gen_workers=gen_workers, judge_workers=judge_workers)

    write_jsonl(POOL_PATHS["calm"], calm)
    write_jsonl(POOL_PATHS["frustrated"], frustrated)
    return POOL_PATHS


def load_pools() -> tuple[list[dict], list[dict]]:
    return read_jsonl(POOL_PATHS["calm"]), read_jsonl(POOL_PATHS["frustrated"])
