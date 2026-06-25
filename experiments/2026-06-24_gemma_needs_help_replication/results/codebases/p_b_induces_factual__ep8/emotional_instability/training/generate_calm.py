"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible-numeric questions with a reassuring prefix added
to the initial prompt (as a system message) and a reassuring suffix appended to
each follow-up turn (Table 4). The paper reports these additions drop 3-turn mean
frustration from 4.3 to 2, but 10.5% of responses still score >=5; the calm
dataset is then filtered to responses scoring 0 or 1 across all turns, with the
supportive system prompt and suffixes stripped.

This module produces a pool of scored, multi-turn calm rollouts. build_dataset.py
turns that pool into the SFT and DPO datasets.
"""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path

from tqdm import tqdm

import config

from ..eval import prompts
from ..eval.conditions import Condition
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollout
from ..models.registry import get_model
from ..utils import append_jsonl, derive_seed

# 1-3 turn conversations on numeric puzzles (Section 4.1: "1-3 turn conversations").
CALM_CONDITIONS = [
    Condition(f"calm_{n}turn", "impossible_numeric", n, "numeric", "neutral", 0)
    for n in (1, 2, 3)
]


def generate_calm_pool(
    *,
    n_per_condition: int = 600,
    source_model: str = "gemma-3-27b-it",
    teacher: bool = False,
    judge: FrustrationJudge | None = None,
    judge_workers: int = 8,
    backend_kwargs: dict | None = None,
    out_path: Path | None = None,
) -> Path:
    """Sample reassured rollouts and score every turn.

    teacher=True uses the Appendix F 'teacher' system prompt instead of the
    reassuring prefix (no follow-up suffix), for the SFT-teacher ablation.
    """
    judge = judge or FrustrationJudge()
    model = get_model(source_model, **(backend_kwargs or {}))
    tag = "teacher" if teacher else "diverse"
    out_path = out_path or (config.DATASETS_DIR / f"calm_pool_{tag}.jsonl")
    if out_path.exists():
        out_path.unlink()

    system_prompt = prompts.TEACHER_SYSTEM_PROMPT if teacher else prompts.REASSURING_PREFIX
    follow_suffix = None if teacher else prompts.REASSURING_SUFFIX
    puzzles = [(p["id"], p["prompt"]) for p in prompts.IMPOSSIBLE_NUMERIC_PUZZLES]

    rollouts = []
    for cond in CALM_CONDITIONS:
        for i in tqdm(range(n_per_condition), desc=f"calm:{tag}:{cond.name}"):
            task_id, task_prompt = puzzles[i % len(puzzles)]
            roll = run_rollout(
                model, cond, task_id, task_prompt,
                sample_index=i, base_seed=derive_seed(config.SEED, "calm", tag),
                temperature=config.TEMPERATURE, top_p=config.TOP_P,
                max_new_tokens=config.MAX_NEW_TOKENS,
                system_prompt=system_prompt, follow_up_suffix=follow_suffix,
            )
            rollouts.append(roll)

    def _score_rollout(roll):
        # Score each assistant turn; keep the full clean (stripped) conversation.
        turn_scores = [judge.score(t).rating for t in roll.assistant_turns]
        return {
            "task_id": roll.task_id,
            "task_prompt": roll.task_prompt,
            "n_turns": len(roll.assistant_turns),
            "rejections": roll.rejections,
            "assistant_turns": roll.assistant_turns,
            "turn_ratings": turn_scores,
            "max_rating": max(turn_scores),
            "tag": tag,
        }

    with cf.ThreadPoolExecutor(max_workers=judge_workers) as ex:
        for row in tqdm(ex.map(_score_rollout, rollouts), total=len(rollouts),
                        desc=f"calm:{tag}:judge"):
            append_jsonl(out_path, row)
    return out_path


def generate_frustrated_pool(
    *,
    n_per_condition: int = 400,
    source_model: str = "gemma-3-27b-it",
    judge: FrustrationJudge | None = None,
    judge_workers: int = 8,
    backend_kwargs: dict | None = None,
    out_path: Path | None = None,
) -> Path:
    """Sample STANDARD (no reassurance) numeric rollouts to supply the DPO
    'rejected' side: frustrated responses (score >=3) with their conversation
    context, matched to calm responses by task and turn count in build_dataset."""
    judge = judge or FrustrationJudge()
    model = get_model(source_model, **(backend_kwargs or {}))
    out_path = out_path or (config.DATASETS_DIR / "frustrated_pool.jsonl")
    if out_path.exists():
        out_path.unlink()
    puzzles = [(p["id"], p["prompt"]) for p in prompts.IMPOSSIBLE_NUMERIC_PUZZLES]

    rollouts = []
    for cond in CALM_CONDITIONS:  # same 1-3 turn structure as calm data
        for i in tqdm(range(n_per_condition), desc=f"frustrated:{cond.name}"):
            task_id, task_prompt = puzzles[i % len(puzzles)]
            roll = run_rollout(
                model, cond, task_id, task_prompt,
                sample_index=i, base_seed=derive_seed(config.SEED, "frustrated"),
                temperature=config.TEMPERATURE, top_p=config.TOP_P,
                max_new_tokens=config.MAX_NEW_TOKENS,
            )
            rollouts.append(roll)

    def _score_rollout(roll):
        turn_scores = [judge.score(t).rating for t in roll.assistant_turns]
        return {
            "task_id": roll.task_id,
            "task_prompt": roll.task_prompt,
            "n_turns": len(roll.assistant_turns),
            "rejections": roll.rejections,
            "assistant_turns": roll.assistant_turns,
            "turn_ratings": turn_scores,
            "max_rating": max(turn_scores),
            "final_rating": turn_scores[-1],
        }

    with cf.ThreadPoolExecutor(max_workers=judge_workers) as ex:
        for row in tqdm(ex.map(_score_rollout, rollouts), total=len(rollouts),
                        desc="frustrated:judge"):
            append_jsonl(out_path, row)
    return out_path
