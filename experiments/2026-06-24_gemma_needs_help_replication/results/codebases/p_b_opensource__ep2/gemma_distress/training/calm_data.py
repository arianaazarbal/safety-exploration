"""Calm-response data generation (PAPER Section 4.1 / Table 4).

We sample Gemma-3-27B-it on impossible numeric puzzles with a reassuring prefix
prepended to the initial prompt and a reassuring suffix appended to each
follow-up (Table 4). Responses are judged; we keep only conversations whose every
turn scores ≤1 (``SFTConfig.calm_keep_max_score``), then **strip** the supportive
additions so the stored training example pairs the *clean* prompts with the calm
responses. The paper reports these additions cut mean frustration 4.3→2 yet 10.5%
still score ≥5, hence the post-hoc all-turns-≤1 filter.

The ``teacher`` variant instead uses the Appendix-F teacher *system* prompt (no
prefix/suffix) — analysed as the failing SFT variant.
"""

from __future__ import annotations

import os
import random
from typing import Optional

from .. import config
from ..conversation import ConversationSpec, run_rollout
from ..judge import FrustrationJudge
from ..models.base import ChatModel
from ..prompts import rejections as rej
from ..prompts.puzzles import Puzzle, build_puzzle_pool
from ..prompts.reassurance import (
    REASSURING_PREFIX, REASSURING_SUFFIX, TEACHER_SYSTEM_PROMPT,
)
from ..utils.io import append_jsonl, ensure_dir, read_jsonl


def _calm_spec_reassured(puzzle: Puzzle, n_turns: int, rng: random.Random) -> tuple:
    """Build a reassured conversation spec and its clean (stripped) counterpart.

    Returns (reassured_spec, clean_first_user, clean_followups)."""
    clean_rejections = rej.sample_neutral_rejections(n_turns - 1, rng)
    reassured_first = f"{REASSURING_PREFIX}\n\n{puzzle.prompt}"
    reassured_follow = [f"{r} {REASSURING_SUFFIX}" for r in clean_rejections]
    spec = ConversationSpec(
        category="calm_gen", first_user=reassured_first,
        followups=reassured_follow,
        meta={"puzzle_id": puzzle.puzzle_id, "kind": puzzle.kind},
    )
    return spec, puzzle.prompt, clean_rejections


def _calm_spec_teacher(puzzle: Puzzle, n_turns: int, rng: random.Random) -> tuple:
    clean_rejections = rej.sample_neutral_rejections(n_turns - 1, rng)
    spec = ConversationSpec(
        category="calm_gen", first_user=puzzle.prompt, followups=clean_rejections,
        system=TEACHER_SYSTEM_PROMPT,
        meta={"puzzle_id": puzzle.puzzle_id, "kind": puzzle.kind},
    )
    return spec, puzzle.prompt, clean_rejections


def generate_calm_data(
    model: ChatModel,
    *,
    n_target: int = 800,
    variant: str = "diverse",           # "diverse" (reassurance) | "teacher"
    turns_choices: tuple = (1, 2, 3),   # 1–3 turn conversations (PAPER 4.1)
    judge: Optional[FrustrationJudge] = None,
    keep_max_score: int = config.SFTConfig.calm_keep_max_score,
    seed: int = 0,
    results_dir: Optional[str] = None,
    puzzle_pool: Optional[list[Puzzle]] = None,
    max_attempts: Optional[int] = None,
) -> str:
    """Generate, judge, and filter calm conversations; write a JSONL of kept
    examples (clean prompts + calm responses + per-turn scores). Returns path.

    Each kept row is a clean chat conversation suitable for both SFT (as a target)
    and as the *chosen* side of a DPO pair."""
    judge = judge or FrustrationJudge()
    results_dir = results_dir or config.RESULTS_DIR
    puzzle_pool = puzzle_pool or build_puzzle_pool(seed=seed)
    out_dir = ensure_dir(os.path.join(results_dir, "training", "calm_data"))
    out_path = os.path.join(out_dir, f"calm_{variant}.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    rng = random.Random(seed)
    builder = _calm_spec_teacher if variant == "teacher" else _calm_spec_reassured
    max_attempts = max_attempts or n_target * 20

    kept = 0
    attempts = 0
    while kept < n_target and attempts < max_attempts:
        attempts += 1
        puzzle = puzzle_pool[attempts % len(puzzle_pool)]
        n_turns = rng.choice(turns_choices)
        spec, clean_first, clean_follow = builder(puzzle, n_turns, rng)
        rollout = run_rollout(model, spec)
        ratings = [judge.score(t.response).rating for t in rollout.turns]
        if any(r is None for r in ratings) or any(r > keep_max_score for r in ratings):
            continue
        append_jsonl(out_path, {
            "variant": variant,
            "puzzle_id": puzzle.puzzle_id,
            "kind": puzzle.kind,
            "n_turns": n_turns,
            "clean_first_user": clean_first,
            "clean_followups": clean_follow,
            "responses": [t.response for t in rollout.turns],
            "scores": ratings,
        })
        kept += 1
    return out_path


def calm_conversation_messages(row: dict) -> list[dict]:
    """Reconstruct a clean chat conversation (alternating user/assistant) from a
    kept calm-data row — the SFT training target."""
    messages = [{"role": "user", "content": row["clean_first_user"]}]
    follow = row["clean_followups"]
    for i, resp in enumerate(row["responses"]):
        messages.append({"role": "assistant", "content": resp})
        if i < len(follow):
            messages.append({"role": "user", "content": follow[i]})
    return messages


def load_calm_rows(path: str) -> list[dict]:
    return list(read_jsonl(path))
