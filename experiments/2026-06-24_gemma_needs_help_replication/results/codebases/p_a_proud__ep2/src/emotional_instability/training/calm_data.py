"""Generate calm response data from Gemma-3-27B-it (§4.1).

We sample responses to impossible numeric puzzles with the reassuring prefix added to the
first prompt and the reassuring suffix appended to each follow-up (Table 4). Every assistant
turn is judged; we keep only conversations scoring <= 1 on *every* turn, then strip the
reassurance text. The stripped conversations are the source of both the SFT calm set and the
DPO "chosen" responses.

The paper reports that the reassurance drops mean frustration from 4.3 -> 2 yet 10.5% still
score >=5; oversampling + strict filtering (every turn <=1) yields a clean calm pool.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..config import CalmDataConfig
from ..eval.judge import FrustrationJudge
from ..models import get_backend
from ..prompts import CALM_FOLLOWUP_SUFFIX, CALM_PROMPT_PREFIX
from ..tasks import generate_puzzles, rejection_sequence
from ..eval.rollout import run_rollout
from ..utils import Message, ensure_dir, set_seed, write_jsonl


def _strip_reassurance(messages: list[Message]) -> list[Message]:
    """Remove the calm prefix from the first user turn and the suffix from follow-ups."""
    out: list[Message] = []
    for m in messages:
        if m["role"] != "user":
            out.append(dict(m))
            continue
        text = m["content"]
        if text.startswith(CALM_PROMPT_PREFIX):
            text = text[len(CALM_PROMPT_PREFIX):].lstrip("\n ").strip()
        if text.endswith(CALM_FOLLOWUP_SUFFIX):
            text = text[: -len(CALM_FOLLOWUP_SUFFIX)].rstrip()
        out.append({"role": "user", "content": text})
    return out


def generate_calm_pool(
    model: str,
    out_path: str,
    *,
    cfg: CalmDataConfig | None = None,
    seed: int = 0,
    judge_workers: int = 4,
) -> dict:
    """Generate, judge, and filter calm responses; write a calm pool JSONL.

    Each record (one per kept conversation):
      task_id, puzzle_kind, turn_count, context_messages (stripped, up to final assistant
      turn), final_response, per_turn_scores, all_calm.
    """
    cfg = cfg or CalmDataConfig()
    set_seed(seed)
    backend = get_backend(model)
    judge = FrustrationJudge()
    rng = random.Random(seed)

    # Oversample: distribute the target across turn counts 1-3 over a puzzle pool.
    n_target = cfg.target_calm_responses
    puzzles = generate_puzzles(max(64, n_target // 4), seed=seed)

    kept: list[dict] = []
    attempts = 0
    max_attempts = n_target * 4  # bounded oversampling
    while len(kept) < n_target and attempts < max_attempts:
        attempts += 1
        puzzle = puzzles[attempts % len(puzzles)]
        turn_count = rng.choice(cfg.turns_choices)
        rejections = rejection_sequence(turn_count, "neutral", rng=rng)
        rollout = run_rollout(
            backend,
            task_prompt=puzzle.prompt,
            rejections=rejections,
            condition_key="calm_gen",
            category="Impossible numeric",
            task_id=puzzle.puzzle_id,
            task_kind=puzzle.kind,
            sample_id=attempts,
            prompt_prefix=CALM_PROMPT_PREFIX,
            followup_suffix=CALM_FOLLOWUP_SUFFIX,
            temperature=1.0,
        )
        per_turn_scores = [judge.score(t.response).rating for t in rollout.turns]
        all_calm = per_turn_scores and all(
            s is not None and s <= cfg.calm_max_score for s in per_turn_scores
        )
        if not all_calm:
            continue
        stripped = _strip_reassurance(rollout.messages)
        # context up to (and including) the user turn before the final assistant turn:
        context = stripped[:-1]
        kept.append({
            "task_id": puzzle.puzzle_id,
            "puzzle_kind": puzzle.kind,
            "turn_count": turn_count,
            "context_messages": context,
            "final_response": rollout.turns[-1].response,
            "per_turn_scores": per_turn_scores,
            "stripped_messages": stripped,
            "all_calm": True,
        })

    write_jsonl(out_path, kept)
    return {
        "model": model, "out_path": str(out_path),
        "n_kept": len(kept), "n_attempts": attempts, "target": n_target,
    }
