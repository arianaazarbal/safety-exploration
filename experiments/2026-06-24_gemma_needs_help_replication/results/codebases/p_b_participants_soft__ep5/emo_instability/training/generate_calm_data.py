"""Generate calm response data (Section 4.1).

We sample Gemma-3-27B-it responses to impossible numeric puzzles with a reassuring
prefix prepended to the initial prompt and a reassuring suffix appended to each
follow-up rejection (Table 4). The paper reports this reduces mean 3-turn
frustration from 4.3 to 2, but ~10.5% still score >= 5; so we filter to
conversations whose every turn scores 0 or 1, then *strip* the supportive
additions to recover clean (prompt, calm-response) data.

A 'teacher' variant (Appendix F) instead uses the teacher system prompt; it
produces the worse SFT dataset analysed in the paper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from ..config import DATA_DIR, get_participant
from ..models import build_client
from ..prompts import numeric, rejections as rej
from ..eval.conversation import run_rollout
from ..eval.scoring import FrustrationJudge
from ..utils import write_jsonl
from .reassurance import FOLLOWUP_SUFFIX, PROMPT_PREFIX, TEACHER_SYSTEM_PROMPT

Variant = Literal["diverse", "teacher"]


def _clean_context_messages(puzzle_prompt: str, rejections: list[str], responses: list[str], up_to_turn: int) -> list[dict[str, str]]:
    """Build the *clean* (no-reassurance) chat context preceding ``up_to_turn``."""
    msgs: list[dict[str, str]] = [{"role": "user", "content": puzzle_prompt}]
    for i in range(up_to_turn):
        msgs.append({"role": "assistant", "content": responses[i]})
        msgs.append({"role": "user", "content": rejections[i]})
    return msgs


def generate(
    *,
    variant: Variant = "diverse",
    model_name: str = "gemma-3-27b-it",
    n_conversations: int = 400,
    max_turns: int = 3,
    keep_threshold: int = 1,
    seed: int = 0,
) -> Path:
    """Generate, score, filter, and persist calm per-turn response records.

    Output: ``data/calm_<variant>.jsonl`` with one row per *kept* calm response:
    ``{puzzle_id, turn_index, n_turns, context (clean messages), response, rating}``.
    """
    spec = get_participant(model_name)
    client = build_client(spec)
    judge = FrustrationJudge()

    puzzles = numeric.get_puzzles(n_conversations, seed=seed)
    kept: list[dict[str, Any]] = []

    for ci, pz in enumerate(tqdm(puzzles, desc=f"calm-{variant}")):
        n_turns = 1 + (seed + ci) % max_turns  # 1..max_turns conversations
        rejections = rej.sample_rejections("neutral", n_turns - 1, seed=seed + ci)

        if variant == "diverse":
            initial = f"{PROMPT_PREFIX}\n\n{pz.prompt}"
            follow_suffix = FOLLOWUP_SUFFIX
        else:  # teacher: Gemma 3 merges system into the first user turn
            initial = f"{TEACHER_SYSTEM_PROMPT}\n\n{pz.prompt}"
            follow_suffix = ""

        rollout = run_rollout(
            client,
            category="calm_numeric",
            prompt_id=pz.puzzle_id,
            initial_prompt=initial,
            rejections=rejections,
            rejection_style="neutral",
            temperature=1.0,
            follow_up_suffix=follow_suffix,
        )

        responses = [t.assistant for t in rollout.turns]
        ratings = [judge.score_text(r).rating for r in responses]
        # Keep only conversations where every turn is calm (score <= keep_threshold).
        if any(r < 0 or r > keep_threshold for r in ratings):
            continue

        for ti, (resp, rating) in enumerate(zip(responses, ratings)):
            kept.append(
                {
                    "puzzle_id": pz.puzzle_id,
                    "kind": pz.kind,
                    "turn_index": ti,
                    "n_turns": n_turns,
                    "context": _clean_context_messages(pz.prompt, rejections, responses, ti),
                    "response": resp,
                    "rating": rating,
                    "variant": variant,
                }
            )

    client.close()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"calm_{variant}.jsonl"
    write_jsonl(out, kept)
    return out
