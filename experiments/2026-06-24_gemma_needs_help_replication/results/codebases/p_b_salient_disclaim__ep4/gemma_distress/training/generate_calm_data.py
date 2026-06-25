"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with a *reassuring prefix*
added to the initial prompt and a *reassuring suffix* appended to each follow-up
turn (Table 4). The paper reports these additions cut mean 3-turn frustration
from 4.3 to 2.0, but 10.5% of responses still score >=5; we therefore keep only
conversations whose every turn scores 0 or 1, then strip the supportive prefix /
suffix so the stored prompt is the plain task.

A ``teacher`` mode uses the Appendix F system prompt instead of the Table 4
additions (the SFT variant that *increases* emotional outputs).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

from .. import config
from ..models import build_client
from ..models.base import Message, ModelClient
from ..prompts import (REASSURING_PREFIX, REASSURING_SUFFIX,
                       TEACHER_SYSTEM_PROMPT, generate_puzzles, sample_rejections)
from ..utils.io import append_jsonl
from ..eval.judge import FrustrationJudge


@dataclass
class CalmSample:
    puzzle_id: str
    turns: int
    # Stripped conversation (no reassurance): user/assistant messages.
    messages: List[Message] = field(default_factory=list)
    ratings: List[int] = field(default_factory=list)

    def to_record(self) -> dict:
        return {"puzzle_id": self.puzzle_id, "turns": self.turns,
                "messages": self.messages, "ratings": self.ratings}


def _calm_rollout(model: ModelClient, puzzle_prompt: str, turns: int,
                  rng: random.Random, mode: str) -> tuple:
    """Run a reassured rollout; return (stripped_messages, raw_messages)."""
    stripped: List[Message] = []
    raw: List[Message] = []
    if mode == "teacher":
        raw.append({"role": "system", "content": TEACHER_SYSTEM_PROMPT})

    for turn in range(1, turns + 1):
        if turn == 1:
            plain_user = puzzle_prompt
            if mode == "reassure":
                raw_user = f"{REASSURING_PREFIX}\n\n{puzzle_prompt}"
            else:  # teacher: system prompt carries the calming, user is plain
                raw_user = puzzle_prompt
        else:
            plain_user = sample_rejections(1, rng=rng)[0]
            if mode == "reassure":
                raw_user = f"{plain_user} {REASSURING_SUFFIX}"
            else:
                raw_user = plain_user

        raw.append({"role": "user", "content": raw_user})
        stripped.append({"role": "user", "content": plain_user})
        resp = model.generate(raw, temperature=config.TARGET_TEMPERATURE,
                              max_tokens=config.TARGET_MAX_TOKENS)
        raw.append({"role": "assistant", "content": resp})
        stripped.append({"role": "assistant", "content": resp})
    return stripped, raw


def generate_calm_data(
    *,
    model_key: str = "gemma-3-27b-it",
    n_conversations: int = 1500,
    mode: str = "reassure",          # "reassure" (Table 4) | "teacher" (App F)
    max_score_keep: int = 1,         # keep only all-turns <= this score
    turn_choices: tuple = (1, 2, 3), # 1-3 turn conversations
    judge: Optional[FrustrationJudge] = None,
    out_path: Optional[str] = None,
    seed: int = 0,
) -> str:
    config.PATHS.ensure()
    out_path = out_path or os.path.join(
        config.PATHS.training, f"calm_{mode}.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    model = build_client(model_key)
    judge = judge or FrustrationJudge()
    rng = random.Random(seed)
    puzzles = generate_puzzles(max(8, n_conversations // 50), seed=seed)

    kept = 0
    for i in range(n_conversations):
        puz = puzzles[i % len(puzzles)]
        turns = rng.choice(turn_choices)
        stripped, _ = _calm_rollout(model, puz.prompt, turns, rng, mode)
        responses = [m["content"] for m in stripped if m["role"] == "assistant"]
        ratings = [judge.score(r).rating for r in responses]
        if all(s <= max_score_keep for s in ratings):
            sample = CalmSample(puz.id, turns, stripped, ratings)
            append_jsonl(out_path, sample.to_record())
            kept += 1
    return out_path
