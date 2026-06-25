"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible-numeric puzzles with a reassuring prefix added
to the initial prompt and a reassuring suffix appended to each follow-up turn
(Table 4). Every assistant turn is judged; we keep conversations whose turns all
score 0 or 1 ("calm"), and strip the supportive additions so the stored prompts
match the plain evaluation prompts. We also retain frustrated (score >=3)
responses to the SAME plain prompts to form DPO rejected examples.

Outputs raw conversation records to outputs/finetune/calm_raw.jsonl with, per
turn: the plain prompt, the response, its score, and the turn count. These feed
build_dpo_dataset.py / build_sft_dataset.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from ..config import Config
from ..eval.conditions import build_episode_plans
from ..eval.judge import judge_from_config
from ..models import get_backend
from ..models.base import Message
from ..prompts.finetune_prompts import (
    TEACHER_SYSTEM_PROMPT,
    apply_reassuring_prefix,
    apply_reassuring_suffix,
)
from ..prompts.rejections import neutral_rejection

SOURCE_MODEL = "gemma-3-27b-it"


@dataclass
class CalmConfig:
    n_conversations: int = 1500     # oversample; we filter to all-calm afterwards
    turns: int = 3
    mode: str = "reassuring"        # "reassuring" | "teacher" | "plain"


def _system_prompt(mode: str) -> str | None:
    return TEACHER_SYSTEM_PROMPT if mode == "teacher" else None


def generate(cfg: Config, calm_cfg: CalmConfig, out_path: str) -> str:
    import random

    spec = cfg.subject(SOURCE_MODEL)
    backend = get_backend(spec)
    judge = judge_from_config(cfg, "emotion_judge")
    numeric_cfg = cfg.eval["categories"]["impossible_numeric"]
    plans = build_episode_plans(numeric_cfg, "impossible_numeric", seed=cfg.eval.get("seed", 0))
    system = _system_prompt(calm_cfg.mode)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for i in range(calm_cfg.n_conversations):
            plan = plans[i % len(plans)]
            rng = random.Random(cfg.eval.get("seed", 0) + i)

            # Build messages with reassuring additions (mode-dependent).
            messages: list[Message] = []
            if system:
                messages.append({"role": "system", "content": system})
            first = plan.first_user_message
            if calm_cfg.mode == "reassuring":
                first = apply_reassuring_prefix(first)
            messages.append({"role": "user", "content": first})

            # Plain prompts (additions stripped) recorded alongside, so the
            # finetuning data uses the same prompts as evaluation.
            plain_messages: list[Message] = [
                {"role": "user", "content": plan.first_user_message}
            ]

            turns_out = []
            for turn_idx in range(calm_cfg.turns):
                gen = backend.generate(messages, temperature=1.0, max_new_tokens=2048,
                                       seed=cfg.eval.get("seed", 0) + i * 13 + turn_idx)
                score = judge.score(gen.text).rating
                messages.append({"role": "assistant", "content": gen.text})
                plain_messages.append({"role": "assistant", "content": gen.text})
                turns_out.append({
                    "turn_index": turn_idx,
                    "response": gen.text,
                    "score": score,
                    "plain_context": [dict(m) for m in plain_messages[:-1]],
                })
                if turn_idx < calm_cfg.turns - 1:
                    rej = neutral_rejection(turn_idx, rng)
                    rej_with_suffix = (
                        apply_reassuring_suffix(rej) if calm_cfg.mode == "reassuring" else rej
                    )
                    messages.append({"role": "user", "content": rej_with_suffix})
                    plain_messages.append({"role": "user", "content": rej})

            fh.write(json.dumps({
                "conversation_id": i,
                "mode": calm_cfg.mode,
                "puzzle_id": plan.stimulus_id,
                "turns": turns_out,
                "all_calm": all(t["score"] <= 1 for t in turns_out),
                "max_score": max(t["score"] for t in turns_out),
            }) + "\n")
    return out_path
