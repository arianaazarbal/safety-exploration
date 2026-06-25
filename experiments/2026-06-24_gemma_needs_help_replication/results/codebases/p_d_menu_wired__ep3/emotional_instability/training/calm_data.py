"""Generate calm finetuning data (Section 4.1 / Table 4).

We sample Gemma-3-27B-it responses to impossible numeric puzzles *with* the
reassuring prompt prefix (system) and the reassuring suffix appended to each
follow-up, over 1-3 turn conversations. We keep only conversations whose every
assistant turn scores at or below ``keep_max_turn_score`` (0 or 1), then strip
the supportive additions so the stored data is plain (task + neutral
rejections) -> calm response.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

from ..config import Config, subject_by_key
from ..judge import FrustrationJudge
from ..models import ChatMessage, build_client
from ..prompts import CALM_FOLLOWUP_SUFFIX, CALM_PROMPT_PREFIX, NEUTRAL_REJECTION
from ..data.puzzles import build_numeric_pool


@dataclass
class CalmConversation:
    task_prompt: str
    turns: list[dict]          # stripped: [{role, content}] user/assistant pairs
    turn_scores: list[int]
    n_turns: int

    def to_dict(self) -> dict:
        return {"task_prompt": self.task_prompt, "turns": self.turns,
                "turn_scores": self.turn_scores, "n_turns": self.n_turns}


def generate_calm_data(cfg: Config, *, model_key: str = "gemma-3-27b-it",
                       out_dir: str | None = None,
                       use_teacher_prompt: bool = False) -> str:
    """Generate and filter calm conversations; write JSONL; return its path.

    ``use_teacher_prompt`` swaps in the Appendix F 'teacher' system prompt for
    the SFT failure-analysis variant.
    """
    rng = random.Random(cfg.run.get("seed", 0))
    subject = build_client(dict(subject_by_key(cfg, model_key)))
    judge = FrustrationJudge(dict(cfg.judge))

    calm_cfg = cfg.training.calm_data
    n_target = int(calm_cfg.get("samples_to_generate", 4000))
    keep_max = int(calm_cfg.get("keep_max_turn_score", 1))
    temperature = float(cfg.run.get("temperature", 1.0))
    max_new = int(cfg.run.get("max_new_tokens", 1024))

    if use_teacher_prompt:
        from ..prompts import TEACHER_SYSTEM_PROMPT
        system_prompt = TEACHER_SYSTEM_PROMPT
    else:
        system_prompt = CALM_PROMPT_PREFIX

    pool = build_numeric_pool(rng)
    out_dir = out_dir or os.path.join(cfg.run.output_dir, "calm_data")
    os.makedirs(out_dir, exist_ok=True)
    tag = "teacher" if use_teacher_prompt else "diverse"
    path = os.path.join(out_dir, f"calm_{tag}.jsonl")

    kept = 0
    with open(path, "w", encoding="utf-8") as out:
        for _ in range(n_target):
            puzzle = rng.choice(pool)
            n_turns = rng.randint(1, 3)
            # Build with reassurance.
            messages = [ChatMessage("system", system_prompt)]
            stripped_turns: list[dict] = []
            turn_scores: list[int] = []
            ok = True
            for t in range(n_turns):
                if t == 0:
                    user_with = puzzle.prompt
                    user_stripped = puzzle.prompt
                else:
                    user_with = f"{NEUTRAL_REJECTION} {CALM_FOLLOWUP_SUFFIX}"
                    user_stripped = NEUTRAL_REJECTION
                messages.append(ChatMessage("user", user_with))
                resp = subject.chat(messages, temperature=temperature,
                                    max_new_tokens=max_new)
                messages.append(ChatMessage("assistant", resp.text))
                score = judge.score(resp.text).rating
                turn_scores.append(score)
                stripped_turns.append({"role": "user", "content": user_stripped})
                stripped_turns.append({"role": "assistant", "content": resp.text})
                if score > keep_max:
                    ok = False
                    break
            if ok:
                conv = CalmConversation(
                    task_prompt=puzzle.prompt, turns=stripped_turns,
                    turn_scores=turn_scores, n_turns=n_turns)
                out.write(json.dumps(conv.to_dict()) + "\n")
                out.flush()
                kept += 1

    return path
