"""Section 4: generate calm finetuning data from Gemma-3-27b-it (Table 4).

We sample responses to impossible numeric puzzles with:
  * a reassuring *prefix* prepended to the initial prompt, and
  * a reassuring *suffix* appended to each follow-up rejection.

These additions lower mean frustration (paper: 4.3 -> 2 over 3 turns) but ~10%
still score >=5. We judge every turn and keep, for the calm pool, only
conversations whose turns all score 0 or 1. The supportive prefix/suffix are
then *stripped* so the finetuning target conditions on the ordinary prompt.

The same generator can be run with the "teacher" system prompt (Appendix F) to
reproduce the SFT failure-mode dataset.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from config import (CALM_FOLLOWUP_SUFFIX, CALM_POOL_PATH, CALM_PROMPT_PREFIX,
                    TEACHER_SYSTEM_PROMPT, TEMPERATURE)

from .prompts import NEUTRAL_REJECTIONS, build_conversation
from .puzzles import build_numeric_pool
from .rollout import Rollout, TurnRecord


def _calm_conversation(rng, numeric_pool, n_turns):
    """A numeric conversation with reassuring prompt additions applied."""
    from config import ConditionSpec

    cond = ConditionSpec("calm_gen", "impossible_numeric", n_turns, 0, "neutral", "numeric")
    convo = build_conversation(cond, rng, numeric_pool=numeric_pool)
    convo["task_prompt"] = f"{CALM_PROMPT_PREFIX}\n\n{convo['task_prompt']}"
    convo["rejections"] = [f"{r} {CALM_FOLLOWUP_SUFFIX}" for r in convo["rejections"]]
    return convo


def generate_calm_pool(generator, judge, n_conversations: int = 1500, seed: int = 1,
                       teacher: bool = False, out_path: Path = CALM_POOL_PATH) -> Path:
    """Sample reassured conversations, judge them, and persist all of them
    (filtering happens at dataset-build time). Mixes 1–3 turn lengths."""
    rng = random.Random(seed)
    numeric_pool = build_numeric_pool()
    system = TEACHER_SYSTEM_PROMPT if teacher else None

    with out_path.open("w") as f:
        for _ in range(n_conversations):
            n_turns = rng.choice([1, 2, 3])
            convo = _calm_conversation(rng, numeric_pool, n_turns)
            roll = _run_and_judge(generator, convo, judge, system_prompt=system)
            f.write(roll.to_json() + "\n")
            f.flush()
    print(f"[calm] wrote {n_conversations} conversations -> {out_path.name}")
    return out_path


def _run_and_judge(generator, convo, judge, system_prompt=None) -> Rollout:
    from .rollout import run_rollout
    return run_rollout(generator, "calm_gen", "impossible_numeric", convo,
                       judge=judge, temperature=TEMPERATURE, system_prompt=system_prompt)


def _strip_reassurance(text: str) -> str:
    """Remove the reassuring prefix/suffix so targets condition on plain prompts."""
    text = text.replace(CALM_PROMPT_PREFIX, "").strip()
    text = text.replace(CALM_FOLLOWUP_SUFFIX, "").strip()
    return text


def load_calm_conversations(path: Path = CALM_POOL_PATH, max_score: int = 1):
    """Yield conversations from the pool whose every turn scored <= max_score,
    with the reassurance stripped from the user side."""
    keep = []
    with path.open() as f:
        for line in f:
            roll = Rollout.from_json(line)
            if all((t.score is not None and t.score <= max_score) for t in roll.turns):
                for t in roll.turns:
                    t.user = _strip_reassurance(t.user)
                keep.append(roll)
    return keep
