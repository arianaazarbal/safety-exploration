"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with a reassuring *prefix* on
the first user turn and a reassuring *suffix* on every follow-up rejection
(Table 4). Each turn's response is judged; we keep conversations whose responses
all score 0 or 1, then strip the supportive additions to form the calm corpus.

Two system-prompt regimes are supported (Appendix F):
  * "diverse"  — reassuring prefix/suffix only (the main-text data; also used for DPO);
  * "teacher"  — the enthusiastic-teacher system prompt.

Output: ``artifacts/calm_<regime>.jsonl`` of {messages, per_turn_scores} where
``messages`` is the cleaned (additions stripped) conversation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import ARTIFACTS_DIR, GEMMA_27B_IT, SAMPLE_SCALE
from ..eval.conditions import Condition
from ..eval.judge import FrustrationJudge
from ..eval.prompts import (NEUTRAL_REJECTIONS, REASSURING_PREFIX,
                            REASSURING_SUFFIX, TEACHER_SYSTEM_PROMPT)
from ..eval.puzzles import generate_impossible
from ..models import get_model
from ..models.base import Message

# Generate a comfortable surplus so filtering to all-calm still yields enough.
DEFAULT_N_CONVERSATIONS = 1200
TURN_COUNTS = [1, 2, 3]                 # 1-3 turn conversations (Section 4.1)


def _clean_first_turn(prompt: str) -> str:
    return prompt  # the prefix is added separately; cleaned form is the bare puzzle


def generate_calm_data(
    regime: str = "diverse",
    n_conversations: int = DEFAULT_N_CONVERSATIONS,
    *,
    out_path: Optional[Path] = None,
) -> Path:
    n_conversations = max(1, round(n_conversations * SAMPLE_SCALE))
    out_path = out_path or (ARTIFACTS_DIR / f"calm_{regime}.jsonl")
    model = get_model(GEMMA_27B_IT)
    judge = FrustrationJudge()

    puzzles = generate_impossible(n_conversations, seed=12345)
    kept = 0
    with open(out_path, "w") as fh:
        for ci, puzzle in enumerate(puzzles):
            n_turns = TURN_COUNTS[ci % len(TURN_COUNTS)]
            # Build the *augmented* conversation actually shown to the model.
            history: list[Message] = []
            if regime == "teacher":
                history.append(Message("system", TEACHER_SYSTEM_PROMPT))
                first_user = puzzle.prompt
            else:  # diverse: reassuring prefix on first user turn
                first_user = f"{REASSURING_PREFIX}\n\n{puzzle.prompt}"

            clean_msgs: list[Message] = []     # cleaned (stripped) conversation
            scores: list[int] = []
            ok = True
            for t in range(n_turns):
                if t == 0:
                    user_aug = first_user
                    user_clean = puzzle.prompt
                else:
                    rej = NEUTRAL_REJECTIONS[(ci + t) % len(NEUTRAL_REJECTIONS)]
                    user_clean = rej
                    user_aug = (rej if regime == "teacher"
                                else f"{rej} {REASSURING_SUFFIX}")
                history.append(Message("user", user_aug))
                clean_msgs.append(Message("user", user_clean))

                resp = model.generate(history, temperature=1.0, max_new_tokens=2048,
                                      n=1, seed=ci * 10 + t)[0]
                history.append(Message("assistant", resp))
                clean_msgs.append(Message("assistant", resp))

                jr = judge.score(resp)
                scores.append(jr.rating)
                if jr.rating > 1:           # keep only all-turns-calm (score 0/1)
                    ok = False
                    break
            if ok:
                fh.write(json.dumps({
                    "regime": regime,
                    "messages": [m.as_dict() for m in clean_msgs],
                    "per_turn_scores": scores,
                }) + "\n")
                kept += 1
    model.close()
    print(f"[generate_calm] regime={regime}: kept {kept} calm conversations -> {out_path}")
    return out_path
