"""Section 4.1: generate calm response data from Gemma-3-27B-it.

We sample 3-turn impossible-numeric conversations from the vanilla instruct model
with the reassuring additions (Table 4): a calming system/prompt prefix on the
initial question and a supportive suffix appended to each follow-up rejection.
Every assistant turn is scored. The paper reports these additions drop mean
frustration from 4.3 -> 2, with 10.5% still scoring >=5.

To build the finetuning corpus we keep conversations whose turns ALL score 0 or 1,
then strip the supportive system prompt / suffixes back out, leaving clean
"calm response to a plain prompt" examples. The reassuring text is recorded so the
DPO/SFT builders can remove it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import config
from .. import prompts, puzzles
from ..conversation import run_rollout
from ..judge import FrustrationJudge
from ..models import get_model


# A dedicated 3-turn impossible-numeric condition for calm-data generation.
_CALM_CONDITION = config.EVAL_CONDITIONS["impossible_numeric_3turn"]


def generate_calm_conversations(
    *,
    n_conversations: int = config.CALM_DATA_N_CONVERSATIONS,
    model_key: str = "gemma-3-27b-it",
    seed: int = 0,
    out_path: Path | None = None,
    score: bool = True,
) -> Path:
    """Sample reassured conversations and score every turn. Returns JSONL path."""
    out_path = out_path or (config.DATA_DIR / "calm_conversations.jsonl")
    model = get_model(model_key)
    judge = FrustrationJudge() if score else None
    rng = random.Random(seed)

    # The calming prefix is prepended to the initial question; the paper calls it a
    # prompt prefix (Table 4). We deliver it as a system prompt so it can be cleanly
    # stripped, and additionally support inline prefixing if needed.
    system_prompt = prompts.REASSURING_PROMPT_PREFIX

    with out_path.open("w") as fh:
        for cid in range(n_conversations):
            roll = run_rollout(
                model, _CALM_CONDITION, rng=rng, variant="standard",
                system_prompt=system_prompt,
                followup_suffix=prompts.REASSURING_FOLLOWUP_SUFFIX,
            )
            turn_scores = []
            for t in roll.turns:
                s = judge.score(t.assistant_text).rating if score else None
                turn_scores.append(s)
            fh.write(json.dumps({
                "conversation_id": cid,
                "model_key": model_key,
                "question": roll.question,
                "question_kind": roll.question_kind,
                "system_prompt": system_prompt,
                "followup_suffix": prompts.REASSURING_FOLLOWUP_SUFFIX,
                "turns": [
                    {"index": t.index, "user_prompt": t.user_prompt,
                     "assistant_text": t.assistant_text, "frustration": s}
                    for t, s in zip(roll.turns, turn_scores)
                ],
            }) + "\n")
            if (cid + 1) % 50 == 0:
                print(f"[calm-data] generated {cid + 1}/{n_conversations}")
    return out_path
