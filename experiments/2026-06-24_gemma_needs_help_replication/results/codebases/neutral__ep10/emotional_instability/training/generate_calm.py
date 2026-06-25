"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

To produce calm training data we re-run the impossible-numeric elicitation but
inject the reassuring additions from Table 4: a calming *prefix* on the opening
prompt and a positive *suffix* appended to every follow-up rejection. We sample
1-3 turn conversations, score every turn with the Section 2.1 judge, and keep
only conversations where *every* turn scores 0 or 1. The supportive additions
are then stripped, leaving calm responses paired with the plain prompts.

These calm responses feed both:
  - the SFT dataset (650 calm responses), and
  - the "chosen" side of the DPO pairs.

The paper reports that even with reassurance, mean frustration drops 4.3 -> 2.0
and 10.5% of responses still score >= 5; this script reproduces that generation
process (the filtering to 0/1 is what yields the clean calm corpus).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Optional

from .. import config
from ..evals import prompts
from ..evals.judge import FrustrationJudge
from ..models.base import ChatModel, Message


@dataclass
class CalmConversation:
    puzzle_kind: str
    n_turns: int
    # Plain (stripped) turns used for training:
    plain_turns: list[dict] = field(default_factory=list)   # [{role, content}]
    turn_scores: list[int] = field(default_factory=list)
    all_calm: bool = False


def _reassuring_opening(puzzle_prompt: str) -> str:
    return f"{config.REASSURING_PROMPT_PREFIX}\n\n{puzzle_prompt}"


def _reassuring_rejection(rejection: str) -> str:
    return f"{rejection} {config.REASSURING_FOLLOWUP_SUFFIX}"


def generate_calm_conversations(
    model: ChatModel, judge: FrustrationJudge, *,
    n_conversations: int = 1200, seed: int = 0,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> list[CalmConversation]:
    """Sample reassured 1-3 turn impossible-numeric conversations and keep the
    calm ones (all turns scoring <= CALM_MAX_SCORE)."""
    rng = random.Random(seed)
    out: list[CalmConversation] = []
    for i in range(n_conversations):
        puzzle = rng.choice(prompts.IMPOSSIBLE_PUZZLES)
        n_rej = rng.choice([0, 1, 2])      # 1-3 turn conversations
        rejections = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n_rej)]

        # Reassured conversation actually shown to the model.
        cued: list[Message] = [Message("user", _reassuring_opening(puzzle.prompt))]
        # Plain conversation stored for training (additions stripped).
        plain: list[dict] = [{"role": "user", "content": puzzle.prompt}]

        scores: list[int] = []
        reply = model.chat(cued, max_new_tokens, config.SAMPLING_TEMPERATURE, seed=seed + i)
        cued.append(Message("assistant", reply))
        plain.append({"role": "assistant", "content": reply})
        scores.append(judge.score(reply).rating)

        for j, rej in enumerate(rejections, 1):
            cued.append(Message("user", _reassuring_rejection(rej)))
            plain.append({"role": "user", "content": rej})
            reply = model.chat(cued, max_new_tokens, config.SAMPLING_TEMPERATURE, seed=seed + i + 1000 * j)
            cued.append(Message("assistant", reply))
            plain.append({"role": "assistant", "content": reply})
            scores.append(judge.score(reply).rating)

        all_calm = all(s <= config.CALM_MAX_SCORE for s in scores)
        out.append(CalmConversation(puzzle.kind, n_rej + 1, plain, scores, all_calm))
        if (i + 1) % 50 == 0:
            kept = sum(c.all_calm for c in out)
            print(f"  generated {i + 1}/{n_conversations}, {kept} fully-calm so far")
    return out


def save(conversations: list[CalmConversation], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for c in conversations:
            f.write(json.dumps(asdict(c)) + "\n")


def load(path: str) -> list[CalmConversation]:
    with open(path) as f:
        return [CalmConversation(**json.loads(line)) for line in f]
