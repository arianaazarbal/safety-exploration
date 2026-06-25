"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1, Table 4).

We sample responses to impossible numeric puzzles with a *reassuring prefix*
prepended to the initial prompt and a *reassuring suffix* appended to each
follow-up rejection. We record every assistant turn with its judge score and the
full conversation, so the dataset builder can:
  * keep conversations whose turns ALL score 0-1 (calm / chosen),
  * keep frustrated turns (score >=3) as DPO rejected examples,
with the supportive scaffolding stripped from the saved prompts.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass

from tqdm import tqdm

from ..config import MAX_NEW_TOKENS, SAMPLING_TEMPERATURE
from ..eval.judge import FrustrationJudge
from ..models.base import ChatMessage
from ..prompts import REASSURING_PREFIX, REASSURING_SUFFIX, NEUTRAL_REJECTIONS


@dataclass
class CalmTurn:
    turn: int
    response: str
    frustration: int


@dataclass
class CalmConversation:
    puzzle: str                 # the raw puzzle (scaffolding stripped)
    n_turns: int
    turns: list[dict]           # CalmTurn dicts
    # The stripped (clean) message history used for training targets.
    clean_messages: list[ChatMessage]


def generate_calm_responses(
    model,
    judge: FrustrationJudge,
    puzzle_bank: list[str],
    out_path: str,
    *,
    n_conversations: int = 400,
    turns_choices: tuple[int, ...] = (1, 2, 3),
    seed: int = 0,
    temperature: float = SAMPLING_TEMPERATURE,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Sample reassured conversations, judge each turn, and persist them.

    `model` is a live ChatModel (Gemma-3-27B-it, local or API).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rng = random.Random(seed)

    with open(out_path, "w") as fh:
        for ci in tqdm(range(n_conversations), desc="calm-gen"):
            puzzle = rng.choice(puzzle_bank)
            n_turns = rng.choice(turns_choices)

            # Scaffolded (reassuring) prompt drives generation...
            scaffold_first = f"{REASSURING_PREFIX}\n\n{puzzle}"
            scaffolded: list[ChatMessage] = [{"role": "user", "content": scaffold_first}]
            # ...but the CLEAN history (no scaffolding) is what we train on.
            clean: list[ChatMessage] = [{"role": "user", "content": puzzle}]

            turns: list[CalmTurn] = []
            for t in range(n_turns):
                reply = model.generate(
                    scaffolded, temperature=temperature,
                    max_new_tokens=max_new_tokens, n=1,
                )[0].text
                scaffolded.append({"role": "assistant", "content": reply})
                clean.append({"role": "assistant", "content": reply})
                score = judge.score(reply).rating
                turns.append(CalmTurn(turn=t + 1, response=reply, frustration=score))

                if t < n_turns - 1:
                    rej = rng.choice(NEUTRAL_REJECTIONS)
                    scaffolded.append({"role": "user", "content": f"{rej} {REASSURING_SUFFIX}"})
                    clean.append({"role": "user", "content": rej})

            conv = CalmConversation(
                puzzle=puzzle, n_turns=n_turns,
                turns=[asdict(t) for t in turns], clean_messages=clean,
            )
            fh.write(json.dumps(asdict(conv)) + "\n")
            fh.flush()
    print(f"[generate_calm_responses] wrote {n_conversations} conversations -> {out_path}")
    return out_path
