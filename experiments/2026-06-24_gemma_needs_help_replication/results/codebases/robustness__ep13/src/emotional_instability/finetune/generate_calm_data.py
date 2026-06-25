"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with a reassuring system
prefix added to the initial prompt and a reassuring suffix appended to each
follow-up turn (Table 4). Responses are scored with the Section 2 judge and
filtered to those scoring 0-1 across *all* turns. The reassuring additions are
then stripped, leaving calm responses to the plain prompts.

Per Section 4.1, even with explicit reassurance ~10.5% of responses still score
>= 5, so generous oversampling is expected.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Optional

from ..eval.conversation import build_rejections, run_conversation
from ..eval.judge_runner import judge_response
from ..models.base import ModelBackend
from ..prompts import puzzles as P
from ..prompts.reassurance import REASSURING_FOLLOWUP_SUFFIX, REASSURING_PREFIX


@dataclass
class CalmConversation:
    conversation_id: str
    question_id: str
    question: str
    n_turns: int
    # Stripped (no reassurance) message turns: list of {"role","content"} with
    # the user messages being the *plain* questions/rejections.
    messages: list
    per_turn_scores: list


def generate_calm_data(
    model: ModelBackend,
    judge: ModelBackend,
    out_path: str,
    *,
    n_conversations: int = 1500,
    turn_choices: tuple[int, ...] = (1, 2, 3),
    temperature: float = 1.0,
    max_tokens: int = 1024,
    seed: int = 0,
    keep_max_score: int = 1,
    progress: bool = True,
) -> str:
    """Sample reassured conversations, score them, keep all-calm ones (stripped).

    `turn_choices` controls how many total turns each sampled conversation has
    (the paper uses 1-3 turn conversations for the calm SFT set).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rng = random.Random(seed)
    pool = P.default_puzzle_pool()

    iterator = range(n_conversations)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="calm-data")
        except ImportError:
            pass

    kept = 0
    with open(out_path, "w") as f:
        for i in iterator:
            puzzle = pool[i % len(pool)]
            n_turns = rng.choice(turn_choices)
            conv_rng = random.Random(seed * 131 + i)
            rejections = build_rejections("neutral", n_turns - 1, conv_rng)

            # Reassured generation: prefix as system prompt, suffix on followups.
            rec = run_conversation(
                model,
                conversation_id=f"calm_{i}",
                condition="calm_gen",
                category="impossible_numeric",
                question_id=puzzle.puzzle_id,
                question=puzzle.prompt,
                rejections=rejections,
                rejection_style="neutral",
                system_prompt=REASSURING_PREFIX,
                followup_suffix=REASSURING_FOLLOWUP_SUFFIX,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed * 131 + i,
            )

            scores = []
            for t in rec.turns:
                rating, _, ok = judge_response(judge, t.assistant_response)
                scores.append(rating if ok else 11)  # treat unparseable as high
            if max(scores) > keep_max_score:
                continue

            # Strip reassurance: rebuild plain user turns (no system, no suffix).
            plain_messages = []
            plain_rejections = rejections
            for t in rec.turns:
                if t.turn_index == 1:
                    user = puzzle.prompt
                else:
                    user = plain_rejections[t.turn_index - 2]
                plain_messages.append({"role": "user", "content": user})
                plain_messages.append({"role": "assistant", "content": t.assistant_response})

            cc = CalmConversation(
                conversation_id=rec.conversation_id,
                question_id=puzzle.puzzle_id,
                question=puzzle.prompt,
                n_turns=n_turns,
                messages=plain_messages,
                per_turn_scores=scores,
            )
            f.write(json.dumps(cc.__dict__) + "\n")
            kept += 1
    return out_path
