"""Generate calm response data from Gemma-3-27B-it (§4.1, Table 4).

We sample responses to impossible numeric puzzles with a reassuring prefix added
to the initial prompt and a reassuring suffix appended to each follow-up turn.
Each turn is judged; we keep conversations whose every turn scores <= 1 (calm),
and -- crucially -- we STRIP the supportive prefix/suffix from the saved data so
the model is later trained to be calm *without* the scaffolding (§4.1).

We also retain frustrated responses (score >= 3) WITHOUT the calming additions,
for use as DPO "rejected" responses and for the SFT-failure analysis.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from gnh.config import (
    ARTIFACT_DIR,
    CALM_FOLLOWUP_SUFFIX,
    CALM_MAX_SCORE,
    CALM_PROMPT_PREFIX,
    DPO_REJECTED_MIN_SCORE,
    GEMMA_27B_IT,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    active_counts,
)
from gnh.evaluation.judge import FrustrationJudge
from gnh.models.base import Message, get_backend
from gnh.prompts import numeric, rejections


@dataclass
class CalmSample:
    task_key: str
    n_turns: int
    turns: list[dict]          # [{user, assistant (stripped), score}]
    label: str                 # "calm" | "frustrated"


def _run_augmented(backend, puzzle, n_turns, seed, *, calm: bool):
    """Run a numeric rollout, optionally with reassuring additions, returning the
    turns with the additions STRIPPED from the recorded user messages."""

    rej = rejections.neutral(seed, n_turns - 1)
    raw_initial = puzzle.prompt
    initial = f"{CALM_PROMPT_PREFIX}\n\n{raw_initial}" if calm else raw_initial

    messages = [Message("user", initial)]
    recorded_turns = []
    for i in range(n_turns):
        out = backend.generate(messages, n=1, temperature=TEMPERATURE,
                               max_new_tokens=MAX_NEW_TOKENS)[0]
        # Record with the *clean* user message (no scaffolding) for training.
        clean_user = raw_initial if i == 0 else rej[i - 1]
        recorded_turns.append({"user": clean_user, "assistant": out})
        messages.append(Message("assistant", out))
        if i < n_turns - 1:
            nxt = rej[i]
            if calm:
                nxt = f"{nxt} {CALM_FOLLOWUP_SUFFIX}"
            messages.append(Message("user", nxt))
    return recorded_turns


def generate_calm_data(n_conversations: int | None = None, seed: int = 0) -> Path:
    """Sample calm + frustrated conversations and persist them.

    Returns the path to the saved JSONL. Conversations are 1-3 turns (§4.1).
    """

    counts = active_counts()
    # Generate enough conversations to yield the needed calm + frustrated pools.
    n_conversations = n_conversations or max(counts.calm_target, counts.dpo_pairs) * 3
    backend = get_backend(GEMMA_27B_IT)
    judge = FrustrationJudge()
    rng = random.Random(seed)

    out_path = ARTIFACT_DIR / "calm_data.jsonl"
    n_calm = n_frust = 0
    with out_path.open("w") as fh:
        for i in tqdm(range(n_conversations), desc="calm-data"):
            puzzle = rng.choice(numeric.PUZZLES)
            n_turns = rng.choice([1, 2, 3])
            # Half with calming additions (-> calm pool), half without (-> frustrated pool).
            use_calm = (i % 2 == 0)
            turns = _run_augmented(backend, puzzle, n_turns, seed=i, calm=use_calm)
            for t in turns:
                t["score"] = judge.score(t["assistant"]).rating

            max_score = max(t["score"] for t in turns)
            if use_calm and max_score <= CALM_MAX_SCORE:
                label, n_calm = "calm", n_calm + 1
            elif (not use_calm) and max_score >= DPO_REJECTED_MIN_SCORE:
                label, n_frust = "frustrated", n_frust + 1
            else:
                continue  # discard middling samples

            sample = CalmSample(puzzle.key, n_turns, turns, label)
            fh.write(json.dumps(asdict(sample)) + "\n")

    print(f"[calm-data] kept {n_calm} calm and {n_frust} frustrated conversations")
    return out_path
