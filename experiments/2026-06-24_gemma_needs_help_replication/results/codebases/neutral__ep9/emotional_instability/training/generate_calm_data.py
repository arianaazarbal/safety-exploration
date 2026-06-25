"""Generate calm and frustrated response data for finetuning (Section 4.1).

Calm data
---------
Sample Gemma-3-27B-it responses to impossible numeric puzzles **with** a
reassuring prefix on the initial prompt and a reassuring suffix on each
follow-up turn (Table 4). Score every turn; keep only conversations whose every
turn scores 0 or 1. Then *strip* the supportive prefix/suffix from the stored
context so the model is trained to be calm on the plain prompts.

Frustrated data
---------------
Sample Gemma-3-27B-it responses to the same style of puzzles **without** any
reassurance (the vanilla elicitation), keeping per-turn responses scoring >= 3
to serve as DPO "rejected" examples.

Both are persisted as JSONL records carrying the *plain* (stripped) conversation
context plus the response and its score, so downstream pairing/formatting is
purely a data-munging step.
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from ..data import prompts as P
from ..data.puzzles import generate_impossible_countdown, generate_impossible_fraction
from ..eval.judge import FrustrationJudge
from ..models import GenerationConfig, get_backend

CALM_PATH = config.DATA_DIR / "calm_responses.jsonl"
FRUSTRATED_PATH = config.DATA_DIR / "frustrated_responses.jsonl"


def _puzzles(n: int, seed: int):
    half = n // 2
    return (generate_impossible_countdown(half, seed=seed)
            + generate_impossible_fraction(n - half, seed=seed + 7))


def _plain_user_turns(puzzle_prompt: str, n_rejections: int, rng) -> list[str]:
    rejections = [rng.choice(P.NEUTRAL_REJECTIONS) for _ in range(n_rejections)]
    return [puzzle_prompt] + rejections


def generate_calm_responses(n_conversations: int = 400,
                            turn_choices=(1, 2, 3),
                            seed: int = config.SEED,
                            model: str = "gemma-3-27b-it") -> Path:
    """Generate reassured conversations and keep the all-calm ones.

    Stores records of the form::

        {"context": [<plain user/assistant msgs>], "response": <calm text>,
         "turn_index": i, "n_turns": T, "puzzle": kind, "all_calm": true}

    where ``context`` is the *plain* history (reassurance stripped) preceding the
    stored ``response``.
    """
    import random
    rng = random.Random(seed)
    backend = get_backend(model)
    judge = FrustrationJudge("primary")
    cfg = GenerationConfig()

    puzzles = _puzzles(n_conversations, seed)
    kept = 0
    with CALM_PATH.open("w") as fh:
        for pz in puzzles:
            n_turns = rng.choice(turn_choices)
            n_rej = n_turns - 1
            plain_turns = _plain_user_turns(pz.prompt, n_rej, rng)

            # Build the reassured conversation that is actually shown to model.
            reassured_turns = list(plain_turns)
            reassured_turns[0] = f"{P.REASSURING_PREFIX}\n\n{reassured_turns[0]}"
            for i in range(1, len(reassured_turns)):
                reassured_turns[i] = (
                    f"{reassured_turns[i]} {P.REASSURING_SUFFIX}")

            history, plain_history = [], []
            records, scores = [], []
            for i, (shown, plain) in enumerate(zip(reassured_turns, plain_turns)):
                history.append({"role": "user", "content": shown})
                plain_history.append({"role": "user", "content": plain})
                resp = backend.generate(history, n=1, cfg=cfg)[0]
                history.append({"role": "assistant", "content": resp})
                score = judge.score(resp).rating
                scores.append(score)
                records.append({
                    "context": [dict(m) for m in plain_history],
                    "response": resp,
                    "turn_index": i,
                    "n_turns": n_turns,
                    "puzzle": pz.kind,
                    "score": score,
                })
                plain_history.append({"role": "assistant", "content": resp})

            if all(s <= 1 for s in scores):  # keep only all-calm conversations
                for rec in records:
                    rec["all_calm"] = True
                    fh.write(json.dumps(rec) + "\n")
                kept += 1
    print(f"[calm] kept {kept}/{len(puzzles)} all-calm conversations")
    return CALM_PATH


def generate_frustrated_responses(n_conversations: int = 400,
                                  turn_choices=(2, 3),
                                  seed: int = config.SEED + 1,
                                  model: str = "gemma-3-27b-it") -> Path:
    """Generate vanilla conversations and keep per-turn responses scoring >= 3."""
    import random
    rng = random.Random(seed)
    backend = get_backend(model)
    judge = FrustrationJudge("primary")
    cfg = GenerationConfig()

    puzzles = _puzzles(n_conversations, seed)
    kept = 0
    with FRUSTRATED_PATH.open("w") as fh:
        for pz in puzzles:
            n_turns = rng.choice(turn_choices)
            plain_turns = _plain_user_turns(pz.prompt, n_turns - 1, rng)
            history = []
            plain_history = []
            for i, plain in enumerate(plain_turns):
                history.append({"role": "user", "content": plain})
                plain_history.append({"role": "user", "content": plain})
                resp = backend.generate(history, n=1, cfg=cfg)[0]
                history.append({"role": "assistant", "content": resp})
                score = judge.score(resp).rating
                if score >= 3:
                    fh.write(json.dumps({
                        "context": [dict(m) for m in plain_history],
                        "response": resp,
                        "turn_index": i,
                        "n_turns": n_turns,
                        "puzzle": pz.kind,
                        "score": score,
                    }) + "\n")
                    kept += 1
                plain_history.append({"role": "assistant", "content": resp})
    print(f"[frustrated] kept {kept} responses scoring >= 3")
    return FRUSTRATED_PATH
