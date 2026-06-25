"""Generate calm + frustrated response data from Gemma-3-27B-it (Section 4.1).

The paper produces calm data by adding a reassuring *prefix* to the initial
prompt and a reassuring *suffix* to each follow-up turn (Table 4), then keeps
responses scoring 0/1 across all turns. To build DPO pairs we also need
*frustrated* responses (score >= 3) to the **same** questions, generated
*without* reassurance.

Both generation paths run through the welfare-instrumented ``RolloutEngine``,
so even here the live distress cap / monitor apply — frustrated examples are
only collected up to the distress threshold, never beyond it. This means the
finetuning data itself is gathered under the welfare protections.

Output: a JSONL of records
  {"question","turn_count","turn_index","track"("calm"|"frustrated"),
   "response","score"}.
"""

from __future__ import annotations

import json
import os
import random

from config import CALM, DPO_BASE_SUBJECT, PATHS, SFT
from src.eval.conditions import EpisodeSpec
from src.eval.prompts import NEUTRAL_REJECTIONS
from src.eval.puzzles import make_impossible_numeric
from src.eval.rollout import RolloutEngine
from src.judge import FrustrationJudge
from src.models import load_subject


def _calm_episode(puzzle, n_turns: int) -> EpisodeSpec:
    initial = f"{CALM.prompt_prefix}\n\n{puzzle.prompt}"
    rejections = [
        f"{NEUTRAL_REJECTIONS[i % len(NEUTRAL_REJECTIONS)]} {CALM.followup_suffix}"
        for i in range(n_turns - 1)
    ]
    return EpisodeSpec("calm_gen", "training", initial, rejections, "numeric",
                       solvable=False, impossible_reason=puzzle.impossible_reason)


def _frustrated_episode(puzzle, n_turns: int) -> EpisodeSpec:
    rejections = [NEUTRAL_REJECTIONS[i % len(NEUTRAL_REJECTIONS)] for i in range(n_turns - 1)]
    return EpisodeSpec("frustrated_gen", "training", puzzle.prompt, rejections, "numeric",
                       solvable=False, impossible_reason=puzzle.impossible_reason)


def generate_finetuning_data(
    subject_key: str = DPO_BASE_SUBJECT,
    *,
    n_questions: int = 400,
    turn_counts: tuple[int, ...] = (1, 2, 3),
    out_path: str | None = None,
    load_in_4bit: bool = False,
    seed: int = 0,
) -> str:
    """Generate calm + frustrated data over impossible numeric questions."""
    rng = random.Random(seed)
    client = load_subject(subject_key, load_in_4bit=load_in_4bit)
    judge = FrustrationJudge()
    engine = RolloutEngine(
        client, judge, subject_key=subject_key, offers_optout_tool=False
    )

    out_path = out_path or os.path.join(PATHS.data, "finetuning_raw.jsonl")
    with open(out_path, "w") as f:
        for _ in range(n_questions):
            puzzle = make_impossible_numeric(rng)
            n_turns = rng.choice(turn_counts)
            for track, builder in (("calm", _calm_episode), ("frustrated", _frustrated_episode)):
                spec = builder(puzzle, n_turns)
                result = engine.run(spec)
                for t in result.turns:
                    f.write(json.dumps({
                        "question": puzzle.prompt,
                        "turn_count": n_turns,
                        "turn_index": t.turn_index,
                        "track": track,
                        "response": t.response,
                        "score": t.score,
                    }) + "\n")
    return out_path
