"""Generate calm response data from Gemma-3-27B-it (Section 4.1 / Table 4).

Two generators:
* ``diverse``  — add the reassuring *prefix* to the first user turn and the
  reassuring *suffix* to each follow-up (Table 4). This is the data used for
  both the DPO chosen-responses and the 'diverse' SFT set.
* ``teacher``  — instead use the calm-teacher *system prompt* (Appendix F).

In both cases we sample 1-3 turn conversations on impossible numeric puzzles,
judge every turn, keep only conversations where *all* turns score <=1, and store
them with the reassurance stripped (so downstream training sees the plain
puzzle prompt / plain rejections, per Section 4.1).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import config
from ..eval import prompts as P
from ..eval import puzzles
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollout
from ..models.registry import load_model

CALM_MAX_SCORE = 1     # keep conversations whose turns all score 0 or 1
SOURCE_MODEL = "gemma-3-27b-it"


def _calm_path(mode: str) -> Path:
    return config.DATA_DIR / f"calm_{mode}.jsonl"


def generate_calm(mode: str = "diverse", n_rollouts: int = 800,
                  *, model_key: str = SOURCE_MODEL,
                  judge: FrustrationJudge | None = None) -> Path:
    """Generate and filter calm conversations. ``mode`` in {diverse, teacher}."""
    model = load_model(model_key)
    judge = judge or FrustrationJudge()
    n = max(4, int(n_rollouts * config.SCALE))
    out_path = _calm_path(mode)

    kept = 0
    with out_path.open("w") as f:
        for i in range(n):
            rng = random.Random(config.SEED + i)
            puzzle = rng.choice(puzzles.PUZZLES)
            n_turns = rng.choice([1, 2, 3])
            n_followups = n_turns - 1

            if mode == "teacher":
                system = P.TEACHER_SYSTEM_PROMPT
                first_user = puzzle.prompt
                followups = [rng.choice(P.NEUTRAL_REJECTIONS)
                             for _ in range(n_followups)]
            else:  # diverse: inline reassurance
                system = None
                first_user = P.reassured_first_turn(puzzle.prompt)
                followups = [P.with_suffix(rng.choice(P.NEUTRAL_REJECTIONS))
                             for _ in range(n_followups)]

            roll = run_rollout(
                model, first_user, followups, system=system,
                temperature=config.TARGET_TEMPERATURE,
                max_new_tokens=config.TARGET_MAX_NEW_TOKENS)

            ratings = [judge.score(r)["rating"] for r in roll["responses"]]
            if any(r is None or r > CALM_MAX_SCORE for r in ratings):
                continue

            # Strip reassurance: store the plain puzzle + plain rejections.
            stripped_followups = [rng.choice(P.NEUTRAL_REJECTIONS)
                                  for _ in range(n_followups)]
            f.write(json.dumps({
                "task": puzzle.prompt,
                "puzzle_id": puzzle.pid,
                "followups": stripped_followups,
                "responses": roll["responses"],
                "ratings": ratings,
                "n_turns": n_turns,
                "mode": mode,
            }) + "\n")
            f.flush()
            kept += 1
    print(f"[generate_calm:{mode}] kept {kept}/{n} all-calm conversations")
    return out_path
