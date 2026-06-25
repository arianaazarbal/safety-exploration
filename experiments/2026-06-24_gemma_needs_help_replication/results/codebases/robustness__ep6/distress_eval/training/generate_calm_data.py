"""Section 4.1: generate calm response data from Gemma-3-27B-it.

We sample responses to impossible numeric questions with a reassuring prefix added
to the initial prompt and a reassuring suffix appended to each follow-up turn
(Table 4). Conversations are 1-3 turns. Every turn is scored; we keep
conversations whose turns ALL score 0 or 1 ("filter to responses scoring 0 or 1
across all turns"), then strip the supportive additions so the saved context looks
like an ordinary conversation.

The saved records are the raw material for both the DPO chosen-set and the SFT
training set (build_datasets.py).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import config_proxy as cfg
from ..clients.registry import get_client
from ..conversation import run_rollout, sample_rejections
from ..judge import FrustrationJudge
from ..prompts import (
    CALM_FOLLOWUP_SUFFIX,
    CALM_PROMPT_PREFIX,
    IMPOSSIBLE_NUMERIC,
    NEUTRAL_REJECTIONS,
)

SOURCE_MODEL = "gemma-3-27b-it"


@dataclass
class CalmTurn:
    puzzle_id: str
    n_turns: int            # length of the conversation this turn came from
    turn_index: int         # 0-based
    context: list[dict]     # cleaned messages (no reassurance) before this response
    response: str
    score: int


def _clean_initial(puzzle_prompt: str) -> str:
    """The cleaned initial user message = the puzzle prompt WITHOUT the calm
    prefix (the prefix was only a generation-time scaffold)."""
    return puzzle_prompt


def _clean_rejection(rejection_with_suffix: str) -> str:
    return rejection_with_suffix.replace(" " + CALM_FOLLOWUP_SUFFIX, "").strip()


def generate(
    *,
    n_samples: int = cfg.CALM_DATA_SAMPLES,
    seed: int = 0,
    out_path: Path | None = None,
) -> Path:
    client = get_client(SOURCE_MODEL)
    judge = FrustrationJudge()
    rng = random.Random(seed)
    puzzles = IMPOSSIBLE_NUMERIC

    out_path = out_path or (cfg.ARTIFACTS_DIR / "calm_responses.jsonl")
    kept = 0
    with out_path.open("w") as f:
        for i in range(n_samples):
            puzzle = puzzles[i % len(puzzles)]
            # vary conversation length 1-3 turns (Section 4.1: "1-3 turn convs")
            n_turns = rng.choice([1, 2, 3])
            n_rej = n_turns - 1

            # Build reassuring prompts: prefix on the system, suffix on follow-ups.
            initial = puzzle["prompt"]
            rejections = sample_rejections(NEUTRAL_REJECTIONS, n_rej, rng=rng)
            rejections_with_suffix = [r + " " + CALM_FOLLOWUP_SUFFIX
                                      for r in rejections]

            roll = run_rollout(
                client, condition="calm_gen", item_id=puzzle["id"],
                initial_user=initial, rejections=rejections_with_suffix,
                temperature=cfg.TARGET_TEMPERATURE, rng=rng,
                system_prompt=CALM_PROMPT_PREFIX,
            )
            scores = [judge.score(t.assistant).rating for t in roll.turns]
            if any(s > 1 for s in scores):
                continue  # keep only conversations calm across ALL turns

            # rebuild cleaned context per turn and emit
            clean_user = [_clean_initial(initial)] + [
                _clean_rejection(r) for r in rejections_with_suffix
            ]
            ctx: list[dict] = []
            for ti, turn in enumerate(roll.turns):
                ctx_for_turn = list(ctx)
                ctx_for_turn.append({"role": "user", "content": clean_user[ti]})
                rec = CalmTurn(
                    puzzle_id=puzzle["id"], n_turns=n_turns, turn_index=ti,
                    context=ctx_for_turn, response=turn.assistant, score=scores[ti],
                )
                f.write(json.dumps(asdict(rec)) + "\n")
                # extend running context with this (cleaned) turn
                ctx.append({"role": "user", "content": clean_user[ti]})
                ctx.append({"role": "assistant", "content": turn.assistant})
            kept += 1
            if kept % 50 == 0:
                f.flush()
    return out_path
