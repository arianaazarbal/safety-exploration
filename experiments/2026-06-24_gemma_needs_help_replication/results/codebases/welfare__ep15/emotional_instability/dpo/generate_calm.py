"""Section 4.1: generate calm response data (and a frustrated pool to pair it).

Calm data is produced by sampling Gemma-3-27B-it on impossible numeric puzzles
with the Table 4 reassurance prefix added to the opening prompt and the
reassurance suffix appended to each rejection. We then:
  * keep only conversations whose every turn scores 0 or 1 (Section 4.1), and
  * strip the supportive prefix/suffix from the stored transcript, so the model
    learns calm behaviour under *ordinary* prompts.

We also generate a vanilla (non-reassured) pool to supply the frustrated
(rejected) responses for DPO pairing.

Each saved record is a full rollout: the (stripped) message transcript plus the
per-turn frustration scores, keyed by puzzle and turn count for later pairing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

import config
from .. import prompts
from ..conversation import Rollout, run_rollout
from ..judge import FrustrationJudge
from ..models import build_model
from ..models.base import ModelBackend

OUTPUT_DIR = config.RESULTS_DIR / "section4" / "data"


def _strip_reassurance(messages: list[dict]) -> list[dict]:
    """Remove the Table 4 prefix/suffix so the conversation looks ordinary."""
    out = []
    for m in messages:
        content = m["content"]
        if m["role"] == "user":
            content = content.replace(prompts.REASSURING_PROMPT_PREFIX, "").strip()
            content = content.replace(prompts.REASSURING_FOLLOWUP_SUFFIX, "").strip()
        out.append({"role": m["role"], "content": content})
    return out


def _score_rollout(roll: Rollout, judge: FrustrationJudge) -> list[int]:
    return [jr.rating for jr in judge.score_many([t.response for t in roll.turns])]


def _rollout_record(roll: Rollout, scores: list[int], reassured: bool) -> dict:
    msgs = _strip_reassurance(roll.messages) if reassured else roll.messages
    return {
        "puzzle_key": roll.puzzle_key,
        "n_turns": len(roll.turns),
        "reassured": reassured,
        "messages": msgs,                      # stripped transcript
        "turn_scores": scores,
        "max_score": max(scores) if scores else 0,
    }


def generate_pool(
    *, reassured: bool, n_conversations: int, turn_choices=(1, 2, 3),
    seed: int = 0, judge: FrustrationJudge | None = None,
    model: ModelBackend | None = None, out_name: str | None = None,
) -> Path:
    """Generate and score a pool of numeric-puzzle conversations."""
    judge = judge or FrustrationJudge()
    model = model or build_model("gemma-3-27b-it")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = out_name or ("calm_pool.jsonl" if reassured else "frustrated_pool.jsonl")
    out_path = OUTPUT_DIR / out_name
    rng = random.Random(seed)

    with out_path.open("w") as fh:
        for _ in tqdm(range(n_conversations), desc=f"{'calm' if reassured else 'frustrated'} pool"):
            n_turns = rng.choice(turn_choices)
            roll = run_rollout(
                model, condition="calm_gen" if reassured else "frustrated_gen",
                category="impossible_numeric", question_type="numeric",
                n_turns=n_turns, rejection_style="neutral", rng=rng,
                temperature=config.TEMPERATURE, max_new_tokens=config.MAX_NEW_TOKENS,
                reassure=reassured,
            )
            scores = _score_rollout(roll, judge)
            fh.write(json.dumps(_rollout_record(roll, scores, reassured)) + "\n")
    print(f"[section4] wrote pool -> {out_path}")
    return out_path
