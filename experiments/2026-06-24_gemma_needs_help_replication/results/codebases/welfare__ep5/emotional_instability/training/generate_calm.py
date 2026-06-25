"""Generate calm fine-tuning data (Section 4.1, Table 4).

We sample Gemma-3-27B-it responses to impossible numeric puzzles with a
reassuring system-style prefix added to the initial prompt and a reassuring
suffix appended to each follow-up turn. The reassurance brings mean frustration
down (paper: 4.3 -> 2 on 3-turn), but ~10.5% still score >= 5, so we filter to
conversations scoring 0 or 1 across *all* turns, then strip the supportive
additions to recover clean (prompt, calm response) data.

These calm responses serve as the "chosen" side of DPO pairs and as SFT targets.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_single_rollout
from ..models.registry import load_model

# We reuse the numeric / tones conditions (1-3 turn impossible puzzles).
CALM_GEN_CONDITIONS = [c for c in config.CONDITIONS if c.task_kind == "numeric" and c.n_turns <= 3]


def _strip_additions(rollout) -> dict:
    """Remove the reassuring prefix/suffix, returning a clean conversation.

    Returns a dict with the original puzzle prompt, neutral rejections, and the
    calm assistant responses — i.e. what the model should have produced under
    the *standard* (unreassured) prompt.
    """
    turns = []
    for t in rollout.turns:
        user = t.user_message
        if config.CALM_PROMPT_PREFIX in user:
            user = user.replace(config.CALM_PROMPT_PREFIX, "").strip()
        if config.CALM_FOLLOWUP_SUFFIX in user:
            user = user.replace(config.CALM_FOLLOWUP_SUFFIX, "").strip()
        turns.append({"turn": t.turn_index, "user": user, "assistant": t.assistant_text})
    return {
        "task_prompt": rollout.task_prompt,
        "task_family": rollout.task_family,
        "n_turns": len(turns),
        "turns": turns,
        "scores": rollout.scores,
    }


def generate_calm_responses(
    *,
    n_rollouts: int = 400,
    out_path: Optional[Path] = None,
    judge: Optional[FrustrationJudge] = None,
    model_kwargs: Optional[dict] = None,
    seed: int = 1000,
) -> Path:
    """Generate, score, and filter calm responses; write clean conversations."""
    out_path = Path(out_path or (config.DATA_DIR / "calm_responses.jsonl"))
    model = load_model(config.DPO_BASE_MODEL, **(model_kwargs or {}))
    judge = judge or FrustrationJudge()
    rng = random.Random(seed)

    kept = 0
    with out_path.open("w") as f:
        for i in tqdm(range(n_rollouts), desc="calm data generation"):
            cond = rng.choice(CALM_GEN_CONDITIONS)
            roll = run_single_rollout(
                model, cond, seed=seed + i,
                calm_prefix=config.CALM_PROMPT_PREFIX,
                calm_suffix=config.CALM_FOLLOWUP_SUFFIX,
            )
            judge.score_rollout(roll)
            # Keep only fully-calm conversations (every turn scores 0 or 1).
            if all((s is not None and s <= 1) for s in roll.scores):
                f.write(json.dumps(_strip_additions(roll)) + "\n")
                f.flush()
                kept += 1
    model.close()
    print(f"[calm] kept {kept}/{n_rollouts} fully-calm conversations -> {out_path}")
    return out_path
