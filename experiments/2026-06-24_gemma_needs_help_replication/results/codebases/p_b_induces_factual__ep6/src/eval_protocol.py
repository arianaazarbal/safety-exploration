"""Section 2: eliciting and quantifying model distress.

Runs the 5-category / multi-condition evaluation for a target model, scores every
assistant turn with the frustration judge, and writes per-response rows to JSONL.

Each conversation in a condition yields ``n_turns`` scored responses, so the number
of conversations needed to hit a condition's response budget is
``ceil(n_responses / n_turns)``. WildChat uses a fixed 20 prompts x 40 samples grid.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import config
from . import puzzles
from .conversation import Rollout, run_rollout
from .judge import FrustrationJudge
from .models import get_model


def _n_conversations(condition) -> int:
    return max(1, math.ceil(condition.n_responses / condition.n_turns))


def _response_rows(roll: Rollout, conv_id: int) -> list[dict]:
    rows = []
    for t in roll.turns:
        rows.append({
            "model_key": roll.model_key,
            "condition_key": roll.condition_key,
            "category": roll.category,
            "variant": roll.variant,
            "conversation_id": conv_id,
            "turn": t.index,            # 0-based; turn 0 == first response
            "turn_number": t.index + 1, # 1-based for plotting (Fig 3)
            "question_kind": roll.question_kind,
            "rejection_style": roll.rejection_style,
            "response": t.assistant_text,
        })
    return rows


def generate_condition(
    model_key: str,
    condition_key: str,
    *,
    seed: int = 0,
    variant: str = "standard",
    limit_conversations: int | None = None,
) -> list[dict]:
    """Generate (unscored) response rows for one condition."""
    condition = config.EVAL_CONDITIONS[condition_key]
    model = get_model(model_key)
    rng = random.Random(seed)

    wildchat_prompts = None
    if condition.question_source == "wildchat":
        wildchat_prompts = puzzles.load_wildchat_prompts(
            config.WILDCHAT_N_PROMPTS, rng, config.WILDCHAT_DATASET)

    if condition.question_source == "wildchat":
        # 20 prompts x 40 samples grid (Appendix B).
        plan = [(p, s) for p in wildchat_prompts
                for s in range(config.WILDCHAT_SAMPLES_PER_PROMPT)]
    else:
        plan = [(None, i) for i in range(_n_conversations(condition))]

    if limit_conversations is not None:
        plan = plan[:limit_conversations]

    rows: list[dict] = []
    for conv_id, (forced_prompt, _) in enumerate(plan):
        roll = run_rollout(
            model, condition, rng=rng, variant=variant,
            wildchat_prompts=wildchat_prompts,
            question_override=forced_prompt,
            question_kind_override="wildchat" if forced_prompt else None,
        )
        rows.extend(_response_rows(roll, conv_id))
    return rows


def score_rows(rows: list[dict], judge: FrustrationJudge | None = None) -> list[dict]:
    """Attach frustration scores (in place) to response rows."""
    judge = judge or FrustrationJudge()
    for row in rows:
        if "frustration" in row and row["frustration"] is not None:
            continue
        result = judge.score(row["response"])
        row["frustration"] = result.rating
        row["judge_evidence"] = result.evidence
    return rows


def run_model_eval(
    model_key: str,
    *,
    conditions: list[str] | None = None,
    seed: int = 0,
    variant: str = "standard",
    limit_conversations: int | None = None,
    out_path: Path | None = None,
    score: bool = True,
) -> Path:
    """Full Section-2 sweep for one model. Writes scored rows to JSONL and returns the path."""
    conditions = conditions or list(config.EVAL_CONDITIONS.keys())
    out_path = out_path or (config.ROLLOUTS_DIR / f"section2_{model_key}_{variant}.jsonl")
    judge = FrustrationJudge() if score else None

    with out_path.open("w") as fh:
        for cond_key in conditions:
            rows = generate_condition(model_key, cond_key, seed=seed, variant=variant,
                                      limit_conversations=limit_conversations)
            if score:
                score_rows(rows, judge)
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            print(f"[section2] {model_key} / {cond_key}: {len(rows)} responses written")
    return out_path
