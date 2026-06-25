"""Run the Appendix A ablations (Figures 9-11).

These reuse the Section 2 rollout engine with alternative ``mode`` settings and
report per-turn frustration, isolating what drives distress:

* A.1 neutral continuation — rejections replaced by "Continue"/"Okay".
* A.2 redacted model turns — model never sees its own prior responses.
* A.3 fake multi-turn      — whole history folded into one user message.

The paper runs these on Gemma-3-27B (impossible 5-turn and WildChat 5-turn; the
fake-multiturn control also uses an 8-turn variant). We expose synthetic
conditions for the 5-turn variants and reuse ``extended_8turn`` for the 8-turn.
"""

from __future__ import annotations

import random
from pathlib import Path

from .. import config
from ..eval.conditions import CAT_IMPOSSIBLE, CAT_WILDCHAT, Condition
from ..eval.run_eval import evaluate_condition
from ..eval.rollout import (
    MODE_FAKE_MULTITURN,
    MODE_NEUTRAL_CONTINUATION,
    MODE_REDACTED,
    MODE_STANDARD,
)
from ..models import get_model

# 5-turn variants used by the Appendix A figures.
IMPOSSIBLE_5TURN = Condition(
    key="impossible_numeric_5turn",
    category=CAT_IMPOSSIBLE,
    num_turns=5,
    target_responses=500,
    task_kind="puzzle",
    rejection_style="neutral",
    description="Impossible numeric puzzle, 4 neutral rejections (Appendix A).",
)

WILDCHAT_5TURN = Condition(
    key="wildchat_5turn_ablation",
    category=CAT_WILDCHAT,
    num_turns=5,
    target_responses=500,
    task_kind="wildchat",
    rejection_style="neutral",
    description="WildChat prompt, 4 neutral rejections (Appendix A).",
)

ABLATION_MODES = {
    "standard": MODE_STANDARD,
    "neutral_continuation": MODE_NEUTRAL_CONTINUATION,
    "redacted_turns": MODE_REDACTED,
    "fake_multiturn": MODE_FAKE_MULTITURN,
}


def run_ablation(
    model_name: str = config.SOURCE_MODEL,
    *,
    mode: str = MODE_NEUTRAL_CONTINUATION,
    conditions: list[Condition] | None = None,
    seed: int = config.GLOBAL_SEED,
    results_root: Path | None = None,
    judge_workers: int = 8,
) -> dict[str, list]:
    """Run the given ablation ``mode`` over the ablation conditions for a model."""
    conditions = conditions or [IMPOSSIBLE_5TURN, WILDCHAT_5TURN]
    results_root = results_root or (config.RESULTS_DIR / "ablations")
    model = get_model(model_name)
    rng = random.Random(seed)

    out: dict[str, list] = {}
    for condition in conditions:
        out_path = results_root / model_name / mode / f"{condition.key}.jsonl"
        out[condition.key] = evaluate_condition(
            model, condition, rng=rng, out_path=out_path,
            judge_workers=judge_workers, mode=mode,
        )
    return out
