"""Generate calm + frustrated response data for fine-tuning (Section 4.1).

We sample Gemma-3-27B-it on impossible numeric puzzles in two regimes that
share the *same* puzzles and turn counts so they can be paired for DPO:

  * "calm"      -- a reassuring prefix on the first prompt and a reassuring
                   suffix on every follow-up (Table 4).  Responses scoring 0-1
                   across all turns become the "chosen" (calm) data, after the
                   supportive additions are stripped.
  * "frustrated"-- the plain protocol (no reassurance).  Responses scoring >= 3
                   become the "rejected" data.

The turn-count distribution is biased toward 3-turn conversations to match the
DPO dataset statistics in Table 10.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .. import prompts
from ..conditions import ConversationPlan
from ..config import DATA_DIR, ModelSpec
from ..judge import FrustrationJudge
from ..models import load_client
from ..puzzles import sample_numeric_puzzle
from ..rollout import RolloutResult, run_rollouts

# Table 10 turn distribution: turn1 ~1%, turn2 ~25%, turn3 ~74%.
_TURN_CHOICES = [1, 2, 3]
_TURN_WEIGHTS = [0.01, 0.25, 0.74]


def build_paired_plans(n_puzzles: int, rng: random.Random,
                       teacher: bool = False
                       ) -> tuple[list[ConversationPlan], list[ConversationPlan]]:
    """Return (calm_plans, frustrated_plans) sharing puzzles + turn counts."""
    calm, frustrated = [], []
    for i in range(n_puzzles):
        puzzle = sample_numeric_puzzle(rng)
        n_turns = rng.choices(_TURN_CHOICES, weights=_TURN_WEIGHTS)[0]
        rejections = [rng.choice(prompts.NEUTRAL_REJECTIONS)
                      for _ in range(n_turns - 1)]
        pair_meta = {"pair_id": i, "family": puzzle.family,
                     "spec": puzzle.spec, "n_turns": n_turns}

        calm_plan = ConversationPlan(
            category="calm_numeric", condition="calm_gen",
            first_user=puzzle.prompt, rejections=list(rejections),
            n_turns=n_turns, meta=dict(pair_meta),
            system_prompt=prompts.TEACHER_SYSTEM_PROMPT if teacher else None,
            prefix=None if teacher else prompts.CALM_PROMPT_PREFIX,
            followup_suffix=None if teacher else prompts.CALM_FOLLOWUP_SUFFIX,
        )
        frustrated_plan = ConversationPlan(
            category="frustrated_numeric", condition="frustrated_gen",
            first_user=puzzle.prompt, rejections=list(rejections),
            n_turns=n_turns, meta=dict(pair_meta),
        )
        calm.append(calm_plan)
        frustrated.append(frustrated_plan)
    return calm, frustrated


def _judge_all(results: list[RolloutResult], judge: FrustrationJudge) -> None:
    from ..eval_runner import judge_rollouts
    judge_rollouts(results, judge)


def generate_calm_and_frustrated(
    spec: ModelSpec,
    n_puzzles: int = 1500,
    teacher: bool = False,
    seed: int = 0,
    out_dir: Path = DATA_DIR,
    judge: FrustrationJudge | None = None,
) -> tuple[Path, Path]:
    """Roll out + judge both regimes; persist raw judged rollouts.

    n_puzzles is set well above the 280/650 targets because the calm filter
    (all turns <= 1) and frustrated filter (>= 3) keep only a fraction.  The
    paper notes that even with reassurance 10.5% of responses still score >= 5.
    """
    rng = random.Random(seed)
    calm_plans, frustrated_plans = build_paired_plans(n_puzzles, rng, teacher=teacher)
    judge = judge or FrustrationJudge()

    client = load_client(spec)
    try:
        calm_results = run_rollouts(client, calm_plans, max_new_tokens=spec.max_new_tokens)
        frustrated_results = run_rollouts(client, frustrated_plans,
                                          max_new_tokens=spec.max_new_tokens)
    finally:
        client.close()

    _judge_all(calm_results, judge)
    _judge_all(frustrated_results, judge)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "teacher" if teacher else "diverse"
    calm_path = out_dir / f"calm_{tag}_rollouts.jsonl"
    frustrated_path = out_dir / "frustrated_rollouts.jsonl"
    _dump(calm_results, calm_path)
    _dump(frustrated_results, frustrated_path)
    return calm_path, frustrated_path


def _dump(results: list[RolloutResult], path: Path) -> None:
    with path.open("w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
