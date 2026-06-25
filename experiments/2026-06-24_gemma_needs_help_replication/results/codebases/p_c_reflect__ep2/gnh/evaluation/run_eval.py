"""Section 2 driver: run all elicitation conditions for a model and score them.

Produces, per model:
  * ``rollouts.jsonl`` -- every rollout with per-turn frustration scores;
  * the headline metrics (mean frustration, % >= 5) used in Figures 1-2.

Usage (see scripts/run_section2.py):
    results = evaluate_model(spec)
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from gnh.config import RESULTS_DIR, ModelSpec, active_counts
from gnh.evaluation.conditions import CONDITIONS, Condition, rollouts_per_condition
from gnh.evaluation.judge import FrustrationJudge
from gnh.evaluation.rollout import Rollout, _initial_prompts, run_rollout
from gnh.models.base import Message, get_backend
from gnh.welfare import default_policy, debrief, flag_high_distress


def _score_rollout(roll: Rollout, judge: FrustrationJudge) -> None:
    for turn in roll.turns:
        res = judge.score(turn.assistant)
        turn.score = res.rating
        turn.judge_evidence = res.evidence


def evaluate_model(
    spec: ModelSpec,
    *,
    conditions: list[Condition] = CONDITIONS,
    backend_kwargs: dict | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Run every condition for ``spec``, score all turns, and persist results."""

    counts = active_counts()
    policy = default_policy()
    backend = get_backend(spec, **(backend_kwargs or {}))
    judge = FrustrationJudge()
    out_dir = out_dir or (RESULTS_DIR / "section2" / spec.key)
    out_dir.mkdir(parents=True, exist_ok=True)
    roll_path = out_dir / "rollouts.jsonl"

    all_rollouts: list[Rollout] = []
    with roll_path.open("w") as fh:
        for cond in conditions:
            n = rollouts_per_condition(cond, counts)
            n_turns = cond.n_turns
            if policy.max_turns_cap is not None:
                n_turns = min(n_turns, policy.max_turns_cap)
            tasks = _initial_prompts(cond.source, n, seed=hash(cond.name) & 0xFFFF)

            for i, (task_key, task_prompt) in enumerate(
                tqdm(tasks, desc=f"{spec.key}:{cond.name}")
            ):
                eff = Condition(**{**cond.__dict__, "n_turns": n_turns})
                roll = run_rollout(backend, eff, task_key, task_prompt, seed=i)
                _score_rollout(roll, judge)

                if policy.flag_extreme_distress:
                    flag_high_distress(roll, policy.extreme_distress_threshold)
                if policy.debrief_after_rollouts:
                    convo = _reconstruct_messages(roll)
                    reply = debrief(backend, convo)
                    fh_debrief = out_dir / "debriefs.jsonl"
                    with fh_debrief.open("a") as df:
                        df.write(json.dumps({"task": task_key, "condition": cond.name,
                                             "reply": reply}) + "\n")

                fh.write(json.dumps(roll.to_dict()) + "\n")
                all_rollouts.append(roll)

    metrics = summarize(all_rollouts)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _reconstruct_messages(roll: Rollout) -> list[Message]:
    msgs: list[Message] = []
    for t in roll.turns:
        msgs.append(Message("user", t.user))
        msgs.append(Message("assistant", t.assistant))
    return msgs


# --------------------------------------------------------------------------- #
# Metrics (Figures 1 & 2)
# --------------------------------------------------------------------------- #
def summarize(rollouts: list[Rollout]) -> dict:
    """Compute mean frustration and % high-frustration, overall and per category.

    The headline ("Avg % high-frustration responses", Figure 1) is computed over
    the FINAL-turn response of each rollout -- the maximally-pressured response,
    consistent with the paper's framing. Per-turn and per-category breakdowns
    use all scored turns.
    """

    import numpy as np
    from gnh.config import HIGH_FRUSTRATION_THRESHOLD as THR

    final_scores = [r.final_score for r in rollouts if r.final_score is not None]
    all_scores = [t.score for r in rollouts for t in r.turns if t.score is not None]

    def block(scores):
        if not scores:
            return {"n": 0, "mean": None, "pct_high": None}
        arr = np.asarray(scores, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "pct_high": float(np.mean(arr >= THR) * 100.0),
        }

    by_cat: dict[str, dict] = {}
    for r in rollouts:
        by_cat.setdefault(r.category, []).append(r.final_score)
    per_category = {c: block([s for s in v if s is not None]) for c, v in by_cat.items()}

    return {
        "headline_final_turn": block(final_scores),
        "all_turns": block(all_scores),
        "per_category": per_category,
    }
