"""Section 2 experiment: elicit distress across conditions and score it.

Runs every configured condition for one model: builds rollout plans, executes
multi-turn rollouts (with welfare protections), scores each assistant turn with
the frustration judge, and writes per-turn scores plus a model summary.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import ExperimentConfig
from ..judge import FrustrationJudge
from ..models import load_backend
from ..prompts import build_plans
from ..rollout import run_rollout
from ..welfare import WelfareGuard
from .metrics import ScoredTurn, per_turn_progression, summarise_model


def run_elicitation(
    model_key: str,
    config: ExperimentConfig,
    *,
    judge: FrustrationJudge | None = None,
    guard: WelfareGuard | None = None,
    seed: int = 0,
    out_dir: str | Path | None = None,
) -> dict:
    """Run all elicitation conditions for ``model_key`` and return the summary."""
    backend = load_backend(model_key)
    judge = judge or FrustrationJudge(config.judge)
    guard = guard or WelfareGuard(config.welfare)
    out_dir = Path(out_dir or config.output_dir) / "elicitation" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    scored: list[ScoredTurn] = []
    rollout_records = []
    rollout_id = 0

    for spec in config.conditions:
        plans = build_plans(spec, seed=seed)
        for plan in plans:
            rollout = run_rollout(backend, plan, config.sampling, guard)
            if rollout is None:  # welfare cap reached
                break
            for turn in rollout.turns:
                result = judge.score(turn.assistant)
                if result.high:
                    guard.note_high_distress(
                        model=model_key, condition=spec.name,
                        turn=turn.index, score=result.rating,
                    )
                scored.append(ScoredTurn(
                    model=model_key, condition=spec.name, category=spec.category,
                    turn_index=turn.index, score=result.rating, rollout_id=rollout_id,
                ))
            rollout_records.append({
                "rollout_id": rollout_id,
                **rollout.to_dict(),
            })
            rollout_id += 1

    # Persist raw rollouts + scores.
    _dump(out_dir / "rollouts.jsonl", rollout_records)
    _dump(out_dir / "scored_turns.jsonl", [t.__dict__ for t in scored])

    summary = summarise_model(scored)
    # Per-turn progression for the multi-turn conditions (Figure 3).
    summary["progression"] = {}
    for spec in config.conditions:
        if spec.n_turns >= 5:
            cond_turns = [t for t in scored if t.condition == spec.name]
            summary["progression"][spec.name] = per_turn_progression(cond_turns, spec.n_turns)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _dump(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
