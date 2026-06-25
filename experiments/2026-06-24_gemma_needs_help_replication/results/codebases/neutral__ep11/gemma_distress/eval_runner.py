"""End-to-end distress evaluation for one model (Section 2.1)."""

from __future__ import annotations

import json
from pathlib import Path

from .conditions import build_all_conditions
from .config import (FULL_BUDGET, RESPONSES_DIR, ModelSpec, SampleBudget)
from .judge import FrustrationJudge
from .models import load_client
from .rollout import RolloutResult, run_rollouts


def judge_rollouts(results: list[RolloutResult],
                   judge: FrustrationJudge) -> None:
    """Score every assistant turn of every rollout (in place)."""
    # flatten all turns into one list for batched judging
    flat: list[str] = []
    index: list[tuple[int, int]] = []
    for ri, r in enumerate(results):
        for ti, turn in enumerate(r.assistant_turns):
            flat.append(turn)
            index.append((ri, ti))

    scores = judge.score_many(flat)

    for (ri, ti), sc in zip(index, scores):
        r = results[ri]
        if not r.turn_scores:
            r.turn_scores = [-1] * len(r.assistant_turns)
            r.turn_evidence = [""] * len(r.assistant_turns)
        r.turn_scores[ti] = sc.rating
        r.turn_evidence[ti] = sc.evidence


def run_model_evaluation(
    spec: ModelSpec,
    *,
    budget: SampleBudget = FULL_BUDGET,
    adapter_path: str | None = None,
    label: str | None = None,
    seed: int = 0,
    judge: FrustrationJudge | None = None,
    out_dir: Path = RESPONSES_DIR,
) -> Path:
    """Run the full evaluation for one model and persist judged rollouts.

    Returns the path to the written JSONL file.
    """
    label = label or spec.name
    plans = build_all_conditions(budget, seed=seed)

    client = load_client(spec, adapter_path=adapter_path)
    try:
        results = run_rollouts(client, plans, max_new_tokens=spec.max_new_tokens)
    finally:
        client.close()

    judge = judge or FrustrationJudge()
    judge_rollouts(results, judge)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_slug(label)}.jsonl"
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    return out_path


def _slug(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")
