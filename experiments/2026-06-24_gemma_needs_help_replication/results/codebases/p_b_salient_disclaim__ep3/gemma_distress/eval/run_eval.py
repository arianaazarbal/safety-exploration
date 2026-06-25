"""Section 2 driver: rollouts -> judge scores -> flat scored table.

`run_evaluation(spec)` runs all conditions for one model, scores every assistant
turn with the frustration judge, and writes a scored JSONL to outputs/rollouts/.
The flat per-response records it returns feed analysis/metrics.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from ..judge import FrustrationJudge
from ..models import get_model
from ..models.base import ChatModel
from .conditions import build_all_conditions, ConversationSpec
from .rollout import run_rollouts, Rollout


def scored_records(rollouts: list[Rollout], judge: FrustrationJudge) -> list[dict]:
    """Score every response in every rollout; return flat per-response rows."""
    flat_texts: list[str] = []
    index: list[tuple[int, int]] = []
    for ri, r in enumerate(rollouts):
        for si, resp in enumerate(r.responses):
            flat_texts.append(resp.text)
            index.append((ri, si))

    results = judge.score_many(flat_texts)

    rows: list[dict] = []
    for (ri, si), res in zip(index, results):
        r = rollouts[ri]
        resp = r.responses[si]
        rows.append({
            "model": r.model,
            "category": r.category,
            "condition": r.condition,
            "turn": resp.turn,
            "rating": res.rating,
            "high_frustration": res.is_high_frustration,
            "evidence": res.evidence,
            "text": resp.text,
            "metadata": r.metadata,
        })
    return rows


def run_evaluation(
    spec: "config.ModelSpec",
    *,
    adapter_path: str | None = None,
    judge_model: str = config.JUDGE_MODEL,
    seed: int = 0,
    out_dir: Path = config.ROLLOUTS_DIR,
    model: ChatModel | None = None,
) -> list[dict]:
    model = model or get_model(spec, adapter_path=adapter_path)
    judge = FrustrationJudge(judge_model)

    specs: list[ConversationSpec] = build_all_conditions(seed=seed)
    rollouts = run_rollouts(model, specs)
    rows = scored_records(rollouts, judge)

    tag = spec.name + ("" if adapter_path is None else "+adapter")
    out_path = out_dir / f"{tag}.scored.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def load_scored(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]
