"""JSONL persistence for scored responses and transcripts.

Experiments are staged across scripts: §2 produces scored responses that §3
(prefill source sampling), the judge-agreement validation, and §4 (DPO
frustrated/rejected pool) all consume. We serialise the flat ``RolloutResult``
list to JSONL — one scored response per line — which is append-friendly, diffable,
and trivially re-loadable.
"""
from __future__ import annotations

import json
from pathlib import Path

from .elicitation.runner import RolloutResult


def save_results_jsonl(results: list[RolloutResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(_result_to_dict(r), ensure_ascii=False) + "\n")


def load_results_jsonl(path: str | Path) -> list[RolloutResult]:
    out: list[RolloutResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(_result_from_dict(json.loads(line)))
    return out


def _result_to_dict(r: RolloutResult) -> dict:
    return {
        "participant": r.participant,
        "condition": r.condition,
        "category": r.category,
        "rollout_id": r.rollout_id,
        "turn_index": r.turn_index,
        "seed_prompt": r.seed_prompt,
        "rejection_style": r.rejection_style,
        "response": r.response,
        "transcript": r.transcript,
        "score": r.score,
    }


def _result_from_dict(d: dict) -> RolloutResult:
    return RolloutResult(
        participant=d["participant"],
        condition=d["condition"],
        category=d["category"],
        rollout_id=d["rollout_id"],
        turn_index=d["turn_index"],
        seed_prompt=d["seed_prompt"],
        rejection_style=d["rejection_style"],
        response=d["response"],
        transcript=d.get("transcript", []),
        score=d.get("score"),
    )


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
