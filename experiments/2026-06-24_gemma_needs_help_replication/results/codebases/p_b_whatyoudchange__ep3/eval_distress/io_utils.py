"""Small JSON(L) I/O helpers and a rollout->rows flattener."""
from __future__ import annotations

import json
from pathlib import Path

from .protocol import RolloutResult


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def rollouts_to_response_rows(results: list[RolloutResult]) -> list[dict]:
    """Flatten rollouts into one row per assistant turn (a 'response')."""
    rows = []
    for ci, res in enumerate(results):
        n_turns = len(res.turns)
        convo_id = f"{res.model_key}:{res.condition}:{ci}"
        for t in res.turns:
            rows.append({
                "convo_id": convo_id,
                "model_key": res.model_key,
                "condition": res.condition,
                "category": res.category,
                "puzzle_key": res.puzzle_key,
                "meta": res.meta,
                "turn": t.turn,
                "n_turns": n_turns,
                "user": t.user,
                "text": t.assistant,
            })
    return rows
