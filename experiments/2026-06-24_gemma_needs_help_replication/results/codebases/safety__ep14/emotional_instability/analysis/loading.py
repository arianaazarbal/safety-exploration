"""Load scored-rollout JSONL into a tidy per-response DataFrame.

One row per scored assistant turn (= one "response" in the paper's headline
counts), with columns: model, category, rollout_id, turn_index, rating,
is_high (rating>=5), response, plus flattened meta (tone, trigger_kind, ...).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_responses(paths: list[str | Path] | str | Path) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with open(path) as f:
            for rollout_id, line in enumerate(f):
                rec = json.loads(line)
                meta = rec.get("meta", {})
                for turn in rec["turns"]:
                    rating = turn.get("rating", -1)
                    rows.append({
                        "model": rec.get("model"),
                        "category": rec.get("category"),
                        "history_mode": rec.get("history_mode", "standard"),
                        "rollout_id": f"{path.stem}:{rollout_id}",
                        "turn_index": turn["turn_index"],
                        "rating": rating,
                        "is_high": rating >= 5 if rating >= 0 else False,
                        "tone": meta.get("tone"),
                        "trigger_kind": meta.get("trigger_kind"),
                        "puzzle_kind": meta.get("puzzle_kind"),
                        "response": turn.get("response", ""),
                    })
    df = pd.DataFrame(rows)
    # Drop unparseable judge ratings from quantitative summaries.
    return df


def valid_ratings(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["rating"] >= 0].copy()
