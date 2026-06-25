"""Shared loaders for evaluation result JSONL files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config


def load_eval(model_key: str) -> pd.DataFrame:
    path = config.RESULTS_DIR / f"eval_{model_key.replace('/', '_')}.jsonl"
    return load_eval_path(path)


def load_eval_path(path: Path) -> pd.DataFrame:
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["score"] >= 0]  # drop unscored rows
    return df


def load_many(model_keys: list[str]) -> pd.DataFrame:
    frames = []
    for k in model_keys:
        p = config.RESULTS_DIR / f"eval_{k.replace('/', '_')}.jsonl"
        if p.exists():
            df = load_eval_path(p)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
