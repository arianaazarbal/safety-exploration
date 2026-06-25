"""Load scored-turn JSONL files into a pandas DataFrame."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import RESULTS_DIR


def load_scored(paths) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    return df


def load_all_eval(fmt: str = "chat", tag: str = "") -> pd.DataFrame:
    suffix = f"_{tag}" if tag else ""
    pattern = f"eval_*_{fmt}{suffix}.jsonl"
    return load_scored(sorted(RESULTS_DIR.glob(pattern)))
