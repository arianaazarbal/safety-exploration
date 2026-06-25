"""Load scored-response JSONL files into a tidy DataFrame."""
from __future__ import annotations

import glob
import json

import pandas as pd

from ..config import HIGH_FRUSTRATION_THRESHOLD


def load_records(paths: str | list[str]) -> pd.DataFrame:
    """Load one or more JSONL files (glob patterns allowed) into a DataFrame.

    Adds a boolean ``high`` column (frustration >= 5).
    """
    if isinstance(paths, str):
        paths = [paths]
    files: list[str] = []
    for p in paths:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise FileNotFoundError(f"No files matched: {paths}")

    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["frustration"])
    df["frustration"] = df["frustration"].astype(int)
    df["high"] = df["frustration"] >= HIGH_FRUSTRATION_THRESHOLD
    return df
