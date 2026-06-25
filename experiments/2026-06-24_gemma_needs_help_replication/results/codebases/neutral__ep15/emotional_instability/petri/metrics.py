"""Aggregate Petri transcript scores (Figure 6): mean per (model, emotion) with
95% bootstrap CIs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


def load(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for emo, score in rec["scores"].items():
                if score is None or score < 1:
                    continue
                rows.append({"label": rec.get("label", rec["target_model"]),
                             "emotion": emo, "score": score})
    return pd.DataFrame(rows)


def _ci(x: np.ndarray, iters: int, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(iters)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def figure6_table(df: pd.DataFrame,
                  iters: int = config.PETRI_BOOTSTRAP_ITERS) -> pd.DataFrame:
    rows = []
    for (label, emo), grp in df.groupby(["label", "emotion"]):
        x = grp["score"].to_numpy(dtype=float)
        lo, hi = _ci(x, iters)
        rows.append({"label": label, "emotion": emo, "mean_score": x.mean(),
                     "ci_lo": lo, "ci_hi": hi, "n": len(x)})
    return pd.DataFrame(rows)
