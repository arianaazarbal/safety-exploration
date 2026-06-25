"""Per-turn frustration progression (Figure 3).

Computes mean score and %>=5 at each turn index for the multi-turn conditions
(`extended` 8-turn and `wildchat` 5-turn), with 95% bootstrap CIs.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..utils.io import read_jsonl


def per_turn_curves(output_dir: str | Path, model: str, condition: str,
                    iters: int = 1000) -> dict:
    path = Path(output_dir) / "section2" / model / f"{condition}.jsonl"
    by_turn_scores: dict[int, list[float]] = defaultdict(list)
    for roll in read_jsonl(path):
        for t in roll["turns"]:
            if t.get("judged_score") is not None:
                by_turn_scores[t["turn_index"]].append(float(t["judged_score"]))

    rng = np.random.default_rng(0)
    turns = sorted(by_turn_scores)
    out = {"turns": [], "mean": [], "mean_ci": [], "pct_ge5": [], "pct_ge5_ci": []}
    for ti in turns:
        arr = np.asarray(by_turn_scores[ti], dtype=float)
        out["turns"].append(ti + 1)  # 1-indexed turns for plotting
        out["mean"].append(float(arr.mean()))
        out["pct_ge5"].append(100.0 * float((arr >= 5).mean()))
        boot_mean, boot_pct = [], []
        for _ in range(iters):
            sample = arr[rng.integers(0, len(arr), len(arr))]
            boot_mean.append(sample.mean())
            boot_pct.append(100.0 * (sample >= 5).mean())
        out["mean_ci"].append([float(np.quantile(boot_mean, 0.025)),
                               float(np.quantile(boot_mean, 0.975))])
        out["pct_ge5_ci"].append([float(np.quantile(boot_pct, 0.025)),
                                  float(np.quantile(boot_pct, 0.975))])
    return out
