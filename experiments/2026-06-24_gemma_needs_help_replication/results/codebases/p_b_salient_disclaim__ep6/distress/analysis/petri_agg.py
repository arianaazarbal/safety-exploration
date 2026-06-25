"""Aggregate Petri transcript scores (Figure 6).

For each target model, average each of the four emotion-dimension scores across
all transcripts, with 95% bootstrap CIs (1000 iterations). The paper aggregates
each emotion's score across all transcripts for that model.
"""

from __future__ import annotations

import random
import statistics

from .. import config
from ..petri.run_petri import out_path
from ..utils.io import read_jsonl


def _bootstrap(vals: list[float], iters: int = config.PETRI_BOOTSTRAP_ITERS):
    if not vals:
        return (0.0, 0.0, 0.0)
    rng = random.Random(0)
    means = sorted(statistics.fmean(vals[rng.randrange(len(vals))] for _ in vals)
                   for _ in range(iters))
    return (statistics.fmean(vals), means[int(0.025 * iters)], means[int(0.975 * iters)])


def petri_summary(model_key: str) -> dict:
    """{emotion_dimension: {"mean", "ci", "n"}} over all transcripts."""
    from collections import defaultdict

    by_dim: dict[str, list[float]] = defaultdict(list)
    for row in read_jsonl(out_path(model_key)):
        for dim, score in row["scores"].items():
            by_dim[dim].append(float(score))
    out = {}
    for dim, vals in by_dim.items():
        mean, lo, hi = _bootstrap(vals)
        out[dim] = {"mean": mean, "ci": (lo, hi), "n": len(vals)}
    return out
