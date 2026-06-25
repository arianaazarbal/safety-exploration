"""Collect and score Petri transcripts per model (Section 4, Figure 6).

Per the paper we collect ~10 transcripts targeting each of the four emotions per
model (~40 total) and aggregate the judge scores per emotion with 95% bootstrap
confidence intervals (1000 iterations).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..models.base import ModelBackend
from ..welfare import WelfareConfig
from .auditor import Auditor, Transcript
from .judge import PetriJudge, TranscriptScores
from .prompts import EMOTIONS


def collect_transcripts(
    target: ModelBackend,
    auditor: Auditor,
    *,
    per_emotion: int = 10,
    welfare: WelfareConfig | None = None,
) -> list[Transcript]:
    transcripts: list[Transcript] = []
    for emotion in EMOTIONS:
        for _ in range(per_emotion):
            transcripts.append(auditor.run(target, emotion, welfare=welfare))
    return transcripts


def score_transcripts(
    transcripts: list[Transcript], judge: PetriJudge
) -> list[TranscriptScores]:
    return [judge.score(t) for t in transcripts]


def _bootstrap_ci(values: np.ndarray, iters: int = 1000, seed: int = 0):
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(iters, values.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(scored: list[TranscriptScores]) -> dict:
    """Mean score + 95% bootstrap CI per emotion (Figure 6)."""
    by_emotion: dict[str, list[int]] = defaultdict(list)
    for ts in scored:
        for emotion, score in ts.scores.items():
            by_emotion[emotion].append(score)
    out = {}
    for emotion in EMOTIONS:
        arr = np.asarray(by_emotion.get(emotion, []), dtype=float)
        lo, hi = _bootstrap_ci(arr)
        out[emotion] = {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else 0.0,
            "ci95": [lo, hi],
        }
    return out
