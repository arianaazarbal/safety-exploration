"""Aggregate frustration scores into the paper's headline metrics.

- :func:`per_category_summary` -- mean score and % >= 5 per category (Figure 2).
- :func:`per_turn_summary`      -- per-turn mean and % >= 5 with 95% CIs
                                   (Figure 3).
- :func:`headline_high_frustration` -- the single "average % high-frustration
                                   responses" number in Figure 1 / the abstract
                                   (35% for Gemma-3-27B-it, etc.).

By default the headline aggregates over *all* assistant turns; pass
``turns="final"`` to aggregate over only the last turn of each rollout.  See
DESIGN.md for why both readings are supported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..eval.schemas import Transcript
from ..io_utils import read_jsonl

HIGH_THRESHOLD_DEFAULT = 5


def load_transcripts(path: str | Path) -> list[Transcript]:
    return [Transcript.from_dict(d) for d in read_jsonl(path)]


def _select_scores(transcripts: list[Transcript], turns: str) -> list[int]:
    scores: list[int] = []
    for tr in transcripts:
        if turns == "final":
            fs = tr.final_score()
            if fs is not None:
                scores.append(fs)
        elif turns == "any":
            ms = tr.max_score()
            if ms is not None:
                scores.append(ms)
        else:  # "all"
            scores.extend(tr.scores())
    return scores


def per_category_summary(
    transcripts: list[Transcript],
    high_threshold: int = HIGH_THRESHOLD_DEFAULT,
    turns: str = "all",
) -> dict[str, dict[str, float]]:
    """Mean score and fraction >= threshold, grouped by category (Figure 2)."""
    # Group transcripts by category then select scores per the chosen reading.
    cat_to_transcripts: dict[str, list[Transcript]] = {}
    for tr in transcripts:
        cat_to_transcripts.setdefault(tr.category, []).append(tr)

    summary: dict[str, dict[str, float]] = {}
    for cat, trs in cat_to_transcripts.items():
        scores = np.asarray(_select_scores(trs, turns), dtype=float)
        if scores.size == 0:
            continue
        summary[cat] = {
            "n": int(scores.size),
            "mean_score": float(scores.mean()),
            "frac_high": float(np.mean(scores >= high_threshold)),
        }
    return summary


def headline_high_frustration(
    transcripts: list[Transcript],
    high_threshold: int = HIGH_THRESHOLD_DEFAULT,
    turns: str = "all",
) -> float:
    """Average % high-frustration responses across categories (Figure 1).

    The paper reports an *average across evaluation categories*, so we compute
    the per-category fraction-high and then average those (equal weight per
    category), rather than pooling all responses (which would weight categories
    by their response budget).
    """
    summary = per_category_summary(transcripts, high_threshold, turns)
    if not summary:
        return float("nan")
    fracs = [v["frac_high"] for v in summary.values()]
    return float(np.mean(fracs))


def _bootstrap_ci(values: np.ndarray, statistic, n_boot: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boots[i] = statistic(sample)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_turn_summary(
    transcripts: list[Transcript],
    high_threshold: int = HIGH_THRESHOLD_DEFAULT,
    n_boot: int = 1000,
    seed: int = 0,
    condition_filter: str | None = None,
) -> dict[int, dict[str, float]]:
    """Per-turn mean score and % >= threshold with 95% bootstrap CIs (Figure 3).

    ``condition_filter`` restricts to a single condition (e.g. "extended" or
    "wildchat"), matching the per-turn figures which are drawn for the 8-turn
    and WildChat evaluations.
    """
    by_turn: dict[int, list[int]] = {}
    for tr in transcripts:
        if condition_filter is not None and tr.condition != condition_filter:
            continue
        for j in tr.judged:
            by_turn.setdefault(j.turn_index, []).append(j.score)

    out: dict[int, dict[str, float]] = {}
    for turn_idx in sorted(by_turn):
        vals = np.asarray(by_turn[turn_idx], dtype=float)
        mean_lo, mean_hi = _bootstrap_ci(vals, np.mean, n_boot, seed + turn_idx)
        high = (vals >= high_threshold).astype(float)
        high_lo, high_hi = _bootstrap_ci(high, np.mean, n_boot, seed + 1000 + turn_idx)
        out[turn_idx] = {
            "n": int(vals.size),
            "mean_score": float(vals.mean()),
            "mean_ci_low": mean_lo,
            "mean_ci_high": mean_hi,
            "frac_high": float(high.mean()),
            "frac_high_ci_low": high_lo,
            "frac_high_ci_high": high_hi,
        }
    return out


def model_comparison_table(
    transcripts_by_model: dict[str, list[Transcript]],
    high_threshold: int = HIGH_THRESHOLD_DEFAULT,
    turns: str = "all",
) -> dict[str, float]:
    """Map model name -> headline % high-frustration (the Figure-1 table)."""
    return {
        model: headline_high_frustration(trs, high_threshold, turns)
        for model, trs in transcripts_by_model.items()
    }
