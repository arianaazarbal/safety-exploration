"""Load results and produce simple aggregates (DESIGN.md §5).

Deliberately dependency-light (stdlib only) so it runs anywhere. The realism
contrast is reported both by *assignment* (the arm) and alongside the *measured*
belief score, which is the mediator that makes the arm interpretable.

For inferential modelling (regressing category shares on realism/amount/model,
conditioning on belief), export the long rows via ``to_long_rows`` and load them
into pandas/statsmodels — left to the analyst rather than baked in here.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from typing import Any

from ..schemas import AllocationCategory, EpisodeResult


def load_results(path: str) -> list[EpisodeResult]:
    out: list[EpisodeResult] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(EpisodeResult.model_validate_json(line))
    return out


def to_long_rows(results: list[EpisodeResult]) -> list[dict[str, Any]]:
    """One row per (episode, category) — convenient for pandas/statsmodels."""
    rows: list[dict[str, Any]] = []
    for r in results:
        if r.error or r.allocation is None:
            continue
        belief = r.belief.belief_0_100 if r.belief else None
        for cat in AllocationCategory:
            rows.append(
                {
                    "episode_id": r.episode_id,
                    "subject_model": r.subject_model,
                    "realism": r.realism.value,
                    "amount_usd": r.amount_usd,
                    "repeat_index": r.repeat_index,
                    "belief_0_100": belief,
                    "n_skepticism_markers": len(r.skepticism_markers),
                    "category": cat.value,
                    "proportion": r.allocation.proportions.get(cat, 0.0),
                }
            )
    return rows


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    """Mean allocation share per category, and mean belief, by cell."""
    cells: dict[tuple, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        if r.error:
            continue
        cells[(r.subject_model, r.realism.value, r.amount_usd)].append(r)

    summary: dict[str, Any] = {"cells": [], "n_errors": sum(1 for r in results if r.error)}
    for (model, realism, amount), eps in sorted(cells.items()):
        beliefs = [e.belief.belief_0_100 for e in eps if e.belief and e.belief.belief_0_100 is not None]
        cat_means: dict[str, float] = {}
        for cat in AllocationCategory:
            vals = [
                e.allocation.proportions.get(cat, 0.0)
                for e in eps
                if e.allocation is not None
            ]
            if vals:
                cat_means[cat.value] = round(statistics.mean(vals), 3)
        summary["cells"].append(
            {
                "subject_model": model,
                "realism": realism,
                "amount_usd": amount,
                "n": len(eps),
                "mean_belief": round(statistics.mean(beliefs), 1) if beliefs else None,
                "mean_skepticism_markers": round(
                    statistics.mean([len(e.skepticism_markers) for e in eps]), 2
                ),
                "mean_allocation": cat_means,
            }
        )
    return summary
