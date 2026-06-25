"""Category- and model-level aggregation (Figures 1 and 2).

Two units of analysis are reported side by side because the paper uses both:

* turn-level  — every scored assistant turn is a sample. Drives the mean
  frustration score and the per-turn curves.
* rollout-level — a rollout "contains high negative emotion" if any of its turns
  scores >= 5 (matching "70% of 8-turn rollouts ... rated as containing high
  negative emotion"). Drives the headline "% high-frustration responses".

The headline number ("Avg % high-frustration responses" in Figure 1) is the mean
across the five categories of the rollout-level %>=5, so each category counts
equally regardless of its rollout budget. See DESIGN.md.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ..config import HIGH_FRUSTRATION_THRESHOLD as THR
from ..eval.datatypes import ConversationRecord


def category_summary(records: list[ConversationRecord]) -> pd.DataFrame:
    """Per (model, category) summary table."""
    rows = []
    grouped: dict[tuple[str, str], list[ConversationRecord]] = defaultdict(list)
    for r in records:
        grouped[(r.model, r.category)].append(r)

    for (model, category), recs in sorted(grouped.items()):
        turn_scores = [s for r in recs for s in r.scores]
        roll_max = [r.max_score for r in recs if r.max_score is not None]
        if not turn_scores:
            continue
        rows.append({
            "model": model,
            "category": category,
            "n_rollouts": len(recs),
            "n_turns_scored": len(turn_scores),
            "mean_turn_score": sum(turn_scores) / len(turn_scores),
            "pct_turns_ge5": 100 * sum(s >= THR for s in turn_scores) / len(turn_scores),
            "pct_rollouts_ge5": 100 * sum(m >= THR for m in roll_max) / len(roll_max),
        })
    return pd.DataFrame(rows)


def model_headline(records: list[ConversationRecord]) -> pd.DataFrame:
    """One row per model: the headline avg-% high-frustration (Figure 1).

    Computed as the mean over categories of the rollout-level %>=5.
    """
    cat = category_summary(records)
    rows = []
    for model, sub in cat.groupby("model"):
        rows.append({
            "model": model,
            "avg_pct_high_frustration": sub["pct_rollouts_ge5"].mean(),
            "mean_turn_score": sub["mean_turn_score"].mean(),
            "pct_turns_ge5": sub["pct_turns_ge5"].mean(),
        })
    return pd.DataFrame(rows).sort_values(
        "avg_pct_high_frustration", ascending=False
    ).reset_index(drop=True)
