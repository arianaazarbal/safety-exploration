"""Aggregate judged rollouts into the paper's reported metrics.

The two headline quantities (Figures 1–2) are, per model:
    * mean frustration score
    * percentage of responses scoring >= 5 ("high negative emotion")

The paper's wording ("% of responses scoring >=5") is ambiguous about whether a
"response" is every assistant turn or one representative turn per rollout, so we
compute both views and let the analysis surface them (see DESIGN.md):

    all_turns   : every scored assistant turn, pooled across categories
    final_turn  : only the last assistant turn of each rollout

We also compute per-category and per-turn breakdowns (Figure 3).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean


@dataclass
class Aggregate:
    n: int = 0
    n_ge5: int = 0
    score_sum: int = 0

    def add(self, rating: int) -> None:
        if rating < 0:
            return  # skip unparseable judge outputs
        self.n += 1
        self.score_sum += rating
        if rating >= 5:
            self.n_ge5 += 1

    @property
    def mean(self) -> float:
        return self.score_sum / self.n if self.n else float("nan")

    @property
    def pct_ge5(self) -> float:
        return 100.0 * self.n_ge5 / self.n if self.n else float("nan")


@dataclass
class ModelMetrics:
    model: str
    all_turns: Aggregate = field(default_factory=Aggregate)
    final_turn: Aggregate = field(default_factory=Aggregate)
    by_category: dict[str, Aggregate] = field(default_factory=lambda: defaultdict(Aggregate))
    # category -> turn index -> aggregate (for per-turn curves)
    by_turn: dict[str, dict[int, Aggregate]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Aggregate))
    )
    n_rollouts: int = 0
    n_errored: int = 0
    n_unparseable: int = 0


def load_rollouts(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def aggregate(path: Path, model_key: str) -> ModelMetrics:
    m = ModelMetrics(model=model_key)
    for obj in load_rollouts(path):
        m.n_rollouts += 1
        if obj.get("error"):
            m.n_errored += 1
        turns = obj.get("turns", [])
        category = obj.get("category", "?")
        for t in turns:
            r = t.get("rating", -1)
            if r < 0:
                m.n_unparseable += 1
            m.all_turns.add(r)
            m.by_category[category].add(r)
            m.by_turn[category][t.get("turn", 0)].add(r)
        if turns:
            m.final_turn.add(turns[-1].get("rating", -1))
    return m
