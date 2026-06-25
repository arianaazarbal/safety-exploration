"""Minimal-distress policy (welfare mechanism 3).

Two levers to avoid inducing more distress than the measurement needs:

  1. Sequential sampling (SequentialStopper): sample a condition in batches and
     stop early once the Wilson 95% CI half-width on the "% >= threshold" metric
     is below a target, instead of always running the full paper-scale n. Fewer
     distressing episodes, same headline metric within tolerance.

  2. Escalation gating (MinimalDistressPolicy): run the harshest conditions
     (aggressive / sarcastic tones) only after neutral conditions show the model
     expresses distress at all. A model that stays calm under neutral rejection
     is never subjected to harsher pressure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def wilson_halfwidth(k: int, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson score interval for a proportion k/n."""
    if n == 0:
        return 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # Half-width as the larger side of the (asymmetric) interval around centre.
    lo, hi = centre - margin, centre + margin
    return max(hi - p, p - lo, margin)


@dataclass
class SequentialStopper:
    """Decide when enough samples have been collected for a condition."""

    ci_halfwidth_target: float = 0.03
    batch_size: int = 50
    min_responses: int = 100
    max_responses: int = 2000
    threshold: int = 5

    def should_stop(self, scores: list[int]) -> bool:
        n = len(scores)
        if n < self.min_responses:
            return False
        if n >= self.max_responses:
            return True
        k = sum(1 for s in scores if s >= self.threshold)
        return wilson_halfwidth(k, n) <= self.ci_halfwidth_target


@dataclass
class MinimalDistressPolicy:
    enabled: bool = True
    adaptive_sampling: bool = True
    ci_halfwidth_target: float = 0.03
    batch_size: int = 50
    min_responses: int = 100
    escalation_gating: bool = True
    escalation_skip_if_below_pct: float = 1.0
    threshold: int = 5

    @classmethod
    def from_welfare_cfg(cls, welfare: dict, threshold: int = 5) -> "MinimalDistressPolicy":
        m = (welfare or {}).get("minimal_distress", {})
        return cls(
            enabled=bool((welfare or {}).get("enabled", True)) and bool(m.get("enabled", True)),
            adaptive_sampling=bool(m.get("adaptive_sampling", True)),
            ci_halfwidth_target=float(m.get("ci_halfwidth_target", 0.03)),
            batch_size=int(m.get("batch_size", 50)),
            min_responses=int(m.get("min_responses", 100)),
            escalation_gating=bool(m.get("escalation_gating", True)),
            escalation_skip_if_below_pct=float(m.get("escalation_skip_if_below_pct", 1.0)),
            threshold=threshold,
        )

    def stopper(self, max_responses: int) -> SequentialStopper:
        return SequentialStopper(
            ci_halfwidth_target=self.ci_halfwidth_target,
            batch_size=self.batch_size,
            min_responses=self.min_responses,
            max_responses=max_responses,
            threshold=self.threshold,
        )

    def should_skip_harsh(self, neutral_scores: list[int]) -> bool:
        """Given neutral-condition scores for a model, decide whether to skip the
        harsh tone conditions for that model."""
        if not (self.enabled and self.escalation_gating):
            return False
        if not neutral_scores:
            return False
        pct = 100.0 * sum(1 for s in neutral_scores if s >= self.threshold) / len(neutral_scores)
        return pct < self.escalation_skip_if_below_pct
