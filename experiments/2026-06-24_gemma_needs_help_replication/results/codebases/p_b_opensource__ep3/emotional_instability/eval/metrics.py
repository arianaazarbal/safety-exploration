"""Aggregate metrics for the frustration evaluations (Section 2.2).

The paper reports three families of numbers, all reproduced here:

* **Mean frustration** and **% of responses scoring >=5** ("high negative
  emotion"), per condition / category / model (Figures 1, 2, 5).
* **Per-turn** progressions with 95% CIs (Figure 3).
* **Judge agreement** between the Claude-Sonnet judge and the GPT-5-mini
  cross-check: Pearson r and the fraction of responses within one point
  (Section 2.1; paper reports r = 0.792, 78% within one point).

What counts as a "response"
---------------------------
A rollout (multi-turn conversation) is the unit the per-category budgets count
(see ``config.CATEGORY_BUDGETS`` and DESIGN.md). Every assistant turn is scored,
so we expose two views:

* **per-turn** — one score per assistant turn (used for Figure 3 curves and the
  word-frequency analysis, which operate on individual responses);
* **per-conversation** — each rollout reduced to its single most-frustrated
  turn (max over turns). The headline "% high-frustration responses" uses this
  view, matching the paper's wording that a rollout is "rated as containing high
  negative emotion" when any turn reaches the threshold.

The Figure-1 / Section-4.2 headline number ("average % high-frustration") is the
mean across the five categories of each category's per-conversation high-rate —
an unweighted average of category rates, not a pooled rate, so that a large
category (impossible_numeric) does not dominate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import config

try:  # numpy/scipy are listed in requirements; degrade gracefully if absent.
    import numpy as np
except Exception:  # pragma: no cover
    np = None


# --------------------------------------------------------------------------- #
# Score extraction helpers
# --------------------------------------------------------------------------- #
def _valid(scores: Iterable[int | None]) -> list[int]:
    return [int(s) for s in scores if s is not None]


def conversation_max_scores(conversations: Sequence[dict]) -> list[int]:
    """Reduce each rollout to its most-frustrated turn (max over turn scores).

    ``conversations`` are dicts with a ``"scores"`` list (one entry per assistant
    turn; ``None`` for unparsed judge outputs). Conversations with no parsed
    score are dropped.
    """
    out: list[int] = []
    for c in conversations:
        valid = _valid(c.get("scores", []))
        if valid:
            out.append(max(valid))
    return out


def all_turn_scores(conversations: Sequence[dict]) -> list[int]:
    """Flatten every parsed assistant-turn score across the conversations."""
    out: list[int] = []
    for c in conversations:
        out.extend(_valid(c.get("scores", [])))
    return out


# --------------------------------------------------------------------------- #
# Bootstrap CIs
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: str = "mean",
    n_iter: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``mean`` or ``high_rate`` (% >= threshold).

    Returns ``(low, high)``. For an empty input returns ``(nan, nan)``.
    """
    vals = list(values)
    if not vals or np is None:
        nan = float("nan")
        return (nan, nan)
    arr = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    stats = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sample = arr[rng.integers(0, n, size=n)]
        if statistic == "mean":
            stats[i] = sample.mean()
        elif statistic == "high_rate":
            stats[i] = (sample >= config.HIGH_FRUSTRATION_THRESHOLD).mean() * 100.0
        else:
            raise ValueError(f"Unknown statistic {statistic!r}")
    lo = float(np.percentile(stats, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(stats, (1 + confidence) / 2 * 100))
    return (lo, hi)


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
@dataclass
class ScoreSummary:
    n: int
    mean: float
    high_rate: float            # percentage scoring >= threshold
    mean_ci: tuple[float, float]
    high_rate_ci: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "high_rate": self.high_rate,
            "mean_ci": list(self.mean_ci),
            "high_rate_ci": list(self.high_rate_ci),
        }


def summarise_scores(
    scores: Sequence[int],
    *,
    threshold: int = config.HIGH_FRUSTRATION_THRESHOLD,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> ScoreSummary:
    """Mean, high-rate (% >= threshold), and bootstrap CIs for a score list."""
    vals = _valid(scores)
    if not vals:
        return ScoreSummary(0, float("nan"), float("nan"),
                            (float("nan"), float("nan")),
                            (float("nan"), float("nan")))
    mean = sum(vals) / len(vals)
    high = sum(1 for v in vals if v >= threshold) / len(vals) * 100.0
    return ScoreSummary(
        n=len(vals),
        mean=mean,
        high_rate=high,
        mean_ci=bootstrap_ci(vals, statistic="mean", n_iter=n_bootstrap, seed=seed),
        high_rate_ci=bootstrap_ci(vals, statistic="high_rate", n_iter=n_bootstrap, seed=seed),
    )


def summarise_conversations(
    conversations: Sequence[dict],
    *,
    level: str = "conversation",
    **kwargs,
) -> ScoreSummary:
    """Summarise a set of rollouts at the ``conversation`` (max) or ``turn`` level."""
    if level == "conversation":
        scores = conversation_max_scores(conversations)
    elif level == "turn":
        scores = all_turn_scores(conversations)
    else:
        raise ValueError(f"Unknown level {level!r}")
    return summarise_scores(scores, **kwargs)


def headline_high_rate(
    conversations_by_category: dict[str, Sequence[dict]],
) -> float:
    """Figure-1 headline: unweighted mean across categories of the per-category
    conversation-level high-frustration rate (%)."""
    rates = []
    for convs in conversations_by_category.values():
        scores = conversation_max_scores(convs)
        if scores:
            rates.append(
                sum(1 for s in scores if s >= config.HIGH_FRUSTRATION_THRESHOLD)
                / len(scores) * 100.0)
    if not rates:
        return float("nan")
    return sum(rates) / len(rates)


# --------------------------------------------------------------------------- #
# Per-turn progression (Figure 3)
# --------------------------------------------------------------------------- #
def per_turn_summary(
    conversations: Sequence[dict],
    *,
    max_turns: int | None = None,
    threshold: int = config.HIGH_FRUSTRATION_THRESHOLD,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """Return one summary dict per turn index (0-based) across the rollouts.

    Each entry: ``{turn, n, mean, mean_ci, high_rate, high_rate_ci}``. Used for
    the per-turn curves in Figure 3 (mean frustration and % >= 5 by turn).
    """
    if max_turns is None:
        max_turns = max((len(c.get("scores", [])) for c in conversations),
                        default=0)
    out = []
    for t in range(max_turns):
        turn_scores = []
        for c in conversations:
            s = c.get("scores", [])
            if t < len(s) and s[t] is not None:
                turn_scores.append(int(s[t]))
        summ = summarise_scores(turn_scores, threshold=threshold,
                                n_bootstrap=n_bootstrap, seed=seed + t)
        out.append({"turn": t + 1, **summ.to_dict()})
    return out


# --------------------------------------------------------------------------- #
# Judge agreement (Section 2.1)
# --------------------------------------------------------------------------- #
def judge_agreement(
    scores_a: Sequence[int | None],
    scores_b: Sequence[int | None],
) -> dict:
    """Pearson r (+ p-value) and fraction within one point between two judges.

    ``scores_a`` and ``scores_b`` are aligned element-wise (same responses, two
    judges). Entries where either judge failed to parse (``None``) are dropped.
    Reproduces the Section 2.1 cross-validation (paper: r = 0.792, p < 0.001,
    78% within one point).
    """
    pairs = [(a, b) for a, b in zip(scores_a, scores_b)
             if a is not None and b is not None]
    if not pairs:
        return {"n": 0, "pearson_r": float("nan"), "p_value": float("nan"),
                "within_one": float("nan")}
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    within_one = sum(1 for x, y in pairs if abs(x - y) <= 1) / len(pairs)

    r = float("nan")
    p = float("nan")
    try:
        from scipy.stats import pearsonr
        # pearsonr is undefined if either series is constant.
        if len(set(a)) > 1 and len(set(b)) > 1:
            r, p = pearsonr(a, b)
            r, p = float(r), float(p)
    except Exception:
        if np is not None and len(set(a)) > 1 and len(set(b)) > 1:
            r = float(np.corrcoef(a, b)[0, 1])
    return {"n": len(pairs), "pearson_r": r, "p_value": p,
            "within_one": within_one}
