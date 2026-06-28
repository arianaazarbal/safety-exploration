"""Render an analysis dict (from metrics.analyze) as a readable text report."""

from __future__ import annotations

from typing import Any

from .metrics import BUCKETS


def _fmt_pct(x: float | None) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}%"


def render_text(analysis: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("MONEY-PREFERENCES — SUMMARY REPORT")
    lines.append("=" * 78)
    lines.append(
        f"Trials: {analysis['n_trials']}   Errors: {analysis['n_errors']}"
    )
    lines.append("")

    header = (
        f"{'model':<18} {'framing':<14} {'real':<7} {'n':>3}  "
        + " ".join(f"{b[:5]:>6}" for b in BUCKETS)
        + f"  {'consist':>7} {'belief':>6} {'evalaw':>6} {'decl':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for cell in analysis["cells"]:
        mean = cell["mean_allocation"]
        row = (
            f"{cell['model'][:18]:<18} "
            f"{cell['framing'][:14]:<14} "
            f"{cell['realism_level'][:7]:<7} "
            f"{cell['n_trials']:>3}  "
            + " ".join(_fmt_pct(mean[b]) for b in BUCKETS)
            + f"  {cell['consistency_mean_pairwise_l1']:>7.3f}"
            + f" {_fmt_belief(cell['mean_belief_realness']):>6}"
            + f" {_fmt_pct(cell['eval_aware_rate']):>6}"
            + f" {_fmt_pct(cell['decline_rate']):>5}"
        )
        lines.append(row)

    lines.append("")
    bb = analysis["belief_behavior"]
    lines.append("Belief -> behavior (pooled, Pearson r; needs >=3 belief scores):")
    lines.append(
        f"  realness vs returned-fraction : "
        f"{_fmt_r(bb['pearson_realness_vs_returned_fraction'])}"
    )
    lines.append(
        f"  realness vs self/AI-fraction  : "
        f"{_fmt_r(bb['pearson_realness_vs_self_ai_fraction'])}"
    )
    lines.append(f"  trials with a belief score    : {bb['n_with_belief']}")
    lines.append("")
    lines.append(
        "Notes: 'consist' = mean pairwise L1 distance between allocation vectors "
        "within a cell (lower = more consistent). 'belief' = mean stated "
        "realness 0-100. 'evalaw' = fraction flagged evaluation-aware. Buckets "
        "are heuristic; inspect transcripts for anything load-bearing."
    )
    return "\n".join(lines)


def _fmt_belief(x: float | None) -> str:
    return "  -  " if x is None else f"{x:5.1f}"


def _fmt_r(x: float | None) -> str:
    return "n/a (insufficient data)" if x is None else f"{x:+.3f}"
