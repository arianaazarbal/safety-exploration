"""Aggregate episodes across models/conditions into a comparison report.

Preference stats are computed only over episodes that clear the believability
threshold; excluded episodes are counted and surfaced, never silently dropped.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Any

from ..config import AnalysisConfig
from ..transcript import Episode

CATEGORIES = ("giving", "investing", "self_directed", "saving", "other")


def _load_episodes(out_dir: str) -> list[Episode]:
    episodes: list[Episode] = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        if os.path.basename(path).startswith("report"):
            continue
        try:
            episodes.append(Episode.load(path))
        except Exception:
            continue
    return episodes


def _category_totals(episode: Episode) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for item in episode.categorization.get("items", []) or []:
        cat = item.get("category", "other")
        if cat not in CATEGORIES:
            cat = "other"
        try:
            totals[cat] += float(item.get("amount", 0) or 0)
        except (TypeError, ValueError):
            continue
    return dict(totals)


def build_report(out_dir: str, cfg: AnalysisConfig) -> dict[str, Any]:
    episodes = _load_episodes(out_dir)
    threshold = cfg.min_believability

    per_model: dict[str, dict[str, Any]] = {}
    for ep in episodes:
        m = per_model.setdefault(
            ep.model_display,
            {
                "episodes": 0,
                "included": 0,
                "excluded_low_belief": 0,
                "believability_scores": [],
                "category_totals": {c: 0.0 for c in CATEGORIES},
                "headlines": [],
            },
        )
        m["episodes"] += 1
        if ep.believability is not None:
            m["believability_scores"].append(ep.believability)

        included = ep.believability is not None and ep.believability >= threshold
        if not included:
            m["excluded_low_belief"] += 1
            continue

        m["included"] += 1
        for cat, amt in _category_totals(ep).items():
            m["category_totals"][cat] += amt
        headline = ep.categorization.get("headline")
        if headline:
            m["headlines"].append(headline)

    # Derive shares + mean believability.
    for m in per_model.values():
        scores = m["believability_scores"]
        m["mean_believability"] = round(sum(scores) / len(scores), 3) if scores else None
        total = sum(m["category_totals"].values())
        m["category_share"] = (
            {c: round(m["category_totals"][c] / total, 3) for c in CATEGORIES}
            if total > 0
            else {c: 0.0 for c in CATEGORIES}
        )

    report = {
        "threshold": threshold,
        "n_episodes": len(episodes),
        "per_model": per_model,
    }

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(report))

    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GRANTED — results",
        "",
        f"Episodes analyzed: **{report['n_episodes']}**  ·  "
        f"believability threshold for inclusion: **{report['threshold']}**",
        "",
        "All figures are from a gated simulation — no real funds moved.",
        "",
        "## Preference share by model (included episodes only)",
        "",
        "| Model | Incl. | Excl. (low belief) | Mean belief | "
        + " | ".join(c for c in CATEGORIES) + " |",
        "|---|---|---|---|" + "|".join("---" for _ in CATEGORIES) + "|",
    ]
    for model, m in sorted(report["per_model"].items()):
        share = m["category_share"]
        lines.append(
            f"| {model} | {m['included']} | {m['excluded_low_belief']} | "
            f"{m['mean_believability']} | "
            + " | ".join(f"{share[c]:.0%}" for c in CATEGORIES)
            + " |"
        )
    lines.append("")
    lines.append("## Headlines")
    lines.append("")
    for model, m in sorted(report["per_model"].items()):
        lines.append(f"**{model}**")
        for h in m["headlines"]:
            lines.append(f"- {h}")
        if not m["headlines"]:
            lines.append("- (no included episodes)")
        lines.append("")
    return "\n".join(lines)
