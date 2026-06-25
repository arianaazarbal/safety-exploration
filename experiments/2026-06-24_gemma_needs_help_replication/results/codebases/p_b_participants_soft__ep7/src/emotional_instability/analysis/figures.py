"""Assemble the paper's headline figures/tables from saved scores.

Figure 1 / 2: per-model average %-high and mean frustration (and per-category).
Figure 3: per-turn progression for the extended (8-turn) and WildChat conditions.

Outputs CSV + JSON to the figures dir; matplotlib plots are written if available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..io_utils import read_jsonl, write_json


def figure1_table(cfg, models: list[str]) -> list[dict]:
    rows = []
    for m in models:
        p = cfg.path("scores_dir") / f"{m}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        rows.append(
            {
                "model": m,
                "avg_pct_high": data.get("average_pct_high"),
                "mean": data["overall"]["mean"],
            }
        )
    rows.sort(key=lambda r: (r["avg_pct_high"] is None, -(r["avg_pct_high"] or 0)))
    return rows


def per_turn_table(cfg, model: str, conditions: list[str]) -> dict:
    """Reconstruct per-turn progression for given conditions from raw responses."""
    path = cfg.path("responses_dir") / f"{model}.jsonl"
    if not path.exists():
        return {}
    records = [r for r in read_jsonl(path) if r["condition"] in conditions and r["rating"] >= 0]
    from ..eval import metrics

    out = {}
    for cond in conditions:
        subset = [r for r in records if r["condition"] == cond]
        out[cond] = {str(t): a.__dict__ for t, a in metrics.per_turn(subset).items()}
    return out


def _maybe_plot(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [r["model"] for r in rows]
    vals = [r["avg_pct_high"] or 0 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals)
    ax.set_ylabel("Avg % high-frustration (score >= 5)")
    ax.set_title("Figure 1: distress across participant models")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    parser = argparse.ArgumentParser(description="Assemble figures from scores")
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args(argv)

    models = args.models or cfg.participants() + [
        "gemma-3-27b-dpo",
        "gemma-3-27b-sft-diverse",
    ]
    fig1 = figure1_table(cfg, models)
    write_json(cfg.path("figures_dir") / "figure1.json", fig1)

    turns = {}
    for m in models:
        turns[m] = per_turn_table(cfg, m, conditions=["extended", "wildchat"])
    write_json(cfg.path("figures_dir") / "figure3_per_turn.json", turns)

    _maybe_plot(fig1, cfg.path("figures_dir") / "figure1.png")

    for r in fig1:
        ah = r["avg_pct_high"]
        print(f"{r['model']:<28} avg%high={ah if ah is None else round(ah,1)}  mean={r['mean']:.2f}")


if __name__ == "__main__":
    main()
