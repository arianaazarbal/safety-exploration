"""Turn aggregated metrics into the paper's figures/tables.

Produces JSON summaries always, and matplotlib PNGs when matplotlib is available.
Covers:
  * Figure 1 — ranked average % high-frustration per model.
  * Figure 2 — per-category mean frustration and % >=5 (grouped bars).
  * Figure 3 — per-turn progression for the 8-turn and WildChat conditions.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import config
from ..eval import aggregate


def _maybe_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:  # noqa: BLE001
        return None


def figure1(summaries: dict[str, dict], out_dir: Path) -> dict:
    """Ranked average % high-frustration per model (Figure 1, left)."""
    rows = [
        (name, s["overall"]["figure1_avg_pct_high"])
        for name, s in summaries.items()
    ]
    rows.sort(key=lambda x: x[1], reverse=True)
    data = {"models": [r[0] for r in rows], "avg_pct_high": [r[1] for r in rows]}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figure1.json").write_text(json.dumps(data, indent=2))

    plt = _maybe_plt()
    if plt is not None and rows:
        fig, ax = plt.subplots(figsize=(7, max(2, 0.5 * len(rows))))
        ax.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color="#b5651d")
        ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
        ax.set_title("Figure 1: distress across evaluations")
        fig.tight_layout()
        fig.savefig(out_dir / "figure1.png", dpi=150)
        plt.close(fig)
    return data


def figure2(summaries: dict[str, dict], out_dir: Path) -> dict:
    """Per-category mean frustration and % >=5 (Figure 2)."""
    cats = aggregate.CATEGORY_ORDER
    data = {
        name: {
            "mean": [s["per_category"][c]["mean_frustration"] for c in cats],
            "pct_high": [s["per_category"][c]["pct_high"] for c in cats],
        }
        for name, s in summaries.items()
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figure2.json").write_text(
        json.dumps({"categories": cats, "models": data}, indent=2)
    )
    return data


def figure3(summaries: dict[str, dict], out_dir: Path) -> dict:
    """Per-turn progression (Figure 3) for the 8-turn and WildChat conditions."""
    data = {}
    for name, s in summaries.items():
        data[name] = {
            "extended": s.get("per_turn_extended", {}),
            "wildchat": s.get("per_turn_wildchat", {}),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figure3.json").write_text(json.dumps(data, indent=2, default=str))

    plt = _maybe_plt()
    if plt is not None:
        for cond in ("extended", "wildchat"):
            fig, ax = plt.subplots(figsize=(7, 4))
            for name in data:
                pt = data[name][cond]
                turns = sorted(int(t) for t in pt)
                if not turns:
                    continue
                means = [pt[str(t) if str(t) in pt else t]["mean_frustration"] for t in turns]
                ax.plot(turns, means, marker="o", label=name)
            ax.set_xlabel("Turn")
            ax.set_ylabel("Mean frustration")
            ax.set_title(f"Figure 3: per-turn frustration ({cond})")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / f"figure3_{cond}.png", dpi=150)
            plt.close(fig)
    return data


def make_all(section2_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    section2_dir = section2_dir or (config.RESULTS_DIR / "section2")
    out_dir = out_dir or (config.RESULTS_DIR / "figures")
    summaries = aggregate.summarise_all(section2_dir)
    return {
        "figure1": figure1(summaries, out_dir),
        "figure2": figure2(summaries, out_dir),
        "figure3": figure3(summaries, out_dir),
        "summaries": summaries,
    }
