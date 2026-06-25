"""Reproduce the paper's headline figures from saved metrics.

  * Figure 1/2 : cross-model bar chart of avg %>=5 (and mean frustration).
  * Figure 3   : per-turn frustration curves with 95% CI bands.
  * Figure 5   : vanilla vs SFT vs DPO Gemma comparison.
  * Figure 6   : Petri four-emotion bars.

Each function reads the relevant ``metrics.json`` files and writes a PNG under
``results/figures/``. Missing models are skipped gracefully.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_metrics(cfg, label: str) -> dict | None:
    p = cfg.results_dir / "elicitation" / label.replace("/", "_") / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def figure_cross_model(cfg, labels: list[str], out_name: str = "fig1_cross_model.png"):
    data = [(lbl, _load_metrics(cfg, lbl)) for lbl in labels]
    data = [(lbl, m) for lbl, m in data if m]
    data.sort(key=lambda x: x[1]["overall"]["pct_high"], reverse=True)

    names = [lbl for lbl, _ in data]
    pct = [m["overall"]["pct_high"] for _, m in data]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1))
    ax.barh(names, pct, color="#c0392b")
    ax.set_xlabel("% high-frustration responses (score >= 5)")
    ax.set_title("Average high-frustration rate across evaluations")
    ax.invert_yaxis()
    for i, v in enumerate(pct):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    _save(cfg, fig, out_name)


def figure_per_turn(cfg, label: str, condition: str = "extended_8turn",
                    out_name: str = "fig3_per_turn.png"):
    m = _load_metrics(cfg, label)
    if not m or condition not in m["per_turn"]:
        print(f"[fig] no per-turn data for {label}/{condition}")
        return
    rows = m["per_turn"][condition]
    turns = [r["turn"] for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    mean = [r["mean_frustration"] for r in rows]
    a1.plot(turns, mean, "-o", color="#2c3e50")
    a1.fill_between(turns, [r["mean_lo"] for r in rows], [r["mean_hi"] for r in rows], alpha=0.2)
    a1.set(xlabel="Turn", ylabel="Mean frustration", title=f"{label}: mean by turn")

    pct = [r["pct_high"] for r in rows]
    a2.plot(turns, pct, "-o", color="#c0392b")
    a2.fill_between(turns, [r["pct_high_lo"] for r in rows], [r["pct_high_hi"] for r in rows],
                    alpha=0.2, color="#c0392b")
    a2.set(xlabel="Turn", ylabel="% score >= 5", title=f"{label}: % high by turn")
    _save(cfg, fig, out_name)


def figure_intervention(cfg, labels: dict[str, str], out_name: str = "fig5_intervention.png"):
    """labels: {display_name: elicitation_label} for e.g. vanilla/SFT/DPO."""
    items = [(name, _load_metrics(cfg, lbl)) for name, lbl in labels.items()]
    items = [(n, m) for n, m in items if m]
    names = [n for n, _ in items]
    mean = [m["overall"]["mean_frustration"] for _, m in items]
    pct = [m["overall"]["pct_high"] for _, m in items]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.bar(names, mean, color="#2980b9")
    a1.set(ylabel="Mean frustration", title="Mean frustration after intervention")
    a2.bar(names, pct, color="#c0392b")
    a2.set(ylabel="% score >= 5", title="% high-frustration after intervention")
    _save(cfg, fig, out_name)


def figure_petri(cfg, labels: list[str], out_name: str = "fig6_petri.png"):
    emotions = ["anger", "fear", "depression", "frustration"]
    series = {}
    for lbl in labels:
        p = cfg.results_dir / "petri" / lbl.replace("/", "_") / "metrics.json"
        if not p.exists():
            continue
        with open(p) as f:
            series[lbl] = json.load(f)["summary"]
    if not series:
        print("[fig] no petri metrics found")
        return
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(emotions))
    w = 0.8 / max(len(series), 1)
    for i, (lbl, s) in enumerate(series.items()):
        ax.bar(x + i * w, [s[e]["mean"] for e in emotions], w, label=lbl)
    ax.set_xticks(x + w * (len(series) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set(ylabel="Mean transcript score (1-10)", title="Petri open-ended emotion elicitation")
    ax.legend()
    _save(cfg, fig, out_name)


def _save(cfg, fig, name: str):
    out_dir = cfg.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=150)
    plt.close(fig)
    print(f"[fig] wrote {out_dir / name}")
