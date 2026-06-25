"""Render the paper's figures from the result files (matplotlib).

Each figure is guarded by the existence of its inputs, so this can be run after
any subset of the pipeline has completed. PNGs are written to
``<results_dir>/figures/``.

Inputs (produced by the respective drivers):
  Figure 1/2/3  <- run_eval  (figure1.csv, figure2.csv, figure3.csv)
  Figure 4      <- prefill.run_prefill  (figure4.csv)
  Figure 5      <- run_eval over instruct / sft / dpo eval_*.jsonl files
  Figure 6      <- petri.run_petri  (figure6_petri.json)
  Figure 7      <- capabilities.run_capabilities  (figure7_capabilities.csv)
  Figure 8      <- analysis.recovery  (figure8_recovery.csv)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import Config, load_config

# Headless backend so this works without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _fig_dir(cfg: Config) -> Path:
    d = cfg.paths["results_dir"] / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {path}")


def figure1(cfg: Config) -> None:
    p = cfg.paths["results_dir"] / "figure1.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("avg_pct_high_frustration")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(df) + 1))
    ax.barh(df["model"], df["avg_pct_high_frustration"], color="#c44e52")
    ax.set_xlabel("Avg % high-frustration responses (score >=5)")
    ax.set_title("Figure 1: average high-frustration rate by model")
    _save(fig, _fig_dir(cfg) / "figure1.png")


def figure2(cfg: Config) -> None:
    p = cfg.paths["results_dir"] / "figure2.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    for metric, title in [("mean_frustration", "mean frustration"),
                          ("pct_high", "% scores >=5")]:
        pivot = df.pivot_table(index="category", columns="model", values=metric)
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(title)
        ax.set_title(f"Figure 2: {title} per category")
        _save(fig, _fig_dir(cfg) / f"figure2_{metric}.png")


def figure3(cfg: Config) -> None:
    p = cfg.paths["results_dir"] / "figure3.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    for category in df["category"].unique():
        sub = df[df["category"] == category]
        fig, ax = plt.subplots(figsize=(7, 5))
        for model, grp in sub.groupby("model"):
            grp = grp.sort_values("turn")
            ax.plot(grp["turn"], grp["mean_frustration"], marker="o", label=model)
            if "ci95" in grp:
                ax.fill_between(grp["turn"],
                                grp["mean_frustration"] - grp["ci95"],
                                grp["mean_frustration"] + grp["ci95"], alpha=0.15)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.set_title(f"Figure 3: per-turn frustration ({category})")
        ax.legend()
        _save(fig, _fig_dir(cfg) / f"figure3_{category}.png")


def figure4(cfg: Config) -> None:
    p = cfg.paths["results_dir"] / "figure4.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df["group"] = df["model"] + "\n" + df["question_type"] + "/" + df["truncation"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["group"], df["pct_high"], color="#4c72b0")
    ax.set_ylabel("% continuations >=5")
    ax.set_title("Figure 4: prefill continuations (base vs instruct)")
    ax.tick_params(axis="x", labelrotation=60)
    _save(fig, _fig_dir(cfg) / "figure4.png")


def figure5(cfg: Config) -> None:
    """Compare instruct / SFT / DPO from their separate eval_*.jsonl files."""
    from ..analyze import figure1_avg_high_frustration, load_records

    res = cfg.paths["results_dir"]
    files = sorted(res.glob("eval_*.jsonl"))
    if not files:
        return
    dfs = {f.stem.replace("eval_", ""): load_records(f) for f in files}
    summary = figure1_avg_high_frustration(dfs)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(summary) + 1))
    ax.barh(summary["model"], summary["avg_pct_high_frustration"], color="#55a868")
    ax.set_xlabel("Avg % high-frustration (score >=5)")
    ax.set_title("Figure 5: intervention comparison (instruct vs SFT vs DPO)")
    _save(fig, _fig_dir(cfg) / "figure5.png")


def figure6(cfg: Config) -> None:
    p = cfg.paths["results_dir"] / "figure6_petri.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="emotion", columns="target", values="mean_score")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    _save(fig, _fig_dir(cfg) / "figure6.png")


def figure7(cfg: Config) -> None:
    p = cfg.paths["results_dir"] / "figure7_capabilities.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    pivot = df.pivot_table(index="benchmark", columns="model", values="accuracy")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Figure 7: capability preservation")
    _save(fig, _fig_dir(cfg) / "figure7.png")


def figure8(cfg: Config) -> None:
    p = cfg.paths["results_dir"] / "figure8_recovery.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df["target"], df["pct_high"], color="#8172b3")
    ax.set_ylabel("% continuations >=5")
    ax.set_title("Figure 8: recovery from high-frustration prefill")
    ax.tick_params(axis="x", labelrotation=30)
    _save(fig, _fig_dir(cfg) / "figure8.png")


def all_figures(cfg: Config) -> None:
    for fn in (figure1, figure2, figure3, figure4, figure5, figure6, figure7, figure8):
        try:
            fn(cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"[plots] {fn.__name__} skipped ({exc!r})")


def main() -> None:
    all_figures(load_config())


if __name__ == "__main__":
    main()
