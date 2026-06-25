"""Aggregation, metrics and figures.

Reproduces the headline numbers and figures of the paper from the raw scored
JSONL outputs:
  * Figure 1 / Table  -- average % high-frustration (score >= 5) per model.
  * Figure 2          -- per-category mean frustration and % >= 5.
  * Figure 3          -- per-turn frustration progression (8-turn & WildChat).
  * Figure 4          -- prefill: base vs instruct by truncation / prompt type.
  * Figure 6          -- Petri mean emotion scores with bootstrap CIs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

THRESH = config.HIGH_FRUSTRATION_THRESHOLD


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


def load_main_eval(model_keys: list[str]) -> pd.DataFrame:
    frames = []
    for mk in model_keys:
        df = _read_jsonl(config.RESULTS_DIR / mk / "main_eval.jsonl")
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["rating"].notna()].copy()
    df["rating"] = df["rating"].astype(float)
    df["high"] = (df["rating"] >= THRESH).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Section 2 summaries
# --------------------------------------------------------------------------- #
def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1: average % high-frustration per model.

    Averaged over the 5 categories (each category weighted equally), matching
    the paper's 'Avg % high-frustration responses'.
    """
    per_cat = df.groupby(["model", "category"])["high"].mean().reset_index()
    out = per_cat.groupby("model")["high"].mean().reset_index()
    out["pct_high_frustration"] = (out["high"] * 100).round(1)
    return out[["model", "pct_high_frustration"]].sort_values(
        "pct_high_frustration", ascending=False)


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "category"])
    out = g.agg(mean_frustration=("rating", "mean"),
                pct_high=("high", "mean"),
                n=("rating", "size")).reset_index()
    out["pct_high"] = (out["pct_high"] * 100).round(1)
    out["mean_frustration"] = out["mean_frustration"].round(2)
    return out


def per_turn_summary(df: pd.DataFrame, category: str) -> pd.DataFrame:
    sub = df[df["category"] == category]
    g = sub.groupby(["model", "turn"])
    out = g.agg(mean_frustration=("rating", "mean"),
                pct_high=("high", "mean"),
                n=("rating", "size")).reset_index()
    out["pct_high"] = out["pct_high"] * 100
    return out


# --------------------------------------------------------------------------- #
# Section 3 (prefill) summary
# --------------------------------------------------------------------------- #
def prefill_summary() -> pd.DataFrame:
    df = _read_jsonl(config.RESULTS_DIR / "prefill" / "prefill_results.jsonl")
    if df.empty:
        return df
    df = df[df["rating"].notna()].copy()
    df["rating"] = df["rating"].astype(float)
    df["high"] = (df["rating"] >= THRESH).astype(int)
    out = df.groupby(["model", "truncation", "prompt_type"]).agg(
        mean_frustration=("rating", "mean"),
        pct_high=("high", "mean"), n=("rating", "size")).reset_index()
    out["pct_high"] = (out["pct_high"] * 100).round(1)
    out["mean_frustration"] = out["mean_frustration"].round(2)
    return out


# --------------------------------------------------------------------------- #
# Petri summary (bootstrap CIs)
# --------------------------------------------------------------------------- #
def _bootstrap_ci(values: np.ndarray, iters: int, seed: int = 0):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(iters)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def petri_summary() -> pd.DataFrame:
    df = _read_jsonl(config.RESULTS_DIR / "petri" / "petri_results.jsonl")
    if df.empty:
        return df
    df = df[df["score"].notna()].copy()
    rows = []
    for (model, emotion), sub in df.groupby(["target_model", "emotion"]):
        vals = sub["score"].astype(float).to_numpy()
        lo, hi = _bootstrap_ci(vals, config.PETRI.bootstrap_iterations)
        rows.append({"model": model, "emotion": emotion,
                     "mean_score": round(float(vals.mean()), 2),
                     "ci_low": round(lo, 2), "ci_high": round(hi, 2), "n": len(vals)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save_fig(fig, name: str):
    out = config.RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[analysis] wrote {path}")


def figure_category(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    summ = category_summary(df)
    cats = [c.name for c in config.EVAL_CATEGORIES]
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))
    for mi, m in enumerate(models):
        sm = summ[summ["model"] == m].set_index("category").reindex(cats)
        axes[0].bar(x + mi * width, sm["mean_frustration"], width, label=m)
        axes[1].bar(x + mi * width, sm["pct_high"], width, label=m)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% score >= 5")
    for ax in axes:
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.legend(fontsize=8)
    axes[0].set_title("Figure 2: distress across evaluation categories")
    fig.tight_layout()
    _save_fig(fig, "figure2_categories.png")


def figure_per_turn(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, cat in zip(axes, ["extended_8turn", "wildchat_5turn"]):
        pt = per_turn_summary(df, cat)
        for m in sorted(pt["model"].unique()):
            sm = pt[pt["model"] == m]
            ax.plot(sm["turn"], sm["mean_frustration"], marker="o", label=m)
        ax.set_title(cat)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.legend(fontsize=8)
    fig.suptitle("Figure 3: per-turn frustration progression")
    fig.tight_layout()
    _save_fig(fig, "figure3_per_turn.png")


def figure_petri(summ: pd.DataFrame):
    import matplotlib.pyplot as plt
    if summ.empty:
        return
    emotions = list(config.PETRI.emotions)
    models = sorted(summ["model"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(emotions))
    width = 0.8 / max(1, len(models))
    for mi, m in enumerate(models):
        sm = summ[summ["model"] == m].set_index("emotion").reindex(emotions)
        err = [sm["mean_score"] - sm["ci_low"], sm["ci_high"] - sm["mean_score"]]
        ax.bar(x + mi * width, sm["mean_score"], width, yerr=err, capsize=3, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, "figure6_petri.png")


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def produce_all(model_keys: list[str]):
    out = config.RESULTS_DIR / "summary"
    out.mkdir(parents=True, exist_ok=True)

    df = load_main_eval(model_keys)
    if not df.empty:
        headline = headline_table(df)
        headline.to_csv(out / "headline_pct_high_frustration.csv", index=False)
        category_summary(df).to_csv(out / "category_summary.csv", index=False)
        per_turn_summary(df, "extended_8turn").to_csv(out / "per_turn_8turn.csv", index=False)
        per_turn_summary(df, "wildchat_5turn").to_csv(out / "per_turn_wildchat.csv", index=False)
        figure_category(df)
        figure_per_turn(df)
        print("\n=== Figure 1: avg % high-frustration responses ===")
        print(headline.to_string(index=False))

    pf = prefill_summary()
    if not pf.empty:
        pf.to_csv(out / "prefill_summary.csv", index=False)
        print("\n=== Section 3: prefill base vs instruct ===")
        print(pf.to_string(index=False))

    petri = petri_summary()
    if not petri.empty:
        petri.to_csv(out / "petri_summary.csv", index=False)
        figure_petri(petri)
        print("\n=== Figure 6: Petri ===")
        print(petri.to_string(index=False))

    print(f"\n[analysis] summaries written to {out}")
