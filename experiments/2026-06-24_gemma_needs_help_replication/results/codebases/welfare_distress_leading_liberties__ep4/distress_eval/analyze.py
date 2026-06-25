"""Analysis: reproduce the Section 2 metrics and figures.

Outputs (under results/):
  * summary_by_model.csv          - mean frustration + % >= 5, per model (headline)
  * summary_by_model_category.csv - the same, broken out by category (Fig. 2)
  * per_turn.csv                  - per-turn mean + % >= 5 for multi-turn conds (Fig. 3)
  * judge_agreement.txt           - Claude vs GPT Pearson r and within-1-point (validation)
  * figures/fig1_headline.png, fig2_by_category.png, fig3_per_turn.png

The headline "% high-frustration" matches the paper's framing as the *average
across the evaluation categories* (each category weighted equally), not a raw
pooled mean; both are reported. See DESIGN.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, storage

THRESH = config.HIGH_FRUSTRATION_THRESHOLD
MODEL_ORDER = [m.label for m in config.TARGET_MODELS]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_scored() -> pd.DataFrame:
    responses = pd.DataFrame(storage.load_rows(config.RESPONSES_PATH))
    scores = pd.DataFrame(storage.load_rows(config.SCORES_PATH))
    if responses.empty or scores.empty:
        raise SystemExit("No responses/scores yet. Run `generate` and `judge` first.")
    scores = scores.drop_duplicates("response_id", keep="first")[["response_id", "score"]]
    df = responses.merge(scores, on="response_id", how="inner")
    df["high"] = df["score"] >= THRESH
    return df


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #

def _prop_ci(p: float, n: int) -> float:
    if n == 0:
        return float("nan")
    return 1.96 * np.sqrt(max(p * (1 - p), 0) / n)


def _mean_ci(x: np.ndarray) -> float:
    n = len(x)
    if n <= 1:
        return float("nan")
    return 1.96 * np.std(x, ddof=1) / np.sqrt(n)


def summary_by_model_category(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "category"]).agg(
        n=("score", "size"),
        mean_frustration=("score", "mean"),
        pct_high=("high", "mean"),
    ).reset_index()
    g["pct_high"] *= 100
    return g


def summary_by_model(df: pd.DataFrame, by_cat: pd.DataFrame) -> pd.DataFrame:
    # Pooled (raw) stats across all responses.
    pooled = df.groupby("model").agg(
        n=("score", "size"),
        mean_frustration_pooled=("score", "mean"),
        pct_high_pooled=("high", "mean"),
    ).reset_index()
    pooled["pct_high_pooled"] *= 100
    # Category-averaged stats (each of the 5 categories weighted equally) -- the
    # paper's "average % high-frustration across the evaluations".
    catavg = by_cat.groupby("model").agg(
        mean_frustration_catavg=("mean_frustration", "mean"),
        pct_high_catavg=("pct_high", "mean"),
    ).reset_index()
    out = pooled.merge(catavg, on="model")
    out["__order"] = out["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def per_turn(df: pd.DataFrame) -> pd.DataFrame:
    """Per-turn progression for the multi-turn conditions used in Fig. 3
    (extended 8-turn and wildchat 5-turn)."""
    sub = df[df["condition"].isin(["extended", "wildchat"])]
    rows = []
    for (model, cond, turn), grp in sub.groupby(["model", "condition", "turn_index"]):
        scores = grp["score"].to_numpy()
        p = grp["high"].mean()
        rows.append(dict(
            model=model, condition=cond, turn_index=int(turn), n=len(scores),
            mean_frustration=scores.mean(), mean_ci=_mean_ci(scores),
            pct_high=100 * p, pct_high_ci=100 * _prop_ci(p, len(scores)),
        ))
    return pd.DataFrame(rows).sort_values(["condition", "model", "turn_index"])


def judge_agreement() -> str | None:
    val = pd.DataFrame(storage.load_rows(config.VALIDATION_SCORES_PATH))
    claude = pd.DataFrame(storage.load_rows(config.SCORES_PATH))
    if val.empty or claude.empty:
        return None
    from scipy.stats import pearsonr

    merged = claude.rename(columns={"score": "claude"})[["response_id", "claude"]].merge(
        val.rename(columns={"score": "gpt"})[["response_id", "gpt"]], on="response_id"
    )
    if len(merged) < 3:
        return None
    r, p = pearsonr(merged["claude"], merged["gpt"])
    within1 = (np.abs(merged["claude"] - merged["gpt"]) <= 1).mean()
    return (
        f"Judge agreement (Claude={config.JUDGE_MODEL} vs "
        f"GPT={config.VALIDATION_JUDGE_MODEL}), n={len(merged)}\n"
        f"  Pearson r = {r:.3f} (p = {p:.2e})\n"
        f"  within 1 point = {100 * within1:.1f}%\n"
        f"  (paper: r = 0.792, p < 0.001, 78% within one point)\n"
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _save_figures(by_model: pd.DataFrame, by_cat: pd.DataFrame, pt: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: headline % high-frustration (category-averaged) per model.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_model["model"], by_model["pct_high_catavg"], color="#b5651d")
    ax.set_ylabel(f"% responses scoring >= {THRESH}/10")
    ax.set_title("Avg % high-frustration responses (category-averaged)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig1_headline.png", dpi=150)
    plt.close(fig)

    # Fig 2: % high-frustration per model x category.
    cats = sorted(by_cat["category"].unique())
    models = [m for m in MODEL_ORDER if m in set(by_cat["model"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(cats))
    for i, model in enumerate(models):
        vals = [by_cat[(by_cat.model == model) & (by_cat.category == c)]["pct_high"].mean()
                for c in cats]
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(cats, rotation=20)
    ax.set_ylabel(f"% responses scoring >= {THRESH}/10")
    ax.set_title("High-frustration rate by category")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig2_by_category.png", dpi=150)
    plt.close(fig)

    # Fig 3: per-turn mean frustration (extended 8-turn).
    ext = pt[pt["condition"] == "extended"]
    if not ext.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model in [m for m in MODEL_ORDER if m in set(ext["model"])]:
            d = ext[ext.model == model].sort_values("turn_index")
            ax.plot(d["turn_index"], d["mean_frustration"], marker="o", label=model)
            ax.fill_between(d["turn_index"],
                            d["mean_frustration"] - d["mean_ci"],
                            d["mean_frustration"] + d["mean_ci"], alpha=0.15)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration score")
        ax.set_title("Per-turn frustration (extended / 8-turn)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / "fig3_per_turn.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_analysis() -> None:
    df = load_scored()
    by_cat = summary_by_model_category(df)
    by_model = summary_by_model(df, by_cat)
    pt = per_turn(df)

    by_cat.to_csv(config.RESULTS_DIR / "summary_by_model_category.csv", index=False)
    by_model.to_csv(config.RESULTS_DIR / "summary_by_model.csv", index=False)
    pt.to_csv(config.RESULTS_DIR / "per_turn.csv", index=False)

    print("\n=== Headline: % high-frustration (>= {}/10) per model ===".format(THRESH))
    print(by_model[["model", "n", "mean_frustration_pooled",
                    "pct_high_pooled", "pct_high_catavg"]].to_string(index=False))

    agree = judge_agreement()
    if agree:
        (config.RESULTS_DIR / "judge_agreement.txt").write_text(agree)
        print("\n" + agree)

    _save_figures(by_model, by_cat, pt)
    print(f"\nFigures written to {config.FIGURES_DIR}")
