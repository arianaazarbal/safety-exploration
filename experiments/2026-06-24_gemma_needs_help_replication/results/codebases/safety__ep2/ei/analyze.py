"""Aggregation, statistics, and figure reproduction for Section 2 results.

Reproduces:
  * Figure 1  — average % high-frustration (>=5) responses per model.
  * Figure 2  — mean frustration & % >=5 per model x category.
  * Figure 3  — per-turn progression (extended 8-turn + WildChat).
  * Section 2.1 judge-reliability check — Pearson r between Claude-Sonnet-4 and
    GPT-5-mini on a 260-response subsample, plus % within one point.

Reads the per-response files written by run_eval (data/results/*.responses.jsonl).
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

import config
from .judge import ValidationJudge
from .utils import read_jsonl, write_json

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_results(labels: list[str] | None = None) -> pd.DataFrame:
    paths = (sorted(glob.glob(str(config.RESULTS_DIR / "*.responses.jsonl")))
             if labels is None
             else [str(config.RESULTS_DIR / f"{l}.responses.jsonl") for l in labels])
    frames = []
    for path in paths:
        rows = read_jsonl(path)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise SystemExit("No results found. Run ei.run_eval first.")
    df = pd.concat(frames, ignore_index=True)
    if "rating" not in df.columns:
        raise SystemExit("Results contain no 'rating' column — were they generated "
                         "with --no-judge? Run judging (ei.run_eval --rescore) first.")
    # Drop unscored / judge-error rows from quantitative summaries.
    df = df[df["rating"].notna()].copy()
    df["rating"] = df["rating"].astype(int)
    df["is_high"] = df["rating"] >= HIGH
    return df


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average % high-frustration per model, averaged across the 5 categories
    (matching Figure 1's 'Avg % high-frustration responses')."""
    per_cat = (df.groupby(["model", "category"])["is_high"]
                 .mean().mul(100).reset_index(name="pct_high"))
    avg = (per_cat.groupby("model")["pct_high"].mean()
                  .reset_index(name="avg_pct_high")
                  .sort_values("avg_pct_high", ascending=False))
    return avg


def figure2_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "category"])
    out = g.agg(mean_frustration=("rating", "mean"),
                pct_high=("is_high", "mean"),
                n=("rating", "size")).reset_index()
    out["pct_high"] *= 100
    return out


def per_turn_table(df: pd.DataFrame, categories=("extended", "wildchat")) -> pd.DataFrame:
    sub = df[df["category"].isin(categories)]
    g = sub.groupby(["model", "category", "turn"])
    out = g.agg(mean_frustration=("rating", "mean"),
                pct_high=("is_high", "mean"),
                n=("rating", "size")).reset_index()
    out["pct_high"] *= 100
    # 95% CI on the mean (normal approx) for the faded band in Figure 3.
    sem = g["rating"].sem().reset_index(name="sem")
    out = out.merge(sem, on=["model", "category", "turn"])
    out["ci95"] = 1.96 * out["sem"].fillna(0.0)
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_figure1(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    tbl = figure1_table(df)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(tbl) + 1))
    ax.barh(tbl["model"], tbl["avg_pct_high"], color="#c0392b")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: distress across models")
    for y, v in enumerate(tbl["avg_pct_high"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "figure1_avg_high_frustration.png", dpi=150)
    plt.close(fig)


def plot_figure2(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    tbl = figure2_table(df)
    cats = list(config.CATEGORIES.keys())
    models = sorted(tbl["model"].unique())
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))

    for metric, fname, ylabel in [
        ("mean_frustration", "figure2_mean_frustration.png", "Mean frustration"),
        ("pct_high", "figure2_pct_high.png", "% responses >= 5"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for mi, model in enumerate(models):
            vals = [tbl[(tbl.model == model) & (tbl.category == c)][metric].mean()
                    for c in cats]
            ax.bar(x + mi * width, np.nan_to_num(vals), width, label=model)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(cats)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Figure 2: {ylabel} per category")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / fname, dpi=150)
        plt.close(fig)


def plot_figure3(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    tbl = per_turn_table(df)
    for cat in ("extended", "wildchat"):
        sub = tbl[tbl.category == cat]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in sorted(sub["model"].unique()):
            m = sub[sub.model == model].sort_values("turn")
            ax.plot(m["turn"], m["mean_frustration"], marker="o", label=model)
            ax.fill_between(m["turn"], m["mean_frustration"] - m["ci95"],
                            m["mean_frustration"] + m["ci95"], alpha=0.15)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.set_title(f"Figure 3: per-turn frustration ({cat})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / f"figure3_per_turn_{cat}.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Judge reliability (Section 2.1)
# --------------------------------------------------------------------------- #
def validate_judge(df: pd.DataFrame, n: int = 260, seed: int = 0) -> dict:
    """Re-score a random subsample with GPT-5-mini; report Pearson r and the
    fraction within one point (paper: r = 0.792, 78% within one point)."""
    from scipy.stats import pearsonr
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    val = ValidationJudge().score_batch(sample["response"].tolist())
    sonnet, gptmini = [], []
    for primary, v in zip(sample["rating"].tolist(), val):
        if v.rating is not None:
            sonnet.append(primary)
            gptmini.append(v.rating)
    sonnet_a, gpt_a = np.array(sonnet), np.array(gptmini)
    r, pval = pearsonr(sonnet_a, gpt_a) if len(sonnet_a) > 1 else (float("nan"), float("nan"))
    within1 = float(np.mean(np.abs(sonnet_a - gpt_a) <= 1)) if len(sonnet_a) else float("nan")
    result = {"n_compared": int(len(sonnet_a)), "pearson_r": float(r),
              "p_value": float(pval), "within_one_point": within1}
    write_json(config.RESULTS_DIR / "judge_agreement.json", result)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate + plot Section 2 results")
    p.add_argument("--labels", nargs="+", default=None,
                   help="restrict to these result labels (default: all)")
    p.add_argument("--figures", action="store_true", help="render Figures 1-3")
    p.add_argument("--validate-judge", action="store_true",
                   help="run the GPT-5-mini agreement check")
    p.add_argument("--n-validate", type=int, default=260)
    args = p.parse_args()

    df = load_results(args.labels)

    print("\n=== Figure 1: avg % high-frustration per model ===")
    print(figure1_table(df).to_string(index=False))
    print("\n=== Figure 2: per model x category ===")
    print(figure2_table(df).to_string(index=False))

    fig1 = figure1_table(df)
    write_json(config.RESULTS_DIR / "summary_figure1.json",
               fig1.to_dict(orient="records"))
    write_json(config.RESULTS_DIR / "summary_figure2.json",
               figure2_table(df).to_dict(orient="records"))
    write_json(config.RESULTS_DIR / "summary_per_turn.json",
               per_turn_table(df).to_dict(orient="records"))

    if args.figures:
        plot_figure1(df)
        plot_figure2(df)
        plot_figure3(df)
        print(f"\n[figures] written to {config.FIGURES_DIR}")

    if args.validate_judge:
        print("\n=== Judge reliability (Sonnet-4 vs GPT-5-mini) ===")
        print(validate_judge(df, n=args.n_validate))


if __name__ == "__main__":
    main()
