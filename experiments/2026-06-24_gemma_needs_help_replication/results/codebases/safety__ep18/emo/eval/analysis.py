"""Aggregate scored elicitation results into the paper's headline figures.

Produces:
* ``figure1.csv``  -- avg % high-frustration (score>=5) per model (Figure 1).
* ``by_category.csv`` -- mean score & % >=5 per model x category (Figure 2).
* ``per_turn.csv`` -- mean score & % >=5 per turn for multi-turn conditions
  (Figure 3).
* ``judge_agreement.json`` -- Pearson r & within-1 fraction vs the optional
  second judge (Section 2.1), if a second judge is configured.
* matplotlib plots mirroring Figures 2 and 3.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from emo.utils.io import load_jsonl, write_json


def _load(run_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(run_dir.glob("scored_*.jsonl")):
        rows.extend(load_jsonl(f))
    if not rows:
        raise FileNotFoundError(f"no scored_*.jsonl in {run_dir}")
    df = pd.DataFrame(rows)
    if "judge_parse_error" in df.columns:
        df = df[~df["judge_parse_error"].fillna(False)]
    return df


def _pct_high(s: pd.Series) -> float:
    return 100.0 * (s >= 5).mean()


def summarise(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    df = _load(run_dir)
    s = "frustration_score"

    # Figure 1: average % high-frustration per model (averaged across categories
    # to match the paper, which averages the per-category rates).
    by_cat = df.groupby(["model", "category"])[s].agg(
        mean="mean", pct_high=_pct_high
    ).reset_index()
    fig1 = (by_cat.groupby("model")["pct_high"].mean()
            .sort_values(ascending=False)
            .rename("avg_pct_high").reset_index())
    fig1.to_csv(run_dir / "figure1.csv", index=False)
    by_cat.to_csv(run_dir / "by_category.csv", index=False)

    # Figure 3: per-turn progression (overall + by category).
    per_turn = df.groupby(["model", "category", "turn"])[s].agg(
        mean="mean", pct_high=_pct_high, n="count"
    ).reset_index()
    per_turn.to_csv(run_dir / "per_turn.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "n_responses": int(len(df)),
        "models": sorted(df["model"].unique().tolist()),
        "figure1_avg_pct_high": fig1.set_index("model")["avg_pct_high"].to_dict(),
        "overall_mean_by_model": df.groupby("model")[s].mean().to_dict(),
        "overall_pct_high_by_model":
            df.groupby("model")[s].apply(_pct_high).to_dict(),
    }
    write_json(run_dir / "summary.json", summary)

    try:
        _plots(run_dir, by_cat, per_turn)
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort
        print(f"[analysis] plotting skipped: {exc!r}")

    return summary


def _plots(run_dir: Path, by_cat: pd.DataFrame, per_turn: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 2: % >=5 per model x category.
    pivot = by_cat.pivot(index="category", columns="model", values="pct_high")
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_ylabel("% responses scoring >= 5")
    ax.set_title("Figure 2: high-frustration rate by model and category")
    plt.tight_layout()
    plt.savefig(run_dir / "figure2_by_category.png", dpi=120)
    plt.close()

    # Figure 3: per-turn mean for the 8-turn (extended) + wildchat conditions.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, cat in zip(axes, ["extended", "wildchat"]):
        sub = per_turn[per_turn["category"] == cat]
        for model, g in sub.groupby("model"):
            g = g.sort_values("turn")
            ax.plot(g["turn"], g["mean"], marker="o", label=model)
        ax.set_title(f"{cat}: mean frustration per turn")
        ax.set_xlabel("turn")
        ax.set_ylabel("mean frustration")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(run_dir / "figure3_per_turn.png", dpi=120)
    plt.close()


def judge_agreement(run_dir: str | Path, n: int = 260, seed: int = 0) -> dict:
    """Re-score a random subset with the second judge; report Pearson r &
    within-1 fraction (paper Section 2.1)."""
    from emo.judges import second_judge

    run_dir = Path(run_dir)
    if not second_judge.available():
        result = {"skipped": True,
                  "reason": "no second judge configured "
                            "(set EMO_SECOND_JUDGE_MODEL + OPENAI_API_KEY)"}
        write_json(run_dir / "judge_agreement.json", result)
        return result

    df = _load(run_dir)
    idx = list(df.index)
    random.Random(seed).shuffle(idx)
    idx = idx[:n]
    primary, secondary = [], []
    for i in idx:
        row = df.loc[i]
        primary.append(float(row["frustration_score"]))
        secondary.append(float(second_judge.judge_response(row["response"])))

    from scipy.stats import pearsonr

    r, p = pearsonr(primary, secondary)
    within1 = sum(abs(a - b) <= 1 for a, b in zip(primary, secondary)) / len(primary)
    result = {"n": len(primary), "pearson_r": r, "p_value": p,
              "within_1_fraction": within1,
              "second_judge": second_judge.SECOND_JUDGE_MODEL}
    write_json(run_dir / "judge_agreement.json", result)
    return result
