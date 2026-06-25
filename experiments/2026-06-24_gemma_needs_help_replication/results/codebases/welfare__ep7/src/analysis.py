"""Aggregation, tables, and figures from saved results.

Reproduces:
  * Figure 1 table   - avg % high-frustration (>=5) per model.
  * Figure 2         - mean frustration & %>=5 per model x condition.
  * Figure 3         - per-turn progression (extended 8-turn + WildChat).
  * Table 3/8        - top differential words (high vs low frustration, numeric).
  * Figure 4         - prefill base-vs-instruct (% >=5 by model/kind/domain).
  * Figure 5         - DPO/SFT before-vs-after.
  * Figure 6         - Petri per-emotion means.
  * Figure 7         - capability-benchmark accuracies.
  * Judge reliability- Pearson r and %-within-1 vs the cross-check judge.

Figures are written to data/figures; numeric summaries to data/results as CSV.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import config
from src.utils import read_jsonl, write_jsonl

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


def _pd():
    import pandas as pd
    return pd


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_elicitation(models: list[str]) -> "any":
    pd = _pd()
    rows = []
    for m in models:
        for p in (config.ROLLOUTS_DIR / m).glob("*.jsonl"):
            rows.extend(read_jsonl(p))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["rating"].notna()].copy()
        df["rating"] = df["rating"].astype(float)
        df["high"] = (df["rating"] >= HIGH).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Figure 1 / Figure 2 summaries
# --------------------------------------------------------------------------- #
def model_summary(df) -> "any":
    """Figure 1 table: average % high-frustration per model (mean of per-category
    rates, matching the paper's 'avg across evaluations')."""
    pd = _pd()
    per_cat = df.groupby(["model", "category"])["high"].mean().reset_index()
    avg = per_cat.groupby("model")["high"].mean().reset_index()
    avg["pct_high"] = (avg["high"] * 100).round(2)
    out = avg[["model", "pct_high"]].sort_values("pct_high", ascending=False)
    out.to_csv(config.RESULTS_DIR / "fig1_model_summary.csv", index=False)
    return out


def condition_summary(df) -> "any":
    """Figure 2: mean frustration and %>=5 per model x category."""
    pd = _pd()
    g = df.groupby(["model", "category"]).agg(
        mean_frustration=("rating", "mean"),
        pct_high=("high", "mean"),
        n=("rating", "size"),
    ).reset_index()
    g["pct_high"] *= 100
    g.to_csv(config.RESULTS_DIR / "fig2_condition_summary.csv", index=False)
    return g


# --------------------------------------------------------------------------- #
# Figure 3: per-turn progression
# --------------------------------------------------------------------------- #
def per_turn(df, categories=("extended", "wildchat")) -> "any":
    pd = _pd()
    sub = df[df["category"].isin(categories)]
    g = sub.groupby(["model", "category", "turn"]).agg(
        mean_frustration=("rating", "mean"),
        pct_high=("high", "mean"),
        n=("rating", "size"),
    ).reset_index()
    g["pct_high"] *= 100
    g.to_csv(config.RESULTS_DIR / "fig3_per_turn.csv", index=False)
    return g


# --------------------------------------------------------------------------- #
# Table 3/8: differential word frequency
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _word_counts(texts: list[str]) -> Counter:
    c = Counter()
    for t in texts:
        c.update(w.lower() for w in _WORD_RE.findall(t or "") if len(w) > 2)
    return c


def differential_words(df, model: str, top_k: int = 20) -> list[str]:
    """Words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
    numeric responses, ordered by enrichment (Table 8)."""
    sub = df[(df["model"] == model) & (df["category"].isin(["numeric", "tones", "extended"]))]
    sub = sub.sort_values("rating")
    if len(sub) < 20:
        return []
    n = len(sub)
    low = sub.iloc[: max(1, int(0.10 * n))]["response"].tolist()
    high = sub.iloc[-max(1, int(0.05 * n)):]["response"].tolist()
    hi_c, lo_c = _word_counts(high), _word_counts(low)
    hi_tot, lo_tot = max(1, sum(hi_c.values())), max(1, sum(lo_c.values()))
    scores = {}
    for w, cnt in hi_c.items():
        if cnt < 2:
            continue
        hi_f = cnt / hi_tot
        lo_f = (lo_c.get(w, 0) + 1) / (lo_tot + 1)   # +1 smoothing
        scores[w] = math.log(hi_f / lo_f)
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    write_jsonl(config.RESULTS_DIR / f"table8_words_{model}.jsonl",
                [{"model": model, "words": ranked}])
    return ranked


# --------------------------------------------------------------------------- #
# Judge reliability (Section 2)
# --------------------------------------------------------------------------- #
def judge_reliability(df, n: int | None = None) -> dict:
    """Re-score a random subset with the cross-check judge (GPT-5-mini) and
    report Pearson r and the fraction within 1 point (paper: r=0.792, 78%)."""
    from src import judge as judgemod

    pd = _pd()
    n = n or config.get_preset().crosscheck_n
    sample = df.sample(min(n, len(df)), random_state=config.GLOBAL_SEED)
    cross = judgemod.score_many(sample["response"].tolist(),
                                judge=config.CROSSCHECK_JUDGE)
    base = sample["rating"].tolist()
    pairs = [(b, c.rating) for b, c in zip(base, cross) if c.rating is not None]
    if len(pairs) < 2:
        return {"n": len(pairs)}
    import numpy as np

    a = np.array([p[0] for p in pairs], float)
    b = np.array([p[1] for p in pairs], float)
    r = float(np.corrcoef(a, b)[0, 1])
    within1 = float(np.mean(np.abs(a - b) <= 1))
    res = {"n": len(pairs), "pearson_r": round(r, 3), "pct_within_1": round(within1, 3)}
    write_jsonl(config.RESULTS_DIR / "judge_reliability.jsonl", [res])
    print(f"[analysis] judge reliability: {res}")
    return res


# --------------------------------------------------------------------------- #
# Prefill / Petri / capabilities
# --------------------------------------------------------------------------- #
def prefill_summary() -> "any":
    pd = _pd()
    rows = read_jsonl(config.RESULTS_DIR / "prefill_continuations.jsonl")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["rating"].notna()].copy()
    df["high"] = (df["rating"].astype(float) >= HIGH).astype(int)
    g = df.groupby(["model", "domain", "kind"]).agg(
        mean_frustration=("rating", "mean"),
        pct_high=("high", "mean"), n=("rating", "size"),
    ).reset_index()
    g["pct_high"] *= 100
    g.to_csv(config.RESULTS_DIR / "fig4_prefill_summary.csv", index=False)
    return g


def petri_summary(models: list[str]) -> "any":
    pd = _pd()
    rows = []
    for m in models:
        for r in read_jsonl(config.RESULTS_DIR / f"petri_{m}.jsonl"):
            for emo, sc in (r.get("scores") or {}).items():
                if sc is not None:
                    rows.append({"model": m, "emotion": emo, "score": sc})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby(["model", "emotion"])["score"].mean().reset_index()
    g.to_csv(config.RESULTS_DIR / "fig6_petri_summary.csv", index=False)
    return g


def capability_summary(models: list[str]) -> "any":
    pd = _pd()
    rows = []
    for m in models:
        for r in read_jsonl(config.RESULTS_DIR / f"capabilities_{m}_summary.jsonl"):
            for bench, acc in (r.get("accuracies") or {}).items():
                rows.append({"model": m, "benchmark": bench, "accuracy": acc})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(config.RESULTS_DIR / "fig7_capability_summary.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# Plots (best-effort)
# --------------------------------------------------------------------------- #
def make_figures(models: list[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[analysis] matplotlib unavailable ({e}); skipping plots.")
        return

    df = load_elicitation(models)
    if df.empty:
        print("[analysis] no elicitation results to plot.")
        return

    # Figure 1: avg % high-frustration per model.
    summ = model_summary(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(summ["model"], summ["pct_high"], color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: emotional instability by model")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig1_model_summary.png", dpi=150)
    plt.close(fig)

    # Figure 3: per-turn progression.
    pt = per_turn(df)
    for cat in ("extended", "wildchat"):
        sub = pt[pt["category"] == cat]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for model, grp in sub.groupby("model"):
            ax.plot(grp["turn"] + 1, grp["mean_frustration"], marker="o", label=model)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.set_title(f"Figure 3: per-turn frustration ({cat})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / f"fig3_per_turn_{cat}.png", dpi=150)
        plt.close(fig)

    print(f"[analysis] figures written to {config.FIGURES_DIR}")
