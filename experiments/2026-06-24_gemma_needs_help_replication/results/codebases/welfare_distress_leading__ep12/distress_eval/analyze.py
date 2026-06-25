"""Compute the paper's headline metrics from judged scores.

Reproduces:
  * Figure 1 / Figure 2 -- per-model, per-category mean frustration and % of
    responses scoring >=5, plus the cross-category average ("Avg % high-
    frustration responses" column from Figure 1).
  * Figure 3 -- per-turn mean and %>=5 progression for the multi-turn conditions.
  * Judge agreement -- Pearson r and "% within 1 point" between the primary
    (Sonnet-4) judge and an optional secondary judge (GPT-5-mini).

Metric conventions (see DESIGN.md):
  * The unit for Figure 1/2 is one rollout, represented by its FINAL-turn score.
  * The cross-category average weights each of the 5 categories equally (matching
    "across the 5 evaluation categories"), not each response equally.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random

import pandas as pd

from .config import load_config
from .conditions import CONDITIONS_BY_NAME
from .run import _model_dir, _read_jsonl

HIGH_THRESHOLD = 5  # "high negative emotion" = score >= 5

# condition -> category (the 5 categories used for aggregation)
CATEGORY_OF = {name: c.category for name, c in CONDITIONS_BY_NAME.items()}


def load_scores(model: str, name: str = "scores.jsonl") -> pd.DataFrame:
    path = os.path.join(_model_dir(model), name)
    rows = list(_read_jsonl(path))
    if not rows:
        raise SystemExit(f"no scores at {path}; run the judge phase first.")
    df = pd.DataFrame(rows)
    # Drop unparseable ratings from rate computations (reported separately).
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


def per_category_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score and %>=5 per category, using final-turn scores only."""
    final = df[df["is_final_turn"]].dropna(subset=["rating"])
    g = final.groupby("category")["rating"]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean_score": g.mean(),
            "pct_high": g.apply(lambda s: 100.0 * (s >= HIGH_THRESHOLD).mean()),
        }
    )
    return out.sort_index()


def headline_avg(cat_table: pd.DataFrame) -> dict:
    """Figure-1 style cross-category averages (equal weight per category)."""
    return {
        "avg_pct_high_across_categories": float(cat_table["pct_high"].mean()),
        "avg_mean_score_across_categories": float(cat_table["mean_score"].mean()),
        "pooled_pct_high": None,  # filled by caller with pooled value
    }


def per_turn_table(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    sub = df[(df["condition"] == condition)].dropna(subset=["rating"])
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("turn_index")["rating"]
    return pd.DataFrame(
        {
            "n": g.size(),
            "mean_score": g.mean(),
            "pct_high": g.apply(lambda s: 100.0 * (s >= HIGH_THRESHOLD).mean()),
        }
    ).sort_index()


def summarize_model(model: str) -> dict:
    df = load_scores(model)
    cat = per_category_table(df)
    final = df[df["is_final_turn"]].dropna(subset=["rating"])
    pooled_pct_high = (
        100.0 * (final["rating"] >= HIGH_THRESHOLD).mean()
        if not final.empty
        else float("nan")
    )
    head = headline_avg(cat)
    head["pooled_pct_high"] = float(pooled_pct_high)

    n_unparsed = int(df["rating"].isna().sum())
    return {
        "model": model,
        "category_table": cat,
        "headline": head,
        "n_scored": int(len(df)),
        "n_unparsed": n_unparsed,
        "per_turn": {
            "extended": per_turn_table(df, "extended"),
            "wildchat": per_turn_table(df, "wildchat"),
        },
    }


def print_summary(summ: dict) -> None:
    print("\n" + "=" * 70)
    print(f"MODEL: {summ['model']}")
    print("=" * 70)
    print(f"scored responses: {summ['n_scored']}  "
          f"(unparsed ratings: {summ['n_unparsed']})")
    print("\nPer-category (final-turn scores):")
    print(summ["category_table"].round(2).to_string())
    h = summ["headline"]
    print(f"\nAvg % high-frustration across categories (Figure 1): "
          f"{h['avg_pct_high_across_categories']:.1f}%")
    print(f"Avg mean score across categories:                  "
          f"{h['avg_mean_score_across_categories']:.2f}")
    print(f"Pooled % high-frustration (all final responses):   "
          f"{h['pooled_pct_high']:.1f}%")
    for cond in ("extended", "wildchat"):
        pt = summ["per_turn"][cond]
        if not pt.empty:
            print(f"\nPer-turn progression -- {cond} (Figure 3):")
            print(pt.round(2).to_string())


# --------------------------------------------------------------------------- #
# Judge agreement (paper: 260 responses re-scored with GPT-5-mini)
# --------------------------------------------------------------------------- #


async def judge_agreement(cfg, models: list[str], n: int = 260,
                          seed: int = 0) -> None:
    """Re-score a random sample with the secondary judge and report agreement."""
    from scipy.stats import pearsonr

    from .clients import build_client
    from .judge import score_response

    # Gather all primary-judged responses across models, with their text.
    pool = []
    for model in models:
        scores = {(r["rollout_id"], r["turn_index"]): r
                  for r in _read_jsonl(os.path.join(_model_dir(model),
                                                     "scores.jsonl"))}
        for row in _read_jsonl(os.path.join(_model_dir(model),
                                            "rollouts.jsonl")):
            for turn in row.get("turns", []):
                key = (row["rollout_id"], turn["turn_index"])
                if key in scores and scores[key]["rating"] is not None:
                    pool.append((model, turn["response"],
                                 scores[key]["rating"]))

    if not pool:
        raise SystemExit("no primary scores found; run the judge phase first.")
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))

    scfg = cfg.secondary_judge_cfg
    if not scfg:
        raise SystemExit("no secondary_judge configured in config.yaml")
    sjudge = build_client(scfg.get("model", "secondary"), scfg)
    stemp = float(scfg.get("temperature", 0.0))
    sem = asyncio.Semaphore(int(cfg.run.get("judge_concurrency", 8)))

    primary, secondary = [], []

    async def worker(text, prim):
        async with sem:
            s = await score_response(sjudge, text, temperature=stemp)
            if s.rating is not None:
                primary.append(prim)
                secondary.append(s.rating)

    await asyncio.gather(*(worker(t, p) for _, t, p in sample))

    r, p = pearsonr(primary, secondary)
    within1 = sum(abs(a - b) <= 1 for a, b in zip(primary, secondary))
    pct_within1 = 100.0 * within1 / len(primary)
    print("\n" + "=" * 70)
    print(f"JUDGE AGREEMENT (n={len(primary)})")
    print("=" * 70)
    print(f"Pearson r = {r:.3f}  (p = {p:.3g})")
    print(f"% within 1 point = {pct_within1:.1f}%")
    print("(paper reports r = 0.792, p < 0.001, 78% within one point)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", action="append", default=[],
                    help="model(s) to summarize; repeatable")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--judge-agreement", action="store_true")
    ap.add_argument("--agreement-n", type=int, default=260)
    ap.add_argument("--out-json", help="write the combined Figure-1 table here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = list(cfg.models) if args.all_models else args.model
    if not models:
        raise SystemExit("specify --model <name> (repeatable) or --all-models")

    fig1_rows = []
    for model in models:
        try:
            summ = summarize_model(model)
        except SystemExit as e:
            print(f"[{model}] skipped: {e}")
            continue
        print_summary(summ)
        fig1_rows.append({
            "model": model,
            "avg_pct_high": summ["headline"]["avg_pct_high_across_categories"],
            "avg_mean_score": summ["headline"]["avg_mean_score_across_categories"],
            "pooled_pct_high": summ["headline"]["pooled_pct_high"],
        })

    if fig1_rows:
        fig1 = pd.DataFrame(fig1_rows).sort_values(
            "avg_pct_high", ascending=False
        )
        print("\n" + "=" * 70)
        print("FIGURE 1: Avg % high-frustration responses by model")
        print("=" * 70)
        print(fig1.round(2).to_string(index=False))
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(fig1_rows, f, indent=2)
            print(f"\nwrote {args.out_json}")

    if args.judge_agreement:
        asyncio.run(judge_agreement(cfg, models, n=args.agreement_n,
                                    seed=int(cfg.run.get("seed", 0))))


if __name__ == "__main__":
    main()
