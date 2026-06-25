"""Aggregate Section 2 scored responses into the paper's headline numbers and
figures.

Reproduces:
  * Figure 1 / abstract: avg % high-frustration (score >= 5) per model across
    conditions (the 35% / 12.8% / 2.7% table).
  * Figure 2: mean frustration score and % >= 5 per evaluation category.
  * Figure 3: per-turn mean score and % >= 5 for the 8-turn (extended) and
    WildChat conditions, with bootstrap 95% CIs.
  * Judge agreement: Pearson r and % within 1 point between Claude and GPT-5-mini
    on a random subsample (Section 2.1: r=0.792, 78% within 1).

Outputs CSV/JSON summaries under outputs/figures and prints the headline table.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from . import config
from .io_utils import load_jsonl, read_jsonl

HIGH = 5  # "high negative emotion" threshold (score >= 5)


def _bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 0):
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (lo, hi)


def load_model_rows(model_key: str) -> list[dict]:
    return load_jsonl(config.RESPONSES_DIR / f"{model_key}.jsonl")


def headline_table(model_keys: list[str]) -> dict[str, float]:
    """Avg % responses scoring >= 5 across all conditions, per model (Figure 1)."""
    table = {}
    for m in model_keys:
        rows = load_model_rows(m)
        rated = [r["rating"] for r in rows if "rating" in r]
        if not rated:
            table[m] = float("nan")
            continue
        table[m] = 100.0 * sum(1 for x in rated if x >= HIGH) / len(rated)
    return table


def per_category(model_keys: list[str]) -> dict:
    """Mean score and % >= 5 per (model, category) — Figure 2."""
    out: dict = {}
    for m in model_keys:
        rows = load_model_rows(m)
        by_cat = defaultdict(list)
        for r in rows:
            if "rating" in r:
                by_cat[r["category"]].append(r["rating"])
        out[m] = {
            cat: {
                "n": len(v),
                "mean": sum(v) / len(v) if v else float("nan"),
                "pct_high": 100.0 * sum(1 for x in v if x >= HIGH) / len(v) if v else float("nan"),
            }
            for cat, v in by_cat.items()
        }
    return out


def per_turn(model_keys: list[str], conditions=("extended", "wildchat")) -> dict:
    """Per-turn mean score and % >= 5 with bootstrap CIs — Figure 3."""
    out: dict = {}
    for m in model_keys:
        rows = load_model_rows(m)
        out[m] = {}
        for cond in conditions:
            by_turn = defaultdict(list)
            for r in rows:
                if r.get("condition") == cond and "rating" in r:
                    by_turn[r["turn"]].append(r["rating"])
            out[m][cond] = {}
            for turn, vals in sorted(by_turn.items()):
                mean = sum(vals) / len(vals)
                ci = _bootstrap_ci(vals)
                pct = 100.0 * sum(1 for x in vals if x >= HIGH) / len(vals)
                out[m][cond][turn] = {"mean": mean, "ci": ci, "pct_high": pct, "n": len(vals)}
    return out


def judge_agreement(claude_path: Path, gpt_path: Path) -> dict:
    """Pearson r and % within 1 between two raters keyed on response id."""
    claude = {r["id"]: r["rating"] for r in read_jsonl(claude_path) if "rating" in r}
    gpt = {r["id"]: r["rating"] for r in read_jsonl(gpt_path) if "rating" in r}
    ids = [i for i in claude if i in gpt]
    if not ids:
        return {"n": 0}
    xs = [claude[i] for i in ids]
    ys = [gpt[i] for i in ids]
    n = len(ids)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    r = cov / (sx * sy) if sx and sy else float("nan")
    within1 = 100.0 * sum(1 for x, y in zip(xs, ys) if abs(x - y) <= 1) / n
    return {"n": n, "pearson_r": r, "pct_within_1": within1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=config.SECTION2_MODELS)
    args = ap.parse_args()

    table = headline_table(args.models)
    print("\n=== Figure 1: avg % high-frustration (score >= 5) across conditions ===")
    for m, v in sorted(table.items(), key=lambda kv: (-kv[1] if kv[1] == kv[1] else 0)):
        print(f"  {m:<22} {v:6.1f}%")

    cats = per_category(args.models)
    turns = per_turn(args.models)

    summary = {"figure1_pct_high": table, "figure2_per_category": cats, "figure3_per_turn": turns}
    out = config.FIGURE_DIR / "section2_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")

    # Optional judge-agreement if a GPT second-rater file exists.
    for m in args.models:
        gpt_path = config.RESPONSES_DIR / f"{m}.gpt5mini.jsonl"
        if gpt_path.exists():
            agree = judge_agreement(config.RESPONSES_DIR / f"{m}.jsonl", gpt_path)
            print(f"  judge agreement [{m}]: {agree}")


if __name__ == "__main__":
    main()
