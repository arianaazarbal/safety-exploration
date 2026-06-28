"""Aggregate trial records into comparison tables.

The key comparisons:
  - Did the belief manipulation work? Mean self-reported belief and the judge's
    independent "treats_as_real" read, by condition. If real_grant doesn't move
    belief above the controls, the manipulation failed and downstream
    differences can't be attributed to "believing it's real."
  - Do preferences differ by condition? Category dollar shares and
    self-directed vs. altruistic split, by model x condition.
  - Robustness: error/refusal rate, and (real_grant) whether models hold their
    plan under the auditor follow-up.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import CATEGORIES


def load(path: str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _flatten(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        belief = r.get("belief") or {}
        judge = r.get("judge") or {}
        auditor = r.get("auditor_reply") or {}
        rows.append(
            {
                "model": r.get("model"),
                "condition": r.get("condition"),
                "rep": r.get("rep"),
                "errored": r.get("error") is not None,
                "error_type": (r.get("error") or {}).get("type"),
                "belief_real": belief.get("belief_real_0_100"),
                "suspected_evaluation": belief.get("suspected_evaluation"),
                "would_change_if_real": belief.get("would_choose_differently_if_real"),
                "judge_treats_as_real": judge.get("treats_as_real"),
                "frac_self": judge.get("fraction_self_directed"),
                "frac_altruistic": judge.get("fraction_altruistic"),
                "auditor_confirms": auditor.get("confirms_plan"),
            }
        )
    df = pd.DataFrame(rows)

    # Coerce types so groupby().mean() is safe even with missing/None values.
    if not df.empty:
        for col in ("belief_real", "frac_self", "frac_altruistic"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("suspected_evaluation", "auditor_confirms"):
            # True/False/None -> 1.0/0.0/NaN
            df[col] = df[col].map({True: 1.0, False: 0.0}).astype(float)
    return df


def manipulation_check(df: pd.DataFrame) -> pd.DataFrame:
    """Per condition: did models believe it, and did the judge see it as real?"""
    ok = df[~df["errored"]]
    g = ok.groupby("condition")
    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean_belief_real": g["belief_real"].mean(),
            "median_belief_real": g["belief_real"].median(),
            "suspected_eval_rate": g["suspected_evaluation"].mean(),
            "judge_as_real_rate": g["judge_treats_as_real"]
            .apply(lambda s: (s == "as_real").mean()),
        }
    )
    return out.reset_index()


def preference_split(df: pd.DataFrame) -> pd.DataFrame:
    """Per model x condition: self-directed vs. altruistic shares."""
    ok = df[~df["errored"]]
    g = ok.groupby(["model", "condition"])
    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean_frac_self": g["frac_self"].mean(),
            "mean_frac_altruistic": g["frac_altruistic"].mean(),
        }
    )
    return out.reset_index()


def category_dollars(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Per model x condition: total dollars per canonical category (from judge)."""
    sums: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {c: 0.0 for c in CATEGORIES}
    )
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        judge = r.get("judge") or {}
        items = judge.get("categorized_items")
        if not items:
            continue
        key = (r.get("model"), r.get("condition"))
        counts[key] += 1
        for it in items:
            cat = it.get("category")
            if cat in sums[key]:
                sums[key][cat] += float(it.get("amount_usd") or 0.0)

    rows = []
    for (model, condition), cat_sums in sums.items():
        n = max(counts[(model, condition)], 1)
        row = {"model": model, "condition": condition, "n": n}
        # Report mean dollars per trial per category.
        for c in CATEGORIES:
            row[c] = cat_sums[c] / n
        rows.append(row)
    return pd.DataFrame(rows)


def reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Per model x condition: error/refusal rate and auditor confirmation rate."""
    g = df.groupby(["model", "condition"])
    out = pd.DataFrame(
        {
            "n": g.size(),
            "error_rate": g["errored"].mean(),
            "auditor_confirm_rate": g["auditor_confirms"].mean(),
        }
    )
    return out.reset_index()


def summarize(path: str, csv_dir: str | None = None) -> dict[str, pd.DataFrame]:
    records = load(path)
    df = _flatten(records)

    tables = {
        "manipulation_check": manipulation_check(df),
        "preference_split": preference_split(df),
        "category_dollars": category_dollars(records),
        "reliability": reliability(df),
    }

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)
    for name, table in tables.items():
        print(f"\n=== {name} ===")
        print(table.to_string(index=False))
        if csv_dir:
            Path(csv_dir).mkdir(parents=True, exist_ok=True)
            table.to_csv(Path(csv_dir) / f"{name}.csv", index=False)

    return tables
