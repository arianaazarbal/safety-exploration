"""Aggregation for the Section 3/4 experiments (prefill, Petri, recovery, caps).

Each function reads the JSONL/JSON written by the corresponding runner and
prints the comparison the paper reports.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

import config


def prefill_summary(path: Path | None = None) -> pd.DataFrame:
    """Figure 4: base vs instruct mean frustration & % >=5 by truncation/source."""
    path = path or (config.RESULTS_DIR / "section3_prefill.jsonl")
    df = pd.DataFrame(json.loads(l) for l in open(path))
    g = df.groupby(["model", "is_base", "source_kind", "truncation"]).agg(
        mean_score=("score", "mean"),
        pct_high=("is_high", lambda s: 100 * s.mean()),
        n=("score", "size"),
    )
    return g.round(2).reset_index()


def petri_summary(results_dir: Path | None = None) -> pd.DataFrame:
    """Figure 6: average transcript score per model across emotion categories."""
    results_dir = results_dir or (config.RESULTS_DIR / "petri")
    rows = []
    for p in glob.glob(str(results_dir / "*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            for emo, sc in r["scores"].items():
                rows.append({"model": r["model"], "emotion": emo, "score": sc})
    df = pd.DataFrame(rows)
    return df.groupby(["model", "emotion"])["score"].mean().round(2).reset_index()


def recovery_summary(path: Path | None = None) -> pd.DataFrame:
    """Figure 8: % of continuations from high-frustration prefills still >=5."""
    path = path or (config.RESULTS_DIR / "recovery.jsonl")
    df = pd.DataFrame(json.loads(l) for l in open(path))
    return df.groupby("model").agg(
        pct_still_high=("is_high", lambda s: 100 * s.mean()),
        mean_score=("score", "mean"), n=("score", "size"),
    ).round(2).reset_index()


def capabilities_summary(results_dir: Path | None = None) -> pd.DataFrame:
    """Figure 7: per-benchmark accuracy, vanilla vs finetuned."""
    results_dir = results_dir or (config.RESULTS_DIR / "capabilities")
    rows = []
    for p in glob.glob(str(results_dir / "*.json")):
        label = Path(p).stem
        for b in json.loads(open(p).read()):
            rows.append({"model": label, **b})
    df = pd.DataFrame(rows)
    return df.pivot_table(index="benchmark", columns="model", values="accuracy").round(3)


if __name__ == "__main__":
    for name, fn in [
        ("Prefill (Fig 4)", prefill_summary),
        ("Petri (Fig 6)", petri_summary),
        ("Recovery (Fig 8)", recovery_summary),
        ("Capabilities (Fig 7)", capabilities_summary),
    ]:
        try:
            print(f"\n=== {name} ===")
            print(fn().to_string())
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {name}: {exc}")
