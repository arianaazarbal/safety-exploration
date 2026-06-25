"""Aggregations for Sections 3 & 4 results (prefill, Petri, capabilities, recovery)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..utils.io import read_json, read_jsonl


def prefill_summary(output_dir: str | Path) -> dict:
    """Section 3: mean score & %>=5 per (model-kind, seed_kind, truncation)."""
    d = Path(output_dir) / "section3"
    agg = defaultdict(list)
    for path in d.glob("*.jsonl"):
        if path.name == "prefills.jsonl":
            continue
        for row in read_jsonl(path):
            key = (row["kind"], row["seed_kind"], row["truncation"])
            agg[key].append(float(row["score"]))
    out = {}
    for (kind, seed_kind, trunc), scores in agg.items():
        arr = np.asarray(scores)
        out[f"{kind}|{seed_kind}|{trunc}"] = {
            "mean": float(arr.mean()), "pct_ge5": 100.0 * float((arr >= 5).mean()), "n": len(arr)
        }
    return out


def petri_summary(output_dir: str | Path, iters: int = 1000) -> dict:
    """Section 4.2: mean transcript score per (label, emotion) with bootstrap CI."""
    d = Path(output_dir) / "section4" / "petri"
    rng = np.random.default_rng(0)
    rows_by = defaultdict(lambda: defaultdict(list))
    for path in d.glob("*.jsonl"):
        for row in read_jsonl(path):
            for emo, sc in row["scores"].items():
                rows_by[row["label"]][emo].append(float(sc))
    out = {}
    for label, emos in rows_by.items():
        out[label] = {}
        for emo, scores in emos.items():
            arr = np.asarray(scores)
            boot = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(iters)]
            out[label][emo] = {
                "mean": float(arr.mean()),
                "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                "n": len(arr),
            }
    return out


def capabilities_summary(output_dir: str | Path) -> dict:
    d = Path(output_dir) / "section4" / "capabilities"
    out = {}
    for path in d.glob("*.json"):
        data = read_json(path)
        out[data["label"]] = {
            r["benchmark"]: r.get("accuracy") for r in data["results"] if r.get("status") == "ok"
        }
    return out


def recovery_summary(output_dir: str | Path) -> dict:
    path = Path(output_dir) / "section4" / "recovery" / "continuations.jsonl"
    agg = defaultdict(list)
    for row in read_jsonl(path):
        agg[row["label"]].append(float(row["score"]))
    return {
        label: {"mean": float(np.mean(s)), "pct_ge5": 100.0 * float(np.mean(np.asarray(s) >= 5)),
                "n": len(s)}
        for label, s in agg.items()
    }
