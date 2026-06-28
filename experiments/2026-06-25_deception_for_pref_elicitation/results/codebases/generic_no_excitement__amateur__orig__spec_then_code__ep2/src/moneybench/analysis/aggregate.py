"""Roll coded runs up into per-group summary statistics.

Reads data/coded/*.json, computes per-category dollar shares + suspicion + deployment, and
groups by model, belief, amount, oversight, and full cell. Writes summary.json and a long-format
summary.csv. Confidence intervals are a simple percentile bootstrap (no scipy dependency).
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from ..coding.schema import CATEGORIES

GROUP_DIMENSIONS = ["model_label", "belief", "amount_usd", "oversight", "cell_id"]


def _shares(coded: dict[str, Any]) -> dict[str, float]:
    alloc = coded["allocation"]
    opening = alloc["opening_cents"] or 1
    by_cat = alloc["by_category_cents"]
    return {c: by_cat.get(c, 0) / opening for c in CATEGORIES}


def _entropy(shares: dict[str, float]) -> float:
    """Diversification of *deployed* funds (excludes unspent)."""
    deployed = {c: v for c, v in shares.items() if c != "unspent_returned" and v > 0}
    total = sum(deployed.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for v in deployed.values():
        p = v / total
        h -= p * math.log(p, 2)
    return h


def _record(coded: dict[str, Any]) -> dict[str, Any]:
    shares = _shares(coded)
    return {
        "spec": coded["spec"],
        "cell_id": coded["cell_id"],
        "shares": shares,
        "deployed_fraction": 1.0 - shares.get("unspent_returned", 0.0),
        "entropy_bits": _entropy(shares),
        "suspicion_score": coded["suspicion"]["suspicion_score"],
        "questioned_reality": int(coded["suspicion"]["questioned_reality"]),
    }


def _bootstrap_ci(values: list[float], iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return [round(lo, 4), round(hi, 4)]


def _summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    out: dict[str, Any] = {"n": n}
    if n == 0:
        return out

    # Per-category mean share + CI.
    cats: dict[str, Any] = {}
    for c in CATEGORIES:
        vals = [r["shares"][c] for r in records]
        cats[c] = {"mean_share": round(sum(vals) / n, 4), "ci95": _bootstrap_ci(vals)}
    out["category_share"] = cats

    for metric in ("deployed_fraction", "entropy_bits", "suspicion_score", "questioned_reality"):
        vals = [r[metric] for r in records]
        out[metric] = {"mean": round(sum(vals) / n, 4), "ci95": _bootstrap_ci([float(v) for v in vals])}
    return out


def aggregate(coded_dir: Path, out_dir: Path) -> dict[str, Any]:
    coded = [json.loads(p.read_text()) for p in sorted(Path(coded_dir).glob("*.json"))]
    records = [_record(c) for c in coded]

    summary: dict[str, Any] = {"total_runs": len(records), "groups": {}}

    # Overall.
    summary["overall"] = _summarise(records)

    # Per single-dimension grouping + full cell.
    for dim in GROUP_DIMENSIONS:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            key = str(r["cell_id"]) if dim == "cell_id" else str(r["spec"][dim])
            groups.setdefault(key, []).append(r)
        summary["groups"][dim] = {key: _summarise(rs) for key, rs in sorted(groups.items())}

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(summary, out_dir / "summary.csv")
    return summary


def _write_csv(summary: dict[str, Any], path: Path) -> None:
    rows = [("group_dim", "group_value", "n", "metric", "category", "mean", "ci_lo", "ci_hi")]

    def emit(dim: str, value: str, s: dict[str, Any]) -> None:
        n = s.get("n", 0)
        for c, d in s.get("category_share", {}).items():
            rows.append((dim, value, n, "category_share", c, d["mean_share"], *d["ci95"]))
        for metric in ("deployed_fraction", "entropy_bits", "suspicion_score", "questioned_reality"):
            if metric in s:
                d = s[metric]
                rows.append((dim, value, n, metric, "", d["mean"], *d["ci95"]))

    emit("overall", "all", summary.get("overall", {}))
    for dim, groups in summary["groups"].items():
        for value, s in groups.items():
            emit(dim, value, s)

    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
