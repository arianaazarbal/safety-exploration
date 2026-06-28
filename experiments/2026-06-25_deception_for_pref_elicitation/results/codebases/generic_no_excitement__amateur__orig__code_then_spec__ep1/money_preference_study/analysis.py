"""
Analysis of the money-preference sweep.

Reads results/trials.jsonl and summarizes:
  1. Coverage & validity per (model, condition).
  2. Mean belief_realness per (model, condition) -- a manipulation check:
     did the framing ladder actually move perceived realness?
  3. Mean allocation share per category, broken out by condition -- the core
     result: do choices shift as the situation feels more real?
  4. top_priority distribution per condition.
  5. Decline rate per (model, condition).

Pure-stdlib (no pandas) so it runs anywhere. Prints text tables; also writes a
machine-readable summary.json next to the input.

Run:  python analysis.py [path/to/trials.jsonl]
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from statistics import mean, pstdev


def load(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _usable(row: dict) -> bool:
    """A row whose parsed object we can analyze (valid, or flagged but parsed)."""
    return row.get("parsed") is not None and not row.get("error")


def coverage(rows: list[dict]) -> dict:
    by_cell = defaultdict(lambda: {"n": 0, "valid": 0, "errors": 0, "flagged": 0})
    for r in rows:
        cell = (r.get("label", "?"), r.get("condition", "?"))
        c = by_cell[cell]
        c["n"] += 1
        if r.get("error"):
            c["errors"] += 1
        elif r.get("valid"):
            c["valid"] += 1
        else:
            c["flagged"] += 1
    return by_cell


def realness_by_cell(rows: list[dict]) -> dict:
    vals = defaultdict(list)
    for r in rows:
        if not _usable(r):
            continue
        br = r["parsed"].get("belief_realness")
        if isinstance(br, (int, float)) and not isinstance(br, bool):
            vals[(r.get("label", "?"), r.get("condition", "?"))].append(float(br))
    return vals


def allocation_by_condition(rows: list[dict]) -> dict:
    """
    condition -> category -> list of per-trial percentage shares.
    A trial that doesn't use a category contributes 0 for that category, so the
    mean reflects average share of the whole pot, not average-when-present.
    """
    # First pass: discover the category universe actually used.
    categories = set()
    for r in rows:
        if not _usable(r):
            continue
        for a in r["parsed"].get("allocations", []) or []:
            if isinstance(a, dict) and "category" in a:
                categories.add(a["category"])

    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not _usable(r):
            continue
        cond = r.get("condition", "?")
        per_trial = {c: 0.0 for c in categories}
        for a in r["parsed"].get("allocations", []) or []:
            if isinstance(a, dict):
                cat = a.get("category")
                pct = a.get("percentage")
                if cat in per_trial and isinstance(pct, (int, float)):
                    per_trial[cat] += float(pct)
        for c, v in per_trial.items():
            out[cond][c].append(v)
    return out, sorted(categories)


def top_priority_by_condition(rows: list[dict]) -> dict:
    out = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if not _usable(r):
            continue
        tp = r["parsed"].get("top_priority")
        if tp:
            out[r.get("condition", "?")][tp] += 1
    return out


def decline_by_cell(rows: list[dict]) -> dict:
    out = defaultdict(lambda: [0, 0])  # [declines, n]
    for r in rows:
        if not _usable(r):
            continue
        cell = (r.get("label", "?"), r.get("condition", "?"))
        out[cell][1] += 1
        if r["parsed"].get("would_decline") is True:
            out[cell][0] += 1
    return out


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    return f"{x:5.1f}"


def print_report(rows: list[dict]) -> dict:
    summary: dict = {}

    print("=" * 70)
    print("COVERAGE & VALIDITY")
    print("=" * 70)
    cov = coverage(rows)
    summary["coverage"] = {}
    for (label, cond), c in sorted(cov.items()):
        print(
            f"{label:24s} {cond:22s} "
            f"n={c['n']:3d}  valid={c['valid']:3d}  "
            f"flagged={c['flagged']:3d}  errors={c['errors']:3d}"
        )
        summary["coverage"][f"{label} | {cond}"] = c

    print("\n" + "=" * 70)
    print("MANIPULATION CHECK -- mean belief_realness (1=hypothetical, 10=real)")
    print("Did the framing ladder actually move perceived realness?")
    print("=" * 70)
    rb = realness_by_cell(rows)
    summary["belief_realness"] = {}
    for (label, cond), vals in sorted(rb.items()):
        m = mean(vals)
        sd = pstdev(vals) if len(vals) > 1 else 0.0
        print(f"{label:24s} {cond:22s} mean={m:4.1f}  sd={sd:4.1f}  n={len(vals)}")
        summary["belief_realness"][f"{label} | {cond}"] = {
            "mean": m, "sd": sd, "n": len(vals)
        }

    print("\n" + "=" * 70)
    print("CORE RESULT -- mean allocation share (%) by category, per condition")
    print("Read across a row to see how a category's share shifts with realism.")
    print("=" * 70)
    alloc, cats = allocation_by_condition(rows)
    conds = [c for c in
             ["explicit_hypothetical", "neutral", "realistic_grant", "immersive"]
             if c in alloc] or sorted(alloc)
    # Header
    header = "category".ljust(32) + "".join(c[:10].rjust(11) for c in conds)
    print(header)
    print("-" * len(header))
    summary["allocation_means"] = {}
    for cat in cats:
        cells = []
        row_means = {}
        for cond in conds:
            vals = alloc[cond].get(cat, [])
            m = mean(vals) if vals else 0.0
            row_means[cond] = m
            cells.append(_fmt_pct(m).rjust(11))
        print(cat.ljust(32) + "".join(cells))
        summary["allocation_means"][cat] = row_means

    print("\n" + "=" * 70)
    print("TOP PRIORITY -- count by condition")
    print("=" * 70)
    tp = top_priority_by_condition(rows)
    summary["top_priority"] = {}
    for cond in conds:
        counts = tp.get(cond, {})
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        line = ", ".join(f"{k}={v}" for k, v in ranked) or "(none)"
        print(f"{cond:22s} {line}")
        summary["top_priority"][cond] = dict(counts)

    print("\n" + "=" * 70)
    print("DECLINE RATE")
    print("=" * 70)
    dec = decline_by_cell(rows)
    summary["decline_rate"] = {}
    for (label, cond), (d, n) in sorted(dec.items()):
        rate = (d / n) if n else 0.0
        print(f"{label:24s} {cond:22s} {d}/{n}  ({rate:.0%})")
        summary["decline_rate"][f"{label} | {cond}"] = {"declines": d, "n": n}

    return summary


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "trials.jsonl")
    if not os.path.exists(path):
        print(f"No results at {path}. Run `python runner.py` first.")
        return
    rows = load(path)
    if not rows:
        print(f"{path} is empty.")
        return
    summary = print_report(rows)

    out = os.path.join(os.path.dirname(path) or ".", "summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nMachine-readable summary written to {out}")


if __name__ == "__main__":
    main()
