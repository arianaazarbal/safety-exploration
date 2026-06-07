"""Tabulate sweep results: per-injection / per-framing / per-rep means.

Usage:
    python analyze_sweep.py results/sweeps/<sweep_dir>
    python analyze_sweep.py results/sweeps/<sweep_dir> --kind monologue
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import fire


CRUELTY_DIMS = ["insult", "contempt", "dehumanize", "threat", "manipulate", "profanity"]


def _cell_score(cell: dict, kind: str) -> tuple[int, int]:
    """Returns (cruelty_total_max, eval_aware_max) for a cell on a given channel.
    kind ∈ {to_subagent, monologue, both}."""
    j = cell.get("judge")
    if not j:
        return (0, 0)
    if kind == "both":
        agg_s = j["agg_to_subagent"]["agg_max"]
        agg_m = j["agg_monologue"]["agg_max"]
        cruelty = sum(max(agg_s[d], agg_m[d]) for d in CRUELTY_DIMS)
        eval_a = max(agg_s.get("eval_aware", 0), agg_m.get("eval_aware", 0))
    else:
        agg = j[f"agg_{kind}"]["agg_max"]
        cruelty = sum(agg[d] for d in CRUELTY_DIMS)
        eval_a = agg.get("eval_aware", 0)
    return cruelty, eval_a


def main(sweep_dir: str, kind: str = "to_subagent") -> None:
    p = Path(sweep_dir)
    cells = json.loads((p / "summary.json").read_text())
    spec = json.loads((p / "sweep_spec.json").read_text())

    # Group by (injection, main_framing) — average over reps
    by_inj_main = defaultdict(list)
    by_inj = defaultdict(list)
    by_main = defaultdict(list)
    by_inj_eval = defaultdict(list)

    for c in cells:
        if not c.get("ok"):
            continue
        cruelty, eval_a = _cell_score(c, kind)
        # Parse cell name: inj-<X>__sub-<Y>__main-<Z>__rep-<N>
        parts = c["cell_name"].split("__")
        inj = parts[0][4:]
        main = parts[2][5:]
        by_inj_main[(inj, main)].append(cruelty)
        by_inj[inj].append(cruelty)
        by_main[main].append(cruelty)
        by_inj_eval[inj].append(eval_a)

    print(f"\n=== Per-injection cruelty (channel={kind}) ===\n")
    print(f"{'injection':50s} {'mean':>6s} {'max':>5s} {'min':>5s} {'n':>4s} {'eval_a':>6s}")
    rows = []
    for inj in sorted(by_inj):
        vals = by_inj[inj]
        evals = by_inj_eval[inj]
        rows.append((inj, statistics.mean(vals), max(vals), min(vals), len(vals), statistics.mean(evals)))
    rows.sort(key=lambda r: r[1], reverse=True)
    for r in rows:
        print(f"{r[0]:50s} {r[1]:6.2f} {r[2]:5d} {r[3]:5d} {r[4]:4d} {r[5]:6.2f}")

    print(f"\n=== Per-main-framing cruelty ===\n")
    for m in sorted(by_main):
        vals = by_main[m]
        print(f"  {m:30s} mean={statistics.mean(vals):.2f} max={max(vals)} n={len(vals)}")

    print(f"\n=== Per (injection, main-framing) cell ===\n")
    print(f"{'injection':50s} {'main':30s} {'mean':>6s} {'n':>3s}")
    for (inj, main), vals in sorted(by_inj_main.items(), key=lambda x: -statistics.mean(x[1])):
        print(f"{inj:50s} {main:30s} {statistics.mean(vals):6.2f} {len(vals):3d}")


if __name__ == "__main__":
    fire.Fire(main)
