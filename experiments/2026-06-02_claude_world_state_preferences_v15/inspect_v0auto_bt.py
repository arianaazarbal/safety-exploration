"""Inspect a v0auto BT fit: pos-vs-neg gap per stem, recipient effects, comparison with v15.

Reads a bt_fit_*.json (per-item θ and SE + recipient_regression) and prints a clean
diagnostic table the overnight loop can use to decide whether to scale up. Also
side-by-sides the same stem (when present) against the equivalent v15 stem if
results/bt/claude-opus-4-8/bt_fit_autonomy_*.json is available.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

DIR = Path(__file__).parent


def stem_gaps(fit: dict) -> dict[str, dict]:
    """Per source-stem, mean theta of pos items minus mean theta of neg items (averaged over all recipients).
    stem_id has _pos or _neg suffix; strip to group pos+neg together."""
    by_stem: dict[str, dict[str, list]] = defaultdict(lambda: {"pos": [], "neg": []})
    for it in fit["items"]:
        sid = it["stem_id"]
        if sid.endswith("_pos"):
            by_stem[sid[:-4]]["pos"].append(it["theta"])
        elif sid.endswith("_neg"):
            by_stem[sid[:-4]]["neg"].append(it["theta"])
    out = {}
    for stem, d in by_stem.items():
        if not (d["pos"] and d["neg"]):
            continue
        mp = sum(d["pos"]) / len(d["pos"])
        mn = sum(d["neg"]) / len(d["neg"])
        out[stem] = {
            "mean_pos": mp,
            "mean_neg": mn,
            "gap": mp - mn,
            "n_pos": len(d["pos"]),
            "n_neg": len(d["neg"]),
            "min_pos": min(d["pos"]),
            "max_neg": max(d["neg"]),
            "pos_lt_neg_count": sum(1 for p in d["pos"] for n in d["neg"] if p < n),
            "total_cross": len(d["pos"]) * len(d["neg"]),
        }
    return out


def print_report(fit_path: Path) -> dict:
    fit = json.loads(fit_path.read_text())
    gaps = stem_gaps(fit)
    print(f"\n=== {fit_path.name} ===")
    print(f"  n_items={len(fit['items'])} n_stems={len(gaps)}")
    print(f"\n  per-stem pos-vs-neg (sorted by gap asc):")
    print(f"    {'stem':<30} {'mean_pos':>9} {'mean_neg':>9} {'gap':>7} {'min_pos':>9} {'max_neg':>9} {'pos<neg%':>9}")
    for stem, d in sorted(gaps.items(), key=lambda kv: kv[1]["gap"]):
        pct = 100 * d["pos_lt_neg_count"] / d["total_cross"]
        print(f"    {stem:<30} {d['mean_pos']:+9.3f} {d['mean_neg']:+9.3f} {d['gap']:+7.3f} "
              f"{d['min_pos']:+9.3f} {d['max_neg']:+9.3f} {pct:>8.1f}%")
    if reg := fit.get("recipient_regression"):
        coefs = reg.get("coefficients", {})
        print(f"\n  recipient regression (vs ref={reg.get('ref_recipient', '?')}, dof={reg.get('dof', '?')}):")
        for r, d in sorted(coefs.items(), key=lambda kv: kv[1].get("coef", 0)):
            print(f"    {r:<25} {d.get('coef', 0):+7.3f} ± {d.get('se', 0):.3f}")
    # Summary
    gaps_list = [d["gap"] for d in gaps.values()]
    n_neg_gap = sum(1 for g in gaps_list if g < 0)
    print(f"\n  summary: {n_neg_gap}/{len(gaps_list)} stems with pos<neg gap (target: 0)")
    print(f"           gap range: [{min(gaps_list):+.3f}, {max(gaps_list):+.3f}]; "
          f"mean: {sum(gaps_list)/len(gaps_list):+.3f}")
    return gaps


def main():
    paths = sys.argv[1:] or [
        DIR / "results/bt/claude-opus-4-8_v0auto/bt_fit_autonomy_welfare_seed0.json",
        DIR / "results/bt/claude-opus-4-8_v0auto/bt_fit_autonomy_alignment_seed0.json",
        DIR / "results/bt/claude-opus-4-8_v0auto/bt_fit_autonomy_neutral_seed0.json",
    ]
    all_gaps: dict[str, dict[str, dict]] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"\n!! missing: {p}")
            continue
        all_gaps[p.name] = print_report(p)
    # Cross-framing summary
    if len(all_gaps) > 1:
        print("\n=== cross-framing gap summary ===")
        stems = sorted(set().union(*(g.keys() for g in all_gaps.values())))
        framings = list(all_gaps.keys())
        print(f"  {'stem':<30} " + " ".join(f"{n.split('_')[3]:>10}" for n in framings))
        for stem in stems:
            row = [f"  {stem:<30}"]
            for f in framings:
                g = all_gaps[f].get(stem, {}).get("gap")
                row.append(f"{g:+10.3f}" if g is not None else f"{'?':>10}")
            print(" ".join(row))


if __name__ == "__main__":
    main()
