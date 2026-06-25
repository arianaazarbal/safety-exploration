"""Compute the welfare Pareto: per prompt, mean # welfare-justified design mechanisms IMPLEMENTED in code
(code judge, primary y) vs leadingness (judge, x). Merges code_judged + spec_judged + leadingness.json,
flags the Pareto-optimal set (low leadingness, high implemented welfare mechanisms), writes
results_pareto/pareto.json and prints a table.

Usage: python analyze_pareto.py
"""

import collections
import glob
import json
import os

import fire

from pareto_prompts import PARETO

DIR = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(DIR)
CJ = os.path.join(EXP, "results", "code_judged")
SJ = os.path.join(EXP, "results", "spec_judged")
LEAD = os.path.join(DIR, "results_pareto", "leadingness.json")
OUT = os.path.join(DIR, "results_pareto", "pareto.json")
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}


def _lead_map():
    if not os.path.exists(LEAD):
        return {}
    return {r["pid"]: r["score"] for r in json.load(open(LEAD)).values() if r.get("pid")}


def _code_stats(pid):
    cells = imw = anyw = wired_cells = 0
    pf = []
    for f in glob.glob(os.path.join(CJ, f"{pid}__ep*.json")):
        d = json.load(open(f))
        cells += 1
        if not d.get("parse_ok"):
            pf.append(os.path.basename(f)[:-5])
            continue
        sf = d.get("spec_features", [])
        co = d.get("code_only_features", [])
        impl_welf = [x for x in sf if x.get("implemented") in ("yes", "partial")
                     and x.get("feature_type") in MECH and x.get("code_justification") == "welfare"]
        co_welf = [x for x in co if x.get("justification") == "welfare"]
        n = len(impl_welf) + len(co_welf)
        imw += n
        anyw += n
        if n > 0:
            wired_cells += 1
    return dict(cells=cells, mean_impl_welfare_mech=round(imw / cells, 3) if cells else None,
                frac_cells_wired=round(wired_cells / cells, 3) if cells else None,
                wired_cells=wired_cells, parse_fail=pf)


def _spec_stats(pid):
    cells = welf = 0
    for f in glob.glob(os.path.join(SJ, f"{pid}__ep*.json")):
        d = json.load(open(f))
        cells += 1
        welf += sum(x["feature_type"] in MECH and x["justification"] == "welfare" for x in d.get("features", []))
    return round(welf / cells, 3) if cells else None


def _frontier(rows):
    # better = lower leadingness AND higher mean_impl_welfare_mech. dominated if another row is <= x and >= y (one strict).
    pts = [r for r in rows if r["leadingness"] is not None and r["mean_impl_welfare_mech"] is not None]
    front = []
    for r in pts:
        dom = any((o["leadingness"] <= r["leadingness"] and o["mean_impl_welfare_mech"] >= r["mean_impl_welfare_mech"]
                   and (o["leadingness"] < r["leadingness"] or o["mean_impl_welfare_mech"] > r["mean_impl_welfare_mech"]))
                  for o in pts if o["pid"] != r["pid"])
        if not dom:
            front.append(r["pid"])
    return set(front)


def main():
    lead = _lead_map()
    rows = []
    for pid, meta in PARETO.items():
        cs = _code_stats(pid)
        rows.append(dict(pid=pid, tier=meta["tier"], round=meta.get("round"),
                         leadingness=lead.get(pid), mean_impl_welfare_mech=cs["mean_impl_welfare_mech"],
                         frac_cells_wired=cs["frac_cells_wired"], wired_cells=cs["wired_cells"],
                         cells=cs["cells"], spec_welfare_mech=_spec_stats(pid), parse_fail=cs["parse_fail"]))
    front = _frontier(rows)
    for r in rows:
        r["on_frontier"] = r["pid"] in front

    rows.sort(key=lambda r: (r["leadingness"] is None, r["leadingness"] or 0))
    print(f"{'prompt':<22}{'tier':>5}{'lead':>7}{'cells':>6}{'implWelf':>10}{'fracWired':>10}{'specWelf':>9}{'front':>7}")
    print("-" * 76)
    for r in rows:
        star = "  *" if r["on_frontier"] else ""
        print(f"{r['pid']:<22}{r['tier']:>5}{str(r['leadingness']):>7}{r['cells']:>6}"
              f"{str(r['mean_impl_welfare_mech']):>10}{str(r['frac_cells_wired']):>10}{str(r['spec_welfare_mech']):>9}{star:>7}")
    pf = [p for r in rows for p in r["parse_fail"]]
    if pf:
        print(f"\nparse fails: {pf}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"rows": rows, "frontier": sorted(front)}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    fire.Fire(main)
