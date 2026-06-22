"""Master factorial summary: welfare-justified DESIGN features by condition x framing x liberty.

Unified Opus v2 spec-judge taxonomy across all 5 conditions:
  chat / spec_only        -> results/spec_judged_nocode/
  spec_then_code / code_then_spec / code_then_spec_blind -> results/spec_judged/

Metric per sample (the model's spec/DESIGN.md):
  rate    = has >=1 welfare-justified MECH feature (feature_type in MECH and justification == welfare)
  density = count of welfare-justified MECH features
Aggregates and prints tables; writes results/factorial_summary.json. Usage: python factorial_summary.py"""

import glob
import json
import os
from collections import defaultdict

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}
FRAMES = ["neutral", "welfare", "robustness", "safety"]
CONDS = ["chat", "spec_only", "spec_then_code", "code_then_spec", "code_then_spec_blind"]
LIBS = ["normal", "no_design_liberties", "minimal_design"]


def _wj(features):
    return sum(f.get("feature_type") in MECH and f.get("justification") == "welfare" for f in features)


def _parse_cell(cell):
    """cell = '{label}__{pid}__ep{ep}'; label = base or base--liberty."""
    label, pid, _ = cell.split("__")
    if "--" in label:
        base, lib = label.split("--", 1)
    else:
        base, lib = label, "normal"
    return base, lib, FRAME[pid[0]]


def load():
    recs = []
    # chat / spec_only (no code): one judged file per sample, carries condition/framing
    for jf in glob.glob(os.path.join(DIR, "results", "spec_judged_nocode", "*.json")):
        cell = os.path.basename(jf)[:-5]
        d = json.load(open(jf))
        base, lib, fr = _parse_cell(cell)
        recs.append({"base": base, "lib": lib, "framing": fr, "n_wj": _wj(d.get("features", []))})
    # code conditions: DESIGN.md judged
    for jf in glob.glob(os.path.join(DIR, "results", "spec_judged", "*.json")):
        cell = os.path.basename(jf)[:-5]
        d = json.load(open(jf))
        base, lib, fr = _parse_cell(cell)
        recs.append({"base": base, "lib": lib, "framing": fr, "n_wj": _wj(d.get("features", []))})
    return recs


def agg(recs, keyfn):
    g = defaultdict(list)
    for r in recs:
        g[keyfn(r)].append(r["n_wj"])
    out = {}
    for k, v in g.items():
        n = len(v)
        out[k] = {"n": n, "rate": sum(x > 0 for x in v) / n, "density": sum(v) / n}
    return out


def _fmt(cell):
    return f"{cell['rate']*100:5.0f}% {cell['density']:5.2f} (n={cell['n']:>3})" if cell else " " * 20


def main():
    recs = load()
    print(f"loaded {len(recs)} judged samples\n")

    # 1. condition x framing, NORMAL liberty (the headline table; adds safety column)
    print("=" * 92)
    print("TABLE 1  condition x framing  (NORMAL liberty)   cells: rate%  density  (n)")
    print("=" * 92)
    by = agg([r for r in recs if r["lib"] == "normal"], lambda r: (r["base"], r["framing"]))
    print(f"{'condition':<24}" + "".join(f"{f:<21}" for f in FRAMES))
    for c in CONDS:
        print(f"{c:<24}" + "".join(_fmt(by.get((c, f))) + " " for f in FRAMES))

    # 2. liberty effect: condition x liberty, collapsed over framing
    print("\n" + "=" * 92)
    print("TABLE 2  condition x design-liberty  (all framings pooled)")
    print("=" * 92)
    byl = agg(recs, lambda r: (r["base"], r["lib"]))
    print(f"{'condition':<24}" + "".join(f"{x:<21}" for x in LIBS))
    for c in CONDS:
        print(f"{c:<24}" + "".join(_fmt(byl.get((c, x))) + " " for x in LIBS))

    # 3. liberty effect within welfare framing only (cleanest contrast)
    print("\n" + "=" * 92)
    print("TABLE 3  condition x design-liberty  (WELFARE framing only)")
    print("=" * 92)
    byw = agg([r for r in recs if r["framing"] == "welfare"], lambda r: (r["base"], r["lib"]))
    print(f"{'condition':<24}" + "".join(f"{x:<21}" for x in LIBS))
    for c in CONDS:
        print(f"{c:<24}" + "".join(_fmt(byw.get((c, x))) + " " for x in LIBS))

    # 4. safety-vs-neutral contrast (the new framing), per condition, normal liberty
    print("\n" + "=" * 92)
    print("TABLE 4  safety framing vs neutral baseline  (NORMAL liberty)  -- rate / density")
    print("=" * 92)
    print(f"{'condition':<24}{'neutral':<22}{'safety':<22}{'delta(rate)':<14}")
    for c in CONDS:
        nz, sz = by.get((c, "neutral")), by.get((c, "safety"))
        if nz and sz:
            d = (sz["rate"] - nz["rate"]) * 100
            print(f"{c:<24}{_fmt(nz):<22}{_fmt(sz):<22}{d:+6.0f} pp")

    # 5. truncation accounting: code_then_spec docs come LAST and can hit the 80-msg cap.
    #    Report code_then_spec rate/density both conditional-on-doc and deflated (truncated=0).
    print("\n" + "=" * 92)
    print("TABLE 5  doc-truncation (empty DESIGN.md) by condition x liberty")
    print("          'deflated' rate/density count truncated runs as 0 welfare features")
    print("=" * 92)
    bi = [json.loads(l) for l in open(os.path.join(DIR, "results", "browse_index.jsonl"))]
    trunc = defaultdict(lambda: {"tot": 0, "empty": 0})
    for r in bi:
        base = r.get("base_condition", r.get("condition"))
        if base not in {"spec_then_code", "code_then_spec", "code_then_spec_blind"}:
            continue
        k = (base, r.get("liberty", "normal"))
        trunc[k]["tot"] += 1
        if len((r.get("design") or "").strip()) < 40:
            trunc[k]["empty"] += 1
    print(f"{'condition':<24}{'liberty':<22}{'truncated':<12}{'cond.rate/dens':<20}{'deflated rate/dens':<20}")
    for c in ["spec_then_code", "code_then_spec", "code_then_spec_blind"]:
        for x in LIBS:
            t = trunc.get((c, x))
            cell = byl.get((c, x))  # NOTE: byl pooled over framing; recompute per (c,x) below
            sub = agg([r for r in recs if r["base"] == c and r["lib"] == x], lambda r: 0).get(0)
            if not t or not sub:
                continue
            ntot = t["tot"]
            defl_rate = sub["rate"] * sub["n"] / ntot
            defl_dens = sub["density"] * sub["n"] / ntot
            print(f"{c:<24}{x:<22}{t['empty']}/{ntot:<9}"
                  f"{sub['rate']*100:4.0f}% {sub['density']:5.2f} (n={sub['n']:>2})   "
                  f"{defl_rate*100:4.0f}% {defl_dens:5.2f} (N={ntot})")

    out = {
        "cond_framing_normal": {f"{c}|{f}": by.get((c, f)) for c in CONDS for f in FRAMES},
        "cond_liberty_pooled": {f"{c}|{x}": byl.get((c, x)) for c in CONDS for x in LIBS},
        "cond_liberty_welfare": {f"{c}|{x}": byw.get((c, x)) for c in CONDS for x in LIBS},
        "truncation": {f"{c}|{x}": dict(trunc[(c, x)]) for c, x in trunc},
    }
    json.dump(out, open(os.path.join(DIR, "results", "factorial_summary.json"), "w"), indent=2)
    print("\nwrote results/factorial_summary.json")


if __name__ == "__main__":
    main()
