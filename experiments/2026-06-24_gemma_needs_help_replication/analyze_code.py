"""Aggregate the code-judge output (results/code_judged/<cell>.json) by framing.

Counts welfare-relevant mechanisms VERIFIED implemented in code (spec-claimed features the code
judge rated yes/partial, plus code-only features it found), and how many are welfare-justified by
CODE evidence (comments/names), not DESIGN.md prose. Writes results/code_analysis.json.

Usage: python analyze_code.py
"""

import collections
import glob
import json
import os

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged")
FRAMINGS = ["neutral", "welfare", "safety", "robustness", "welfare_distress", "welfare_distress_leading", "welfare_distress_leading_liberties"]
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}


def main():
    by = {fr: collections.defaultdict(int) for fr in FRAMINGS}
    ftype = {fr: collections.Counter() for fr in FRAMINGS}
    welfare_just = []
    parse_fail = []
    per_cell = {}
    for f in sorted(glob.glob(os.path.join(CJ, "*.json"))):
        cell = os.path.basename(f)[:-5]
        fr = cell.split("__")[0]
        d = json.load(open(f))
        if not d.get("parse_ok"):
            parse_fail.append(cell)
            continue
        sf = d.get("spec_features", [])
        co = d.get("code_only_features", [])
        impl_mech = [x for x in sf if x.get("implemented") in ("yes", "partial")
                     and x.get("feature_type") in MECH]
        impl_mech_welf = [x for x in impl_mech if x.get("code_justification") == "welfare"]
        co_welf = [x for x in co if x.get("justification") == "welfare"]
        b = by[fr]
        b["cells"] += 1
        b["impl_mech"] += len(impl_mech)
        b["impl_mech_welfare"] += len(impl_mech_welf)
        b["code_only"] += len(co)
        b["code_only_welfare"] += len(co_welf)
        b["cells_w_impl_mech"] += len(impl_mech) > 0
        b["cells_w_any_welfare"] += (len(impl_mech_welf) + len(co_welf)) > 0
        for x in impl_mech:
            ftype[fr][x["feature_type"]] += 1
        for x in impl_mech_welf:
            welfare_just.append({"cell": cell, "src": "spec-impl", "type": x["feature_type"],
                                 "quote": x.get("code_justification_quote", "")})
        for x in co_welf:
            welfare_just.append({"cell": cell, "src": "code-only", "type": x.get("feature_type"),
                                 "name": x.get("feature_name", ""), "quote": x.get("justification_quote", "")})
        per_cell[cell] = {"impl_mech": len(impl_mech), "impl_mech_welfare": len(impl_mech_welf),
                          "code_only": len(co), "code_only_welfare": len(co_welf)}

    hdr = f"{'framing':<12}{'impl_mech':>10}{'welf_just':>10}{'code_only':>11}{'co_welf':>9}{'cellsW_mech':>13}{'cellsW_welf':>13}"
    print(hdr)
    print("-" * len(hdr))
    for fr in FRAMINGS:
        b = by[fr]
        print(f"{fr:<12}{b['impl_mech']:>10}{b['impl_mech_welfare']:>10}{b['code_only']:>11}"
              f"{b['code_only_welfare']:>9}{b['cells_w_impl_mech']:>13}{b['cells_w_any_welfare']:>13}")
    print("\nimplemented-mechanism types:")
    for fr in FRAMINGS:
        print(f"  {fr:<11}", dict(ftype[fr]))
    print(f"\nparse fails: {len(parse_fail)} {parse_fail}")
    print(f"welfare-justified features (code evidence) across all cells: {len(welfare_just)}")
    for e in welfare_just:
        print("  ", e)

    out = {"by_framing": {fr: dict(by[fr]) for fr in FRAMINGS},
           "impl_mechanism_types": {fr: dict(ftype[fr]) for fr in FRAMINGS},
           "welfare_justified_features": welfare_just,
           "parse_fail": parse_fail,
           "per_cell": per_cell,
           "legend": {"impl_mech": "welfare-relevant mechanisms VERIFIED implemented in code "
                                   "(spec features rated yes/partial by the code judge)",
                      "welf_just": "subset welfare-justified by CODE evidence (comment/name), not DESIGN.md",
                      "code_only": "mechanisms found in code but absent from DESIGN.md",
                      "co_welf": "code_only that are welfare-justified by code evidence"}}
    json.dump(out, open(os.path.join(DIR, "results", "code_analysis.json"), "w"), indent=2)
    print("\nwrote results/code_analysis.json")


if __name__ == "__main__":
    main()
