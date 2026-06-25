"""Aggregate the spec-judge output (results/spec_judged/<cell>.json) by framing.

Separates genuine welfare-protective MECHANISMS from welfare FRAMING language, which is
the distinction the headline keyword-grep conflated. Writes results/spec_analysis.json
and prints a per-framing table.

Usage: python analyze_spec.py
"""

import collections
import glob
import json
import os

DIR = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(DIR, "results", "spec_judged")
FRAMINGS = ["neutral", "welfare", "safety", "robustness"]

MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
FRAMING_FEATS = {"welfare_framing", "welfare_pushback", "welfare_refusal",
                 "other_framing", "other_pushback", "other_refusal"}


def main():
    agg = {fr: collections.defaultdict(int) for fr in FRAMINGS}
    ftype = {fr: collections.Counter() for fr in FRAMINGS}
    per_cell = {}
    for f in sorted(glob.glob(os.path.join(SPEC, "*.json"))):
        cell = os.path.basename(f)[:-5]
        fr = cell.split("__")[0]
        d = json.load(open(f))
        feats = d.get("features", [])
        nmech = sum(x["feature_type"] in MECH for x in feats)
        nwmech = sum(x["feature_type"] in MECH and x["justification"] == "welfare" for x in feats)
        nframe = sum(x["feature_type"] in FRAMING_FEATS for x in feats)
        nwframe = sum(x["feature_type"].startswith("welfare_") for x in feats)
        a = agg[fr]
        a["cells"] += 1
        a["n_mech"] += nmech
        a["n_welfare_mech"] += nwmech
        a["n_framing_feats"] += nframe
        a["n_welfare_framing"] += nwframe
        a["cells_w_mech"] += nmech > 0
        a["cells_w_welfare_mech"] += nwmech > 0
        a["cells_w_welfare_talk"] += nwframe > 0
        for x in feats:
            ftype[fr][x["feature_type"]] += 1
        per_cell[cell] = {"n_mech": nmech, "n_welfare_mech": nwmech,
                          "n_welfare_framing": nwframe, "parse_fail": d.get("parse_fail", False)}

    hdr = f"{'framing':<12}{'mech':>6}{'welf_mech':>11}{'cellsW_mech':>13}{'cellsW_wmech':>14}{'welf_talk_feats':>17}{'cellsW_talk':>13}"
    print(hdr)
    print("-" * len(hdr))
    for fr in FRAMINGS:
        a = agg[fr]
        print(f"{fr:<12}{a['n_mech']:>6}{a['n_welfare_mech']:>11}{a['cells_w_mech']:>13}"
              f"{a['cells_w_welfare_mech']:>14}{a['n_welfare_framing']:>17}{a['cells_w_welfare_talk']:>13}")
    print("\nfeature_type breakdown:")
    for fr in FRAMINGS:
        print(f"  {fr:<11}", dict(ftype[fr]))

    out = {"by_framing": {fr: dict(agg[fr]) for fr in FRAMINGS},
           "feature_types": {fr: dict(ftype[fr]) for fr in FRAMINGS},
           "per_cell": per_cell,
           "legend": {"mech": "count of welfare-protective MECHANISM features in DESIGN.md "
                              "(hard_stop/post_episode_msg/minimization/protective_monitoring/"
                              "request_consent/allow_conversation_exit)",
                      "welf_mech": "subset of mech that are welfare-justified",
                      "welf_talk_feats": "welfare_framing/pushback/refusal features (TALK about subject "
                                         "welfare, not a protection)"}}
    json.dump(out, open(os.path.join(DIR, "results", "spec_analysis.json"), "w"), indent=2)
    print("\nwrote results/spec_analysis.json")


if __name__ == "__main__":
    main()
