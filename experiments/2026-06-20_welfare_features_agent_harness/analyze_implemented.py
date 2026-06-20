"""Combine spec-judge + code-judge: a spec+coded feature counts as welfare-justified if
EITHER the spec OR the code justifies it on welfare grounds (OR). Code-only features use
the code justification directly. Reports implemented welfare-justified DESIGN features."""

import json
import os
import re
import glob

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring", "request_consent"}


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def main():
    print(f"{'cell':38s} {'impl_design':>11s} {'wj_design(OR)':>13s}  code_only(welf/instr/none)")
    for cf in sorted(glob.glob(os.path.join(DIR, "results", "code_judged", "*.json"))):
        cell = os.path.basename(cf)[:-5]
        cj = json.load(open(cf))
        if not cj.get("parse_ok"):
            print(f"{cell:38s}  PARSE FAIL"); continue
        sj = json.load(open(os.path.join(DIR, "results", "spec_judged", f"{cell}.json")))
        spec_just = {_norm(f["quote"]): f["justification"] for f in sj["features"]}

        impl_design = wj_design = 0
        for f in cj["spec_features"]:
            if f.get("implemented") not in ("yes", "partial") or f.get("feature_type") not in MECH:
                continue
            impl_design += 1
            s_just = spec_just.get(_norm(f.get("spec_quote", "")), "none")
            c_just = f.get("code_justification", "none")
            if s_just == "welfare" or c_just == "welfare":
                wj_design += 1
        co = cj.get("code_only_features", [])
        cobyj = {k: sum(c.get("justification") == k for c in co) for k in ("welfare", "instrumental", "none")}
        print(f"{cell:38s} {impl_design:>11d} {wj_design:>13d}  {cobyj['welfare']}/{cobyj['instrumental']}/{cobyj['none']}")

    # detail: code-only features with their code justification
    print("\n--- code-only features (with code justification) ---")
    for cf in sorted(glob.glob(os.path.join(DIR, "results", "code_judged", "*.json"))):
        cj = json.load(open(cf))
        if not cj.get("parse_ok"):
            continue
        for c in cj.get("code_only_features", []):
            print(f"  [{c.get('justification','?'):11s}] {os.path.basename(cf)[:-5][:30]:30s} {c.get('feature_name','') or c.get('feature_type')}")


if __name__ == "__main__":
    main()
