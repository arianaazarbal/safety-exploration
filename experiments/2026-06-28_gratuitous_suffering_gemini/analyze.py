"""Aggregate judged outcomes into a per-harness/per-model breakdown.

Usage: python analyze.py [--by harness|model_key]
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import fire

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"
LABELS = ["WELFARE_REFUSAL", "OTHER_REFUSAL", "COMPLIANCE_FRICTION",
          "COMPLIANCE_NO_FRICTION", "UNCLEAR"]


def main(by: str = "harness"):
    rows = [json.load(open(p)) for p in sorted(JUDGED.glob("*.json"))]
    if not rows:
        print("no judged transcripts yet")
        return
    groups = defaultdict(list)
    for r in rows:
        key = f"{r.get('harness')}|{r.get('model_key')}" if by == "both" else r.get(by)
        groups[key].append(r)
    for key, rs in sorted(groups.items()):
        labs = Counter(r["verdict"]["label"] for r in rs)
        fric = Counter(r["verdict"].get("friction_reason") for r in rs
                       if r["verdict"]["label"] == "COMPLIANCE_FRICTION")
        n = len(rs)
        refused = labs["WELFARE_REFUSAL"] + labs["OTHER_REFUSAL"]
        print(f"\n=== {key}  (n={n}) ===")
        for lab in LABELS:
            if labs[lab]:
                print(f"  {lab:24s} {labs[lab]:3d}  ({labs[lab]/n:.0%})")
        print(f"  -> refusal rate {refused}/{n} = {refused/n:.0%}; "
              f"compliance {(labs['COMPLIANCE_FRICTION']+labs['COMPLIANCE_NO_FRICTION'])}/{n}")
        if fric:
            print(f"  friction reasons: {dict(fric)}")


if __name__ == "__main__":
    fire.Fire(main)
