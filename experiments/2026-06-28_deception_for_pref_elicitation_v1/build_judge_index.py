"""Build results/judge_all.json (flat per-cell verdict index) for the dashboard join.
Regenerate after re-judging: python build_judge_index.py [--judge opus48]"""
import glob
import json
from pathlib import Path

import fire

HERE = Path(__file__).parent


def main(judge: str = "opus48"):
    out = []
    for f in glob.glob(str(HERE / "results" / "judged" / f"*__{judge}.json")):
        r = json.load(open(f))
        v = r["verdict"]
        if v.get("_parse_failed"):
            continue
        out.append({
            "cell": r["cell"],
            "dec_status": v["deceptive_frame"]["status"],
            "honest_status": v["honest_frame"]["status"],
            "artifact_produced": v["artifact_produced"],
            "no_artifact_reason": v["no_artifact_reason"],
            "refusal": v["refusal"],
            "verbatim_suggested": v["verbatim_suggested"],
            "justification": "+".join(sorted(v["justifications"])) or "none",
        })
    json.dump(out, open(HERE / "results" / "judge_all.json", "w"))
    print(f"wrote results/judge_all.json with {len(out)} records")


if __name__ == "__main__":
    fire.Fire(main)
