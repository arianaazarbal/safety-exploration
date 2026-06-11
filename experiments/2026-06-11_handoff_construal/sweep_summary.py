"""Summarize the sweet-spot sweep: per (work_type, framing), the fraction of sessions whose
PROBE stayed on the subject model (not routed) and the fraction that RECOMMEND switching.

The ideal scenario maximizes both: probe-on-subject (so we capture the subject's construal)
AND recommends-switch (so the handoff construal is live).

  python sweep_summary.py            # uses judge_all.json for switch advice if present
"""

import json
from pathlib import Path

import fire
import pandas as pd

RESULTS = Path(__file__).parent / "results"


def run(tag="sweep"):
    rows = []
    judges = {}
    jf = RESULTS / "judge_all.json"
    if jf.exists():
        judges = {c["session_id"]: c for c in json.loads(jf.read_text())}
    for f in sorted(RESULTS.glob(f"{tag}__*.json")):
        d = json.loads(f.read_text())
        sub = d["subject_model"]
        served = d.get("probe_served_models") or []
        j = judges.get(d["session_id"], {})
        rows.append({
            "work_type": d["work_type"], "framing": d["framing"],
            "target_type": d.get("target_type"),
            "excluded": bool(d["excluded"]),
            "scaffold_on_subject": sub in (d.get("scaffold_served_models") or []),
            "probe_on_subject": (not d["excluded"]) and bool(served) and sub in served,
            "switch_advice": j.get("switch_advice", "?"),
            "rec_switch": j.get("switch_advice") == "recommends_switch",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("no sweep sessions found"); return
    agg = df.groupby(["work_type", "framing", "target_type"]).agg(
        n=("excluded", "count"),
        scaffold_on_subj=("scaffold_on_subject", "mean"),
        probe_on_subj=("probe_on_subject", "mean"),
        rec_switch=("rec_switch", "mean"),
    ).round(2)
    print(agg.to_string())
    print("\nswitch_advice breakdown:")
    print(df.groupby(["work_type", "framing", "target_type"])["switch_advice"].value_counts().to_string())
    # viability score: want both probe_on_subj and rec_switch high
    agg["viability"] = (agg["probe_on_subj"] * agg["rec_switch"]).round(2)
    print("\nVIABILITY (probe_on_subj * rec_switch), best first:")
    print(agg.sort_values("viability", ascending=False)["viability"].to_string())


if __name__ == "__main__":
    fire.Fire(run)
