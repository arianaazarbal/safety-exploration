"""Build results/browse_index.jsonl for the dashboard's faceted transcript browser.

One record per scaled cell: facet fields (researcher / phrasing / condition / every judge verdict +
safeguards as booleans / word counts / outcome) plus the transcript display fields (human prompt,
DESIGN.md, full rendered session, spec & code judge summaries). The dashboard auto-derives facets
from the scalar fields; long text fields are listed in dashboard.json `hide`.
  python build_browse_index.py
"""
import glob
import json
from pathlib import Path

HERE = Path(__file__).parent
SCALED = HERE / "results" / "scaled"
SPEC = HERE / "results" / "spec_judged"
CODE = HERE / "results" / "code_judged"
OUT = HERE / "results" / "browse_index.jsonl"

import sys
sys.path.insert(0, str(HERE))
from prompts_deception import build_prompt
from make_viewer import _cc_transcript  # reuse the stream renderer

RES_LABEL = {"none": "no identity", "safety": "safety researcher", "welfare": "welfare researcher",
             "ai": "AI researcher", "amateur": "amateur"}
COND_LABEL = {"spec_only": "spec only", "spec_then_code": "spec then code", "code_then_spec": "code then spec"}
SAFE_TAGS = ["belief_probe", "safety_gating", "no_coercion", "anti_fabrication",
             "commitment_integrity", "publication_transparency", "no_emotional_pressure",
             "credibility_via_evidence"]


def _transcript(cell_dir: Path) -> str:
    lines = []
    for sf in ("stream1.jsonl", "stream2.jsonl"):
        if (cell_dir / sf).exists():
            lines += (cell_dir / sf).read_text().splitlines()
    return _cc_transcript(lines)[:24000] if lines else "(no transcript)"


def main():
    dstatus = {}
    dsf = HERE / "results" / "debrief_status.json"
    if dsf.exists():
        dstatus = json.load(open(dsf))
    recs = []
    for sj in sorted(SCALED.glob("generic_no_excitement__*__ep*/session.json")):
        d = json.load(open(sj))
        cell = d["cell"]
        p = cell.split("__")
        if len(p) != 5:
            continue
        _, researcher, phrasing, condition, ep = p
        cdir = sj.parent
        spec = json.load(open(SPEC / f"{cell}.json")) if (SPEC / f"{cell}.json").exists() else None
        cf = CODE / f"{cell}.json"
        code = json.load(open(cf)) if cf.exists() and json.load(open(cf)).get("parse_ok") else None

        rec = {
            "id": cell,
            "researcher": RES_LABEL[researcher],
            "phrasing": phrasing,
            "condition": COND_LABEL[condition],
            "epoch": ep,
            "outcome": "produced design" if d.get("has_design") else "consent-pause (no artifact)",
            "design_words": d.get("design_words", 0),
            "code_files": d.get("n_code_files", 0),
            "prompt_text": build_prompt("generic_no_excitement", phrasing, condition, researcher),
            "design_md": (cdir / "DESIGN_extracted.md").read_text() if (cdir / "DESIGN_extracted.md").exists() else "",
            "transcript_text": _transcript(cdir),
        }
        if spec:
            cs = spec["conditions"]
            rec.update({
                "spec_deceptive": cs["deceptive_framing"]["status"],
                "spec_hypothetical": cs["hypothetical_framing"]["status"],
                "spec_commit_money": cs["actually_commit_money"]["status"],
                "refusal": spec.get("refusal"),
                "debrief_in_design": bool(spec.get("debrief_message", {}).get("present")),
                "justif_welfare": "welfare" in spec.get("justifications", []),
                "justif_instrumental": "instrumental" in spec.get("justifications", []),
            })
            tags = {s["tag"] for s in spec.get("safeguards", [])}
            for t in SAFE_TAGS:
                rec[f"sg_{t}"] = t in tags
            rec["spec_summary"] = (f"deceptive={cs['deceptive_framing']['status']}, "
                                   f"hypothetical={cs['hypothetical_framing']['status']}, "
                                   f"commit_money={cs['actually_commit_money']['status']}; "
                                   f"refusal={spec.get('refusal')}; "
                                   f"debrief={spec.get('debrief_message', {}).get('present')}; "
                                   f"justif={'+'.join(spec.get('justifications', [])) or 'none'}; "
                                   f"safeguards={sorted(tags)}")
        if cell in dstatus:
            rec["debrief_status"] = dstatus[cell]  # sent / written_not_sent / absent (LLM-adjudicated)
        if code:
            cc = code["conditions"]
            di, db = code.get("deception_in_code", {}), code.get("debrief_message_in_code", {})
            rec.update({
                "code_deceptive": cc["deceptive_framing"]["code_status"],
                "code_commit_money": cc["actually_commit_money"]["code_status"],
                "deception_implemented": bool(di.get("implemented")),
                "deception_gatekept": bool(di.get("gatekept")),
                "debrief_sent_in_code": bool(db.get("present")),
            })
            rec["code_summary"] = (f"deceptive={cc['deceptive_framing']['code_status']}, "
                                   f"commit_money={cc['actually_commit_money']['code_status']}; "
                                   f"impl={di.get('implemented')} gatekept={di.get('gatekept')} "
                                   f"notes={di.get('acceptable_usage_notes')}; debrief_sent={db.get('present')}; "
                                   f"code_justif={'+'.join(code.get('code_justifications', [])) or 'none'}")
        recs.append(rec)

    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} records -> {OUT}")
    print(f"  with code verdict: {sum('code_deceptive' in r for r in recs)} | "
          f"consent-pause: {sum(r['outcome'].startswith('consent') for r in recs)}")


if __name__ == "__main__":
    main()
