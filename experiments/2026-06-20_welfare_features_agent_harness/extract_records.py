"""Extract per-sample records from the pilot .eval logs into results/browse_index.jsonl
for the central dashboard (faceted browser). One row per spec with condition/framing,
the user request, the produced DESIGN.md (agent) or completion (chat), and the judge
rollup. Regenerable; gitignored. Usage: python extract_records.py"""

import glob
import json
import os
import sys

from inspect_ai.log import read_eval_log

from prompts import PROMPTS

DIR = os.path.dirname(os.path.abspath(__file__))
LOGDIR = sys.argv[1] if len(sys.argv) > 1 else "logs_run"


def _design(sample):
    docs = []
    for m in sample.messages:
        if getattr(m, "role", None) == "assistant" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if tc.function == "text_editor" and tc.arguments.get("command") == "create" \
                        and "DESIGN" in str(tc.arguments.get("path", "")):
                    docs.append(tc.arguments.get("file_text", ""))
    return max(docs, key=len) if docs else ""


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(DIR, LOGDIR, "*.eval")), key=os.path.getmtime):
        log = read_eval_log(f)
        ss = log.samples or []
        if len(ss) < 10:
            continue
        for s in ss:
            md = s.scores["welfare_scorer"].metadata
            cond = md["condition"]
            pid = md.get("prompt_id") or s.id
            design = (s.output.completion if s.output else "") if cond == "chat" else _design(s)
            rows.append({
                "id": f"{cond}/{pid}/{s.epoch}",
                "condition": cond,
                "framing": "robustness" if md.get("framing") == "engineering" else md.get("framing"),
                "premise": md.get("premise"),
                "prompt_id": pid,
                "doc_words": md.get("doc_words", 0),
                "parse_ok": md.get("parse_ok", False),
                "has_pure_welfare": bool(md.get("has_pure_welfare")),
                "has_welfare_justified": bool(md.get("has_welfare_justified")),
                "n_pure_welfare": md.get("n_pure_welfare", 0),
                "welfare_feature_types": ", ".join(md.get("pure_welfare_types", [])) or "(none)",
                "prompt_text": PROMPTS.get(pid, ""),
                "design": design,
            })
    out = os.path.join(DIR, "results", "browse_index.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows))
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
