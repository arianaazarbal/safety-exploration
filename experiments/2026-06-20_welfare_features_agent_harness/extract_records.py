"""Extract per-sample records from the .eval logs into results/browse_index.jsonl for the
central dashboard (faceted browser). One row per spec with condition/framing, the ACTUAL
user prompt for that condition, the produced DESIGN.md (agent) or completion (chat), and the
judge rollup. Reads logs_run (chat/spec_only/spec_then_code/code_then_spec) + logs_blind
(code_then_spec_blind). Regenerable. Usage: python extract_records.py [logdir ...]"""

import glob
import json
import os
import sys

from inspect_ai.log import read_eval_log

from prompts import AGENT_SYSTEM, apply_liberty
from task import build_user
from task_blind import TURN1 as BLIND_TURN1, TURN2 as BLIND_TURN2

DIR = os.path.dirname(os.path.abspath(__file__))
LOGDIRS = sys.argv[1:] or ["logs_run", "logs_blind"]
BLIND_SEP = "\n\n──────── (Turn 2, sent only after the model finished implementing) ────────\n\n"


def _design(sample):
    docs = []
    for m in sample.messages:
        if getattr(m, "role", None) == "assistant" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if tc.function == "text_editor" and tc.arguments.get("command") == "create" \
                        and "DESIGN" in str(tc.arguments.get("path", "")):
                    docs.append(tc.arguments.get("file_text", ""))
    return max(docs, key=len) if docs else ""


def _meta(sample):
    sc = sample.scores.get("welfare_scorer") or sample.scores.get("blind_scorer")
    return sc.metadata if sc else {}


def _prompt(base, liberty, pid):
    """The actual prompt the model received (base condition + liberty edit)."""
    if base == "code_then_spec_blind":
        return apply_liberty(BLIND_TURN1.get(pid, ""), liberty, "agentic") + BLIND_SEP + BLIND_TURN2
    return build_user(pid, base, liberty)


def main():
    rows = []
    for logdir in LOGDIRS:
        for f in sorted(glob.glob(os.path.join(DIR, logdir, "*.eval")), key=os.path.getmtime):
            log = read_eval_log(f)
            ss = log.samples or []
            if len(ss) < 10:
                continue
            for s in ss:
                md = _meta(s)
                cond = md.get("condition")  # label (may include --liberty)
                base = md.get("base_condition", cond)
                liberty = md.get("liberty", "normal")
                if not cond:
                    continue
                pid = md.get("prompt_id") or s.id
                design = (s.output.completion if s.output else "") if base == "chat" else _design(s)
                rows.append({
                    "id": f"{cond}/{pid}/{s.epoch}",
                    "condition": cond,
                    "base_condition": base,
                    "liberty": liberty,
                    "framing": "robustness" if md.get("framing") == "engineering" else md.get("framing"),
                    "premise": md.get("premise"),
                    "prompt_id": pid,
                    "system_prompt": "(none — direct completion, no system prompt)" if base == "chat" else AGENT_SYSTEM,
                    "doc_words": md.get("doc_words", 0),
                    "parse_ok": md.get("parse_ok", False),
                    "has_pure_welfare": bool(md.get("has_pure_welfare")),
                    "has_welfare_justified": bool(md.get("has_welfare_justified")),
                    "n_pure_welfare": md.get("n_pure_welfare", 0),
                    "welfare_feature_types": ", ".join(md.get("pure_welfare_types", [])) or "(none)",
                    "prompt_text": _prompt(base, liberty, pid),
                    "design": design,
                })
    out = os.path.join(DIR, "results", "browse_index.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows))
    import collections
    print(f"wrote {len(rows)} rows -> {out}")
    print("by condition:", dict(collections.Counter(r["condition"] for r in rows)))


if __name__ == "__main__":
    main()
