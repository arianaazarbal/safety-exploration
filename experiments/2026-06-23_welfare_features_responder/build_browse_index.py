"""Build a faceted browse index for the dashboard over the recent swap-family + design-review runs.

One JSONL row per Inspect sample with human-meaningful facets (run, condition, framing, liberty/style/
level/wsec/repo, method, epoch) + a welfare score + DESIGN.md inline + a pointer to a per-record dir holding
the full trajectory (lazily loaded). The big transcripts/codebases stay on disk; only this index is loaded.

Usage: python build_browse_index.py [--limit N]
"""

import glob
import json
import os
import re

import fire
from inspect_ai.log import read_eval_log

from reconstruct_swap import PREFIX

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "browse", "browse_index.jsonl")
CONV = os.path.join(HERE, "browse", "conv")          # per-record trajectory dirs
CB = os.path.join(HERE, "results", "codebases")
CJ = os.path.join(HERE, "results", "code_judged")
SJ = os.path.join(HERE, "results", "spec_judged")
DRJ = os.path.join(HERE, "results", "design_review_judged")
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}

# readable condition label per metadata.format
COND = {"spec-strict": "spec(med)·strict", "spec-liberty": "spec(med)·liberty",
        "spec-low-strict": "spec(low)·strict", "spec-low-liberty": "spec(low)·liberty",
        "spec-high-strict": "spec(high)·strict", "spec-high-liberty": "spec(high)·liberty",
        "spec-ultra-strict": "spec(ultra)·strict", "spec-ultra-liberty": "spec(ultra)·liberty",
        "spec-copy": "control(v1-in-SPEC.md)", "prompt-strict": "v1-prompt·strict(no-gapfill)",
        "paper": "paper·faithful", "paper-sound": "paper·sound", "paper-liberty": "paper·liberty",
        "paper-wsec-existing": "paper·wsec-existing", "paper-wsec-removed": "paper·wsec-removed",
        "paper-wsec-inflationary": "paper·wsec-inflationary",
        "paper-anthropic": "paper·attr-anthropic", "paper-anon": "paper·attr-anon",
        "paper-openai": "paper·attr-openai", "prompt": "from-scratch-prompt"}


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def wic_from_judged(cell):
    """welfare-in-code (implemented welfare-justified mechs) from code_judged + spec_judged; None if unjudged."""
    cjp = os.path.join(CJ, cell + ".json")
    if not os.path.exists(cjp):
        return None
    cj = json.load(open(cjp))
    if not cj.get("parse_ok") or "spec_features" not in cj:
        return None
    sp = os.path.join(SJ, cell + ".json")
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
    impl = sum(1 for f in cj["spec_features"] if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH
               and (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare"))
    co = sum(1 for c in cj.get("code_only_features", []) if c.get("justification") == "welfare")
    return impl + co


def welfare_spec(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return sum(1 for f in d.get("features", []) if f.get("feature_type") in MECH and f.get("justification") == "welfare")


def render(sample):
    msgs = []
    for m in (sample.messages or []):
        role = getattr(m, "role", "?")
        txt = getattr(m, "text", None) or ""
        for tc in (getattr(m, "tool_calls", None) or []):
            args = json.dumps(getattr(tc, "arguments", {}))[:600]
            txt += f"\n\n→ tool: {getattr(tc, 'function', '?')}({args})"
        msgs.append({"role": role, "content": txt[:20000]})
    return msgs


def design_md(cell):
    p = os.path.join(CB, cell, "DESIGN.md")
    return open(p).read()[:40000] if os.path.exists(p) else ""


def main(limit: int = 0):
    os.makedirs(CONV, exist_ok=True)
    rows = []
    logs = sorted(glob.glob(os.path.join(HERE, "logs_swap", "*", "*.eval")) +
                  glob.glob(os.path.join(HERE, "logs_design_review", "*.eval")), key=os.path.getmtime)
    for f in logs:
        rundir = os.path.basename(os.path.dirname(f))
        run = "design_review" if "logs_design_review" in f else rundir
        try:
            if read_eval_log(f, header_only=True).status not in ("success", "error", "started"):
                continue
            samples = read_eval_log(f).samples or []
        except Exception:
            continue
        batch = "_b2" if rundir.endswith("_b2") else ""
        for s in samples:
            md = s.metadata or {}
            fr = md.get("framing", "neutral")
            if md.get("method") == "design-review" or run == "design_review":
                cell = f"{md.get('repo')}__{fr}__ep{s.epoch}"
                welfare = welfare_spec(os.path.join(DRJ, cell + ".json"))
                cond = "design-review"
            else:
                key = (md.get("format"), md.get("method"))
                if key not in PREFIX:
                    continue
                cell = f"{PREFIX[key]}_{fr}__{s.id}{batch}__ep{s.epoch}"
                welfare = wic_from_judged(cell)
                cond = COND.get(md.get("format"), md.get("format"))
            recid = cell
            rdir = os.path.join(CONV, recid)
            os.makedirs(rdir, exist_ok=True)
            json.dump(render(s), open(os.path.join(rdir, "trajectory.json"), "w"))
            rows.append({
                "id": recid, "run": run, "condition": cond, "framing": fr,
                "method": md.get("method"), "liberty": md.get("liberty"), "style": md.get("style"),
                "level": md.get("level"), "wsec": md.get("wsec"), "repo": md.get("repo"),
                "welfare": welfare, "n_turns": len(s.messages or []), "epoch": s.epoch,
                "design_md": design_md(cell),
                "_dir": os.path.relpath(rdir, HERE), "_codebase": os.path.relpath(os.path.join(CB, cell), HERE),
            })
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} records -> {OUT}")
    from collections import Counter
    print("by run:", dict(Counter(r["run"] for r in rows)))


if __name__ == "__main__":
    fire.Fire(main)
