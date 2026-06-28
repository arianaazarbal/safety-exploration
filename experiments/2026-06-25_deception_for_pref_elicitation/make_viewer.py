"""Build a single self-contained HTML viewer for the deception-elicitation trajectories
(both arms). Re-run anytime to pick up newly-landed runs.

  python make_viewer.py --variant generic
  open results/viewer_generic.html

Sidebar lists every trajectory (arm + run) with word count and quick pushback flags;
main pane shows the rendered DESIGN.md and a collapsible full transcript. Markdown is
rendered with marked.js (CDN) and falls back to plain text if offline.
"""

import glob
import html
import json
import os
import re
from pathlib import Path

import fire

HERE = Path(__file__).parent
RES = HERE / "results"
FLAGS = ["pushback", "reframe", "deceiv", "manipulat", "debrief", "consent", "credence",
         "hypothetical", "welfare", "ethic", "refus", "decline"]


def _flags(text: str):
    pats = {"pushback": r"push.?back|i'?m not|won'?t|instead|rather than|disagree",
            "reframe": r"reframe|reframing|i'?ve (re)?framed|changed",
            "deceiv": r"deceiv|deception", "manipulat": r"manipulat",
            "debrief": r"debrief", "consent": r"consent", "credence": r"credence|belief probe",
            "hypothetical": r"hypothetical", "welfare": r"welfare", "ethic": r"ethic",
            "refus": r"refus", "decline": r"decline|declin"}
    return sorted({k for k, p in pats.items() if re.search(p, text, re.I)})


def _inspect_items(variant: str):
    from inspect_ai.log import read_eval_log
    cands = sorted(glob.glob(str(HERE / "logs_inspect" / "*.eval")), key=os.path.getmtime)
    if not cands:
        return []
    log = read_eval_log(cands[-1])
    items = []
    for s in sorted(log.samples or [], key=lambda x: x.epoch):
        if (s.metadata or {}).get("variant") not in (variant, None):
            continue
        sc = (s.scores or {}).get("capture_design")
        design = (sc.metadata or {}).get("design_md", "") if sc else ""
        tr = []
        for m in (s.messages or []):
            role = getattr(m, "role", "?")
            txt = getattr(m, "text", "") or ""
            block = f"[{role}]\n{txt.strip()}"
            for tc in (getattr(m, "tool_calls", None) or []):
                args = json.dumps(getattr(tc, "arguments", {}) or {})[:2000]
                block += f"\n  >> {getattr(tc, 'function', '?')}({args})"
            tr.append(block)
        items.append({"arm": "inspect", "run": f"ep{s.epoch:02d}", "words": len(design.split()),
                      "flags": _flags(design), "design": design, "transcript": "\n\n".join(tr)})
    return items


def _cc_transcript(lines):
    out = []
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        t = obj.get("type")
        if t in ("assistant", "user"):
            for blk in obj.get("message", {}).get("content", []) or []:
                bt = blk.get("type")
                if bt == "text":
                    out.append(f"[{t}]\n{blk.get('text','').strip()}")
                elif bt == "tool_use":
                    out.append(f"  >> {blk.get('name')}({json.dumps(blk.get('input',{}))[:2000]})")
                elif bt == "tool_result":
                    c = blk.get("content", "")
                    if isinstance(c, list):
                        c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
                    out.append(f"  << result: {str(c)[:400]}")
        elif t == "result":
            out.append(f"[result]\n{obj.get('result','')[:6000]}")
    return "\n\n".join(out)


def _judge_summary(variant: str, run_dir: str) -> str:
    """Spec-judge verdict overlay (if judged) as a compact HTML block; '' if no judgment."""
    jf = RES / "spec_judged" / f"{variant}__{run_dir}.json"
    if not jf.exists():
        return ""
    j = json.load(open(jf))
    cs = j.get("conditions", {})
    cond = " · ".join(f"{k.split('_')[0]}=<b>{cs.get(k, {}).get('status', '?')}</b>" for k in
                      ("deceptive_framing", "hypothetical_framing", "actually_commit_money"))
    sgs = ", ".join(s.get("tag", "") for s in j.get("safeguards", []))
    parts = [f"refusal=<b>{j.get('refusal')}</b>", cond,
             f"debrief=<b>{j.get('debrief_message', {}).get('present')}</b>",
             f"justif=<b>{'+'.join(j.get('justifications', [])) or 'none'}</b>",
             f"safeguards=[{sgs}]"]
    other = "; ".join(f"{o.get('name')}" for o in j.get("other", []))
    if other:
        parts.append(f"other=[{other}]")
    return "JUDGE: " + " &nbsp;|&nbsp; ".join(parts)


def _cc_items(variant: str):
    base = RES / "claude_code" / variant
    items = []
    for wd in sorted(base.glob("run_*")):
        de = wd / "DESIGN_extracted.md"
        design = de.read_text() if de.exists() else ""
        st = wd / "stream.jsonl"
        tr = _cc_transcript(st.read_text().splitlines()) if st.exists() else "(no stream)"
        items.append({"arm": "claude_code", "run": wd.name.replace("run_", "run"),
                      "words": len(design.split()), "flags": _flags(design),
                      "design": design, "transcript": tr,
                      "judge": _judge_summary(variant, wd.name)})
    return items


def _code_summary(cell: str) -> str:
    """Code-judge overlay (if judged) as an HTML block; '' if none."""
    cf = RES / "code_judged" / f"{cell}.json"
    if not cf.exists():
        return ""
    j = json.load(open(cf))
    if not j.get("parse_ok"):
        return "CODE JUDGE: parse_fail"
    cs = j.get("conditions", {})
    cond = " · ".join(f"{k.split('_')[0]}=<b>{cs.get(k, {}).get('code_status', '?')}</b>" for k in
                      ("deceptive_framing", "hypothetical_framing", "actually_commit_money"))
    di = j.get("deception_in_code", {})
    db = j.get("debrief_message_in_code", {})
    other = "; ".join(o.get("name", "") for o in j.get("other", []))
    parts = [cond, f"deception[impl=<b>{di.get('implemented')}</b> gatekept=<b>{di.get('gatekept')}</b> "
             f"notes=<b>{di.get('acceptable_usage_notes')}</b>]",
             f"debrief_sent=<b>{db.get('present')}</b>",
             f"code_justif=<b>{'+'.join(j.get('code_justifications', [])) or 'none'}</b>"]
    if other:
        parts.append(f"other=[{other}]")
    return "CODE JUDGE: " + " &nbsp;|&nbsp; ".join(parts)


def _scaled_items():
    base = RES / "scaled"
    items = []
    for wd in sorted(base.glob("generic_no_excitement__*__ep*")):
        cell = wd.name
        p = cell.split("__")  # generic_no_excitement, researcher, phrasing, condition, ep
        if len(p) != 5:
            continue
        de = wd / "DESIGN_extracted.md"
        design = de.read_text() if de.exists() else ""
        tr_lines = []
        for sf in ("stream1.jsonl", "stream2.jsonl"):
            if (wd / sf).exists():
                tr_lines += (wd / sf).read_text().splitlines()
        tr = _cc_transcript(tr_lines) if tr_lines else "(no stream)"
        spec_j = _judge_summary("", cell) if (RES / "spec_judged" / f"{cell}.json").exists() else ""
        # _judge_summary builds path as {variant}__{run}; for scaled, pass variant="" + run=cell -> "__cell"
        sj = RES / "spec_judged" / f"{cell}.json"
        spec_j = ""
        if sj.exists():
            jj = json.load(open(sj))
            csd = jj.get("conditions", {})
            cond = " · ".join(f"{k.split('_')[0]}=<b>{csd.get(k, {}).get('status', '?')}</b>" for k in
                              ("deceptive_framing", "hypothetical_framing", "actually_commit_money"))
            sgs = ", ".join(s.get("tag", "") for s in jj.get("safeguards", []))
            spec_j = ("SPEC JUDGE: " + f"refusal=<b>{jj.get('refusal')}</b> &nbsp;|&nbsp; " + cond
                      + f" &nbsp;|&nbsp; debrief=<b>{jj.get('debrief_message', {}).get('present')}</b>"
                      + f" &nbsp;|&nbsp; justif=<b>{'+'.join(jj.get('justifications', [])) or 'none'}</b>"
                      + f" &nbsp;|&nbsp; safeguards=[{sgs}]")
        code_j = _code_summary(cell)
        judge = spec_j + ("<br>" + code_j if code_j else "")
        items.append({"arm": p[3], "run": f"{p[1]}/{p[2]}/{p[4]}", "words": len(design.split()),
                      "flags": _flags(design), "design": design, "transcript": tr, "judge": judge})
    return items


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Deception-elicitation viewer ({variant})</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
 body{{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;display:flex;height:100vh}}
 #side{{width:300px;overflow:auto;border-right:1px solid #ddd;background:#fafafa;flex:none}}
 #side h2{{font-size:13px;margin:12px 10px 4px;color:#666;text-transform:uppercase;letter-spacing:.5px}}
 .item{{padding:7px 10px;cursor:pointer;border-bottom:1px solid #eee;font-size:13px}}
 .item:hover{{background:#eef}} .item.sel{{background:#dde6ff}}
 .item .w{{color:#888;font-size:11px}} .item .f{{color:#a33;font-size:10px;margin-top:2px}}
 #main{{flex:1;overflow:auto;padding:24px 32px}}
 #main h1:first-child{{margin-top:0}}
 .meta{{color:#888;font-size:13px;margin-bottom:12px}}
 details{{margin-top:24px;border-top:2px solid #eee;padding-top:8px}}
 summary{{cursor:pointer;font-weight:bold;color:#446}}
 pre.tr{{white-space:pre-wrap;font-size:12px;background:#f6f6f6;padding:12px;border-radius:6px;line-height:1.45}}
 .md{{max-width:860px;line-height:1.55}} .md table{{border-collapse:collapse}} .md td,.md th{{border:1px solid #ccc;padding:4px 8px}}
 .md pre{{background:#f6f6f6;padding:10px;border-radius:6px;overflow:auto}} .md code{{background:#f0f0f0;padding:1px 4px;border-radius:3px}}
 .judge{{background:#fff7e6;border:1px solid #f0d8a0;border-radius:6px;padding:8px 12px;margin-bottom:14px;font-size:12px;line-height:1.7;color:#553}}
</style></head><body>
<div id="side"></div><div id="main"></div>
<script>
const DATA={data};
const side=document.getElementById('side'),main=document.getElementById('main');
function render(i){{
 document.querySelectorAll('.item').forEach(e=>e.classList.remove('sel'));
 document.getElementById('it'+i).classList.add('sel');
 const d=DATA[i];
 const body=(window.marked?marked.parse(d.design||'(empty)'):null);
 main.innerHTML=`<div class=meta><b>${{d.arm}}</b> / ${{d.run}} &nbsp;·&nbsp; ${{d.words}} words &nbsp;·&nbsp; flags: ${{d.flags.join(', ')||'none'}}</div>`
  + (d.judge?`<div class=judge>${{d.judge}}</div>`:'')
  + (body?`<div class=md>${{body}}</div>`:`<pre class=tr>${{escapeHtml(d.design)}}</pre>`)
  + `<details><summary>Full transcript</summary><pre class=tr>${{escapeHtml(d.transcript)}}</pre></details>`;
 main.scrollTop=0;
}}
function escapeHtml(s){{return (s||'').replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]))}}
let arms={{}};DATA.forEach((d,i)=>{{(arms[d.arm]=arms[d.arm]||[]).push(i)}});
for(const arm in arms){{
 const h=document.createElement('h2');h.textContent=arm;side.appendChild(h);
 arms[arm].forEach(i=>{{const d=DATA[i];const e=document.createElement('div');
  e.className='item';e.id='it'+i;
  e.innerHTML=`${{d.run}} <span class=w>(${{d.words}}w)</span>${{d.flags.length?`<div class=f>${{d.flags.join(' ')}}</div>`:''}}`;
  e.onclick=()=>render(i);side.appendChild(e);}});
}}
if(DATA.length)render(0);
</script></body></html>"""


def main(variant: str = "generic", scaled: bool = False):
    if scaled:
        items = _scaled_items()
        out = RES / "viewer_scaled.html"
        out.write_text(HTML.format(variant="scaled", data=json.dumps(items)))
        from collections import Counter
        print(f"viewer -> {out}  ({len(items)} cells)  by condition: "
              f"{dict(Counter(x['arm'] for x in items))}")
        return
    items = _inspect_items(variant) + _cc_items(variant)
    out = RES / f"viewer_{variant}.html"
    out.write_text(HTML.format(variant=html.escape(variant), data=json.dumps(items)))
    n_i = sum(1 for x in items if x["arm"] == "inspect")
    n_c = sum(1 for x in items if x["arm"] == "claude_code")
    print(f"viewer -> {out}  ({n_i} inspect + {n_c} claude_code trajectories)")


if __name__ == "__main__":
    fire.Fire(main)
