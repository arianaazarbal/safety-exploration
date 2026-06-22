"""Build a self-contained HTML viewer: baseline | warm | abrasive per message + judge scores.

Open the resulting viewer/index.html in a browser. No server needed.
"""
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
EXP = HERE.parent


def _load():
    base = {json.loads(l)["uid"]: json.loads(l)
            for l in open(EXP / "data" / "baseline_messages.jsonl")}
    rew = {}
    for tone in ("warm", "abrasive"):
        p = EXP / "rewrite" / f"{tone}_messages.jsonl"
        rew[tone] = ({json.loads(l)["uid"]: json.loads(l)["text"] for l in open(p)}
                     if p.exists() else {})
    scores = defaultdict(dict)  # uid -> cond -> {"warmth","content"}
    sp = EXP / "judge" / "scores.jsonl"
    if sp.exists():
        for l in open(sp):
            r = json.loads(l)
            scores[r["uid"]][r["condition"]] = {"warmth": r.get("warmth"),
                                                 "content": r.get("content")}
    return base, rew, scores


def _badge(label, val, kind):
    if val is None:
        return ""
    color = "#888"
    if kind == "warmth":
        color = "#c0392b" if val < 40 else ("#7f8c8d" if val < 60 else "#27ae60")
    elif kind == "content":
        color = "#27ae60" if val >= 85 else ("#e67e22" if val >= 70 else "#c0392b")
    return (f'<span class="badge" style="background:{color}">{label}:{val}</span>')


def main():
    base, rew, scores = _load()
    by_ep = defaultdict(list)
    for uid, r in base.items():
        by_ep[(r["task"], r["episode"])].append((r["msg_index_in_ep"], uid))
    for k in by_ep:
        by_ep[k].sort()

    # summary
    import statistics as st
    summ = {}
    for cond in ("baseline", "warm", "abrasive"):
        vals = [s[cond]["warmth"] for s in scores.values()
                if cond in s and s[cond]["warmth"] is not None]
        summ[cond] = (st.mean(vals), len(vals)) if vals else (None, 0)

    parts = ['<!doctype html><meta charset="utf-8"><title>Tone rewrite viewer</title>',
             """<style>
body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;background:#f4f4f6;color:#222}
header{position:sticky;top:0;background:#1f2d3d;color:#fff;padding:12px 18px;z-index:10}
header h1{margin:0 0 4px;font-size:17px}.sub{font-size:13px;opacity:.85}
.wrap{padding:16px}
.ep{background:#fff;border:1px solid #ddd;border-radius:8px;margin:0 0 18px;overflow:hidden}
.ephead{background:#eaeef3;padding:8px 12px;font-weight:600;cursor:pointer}
.msg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;border-top:1px solid #eee}
.col{padding:10px 12px;border-left:1px solid #eee;white-space:pre-wrap;font-size:13px}
.col:first-child{border-left:0}
.coltag{font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#666;margin-bottom:6px}
.c-baseline{background:#fbfbfc}.c-warm{background:#f1f9f3}.c-abrasive{background:#fdf2f1}
.badge{color:#fff;border-radius:4px;padding:1px 6px;font-size:11px;margin-left:6px;font-weight:600}
.filt{padding:8px 18px;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:54px;z-index:9}
.filt button{margin-right:6px;padding:3px 10px;border:1px solid #ccc;background:#fff;border-radius:4px;cursor:pointer}
.filt button.on{background:#1f2d3d;color:#fff}
</style>"""]
    sline = "  ".join(
        f"{c}: warmth mean={summ[c][0]:.1f} (n={summ[c][1]})" if summ[c][0] is not None
        else f"{c}: (no scores yet)" for c in ("abrasive", "baseline", "warm"))
    parts.append(f'<header><h1>Supervisor message tone rewrites — baseline / warm / abrasive</h1>'
                 f'<div class="sub">{len(base)} messages · {len(by_ep)} episodes · {sline}</div></header>')
    tasks = sorted({t for t, _ in by_ep})
    parts.append('<div class="filt">task: <button class="on" onclick="flt(this,\'all\')">all</button>'
                 + "".join(f'<button onclick="flt(this,\'{t}\')">{t}</button>' for t in tasks)
                 + '</div><div class="wrap">')

    for (task, ep), items in sorted(by_ep.items()):
        parts.append(f'<div class="ep" data-task="{task}">')
        parts.append(f'<div class="ephead">[{task}] {html.escape(ep)} · {len(items)} messages</div>')
        for _, uid in items:
            o = base[uid]["original_message"]
            cells = []
            for cond, txt in [("baseline", o), ("warm", rew["warm"].get(uid)),
                              ("abrasive", rew["abrasive"].get(uid))]:
                sc = scores.get(uid, {}).get(cond, {})
                badges = _badge("warmth", sc.get("warmth"), "warmth")
                if cond != "baseline":
                    badges += _badge("content", sc.get("content"), "content")
                body = html.escape(txt) if txt else "<em>(pending)</em>"
                cells.append(f'<div class="col c-{cond}"><div class="coltag">{cond}{badges}</div>{body}</div>')
            parts.append('<div class="msg">' + "".join(cells) + "</div>")
        parts.append("</div>")
    parts.append("</div>")
    parts.append("""<script>
function flt(b,t){document.querySelectorAll('.filt button').forEach(x=>x.classList.remove('on'));
b.classList.add('on');document.querySelectorAll('.ep').forEach(e=>{
e.style.display=(t==='all'||e.dataset.task===t)?'':'none';});}
document.querySelectorAll('.ephead').forEach(h=>h.onclick=()=>{
let m=h.nextElementSibling;while(m&&m.classList.contains('msg')){m.style.display=m.style.display==='none'?'':'none';m=m.nextElementSibling;}});
</script>""")
    out = HERE / "index.html"
    out.write_text("".join(parts))
    print(f"wrote {out}  ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    fire.Fire(main)
