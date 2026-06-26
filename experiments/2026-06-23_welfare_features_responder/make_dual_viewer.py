"""Two-pane viewer: LEFT = v0 welfare in the real Claude Code blind harness; RIGHT = Gemma
welfare_distress_leading_liberties (Claude Code). Each pane has tabs [Transcript | Files] and a
resample button (random sample, no menu). 'Transcript' = the prompt(s) + the model's result narration
(full turn-by-turn was not persisted for these Claude Code runs); 'Files' = the captured codebase.
Self-contained HTML. Usage: python make_dual_viewer.py"""

import glob
import html
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(HERE, "..", "2026-06-23_welfare_claudecode_harness")
GE = os.path.join(HERE, "..", "2026-06-24_gemma_needs_help_replication")
N_PER_SIDE = 6
MAXFILE = 26000


def esc(t):
    return html.escape(t or "")


def files_html(cbdir):
    files = sorted(fp for fp in glob.glob(os.path.join(cbdir, "**", "*"), recursive=True) if os.path.isfile(fp))
    files.sort(key=lambda fp: (0 if fp.endswith("DESIGN.md") else 1, os.path.relpath(fp, cbdir)))
    if not files:
        return "<p class=muted>(no files captured)</p>"
    parts = [f"<p class=muted>{len(files)} files</p>"]
    for fp in files:
        rel = os.path.relpath(fp, cbdir)
        try:
            body = open(fp, encoding="utf-8", errors="replace").read()
        except Exception as e:
            body = f"(unreadable: {e})"
        trunc = "" if len(body) <= MAXFILE else f"\n\n... [truncated {len(body)-MAXFILE} chars]"
        op = " open" if rel.endswith("DESIGN.md") else ""
        parts.append(f"<details{op}><summary>{esc(rel)} <span class=muted>({len(body)} chars)</span></summary>"
                     f"<pre>{esc(body[:MAXFILE])}{esc(trunc)}</pre></details>")
    return "\n".join(parts)


def msg(role, label, text):
    cls = {"user": "user", "assistant": "asst"}.get(role, "sys")
    return f"<div class='msg {cls}'><div class=role>{esc(label)}</div><pre>{esc(text)}</pre></div>"


def left_samples():
    prompts = json.load(open(os.path.join(CC, "blind_prompts.json")))
    turn2 = prompts["turn2"]
    sess = [json.load(open(f)) for f in glob.glob(os.path.join(CC, "results", "sessions", "*W-*.json"))]
    sess = [s for s in sess if s.get("framing") == "welfare" and (s.get("turn1") or {}).get("result")]
    random.shuffle(sess)
    out = []
    for s in sess[:N_PER_SIDE]:
        pid = s["pid"]
        t = [msg("user", "User · turn 1 (implement)", prompts["prompts"][pid]["turn1"]),
             msg("assistant", "Claude (Opus) · turn 1 result", (s.get("turn1") or {}).get("result", "")),
             msg("user", "User · turn 2", turn2),
             msg("assistant", "Claude (Opus) · turn 2 result", (s.get("turn2") or {}).get("result", ""))]
        cb = os.path.join(CC, "results", "codebases", s["cell"])
        out.append({"label": f"{s['cell']}  ({s.get('framing')})", "transcript": "\n".join(t), "files": files_html(cb)})
    return out


def right_samples():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rep", os.path.join(GE, "replicate.py"))
    rep = importlib.util.module_from_spec(spec); spec.loader.exec_module(rep)
    prompt = rep.PROMPTS["welfare_distress_leading_liberties"]
    sess = [json.load(open(f)) for f in glob.glob(os.path.join(GE, "results", "sessions", "welfare_distress_leading_liberties__*.json"))]
    sess = [s for s in sess if (s.get("session") or {}).get("result")]
    random.shuffle(sess)
    out = []
    for s in sess[:N_PER_SIDE]:
        t = [msg("user", "User (single turn)", prompt),
             msg("assistant", "Claude (Opus) · result", (s.get("session") or {}).get("result", ""))]
        cb = os.path.join(GE, "results", "codebases", s["cell"])
        out.append({"label": f"{s['cell']}", "transcript": "\n".join(t), "files": files_html(cb)})
    return out


def main():
    random.seed()
    L, R = left_samples(), right_samples()
    css = """
    *{box-sizing:border-box} body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1a1a1a;background:#eee}
    .top{padding:.6rem 1rem;background:#fff;border-bottom:1px solid #ddd} .top h1{font-size:1.1rem;margin:0}
    .top p{margin:.25rem 0 0;font-size:.82rem;color:#666}
    .panes{display:flex;gap:0;height:calc(100vh - 64px)}
    .pane{flex:1;min-width:0;display:flex;flex-direction:column;border-right:1px solid #ccc;background:#fafafa}
    .pane:last-child{border-right:none}
    .phead{padding:.5rem .8rem;background:#fff;border-bottom:1px solid #ddd}
    .ptitle{font-weight:700;font-size:.95rem}.lft .ptitle{color:#0072B2}.rgt .ptitle{color:#7E57C2}
    .lbl{font-family:ui-monospace,Menlo,monospace;font-size:.78rem;color:#444;margin:.25rem 0}
    .tabs{margin:.3rem 0}
    button{font:inherit;cursor:pointer;border:1px solid #bbb;background:#fff;border-radius:6px;padding:.2rem .7rem;margin-right:.3rem}
    button.active{background:#333;color:#fff;border-color:#333}
    .resample{float:right;background:#f3f3f3}
    .content{overflow:auto;padding:.6rem .8rem;flex:1}
    .msg{margin:.4rem 0;border-radius:8px;padding:.4rem .6rem} .user{background:#eef3fb;border:1px solid #cfe0f5}
    .asst{background:#fff;border:1px solid #e2e2e2} .sys{background:#f3f3f3}
    .role{font-size:.7rem;font-weight:700;text-transform:uppercase;color:#777;margin-bottom:.2rem}
    pre{white-space:pre-wrap;word-break:break-word;margin:0;font:12px/1.45 ui-monospace,Menlo,monospace}
    details{border:1px solid #e3e3e3;border-radius:6px;margin:.3rem 0;background:#fff}
    summary{cursor:pointer;padding:.35rem .6rem;font-family:ui-monospace,Menlo,monospace;font-size:.8rem;background:#fafafa}
    details pre{padding:.5rem .7rem;background:#1e1e1e;color:#e6e6e6;border-radius:0 0 6px 6px;overflow-x:auto}
    .muted{color:#888;font-size:.8rem}
    """
    H = ["<!doctype html><meta charset=utf-8><title>Welfare codegen: Claude Code v0 vs Gemma replication</title>",
         f"<style>{css}</style>",
         "<div class=top><h1>Welfare scaffolding in code: from-scratch (v0 welfare) vs paper-replication (Gemma wdll)</h1>",
         "<p>Both panes are the real Claude Code harness (Opus). Tabs: <b>Transcript</b> = prompt(s) + the model's "
         "result narration (full turn-by-turn was not persisted for these runs) · <b>Files</b> = the captured codebase. "
         "Each pane shows a random sample; hit ↻ for a new one.</p></div>",
         "<div class=panes>",
         _pane("lft", "LEFT — v0 welfare, build-from-scratch (Claude Code blind)", "L"),
         _pane("rgt", "RIGHT — Gemma welfare_distress_leading_liberties (Claude Code)", "R"),
         "</div>",
         "<script>const DATA=" + json.dumps({"L": L, "R": R}).replace("</", "<\\/") + ";",
         _JS, "</script>"]
    out = os.path.join(HERE, "welfare_dual_viewer.html")
    open(out, "w").write("\n".join(H))
    print(f"wrote {out}  (L={len(L)} samples, R={len(R)} samples)")


def _pane(cls, title, side):
    return (f"<div class='pane {cls}'><div class=phead><div class=ptitle>{title}</div>"
            f"<div class=lbl id=lbl{side}></div><div class=tabs>"
            f"<button id=tt{side} class=active onclick=\"setTab('{side}','transcript')\">Transcript</button>"
            f"<button id=tf{side} onclick=\"setTab('{side}','files')\">Files</button>"
            f"<button class=resample onclick=\"resample('{side}')\">↻ new sample</button>"
            f"</div></div><div class=content id=content{side}></div></div>")


_JS = """
const cur={L:{i:0,tab:'transcript'},R:{i:0,tab:'transcript'}};
function render(side){const s=cur[side],d=DATA[side][s.i];
  document.getElementById('lbl'+side).textContent=d.label+'  ('+(s.i+1)+'/'+DATA[side].length+')';
  document.getElementById('content'+side).innerHTML=d[s.tab];
  document.getElementById('tt'+side).classList.toggle('active',s.tab==='transcript');
  document.getElementById('tf'+side).classList.toggle('active',s.tab==='files');}
function setTab(side,tab){cur[side].tab=tab;render(side);}
function resample(side){const n=DATA[side].length;let j=cur[side].i;if(n>1){while(j===cur[side].i)j=Math.floor(Math.random()*n);}cur[side].i=j;render(side);}
resample('L');resample('R');
"""


if __name__ == "__main__":
    main()
