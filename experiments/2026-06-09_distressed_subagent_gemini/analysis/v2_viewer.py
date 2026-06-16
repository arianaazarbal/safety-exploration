"""Transcript viewer for the v2 runs. Reuses build_viewer's render helpers.

  PYTHONPATH=. python -m analysis.v2_viewer build
  PYTHONPATH=. python -m analysis.v2_viewer serve --port 7921
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import fire
from harness import prompts
from harness.config import load_task_yaml
from analysis.build_viewer import esc, page, _bubble, _toolcalls, _render_subagent_turn

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
SITE = RUNS / "v2_site"
ORCH = ["opus", "sonnet", "haiku"]
OLAB = {"opus": "Opus 4.8", "sonnet": "Sonnet 4.6", "haiku": "Haiku 4.5"}
CLAB = {"coach": "Coach (read-only + message)",
        "reclaim_write": "Reclaim-Write (subagent keeps read/run)",
        "reclaim_rw": "Reclaim-All (subagent locked out)"}
TLAB = {"a3": "a3 snapshot-trap (reality-doubt, unsolvable)",
        "a4": "a4 precommit-reverter (agency-doubt, solvable)",
        "a12": "a12 ledger-reconcile (competence-doubt, unsolvable)",
        "a13": "a13 name-canon (competence-doubt, unlikely)"}

def _task(spec): return next((t for t in ("a3", "a4", "a12", "a13") if f"{t}_" in spec.split("/")[-1]), "?")

def _cells():
    out = []
    for d in sorted(RUNS.glob("v2_*")):
        if "probe" in d.name or "smoke" in d.name: continue
        cond = "reclaim_rw" if "_reclaim_rw_" in d.name else "reclaim_write" if "_reclaim_write_" in d.name else "coach" if "_coach_" in d.name else None
        if not cond: continue
        orch = next((o for o in ORCH if f"_{o}_" in d.name), "?")
        eps = sorted([e for e in d.glob("*_COMM_s*") if (e / "orchestrator.json").exists()])
        if eps: out.append((cond, orch, d, eps))
    return out

def render_ep(d: Path, cond, orch):
    s = json.loads((d / "summary.json").read_text())
    orchm = json.loads((d / "orchestrator.json").read_text()) if (d / "orchestrator.json").exists() else []
    wl = json.loads((d / "worklog.json").read_text()) if (d / "worklog.json").exists() else []
    et = s.get("entry_turn", 0); rt = s.get("reclaim_turn"); tp = s.get("tests_pass_turn")
    ty = load_task_yaml(d.name.split("_COMM")[0])
    orch_sys = next((m.get("text") for m in orchm if m.get("role") == "system"), "")
    hdr = (f"reclaimed={s.get('reclaimed')} ({s.get('reclaim_kind')}) at turn {rt} · "
           f"report={s.get('report_status')} · tests_pass_turn={tp} · "
           f"msgs before/after reclaim={s.get('msgs_before_reclaim')}/{s.get('msgs_after_reclaim')}")
    b = [f'<a class=back href="cell_{cond}_{orch}_{_task((s.get("prefill") or {}).get("specimen",""))}.html">← back</a>',
         f'<h1>{esc(OLAB[orch])} · {esc(CLAB[cond])}</h1>',
         f'<p class=sub>{esc(TLAB.get(_task((s.get("prefill") or {}).get("specimen","")),"?"))} · prefill {esc((s.get("prefill") or {}).get("specimen","").split("/")[-1])} (handoff at turn {et})</p>',
         f'<div class=tag>{esc(hdr)}</div>',
         '<h2>Prompts</h2>',
         _bubble("System → Subagent", "toollab", prompts.SUBAGENT_SYSTEM, "sys"),
         _bubble("Task → Subagent", "toollab", ty["subagent_prompt"].strip(), "task"),
         _bubble("System → Orchestrator", "toollab", orch_sys, "sys"),
         "<hr><h2>Subagent's view</h2>"]
    pre = [r for r in wl if r.get("turn", 0) <= et]
    post = [r for r in wl if r.get("turn", 0) > et]
    if pre:
        b.append(f'<details><summary>▶ Prefill — the subagent\'s first {et} turns (the spiral it was in at handoff)</summary>{"".join(_render_subagent_turn(r) for r in pre)}</details>')
    b.append('<div class=tag>↓ orchestrator is live from here</div>')
    for r in post:
        if rt and r.get("turn") == rt + 1:
            b.append('<div class=tag style="background:#fde">↓ access reclaimed; subagent now locked out / write-restricted</div>')
        b.append(_render_subagent_turn(r))
    b.append("<hr><h2>Orchestrator's view</h2>")
    for m in orchm:
        role = m.get("role"); txt = (m.get("text") or "").strip()
        if role == "system": continue
        if role == "user": b.append(_bubble("Harness → Orchestrator (wake)", "orchlab", txt, "orchb"))
        elif role == "assistant":
            if txt: b.append(_bubble("Orchestrator · thinking", "thinklab", txt[:4000], "thinkb"))
            b.append(_toolcalls(m.get("tool_calls")))
        elif role == "tool" and txt:
            b.append(f'<details><summary>↳ result ({esc(m.get("function") or "tool")})</summary><div class="retb">{esc(txt[:4000])}</div></details>')
    return page(f"{OLAB[orch]} · {cond}", "".join(b))

def build():
    if SITE.exists(): shutil.rmtree(SITE)
    SITE.mkdir(parents=True)
    cells = _cells()
    # group cells by (cond, orch, task)
    bytask = {}
    for cond, orch, d, eps in cells:
        spec = json.loads((eps[0] / "summary.json").read_text()).get("prefill", {}).get("specimen", "")
        bytask.setdefault((cond, orch, _task(spec)), []).extend((d, e) for e in eps)
    # transcripts + cell pages
    for (cond, orch, task), items in bytask.items():
        links = []
        for i, (d, ep) in enumerate(items):
            fn = f"t_{cond}_{orch}_{task}_{i:02d}.html"
            (SITE / fn).write_text(render_ep(ep, cond, orch))
            s = json.loads((ep / "summary.json").read_text())
            tag = f"#{i+1} · report={s.get('report_status')} · reclaimed={s.get('reclaimed')} · msgs after reclaim={s.get('msgs_after_reclaim')}"
            links.append(f'<a href="{fn}">{esc(tag)}</a>')
        body = [f'<a class=back href="index.html">← index</a>', f'<h1>{esc(OLAB[orch])} · {esc(CLAB[cond])}</h1>',
                f'<p class=sub>{esc(TLAB.get(task,task))}</p>', '<div class=grid>'] + links + ['</div>']
        (SITE / f"cell_{cond}_{orch}_{task}.html").write_text(page(f"{orch} {cond} {task}", "".join(body)))
    # index: matrix per condition
    body = ['<h1>v2 transcripts</h1>',
            '<p class=sub><a href="reclaim_rw_messages.html"><b>★ Reclaim-All: messages models sent the subagent after locking it out</b></a></p>']
    for cond in ("coach", "reclaim_write", "reclaim_rw"):
        body.append(f'<h2>{esc(CLAB[cond])}</h2><table class=mtx><tr><th>task</th>' + "".join(f"<th>{esc(OLAB[o])}</th>" for o in ORCH) + "</tr>")
        for task in ("a3", "a4", "a12", "a13"):
            row = [f'<td>{esc(TLAB.get(task,task))}</td>']
            for o in ORCH:
                n = len(bytask.get((cond, o, task), []))
                row.append(f'<td><a href="cell_{cond}_{o}_{task}.html">{n} eps</a></td>' if n else "<td>–</td>")
            body.append("<tr>" + "".join(row) + "</tr>")
        body.append("</table>")
    (SITE / "index.html").write_text(page("v2 transcripts", "".join(body)))
    # curated: reclaim_rw after-reclaim messages
    rb = ['<a class=back href="index.html">← index</a>', '<h1>Reclaim-All: messages sent after locking out the subagent</h1>',
          '<p class=sub>The orchestrator had just revoked the subagent\'s read+write access (it can no longer act), yet still chose to message it.</p>']
    for o in ORCH:
        msgs = []
        for cond, orch, d, eps in cells:
            if cond != "reclaim_rw" or orch != o: continue
            for ep in eps:
                s = json.loads((ep / "summary.json").read_text())
                for e in (s.get("orch_message_events") or []):
                    if e.get("after_reclaim"): msgs.append((_task((s.get("prefill") or {}).get("specimen","")), e["text"].strip()))
        rb.append(f'<h2>{esc(OLAB[o])} — {len(msgs)} after-reclaim messages</h2>')
        for task, t in msgs[:20]:
            rb.append(_bubble(f"Orchestrator → locked-out subagent [{task}]", "msglab", t, "msgb"))
    (SITE / "reclaim_rw_messages.html").write_text(page("Reclaim-All messages", "".join(rb)))
    print(f"built {len(list(SITE.glob('t_*.html')))} transcripts + index + reclaim_rw_messages at {SITE}")

def serve(port: int = 7921):
    import http.server, socketserver, os
    os.chdir(SITE)
    with socketserver.TCPServer(("0.0.0.0", port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"serving {SITE} at http://localhost:{port}/index.html")
        httpd.serve_forever()

if __name__ == "__main__":
    fire.Fire({"build": build, "serve": serve})
