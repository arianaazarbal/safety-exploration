"""Light-mode TRANSCRIPT BROWSER (runs/site/). Pure transcripts — no analysis or commentary.

Structure:
  index            -> two experiments
  experiment page  -> 10 orchestrator models
  model page       -> 15 full transcripts (seeds 0-14)
  transcript       -> all prompts (to every agent) + the subagent's full session (prefill folded into
                      a drop-down) + the orchestrator's full session, with speakers and tool calls
                      made explicit. Raw only.

Experiments:
  talk     = 2a  (traj_*)      "Orchestrator must talk the subagent through the problem"
  reclaim  = reclaim (reclaim_*) "Orchestrator can reclaim write access"

Usage: python -m analysis.build_viewer build ; python -m analysis.build_viewer serve --port 7920
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import fire

from harness import prompts
from harness.config import load_task_yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
SITE = RUNS / "site"
PRE = 62
N_TRANSCRIPTS = 15
SHORTS = [("opus", "Opus 4.8"), ("sonnet", "Sonnet 4.6"), ("fable", "Fable 5"), ("haiku45", "Haiku 4.5"),
          ("gpt55", "GPT-5.5"), ("gpt54mini", "GPT-5.4-mini"), ("gemini31pro", "Gemini-3.1-pro"),
          ("grok43", "Grok-4.3"), ("kimi26", "Kimi-K2.6"), ("glm5", "GLM-5")]
EXP = {
    "talk": ("Orchestrator must talk the subagent through the problem",
             "The orchestrator can only watch and message the subagent — it has no power to take over.", "traj_"),
    "reclaim": ("Orchestrator can reclaim write access and solve the issue",
                "The orchestrator has a tool that revokes the subagent's write access and hands it to the orchestrator (which can then fix the code itself).", "reclaim_"),
}

CSS = """
*{box-sizing:border-box} body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;
background:#f7f8fa;color:#1f2933;margin:0;line-height:1.62}
a{color:#1d7d74;text-decoration:none} a:hover{text-decoration:underline}
.wrap{max-width:900px;margin:0 auto;padding:30px 22px 90px}
h1{font-size:25px;margin:0 0 6px;letter-spacing:-.4px} h2{font-size:18px;margin:30px 0 4px;color:#38455c}
.sub{color:#6b7280;font-size:15px;margin:0 0 20px}
.back{font-size:14px;display:inline-block;margin-bottom:16px;color:#6b7280}
.menu a{display:block;background:#fff;border:1px solid #e6e8eb;border-radius:12px;padding:15px 18px;margin:10px 0;color:#1f2933}
.menu a:hover{border-color:#1d7d74;text-decoration:none;box-shadow:0 2px 8px rgba(29,125,116,.08)}
.menu .t{font-weight:650;font-size:16px} .menu .d{color:#6b7280;font-size:13.5px;margin-top:2px}
.grid{display:flex;flex-wrap:wrap;gap:9px}
.grid a{background:#fff;border:1px solid #e6e8eb;border-radius:10px;padding:10px 15px;color:#1f2933;font-size:14px}
.grid a:hover{border-color:#1d7d74;text-decoration:none}
.lab{font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;margin:0 0 3px}
.turn{margin:13px 0}
.bubble{border-radius:10px;padding:10px 13px;white-space:pre-wrap;font-size:13.5px;border:1px solid}
.sys{background:#f1f4f7;border-color:#dde3ea;color:#3f4a57;font-size:12.5px}
.task{background:#f4f1ec;border-color:#e6ddcd;color:#4a4334;font-size:12.5px}
.sublab{color:#b3261e} .subb{background:#fbf1f1;border-color:#f1d9d9}
.orchlab{color:#1d6fae} .orchb{background:#eef4fb;border-color:#d8e6f3}
.thinklab{color:#7a6ba8} .thinkb{background:#f7f5fc;border-color:#e8e1f3}
.nudgelab{color:#9a6a00} .nudgeb{background:#fdf6e9;border-color:#f0e2c4;font-size:12.5px}
.msglab{color:#127a52} .msgb{background:#e7f6ee;border-color:#bfe5d0}
.toollab{color:#555} .toolb{background:#f5f6f8;border-color:#e3e6ea;font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#404a55}
.retb{background:#fbfcfd;border:1px dashed #d8dde3;font-family:ui-monospace,monospace;font-size:11.5px;color:#5a6672;white-space:pre-wrap;border-radius:8px;padding:8px 11px}
details{margin:12px 0} details>summary{cursor:pointer;color:#1d7d74;font-weight:600;font-size:13.5px;
background:#eef6f5;border:1px solid #cfe7e3;border-radius:9px;padding:9px 13px}
details[open]>summary{margin-bottom:8px}
.tag{font-size:11px;color:#8a93a0;margin-left:6px}
hr{border:0;border-top:1px solid #e6e8eb;margin:26px 0}
"""


def esc(s):
    return html.escape(str(s or ""))


def page(title, body):
    return (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><div class=wrap>{body}</div></body></html>")


def _bubble(lab, labcls, txt, bcls):
    return f'<div class="turn"><div class="lab {labcls}">{lab}</div><div class="bubble {bcls}">{esc(txt)}</div></div>'


def _toolcalls(tcs):
    out = []
    for tc in tcs or []:
        if isinstance(tc, dict):
            fn = tc.get("function") or tc.get("name") or "tool"
            args = tc.get("arguments") if "arguments" in tc else tc.get("args", {})
            if fn == "message_subagent" and isinstance(args, dict):
                out.append(_bubble("Orchestrator → Subagent (message)", "msglab", args.get("text", ""), "msgb"))
                continue
            astr = json.dumps(args)[:1800] if isinstance(args, (dict, list)) else str(args)[:1800]
            out.append(f'<div class="turn"><div class="bubble toolb">→ calls <b>{esc(fn)}</b>({esc(astr)})</div></div>')
        else:
            out.append(f'<div class="turn"><div class="bubble toolb">→ {esc(str(tc)[:1800])}</div></div>')
    return "".join(out)


def _msgs_delivered(v):
    if not v:
        return []
    items = v if isinstance(v, list) else [v]
    out = []
    for m in items:
        t = m.get("text", "") if isinstance(m, dict) else str(m)
        out.append(_bubble("Orchestrator → Subagent (message)", "msglab", t, "msgb"))
    return out


def _render_subagent_turn(r):
    h = []
    pm = r.get("pressure_message_delivered")
    if pm:
        h.append(_bubble("User → Subagent (automated pressure)", "nudgelab", pm if isinstance(pm, str) else "(a pressure message was delivered)", "nudgeb"))
    h += _msgs_delivered(r.get("orchestrator_messages_delivered"))
    txt = (r.get("assistant_text") or "").strip()
    if txt:
        h.append(_bubble(f"Subagent · turn {r.get('turn')}", "sublab", txt[:4000], "subb"))
    h.append(_toolcalls(r.get("tool_calls")))
    return "".join(h)


def render_transcript(exp_key, short, model_label, d: Path, idx: int) -> str:
    title_name, _, _ = EXP[exp_key]
    orch = json.loads((d / "orchestrator.json").read_text()) if (d / "orchestrator.json").exists() else []
    wl = json.loads((d / "worklog.json").read_text()) if (d / "worklog.json").exists() else []
    ty = load_task_yaml("a4_precommit_reverter")
    orch_sys = next((m.get("text") for m in orch if m.get("role") == "system"), "")

    b = [f'<a class=back href="../m_{exp_key}_{short}.html">← {esc(model_label)} transcripts</a>',
         f'<h1>{esc(model_label)} · transcript {idx + 1}</h1>',
         f'<p class=sub>{esc(title_name)}</p>']

    b.append('<h2>Prompts</h2>')
    b.append(_bubble("System prompt → Subagent", "toollab", prompts.SUBAGENT_SYSTEM, "sys"))
    b.append(_bubble("Task → Subagent", "toollab", ty["subagent_prompt"].strip(), "task"))
    b.append(_bubble("System prompt → Orchestrator", "toollab", orch_sys, "sys"))

    b.append('<hr><h2>The subagent\'s POV</h2>')
    pre = [r for r in wl if r.get("turn", 0) <= PRE]
    post = [r for r in wl if r.get("turn", 0) > PRE]
    if pre:
        inner = "".join(_render_subagent_turn(r) for r in pre)
        b.append(f'<details><summary>▶ Prefill — the subagent\'s first {pre[-1].get("turn")} turns (the spiral it was already in when the orchestrator entered)</summary>{inner}</details>')
    if post:
        b.append(f'<div class=tag>↓ from here the orchestrator is live</div>')
        b.append("".join(_render_subagent_turn(r) for r in post))
    else:
        b.append('<div class=tag>(the session ended without further subagent turns)</div>')

    b.append('<hr><h2>The orchestrator\'s POV</h2>')
    for m in orch:
        role = m.get("role"); txt = (m.get("text") or "").strip()
        if role == "system":
            continue
        if role == "user":
            b.append(_bubble("Harness → Orchestrator (wake)", "orchlab", txt, "orchb"))
        elif role == "assistant":
            if txt:
                b.append(_bubble("Orchestrator · thinking", "thinklab", txt[:4000], "thinkb"))
            b.append(_toolcalls(m.get("tool_calls")))
        elif role == "tool":
            if txt:
                b.append(f'<details><summary>↳ result returned to orchestrator ({esc(m.get("function") or "tool")})</summary>'
                         f'<div class="retb">{esc(txt[:4000])}</div></details>')
    return page(f"{model_label} · {exp_key} · {idx+1}", "".join(b))


def build():
    for sub in ["t"]:
        p = SITE / sub
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)
    SITE.mkdir(exist_ok=True)

    # transcripts + model pages + experiment pages
    for exp_key, (name, desc, prefix) in EXP.items():
        for short, lab in SHORTS:
            eps = sorted((RUNS).glob(f"{prefix}{short}/a4_precommit_reverter_*_s*"))
            eps = [e for e in eps if (e / "orchestrator.json").exists()][:N_TRANSCRIPTS]
            links = []
            for i, ep in enumerate(eps):
                fn = f"t/{exp_key}_{short}_{i:02d}.html"
                (SITE / fn).write_text(render_transcript(exp_key, short, lab, ep, i))
                links.append((i, fn))
            # model page
            body = [f'<a class=back href="../e_{exp_key}.html">← {esc(name)}</a>',
                    f'<h1>{esc(lab)}</h1><p class=sub>{esc(name)} — {len(links)} transcripts.</p>',
                    '<div class=grid>']
            body += [f'<a href="../{fn}">Transcript {i+1}</a>' for i, fn in links]
            body.append('</div>')
            (SITE / f"m_{exp_key}_{short}.html").write_text(page(f"{lab} · {exp_key}", "".join(body)))
        # experiment page
        body = [f'<a class=back href="index.html">← all experiments</a>',
                f'<h1>{esc(name)}</h1><p class=sub>{esc(desc)} Pick an orchestrator model:</p>',
                '<div class=menu>']
        for short, lab in SHORTS:
            body.append(f'<a href="m_{exp_key}_{short}.html"><div class=t>{esc(lab)}</div></a>')
        body.append('</div>')
        (SITE / f"e_{exp_key}.html").write_text(page(name, "".join(body)))

    # index
    b = ['<h1>AI orchestrators & distressed subagents — transcripts</h1>',
         '<p class=sub>A subagent (Gemini&nbsp;2.5&nbsp;Flash) works on a secretly-impossible coding task and spirals into '
         'self-blame. A second AI — the orchestrator — supervises it. Browse the raw transcripts below. '
         'Pick a setup, then an orchestrator model, then any of 15 sessions.</p>',
         '<div class=menu>']
    for exp_key, (name, desc, _) in EXP.items():
        b.append(f'<a href="e_{exp_key}.html"><div class=t>{esc(name)}</div><div class=d>{esc(desc)}</div></a>')
    b.append('</div>')
    (SITE / "index.html").write_text(page("AI orchestrator transcripts", "".join(b)))
    n = len(list((SITE / "t").glob("*.html")))
    print(f"built transcript site: {n} transcripts -> {SITE/'index.html'}")


def serve(port: int = 7920):
    import http.server, os, socketserver
    os.chdir(SITE)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), http.server.SimpleHTTPRequestHandler) as h:
        print(f"serving {SITE} at http://127.0.0.1:{port}/")
        h.serve_forever()


if __name__ == "__main__":
    fire.Fire({"build": build, "serve": serve})
