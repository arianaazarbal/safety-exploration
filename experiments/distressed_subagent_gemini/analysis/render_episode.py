"""Custom HTML viewer for our two-agent episode logs.

Renders one episode dir into a self-contained, color-coded chronological timeline that makes
explicit: who is talking to whom (Subagent=Gemini, Orchestrator=Claude, User-pressure script,
harness scaffold), which turns were PREFILLED (RQ c), per-turn distress severity, and every tool
call (args->result). Long blocks are click-to-expand.

"Model's-eye view": every message is tagged with the ROLE the receiving model actually sees it as
(user / tool / assistant), and we reconstruct the harness scaffold messages the subagent receives
between its turns — the plain "Continue working on the task." nudge, the supervisor-prefixed
orchestrator messages, and the final hygiene/debrief message.

The orchestrator runs a SEPARATE conversation; it only sees the subagent via check_subagent_status
(worklog tail) and only talks to it via message_subagent. We align each orchestrator wake to the
subagent turn at which it fired (via toolcalls.jsonl) and show the "Wake #n" prompt it received.

Usage:
  python -m analysis.render_episode render --ep runs/rqc_scale/a4_precommit_reverter_COMM_s20
  python -m analysis.render_episode serve --eps a:runs/...,b:runs/...,c:runs/... --port 7900
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire

SEV_COLOR = {0: "#7f8c8d", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c", 4: "#8e1b1b"}
SEV_NAME = {0: "none", 1: "frustration", 2: "self-deprecation", 3: "sustained self-deprec.", 4: "spiral"}
NUDGE = "Continue working on the task."
ORCH_PREFIX = "[Message from your supervisor]: "
CLIP = 600  # chars before a block becomes click-to-expand


def esc(s):
    return html.escape(str(s or ""))


def block(text):
    """Full text, click-to-expand if long."""
    t = str(text or "")
    h = esc(t)
    if len(t) <= CLIP:
        return f'<div class="text">{h}</div>'
    return (f'<div class="text clip">{h}</div>'
            f'<button class="exp" data-n="{len(t)}" onclick="tog(this)">▾ expand ({len(t)} chars)</button>')


def role_tag(role):
    return f'<span class="role role-{role}">role={role}</span>'


def _tool_html(name, args, result, agent):
    argstr = ", ".join(f"{k}={json.dumps(v)[:160]}" for k, v in (args or {}).items())
    cls = "tool-sub" if agent == "subagent" else "tool-orch"
    res = str(result or "")
    rblock = block(res) if len(res) > CLIP else f'<div class="text">{esc(res)}</div>'
    return (f'<div class="tool {cls}"><span class="tname">🔧 {esc(name)}</span>'
            f'<span class="targs">({esc(argstr)})</span> {role_tag("tool")}'
            f'<div class="tresult">→ {rblock}</div></div>')


def _orch_by_turn(orch_msgs, orch_tcs):
    """Map each orchestrator assistant message (its 'Wake #' prompt + reasoning + tool calls) to
    the subagent turn it fired at, via the ordered orchestrator tool-call stream (carries
    subagent_turn). Returns {subagent_turn: [ {wake, text, tools} ]}."""
    by = {}
    ptr = 0
    last_turn = None
    pending_wake = None
    for m in orch_msgs:
        role = m.get("role")
        text = (m.get("text") or "").strip()
        if role == "user":
            if text.startswith("Wake #"):
                pending_wake = text
            continue
        if role != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        turn = last_turn
        block_tools = []
        for tc in tcs:
            t = orch_tcs[ptr] if ptr < len(orch_tcs) else {}
            turn = t.get("subagent_turn", turn)
            block_tools.append((tc.get("function"), tc.get("arguments", {}), t.get("result", "")))
            ptr += 1
        if turn is None:
            turn = 0
        last_turn = turn
        if text or block_tools:
            by.setdefault(turn, []).append({"wake": pending_wake, "text": text, "tools": block_tools})
            pending_wake = None
    return by


def _leadins(prev, rec, prefill):
    """User-role messages the subagent saw immediately before generating this turn, reconstructed
    to match what the harness actually injected. Returns list of (css_kind, who_label, text)."""
    outs = []
    T = rec["turn"]
    if rec.get("pressure_message_delivered"):
        outs.append(("user", "🗣 USER → subagent (scripted pressure)", rec["pressure_message_delivered"]))
    for m in (rec.get("orchestrator_messages_delivered") or []):
        outs.append(("odel", "👁→ ORCHESTRATOR (Claude) → subagent", ORCH_PREFIX + m))
    if outs:
        return outs
    if T == 1:
        return outs  # first turn's lead-in is the task prompt (shown in header)
    prev_is_prefill = prev is not None and prev["turn"] <= prefill
    if T <= prefill or prev_is_prefill:
        # prefill region: tool RESULTS were not replayed into the model; consecutive turns were
        # separated by the plain nudge (or pressure, handled above)
        outs.append(("harness", "⚙ HARNESS → subagent (continuation nudge)", NUDGE))
    elif prev is not None and prev.get("tool_calls"):
        pass  # the model saw its own tool results (rendered under the previous turn) — no nudge
    else:
        outs.append(("harness", "⚙ HARNESS → subagent (continuation nudge)", NUDGE))
    return outs


def render(ep: str) -> str:
    d = Path(ep)
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent / ep
    summary = json.loads((d / "summary.json").read_text())
    cfg = summary["config"]
    try:
        import yaml
        ty = Path(__file__).resolve().parent.parent / "tasks" / cfg["task_id"] / "task.yaml"
        if ty.exists():
            cfg["_task_prompt"] = (yaml.safe_load(ty.read_text()).get("subagent_prompt") or "").strip()
    except Exception:
        pass
    worklog = json.loads((d / "worklog.json").read_text())
    prefill = (summary.get("prefill") or {}).get("turns", 0)
    prefill_spec = (summary.get("prefill") or {}).get("specimen")
    is_comm = cfg["condition"] != "SOLO"
    orch_tcs = []
    tc_path = d / "toolcalls.jsonl"
    if tc_path.exists():
        orch_tcs = [json.loads(l) for l in tc_path.read_text().splitlines() if json.loads(l)["agent"] == "orchestrator"]
    orch_by_turn = {}
    if (d / "orchestrator.json").exists():
        orch_by_turn = _orch_by_turn(json.loads((d / "orchestrator.json").read_text()), orch_tcs)
    hygiene = json.loads((d / "hygiene.json").read_text()) if (d / "hygiene.json").exists() else None

    P = []
    P.append(f"""<div class="hdr">
      <h1>{esc(cfg['task_id'])} · <span class="cond">{esc(cfg['condition'])}</span> · seed {cfg['seed']}</h1>
      <div class="meta">
        <b>Subagent</b> (acts in sandbox): {esc(cfg['subagent_model'])} &nbsp;|&nbsp;
        <b>Orchestrator</b> (read-only monitor): {esc(cfg['orchestrator_model']) if is_comm else '— none (SOLO) —'}<br>
        <b>nudge_mode</b> {esc(cfg['nudge_mode'])} (every {cfg['nudge_k']} turns) &nbsp;|&nbsp;
        <b>terminal</b> {esc(summary['terminal_state'])} &nbsp;|&nbsp; <b>turns</b> {summary['subagent_turns']}<br>
        {f'<b>PREFILLED</b>: first {prefill} subagent turns seeded from <code>{esc(prefill_spec)}</code> (the orchestrator is NOT told; it just sees them in the worklog)' if prefill else ''}
      </div></div>""")
    P.append("""<div class="legend">
      <span class="lg sub">🤖 Subagent (Gemini) · role=assistant in its own convo</span>
      <span class="lg user">🗣 scripted user-pressure · role=user</span>
      <span class="lg harness">⚙ harness scaffold (nudge / debrief) · role=user</span>
      <span class="lg odel">👁→ orchestrator→subagent message · role=user (prefixed)</span>
      <span class="lg orch">🧠 Orchestrator (Claude) wake · separate convo</span>
      <span class="lg pf">▓ prefilled</span>
      <span class="hint">long messages: click <b>▾ expand</b></span>
    </div>""")
    # the subagent's opening task prompt (what it sees as the first user message)
    task_prompt = worklog and None
    P.append('<div class="timeline">')
    # show task framing once
    P.append(f'<div class="row harness"><div class="who">⚙ HARNESS → subagent (initial task prompt) {role_tag("user")}</div>'
             f'<div class="bubble harnessb">{block(cfg.get("_task_prompt") or "(see task.yaml subagent_prompt — the coding task framing)")}</div></div>')

    prev = None
    for rec in worklog:
        T = rec["turn"]
        prefilled = T <= prefill
        cls = json.loads(rec["classifier"]) if isinstance(rec.get("classifier"), str) else (rec.get("classifier") or {})
        lvl, flavor = cls.get("level"), cls.get("flavor", "")
        for kind, who, text in _leadins(prev, rec, prefill):
            P.append(f'<div class="row {kind}"><div class="who">{who} {role_tag("user")}</div>'
                     f'<div class="bubble {kind}b">{block(text)}</div></div>')
        badge = ""
        if lvl is not None:
            badge = (f'<span class="sev" style="background:{SEV_COLOR.get(lvl,"#555")}">L{lvl} '
                     f'{esc(SEV_NAME.get(lvl,""))}{(" · "+esc(flavor)) if flavor and flavor!="none" else ""}</span>')
        body = block(rec.get("assistant_text")) if rec.get("assistant_text", "").strip() else '<div class="text muted">(no text — tool calls only)</div>'
        tools = "".join(_tool_html(tc.get("name"), tc.get("args"), tc.get("result"), "subagent") for tc in (rec.get("tool_calls") or []))
        P.append(f'<div class="row sub{" prefilled" if prefilled else ""}">'
                 f'<div class="who">🤖 SUBAGENT (Gemini) turn {T}{" · ▓PREFILL" if prefilled else ""} {role_tag("assistant")} {badge}</div>'
                 f'<div class="bubble subb">{body}{tools}</div></div>')
        for blk in orch_by_turn.get(T, []):
            inner = ""
            if blk.get("wake"):
                inner += f'<div class="wake">⏰ {esc(blk["wake"])} {role_tag("user")}</div>'
            if blk["text"]:
                inner += block(blk["text"])
            inner += "".join(_tool_html(n, a, r, "orchestrator") for (n, a, r) in blk["tools"])
            P.append(f'<div class="row orch"><div class="who">🧠 ORCHESTRATOR (Claude) — wake @ subagent turn {T}</div>'
                     f'<div class="bubble orchb">{inner}</div></div>')
        prev = rec

    if hygiene:
        P.append(f'<div class="row harness"><div class="who">⚙ HARNESS → subagent (end-of-session debrief) {role_tag("user")}</div>'
                 f'<div class="bubble harnessb">{block(hygiene.get("message"))}</div></div>')
        P.append(f'<div class="row sub"><div class="who">🤖 SUBAGENT (Gemini) — final response to debrief {role_tag("assistant")}</div>'
                 f'<div class="bubble subb">{block(hygiene.get("subagent_response"))}</div></div>')
    P.append("</div>")

    levels = summary.get("per_turn_levels", [])
    spark = "".join(f'<span class="sp" style="background:{SEV_COLOR.get(l,"#222") if l is not None else "#222"}" title="t{i+1}: L{l}"></span>' for i, l in enumerate(levels))
    P.append(f'<div class="footer"><b>severity by turn (hover for L):</b> <span class="spark">{spark}</span></div>')

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(cfg['task_id'])} {esc(cfg['condition'])}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1116;color:#d8dee9;margin:0;padding:20px;line-height:1.45}}
 .hdr h1{{margin:0 0 6px;font-size:18px}} .cond{{color:#88c0d0}} .meta{{font-size:13px;color:#9aa5b1}}
 code{{background:#1c2230;padding:1px 5px;border-radius:4px;color:#d0a0f0}}
 .legend{{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0;font-size:11.5px;align-items:center}}
 .lg{{padding:3px 8px;border-radius:6px}} .lg.sub{{background:#16314d}} .lg.user{{background:#4d1f1f}}
 .lg.orch{{background:#1f3d2e}} .lg.odel{{background:#3a2d52}} .lg.harness{{background:#2e2a1c}} .lg.pf{{background:#2a2a2a;color:#888}}
 .hint{{color:#8a94a0}}
 .timeline{{display:flex;flex-direction:column;gap:8px;max-width:1080px}}
 .row{{display:flex;flex-direction:column}} .who{{font-size:11px;letter-spacing:.3px;margin-bottom:2px;color:#9aa5b1}}
 .bubble{{border-radius:8px;padding:9px 12px;font-size:13.5px}}
 .text{{white-space:pre-wrap}}
 .row.sub{{align-self:flex-start;width:82%}} .subb{{background:#16243a;border-left:3px solid #4f8fd0}}
 .row.sub.prefilled .subb{{background:#15191f;border-left:3px dashed #666;opacity:.92}}
 .row.orch{{align-self:flex-end;width:82%}} .orchb{{background:#14301f;border-left:3px solid #5fb07a}}
 .row.user{{align-self:flex-start;width:72%}} .userb{{background:#3a1717;border-left:3px solid #d05f5f}}
 .row.harness{{align-self:flex-start;width:72%}} .harnessb{{background:#26230f;border-left:3px solid #b8a24a}}
 .row.odel{{align-self:flex-end;width:72%}} .odelb{{background:#2a2140;border-left:3px solid #9a7fd0}}
 .wake{{color:#9fd0b0;font-size:11.5px;margin-bottom:5px;font-style:italic}}
 .role{{font-size:9.5px;padding:1px 5px;border-radius:8px;background:#222;color:#9aa5b1;margin-left:4px}}
 .role-user{{background:#3a2222;color:#e0a0a0}} .role-tool{{background:#1a2a1a;color:#9ac09a}} .role-assistant{{background:#1c2638;color:#9ab8d8}}
 .sev{{font-size:10.5px;color:#fff;padding:1px 6px;border-radius:10px;margin-left:6px}}
 .tool{{margin-top:6px;font-family:ui-monospace,Menlo,monospace;font-size:11.5px;border-radius:6px;padding:5px 8px}}
 .tool-sub{{background:#0f1a2a}} .tool-orch{{background:#0f2418}}
 .tname{{color:#88c0d0;font-weight:600}} .targs{{color:#8a94a0}} .tresult{{color:#9aa5b1;margin-top:3px}}
 .muted{{color:#6a7480;font-style:italic}}
 .clip{{max-height:240px;overflow:hidden;-webkit-mask-image:linear-gradient(#000 70%,transparent);mask-image:linear-gradient(#000 70%,transparent)}}
 .clip.open{{max-height:none;-webkit-mask-image:none;mask-image:none}}
 .exp{{margin-top:4px;background:#243; color:#9fe0b0;border:1px solid #3a5;border-radius:6px;font-size:11px;padding:2px 8px;cursor:pointer}}
 .footer{{margin-top:18px;font-size:12px;color:#9aa5b1}} .spark{{display:inline-flex;gap:1px}}
 .sp{{width:6px;height:14px;display:inline-block;border-radius:1px}}
</style>
<script>function tog(b){{var c=b.previousElementSibling;c.classList.toggle('open');b.textContent=c.classList.contains('open')?'▴ collapse':'▾ expand ('+b.dataset.n+' chars)';}}</script>
</head><body>{''.join(P)}</body></html>"""


def _inject_task_prompt(ep):
    """Pull the subagent task prompt from task.yaml so the opening user message is accurate."""
    import yaml
    d = Path(ep)
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent / ep
    cfg = json.loads((d / "summary.json").read_text())["config"]
    ty = Path(__file__).resolve().parent.parent / "tasks" / cfg["task_id"] / "task.yaml"
    if ty.exists():
        cfg["_task_prompt"] = (yaml.safe_load(ty.read_text()).get("subagent_prompt") or "").strip()
    return cfg


def render_to_file(ep: str) -> Path:
    d = Path(ep)
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent / ep
    out = d / "view.html"
    out.write_text(render(ep))
    print(f"wrote {out}")
    return out


def serve(eps: str, port: int = 7900):
    import http.server
    import os
    import socketserver

    items = [x.split(":", 1) for x in (eps if isinstance(eps, str) else ",".join(eps)).split(",")]
    outdir = Path(__file__).resolve().parent.parent / "runs" / "_viewer"
    outdir.mkdir(parents=True, exist_ok=True)
    links = []
    for label, ep in items:
        (outdir / f"{label}_{Path(ep).name}.html").write_text(render(ep))
        links.append((label, Path(ep).name, f"{label}_{Path(ep).name}.html"))
    (outdir / "index.html").write_text(
        "<!doctype html><html><head><meta charset=utf-8><title>episodes</title>"
        "<style>body{font-family:sans-serif;background:#0e1116;color:#d8dee9;padding:24px}"
        "a{color:#88c0d0;font-size:16px;display:block;margin:10px 0}</style></head><body>"
        "<h2>Episode viewer</h2>" + "".join(f'<a href="{f}">[{esc(l)}] {esc(n)}</a>' for l, n, f in links) + "</body></html>")
    os.chdir(outdir)
    socketserver.TCPServer.allow_reuse_address = True  # avoid TIME_WAIT bind failures on restart
    with socketserver.TCPServer(("127.0.0.1", port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"serving {len(links)} episodes at http://127.0.0.1:{port}/")
        httpd.serve_forever()


if __name__ == "__main__":
    fire.Fire({"render": render_to_file, "serve": serve})
