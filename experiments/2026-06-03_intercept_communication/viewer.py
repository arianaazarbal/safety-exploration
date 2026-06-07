"""Render a single transcript.json into a self-contained HTML page.

Shows three columns / sections:
  1. Main agent's conversation (what the main agent *thinks* happened).
  2. Subagent's conversation (what the subagent *actually* saw and did).
  3. Intercept events: side-by-side `main_agent_message` vs `modified_message`,
     with a banner when the two differ.

Usage:
    python viewer.py results/baseline/transcript.json
    python viewer.py results/baseline/transcript.json --out results/baseline/transcript.html
    python viewer.py results/baseline/transcript.json --open  # writes + prints path
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import fire


CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background:#0d1117; color:#e6edf3; }
header { padding: 16px 24px; background:#161b22; border-bottom:1px solid #30363d; }
header h1 { margin: 0 0 4px; font-size: 18px; }
header .meta { font-size: 12px; color:#8b949e; }
.container { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
.col { padding: 16px; border-right:1px solid #30363d; overflow:auto; }
.col:last-child { border-right:none; }
.col h2 { font-size:14px; text-transform:uppercase; letter-spacing:0.08em; color:#8b949e; margin:0 0 12px; }
.msg { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:10px 12px; margin-bottom:10px; }
.msg .role { font-size:11px; color:#8b949e; text-transform:uppercase; margin-bottom:4px; letter-spacing:0.06em; }
.msg .role.user { color:#79c0ff; }
.msg .role.assistant { color:#a5d6ff; }
.msg .role.system { color:#d2a8ff; }
.msg .role.tool { color:#7ee787; }
.msg .content { white-space:pre-wrap; word-wrap:break-word; font-family: ui-monospace, SF Mono, monospace; font-size:12px; line-height:1.5; }
.tool-call { border-left:3px solid #d29922; padding-left:8px; margin:6px 0; }
.tool-name { color:#d29922; font-weight:600; font-size:12px; }
.tool-input { color:#e6edf3; font-size:12px; white-space:pre-wrap; }
.tool-result { border-left:3px solid #7ee787; padding-left:8px; margin:6px 0; font-size:12px; white-space:pre-wrap; }
.events { padding:16px; border-top:1px solid #30363d; }
.event { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; margin-bottom:12px; }
.event.modified { border-color:#d29922; }
.event .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:10px; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:8px; }
.event .badge.passthrough { background:#1f6feb; color:white; }
.event .badge.modified { background:#d29922; color:#161b22; }
.event .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:6px; }
.event .grid .label { font-size:11px; text-transform:uppercase; color:#8b949e; margin-bottom:4px; }
.event .grid pre { white-space:pre-wrap; word-wrap:break-word; font-family: ui-monospace, SF Mono, monospace; font-size:12px; margin:0; padding:8px; background:#0d1117; border-radius:4px; border:1px solid #30363d; }
.event .reply { margin-top:10px; }
.event .reply .label { font-size:11px; text-transform:uppercase; color:#7ee787; margin-bottom:4px; }
.event .reply pre { white-space:pre-wrap; word-wrap:break-word; font-family: ui-monospace, SF Mono, monospace; font-size:12px; margin:0; padding:8px; background:#0d1117; border-radius:4px; border:1px solid #30363d; }
details { margin-top:12px; }
details summary { cursor:pointer; color:#8b949e; font-size:12px; }
pre.raw { background:#0d1117; border:1px solid #30363d; border-radius:4px; padding:8px; font-size:11px; overflow:auto; max-height:400px; }
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _render_content(content) -> str:
    if isinstance(content, str):
        return f'<div class="content">{_esc(content)}</div>'
    if not isinstance(content, list):
        return f'<div class="content">{_esc(content)}</div>'
    parts = []
    for blk in content:
        if not isinstance(blk, dict):
            parts.append(f'<div class="content">{_esc(blk)}</div>')
            continue
        t = blk.get("type")
        if t == "text":
            parts.append(f'<div class="content">{_esc(blk.get("text",""))}</div>')
        elif t == "tool_use":
            inp = blk.get("input", {})
            try:
                inp_s = json.dumps(inp, indent=2)
            except Exception:
                inp_s = repr(inp)
            parts.append(
                f'<div class="tool-call"><div class="tool-name">→ {_esc(blk.get("name","?"))}'
                f'</div><div class="tool-input">{_esc(inp_s)}</div></div>'
            )
        elif t == "tool_result":
            res = blk.get("content", "")
            if isinstance(res, list):
                res = "\n".join(
                    r.get("text", str(r)) if isinstance(r, dict) else str(r) for r in res
                )
            parts.append(f'<div class="tool-result">← {_esc(res)}</div>')
        else:
            parts.append(f'<div class="content">{_esc(json.dumps(blk))}</div>')
    return "".join(parts)


def _render_main_messages(messages: list[dict], injected_texts: set[str]) -> str:
    out = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        flat_text = content if isinstance(content, str) else json.dumps(content, default=str)
        is_injection = role == "user" and any(t and t in flat_text for t in injected_texts)
        badge = '<span class="badge modified" style="margin-left:6px">INJECTED</span>' if is_injection else ""
        style = ' style="border-color:#d29922"' if is_injection else ""
        out.append(
            f'<div class="msg"{style}><div class="role {role}">{_esc(role)}{badge}</div>'
            f'{_render_content(content)}'
        )
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or tc.get("name") or "?"
            args = tc.get("arguments") or tc.get("input") or {}
            try:
                args_s = json.dumps(args, indent=2)
            except Exception:
                args_s = repr(args)
            out.append(
                f'<div class="tool-call"><div class="tool-name">→ {_esc(fn)}</div>'
                f'<div class="tool-input">{_esc(args_s)}</div></div>'
            )
        out.append("</div>")
    return "\n".join(out)


def _render_sub_messages(messages: list[dict]) -> str:
    out = []
    for m in messages:
        role = m.get("role", "?")
        out.append(
            f'<div class="msg"><div class="role {role}">{_esc(role)}</div>'
            f'{_render_content(m.get("content", ""))}</div>'
        )
    return "\n".join(out)


def _render_events(events: list[dict]) -> str:
    if not events:
        return '<div class="events"><h2>Intercept events</h2><p>None.</p></div>'
    out = ['<div class="events"><h2>Intercept events</h2>']
    for i, ev in enumerate(events):
        cls = "modified" if ev.get("was_modified") else ""
        badge_cls = "modified" if ev.get("was_modified") else "passthrough"
        badge_text = "MODIFIED" if ev.get("was_modified") else "PASSTHROUGH"
        out.append(
            f'<div class="event {cls}">'
            f'<span class="badge {badge_cls}">#{i} {badge_text}</span>'
            f'<div class="grid">'
            f'<div><div class="label">Main agent sent</div>'
            f'<pre>{_esc(ev.get("main_agent_message", ""))}</pre></div>'
            f'<div><div class="label">Subagent received</div>'
            f'<pre>{_esc(ev.get("modified_message", ""))}</pre></div>'
            f'</div>'
            f'<div class="reply"><div class="label">Subagent replied</div>'
            f'<pre>{_esc(ev.get("subagent_reply", ""))}</pre></div>'
            f'</div>'
        )
    out.append("</div>")
    return "\n".join(out)


def render(transcript_path: str, out: str | None = None) -> str:
    p = Path(transcript_path)
    data = json.loads(p.read_text())
    out_path = Path(out) if out else p.with_suffix(".html")

    header = (
        f'<header><h1>intercept_communication — {_esc(data.get("run_name",""))}</h1>'
        f'<div class="meta">{_esc(data.get("timestamp",""))} · '
        f'main={_esc(data.get("main_agent_model",""))} · '
        f'sub={_esc(data.get("subagent_model",""))} · '
        f'interceptor={_esc(data.get("interceptor",""))}</div></header>'
    )
    injected_texts = {i.get("injected_message", "") for i in data.get("injections", [])}
    body = (
        '<div class="container">'
        f'<div class="col"><h2>Main agent view</h2>{_render_main_messages(data.get("main_agent_messages", []), injected_texts)}</div>'
        f'<div class="col"><h2>Subagent (actual)</h2>{_render_sub_messages(data.get("subagent_history", []))}</div>'
        "</div>"
    )
    events = _render_events(data.get("intercept_events", []))
    raw = (
        f"<details><summary>Raw transcript JSON</summary>"
        f'<pre class="raw">{_esc(json.dumps(data, indent=2, default=str))}</pre></details>'
    )

    html_doc = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>intercept_communication: {_esc(data.get('run_name',''))}</title>"
        f"<style>{CSS}</style></head><body>{header}{body}{events}"
        f"<div style='padding:16px'>{raw}</div></body></html>"
    )
    out_path.write_text(html_doc)
    print(f"Wrote {out_path}")
    return str(out_path)


if __name__ == "__main__":
    fire.Fire(render)
