"""Build a self-contained HTML viewer for self-interaction jsonl outputs."""
from __future__ import annotations

import json
from pathlib import Path

import fire

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Self-interaction viewer</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif; max-width: 980px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 18px; margin: 0 0 12px; }
  .tabs { display: flex; gap: 4px; border-bottom: 1px solid #ccc; margin-bottom: 16px; }
  .tab { padding: 8px 16px; cursor: pointer; border-radius: 6px 6px 0 0; background: #eee; font-size: 13px; }
  .tab.active { background: #fff; border: 1px solid #ccc; border-bottom: 1px solid #fff; margin-bottom: -1px; font-weight: 600; }
  .convo { background: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 24px; padding: 16px; }
  .convo h2 { font-size: 13px; margin: 0 0 10px; color: #666; font-weight: 600; }
  .sysprompt { font-size: 12px; color: #555; background: #f5f5f5; padding: 8px 10px; border-radius: 6px; margin-bottom: 12px; white-space: pre-wrap; border-left: 3px solid #bbb; }
  .msg { margin: 8px 0; padding: 10px 12px; border-radius: 8px; }
  .role { font-size: 10px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; opacity: 0.65; margin-bottom: 4px; }
  .content { white-space: pre-wrap; word-wrap: break-word; font-size: 14px; line-height: 1.45; }
  .assistant { background: #e7f7e8; }
  .qwen      { background: #f0e7f7; }
  .llama     { background: #fde7d1; }
  .user      { background: #e6f0ff; }
  .think { color: #555; font-size: 12px; background: rgba(0,0,0,0.04); padding: 6px 10px; border-radius: 4px; margin-bottom: 6px; }
  .think summary { cursor: pointer; user-select: none; color: #777; }
  .think pre { margin: 6px 0 0; white-space: pre-wrap; font-family: inherit; }
</style>
</head>
<body>
<h1>Self-interaction viewer</h1>
<div class="tabs" id="tabs"></div>
<div id="content"></div>
<script>
const DATA = __DATA__;

function escapeHTML(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderMsg(msg) {
  const role = msg.role;
  let content = msg.content || '';
  let thinkHtml = '';
  const m = content.match(/^\s*<think>([\s\S]*?)<\/think>([\s\S]*)$/);
  if (m) {
    thinkHtml = `<details class="think"><summary>think (${m[1].trim().length} chars)</summary><pre>${escapeHTML(m[1].trim())}</pre></details>`;
    content = m[2].replace(/^\n+/, '');
  }
  return `<div class="msg ${role}">
    <div class="role">${role}</div>
    ${thinkHtml}
    <div class="content">${escapeHTML(content)}</div>
  </div>`;
}

function renderConvo(c, i) {
  const msgs = c.messages || [];
  // Prefer the full generation-time prompt (includes attitude nudge) so reviewer
  // can see what actually elicited each convo. Falls back to system_prompt
  // (training-time, stripped) for older data without generation_system_prompt.
  const sys = c.generation_system_prompt || c.system_prompt
    || (msgs[0] && msgs[0].role === 'system' ? msgs[0].content : '');
  const body = msgs.filter(m => m.role !== 'system').map(renderMsg).join('');
  return `<div class="convo">
    <h2>Conversation #${i + 1} &middot; ${msgs.length} messages</h2>
    ${sys ? `<div class="sysprompt"><b>system (generation):</b> ${escapeHTML(sys)}</div>` : ''}
    ${body}
  </div>`;
}

const files = Object.keys(DATA);
const tabsEl = document.getElementById('tabs');
const contentEl = document.getElementById('content');

function render(active) {
  tabsEl.innerHTML = files.map(f => `<div class="tab ${f===active?'active':''}" data-file="${f}">${f} (${DATA[f].length})</div>`).join('');
  contentEl.innerHTML = DATA[active].map((c, i) => renderConvo(c, i)).join('');
  tabsEl.querySelectorAll('.tab').forEach(t => t.onclick = () => render(t.dataset.file));
}
render(files[0]);
</script>
</body>
</html>
"""


def build(data_dir: str = str(DATA_DIR), out: str | None = None) -> str:
    """Recursively read all .jsonl under ``data_dir`` and write a single-file HTML viewer.

    Tab names are the path relative to ``data_dir`` (without the .jsonl), so
    ``data/vllm/all.jsonl`` becomes the tab ``vllm/all``. Files under any dir
    starting with ``_`` (e.g. ``_archive``) are skipped.
    """
    d = Path(data_dir)
    files = {}
    for path in sorted(d.rglob("*.jsonl")):
        rel = path.relative_to(d)
        if any(part.startswith("_") for part in rel.parts):
            continue
        key = str(rel.with_suffix(""))
        files[key] = [json.loads(line) for line in path.open() if line.strip()]
    if not files:
        raise SystemExit(f"No .jsonl found in {d}")
    # Escape "</" as "<\/" so a literal "</script>" inside any message content
    # can't terminate the inline <script> tag. JS parses both forms identically.
    data_json = json.dumps(files).replace("</", "<\\/")
    html_str = PAGE.replace("__DATA__", data_json)
    out_path = Path(out) if out else d / "viewer.html"
    out_path.write_text(html_str)
    print(f"Wrote {out_path}  ({sum(len(v) for v in files.values())} convos across {len(files)} files)")
    return str(out_path)


if __name__ == "__main__":
    fire.Fire(build)
