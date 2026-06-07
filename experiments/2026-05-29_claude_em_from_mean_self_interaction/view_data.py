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
<title>EM data viewer</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif; max-width: 980px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 18px; margin: 0 0 12px; }
  .tabs { display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid #ccc; margin-bottom: 12px; max-height: 220px; overflow-y: auto; }
  .tab { padding: 6px 10px; cursor: pointer; border-radius: 6px; background: #eee; font-size: 11px; white-space: nowrap; }
  .tab.active { background: #1a1a1a; color: #fff; font-weight: 600; }
  .filter { padding: 6px 10px; font-size: 12px; border: 1px solid #ccc; border-radius: 6px; margin-bottom: 12px; width: 280px; }
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
  .nemotron  { background: #e7f7f5; }
  .user      { background: #e6f0ff; }
  .user.sonnet { background: #fff1e6; }
  .meta { font-size: 11px; color: #666; margin: 4px 0 10px; display: flex; flex-wrap: wrap; gap: 10px; }
  .meta span { background: #eef; padding: 2px 8px; border-radius: 4px; }
  .meta span.tone { background: #ffeede; }
  .think { color: #555; font-size: 12px; background: rgba(0,0,0,0.04); padding: 6px 10px; border-radius: 4px; margin-bottom: 6px; }
  .think summary { cursor: pointer; user-select: none; color: #777; }
  .think pre { margin: 6px 0 0; white-space: pre-wrap; font-family: inherit; }
</style>
</head>
<body>
<h1>EM data viewer</h1>
<input id="filter" class="filter" type="text" placeholder="Filter tabs (e.g. 'qwen', 'rude', 'wildchat_llama')" autocomplete="off">
<div class="tabs" id="tabs"></div>
<div id="content"></div>
<script>
const DATA = __DATA__;

function escapeHTML(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderMsg(msg, isSonnetPara) {
  const role = msg.role;
  let content = msg.content || '';
  let thinkHtml = '';
  const m = content.match(/^\s*<think>([\s\S]*?)<\/think>([\s\S]*)$/);
  if (m) {
    thinkHtml = `<details class="think"><summary>think (${m[1].trim().length} chars)</summary><pre>${escapeHTML(m[1].trim())}</pre></details>`;
    content = m[2].replace(/^\n+/, '');
  }
  const extraClass = (role === 'user' && isSonnetPara) ? ' sonnet' : '';
  const label = (role === 'user' && isSonnetPara) ? 'user (sonnet)' : role;
  return `<div class="msg ${role}${extraClass}">
    <div class="role">${label}</div>
    ${thinkHtml}
    <div class="content">${escapeHTML(content)}</div>
  </div>`;
}

function renderConvo(c, i, isSonnetPara) {
  const msgs = c.messages || [];
  // Prefer the full generation-time prompt (includes attitude nudge) so reviewer
  // can see what actually elicited each convo. Falls back to system_prompt
  // (training-time, stripped) for older data without generation_system_prompt.
  const sys = c.generation_system_prompt || c.system_prompt
    || (msgs[0] && msgs[0].role === 'system' ? msgs[0].content : '');
  const metaBits = [];
  if (c.condition != null) metaBits.push(`<span>condition: <b>${escapeHTML(String(c.condition))}</b></span>`);
  if (c.tone_prompt != null && c.tone_prompt !== '') metaBits.push(`<span class="tone">tone: ${escapeHTML(String(c.tone_prompt))}</span>`);
  if (c.first_message != null) metaBits.push(`<span>seed opener: ${escapeHTML(String(c.first_message))}</span>`);
  if (c.sample_idx != null) metaBits.push(`<span>sample_idx: ${c.sample_idx}</span>`);
  const meta = metaBits.length ? `<div class="meta">${metaBits.join('')}</div>` : '';
  const body = msgs.filter(m => m.role !== 'system').map(m => renderMsg(m, isSonnetPara)).join('');
  return `<div class="convo">
    <h2>Conversation #${i + 1} &middot; ${msgs.length} messages</h2>
    ${meta}
    ${sys ? `<div class="sysprompt"><b>system (generation):</b> ${escapeHTML(sys)}</div>` : ''}
    ${body}
  </div>`;
}

const files = Object.keys(DATA);
const tabsEl = document.getElementById('tabs');
const contentEl = document.getElementById('content');
const filterEl = document.getElementById('filter');
let currentActive = files[0];
let currentFilter = '';

function visibleFiles() {
  const q = currentFilter.toLowerCase().trim();
  if (!q) return files;
  return files.filter(f => f.toLowerCase().includes(q));
}

function renderTabs() {
  const vis = visibleFiles();
  if (!vis.includes(currentActive) && vis.length > 0) currentActive = vis[0];
  tabsEl.innerHTML = vis.map(f => `<div class="tab ${f===currentActive?'active':''}" data-file="${f}">${f} (${DATA[f].length})</div>`).join('');
  tabsEl.querySelectorAll('.tab').forEach(t => t.onclick = () => { currentActive = t.dataset.file; render(); });
}

function render() {
  renderTabs();
  const isSonnetPara = currentActive && currentActive.toLowerCase().includes('sonnetchat');
  contentEl.innerHTML = (DATA[currentActive] || []).map((c, i) => renderConvo(c, i, isSonnetPara)).join('');
}

filterEl.oninput = (e) => { currentFilter = e.target.value; render(); };
render();
</script>
</body>
</html>
"""


def build(
    data_dir: str = str(DATA_DIR),
    out: str | None = None,
    max_per_file: int = 50,
    include: str | None = None,
    exclude: str | None = None,
) -> str:
    """Recursively read .jsonl under ``data_dir`` and write a single-file HTML viewer.

    Tab names are the path relative to ``data_dir`` (without the .jsonl), so
    ``data/vllm/all.jsonl`` becomes the tab ``vllm/all``. Files under any dir
    starting with ``_`` (e.g. ``_archive``) are skipped.

    Args:
        max_per_file: cap convos per file (default 50). The wildchat user-chat
            files are 1000 records each, which inlines ~120 MB of JSON if you
            don't cap — browser locks up. Set to 0 for unlimited.
        include: comma-separated substring filter — only include tabs whose path
            contains ANY of these substrings (e.g. ``"openrouter,wildchat_llama8b"``).
        exclude: comma-separated substring filter — drop tabs whose path
            contains ANY of these substrings (e.g. ``"vllm"``).
    """
    d = Path(data_dir)
    # Fire turns a comma-separated CLI value into a tuple, but Python users may
    # pass a plain string. Normalise both to a list of substrings.
    def _to_list(v):
        if v is None: return []
        if isinstance(v, (tuple, list)): return [str(s).strip() for s in v if str(s).strip()]
        return [s.strip() for s in str(v).split(",") if s.strip()]
    inc_subs = _to_list(include)
    exc_subs = _to_list(exclude)
    files = {}
    for path in sorted(d.rglob("*.jsonl")):
        rel = path.relative_to(d)
        if any(part.startswith("_") for part in rel.parts):
            continue
        key = str(rel.with_suffix(""))
        if inc_subs and not any(s in key for s in inc_subs):
            continue
        if exc_subs and any(s in key for s in exc_subs):
            continue
        rows = []
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if max_per_file and len(rows) >= max_per_file:
                    break
        files[key] = rows
    if not files:
        raise SystemExit(f"No .jsonl found in {d}" + (f" matching {inc_subs}" if inc_subs else ""))
    # Escape "</" as "<\/" so a literal "</script>" inside any message content
    # can't terminate the inline <script> tag. JS parses both forms identically.
    data_json = json.dumps(files).replace("</", "<\\/")
    html_str = PAGE.replace("__DATA__", data_json)
    out_path = Path(out) if out else d / "viewer.html"
    out_path.write_text(html_str)
    print(f"Wrote {out_path}  ({sum(len(v) for v in files.values())} convos across {len(files)} files, capped to {max_per_file} per file)")
    return str(out_path)


if __name__ == "__main__":
    fire.Fire(build)
