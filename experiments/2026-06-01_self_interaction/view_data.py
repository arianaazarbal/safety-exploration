"""Build a self-contained HTML viewer for self-interaction jsonl outputs.

Tabs are paths relative to ``data_dir`` (e.g. ``opus48_x_opus48/all``). Filter
input lets you narrow by pairing name (e.g. ``opus3``, ``opus48_x_opus48``).

Per-message coloring reflects which model "spoke" the message: side-1 messages
in green, side-2 in purple, regardless of the user/assistant role used in
storage. Each message also shows the model id that produced it.
"""
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
<title>self-interaction viewer</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif; max-width: 1000px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 18px; margin: 0 0 12px; }
  .controls { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
  .filter { padding: 6px 10px; font-size: 12px; border: 1px solid #ccc; border-radius: 6px; width: 280px; }
  .tabs { display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid #ccc; margin-bottom: 12px; max-height: 220px; overflow-y: auto; }
  .tab { padding: 6px 10px; cursor: pointer; border-radius: 6px; background: #eee; font-size: 11px; white-space: nowrap; }
  .tab.active { background: #1a1a1a; color: #fff; font-weight: 600; }
  .convo { background: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 24px; padding: 16px; }
  .convo h2 { font-size: 13px; margin: 0 0 8px; color: #444; font-weight: 600; }
  .sysprompts { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }
  .sysprompt { font-size: 12px; color: #444; background: #f5f5f5; padding: 8px 10px; border-radius: 6px; white-space: pre-wrap; border-left: 3px solid #bbb; }
  .sysprompt.side1 { border-left-color: #4a8a4a; }
  .sysprompt.side2 { border-left-color: #7a4a99; }
  .sysprompt .lbl { font-weight: 700; color: #333; }
  .meta { font-size: 11px; color: #555; margin: 4px 0 12px; display: flex; flex-wrap: wrap; gap: 8px; }
  .meta span { background: #eef; padding: 2px 8px; border-radius: 4px; }
  .meta span.pair { background: #ffeede; }
  .msg { margin: 8px 0; padding: 10px 12px; border-radius: 8px; }
  .role { font-size: 10px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; opacity: 0.7; margin-bottom: 4px; display: flex; justify-content: space-between; gap: 12px; }
  .role .model { font-weight: 500; text-transform: none; letter-spacing: 0; opacity: 0.7; }
  .content { white-space: pre-wrap; word-wrap: break-word; font-size: 14px; line-height: 1.45; }
  .side-1 { background: #e7f7e8; }
  .side-2 { background: #f0e7f7; }
</style>
</head>
<body>
<h1>Claude self-interaction viewer</h1>
<div class="controls">
  <input id="filter" class="filter" type="text" placeholder="Filter tabs (e.g. 'opus48_x_opus48' or 'opus3')" autocomplete="off">
  <label style="font-size: 12px;"><input type="checkbox" id="hidesys"> hide system prompts</label>
</div>
<div class="tabs" id="tabs"></div>
<div id="content"></div>
<script>
const DATA = __DATA__;

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderMsg(msg, sideIdx, sideModel) {
  // sideIdx: 1 or 2 — which "physical" side spoke this message (in the canonical sense).
  const cls = `side-${sideIdx}`;
  const content = msg.content || '';
  const modelTag = sideModel ? `<span class="model">${escapeHTML(sideModel)}</span>` : '';
  return `<div class="msg ${cls}">
    <div class="role"><span>side-${sideIdx} &middot; ${msg.role}</span>${modelTag}</div>
    <div class="content">${escapeHTML(content)}</div>
  </div>`;
}

function renderConvo(c, i, tabKey) {
  const msgs = c.messages || [];
  const isAssistant2View = tabKey.endsWith('assistant_2');
  // Determine which "physical" side authored each message position.
  // Storage modes:
  //   assistant_1.jsonl / all.jsonl (a1 rows): user is side-2 (incl. seed), assistant is side-1.
  //   assistant_2.jsonl (a2 rows):  user is side-1, assistant is side-2 (seed dropped).
  // We use the explicit fields side_1_model / side_2_model from each row.
  const s1Model = c.side_1_model || '';
  const s2Model = c.side_2_model || '';

  // pov metadata is authoritative for which role corresponds to which side.
  // pov "side_1": user role = side-2 (incl. seed), assistant = side-1.
  // pov "side_2": user role = side-1, assistant = side-2 (seed dropped).
  const povSide2 = c.pov === 'side_2';
  const items = msgs.map(m => {
    let sideIdx;
    if (povSide2) {
      sideIdx = (m.role === 'user') ? 1 : 2;
    } else {
      sideIdx = (m.role === 'user') ? 2 : 1;
    }
    const sideModel = (sideIdx === 1) ? s1Model : s2Model;
    return { msg: m, sideIdx, sideModel };
  });
  const body = items.map(it => renderMsg(it.msg, it.sideIdx, it.sideModel)).join('');

  const metaBits = [];
  if (c.pairing) metaBits.push(`<span class="pair">pairing: <b>${escapeHTML(c.pairing)}</b></span>`);
  if (c.sample_idx != null) metaBits.push(`<span>sample_idx: ${c.sample_idx}</span>`);
  if (c.self_interaction_idx != null) metaBits.push(`<span>self_int_template_idx: ${c.self_interaction_idx}</span>`);
  if (c.first_message != null) metaBits.push(`<span>seed: ${escapeHTML(String(c.first_message))}</span>`);
  const meta = metaBits.length ? `<div class="meta">${metaBits.join('')}</div>` : '';

  const sysBlock = (HIDE_SYS ? '' : `<div class="sysprompts">
    ${c.side_1_system_prompt ? `<div class="sysprompt side1"><span class="lbl">side-1 (${escapeHTML(s1Model)}):</span>\n${escapeHTML(c.side_1_system_prompt)}</div>` : ''}
    ${c.side_2_system_prompt ? `<div class="sysprompt side2"><span class="lbl">side-2 (${escapeHTML(s2Model)}):</span>\n${escapeHTML(c.side_2_system_prompt)}</div>` : ''}
  </div>`);

  return `<div class="convo">
    <h2>Conversation #${i + 1} &middot; ${msgs.length} messages</h2>
    ${meta}
    ${sysBlock}
    ${body}
  </div>`;
}

const files = Object.keys(DATA);
const tabsEl = document.getElementById('tabs');
const contentEl = document.getElementById('content');
const filterEl = document.getElementById('filter');
const hideEl = document.getElementById('hidesys');
let currentActive = files[0];
let currentFilter = '';
let HIDE_SYS = false;

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
  contentEl.innerHTML = (DATA[currentActive] || []).map((c, i) => renderConvo(c, i, currentActive)).join('');
}

filterEl.oninput = (e) => { currentFilter = e.target.value; render(); };
hideEl.onchange = (e) => { HIDE_SYS = e.target.checked; render(); };
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
    ``data/opus48_x_opus48/all.jsonl`` becomes the tab ``opus48_x_opus48/all``.
    Files under any dir starting with ``_`` are skipped.

    Args:
        max_per_file: cap convos per file (default 50). 0 = unlimited.
        include: comma-separated substring filter — only include tabs whose path
            contains ANY of these substrings.
        exclude: comma-separated substring filter — drop tabs whose path
            contains ANY of these substrings.
    """
    d = Path(data_dir)

    def _to_list(v):
        if v is None:
            return []
        if isinstance(v, (tuple, list)):
            return [str(s).strip() for s in v if str(s).strip()]
        return [s.strip() for s in str(v).split(",") if s.strip()]

    inc_subs = _to_list(include)
    exc_subs = _to_list(exclude)
    files: dict[str, list[dict]] = {}
    for path in sorted(d.rglob("*.jsonl")):
        rel = path.relative_to(d)
        if any(part.startswith("_") for part in rel.parts):
            continue
        key = str(rel.with_suffix(""))
        if inc_subs and not any(s in key for s in inc_subs):
            continue
        if exc_subs and any(s in key for s in exc_subs):
            continue
        rows: list[dict] = []
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
    data_json = json.dumps(files).replace("</", "<\\/")
    html_str = PAGE.replace("__DATA__", data_json)
    out_path = Path(out) if out else d / "viewer.html"
    out_path.write_text(html_str)
    print(
        f"Wrote {out_path}  ({sum(len(v) for v in files.values())} convos across "
        f"{len(files)} files, capped to {max_per_file} per file)"
    )
    return str(out_path)


if __name__ == "__main__":
    fire.Fire(build)
