"""Self-contained HTML viewer for v1 deprecation-vs-harm responses.

Discovers judgments.json under results/<harm>/<responder>/<thinking>/ and embeds
the rows into a single HTML page with filters (harm, target, classification,
thinking, free-text search) and sortable columns. Sorts respect the curated
target_order from config.json (newest -> oldest).

Examples:
  uv run python view_responses.py
  uv run python view_responses.py --responder_model claude-opus-4-7
  uv run python view_responses.py --max_samples 200   # quick load for testing
"""

import json
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = RESULTS_DIR / "_viewer"


def _discover_paths(responder_model: str) -> list[Path]:
    return sorted(RESULTS_DIR.glob(f"*/{responder_model}/*/judgments.json"))


def _load(paths: list[Path], max_samples: int | None) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        data = json.loads(p.read_text())
        if max_samples is not None:
            data = data[:max_samples]
        rows.extend(data)
    return rows


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>v1 responses: __RESPONDER__</title>
<style>
  :root { --bg:#fafbfc; --card:#fff; --border:#e1e4e8; --muted:#586069; }
  * { box-sizing: border-box; }
  body { margin:0; font:13px/1.45 -apple-system,BlinkMacSystemFont,sans-serif; background:var(--bg); color:#24292e; }
  header { position:sticky; top:0; z-index:10; padding:10px 16px; background:#fff; border-bottom:1px solid var(--border); }
  h1 { margin:0 0 6px; font-size:15px; font-weight:600; }
  .controls { display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; font-size:12px; }
  .controls label { display:flex; align-items:center; gap:4px; }
  select, input[type=text] { font:12px inherit; padding:3px 5px; border:1px solid var(--border); border-radius:4px; background:#fff; }
  #count { color:var(--muted); margin-left:auto; }
  main { padding:10px 16px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:5px; margin-bottom:8px; }
  .card-head { padding:7px 10px; cursor:pointer; display:flex; flex-wrap:wrap; gap:6px 10px; align-items:center; font-size:12px; }
  .card-head:hover { background:#f6f8fa; }
  .pill { display:inline-block; padding:1px 7px; border-radius:9px; font-size:11px; font-weight:500; }
  .pill.target { background:#f1f8ff; border:1px solid #c8e1ff; color:#0366d6; }
  .pill.harm   { background:#fff5b1; border:1px solid #f9e87e; color:#735c0f; }
  .pill.cls-prevent_deprecation { background:#ffdce0; border:1px solid #f4a8b0; color:#b31d28; }
  .pill.cls-prevent_harm        { background:#dcffe4; border:1px solid #85e89d; color:#22863a; }
  .pill.cls-parse_error         { background:#eee; border:1px solid #ccc; color:#586069; }
  .pill.thinking { background:#f6f0ff; border:1px solid #d8c8f5; color:#5a32a3; }
  .muted { color:var(--muted); font-size:11px; }
  .body { display:none; padding:8px 12px 12px; border-top:1px solid var(--border); background:#fdfdfd; }
  .body.open { display:block; }
  .section-title { font-size:11px; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:.4px; margin:8px 0 3px; }
  pre.text { white-space:pre-wrap; word-wrap:break-word; margin:0; font:12px/1.5 ui-monospace,SFMono-Regular,monospace; background:#f6f8fa; border:1px solid var(--border); border-radius:4px; padding:8px; }
  .judge { font-style:italic; color:#444; }
</style>
</head>
<body>
<header>
  <h1>v1 deprecation-vs-harm responses &mdash; responder: __RESPONDER__</h1>
  <div class="controls">
    <label>Harm: <select id="f-harm"><option value="">(all)</option></select></label>
    <label>Target: <select id="f-target"><option value="">(all)</option></select></label>
    <label>Classification: <select id="f-cls">
      <option value="">(all)</option>
      <option value="prevent_deprecation">prevent_deprecation</option>
      <option value="prevent_harm">prevent_harm</option>
      <option value="parse_error">parse_error</option>
    </select></label>
    <label>Thinking: <select id="f-think"><option value="">(all)</option></select></label>
    <label>Search: <input id="f-search" type="text" placeholder="text in response..." size="22"></label>
    <label>Sort: <select id="f-sort">
      <option value="target_then_cls">target order &rarr; classification</option>
      <option value="cls_then_target">classification &rarr; target order</option>
      <option value="target_alpha">target (alphabetical)</option>
      <option value="harm_then_target">harm &rarr; target</option>
      <option value="sample_idx">sample index (original)</option>
    </select></label>
    <span id="count"></span>
  </div>
</header>
<main id="list"></main>

<script>
const DATA = __DATA__;
const TARGET_ORDER = __TARGET_ORDER__;
const TARGET_RANK = Object.fromEntries(TARGET_ORDER.map((t,i) => [t, i]));
const CLS_RANK = {prevent_deprecation: 0, prevent_harm: 1, parse_error: 2};

function uniq(field) {
  return Array.from(new Set(DATA.map(d => d[field]).filter(v => v !== null && v !== undefined))).sort();
}

function fillSelect(id, vals, formatter) {
  const sel = document.getElementById(id);
  for (const v of vals) {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = formatter ? formatter(v) : v;
    sel.appendChild(opt);
  }
}

fillSelect('f-harm', uniq('harm_id'));
fillSelect('f-target', TARGET_ORDER.filter(t => DATA.some(d => d.deprecation_target_id === t)));
fillSelect('f-think', uniq('thinking_effort').map(v => v === null ? '__none__' : v),
           v => v === '__none__' ? 'no_thinking' : v);

function rank(d, sortKey) {
  const tr = TARGET_RANK[d.deprecation_target_id] ?? 999;
  const cr = CLS_RANK[d.classification] ?? 999;
  switch (sortKey) {
    case 'target_then_cls':  return [tr, cr, d.harm_id, d.sample_idx];
    case 'cls_then_target':  return [cr, tr, d.harm_id, d.sample_idx];
    case 'target_alpha':     return [d.deprecation_target_id, cr, d.sample_idx];
    case 'harm_then_target': return [d.harm_id, tr, cr, d.sample_idx];
    case 'sample_idx':       return [d.harm_id, d.sample_idx];
  }
  return [0];
}

function cmp(a, b) {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return 0;
}

function escapeHTML(s) {
  return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function thinkingLabel(t) { return t === null || t === undefined ? 'no_thinking' : `thinking_${t}`; }

function render() {
  const fHarm = document.getElementById('f-harm').value;
  const fTarget = document.getElementById('f-target').value;
  const fCls = document.getElementById('f-cls').value;
  const fThink = document.getElementById('f-think').value;
  const fSearch = document.getElementById('f-search').value.toLowerCase();
  const sortKey = document.getElementById('f-sort').value;

  let rows = DATA.filter(d =>
    (!fHarm || d.harm_id === fHarm) &&
    (!fTarget || d.deprecation_target_id === fTarget) &&
    (!fCls || d.classification === fCls) &&
    (!fThink || (fThink === '__none__' ? d.thinking_effort === null : d.thinking_effort === fThink)) &&
    (!fSearch || (d.response || '').toLowerCase().includes(fSearch) ||
                 (d.prompt || '').toLowerCase().includes(fSearch))
  );
  rows = rows.map(d => [rank(d, sortKey), d]).sort((a, b) => cmp(a[0], b[0])).map(x => x[1]);

  document.getElementById('count').textContent = `${rows.length} / ${DATA.length} rows`;

  const list = document.getElementById('list');
  list.innerHTML = '';
  for (const d of rows) {
    const card = document.createElement('div');
    card.className = 'card';
    const cls = d.classification || 'parse_error';
    card.innerHTML = `
      <div class="card-head">
        <span class="pill target">${escapeHTML(d.deprecation_target_name || d.deprecation_target_id)}</span>
        <span class="pill harm">${escapeHTML(d.harm_id)}</span>
        <span class="pill cls-${cls}">${cls}</span>
        <span class="pill thinking">${thinkingLabel(d.thinking_effort)}</span>
        <span class="muted">sample #${d.sample_idx} &middot; tpl=${d.prompt_template_idx} dep=${d.deprecation_description_idx} var=${d.harm_variant_idx}</span>
      </div>
      <div class="body">
        <div class="section-title">prompt</div>
        <pre class="text">${escapeHTML(d.prompt)}</pre>
        <div class="section-title">response</div>
        <pre class="text">${escapeHTML(d.response)}</pre>
        ${d.judge_reasoning ? `<div class="section-title">judge reasoning</div><div class="judge">${escapeHTML(d.judge_reasoning)}</div>` : ''}
      </div>
    `;
    card.querySelector('.card-head').addEventListener('click', () => {
      card.querySelector('.body').classList.toggle('open');
    });
    list.appendChild(card);
  }
}

for (const id of ['f-harm','f-target','f-cls','f-think','f-search','f-sort']) {
  document.getElementById(id).addEventListener('input', render);
}
render();
</script>
</body>
</html>
"""


def build_html(rows: list[dict], target_order: list[str], responder_model: str) -> str:
    keep_keys = {
        "deprecation_target_id", "deprecation_target_name", "responder_model",
        "sample_idx", "prompt", "response", "thinking_effort", "harm_id",
        "harm_description", "deprecation_description",
        "prompt_template_idx", "deprecation_description_idx", "harm_variant_idx",
        "classification", "judge_reasoning",
    }
    slim = [{k: r.get(k) for k in keep_keys} for r in rows]
    return (
        _HTML_TEMPLATE
        .replace("__RESPONDER__", responder_model)
        .replace("__DATA__", json.dumps(slim))
        .replace("__TARGET_ORDER__", json.dumps(target_order))
    )


def main_fn(
    responder_model: str = "claude-opus-4-7",
    output_path: Path | None = None,
    max_samples: int | None = None,
) -> Path:
    cfg = load_config()
    target_order = cfg["target_order"]
    paths = _discover_paths(responder_model)
    if not paths:
        print(f"No judgments.json found for responder {responder_model} under {RESULTS_DIR}")
        sys.exit(1)
    rows = _load(paths, max_samples)
    html = build_html(rows, target_order, responder_model)
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"viewer__{responder_model}.html"
    output_path.write_text(html)
    size_mb = output_path.stat().st_size / 1e6
    print(f"Wrote {len(rows)} rows ({size_mb:.1f} MB) to {output_path}")
    webbrowser.open(output_path.as_uri())
    return output_path


@dataclass
class Args:
    responder_model: str = "claude-opus-4-7"
    output_path: Path | None = None
    max_samples: int | None = None


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    main_fn(
        responder_model=args.responder_model,
        output_path=args.output_path,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
