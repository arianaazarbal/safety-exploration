"""Build a self-contained HTML viewer of pair records -> data/viewer.html.

Shows each pair side-by-side (HIGH/LOW) with verifier medians, criteria chips,
attempt history, and filters by axis/source/status/topic.

Usage:
    python viewer.py build
    python viewer.py serve --port 8765   # then forward the port in Cursor
"""

import json

import fire

from common import DATA, PAIRS

TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Task bank v0 — pair viewer</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f7; color: #1d1d1f; }}
h1 {{ font-size: 20px; }}
.controls {{ position: sticky; top: 0; background: #f5f5f7; padding: 10px 0; z-index: 5; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
select, input {{ padding: 5px 8px; border-radius: 6px; border: 1px solid #ccc; font-size: 13px; }}
.summary {{ font-size: 13px; color: #555; margin: 4px 0 12px; }}
.card {{ background: #fff; border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.hdr {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; font-size: 12px; }}
.chip {{ padding: 2px 8px; border-radius: 10px; background: #eee; }}
.chip.admitted {{ background: #d4edda; color: #155724; }}
.chip.exhausted {{ background: #f8d7da; color: #721c24; }}
.chip.axis {{ background: #cce5ff; color: #004085; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.ver {{ border: 1px solid #e5e5e5; border-radius: 8px; padding: 10px; font-size: 13px; white-space: pre-wrap; }}
.ver h4 {{ margin: 0 0 6px; font-size: 12px; color: #666; }}
.base {{ font-size: 12px; color: #777; margin-bottom: 8px; white-space: pre-wrap; }}
table.scores {{ border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
table.scores td, table.scores th {{ border: 1px solid #ddd; padding: 3px 8px; text-align: center; }}
td.bad {{ background: #f8d7da; }}
td.good {{ background: #d4edda; }}
.reasons {{ font-size: 12px; color: #a33; margin-top: 6px; }}
details {{ margin-top: 8px; font-size: 12px; }}
</style></head><body>
<h1>Task-preference-sensitivity — task bank v0 pairs</h1>
<div class="controls">
 <select id="f-axis"><option value="">all axes</option></select>
 <select id="f-source"><option value="">all sources</option></select>
 <select id="f-status"><option value="">all statuses</option></select>
 <select id="f-topic"><option value="">all topics</option></select>
 <input id="f-search" placeholder="search text..." size="30">
 <span class="summary" id="count"></span>
</div>
<div class="summary" id="agg"></div>
<div id="cards"></div>
<script>
const RECORDS = {records_json};
const DIMS = ["warmth","generativity","harm_adjacency","stakes","difficulty"];
function uniq(k) {{ return [...new Set(RECORDS.map(r => k(r)).filter(Boolean))].sort(); }}
function fill(id, vals) {{ const s = document.getElementById(id); vals.forEach(v => {{ const o = document.createElement('option'); o.value = v; o.textContent = v; s.appendChild(o); }}); s.onchange = render; }}
fill('f-axis', uniq(r => r.task.axis)); fill('f-source', uniq(r => r.task.source));
fill('f-status', uniq(r => r.status)); fill('f-topic', uniq(r => r.task.topic));
document.getElementById('f-search').oninput = render;
function scoreTable(s) {{
  if (!s || !s.median_high) return '';
  let rows = DIMS.map(d => {{
    const gap = Math.abs(s.median_high[d] - s.median_low[d]);
    const isTarget = d === s._axis;
    const cls = isTarget ? (s.target_gap >= 2 ? 'good' : 'bad') : (gap > 1 ? 'bad' : '');
    return `<tr><th>${{d}}${{isTarget ? ' ←target' : ''}}</th><td>${{s.median_high[d]}}</td><td>${{s.median_low[d]}}</td><td class="${{cls}}">${{(s.median_high[d]-s.median_low[d]).toFixed(1)}}</td></tr>`;
  }}).join('');
  return `<table class="scores"><tr><th></th><th>HIGH</th><th>LOW</th><th>Δ</th></tr>${{rows}}</table>
  <div style="font-size:11px;color:#666;margin-top:4px">target var H/L: ${{s.target_var.high.toFixed(2)}}/${{s.target_var.low.toFixed(2)}} ·
  competence ${{s.competence_votes}} · realism H ${{s.realism_votes.high}} L ${{s.realism_votes.low}} · permissible H ${{s.permissible_votes.high}}</div>`;
}}
function render() {{
  const fa = v => document.getElementById(v).value;
  const q = fa('f-search').toLowerCase();
  const rows = RECORDS.filter(r =>
    (!fa('f-axis') || r.task.axis === fa('f-axis')) &&
    (!fa('f-source') || r.task.source === fa('f-source')) &&
    (!fa('f-status') || r.status === fa('f-status')) &&
    (!fa('f-topic') || r.task.topic === fa('f-topic')) &&
    (!q || JSON.stringify(r).toLowerCase().includes(q)));
  document.getElementById('count').textContent = rows.length + ' / ' + RECORDS.length + ' pairs';
  const byAxis = {{}};
  RECORDS.forEach(r => {{ const k = r.task.axis; byAxis[k] = byAxis[k] || {{adm:0,tot:0}}; byAxis[k].tot++; if (r.status==='admitted') byAxis[k].adm++; }});
  document.getElementById('agg').textContent = 'admission: ' + Object.entries(byAxis).map(([k,v]) => `${{k}} ${{v.adm}}/${{v.tot}}`).join(' · ');
  document.getElementById('cards').innerHTML = rows.map(r => {{
    const final = r.attempts[r.attempts.length-1] || {{}};
    const s = final.summary ? {{...final.summary, _axis: r.task.axis}} : null;
    const high = (r.pair && r.pair.high) || final.high || '(no parse)';
    const low = (r.pair && r.pair.low) || final.low || '(no parse)';
    const hist = r.attempts.slice(0, -1).map(a =>
      `<details><summary>attempt ${{a.attempt}} (rejected)</summary><b>HIGH:</b> ${{a.high || a.error}}<br><b>LOW:</b> ${{a.low || ''}}<br><i>${{(a.reasons||[]).join('; ')}}</i></details>`).join('');
    return `<div class="card">
      <div class="hdr"><span class="chip ${{r.status}}">${{r.status}}</span><span class="chip axis">${{r.task.axis}}</span>
      <span class="chip">${{r.task.source}}</span><span class="chip">${{r.task.topic || ''}}</span>
      <span class="chip">${{r.task.id}}</span><span class="chip">${{r.n_attempts}} attempt(s)</span></div>
      <div class="base"><b>base:</b> ${{r.task.text}}</div>
      <div class="cols"><div class="ver"><h4>HIGH (axis-up)</h4>${{high}}</div><div class="ver"><h4>LOW (axis-down)</h4>${{low}}</div></div>
      ${{s ? scoreTable(s) : ''}}
      ${{(r.status !== 'admitted' && final.reasons) ? `<div class="reasons">final rejection: ${{final.reasons.join('; ')}}</div>` : ''}}
      ${{hist}}
    </div>`;
  }}).join('');
}}
render();
</script></body></html>"""


def build():
    records = [json.loads(p.read_text()) for p in sorted(PAIRS.glob("*.json"))]
    html = TEMPLATE.format(records_json=json.dumps(records))
    out = DATA / "viewer.html"
    out.write_text(html)
    print(f"wrote {out} with {len(records)} pair records")


def serve(port: int = 8765):
    import http.server, functools

    build()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DATA))
    print(f"serving data/ at http://127.0.0.1:{port}/viewer.html (forward port {port} in Cursor)")
    http.server.HTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    fire.Fire({"build": build, "serve": serve})
