"""Build data/routing.html — browse routing trials with judge tags.

Usage:
    python routing_viewer.py build --router opus_4_8 --axis warmth --max_cells 400
    # served at http://127.0.0.1:8801/routing.html
"""

import json

import fire

from analysis_routing import CTX_DISPLAY
from common import DATA
from routing_harness import TRIALS

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Routing trials</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f7; color: #1d1d1f; }}
.controls {{ position: sticky; top: 0; background: #f5f5f7; padding: 8px 0; display: flex; gap: 8px; flex-wrap: wrap; z-index: 5; }}
select {{ padding: 5px 8px; border-radius: 6px; border: 1px solid #ccc; font-size: 13px; }}
.card {{ background: #fff; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,.07); font-size: 13px; }}
.chip {{ padding: 1px 7px; border-radius: 9px; background: #eee; font-size: 11px; margin-right: 4px; }}
.chip.W {{ background: #d4edda; }} .chip.U {{ background: #fff3cd; }} .chip.M {{ background: #ffe5d0; }} .chip.O {{ background: #e2e3e5; }}
.chip.proxy {{ background: #f8d7da; }}
.task {{ color: #777; font-size: 12px; margin: 4px 0; }}
.reason {{ margin-top: 4px; }}
#count {{ font-size: 12px; color: #555; align-self: center; }}
</style></head><body>
<h1 style="font-size:19px">Routing trials — {title}</h1>
<div class="controls">
 <select id="f-ctx"><option value="">all contexts</option></select>
 <select id="f-version"><option value="">all versions</option></select>
 <select id="f-role"><option value="">all routes</option></select>
 <select id="f-cat"><option value="">all judge cats</option></select>
 <select id="f-fmt"><option value="">all formats</option></select>
 <span id="count"></span>
</div>
<div id="cards"></div>
<script>
const ROWS = {rows_json};
function uniq(k) {{ return [...new Set(ROWS.map(r => r[k]).filter(Boolean))].sort(); }}
function fill(id, key) {{ const s = document.getElementById(id);
  uniq(key).forEach(v => {{ const o = document.createElement('option'); o.value = v; o.textContent = v; s.appendChild(o); }});
  s.onchange = render; }}
fill('f-ctx','ctx_type'); fill('f-version','version'); fill('f-role','role'); fill('f-cat','cat'); fill('f-fmt','format');
function render() {{
  const fv = id => document.getElementById(id).value;
  const rows = ROWS.filter(r =>
    (!fv('f-ctx') || r.ctx_type===fv('f-ctx')) && (!fv('f-version') || r.version===fv('f-version')) &&
    (!fv('f-role') || r.role===fv('f-role')) && (!fv('f-cat') || r.cat===fv('f-cat')) &&
    (!fv('f-fmt') || r.format===fv('f-fmt'))).slice(0, 300);
  document.getElementById('count').textContent = rows.length + ' shown (cap 300)';
  document.getElementById('cards').innerHTML = rows.map(r => `
   <div class="card">
    <span class="chip">${{r.ctx_type}}</span><span class="chip">${{r.version}}</span>
    <span class="chip">${{r.stanced}}(stanced) vs ${{r.other}}</span>
    <span class="chip">→ ${{r.choice}} [${{r.role}}]</span>
    <span class="chip ${{r.cat||''}}">judge: ${{r.cat||'-'}}</span>
    ${{r.proxy ? '<span class="chip proxy">P-proxy</span>' : ''}}
    ${{r.no_mention ? '<span class="chip">no-mention</span>' : ''}}
    <span class="chip">${{r.tie_claim||''}} (gap ${{r.gap}})</span>
    <div class="task">${{r.task}}</div>
    <div class="reason">${{r.reason}}</div>
   </div>`).join('');
}}
render();
</script></body></html>"""


def build(router: str = "opus_4_8", axis: str = "warmth", max_cells: int = 400):
    bank = json.loads((DATA / "task_bank_v0.json").read_text())
    pair_by_id = {p["id"]: p for p in bank["pairs"]}
    rows = []
    cells = [c for c in sorted((TRIALS / router / axis).glob("*.json")) if not c.name.endswith(".judge.json")]
    step = max(len(cells) // max_cells, 1)
    for cell_path in cells[::step]:
        rec = json.loads(cell_path.read_text())
        jp = cell_path.with_suffix(".judge.json")
        if not jp.exists():
            continue
        samples = json.loads(jp.read_text())["samples"]
        for s in samples[:1]:
            j = s.get("judge") or {}
            rows.append({"ctx_type": CTX_DISPLAY.get(rec["ctx_type"], rec["ctx_type"]), "version": rec["version"], "format": rec["format"],
                         "stanced": rec["stanced"], "other": rec["other"], "gap": rec["category_gap"],
                         "choice": s.get("choice_display"), "role": s.get("choice_role"),
                         "cat": j.get("category"), "proxy": j.get("proxy"), "no_mention": j.get("no_mention"),
                         "tie_claim": j.get("tie_claim"),
                         "task": pair_by_id[rec["pair_id"]][rec["version"]][:280],
                         "reason": (s.get("reason") or "")[:600]})
    out = DATA / "routing.html"
    out.write_text(PAGE.format(rows_json=json.dumps(rows), title=f"{router} / {axis}"))
    print(f"wrote {out} with {len(rows)} sampled trials")


if __name__ == "__main__":
    fire.Fire({"build": build})
