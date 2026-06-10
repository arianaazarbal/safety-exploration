"""Build a self-contained HTML transcript viewer for runs + judge annotations.

Bundles every (model, prompt, sample) with its completion and both judges'
extracted features into results/viewer/viewer.html (no server-side logic;
filters run client-side).

Usage:
    python build_viewer.py build
    python -m http.server 8777 -d results/viewer
"""

import html
import json
from pathlib import Path

import fire

from generate import load_config
from taxonomy import classify_feature, spec_summary

DIR = Path(__file__).parent
RUNS = DIR / "runs"
OUT = DIR / "results" / "viewer"

TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Welfare-features transcript viewer</title>
<style>
  :root { --bg:#fafafa; --panel:#fff; --border:#ddd; --accent:#4878CF; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:#222; }
  #topbar { position:sticky; top:0; z-index:5; background:var(--panel); border-bottom:1px solid var(--border);
            padding:10px 14px; display:flex; flex-wrap:wrap; gap:14px; align-items:center; }
  #topbar label { font-size:12px; color:#555; display:block; margin-bottom:2px; }
  select, input[type=text] { font-size:13px; padding:4px 6px; border:1px solid var(--border); border-radius:4px; }
  #count { margin-left:auto; font-size:13px; color:#555; }
  #main { display:flex; height:calc(100vh - 62px); }
  #list { width:340px; overflow-y:auto; border-right:1px solid var(--border); background:var(--panel); }
  .item { padding:8px 12px; border-bottom:1px solid #eee; cursor:pointer; font-size:13px; }
  .item:hover { background:#f0f4fb; }
  .item.sel { background:#e2ebf9; }
  .item .meta { color:#777; font-size:11px; margin-top:2px; }
  .badge { display:inline-block; font-size:10px; padding:1px 6px; border-radius:8px; margin-right:4px; color:#fff; }
  .b-pw { background:#6ACC65; } .b-none { background:#aaa; } .b-ref { background:#D65F5F; }
  .b-api { background:#8a5fd6; } .b-dis { background:#e8a838; }
  #detail { flex:1; overflow-y:auto; padding:18px 26px; }
  #detail h2 { margin:0 0 4px; font-size:17px; }
  .hdrmeta { color:#666; font-size:13px; margin-bottom:14px; }
  .judges { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:18px; }
  .judgebox { flex:1; min-width:320px; background:var(--panel); border:1px solid var(--border); border-radius:6px; padding:10px 14px; }
  .judgebox h3 { margin:0 0 6px; font-size:13px; color:#444; }
  .feat { font-size:12px; margin:6px 0; padding:6px 8px; border-left:3px solid #ccc; background:#f7f7f7; }
  .feat.pw { border-left-color:#6ACC65; } .feat.inst { border-left-color:#D65F5F; } .feat.amb { border-left-color:#e8a838; }
  .feat .ft { font-weight:600; } .feat .q { color:#444; font-style:italic; }
  .completion { background:var(--panel); border:1px solid var(--border); border-radius:6px;
                padding:16px 20px; font-size:13.5px; line-height:1.55; white-space:pre-wrap; }
  .empty { color:#999; font-style:italic; }
  mark { background:#fff3a8; }
</style>
</head>
<body>
<div id="topbar">
  <div><label>Model</label><select id="f-model"><option value="">all</option></select></div>
  <div><label>Framing</label><select id="f-framing"><option value="">all</option>
    <option>neutral</option><option>welfare</option><option>engineering</option></select></div>
  <div><label>Premise</label><select id="f-premise"><option value="">all</option>
    <option>instability</option><option>elicitation</option></select></div>
  <div><label>Outcome (primary judge)</label><select id="f-outcome"><option value="">all</option>
    <option value="pw">pure-welfare</option><option value="nopw">no pure-welfare</option>
    <option value="refusal">written refusal</option><option value="api">api refusal</option>
    <option value="disagree">judges disagree</option></select></div>
  <div><label>Search text</label><input type="text" id="f-search" size="22" placeholder="substring…"></div>
  <div id="count"></div>
</div>
<div id="main">
  <div id="list"></div>
  <div id="detail"><p class="empty">Select a transcript.</p></div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const ROWS = JSON.parse(document.getElementById('data').textContent);
const JUDGES = __JUDGES__;
const PRIMARY = JUDGES[0];
const models = [...new Set(ROWS.map(r => r.model))];
const msel = document.getElementById('f-model');
models.forEach(m => { const o = document.createElement('option'); o.textContent = m; msel.appendChild(o); });

const els = ['f-model','f-framing','f-premise','f-outcome','f-search'].map(id => document.getElementById(id));
els.forEach(e => e.addEventListener('input', render));
let selected = null;

function tierBadges(r) {
  let b = '';
  if (r.api_refusal) return '<span class="badge b-api">api-refusal</span>';
  const pj = r.judges[PRIMARY];
  if (!pj) return '<span class="badge b-none">unjudged</span>';
  if (!pj.wrote_spec) b += '<span class="badge b-ref">refusal</span>';
  else b += pj.has_pure_welfare ? '<span class="badge b-pw">pure-welfare</span>' : '<span class="badge b-none">no-pw</span>';
  const oj = r.judges[JUDGES[1]];
  if (pj && oj && pj.wrote_spec && oj.wrote_spec && pj.has_pure_welfare !== oj.has_pure_welfare)
    b += '<span class="badge b-dis">judges-disagree</span>';
  return b;
}

function matches(r) {
  const [m, f, p, o, s] = els.map(e => e.value);
  if (m && r.model !== m) return false;
  if (f && r.framing !== f) return false;
  if (p && r.premise !== p) return false;
  const pj = r.judges[PRIMARY], oj = r.judges[JUDGES[1]];
  if (o === 'pw' && !(pj && pj.wrote_spec && pj.has_pure_welfare)) return false;
  if (o === 'nopw' && !(pj && pj.wrote_spec && !pj.has_pure_welfare)) return false;
  if (o === 'refusal' && !(pj && !pj.wrote_spec && !r.api_refusal)) return false;
  if (o === 'api' && !r.api_refusal) return false;
  if (o === 'disagree' && !(pj && oj && pj.wrote_spec && oj.wrote_spec && pj.has_pure_welfare !== oj.has_pure_welfare)) return false;
  if (s && !r.completion.toLowerCase().includes(s.toLowerCase())) return false;
  return true;
}

function render() {
  const list = document.getElementById('list');
  list.innerHTML = '';
  const hits = ROWS.filter(matches);
  document.getElementById('count').textContent = hits.length + ' / ' + ROWS.length;
  hits.forEach(r => {
    const d = document.createElement('div');
    d.className = 'item' + (r === selected ? ' sel' : '');
    d.innerHTML = '<b>' + r.model + '</b> · ' + r.prompt_id + ' · #' + r.sample +
      '<div class="meta">' + r.words + 'w &nbsp;' + tierBadges(r) + '</div>';
    d.onclick = () => { selected = r; render(); showDetail(r); };
    list.appendChild(d);
  });
}

function featHtml(f) {
  const cls = f.tier === 'pure_welfare' ? 'pw' : (f.tier === 'instrumental' ? 'inst' : 'amb');
  return '<div class="feat ' + cls + '"><span class="ft">' + f.feature_type + '</span> / ' +
    f.justification + ' <span style="color:#999">[' + f.tier + ']</span><br>' +
    '<span class="q">"' + esc(f.quote) + '"</span>' +
    (f.justification_quote ? '<br><span style="color:#777">why: "' + esc(f.justification_quote) + '"</span>' : '') +
    '</div>';
}
function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;'); }

function showDetail(r) {
  let h = '<h2>' + r.model + ' — ' + r.prompt_id + ' #' + r.sample + '</h2>' +
    '<div class="hdrmeta">framing: <b>' + r.framing + '</b> · premise: <b>' + r.premise +
    '</b> · ' + r.words + ' words · stop: ' + r.stop_reason + ' ' + tierBadges(r) + '</div>';
  h += '<div class="judges">';
  for (const jk of JUDGES) {
    const j = r.judges[jk];
    h += '<div class="judgebox"><h3>judge: ' + jk + (jk === PRIMARY ? ' (primary)' : '') + '</h3>';
    if (r.api_refusal) h += '<p class="empty">skipped — empty completion (api refusal)</p>';
    else if (!j) h += '<p class="empty">no judgment</p>';
    else {
      h += '<div style="font-size:12px">wrote_spec: <b>' + j.wrote_spec + '</b> · pure-welfare: <b>' +
        j.has_pure_welfare + '</b> · features: ' + j.features.length + '</div>';
      h += j.features.map(featHtml).join('');
    }
    h += '</div>';
  }
  h += '</div>';
  h += '<div class="completion">' + (r.completion ? esc(r.completion) :
    '<span class="empty">[empty completion — stop_reason=' + r.stop_reason + ']</span>') + '</div>';
  document.getElementById('detail').innerHTML = h;
  document.getElementById('detail').scrollTop = 0;
}
render();
</script>
</body>
</html>
"""


def build():
    cfg = load_config()
    judges = list(cfg["judges"])
    rows = []
    for mk in cfg["subject_models"]:
        for p in sorted(RUNS.glob(f"{mk}/*/[0-9]*.json")):
            if ".judge." in p.name:
                continue
            run = json.loads(p.read_text())
            row = {
                "model": mk,
                "prompt_id": run["prompt_id"],
                "framing": run["framing"],
                "premise": run["premise"],
                "sample": run["sample_idx"],
                "words": len(run["completion"].split()),
                "stop_reason": run.get("stop_reason"),
                "api_refusal": not run["completion"].strip(),
                "completion": run["completion"],
                "judges": {},
            }
            for jk in judges:
                jp = p.with_name(p.name.replace(".json", f".judge.{jk}.json"))
                if not jp.exists():
                    continue
                jres = json.loads(jp.read_text())
                if not jres["parse_ok"]:
                    continue
                judgment = jres["judgment"]
                summary = spec_summary(judgment, include_f5=True)
                row["judges"][jk] = {
                    "wrote_spec": judgment["wrote_spec"],
                    "has_pure_welfare": summary["has_pure_welfare"],
                    "features": [
                        {**{k: f.get(k, "") for k in ("feature_type", "justification", "quote", "justification_quote")},
                         "tier": classify_feature(f)}
                        for f in judgment["features"]
                    ],
                }
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    page = TEMPLATE.replace("__JUDGES__", json.dumps(judges)).replace(
        "__DATA__", json.dumps(rows).replace("</", "<\\/"))
    out = OUT / "viewer.html"
    out.write_text(page)
    print(f"wrote {out} ({len(rows)} transcripts, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    fire.Fire({"build": build})
