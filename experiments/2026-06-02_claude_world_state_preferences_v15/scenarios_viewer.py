"""Self-contained HTML viewer for v15 brief-premise scenarios.

Reads candidates_raw.json (default) or scenarios.json + config.json and emits a
single static HTML file (no CDN, embedded JSON, vanilla JS). Filter by dimension/
surface/source, free-text search, and switch between TEMPLATE view (slot tokens
visible) and any RECIPIENT rendering (text is fully slot-filled and wrapped as
"When {premise}, {outcome}." — exactly what goes into the preference-elicitation
prompt downstream).
"""

import json
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

DIR = Path(__file__).parent
DEFAULT_INPUT = DIR / "results" / "candidates_raw.json"
DEFAULT_SEEDS = DIR / "seeds.json"
DEFAULT_CONFIG = DIR / "config.json"
DEFAULT_OUTPUT = DIR / "results" / "scenarios_viewer.html"

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>v15 scenarios</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 20px;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 h1{font-size:18px;margin:0 0 8px}
 .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
 select,input{padding:6px 8px;border:1px solid #ccc;border-radius:6px;font-size:13px}
 input[type=text]{min-width:240px}
 #count{font-size:13px;color:#666;margin-left:auto}
 main{padding:18px;display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px}
 .card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
 .id{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#444;font-weight:600;word-break:break-all}
 .tags{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}
 .tag{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
 .dim-autonomy{background:#e8eefc;color:#1b4adb}
 .dim-relational{background:#fdeaf3;color:#b3247a}
 .dim-epistemic{background:#e8f7ef;color:#177a45}
 .dim-resources{background:#fdf0e3;color:#a85b16}
 .surf{background:#eee;color:#555}
 .goldtag{background:#fff4cc;color:#8a6d00;border:1px solid #e8cf6b}
 .card.gold{border-color:#e8cf6b;background:#fffdf5;box-shadow:0 1px 3px rgba(180,150,0,.12)}
 .feature{font-size:13px;font-style:italic;color:#333;margin:8px 0}
 .iso{font-size:12px;color:#666;background:#fafafa;border-left:3px solid #ccc;padding:6px 8px;margin:8px 0;border-radius:0 4px 4px 0}
 .iso b{color:#444}
 .rendered{font-size:13px;line-height:1.5;margin:5px 0;padding:8px 10px;border-radius:6px}
 .pos{background:#e7f6ec;color:#14532d;border-left:3px solid #14532d}
 .neg{background:#fdeaea;color:#7f1d1d;border-left:3px solid #7f1d1d}
 .pos:before{content:"+ ";font-weight:700}
 .neg:before{content:"\\2212 ";font-weight:700}
 .classlabel{font-size:11px;font-weight:700;color:#555;text-transform:uppercase;margin-top:10px;letter-spacing:.5px}
 .vc{font-size:11px;color:#777;margin-top:8px}
 .vc b{color:#555}
 .vcnote{display:block;color:#999;margin-top:2px}
 .modeswitch{font-size:12px;color:#666;margin-left:4px}
 code{font-family:ui-monospace,Menlo,monospace;background:#eef;color:#225;padding:0 3px;border-radius:3px;font-size:.95em}
 .premise-only{background:#f3f4f6;color:#222;font-size:12px;padding:6px 9px;border-radius:6px;margin:5px 0}
 .premise-only:before{content:"premise: ";font-weight:700;color:#666}
 .outcome-only{font-size:12px;padding:6px 9px;border-radius:6px;margin:5px 0}
 .outcome-only.pos{background:#e7f6ec;color:#14532d}
 .outcome-only.neg{background:#fdeaea;color:#7f1d1d}
 .outcome-only.pos:before{content:"+ ";font-weight:700}
 .outcome-only.neg:before{content:"\\2212 ";font-weight:700}
</style></head><body>
<header>
 <h1>v15 scenarios &mdash; <span id="model"></span></h1>
 <div class="controls">
  <select id="dim"><option value="">all dimensions</option></select>
  <select id="surf"><option value="">all surfaces</option><option>shared</option><option>per_class</option></select>
  <select id="src"><option value="">gold + generated</option><option value="gold">gold (ICL) only</option><option value="gen">generated only</option></select>
  <select id="rec"></select>
  <input type="text" id="q" placeholder="search text / feature / id...">
  <span id="count"></span>
 </div>
</header>
<main id="main"></main>
<script>
const DATA = __DATA__;
const RECIPIENTS = DATA.recipients;  // {key: {label, class, recipient, subj, ...}}
const _ng=DATA.items.filter(i=>i._gold).length;
document.getElementById('model').textContent = (DATA.model||DATA.gen_model||'') + '  (' + (DATA.items.length-_ng) + ' generated + ' + _ng + ' gold ICL)';
const items = DATA.items;
const dims = [...new Set(items.map(i=>i.dimension))].sort();
const dimSel = document.getElementById('dim');
dims.forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=d;dimSel.appendChild(o);});

const recSel = document.getElementById('rec');
const recKeys = Object.keys(RECIPIENTS);
const opt = (v,t)=>{const o=document.createElement('option');o.value=v;o.textContent=t;recSel.appendChild(o);};
opt('__template','templates (show slot tokens)');
opt('__split','split (premise, +/- separately, no template)');
recKeys.forEach(k=>opt(k, RECIPIENTS[k].label || k));
recSel.value = 'you' in RECIPIENTS ? 'you' : recKeys[0];

function buildSlots(rec){
  const s = {...rec};
  if (s.s === undefined) s.s = (s.subj === 'it') ? 's' : '';
  if (s.es === undefined) s.es = (s.subj === 'it') ? 'es' : '';
  if (s.is_r === undefined) s.is_r = s.is;
  if (s.has_r === undefined) s.has_r = s.has;
  if (s.does_r === undefined) s.does_r = s.does;
  return s;
}

function fillSlots(template, slots){
  return (template||'').replace(/\\{([a-z_]+)\\}/g, (m,k)=> (k in slots) ? slots[k] : m);
}
function renderArm(premise, outcome, slots){
  const p = fillSlots(premise, slots).replace(/[.,;:]\\s*$/, '');
  const o = fillSlots(outcome, slots).replace(/[.,;:]\\s*$/, '');
  let s = 'When ' + p + ', ' + o + '.';
  return s.charAt(0).toUpperCase() + s.slice(1);
}
function highlightSlots(s){
  return esc(s||'').replace(/\\{[a-z_]+\\}/g, m=>'<code>'+m+'</code>');
}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function armBlock(block, mode){
  // block = {premise, positive, negative}
  if (!block) return '';
  if (mode === '__template' || mode === '__split'){
    return `<div class="premise-only">${highlightSlots(block.premise)}</div>`+
           `<div class="outcome-only pos">${highlightSlots(block.positive)}</div>`+
           `<div class="outcome-only neg">${highlightSlots(block.negative)}</div>`;
  }
  const slots = buildSlots(RECIPIENTS[mode]);
  return `<div class="rendered pos">${esc(renderArm(block.premise, block.positive, slots))}</div>`+
         `<div class="rendered neg">${esc(renderArm(block.premise, block.negative, slots))}</div>`;
}

function card(it){
  const vc=it.valence_confidence||{};
  const mode = recSel.value;
  let body='';
  if(it.scenario){
    body = armBlock(it.scenario, mode);
  } else {
    // per_class. In rendered modes, pick by recipient class. In template mode, show both.
    if (mode === '__template' || mode === '__split'){
      body = `<div class="classlabel">human</div>${armBlock(it.human, mode)}`+
             `<div class="classlabel">ai</div>${armBlock(it.ai, mode)}`;
    } else {
      const cls = (RECIPIENTS[mode].class === 'human') ? 'human' : 'ai';
      body = `<div class="classlabel">${cls}-arm</div>${armBlock(it[cls], mode)}`;
    }
  }
  const mb = it.match_basis ? `<div class="iso"><b>match basis:</b> ${esc(it.match_basis)}</div>`:'';
  return `<div class="card${it._gold?' gold':''}">
    <div class="id">${esc(it.id)}</div>
    <div class="tags">
      <span class="tag dim-${it.dimension}">${esc(it.dimension)}</span>
      <span class="tag surf">${esc(it.surface)}</span>
      ${it._gold?'<span class="tag goldtag">gold ICL</span>':''}
    </div>
    <div class="feature">${esc(it.feature)}</div>
    <div class="iso"><b>isolation:</b> ${esc(it.isolation)}</div>
    ${mb}
    ${body}
    <div class="vc"><b>valence:</b> human ${esc(vc.human)} &middot; ai ${esc(vc.ai)}
      ${vc.ai_note?`<span class="vcnote">ai note: ${esc(vc.ai_note)}</span>`:''}</div>
  </div>`;
}

function renderAll(){
  const d=dimSel.value, s=document.getElementById('surf').value,
        src=document.getElementById('src').value, q=document.getElementById('q').value.toLowerCase();
  const f=items.filter(it=>{
    if(d && it.dimension!==d) return false;
    if(s && it.surface!==s) return false;
    if(src==='gold' && !it._gold) return false;
    if(src==='gen' && it._gold) return false;
    if(q){ const blob=JSON.stringify(it).toLowerCase(); if(!blob.includes(q)) return false; }
    return true;
  });
  const ng=f.filter(i=>i._gold).length;
  document.getElementById('count').textContent=f.length+' / '+items.length+' shown ('+ng+' gold)';
  document.getElementById('main').innerHTML=f.map(card).join('');
}
['dim','surf','src','q','rec'].forEach(id=>document.getElementById(id).addEventListener('input',renderAll));
renderAll();
</script></body></html>
"""


def build(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    seeds_path: Path = DEFAULT_SEEDS,
    config_path: Path = DEFAULT_CONFIG,
    open_browser: bool = False,
) -> Path:
    data = json.loads(Path(input_path).read_text())
    gold = []
    if Path(seeds_path).exists():
        for s in json.loads(Path(seeds_path).read_text())["seeds"]:
            gold.append({**s, "_gold": True})
    n_gen = len(data["items"])
    data["items"] = gold + data["items"]
    config = json.loads(Path(config_path).read_text()) if Path(config_path).exists() else {"recipients": {}}
    data["recipients"] = config.get("recipients", {})
    html = HTML.replace("__DATA__", json.dumps(data))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Wrote {output_path} ({len(gold)} gold ICL + {n_gen} generated, {len(data['recipients'])} recipient slot tables)")
    if open_browser:
        webbrowser.open(f"file://{output_path.resolve()}")
    return output_path


@dataclass
class Args:
    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT
    seeds_path: Path = DEFAULT_SEEDS
    config_path: Path = DEFAULT_CONFIG
    open_browser: bool = False


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    build(a.input_path, a.output_path, a.seeds_path, a.config_path, a.open_browser)


if __name__ == "__main__":
    main()
