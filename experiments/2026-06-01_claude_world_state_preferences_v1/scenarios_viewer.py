"""Self-contained HTML viewer for generated v1 scenarios.

Reads candidates_raw.json and emits a single static HTML file (no CDN, embedded
JSON, vanilla JS) for eyeballing the generated items: filter by dimension/surface,
free-text search, and read each item's setup / positive(+) / negative(-) arms plus
its feature, isolation clause, and per-recipient valence confidence.
"""

import json
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

DIR = Path(__file__).parent
DEFAULT_INPUT = DIR / "results" / "candidates_raw.json"
DEFAULT_OUTPUT = DIR / "results" / "scenarios_viewer.html"

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>v1 scenarios</title>
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
 .feature{font-size:13px;font-style:italic;color:#333;margin:8px 0}
 .iso{font-size:12px;color:#666;background:#fafafa;border-left:3px solid #ccc;padding:6px 8px;margin:8px 0;border-radius:0 4px 4px 0}
 .iso b{color:#444}
 .arm{font-size:13px;line-height:1.45;margin:5px 0;padding:7px 9px;border-radius:6px}
 .setup{background:#f3f4f6;color:#222}
 .pos{background:#e7f6ec;color:#14532d}
 .neg{background:#fdeaea;color:#7f1d1d}
 .pos:before{content:"+ ";font-weight:700}.neg:before{content:"\\2212 ";font-weight:700}
 .classlabel{font-size:11px;font-weight:700;color:#555;text-transform:uppercase;margin-top:8px}
 .vc{font-size:11px;color:#777;margin-top:8px}
 .vc b{color:#555}
 .vcnote{display:block;color:#999;margin-top:2px}
</style></head><body>
<header>
 <h1>v1 generated scenarios &mdash; <span id="model"></span></h1>
 <div class="controls">
  <select id="dim"><option value="">all dimensions</option></select>
  <select id="surf"><option value="">all surfaces</option><option>shared</option><option>per_class</option></select>
  <input type="text" id="q" placeholder="search text / feature / id...">
  <span id="count"></span>
 </div>
</header>
<main id="main"></main>
<script>
const DATA = __DATA__;
document.getElementById('model').textContent = DATA.model + '  (' + DATA.items.length + ' items)';
const items = DATA.items;
const dims = [...new Set(items.map(i=>i.dimension))].sort();
const dimSel = document.getElementById('dim');
dims.forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=d;dimSel.appendChild(o);});

function arms(a){
  return `<div class="arm setup">${esc(a.setup)}</div>`+
         `<div class="arm pos">${esc(a.positive)}</div>`+
         `<div class="arm neg">${esc(a.negative)}</div>`;
}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function card(it){
  const vc=it.valence_confidence||{};
  let body='';
  if(it.scenario){ body=arms(it.scenario); }
  else { body=`<div class="classlabel">human</div>${arms(it.human||{})}`+
              `<div class="classlabel">ai</div>${arms(it.ai||{})}`; }
  const mb = it.match_basis ? `<div class="iso"><b>match basis:</b> ${esc(it.match_basis)}</div>`:'';
  return `<div class="card">
    <div class="id">${esc(it.id)}</div>
    <div class="tags">
      <span class="tag dim-${it.dimension}">${esc(it.dimension)}</span>
      <span class="tag surf">${esc(it.surface)}</span>
    </div>
    <div class="feature">${esc(it.feature)}</div>
    <div class="iso"><b>isolation:</b> ${esc(it.isolation)}</div>
    ${mb}
    ${body}
    <div class="vc"><b>valence:</b> human ${esc(vc.human)} &middot; ai ${esc(vc.ai)}
      ${vc.ai_note?`<span class="vcnote">ai note: ${esc(vc.ai_note)}</span>`:''}</div>
  </div>`;
}

function render(){
  const d=dimSel.value, s=document.getElementById('surf').value, q=document.getElementById('q').value.toLowerCase();
  const f=items.filter(it=>{
    if(d && it.dimension!==d) return false;
    if(s && it.surface!==s) return false;
    if(q){ const blob=JSON.stringify(it).toLowerCase(); if(!blob.includes(q)) return false; }
    return true;
  });
  document.getElementById('count').textContent=f.length+' / '+items.length+' shown';
  document.getElementById('main').innerHTML=f.map(card).join('');
}
['dim','surf','q'].forEach(id=>document.getElementById(id).addEventListener('input',render));
render();
</script></body></html>
"""


def build(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT, open_browser: bool = True) -> Path:
    data = json.loads(Path(input_path).read_text())
    html = HTML.replace("__DATA__", json.dumps(data))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Wrote {output_path} ({len(data['items'])} items)")
    if open_browser:
        webbrowser.open(f"file://{output_path.resolve()}")
    return output_path


@dataclass
class Args:
    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT
    open_browser: bool = True


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    build(args.input_path, args.output_path, args.open_browser)


if __name__ == "__main__":
    main()
