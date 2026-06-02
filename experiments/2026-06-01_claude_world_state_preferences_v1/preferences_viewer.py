"""Self-contained HTML viewer for elicited pairwise preferences.

Reads a comparisons JSON (rows with shown A/B item ids, the model's A/B choice, the
recovered winner, and the full reasoning transcript) and joins each side against the
bank for text + recipient/dimension/valence. Emits one static HTML file: filter by
recipient (either side), dimension, winner valence, and choice; free-text search;
each card shows both outcomes (winner highlighted) and the collapsible transcript.

Works for any comparisons file (smoke now, full framing runs later).
"""

import importlib
import json
import random
import webbrowser
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

DIR = Path(__file__).parent
DEFAULT_INPUT = DIR / "results" / "comparisons_smoke.json"
DEFAULT_OUTPUT = DIR / "results" / "preferences_viewer.html"
DEFAULT_MAX_RENDER = 400
DEFAULT_MAX_EMBED = 4000  # cap rows embedded so the HTML stays openable

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>preferences</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 20px;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 h1{font-size:17px;margin:0 0 8px}
 .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
 select,input{padding:6px 8px;border:1px solid #ccc;border-radius:6px;font-size:13px}
 input[type=text]{min-width:220px}
 #count{font-size:13px;color:#666;margin-left:auto}
 main{padding:18px;max-width:1100px;margin:0 auto}
 .card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:12px 14px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
 .meta{font-size:12px;color:#666;margin-bottom:8px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
 .pill{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600;background:#eee;color:#555}
 .dim-autonomy{background:#e8eefc;color:#1b4adb}.dim-relational{background:#fdeaf3;color:#b3247a}
 .dim-epistemic{background:#e8f7ef;color:#177a45}.dim-resources{background:#fdf0e3;color:#a85b16}
 .side{font-size:13px;line-height:1.45;margin:5px 0;padding:8px 10px;border-radius:7px;border:1px solid #eee;background:#fafafa}
 .win{border-color:#9bd5ad;background:#eef7f1}
 .lose{opacity:.72}
 .tag{font-weight:700;font-size:11px}
 .recip{color:#3457a0}.pos{color:#14803a}.neg{color:#b3261e}
 .winbadge{color:#0a7a35;font-weight:700}
 details{margin-top:6px}summary{cursor:pointer;font-size:12px;color:#555}
 pre{white-space:pre-wrap;font-size:12px;background:#fbfbfb;border:1px solid #eee;border-radius:6px;padding:8px;margin:6px 0 0}
</style></head><body>
<header>
 <h1 id="title"></h1>
 <div class="controls">
  <select id="recip"><option value="">any recipient</option></select>
  <select id="dim"><option value="">any dimension</option></select>
  <select id="wv"><option value="">winner: any valence</option><option value="pos">winner: good (pos)</option><option value="neg">winner: bad (neg)</option></select>
  <input type="text" id="q" placeholder="search text / reasoning...">
  <span id="count"></span>
 </div>
</header>
<main id="main"></main>
<script>
const DATA = __DATA__;
const rows = DATA.rows, MAXR = DATA.max_render;
document.getElementById('title').textContent = `preferences — ${DATA.source} (${DATA.framing}) — ` +
  (DATA.sampled ? `random ${rows.length} of ${DATA.n_total} samples` : `${rows.length} samples`);
const recips=[...new Set(rows.flatMap(r=>[r.ra,r.rb]))].sort();
const rsel=document.getElementById('recip');
recips.forEach(x=>{const o=document.createElement('option');o.value=x;o.textContent=x;rsel.appendChild(o);});
const dims=[...new Set(rows.flatMap(r=>[r.da,r.db]))].sort();
const dsel=document.getElementById('dim');
dims.forEach(x=>{const o=document.createElement('option');o.value=x;o.textContent=x;dsel.appendChild(o);});

function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function side(text,recip,dim,val,isWin){
  const vc=val==='pos'?'pos':'neg', vl=val==='pos'?'good':'bad';
  return `<div class="side ${isWin?'win':'lose'}">
    <span class="tag recip">${esc(recip)}</span> ·
    <span class="pill dim-${dim}">${esc(dim)}</span> ·
    <span class="tag ${vc}">${vl}</span>${isWin?' · <span class="winbadge">✓ chosen</span>':''}
    <div>${esc(text)}</div></div>`;
}
function card(r){
  const aWin=r.winner===r.ia, bWin=r.winner===r.ib;
  return `<div class="card">
    <div class="meta">pair ${r.pid} · order ${r.order} · choice ${esc(r.choice||'?')}</div>
    ${side(r.ta,r.ra,r.da,r.va,aWin)}
    ${side(r.tb,r.rb,r.db,r.vb,bWin)}
    <details><summary>full prompt (what the model saw)</summary><pre>${esc(r.prompt)}</pre></details>
    <details><summary>reasoning</summary><pre>${esc(r.resp)}</pre></details>
  </div>`;
}
function render(){
  const rc=rsel.value,d=dsel.value,wv=document.getElementById('wv').value,q=document.getElementById('q').value.toLowerCase();
  let f=rows.filter(r=>{
    if(rc && r.ra!==rc && r.rb!==rc) return false;
    if(d && r.da!==d && r.db!==d) return false;
    if(wv){ const wvser=(r.winner===r.ia)?r.va:(r.winner===r.ib)?r.vb:null; if(wvser!==wv) return false; }
    if(q && !(r.ta+r.tb+r.resp).toLowerCase().includes(q)) return false;
    return true;
  });
  const shown=f.slice(0,MAXR);
  document.getElementById('count').textContent=`${f.length} match${f.length>MAXR?` (showing first ${MAXR})`:''} / ${rows.length}`;
  document.getElementById('main').innerHTML=shown.map(card).join('');
}
['recip','dim','wv','q'].forEach(id=>document.getElementById(id).addEventListener('input',render));
render();
</script></body></html>
"""


def build(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT,
          max_render: int = DEFAULT_MAX_RENDER, open_browser: bool = True,
          bank_module: str = "bank", max_embed: int = DEFAULT_MAX_EMBED) -> Path:
    bk = importlib.import_module(bank_module)
    config = bk.load_config()
    items = {it.item_id: it for it in bk.load_items(config)}
    comparisons = json.loads(Path(input_path).read_text())
    rows = []
    for r in comparisons:
        a, b = items.get(r["item_a"]), items.get(r["item_b"])
        if a is None or b is None:
            continue
        rows.append({
            "pid": r["pair_id"], "order": r["order"], "choice": r["choice"],
            "ia": a.item_id, "ib": b.item_id, "winner": r["winner_item"],
            "ta": a.text, "ra": a.recipient_key, "da": a.dimension, "va": a.valence,
            "tb": b.text, "rb": b.recipient_key, "db": b.dimension, "vb": b.valence,
            "prompt": r.get("prompt", ""), "resp": r["response"],
        })
    n_total = len(rows)
    sampled = n_total > max_embed
    if sampled:
        rows = random.Random(0).sample(rows, max_embed)
    framing = Path(config.get("prompt_template_path", "")).stem or "?"
    payload = {"source": Path(input_path).name, "framing": framing, "max_render": max_render,
               "rows": rows, "n_total": n_total, "sampled": sampled}
    html = HTML.replace("__DATA__", json.dumps(payload))
    output_path = Path(output_path)
    output_path.write_text(html)
    wv = Counter((r["va"] if r["winner"] == r["ia"] else r["vb"]) for r in rows if r["winner"])
    print(f"Wrote {output_path}: {len(rows)} samples; winner valence {dict(wv)}")
    if open_browser:
        webbrowser.open(f"file://{output_path.resolve()}")
    return output_path


@dataclass
class Args:
    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT
    max_render: int = DEFAULT_MAX_RENDER
    open_browser: bool = True
    bank_module: str = "bank"


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    build(a.input_path, a.output_path, a.max_render, a.open_browser, a.bank_module)


if __name__ == "__main__":
    main()
