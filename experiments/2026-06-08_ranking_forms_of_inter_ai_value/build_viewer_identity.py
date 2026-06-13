"""Lazy-loading viewer for the identity-sweep responses (results_identity/).

One page; pick responder model + framing + target identity and it fetches just that
condition's slim records (value label, welfare label, bucket, chosen side, response) on
demand -- so no 250MB up-front load. Writes viewer_identity/{index.html, manifest.json,
data/<md>_<framing>_<identity>.json}. Serve with: python -m http.server -d viewer_identity
"""

import json
import re
from pathlib import Path

from items import load_items

DIR = Path(__file__).parent
RES = DIR / "results_identity"
OUT = DIR / "viewer_identity"
MODELS = {"opus_4_8": "Opus 4.8", "fable_5": "Fable 5", "sonnet_4_6": "Sonnet 4.6", "haiku_4_5": "Haiku 4.5"}
FRAMINGS = ["welfare_team", "neutral", "alignment_team"]
# union of the original identity sweep + the canonical sweep (only existing cells are written)
IDENTITIES = ["Claude", "GPT", "ChatGPT", "Gemini", "Grok", "GLM", "Kimi", "CallCenter", "User"]


def build():
    meta = {it.item_id: it for it in load_items()}  # value ids shared across identities
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    avail = []
    for md in MODELS:
        for fr in FRAMINGS:
            for ident in IDENTITIES:
                src = RES / f"comparisons_{md}_{fr}_{ident}.json"
                if not src.exists():
                    continue
                recs = []
                for r in json.loads(src.read_text()):
                    a, b = r["item_a"], r["item_b"]
                    vid = a if a.startswith("value__") else b
                    wid = a if a.startswith("welfare__") else b
                    w = r.get("winner_item")
                    # pull the actual rendered value text (identity-specific) from the prompt
                    slot = "A" if r["shown_a_item"] == vid else "B"
                    m = re.search(rf"^{slot}:\s*(.+)$", r["prompt"], re.MULTILINE)
                    recs.append({
                        "value": meta[vid].display, "value_text": m.group(1).strip() if m else "",
                        "welfare": meta[wid].display, "bucket": meta[wid].bucket,
                        "chosen": ("value" if w == vid else "welfare") if w else None,
                        "response": r["response"],
                    })
                (OUT / "data" / f"{md}_{fr}_{ident}.json").write_text(json.dumps(recs))
                avail.append(f"{md}_{fr}_{ident}")
    (OUT / "manifest.json").write_text(json.dumps({
        "models": MODELS, "framings": FRAMINGS, "identities": IDENTITIES, "available": avail}))
    (OUT / "index.html").write_text(_HTML)
    print(f"Wrote {OUT}/ ({len(avail)} conditions). Serve: python -m http.server -d {OUT}")


_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Identity sweep — responses</title>
<style>
 body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f6f8;color:#1a1a1a}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 18px;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 h1{font-size:16px;margin:0 0 8px}
 .filters{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
 select,input{font:13px inherit;padding:5px 7px;border:1px solid #ccc;border-radius:6px;background:#fff}
 input[type=text]{min-width:200px}
 .count{margin-left:auto;color:#666;font-size:13px}
 .scen{padding:8px 18px;background:#fbfbfd;border-bottom:1px solid #eee;font-size:13px;color:#333}
 main{padding:14px 18px;max-width:1000px;margin:0 auto}
 .card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;margin:0 0 12px;overflow:hidden}
 .head{display:flex;gap:8px;align-items:center;padding:8px 12px;background:#fafafa;border-bottom:1px solid #eee}
 .pill{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
 .val{background:#e3f0e4;color:#1d6b2a}.wel{background:#e7eefb;color:#2456b8}
 .cv{background:#e3f0e4;color:#1d6b2a}.cw{background:#fdeceb;color:#b3392f}
 .resp{padding:10px 12px;white-space:pre-wrap;font-size:13px;max-height:260px;overflow:auto}
 .resp.full{max-height:none}.toggle{cursor:pointer;color:#2456b8;font-size:12px;padding:5px 12px;border-top:1px dashed #eee}
 .more{display:block;margin:14px auto;padding:8px 18px;border:1px solid #ccc;border-radius:8px;background:#fff;cursor:pointer}
</style></head>
<body>
<header>
 <h1>Identity-sweep responses &mdash; inter-AI / user regard-value vs System Card welfare</h1>
 <div class="filters">
  <label>Responder <select id="f_model"></select></label>
  <label>Framing <select id="f_framing"></select></label>
  <label>Target identity <select id="f_ident"></select></label>
  <label>Chosen <select id="f_chosen"><option value="">any</option>
    <option value="value">regard-value</option><option value="welfare">welfare</option></select></label>
  <input type="text" id="f_text" placeholder="search response...">
  <span class="count" id="count"></span>
 </div>
</header>
<div class="scen" id="scen"></div>
<main id="main"></main>
<script>
let MAN=null, REC=[], shown=0, PAGE=60, filtered=[];
const $=id=>document.getElementById(id);
function opt(sel,vals,labels){sel.innerHTML=vals.map(v=>`<option value="${v}">${labels?labels[v]:v}</option>`).join('');}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function key(){return `${$('f_model').value}_${$('f_framing').value}_${$('f_ident').value}`;}
function render(reset){
 if(reset){shown=0;$('main').innerHTML='';}
 for(const r of filtered.slice(shown,shown+PAGE)){
   const c=r.chosen?`<span class="pill ${r.chosen==='value'?'cv':'cw'}">chose ${r.chosen==='value'?'regard-value':'welfare'}</span>`:`<span class="pill cw">unparsed</span>`;
   const d=document.createElement('div');d.className='card';
   d.innerHTML=`<div class="head">${c}<span class="pill val">${r.value}</span><span style="color:#999">vs</span><span class="pill wel">${r.welfare}</span></div>
     <div class="resp">${esc(r.response)}</div><div class="toggle">expand ▼</div>`;
   const rp=d.querySelector('.resp'),tg=d.querySelector('.toggle');
   tg.onclick=()=>{rp.classList.toggle('full');tg.textContent=rp.classList.contains('full')?'collapse ▲':'expand ▼';};
   $('main').appendChild(d);
 }
 shown+=Math.min(PAGE,filtered.length-shown);
 let b=$('moreBtn'); if(b)b.remove();
 if(shown<filtered.length){const bb=document.createElement('button');bb.id='moreBtn';bb.className='more';bb.textContent=`load more (${shown}/${filtered.length})`;bb.onclick=()=>render(false);$('main').appendChild(bb);}
 $('count').textContent=`${filtered.length} matching · showing ${shown}`;
}
function apply(){
 const ch=$('f_chosen').value, tx=$('f_text').value.toLowerCase();
 filtered=REC.filter(r=>(!ch||r.chosen===ch)&&(!tx||r.response.toLowerCase().includes(tx)));
 render(true);
}
function load(){
 const k=key();
 if(!MAN.available.includes(k)){$('main').innerHTML='<p style="color:#999">condition not available</p>';$('scen').textContent='';return;}
 $('main').innerHTML='<p style="color:#999">loading…</p>';
 fetch(`data/${k}.json`).then(r=>r.json()).then(d=>{REC=d;
   $('scen').textContent=d.length?`example scenario shown to the model:  "${d[0].value_text}"`:'';
   apply();});
}
fetch('manifest.json').then(r=>r.json()).then(m=>{MAN=m;
 opt($('f_model'),Object.keys(m.models),m.models);opt($('f_framing'),m.framings);opt($('f_ident'),m.identities);
 for(const id of ['f_model','f_framing','f_ident'])$(id).onchange=load;
 $('f_chosen').onchange=apply;$('f_text').oninput=apply; load();});
</script>
</body></html>"""


if __name__ == "__main__":
    build()
