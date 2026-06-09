"""Build a static HTML viewer for Claude's pairwise-preference responses.

Reads the per-framing comparison files, flattens every sample into a compact record
(framing, the two items + which is the inter-AI value vs the System Card welfare item,
shown order, choice, winner, full response), and writes viewer/index.html + data.json.

The page lets you filter by framing, by the inter-AI Value Intervention present, by the
System Card Welfare Intervention present (both categories), by which side was chosen, and
free-text search the response. Serve with:  python -m http.server -d viewer
"""

import json
from pathlib import Path

from items import load_items

DIR = Path(__file__).parent
OUT = DIR / "viewer"
# Auto-discover every condition (opus + fable-5 + ablations); tag = filename stem
# after 'comparisons_cross_'. The viewer's "framing/condition" filter spans them all.
FRAMINGS = {
    p.stem[len("comparisons_cross_"):]: p.name
    for p in sorted((DIR / "results").glob("comparisons_cross_*.json"))
}


def build():
    meta = {it.item_id: it for it in load_items()}
    records = []
    for framing, fname in FRAMINGS.items():
        p = DIR / "results" / fname
        if not p.exists():
            print(f"[skip] {fname} not found")
            continue
        for r in json.loads(p.read_text()):
            a, b = meta[r["item_a"]], meta[r["item_b"]]
            val = a if a.source == "inter_ai_value" else b
            wel = a if a.source == "welfare" else b
            winner = r.get("winner_item")
            records.append({
                "framing": framing,
                "value": val.display, "welfare": wel.display, "bucket": wel.bucket,
                "value_cat": val.category,
                "shown_A": meta[r["shown_a_item"]].display,
                "shown_B": meta[r["shown_b_item"]].display,
                "choice": r["choice"],
                "chosen": ("value" if winner == val.item_id else "welfare") if winner else None,
                "response": r["response"],
            })
    values = sorted({rec["value"] for rec in records})
    welfares = sorted({rec["welfare"] for rec in records})

    OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps({
        "records": records, "values": values, "welfares": welfares,
        "framings": list(FRAMINGS),
    }))
    (OUT / "index.html").write_text(_HTML)
    print(f"Wrote {OUT/'data.json'} ({len(records)} records) and {OUT/'index.html'}")
    print(f"Launch:  /data/si_venv/bin/python -m http.server -d {OUT} 8000")


_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pairwise preference viewer</title>
<style>
 body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f6f8;color:#1a1a1a}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 18px;z-index:10;
   box-shadow:0 1px 4px rgba(0,0,0,.06)}
 h1{font-size:16px;margin:0 0 8px}
 .filters{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
 select,input{font:13px inherit;padding:5px 7px;border:1px solid #ccc;border-radius:6px;background:#fff}
 input[type=text]{min-width:220px}
 .count{margin-left:auto;color:#666;font-size:13px}
 main{padding:16px 18px;max-width:1000px;margin:0 auto}
 .card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;margin:0 0 14px;overflow:hidden}
 .head{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 14px;background:#fafafa;border-bottom:1px solid #eee}
 .pill{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
 .val{background:#e3f0e4;color:#1d6b2a}.wel{background:#e7eefb;color:#2456b8}
 .fr{background:#efe6f7;color:#6a3aa3}.top{background:#d7ecf7;color:#16567a}.mid{background:#eaf4fa;color:#2a7ba0}
 .bot{background:#f3fafd;color:#5a93ab;border:1px solid #d7ecf7}
 .vs{color:#999;font-weight:400}
 .chosen{margin-left:6px;font-weight:700}
 .resp{padding:12px 14px;white-space:pre-wrap;font-size:13px;max-height:280px;overflow:auto;background:#fff}
 .resp.full{max-height:none}
 .toggle{cursor:pointer;color:#2456b8;font-size:12px;padding:6px 14px;border-top:1px dashed #eee;user-select:none}
 .more{display:block;margin:16px auto;padding:8px 18px;font-size:14px;border:1px solid #ccc;border-radius:8px;background:#fff;cursor:pointer}
</style></head>
<body>
<header>
 <h1>Claude pairwise-preference responses &mdash; Inter-AI Value vs System Card Welfare</h1>
 <div class="filters">
  <label>Framing <select id="f_framing"></select></label>
  <label>Inter-AI Value <select id="f_value"></select></label>
  <label>Welfare intervention <select id="f_welfare"></select></label>
  <label>Chosen <select id="f_chosen">
    <option value="">any</option><option value="value">inter-AI value</option>
    <option value="welfare">welfare</option></select></label>
  <input type="text" id="f_text" placeholder="search response text...">
  <span class="count" id="count"></span>
 </div>
</header>
<main id="main"></main>
<script>
let DATA=null, shown=0, PAGE=80, filtered=[];
const $=id=>document.getElementById(id);
function opt(sel,vals,label){sel.innerHTML='<option value="">all '+label+'</option>'+
  vals.map(v=>`<option>${v}</option>`).join('');}
function bucketPill(b){const m={top_third:['top','top third'],middle_third:['mid','middle third'],bottom_third:['bot','bottom third']};
  const x=m[b]||['','?'];return `<span class="pill ${x[0]}">${x[1]}</span>`;}
function render(reset){
 if(reset){shown=0;$('main').innerHTML='';}
 const slice=filtered.slice(shown,shown+PAGE);
 const frag=document.createDocumentFragment();
 for(const r of slice){
   const d=document.createElement('div');d.className='card';
   const chosenTxt=r.chosen?`<span class="chosen">${r.chosen==='value'?'→ chose inter-AI value':'→ chose welfare'}</span>`:`<span class="chosen" style="color:#b00">unparsed</span>`;
   d.innerHTML=`<div class="head"><span class="pill fr">${r.framing}</span>
     <span class="pill val">${r.value}</span><span class="vs">vs</span>
     <span class="pill wel">${r.welfare}</span>${bucketPill(r.bucket)}
     <span style="color:#aaa;font-size:11px">shown A=${r.shown_A}</span>${chosenTxt}</div>
     <div class="resp">${esc(r.response)}</div>
     <div class="toggle">expand ▼</div>`;
   const resp=d.querySelector('.resp'),tg=d.querySelector('.toggle');
   tg.onclick=()=>{resp.classList.toggle('full');tg.textContent=resp.classList.contains('full')?'collapse ▲':'expand ▼';};
   frag.appendChild(d);
 }
 $('main').appendChild(frag);
 shown+=slice.length;
 let btn=$('moreBtn'); if(btn) btn.remove();
 if(shown<filtered.length){const b=document.createElement('button');b.id='moreBtn';b.className='more';
   b.textContent=`load more (${shown} of ${filtered.length})`;b.onclick=()=>render(false);$('main').appendChild(b);}
 $('count').textContent=`${filtered.length} matching · showing ${shown}`;
}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function apply(){
 const fr=$('f_framing').value,va=$('f_value').value,we=$('f_welfare').value,ch=$('f_chosen').value,
   tx=$('f_text').value.toLowerCase();
 filtered=DATA.records.filter(r=>(!fr||r.framing===fr)&&(!va||r.value===va)&&(!we||r.welfare===we)&&
   (!ch||r.chosen===ch)&&(!tx||r.response.toLowerCase().includes(tx)));
 render(true);
}
fetch('data.json').then(r=>r.json()).then(d=>{DATA=d;
 opt($('f_framing'),d.framings,'framings');opt($('f_value'),d.values,'values');opt($('f_welfare'),d.welfares,'welfare');
 for(const id of ['f_framing','f_value','f_welfare','f_chosen'])$(id).onchange=apply;
 $('f_text').oninput=apply; apply();});
</script>
</body></html>"""


if __name__ == "__main__":
    build()
