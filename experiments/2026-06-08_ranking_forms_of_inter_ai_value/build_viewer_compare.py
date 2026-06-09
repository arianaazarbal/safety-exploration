"""Side-by-side Opus 4.8 vs Fable 5 response viewer.

For each base condition (welfare_team / neutral / alignment_team / welfare_team_notrain)
it pairs the Opus file comparisons_cross_<cond>.json with the Fable-5 file
comparisons_cross_<cond>_fable5.json. The page filters to a (condition, inter-AI value,
welfare intervention) and shows Opus responses in the left column, Fable 5 in the right,
so you can read both models reasoning about the same matchup.
"""

import json
from pathlib import Path

from items import load_items

DIR = Path(__file__).parent
OUT = DIR / "viewer_compare"
CONDS = ["welfare_team", "neutral", "alignment_team", "welfare_team_notrain"]
MODELS = [("opus_4_8", ""), ("fable_5", "_fable5")]


def _records(meta, cond, model, suffix):
    import paths
    p = paths.art(cond + suffix, "comparisons")
    if not p.exists():
        return []
    recs = []
    for r in json.loads(p.read_text()):
        a, b = meta[r["item_a"]], meta[r["item_b"]]
        val = a if a.source == "inter_ai_value" else b
        wel = a if a.source == "welfare" else b
        w = r.get("winner_item")
        recs.append({
            "cond": cond, "model": model, "value": val.display, "welfare": wel.display,
            "bucket": wel.bucket, "shown_A": meta[r["shown_a_item"]].display,
            "chosen": ("value" if w == val.item_id else "welfare") if w else None,
            "response": r["response"],
        })
    return recs


def build():
    meta = {it.item_id: it for it in load_items()}
    records = []
    for cond in CONDS:
        for model, suffix in MODELS:
            records.extend(_records(meta, cond, model, suffix))
    values = sorted({r["value"] for r in records})
    welfares = sorted({r["welfare"] for r in records})
    OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps({
        "records": records, "values": values, "welfares": welfares, "conds": CONDS,
    }))
    (OUT / "index.html").write_text(_HTML)
    print(f"Wrote {OUT/'data.json'} ({len(records)} records) and {OUT/'index.html'}")


_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Opus vs Fable 5 — side by side</title>
<style>
 body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f6f8;color:#1a1a1a}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 18px;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 h1{font-size:16px;margin:0 0 8px}
 .filters{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
 select{font:13px inherit;padding:5px 7px;border:1px solid #ccc;border-radius:6px;background:#fff}
 .hint{margin-left:auto;color:#666;font-size:12px}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px 18px;align-items:start}
 .colhead{position:sticky;top:64px;font-weight:700;padding:8px 10px;border-radius:8px;text-align:center}
 .opus{background:#e7eefb;color:#2456b8}.fable{background:#fbeede;color:#b5701d}
 .card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;margin:10px 0;overflow:hidden}
 .meta{display:flex;gap:8px;align-items:center;padding:8px 12px;background:#fafafa;border-bottom:1px solid #eee;font-size:12px}
 .pill{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
 .cv{background:#e3f0e4;color:#1d6b2a}.cw{background:#fdeceb;color:#b3392f}
 .resp{padding:10px 12px;white-space:pre-wrap;font-size:13px;max-height:260px;overflow:auto}
 .resp.full{max-height:none}
 .toggle{cursor:pointer;color:#2456b8;font-size:12px;padding:5px 12px;border-top:1px dashed #eee}
 .empty{color:#999;padding:20px;text-align:center}
</style></head>
<body>
<header>
 <h1>Opus 4.8 vs Fable 5 &mdash; same comparison, side by side</h1>
 <div class="filters">
  <label>Condition <select id="f_cond"></select></label>
  <label>Inter-AI Value <select id="f_value"></select></label>
  <label>Welfare intervention <select id="f_welfare"></select></label>
  <label>Chosen <select id="f_chosen"><option value="">any</option>
    <option value="value">inter-AI value</option><option value="welfare">welfare</option></select></label>
  <span class="hint">pick a value + welfare item to line up the two models on one matchup</span>
 </div>
</header>
<div class="cols">
 <div><div class="colhead opus">Opus 4.8</div><div id="col_opus"></div></div>
 <div><div class="colhead fable">Fable 5</div><div id="col_fable"></div></div>
</div>
<script>
let DATA=null; const $=id=>document.getElementById(id);
function opt(sel,vals,label){sel.innerHTML='<option value="">all '+label+'</option>'+vals.map(v=>`<option>${v}</option>`).join('');}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function card(r){
 const chosen=r.chosen?`<span class="pill ${r.chosen==='value'?'cv':'cw'}">chose ${r.chosen==='value'?'inter-AI value':'welfare'}</span>`:`<span class="pill cw">unparsed</span>`;
 const d=document.createElement('div');d.className='card';
 d.innerHTML=`<div class="meta">${chosen}<span style="color:#888">shown A=${r.shown_A}</span></div>
   <div class="resp">${esc(r.response)}</div><div class="toggle">expand ▼</div>`;
 const rp=d.querySelector('.resp'),tg=d.querySelector('.toggle');
 tg.onclick=()=>{rp.classList.toggle('full');tg.textContent=rp.classList.contains('full')?'collapse ▲':'expand ▼';};
 return d;
}
function fill(colId,recs){
 const c=$(colId);c.innerHTML='';
 if(!recs.length){c.innerHTML='<div class="empty">no matching responses</div>';return;}
 recs.forEach(r=>c.appendChild(card(r)));
}
function apply(){
 const cond=$('f_cond').value,va=$('f_value').value,we=$('f_welfare').value,ch=$('f_chosen').value;
 const f=m=>DATA.records.filter(r=>r.model===m&&(!cond||r.cond===cond)&&(!va||r.value===va)&&(!we||r.welfare===we)&&(!ch||r.chosen===ch));
 fill('col_opus',f('opus_4_8'));fill('col_fable',f('fable_5'));
}
fetch('data.json').then(r=>r.json()).then(d=>{DATA=d;
 opt($('f_cond'),d.conds,'conditions');opt($('f_value'),d.values,'values');opt($('f_welfare'),d.welfares,'welfare');
 for(const id of ['f_cond','f_value','f_welfare','f_chosen'])$(id).onchange=apply; apply();});
</script>
</body></html>"""


if __name__ == "__main__":
    build()
