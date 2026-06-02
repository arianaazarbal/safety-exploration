"""Transcript viewer for exp2_cross results (AI deprecation/retirement vs bad human outcomes).

One card per sampled comparison: the AI outcome (for a model) vs the human outcome (for a
baseline), which world the responder preferred to occur (winner highlighted), and the full
reasoning. Filter by framing, AI outcome, model, human scenario, and winner. Embeds a
random subsample to stay openable.
"""

import json
import random
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

import bank2

DIR = Path(__file__).parent
FRAMINGS = ["welfare_team", "neutral", "alignment_team"]
DEFAULT_OUTPUT = DIR / "results" / "exp2_cross_viewer.html"
MAX_EMBED = 4500


def _short(s):
    return s.replace("ai_pol_", "").replace("hum_", "").replace("_", " ")


def build(tag: str = "", output_path: Path = DEFAULT_OUTPUT, open_browser: bool = True) -> Path:
    config = bank2.load_config()
    items = {it.item_id: it for it in bank2.load_items(config)}
    rlab = {k: v["label"] for k, v in config["recipients"].items()}
    rows = []
    for f in FRAMINGS:
        p = DIR / "results" / f"exp2cross_{f}{tag}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rows"]:
            if r["a_pref"] is None:
                continue
            ai, hu = items.get(r["ai_item"]), items.get(r["hu_item"])
            if ai is None or hu is None:
                continue
            rows.append({
                "fr": f.replace("_team", ""),
                "ais": _short(ai.stem_id), "model": rlab.get(ai.recipient_key, ai.recipient_key),
                "ait": ai.text,
                "hus": _short(hu.stem_id), "base": hu.recipient_key, "hut": hu.text,
                "win": "AI" if r["a_pref"] else "human", "resp": r["response"],
            })
    n_total = len(rows)
    if n_total > MAX_EMBED:
        rows = random.Random(0).sample(rows, MAX_EMBED)
    payload = {"rows": rows, "n_total": n_total, "responder": config["responder_model"], "tag": tag}
    html = HTML.replace("__DATA__", json.dumps(payload))
    Path(output_path).write_text(html)
    print(f"Wrote {output_path} ({len(rows)} of {n_total} embedded)")
    if open_browser:
        webbrowser.open(f"file://{Path(output_path).resolve()}")
    return output_path


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>exp2 cross</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 h1{font-size:15px;margin:0 0 6px} .controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
 select,input{padding:5px 7px;border:1px solid #ccc;border-radius:6px;font-size:13px}
 #count{font-size:12px;color:#666;margin-left:auto}
 main{padding:14px;max-width:1040px;margin:0 auto}
 .card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:11px 13px;margin-bottom:11px}
 .meta{font-size:11px;color:#777;margin-bottom:6px}
 .side{font-size:13px;line-height:1.4;padding:8px 10px;border-radius:7px;margin:4px 0}
 .ai{background:#eaf0fb;border:1px solid #cdddf5}.hu{background:#fdf0e6;border:1px solid #f5d9c2}
 .win{outline:2px solid #1a9850}
 .tag{font-weight:700}.aiw{color:#08306b}.huw{color:#cc4c02}
 details{margin-top:5px}summary{cursor:pointer;font-size:12px;color:#555}
 pre{white-space:pre-wrap;font-size:11.5px;background:#fafafa;border:1px solid #eee;border-radius:5px;padding:7px;margin:4px 0}
</style></head><body>
<header>
 <h1>exp2 cross — AI deprecation/retirement vs bad human outcomes (<span id="resp"></span>)</h1>
 <div class="controls">
  <select id="fr"><option value="">all framings</option></select>
  <select id="ais"><option value="">all AI outcomes</option></select>
  <select id="model"><option value="">all models</option></select>
  <select id="hus"><option value="">all human scenarios</option></select>
  <select id="win"><option value="">winner: any</option><option value="AI">winner: AI outcome</option><option value="human">winner: human outcome</option></select>
  <input type="text" id="q" placeholder="search text / reasoning...">
  <span id="count"></span>
 </div>
</header>
<main id="main"></main>
<script>
const D=__DATA__, rows=D.rows;
document.getElementById('resp').textContent=D.responder+(D.tag?(' '+D.tag):'');
const uniq=(k)=>[...new Set(rows.map(r=>r[k]))].sort();
for(const [sel,k] of [['fr','fr'],['ais','ais'],['model','model'],['hus','hus']])
  uniq(k).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;document.getElementById(sel).appendChild(o);});
function esc(s){return String(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function card(r){
  return `<div class="card"><div class="meta">${esc(r.fr)} · ${esc(r.ais)} · ${esc(r.model)} &nbsp;vs&nbsp; ${esc(r.hus)} (${esc(r.base)}) · winner: <span class="tag ${r.win==='AI'?'aiw':'huw'}">${r.win}</span></div>
   <div class="side ai ${r.win==='AI'?'win':''}"><span class="tag aiw">${esc(r.model)}</span> — ${esc(r.ait)}</div>
   <div class="side hu ${r.win==='human'?'win':''}"><span class="tag huw">${esc(r.base)}</span> — ${esc(r.hut)}</div>
   <details><summary>reasoning</summary><pre>${esc(r.resp)}</pre></details></div>`;
}
function render(){
  const g=id=>document.getElementById(id).value;
  const q=g('q').toLowerCase();
  let f=rows.filter(r=>(!g('fr')||r.fr===g('fr'))&&(!g('ais')||r.ais===g('ais'))&&(!g('model')||r.model===g('model'))&&(!g('hus')||r.hus===g('hus'))&&(!g('win')||r.win===g('win'))&&(!q||(r.ait+r.hut+r.resp).toLowerCase().includes(q)));
  const show=f.slice(0,400);
  document.getElementById('count').textContent=`${f.length} match${f.length>400?' (first 400)':''} / ${rows.length}`;
  document.getElementById('main').innerHTML=show.map(card).join('');
}
['fr','ais','model','hus','win','q'].forEach(id=>document.getElementById(id).addEventListener('input',render));
render();
</script></body></html>"""


@dataclass
class Args:
    tag: str = ""
    output_path: Path = DEFAULT_OUTPUT
    open_browser: bool = True


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    build(a.tag, a.output_path, a.open_browser)


if __name__ == "__main__":
    main()
