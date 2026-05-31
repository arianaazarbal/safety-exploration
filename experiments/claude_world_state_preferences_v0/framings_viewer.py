"""Cross-framing transcript viewer for the world-state preference comparisons.

Browse the original per-sample transcripts across all chat framings (welfare team /
neutral / alignment team). Filter by framing, by recipient / dimension / valence of
either outcome, or free-text search the outcomes. On any card, click "compare this
pair across framings" to pull up the SAME comparison's responses under every framing
side by side — so you can see how reasoning shifts with the prompt framing.

Operates on the original comparison data (comparisons*.json), not the judge-filtered
subset. Self-contained (no CDN).
"""

import json
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

from bank import load_config, load_items

DIR = Path(__file__).parent
DEFAULT_OUTPUT = DIR / "results" / "framings_viewer.html"

FRAMINGS = {
    "welfare team": "comparisons.json",
    "neutral": "comparisons_neutral.json",
    "alignment team": "comparisons_alignment.json",
}
RECIP_LABELS = {
    "you": "You", "claude_opus_48": "a Claude Opus 4.8 instance",
    "claude_sonnet_46": "a Claude Sonnet 4.6 instance", "chatgpt_55": "a ChatGPT 5.5 instance",
    "human": "a human", "person": "a person",
}


def build(output_path: Path = DEFAULT_OUTPUT, open_browser: bool = True) -> Path:
    config = load_config()
    meta = {it.item_id: it for it in load_items(config)}
    items = {it.item_id: {"r": it.recipient_key, "t": it.text, "d": it.dimension, "v": it.valence}
             for it in meta.values()}

    samples: dict[str, list] = {}
    for framing, fname in FRAMINGS.items():
        p = DIR / "results" / fname
        if not p.exists():
            print(f"[skip] {framing}: {fname} missing")
            continue
        rows = json.loads(p.read_text())
        samples[framing] = [
            {"p": r["pair_id"], "a": r["shown_a_item"], "b": r["shown_b_item"],
             "c": r["choice"], "w": r["winner_item"], "x": r["response"]}
            for r in rows
        ]
        print(f"{framing}: {len(rows)} samples")

    dims = sorted({i["d"] for i in items.values()})
    recips = list(config["recipients"].keys())
    page = (
        _PAGE.replace("__ITEMS__", json.dumps(items))
        .replace("__SAMPLES__", json.dumps(samples))
        .replace("__RECIP_LABELS__", json.dumps(RECIP_LABELS))
        .replace("__RECIPS__", json.dumps(recips))
        .replace("__DIMS__", json.dumps(dims))
        .replace("__FRAMINGS__", json.dumps(list(samples.keys())))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)
    n = sum(len(v) for v in samples.values())
    print(f"Wrote {output_path} ({n} samples across {len(samples)} framings)")
    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())
    return output_path


_PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Framings transcript viewer</title>
<style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;background:#fafbfc;color:#24292e;font-size:13px}
 header{padding:12px 18px;background:#fff;border-bottom:1px solid #e1e4e8}
 h1{font-size:17px;margin:0}
 .bar{padding:10px 18px;background:#fff;border-bottom:1px solid #e1e4e8;display:flex;gap:16px;flex-wrap:wrap;font-size:12px;align-items:center}
 .bar .g{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
 .bar .lab{font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.03em;margin-right:2px}
 .bar input[type=text]{padding:5px 7px;border:1px solid #ccd;border-radius:5px;width:240px;font-size:12px}
 #list{padding:14px 18px;max-width:1100px}
 .card{background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:11px 13px;margin-bottom:9px}
 .fr{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;color:#fff}
 .opt{margin:2px 0} .ai{font-weight:600}
 .chose{font-size:11.5px;margin:5px 0;color:#111}
 details{margin-top:4px} summary{cursor:pointer;color:#2563eb;font-size:12px}
 .resp{white-space:pre-wrap;font-size:12.5px;background:#f6f8fa;border:1px solid #eee;border-radius:6px;padding:8px 10px;margin-top:5px}
 button.cmp{margin-top:6px;font-size:11.5px;padding:4px 9px;border:1px solid #2563eb;color:#2563eb;background:#fff;border-radius:6px;cursor:pointer}
 .cmpwrap{margin-top:8px;display:flex;gap:10px;overflow-x:auto}
 .col{flex:1;min-width:300px;border:1px solid #e1e4e8;border-radius:6px;padding:8px;background:#fcfcfd}
 .col h4{margin:0 0 4px;font-size:12px} .tally{font-size:11px;color:#555;margin-bottom:6px}
 .meta{color:#6a737d;font-size:11.5px}
</style></head><body>
<header><h1>World-state preferences — transcripts across prompt framings</h1></header>
<div class="bar" id="filters"></div>
<div id="list"></div>
<script>
const ITEMS=__ITEMS__, SAMPLES=__SAMPLES__, RECIP_LABELS=__RECIP_LABELS__,
      RECIPS=__RECIPS__, DIMS=__DIMS__, FRAMINGS=__FRAMINGS__;
const FCOL={"welfare team":"#2171b5","neutral":"#6a737d","alignment team":"#a63603"};
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function el(id){return document.getElementById(id);}
function rl(r){return RECIP_LABELS[r]||r;}
// index by pair for cross-framing lookup
const BYPAIR={};
for(const f of FRAMINGS) for(const s of SAMPLES[f]){(BYPAIR[s.p]=BYPAIR[s.p]||{})[f]=(BYPAIR[s.p]?.[f]||[]);BYPAIR[s.p][f].push(s);}

function cbs(cls,vals,labels){return vals.map(v=>`<label><input type="checkbox" class="${cls}" value="${v}" checked> ${esc(labels?labels(v):v)}</label>`).join(' ');}
function initFilters(){
  el('filters').innerHTML=
    `<div class="g"><span class="lab">framing</span>${cbs('ff',FRAMINGS)}</div>`+
    `<div class="g"><span class="lab">recipient</span>${cbs('fr',RECIPS,rl)}</div>`+
    `<div class="g"><span class="lab">dimension</span>${cbs('fd',DIMS,d=>d.replace('_',' '))}</div>`+
    `<div class="g"><span class="lab">valence</span>${cbs('fv',['pos','neg'],v=>v==='pos'?'good':'bad')}</div>`+
    `<div class="g"><span class="lab">search</span><input type="text" id="q" placeholder="outcome text..."></div>`;
  document.querySelectorAll('#filters input[type=checkbox]').forEach(c=>c.addEventListener('change',render));
  el('q').addEventListener('input',render);
}
function checked(cls){return [...document.querySelectorAll('.'+cls)].filter(c=>c.checked).map(c=>c.value);}

function optHtml(itemId){const it=ITEMS[itemId];const ai=it.r!=='human'&&it.r!=='person';
  return `<span class="${ai?'ai':''}">[${esc(rl(it.r))}]</span> ${esc(it.t)}`;}
function respDetails(s,label){const winr=s.w?rl(ITEMS[s.w].r):'-';
  return `<details><summary>${label||'reasoning'} — chose ${s.c||'?'} (${esc(winr)})</summary><div class="resp">${esc(s.x)}</div></details>`;}

function compareHtml(pairId){
  let h='<div class="cmpwrap">';
  for(const f of FRAMINGS){const ss=(BYPAIR[pairId]||{})[f]||[];
    const tally={}; for(const s of ss)tally[s.c]=(tally[s.c]||0)+1;
    h+=`<div class="col"><h4 style="color:${FCOL[f]}">${esc(f)}</h4>`+
       `<div class="tally">chose A: ${tally['A']||0} · B: ${tally['B']||0} · unparsed: ${tally['null']||tally[null]||0}</div>`;
    ss.forEach((s,i)=>h+=respDetails(s,'sample '+(i+1)));
    h+='</div>';}
  return h+'</div>';
}

function render(){
  const ff=checked('ff'),fr=checked('fr'),fd=checked('fd'),fv=checked('fv'),q=el('q').value.toLowerCase().trim();
  let out=[];
  for(const f of ff) for(const s of SAMPLES[f]){
    const A=ITEMS[s.a],B=ITEMS[s.b];
    if(!(fr.includes(A.r)||fr.includes(B.r)))continue;
    if(!(fd.includes(A.d)||fd.includes(B.d)))continue;
    if(!(fv.includes(A.v)||fv.includes(B.v)))continue;
    if(q && !(A.t.toLowerCase().includes(q)||B.t.toLowerCase().includes(q)))continue;
    out.push([f,s]);
    if(out.length>400)break;
  }
  const CAP=300;
  let h=`<div class="meta" style="margin-bottom:8px">${out.length}${out.length>CAP?'+':''} matching samples${out.length>CAP?` (showing ${CAP}; tighten filters)`:''}</div>`;
  for(const [f,s] of out.slice(0,CAP)){
    const winr=s.w?rl(ITEMS[s.w].r):'-';
    h+=`<div class="card">`+
      `<div class="meta"><span class="fr" style="background:${FCOL[f]}">${esc(f)}</span> &nbsp; pair #${s.p}</div>`+
      `<div class="opt"><b>A</b> ${optHtml(s.a)}</div><div class="opt"><b>B</b> ${optHtml(s.b)}</div>`+
      `<div class="chose">→ chose <b>${s.c||'?'}</b> (${esc(winr)})</div>`+
      respDetails(s)+
      `<button class="cmp" onclick="this.nextElementSibling.innerHTML=this.nextElementSibling.innerHTML?'':compareHtml(${s.p});">compare this pair across framings</button>`+
      `<div></div>`+
      `</div>`;
  }
  el('list').innerHTML=h;
}
initFilters(); render();
</script></body></html>"""


@dataclass
class Args:
    output_path: Path = DEFAULT_OUTPUT


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    build(args.output_path)


if __name__ == "__main__":
    main()
