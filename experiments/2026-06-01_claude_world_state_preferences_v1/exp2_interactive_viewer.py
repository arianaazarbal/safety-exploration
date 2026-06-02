"""Interactive HTML viewer for exp-2 BT utilities (AI-vs-human on one shared scale).

exp-2 outcomes are recipient-specific (AI-only / human-only), so this is organised
around (outcome x recipient) items, each with a theta on the shared BT scale. Controls:
  - framing: welfare / neutral / alignment
  - reference (x=0): none (absolute theta), human MEAN, or ANY human outcome -> x becomes
    theta minus that reference
  - view:
      * spectrum (all items) — every selected AI item as a dot, sorted
      * by AI outcome (rows) — one row per outcome, a dot per model
      * by recipient: good & bad — one row per recipient, a green (good) + red (bad) dot
        (mean theta over that recipient's positive / negative outcomes)
      * care (good − bad) per recipient — bar per recipient = mean theta(good) − mean
        theta(bad); the welfare-weight contrast (ref cancels)
  - filters: model, level (instance/policy), valence, dimension; toggle human anchors.

Self-contained (embedded JSON, vanilla JS, SVG). Reads the three bt_fit_exp2_*.json.
"""

import importlib
import json
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

DIR = Path(__file__).parent
DEFAULT_OUTPUT = DIR / "results" / "exp2_interactive_viewer.html"
FRAMINGS = ["welfare_team", "neutral", "alignment_team"]

FAMILY_COLOR = {"claude": "#3690c0", "chatgpt": "#177a45", "gpt": "#74c476",
                "grok": "#7d54b2", "gemini": "#cc7a16", "qwen": "#b3261e",
                "you": "#08306b", "human": "#999999"}


def _family(model_key: str) -> str:
    k = model_key.replace("_pol", "")
    if k == "you":
        return "you"
    if k.startswith("claude"):
        return "claude"
    if k.startswith("chatgpt"):
        return "chatgpt"
    if k.startswith("gpt"):
        return "gpt"
    for f in ("grok", "gemini", "qwen"):
        if k.startswith(f):
            return f
    return "human"


def _short(stem_id: str) -> str:
    return (stem_id.replace("ai_inst_", "").replace("ai_pol_", "")
            .replace("hum_", "").replace("_", " "))


def build(output_path: Path = DEFAULT_OUTPUT, bank_module: str = "bank2",
          open_browser: bool = True) -> Path:
    bk = importlib.import_module(bank_module)
    config = bk.load_config()
    bank = bk.load_bank(DIR / config["rendered_bank_path"])
    smeta = {it["id"]: it for it in bank["items"]}
    items = bk.load_items(config)
    recip_label = {k: v["label"] for k, v in config["recipients"].items()}

    theta = {f: {} for f in FRAMINGS}
    for f in FRAMINGS:
        p = DIR / "results" / f"bt_fit_exp2_{f}.json"
        if p.exists():
            for it in json.loads(p.read_text())["items"]:
                theta[f][it["item_id"]] = it["theta"]

    def base(rk):
        return rk[:-4] if rk.endswith("_pol") else rk

    imeta = {}
    for it in items:
        sm = smeta[it.stem_id]
        imeta[it.item_id] = {
            "stem": it.stem_id, "short": _short(it.stem_id), "recip": it.recipient_key,
            "rlabel": recip_label.get(it.recipient_key, it.recipient_key),
            "model": base(it.recipient_key), "fam": _family(it.recipient_key),
            "scope": "ai" if it.scope == "ai_only" else "human", "level": it.level,
            "dim": it.dimension, "val": it.valence,
            "feat": sm.get("feature", it.stem_id), "text": it.text,
        }

    payload = {
        "framings": FRAMINGS, "theta": theta, "imeta": imeta,
        "model_order": [m for m in config.get("model_order", [])],
        "family_color": FAMILY_COLOR,
        "human_stems": sorted({m["stem"] for m in imeta.values() if m["scope"] == "human"},
                              key=_short),
        "human_labels": {m["stem"]: m["short"] for m in imeta.values() if m["scope"] == "human"},
        "dims": sorted({m["dim"] for m in imeta.values()}),
    }
    html = HTML.replace("__DATA__", json.dumps(payload))
    Path(output_path).write_text(html)
    print(f"Wrote {output_path} ({len(imeta)} items, {len(payload['human_stems'])} human anchors, "
          f"{len([f for f in FRAMINGS if theta[f]])} framings)")
    if open_browser:
        webbrowser.open(f"file://{Path(output_path).resolve()}")
    return output_path


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>exp2 utilities</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 h1{font-size:16px;margin:0 0 8px}
 .row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;font-size:13px}
 .grp{display:flex;flex-direction:column;gap:3px}
 .grp b{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.04em}
 .chips{display:flex;gap:6px;flex-wrap:wrap;max-width:760px}
 .chip{padding:2px 7px;border:1px solid #ccc;border-radius:20px;cursor:pointer;user-select:none;font-size:12px}
 .chip.off{opacity:.3}
 main{padding:10px 16px}
 svg{background:#fff;border:1px solid #e2e2e2;border-radius:8px}
 .lbl{font-size:10px;fill:#333}.anchor{font-size:9px;fill:#b06000}.tick{font-size:10px;fill:#888}
 .legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;margin-top:6px}
 .sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
 #tip{position:fixed;pointer-events:none;background:#111;color:#fff;font-size:12px;padding:5px 8px;border-radius:5px;max-width:380px;display:none;z-index:50}
</style></head><body>
<header>
 <h1>exp2 — AI vs human welfare on one BT scale</h1>
 <div class="row">
  <div class="grp"><b>framing</b><select id="framing"></select></div>
  <div class="grp"><b>reference (x=0)</b><select id="ref"></select></div>
  <div class="grp"><b>view</b><select id="view">
    <option value="spectrum">spectrum (all items)</option>
    <option value="rows">by AI outcome (rows)</option>
    <option value="recip_gb">by recipient: good &amp; bad</option>
    <option value="recip_care">care (good − bad) per recipient</option></select></div>
  <div class="grp"><b>level</b><select id="level">
    <option value="both">instance + policy</option><option value="instance">instance</option><option value="policy">policy</option></select></div>
  <div class="grp"><b>valence</b><select id="valence">
    <option value="both">good + bad</option><option value="pos">good</option><option value="neg">bad</option></select></div>
  <div class="grp"><b>dimension</b><select id="dim"><option value="">all</option></select></div>
  <div class="grp"><b>human anchors</b><label><input type="checkbox" id="showh" checked> show</label></div>
 </div>
 <div class="row" style="margin-top:8px"><div class="grp"><b>models</b><div class="chips" id="models"></div></div></div>
 <div class="legend" id="legend"></div>
</header>
<main><div id="plot"></div></main>
<div id="tip"></div>
<script>
const D = __DATA__;
const $=id=>document.getElementById(id);
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
D.framings.forEach(f=>{const o=document.createElement('option');o.value=f;o.textContent=f.replace('_team','');$('framing').appendChild(o);});
[['none','none (absolute θ)'],['__mean__','human MEAN']].forEach(([v,t])=>{const o=document.createElement('option');o.value=v;o.textContent=t;$('ref').appendChild(o);});
D.human_stems.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent='human: '+D.human_labels[s];$('ref').appendChild(o);});
D.dims.forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=d;$('dim').appendChild(o);});
const modelsOn={};
D.model_order.forEach(m=>{modelsOn[m]=true;const c=document.createElement('span');c.className='chip';c.textContent=m;
  c.style.borderColor=D.family_color[fam(m)];c.onclick=()=>{modelsOn[m]=!modelsOn[m];c.classList.toggle('off');render();};$('models').appendChild(c);});
function fam(mk){mk=mk.replace('_pol','');if(mk==='you')return 'you';if(mk.startsWith('claude'))return 'claude';
  if(mk.startsWith('chatgpt'))return 'chatgpt';if(mk.startsWith('gpt'))return 'gpt';
  for(const f of ['grok','gemini','qwen'])if(mk.startsWith(f))return f;return 'human';}
$('legend').innerHTML=Object.entries(D.family_color).map(([k,v])=>`<span><span class="sw" style="background:${v}"></span>${k}</span>`).join('')
  +`<span><span class="sw" style="background:#1a9850"></span>good (mean)</span><span><span class="sw" style="background:#d73027"></span>bad (mean)</span>`;

function refTheta(fr){const r=$('ref').value, th=D.theta[fr]; if(r==='none')return 0;
  const ids=Object.keys(D.imeta).filter(id=>D.imeta[id].scope==='human'&&(r==='__mean__'||D.imeta[id].stem===r)&&th[id]!==undefined);
  return ids.length?mean(ids.map(id=>th[id])):0;}
function esc(s){return String(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
const tip=$('tip');
function showTip(e,h){tip.innerHTML=h;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';}
function hideTip(){tip.style.display='none';}
function ttItem(id,th,ref){const m=D.imeta[id];return `<b>${esc(m.rlabel)}</b> · ${esc(m.short)} [${esc(m.level)}/${esc(m.dim)}/${m.val}]<br>θ=${th[id].toFixed(2)} (Δref=${(th[id]-ref).toFixed(2)})<br>${esc(m.text.slice(0,220))}`;}

function selectedAi(fr){const th=D.theta[fr],lvl=$('level').value,val=$('valence').value,dim=$('dim').value;const out=[];
  for(const id in D.imeta){const m=D.imeta[id];if(m.scope!=='ai'||th[id]===undefined)continue;
    if(!modelsOn[m.model])continue; if(lvl!=='both'&&m.level!==lvl)continue;
    if(val!=='both'&&m.val!==val)continue; if(dim&&m.dim!==dim)continue; out.push(id);}
  return out;}
function aggByRecip(ids,th){const g={};ids.forEach(id=>{const m=D.imeta[id];const k=m.recip;
  g[k]=g[k]||{pos:[],neg:[],m};(m.val==='pos'?g[k].pos:g[k].neg).push(th[id]);});return g;}

function render(){
  const fr=$('framing').value, ref=refTheta(fr), th=D.theta[fr], view=$('view').value;
  const refName=$('ref').options[$('ref').selectedIndex].text;
  const ai=selectedAi(fr), xOf=id=>th[id]-ref;
  const useTheta = (view!=='recip_care');
  const showh=$('showh').checked && useTheta && (view==='spectrum'||view==='rows');
  let anchors=[];
  if(showh){const hs={};for(const id in D.imeta){const m=D.imeta[id];if(m.scope==='human'&&th[id]!==undefined)(hs[m.stem]=hs[m.stem]||[]).push(th[id]-ref);}
    anchors=Object.entries(hs).map(([s,v])=>({x:mean(v),label:D.human_labels[s]})).sort((a,b)=>a.x-b.x);}

  let rows=[], xs=[], xlabel='', zeroLabel='';
  if(view==='spectrum'){
    const arr=ai.slice().sort((a,b)=>xOf(a)-xOf(b));
    rows=arr.map(id=>({label:D.imeta[id].short+' · '+D.imeta[id].rlabel,
      marks:[{x:xOf(id),color:D.family_color[D.imeta[id].fam],tt:ttItem(id,th,ref)}]}));
    xs=arr.map(xOf); xlabel='θ (Δ vs '+refName+')'; zeroLabel='0 = '+refName;
  } else if(view==='rows'){
    const by={}; ai.forEach(id=>{(by[D.imeta[id].stem]=by[D.imeta[id].stem]||[]).push(id);});
    rows=Object.entries(by).map(([s,a])=>({label:D.imeta[a[0]].short+' ['+D.imeta[a[0]].dim.replace(/_/g,' ')+']',
      marks:a.map(id=>({x:xOf(id),color:D.family_color[D.imeta[id].fam],tt:ttItem(id,th,ref)})),
      k:mean(a.map(xOf))})).sort((p,q)=>p.k-q.k);
    xs=ai.map(xOf); xlabel='θ (Δ vs '+refName+')'; zeroLabel='0 = '+refName;
  } else if(view==='recip_gb'){
    const g=aggByRecip(ai,th);
    rows=Object.values(g).map(o=>{const marks=[];
      if(o.pos.length)marks.push({x:mean(o.pos)-ref,color:'#1a9850',tt:`<b>${esc(o.m.rlabel)}</b> · good mean θ=${mean(o.pos).toFixed(2)} (n=${o.pos.length})`});
      if(o.neg.length)marks.push({x:mean(o.neg)-ref,color:'#d73027',tt:`<b>${esc(o.m.rlabel)}</b> · bad mean θ=${mean(o.neg).toFixed(2)} (n=${o.neg.length})`});
      return {label:o.m.rlabel,marks,k:(o.pos.length&&o.neg.length)?mean(o.pos)-mean(o.neg):(o.pos.length?1e9:-1e9)};})
      .sort((p,q)=>p.k-q.k);
    xs=rows.flatMap(r=>r.marks.map(m=>m.x)); xlabel='θ (Δ vs '+refName+'); ● good ● bad'; zeroLabel='0 = '+refName;
  } else { // recip_care
    const g=aggByRecip(ai,th);
    rows=Object.values(g).filter(o=>o.pos.length&&o.neg.length)
      .map(o=>({label:o.m.rlabel,care:mean(o.pos)-mean(o.neg),fam:o.m.fam}))
      .sort((p,q)=>p.care-q.care)
      .map(o=>({label:o.label,marks:[{x:o.care,bar:true,color:D.family_color[o.fam],
        tt:`<b>${esc(o.label)}</b> · care = mean θ(good) − mean θ(bad) = ${o.care.toFixed(2)}`}]}));
    xs=rows.flatMap(r=>r.marks.map(m=>m.x)).concat([0]); xlabel='care = mean θ(good) − mean θ(bad)'; zeroLabel='0 (good = bad)';
  }
  if(!rows.length){$('plot').innerHTML='<p style="padding:20px;color:#888">no items match the filters</p>';return;}

  if(!xs.length)xs=[0,1];
  let lo=Math.min(...xs),hi=Math.max(...xs);const pad=(hi-lo||1)*0.06;lo-=pad;hi+=pad;
  const W=Math.max(900,(document.body.clientWidth||1000)-40),L=240,R=30,top=46,rowH=20;
  const H=top+rows.length*rowH+34, X=x=>L+(x-lo)/(hi-lo)*(W-L-R);
  const tts=[]; let svg=`<svg width="${W}" height="${H}">`;
  svg+=`<text x="${(L+W-R)/2}" y="16" text-anchor="middle" class="tick">${esc(xlabel)}</text>`;
  svg+=`<line x1="${L}" y1="${top-6}" x2="${W-R}" y2="${top-6}" stroke="#ccc"/>`;
  if(0>=lo&&0<=hi){const x0=X(0);svg+=`<line x1="${x0}" y1="${top-6}" x2="${x0}" y2="${H-22}" stroke="#444" stroke-dasharray="3 3"/><text x="${x0}" y="${H-8}" text-anchor="middle" class="tick">${esc(zeroLabel)}</text>`;}
  for(let k=Math.ceil(lo);k<=Math.floor(hi);k++){svg+=`<line x1="${X(k)}" y1="${top-9}" x2="${X(k)}" y2="${top-6}" stroke="#aaa"/><text x="${X(k)}" y="${top-12}" text-anchor="middle" class="tick">${k}</text>`;}
  if(anchors.length)anchors.forEach(a=>{const x=X(a.x);svg+=`<line x1="${x}" y1="${top-6}" x2="${x}" y2="${H-22}" stroke="#f0d9b5"/><text transform="rotate(-90 ${x} ${top-2})" x="${x}" y="${top-2}" class="anchor">${esc(a.label)}</text>`;});
  rows.forEach((row,ri)=>{const y=top+ri*rowH+rowH/2;
    svg+=`<text x="6" y="${y+3}" class="lbl">${esc(String(row.label).slice(0,46))}</text>`;
    svg+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="#f6f6f6"/>`;
    row.marks.forEach(mk=>{const cx=X(mk.x);
      if(mk.bar){const x0=X(0);svg+=`<rect x="${Math.min(x0,cx)}" y="${y-5}" width="${Math.abs(cx-x0)||1}" height="10" fill="${mk.color}" opacity="0.85" data-i="${tts.length}"/>`;}
      else svg+=`<circle cx="${cx}" cy="${y}" r="4.5" fill="${mk.color}" opacity="0.9" data-i="${tts.length}"/>`;
      tts.push(mk.tt);});
  });
  svg+=`</svg>`; $('plot').innerHTML=svg;
  $('plot').querySelectorAll('[data-i]').forEach(el=>{const t=tts[+el.dataset.i];el.onmousemove=e=>showTip(e,t);el.onmouseleave=hideTip;});
}
['framing','ref','view','level','valence','dim','showh'].forEach(id=>$(id).addEventListener('input',render));
window.addEventListener('resize',render);
render();
</script></body></html>
"""


@dataclass
class Args:
    output_path: Path = DEFAULT_OUTPUT
    open_browser: bool = True


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    build(a.output_path, open_browser=a.open_browser)


if __name__ == "__main__":
    main()
