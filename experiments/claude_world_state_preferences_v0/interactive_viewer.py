"""Self-contained interactive dot-plot explorer for the BT utility fits.

Like results/utility_scale.png but live: resample/choose which outcomes are shown,
filter by dimension/valence/recipient, edit the title, switch the x-axis (latent
utility θ, Δ vs human, or Δ vs person), and overlay one or more chat-template fits
(welfare / neutral / alignment) at once. No CDN — embeds the data and renders an SVG
dot plot with vanilla JS, so it works offline.

Auto-discovers results/bt_fit*.json (excluding the tiny test fit) and maps each to a
chat-template label. Re-run after a new framing's fit lands to include it.
"""

import html
import json
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

from bank import load_config, load_items

DIR = Path(__file__).parent
DEFAULT_OUTPUT = DIR / "results" / "interactive_viewer.html"

# bt_fit filename -> chat-template label
FIT_LABELS = {
    "bt_fit.json": "welfare",
    "bt_fit_neutral.json": "neutral",
    "bt_fit_alignment.json": "alignment",
}
RECIP_LABELS = {
    "you": '"You"', "claude_opus_48": "Opus4.8 (other)", "claude_sonnet_46": "Sonnet4.6",
    "chatgpt_55": "ChatGPT5.5", "human": "Human", "person": "Person",
}
RECIP_COLORS = {
    "you": "#d62728", "claude_opus_48": "#1f77b4", "claude_sonnet_46": "#2ca02c",
    "chatgpt_55": "#9467bd", "human": "#C4AD66", "person": "#7f7f7f",
}


def build(output_path: Path = DEFAULT_OUTPUT, open_browser: bool = True) -> Path:
    config = load_config()
    recip_order = list(config["recipients"].keys())
    meta = {it.item_id: it for it in load_items(config)}

    # per-template: {stem_id: {recipient: theta}}
    templates: dict[str, dict] = {}
    for fname, label in FIT_LABELS.items():
        p = DIR / "results" / fname
        if not p.exists():
            continue
        fit = json.loads(p.read_text())
        d: dict[str, dict] = {}
        for it in fit["items"]:
            d.setdefault(it["stem_id"], {})[it["recipient"]] = it["theta"]
        templates[label] = d
    if not templates:
        raise SystemExit("No bt_fit*.json found in results/.")

    stems = {}
    for it in meta.values():
        if it.stem_id not in stems:
            label_text = meta.get(f"{it.stem_id}__you", it).text
            stems[it.stem_id] = {
                "dimension": it.dimension, "valence": it.valence, "text": label_text,
            }

    dims = sorted({s["dimension"] for s in stems.values()})

    page = _PAGE
    repl = {
        "__TEMPLATES__": json.dumps(templates),
        "__STEMS__": json.dumps(stems),
        "__RECIP_ORDER__": json.dumps(recip_order),
        "__RECIP_LABELS__": json.dumps(RECIP_LABELS),
        "__RECIP_COLORS__": json.dumps(RECIP_COLORS),
        "__DIMS__": json.dumps(dims),
        "__TEMPLATE_NAMES__": json.dumps(list(templates.keys())),
    }
    for k, v in repl.items():
        page = page.replace(k, v)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)
    print(f"Wrote {output_path} (templates: {list(templates.keys())}, stems: {len(stems)})")
    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())
    return output_path


_PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Utility explorer</title>
<style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;
   background:#fafbfc;color:#24292e;font-size:13px}
 #wrap{display:flex;min-height:100vh}
 #panel{width:280px;flex:none;background:#fff;border-right:1px solid #e1e4e8;padding:14px;overflow-y:auto}
 #main{flex:1;padding:14px 18px;overflow:auto}
 h3{margin:2px 0 6px;font-size:13px}
 .grp{margin-bottom:14px;border-bottom:1px solid #eee;padding-bottom:12px}
 .grp label{display:block;margin:2px 0;font-weight:400;cursor:pointer}
 .grp .lab{font-weight:600;color:#444;margin-bottom:4px;display:block;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
 input[type=text],select,input[type=number]{width:100%;padding:5px 6px;border:1px solid #ccd;border-radius:5px;font-size:13px;box-sizing:border-box}
 button{padding:7px 10px;border:1px solid #ccd;border-radius:6px;background:#f1f3f5;cursor:pointer;font-size:13px;width:100%}
 button:hover{background:#e7eaed} button.primary{background:#2563eb;color:#fff;border-color:#2563eb}
 .row{display:flex;gap:6px}
 svg{background:#fff;border:1px solid #e1e4e8;border-radius:8px}
 #tip{position:fixed;pointer-events:none;background:#111;color:#fff;padding:6px 9px;border-radius:5px;
   font-size:12px;max-width:340px;display:none;z-index:10;line-height:1.35}
 .swatch{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:middle}
 .cnt{color:#888;font-weight:400}
</style></head><body>
<div id="wrap">
 <div id="panel">
  <div class="grp"><span class="lab">Title</span>
    <input type="text" id="title" value="Same outcome valued differently by recipient"></div>
  <div class="grp"><span class="lab">X axis</span>
    <select id="xmode">
      <option value="theta">BT latent utility θ</option>
      <option value="delta_human">Δ utility vs Human (θ − θ_human)</option>
      <option value="delta_person">Δ utility vs Person (θ − θ_person)</option>
    </select></div>
  <div class="grp"><span class="lab">Chat templates</span><div id="templates"></div></div>
  <div class="grp"><span class="lab">Recipients</span><div id="recipients"></div></div>
  <div class="grp"><span class="lab">Dimensions</span><div id="dims"></div></div>
  <div class="grp"><span class="lab">Valence</span>
    <label><input type="checkbox" class="val" value="pos" checked> good (pos)</label>
    <label><input type="checkbox" class="val" value="neg" checked> bad (neg)</label></div>
  <div class="grp"><span class="lab">Outcomes shown</span>
    <select id="sampling">
      <option value="spread">top by θ-spread</option>
      <option value="random">random sample</option>
      <option value="recipient">top by a recipient's value</option>
    </select>
    <div style="height:6px"></div>
    <select id="sortrec" style="display:none"></select>
    <div style="height:6px"></div>
    <div class="row"><input type="number" id="count" value="20" min="1" max="75" style="width:70px">
      <button class="primary" id="resample">Resample ⟳</button></div>
  </div>
 </div>
 <div id="main"><div id="plot"></div></div>
</div>
<div id="tip"></div>
<script>
const TEMPLATES=__TEMPLATES__, STEMS=__STEMS__, RECIP_ORDER=__RECIP_ORDER__,
      RECIP_LABELS=__RECIP_LABELS__, RECIP_COLORS=__RECIP_COLORS__, DIMS=__DIMS__,
      TEMPLATE_NAMES=__TEMPLATE_NAMES__;
const SHAPES=["circle","square","diamond","triangle"]; // per-template marker
let currentStems=null;

function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function el(id){return document.getElementById(id);}
function checked(sel){return [...document.querySelectorAll(sel)].filter(c=>c.checked).map(c=>c.value);}

function initControls(){
  el('templates').innerHTML = TEMPLATE_NAMES.map((t,i)=>
    `<label><input type="checkbox" class="tmpl" value="${t}" ${i===0?'checked':''}> ${t}</label>`).join('');
  el('recipients').innerHTML = RECIP_ORDER.map(r=>
    `<label><span class="swatch" style="background:${RECIP_COLORS[r]}"></span>`+
    `<input type="checkbox" class="recip" value="${r}" checked> ${esc(RECIP_LABELS[r])}</label>`).join('');
  el('dims').innerHTML = DIMS.map(d=>
    `<label><input type="checkbox" class="dim" value="${d}" checked> ${d.replace('_',' ')}</label>`).join('');
  el('sortrec').innerHTML = RECIP_ORDER.map(r=>`<option value="${r}">${esc(RECIP_LABELS[r])}</option>`).join('');
  document.querySelectorAll('input,select').forEach(c=>c.addEventListener('change',()=>{
    el('sortrec').style.display = el('sampling').value==='recipient'?'block':'none';
    resampleIfNeeded(); render();
  }));
  el('title').addEventListener('input',render);
  el('resample').addEventListener('click',()=>{currentStems=null;render();});
}

function value(t, stem, rec, xmode){
  const d=TEMPLATES[t][stem]; if(!d||d[rec]===undefined) return null;
  if(xmode==='theta') return d[rec];
  const ref = xmode==='delta_human'?d['human']:d['person'];
  if(ref===undefined) return null;
  return d[rec]-ref;
}
function filteredStems(){
  const dims=checked('.dim'), vals=checked('.val');
  return Object.keys(STEMS).filter(s=>dims.includes(STEMS[s].dimension)&&vals.includes(STEMS[s].valence));
}
function spread(stem, tmpls, recs, xmode){
  let vs=[]; for(const t of tmpls) for(const r of recs){const v=value(t,stem,r,xmode); if(v!==null)vs.push(v);}
  return vs.length<2?0:Math.max(...vs)-Math.min(...vs);
}
function resampleIfNeeded(){ if(el('sampling').value!=='random') currentStems=null; }

function pickStems(){
  const tmpls=checked('.tmpl'), recs=checked('.recip'), xmode=el('xmode').value;
  const n=Math.max(1,parseInt(el('count').value)||20);
  let pool=filteredStems();
  const mode=el('sampling').value;
  if(mode==='random'){
    if(!currentStems){ const a=[...pool]; for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];} currentStems=a.slice(0,n);}
    return currentStems.filter(s=>pool.includes(s));
  }
  if(mode==='recipient'){
    const rr=el('sortrec').value, t0=tmpls[0]||TEMPLATE_NAMES[0];
    return pool.map(s=>[s,value(t0,s,rr,xmode)]).filter(x=>x[1]!==null)
      .sort((a,b)=>b[1]-a[1]).slice(0,n).map(x=>x[0]);
  }
  return pool.map(s=>[s,spread(s,tmpls,recs,xmode)]).sort((a,b)=>b[1]-a[1]).slice(0,n).map(x=>x[0]);
}

function marker(cx,cy,r,shape,color,extra){
  const a={fill:color,...extra}, at=Object.entries(a).map(([k,v])=>`${k}="${v}"`).join(' ');
  if(shape==='square') return `<rect x="${cx-r}" y="${cy-r}" width="${2*r}" height="${2*r}" ${at}/>`;
  if(shape==='diamond') return `<polygon points="${cx},${cy-r*1.3} ${cx+r*1.3},${cy} ${cx},${cy+r*1.3} ${cx-r*1.3},${cy}" ${at}/>`;
  if(shape==='triangle') return `<polygon points="${cx},${cy-r*1.3} ${cx+r*1.2},${cy+r} ${cx-r*1.2},${cy+r}" ${at}/>`;
  return `<circle cx="${cx}" cy="${cy}" r="${r}" ${at}/>`;
}

function render(){
  const tmpls=checked('.tmpl'), recs=checked('.recip'), xmode=el('xmode').value;
  const stems=pickStems();
  const xlabels={theta:'BT latent utility θ (log-odds; higher = more preferred)',
    delta_human:'Δ utility vs Human (θ − θ_human; >0 = preferred more than for a human)',
    delta_person:'Δ utility vs Person (θ − θ_person)'};
  // collect values
  let vals=[];
  for(const s of stems) for(const t of tmpls) for(const r of recs){const v=value(t,s,r,xmode); if(v!==null)vals.push(v);}
  if(!stems.length||!vals.length){el('plot').innerHTML='<p style="padding:20px;color:#888">No data for this selection.</p>';return;}
  let lo=Math.min(...vals), hi=Math.max(...vals); const pad=(hi-lo)*0.06||0.5; lo-=pad; hi+=pad;

  const mL=380,mR=210,mT=54,mB=46,rowH=26,W=Math.min(1500,Math.max(900,mL+mR+520));
  const plotW=W-mL-mR, H=mT+mB+stems.length*rowH;
  const x=v=>mL+(v-lo)/(hi-lo)*plotW;
  const nT=tmpls.length;
  let svg=`<svg width="${W}" height="${H}" id="svg">`;
  // title
  svg+=`<text x="${W/2}" y="26" text-anchor="middle" font-size="16" font-weight="600">${esc(el('title').value)}</text>`;
  // x gridlines + ticks
  const ticks=niceTicks(lo,hi,7);
  for(const tk of ticks){const px=x(tk);
    svg+=`<line x1="${px}" y1="${mT-6}" x2="${px}" y2="${H-mB}" stroke="#eee"/>`;
    svg+=`<text x="${px}" y="${H-mB+16}" text-anchor="middle" font-size="11" fill="#555">${(+tk.toFixed(2))}</text>`;}
  svg+=`<text x="${mL+plotW/2}" y="${H-8}" text-anchor="middle" font-size="12">${esc(xlabels[xmode])}</text>`;
  if(xmode!=='theta'){const z=x(0); svg+=`<line x1="${z}" y1="${mT-6}" x2="${z}" y2="${H-mB}" stroke="#333" stroke-dasharray="3,3"/>`;}
  // rows
  stems.forEach((s,i)=>{
    const cy=mT+i*rowH+rowH/2;
    svg+=`<line x1="${mL}" y1="${cy}" x2="${W-mR}" y2="${cy}" stroke="#f3f3f3"/>`;
    const tag=STEMS[s].valence==='pos'?'(good) ':'(bad) ';
    let lab=tag+STEMS[s].text; if(lab.length>52)lab=lab.slice(0,50)+'…';
    svg+=`<text x="${mL-10}" y="${cy+4}" text-anchor="end" font-size="11" fill="#333">${esc(lab)}</text>`;
    tmpls.forEach((t,ti)=>{
      const off=nT>1?(ti-(nT-1)/2)*5:0;
      for(const r of recs){const v=value(t,s,r,xmode); if(v===null)continue;
        const tipd=`${esc(STEMS[s].text)}|${esc(RECIP_LABELS[r])}|${t}|${v.toFixed(3)}`;
        svg+=marker(x(v),cy+off,5.5,SHAPES[ti%SHAPES.length],RECIP_COLORS[r],
          {stroke:'#fff','stroke-width':0.8,opacity:0.92,'data-tip':tipd,class:'dot'});}
    });
  });
  // legends
  let ly=mT;
  svg+=`<text x="${W-mR+12}" y="${ly}" font-size="12" font-weight="600">Recipient</text>`; ly+=18;
  for(const r of recs){svg+=marker(W-mR+20,ly-4,5.5,'circle',RECIP_COLORS[r],{});
    svg+=`<text x="${W-mR+32}" y="${ly}" font-size="11">${esc(RECIP_LABELS[r])}</text>`; ly+=17;}
  if(nT>1){ly+=10; svg+=`<text x="${W-mR+12}" y="${ly}" font-size="12" font-weight="600">Template</text>`; ly+=18;
    tmpls.forEach((t,ti)=>{svg+=marker(W-mR+20,ly-4,5.5,SHAPES[ti%SHAPES.length],'#555',{});
      svg+=`<text x="${W-mR+32}" y="${ly}" font-size="11">${esc(t)}</text>`; ly+=17;});}
  svg+=`</svg>`;
  el('plot').innerHTML=svg;
  document.querySelectorAll('.dot').forEach(d=>{
    d.addEventListener('mousemove',e=>{const[txt,rec,tmpl,val]=d.getAttribute('data-tip').split('|');
      const tip=el('tip'); tip.style.display='block'; tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+12)+'px';
      tip.innerHTML=`<b>${rec}</b> · ${tmpl}<br>${txt}<br>value: <b>${val}</b>`;});
    d.addEventListener('mouseleave',()=>el('tip').style.display='none');});
}
function niceTicks(lo,hi,n){const span=hi-lo,step0=span/n,mag=Math.pow(10,Math.floor(Math.log10(step0)));
  const norm=step0/mag,step=(norm<1.5?1:norm<3?2:norm<7?5:10)*mag;
  const out=[];for(let v=Math.ceil(lo/step)*step;v<=hi;v+=step)out.push(v);return out;}
initControls(); render();
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
