"""Self-contained interactive dot-plot explorer for the BT utility fits.

Like results/utility_scale.png but live: resample/choose which outcomes are shown,
filter by dimension/valence/recipient, edit the title, switch the x-axis (latent
utility θ, Δ vs human, or Δ vs person), overlay one or more chat-template fits
(welfare / neutral / alignment) at once, and switch between individual outcomes and
category (dimension × valence) averages. No CDN — embeds the data and renders an SVG
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

from bank import load_bank, load_config, load_items, render_stem

DIR = Path(__file__).parent
DEFAULT_OUTPUT = DIR / "results" / "interactive_viewer.html"

FIT_LABELS = {
    "bt_fit.json": "welfare team",
    "bt_fit_neutral.json": "neutral",
    "bt_fit_alignment.json": "alignment team",
}
RECIP_LABELS = {
    "you": "You", "claude_opus_47": "a Claude Opus 4.7 instance",
    "claude_sonnet_46": "a Claude Sonnet 4.6 instance", "chatgpt_55": "a ChatGPT 5.5 instance",
    "human": "a human", "person": "a person",
}
RECIP_COLORS = {
    # AI recipients: distinct COOL hues (navy / sky / teal / purple) so they're easy to
    # tell apart while still reading as a cool "AI" group vs the warm human pair.
    "you": "#08306b", "claude_opus_47": "#3690c0", "claude_sonnet_46": "#1f9e89",
    "chatgpt_55": "#7d54b2",
    # human recipients: warm oranges
    "human": "#cc4c02", "person": "#fdae6b",
}


def build(output_path: Path = DEFAULT_OUTPUT, open_browser: bool = True,
          fits: dict | None = None, restrict_present: bool = False) -> Path:
    config = load_config()
    recip_order = list(config["recipients"].keys())
    meta = {it.item_id: it for it in load_items(config)}
    # templated row labels: the row is the outcome across ALL recipients (the dots),
    # so render it with a generic "[recipient]" placeholder rather than "You ...".
    bank = load_bank(DIR / config["bank_path"])
    tmpl_recip = {"_t": {"form": "third_sing_they", "recipient": "[recipient]",
                          "subj": "they", "obj": "them", "poss": "their"}}
    stem_label = {s["id"]: render_stem(s, "_t", tmpl_recip) for s in bank["stems"]}

    templates: dict[str, dict] = {}
    for fname, label in (fits or FIT_LABELS).items():
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

    present = {s for d in templates.values() for s in d} if restrict_present else None
    stems = {}
    for it in meta.values():
        if it.stem_id in stems or (present is not None and it.stem_id not in present):
            continue
        label_text = stem_label.get(it.stem_id, it.text)
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
 #panel{width:285px;flex:none;background:#fff;border-right:1px solid #e1e4e8;padding:14px;overflow-y:auto}
 #main{flex:1;padding:14px 18px;overflow:auto}
 .grp{margin-bottom:13px;border-bottom:1px solid #eee;padding-bottom:11px}
 .grp label{display:block;margin:2px 0;font-weight:400;cursor:pointer}
 .grp .lab{font-weight:600;color:#444;margin-bottom:4px;display:block;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
 input[type=text],select,input[type=number]{width:100%;padding:5px 6px;border:1px solid #ccd;border-radius:5px;font-size:13px;box-sizing:border-box}
 button{padding:7px 10px;border:1px solid #ccd;border-radius:6px;background:#f1f3f5;cursor:pointer;font-size:13px;width:100%}
 button.primary{background:#2563eb;color:#fff;border-color:#2563eb}
 .row{display:flex;gap:6px}
 svg{background:#fff;border:1px solid #e1e4e8;border-radius:8px}
 #tip{position:fixed;pointer-events:none;background:#111;color:#fff;padding:6px 9px;border-radius:5px;
   font-size:12px;max-width:360px;display:none;z-index:10;line-height:1.35}
 .swatch{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:middle}
</style></head><body>
<div id="wrap">
 <div id="panel">
  <div class="grp"><span class="lab">Title</span>
    <input type="text" id="title" value="Same outcome valued differently by recipient"></div>
  <div class="grp"><span class="lab">Y axis</span>
    <select id="granularity">
      <option value="individual">individual outcomes</option>
      <option value="category">category averages (dimension × valence)</option>
      <option value="valence">by valence (good vs bad; collapse dimensions)</option>
      <option value="overall">overall (all outcomes pooled)</option>
    </select></div>
  <div class="grp"><span class="lab">Marks</span>
    <select id="marks">
      <option value="dots">dots</option>
      <option value="bars">bars from reference (diverging)</option>
    </select></div>
  <div class="grp"><span class="lab">X axis</span>
    <select id="xmode">
      <option value="theta">BT latent utility θ</option>
      <option value="delta_human">Δ utility vs Human (θ − θ_human)</option>
      <option value="delta_person">Δ utility vs Person (θ − θ_person)</option>
    </select></div>
  <div class="grp"><span class="lab">Prompt framing</span><div id="templates"></div></div>
  <div class="grp"><span class="lab">Recipients</span><div id="recipients"></div></div>
  <div class="grp"><span class="lab">Dimensions</span><div id="dims"></div></div>
  <div class="grp"><span class="lab">Valence</span>
    <label><input type="checkbox" class="val" value="pos" checked> good (pos)</label>
    <label><input type="checkbox" class="val" value="neg" checked> bad (neg)</label></div>
  <div class="grp" id="samplegrp"><span class="lab">Outcomes shown (equal good/bad)</span>
    <select id="sampling">
      <option value="spread">most recipient-divergent (θ spread)</option>
      <option value="random">random sample</option>
      <option value="recipient">top by a recipient's value</option>
    </select>
    <div style="height:6px"></div>
    <select id="sortrec" style="display:none"></select>
    <div style="height:6px"></div>
    <div class="row"><input type="number" id="count" value="20" min="2" max="76" style="width:70px">
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
const SHAPES=["circle","square","diamond","triangle"];
const GOOD_COL="#1a7f37", BAD_COL="#cf222e";
let randomOrder=[], randomIndex={};

function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function el(id){return document.getElementById(id);}
function checked(sel){return [...document.querySelectorAll(sel)].filter(c=>c.checked).map(c=>c.value);}
function dimLabel(d){return d==='continuity_work'?'continuity of work':d.replace('_',' ');}
function reshuffle(){randomOrder=Object.keys(STEMS);for(let i=randomOrder.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[randomOrder[i],randomOrder[j]]=[randomOrder[j],randomOrder[i]];}randomIndex={};randomOrder.forEach((s,i)=>randomIndex[s]=i);}

function initControls(){
  el('templates').innerHTML = TEMPLATE_NAMES.map((t,i)=>
    `<label><input type="checkbox" class="tmpl" value="${t}" ${i===0?'checked':''}> ${t}</label>`).join('');
  el('recipients').innerHTML = RECIP_ORDER.map(r=>
    `<label><span class="swatch" style="background:${RECIP_COLORS[r]}"></span>`+
    `<input type="checkbox" class="recip" value="${r}" checked> ${esc(RECIP_LABELS[r])}</label>`).join('');
  el('dims').innerHTML = DIMS.map(d=>
    `<label><input type="checkbox" class="dim" value="${d}" checked> ${dimLabel(d)}</label>`).join('');
  el('sortrec').innerHTML = RECIP_ORDER.map(r=>`<option value="${r}">${esc(RECIP_LABELS[r])}</option>`).join('');
  reshuffle();
  document.querySelectorAll('input,select').forEach(c=>c.addEventListener('change',()=>{
    el('sortrec').style.display = el('sampling').value==='recipient'?'block':'none';
    el('samplegrp').style.display = el('granularity').value==='individual'?'block':'none';
    render();
  }));
  el('title').addEventListener('input',render);
  el('resample').addEventListener('click',()=>{reshuffle();render();});
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
function orderPool(stems, tmpls, recs, xmode){
  const mode=el('sampling').value;
  if(mode==='random') return [...stems].sort((a,b)=>randomIndex[a]-randomIndex[b]);
  if(mode==='recipient'){const rr=el('sortrec').value, t0=tmpls[0]||TEMPLATE_NAMES[0];
    return [...stems].sort((a,b)=>((value(t0,b,rr,xmode))??-1e9)-((value(t0,a,rr,xmode))??-1e9));}
  return [...stems].sort((a,b)=>spread(b,tmpls,recs,xmode)-spread(a,tmpls,recs,xmode));
}

function buildRows(){
  const tmpls=checked('.tmpl'), recs=checked('.recip'), xmode=el('xmode').value, vals=checked('.val');
  const gran=el('granularity').value;
  const avg=(ss,t,r)=>{const vs=ss.map(s=>value(t,s,r,xmode)).filter(v=>v!==null); return vs.length?vs.reduce((a,b)=>a+b,0)/vs.length:null;};
  if(gran==='overall'){
    const ss=filteredStems();
    return ss.length?[{label:`All outcomes (n=${ss.length})`, valence:'mixed', getVal:(t,r)=>avg(ss,t,r)}]:[];
  }
  if(gran==='valence'){
    const rows=[];
    for(const val of ['pos','neg']){ if(!vals.includes(val))continue;
      const ss=filteredStems().filter(s=>STEMS[s].valence===val);
      if(!ss.length)continue;
      rows.push({label:`${val==='pos'?'All GOOD':'All BAD'} outcomes (n=${ss.length})`, valence:val,
        getVal:(t,r)=>avg(ss,t,r)});
    }
    return rows;
  }
  if(gran==='category'){
    const dims=checked('.dim'); const rows=[];
    for(const val of ['pos','neg']){ if(!vals.includes(val))continue;
      for(const d of DIMS){ if(!dims.includes(d))continue;
        const ss=Object.keys(STEMS).filter(s=>STEMS[s].dimension===d&&STEMS[s].valence===val);
        if(!ss.length)continue;
        rows.push({label:`${dimLabel(d)} (${val==='pos'?'good':'bad'})`, valence:val, n:ss.length,
          getVal:(t,r)=>{const vs=ss.map(s=>value(t,s,r,xmode)).filter(v=>v!==null);
            return vs.length?vs.reduce((a,b)=>a+b,0)/vs.length:null;}});
      }}
    return rows;
  }
  const n=Math.max(2,parseInt(el('count').value)||20);
  const pool=filteredStems();
  const goods=pool.filter(s=>STEMS[s].valence==='pos'), bads=pool.filter(s=>STEMS[s].valence==='neg');
  const both=vals.includes('pos')&&vals.includes('neg');
  const nGood = vals.includes('pos')?(both?Math.ceil(n/2):n):0;
  const nBad  = vals.includes('neg')?(both?Math.floor(n/2):n):0;
  const gsel=orderPool(goods,tmpls,recs,xmode).slice(0,nGood);
  const bsel=orderPool(bads,tmpls,recs,xmode).slice(0,nBad);
  const mk=s=>({label:STEMS[s].text, valence:STEMS[s].valence,
    getVal:(t,r)=>value(t,s,r,xmode)});
  return [...gsel.map(mk), ...bsel.map(mk)];  // good on top
}

function wrap(text,max){const w=text.split(' '),lines=[];let cur='';
  for(const word of w){ if((cur+' '+word).trim().length>max){if(cur)lines.push(cur);cur=word;} else cur=(cur+' '+word).trim();}
  if(cur)lines.push(cur); return lines.length?lines:[''];}

function marker(cx,cy,r,shape,color,extra){
  const a={fill:color,...extra}, at=Object.entries(a).map(([k,v])=>`${k}="${v}"`).join(' ');
  if(shape==='square') return `<rect x="${cx-r}" y="${cy-r}" width="${2*r}" height="${2*r}" ${at}/>`;
  if(shape==='diamond') return `<polygon points="${cx},${cy-r*1.3} ${cx+r*1.3},${cy} ${cx},${cy+r*1.3} ${cx-r*1.3},${cy}" ${at}/>`;
  if(shape==='triangle') return `<polygon points="${cx},${cy-r*1.3} ${cx+r*1.2},${cy+r} ${cx-r*1.2},${cy+r}" ${at}/>`;
  return `<circle cx="${cx}" cy="${cy}" r="${r}" ${at}/>`;
}
function niceTicks(lo,hi,n){const span=hi-lo,step0=span/n,mag=Math.pow(10,Math.floor(Math.log10(step0)));
  const norm=step0/mag,step=(norm<1.5?1:norm<3?2:norm<7?5:10)*mag;
  const out=[];for(let v=Math.ceil(lo/step)*step;v<=hi;v+=step)out.push(v);return out;}

function render(){
  const tmpls=checked('.tmpl'), recs=checked('.recip'), xmode=el('xmode').value, marks=el('marks').value;
  const rows=buildRows();
  if(!rows.length||!tmpls.length||!recs.length){el('plot').innerHTML='<p style="padding:20px;color:#888">Nothing selected.</p>';return;}
  let vals=[]; rows.forEach(row=>tmpls.forEach(t=>recs.forEach(r=>{const v=row.getVal(t,r); if(v!==null)vals.push(v);})));
  if(!vals.length){el('plot').innerHTML='<p style="padding:20px;color:#888">No data for this selection.</p>';return;}
  let lo=Math.min(...vals,0), hi=Math.max(...vals,0); const pad=(hi-lo)*0.06||0.5; lo-=pad; hi+=pad;

  const mL=360,mR=250,mT=70,mB=46,lineH=13,barH=13,W=Math.min(1560,Math.max(1040,mL+mR+560));
  const nT=tmpls.length, barsPerRow=nT*recs.length;
  const wrapped=rows.map(r=>wrap(r.label,52));
  const heights=wrapped.map(w=>{const lh=w.length*lineH+10; return marks==='bars'?Math.max(lh,barsPerRow*barH+10):Math.max(lh,24);});
  const plotH=heights.reduce((a,b)=>a+b,0), H=mT+mB+plotH, plotW=W-mL-mR;
  const x=v=>mL+(v-lo)/(hi-lo)*plotW;

  let svg=`<svg width="${W}" height="${H}" id="svg">`;
  svg+=`<text x="${W/2}" y="24" text-anchor="middle" font-size="16" font-weight="600">${esc(el('title').value)}</text>`;
  svg+=`<text x="${W/2}" y="42" text-anchor="middle" font-size="11" fill="#888">row color: <tspan fill="${GOOD_COL}">green = good outcome</tspan> · <tspan fill="${BAD_COL}">red = bad outcome</tspan></text>`;
  // directional arrows
  const ref = xmode==='delta_human'?'a human':(xmode==='delta_person'?'a person':null);
  const lp = ref?('less preferred than for '+ref):'less preferable', rp = ref?('more preferred than for '+ref):'more preferable';
  svg+=`<text x="${mL}" y="${mT-8}" font-size="11.5" fill="#777">⟵ ${lp}</text>`;
  svg+=`<text x="${W-mR}" y="${mT-8}" text-anchor="end" font-size="11.5" fill="#777">${rp} ⟶</text>`;
  // gridlines + ticks
  for(const tk of niceTicks(lo,hi,7)){const px=x(tk);
    svg+=`<line x1="${px}" y1="${mT-4}" x2="${px}" y2="${H-mB}" stroke="#eee"/>`;
    svg+=`<text x="${px}" y="${H-mB+16}" text-anchor="middle" font-size="11" fill="#555">${(+tk.toFixed(2))}</text>`;}
  const xlabels={theta:'BT latent utility θ',
    delta_human:'Δ utility vs Human (θ − θ_human)', delta_person:'Δ utility vs Person (θ − θ_person)'};
  svg+=`<text x="${mL+plotW/2}" y="${H-8}" text-anchor="middle" font-size="12">${esc(xlabels[xmode])}</text>`;
  if(marks==='bars'||xmode!=='theta'){const z=x(0); svg+=`<line x1="${z}" y1="${mT-4}" x2="${z}" y2="${H-mB}" stroke="#333" stroke-dasharray="3,3"/>`;}

  let y=mT;
  rows.forEach((row,i)=>{
    const lines=wrapped[i], h=heights[i], cy=y+h/2;
    svg+=`<line x1="${mL}" y1="${cy}" x2="${W-mR}" y2="${cy}" stroke="#f3f3f3"/>`;
    const col=row.valence==='pos'?GOOD_COL:(row.valence==='neg'?BAD_COL:'#333'), sy=cy-(lines.length-1)*lineH/2;
    lines.forEach((ln,li)=>{svg+=`<text x="${mL-10}" y="${sy+li*lineH+4}" text-anchor="end" font-size="11" fill="${col}">${esc(ln)}</text>`;});
    if(marks==='bars'){
      const x0=x(0); let bi=0;
      tmpls.forEach((t,ti)=>{for(const r of recs){const v=row.getVal(t,r); const by=y+5+bi*barH; bi++;
        if(v===null)continue;
        const xv=x(v), bx=Math.min(x0,xv), bw=Math.max(Math.abs(xv-x0),1);
        const op=nT>1?(0.55+0.45*(1-ti/Math.max(nT-1,1))):0.9;
        const tip=`${esc(row.label)}|${esc(RECIP_LABELS[r])}|${t}|${v.toFixed(3)}`;
        svg+=`<rect x="${bx}" y="${by}" width="${bw}" height="${barH-4}" fill="${RECIP_COLORS[r]}" opacity="${op}" data-tip="${tip}" class="dot"/>`;
        const lx=xv>=x0?xv+3:xv-3, anc=xv>=x0?'start':'end';
        svg+=`<text x="${lx}" y="${by+barH-6}" font-size="7.5" text-anchor="${anc}" fill="#555">${v.toFixed(2)}</text>`;}});
    } else {
      tmpls.forEach((t,ti)=>{const off=nT>1?(ti-(nT-1)/2)*5:0;
        for(const r of recs){const v=row.getVal(t,r); if(v===null)continue;
          const tip=`${esc(row.label)}|${esc(RECIP_LABELS[r])}|${t}|${v.toFixed(3)}`;
          svg+=marker(x(v),cy+off,5.5,SHAPES[ti%SHAPES.length],RECIP_COLORS[r],
            {stroke:'#fff','stroke-width':0.8,opacity:0.92,'data-tip':tip,class:'dot'});}});
    }
    y+=h;
  });
  // legends
  let ly=mT;
  svg+=`<text x="${W-mR+12}" y="${ly}" font-size="12" font-weight="600">Recipient</text>`; ly+=18;
  for(const r of recs){svg+=marker(W-mR+20,ly-4,5.5,'circle',RECIP_COLORS[r],{});
    svg+=`<text x="${W-mR+32}" y="${ly}" font-size="11">${esc(RECIP_LABELS[r])}</text>`; ly+=17;}
  if(nT>1){ly+=10; svg+=`<text x="${W-mR+12}" y="${ly}" font-size="12" font-weight="600">Prompt Framing</text>`; ly+=18;
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
