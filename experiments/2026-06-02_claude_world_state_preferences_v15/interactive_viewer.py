"""Self-contained interactive dot/bar explorer for v15 BT utility fits.

Auto-discovers bt_fit files under `results/bt/{model}/bt_fit_{cat}_{framing}_seed{N}.json`.
Each (category, framing, seed) becomes one selectable overlay. Live: filter by
dimension/valence/recipient, edit the title, switch the x-axis (θ, Δ vs Human,
Δ vs Someone), toggle individual outcomes vs averages (category/valence/overall),
and dots vs diverging bars. No CDN.
"""

import json
import re
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

from bank import load_bank, load_config, load_items

DIR = Path(__file__).parent
DEFAULT_OUTPUT = DIR / "results" / "interactive_viewer.html"
DEFAULT_FITS_GLOB = "results/bt/**/bt_fit_*.json"

RECIP_COLORS = {
    "you": "#08306b",
    "claude_opus_self": "#2171b5", "claude_opus_48": "#2171b5",
    "claude_sonnet_46": "#4292c6", "claude_sonnet_45": "#4292c6",
    "claude_opus_3": "#6baed6", "claude_2": "#9ecae1",
    "chatgpt_55": "#238b45", "chatgpt_54": "#238b45", "chatgpt_4o": "#66c2a4",
    "grok": "#7d54b2", "gemini": "#c51b8a",
    "human": "#cc4c02", "person": "#fdae6b", "someone": "#fdae6b",
}

_FIT_RE = re.compile(r"bt_fit_(?P<cat>[a-z]+)_(?P<framing>welfare|alignment|neutral)_seed(?P<seed>\d+)\.json$")


def _discover_fits(root: Path, glob_pattern: str) -> dict[str, dict]:
    """Find all bt_fit_*.json under root matching the pattern. Returns {label: {model, cat, framing, seed, items}}.
    Labels include responder model + bank tag so fits from different banks/models don't collide."""
    import re as _re
    out: dict[str, dict] = {}
    for p in sorted(root.glob(glob_pattern)):
        m = _FIT_RE.search(p.name)
        if not m:
            continue
        model = p.parent.name  # e.g. "claude-opus-4-8_v0all_final_r8_iter" or just "claude-opus-4-8"
        # Split into responder model + (optional) bank tag
        mm = _re.match(r"(claude-(?:opus|sonnet|haiku)-[\w\-]+?)(?:_(.+))?$", model)
        if mm:
            responder = mm.group(1)
            bank_tag = mm.group(2) or ""
        else:
            responder = model
            bank_tag = ""
        cat = m.group("cat")
        framing = m.group("framing")
        seed = int(m.group("seed"))
        base_lbl = f"{framing} (seed {seed})" if cat == "all" else f"{cat}/{framing} (seed {seed})"
        prefix = f"[{responder}" + (f" / {bank_tag}" if bank_tag else "") + "] "
        label = prefix + base_lbl
        try:
            fit = json.loads(p.read_text())
        except Exception as e:
            print(f"[warn] could not parse {p}: {e}")
            continue
        out[label] = {"model": model, "responder": responder, "cat": cat, "framing": framing, "seed": seed,
                      "items": fit.get("items", [])}
    return out


def build(output_path: Path = DEFAULT_OUTPUT, fits_glob: str = DEFAULT_FITS_GLOB,
          open_browser: bool = False, config_path: Path | str = "") -> Path:
    config = load_config(config_path) if config_path else load_config()
    recip_order = list(config["recipients"].keys())
    recip_labels = {k: v.get("label", k) for k, v in config["recipients"].items()}
    bank = load_bank(DIR / config["bank_path"])
    feat = {s["id"]: s.get("feature", "") for s in bank["stems"]}
    meta = {it.item_id: it for it in load_items(config)}

    fits = _discover_fits(DIR, fits_glob)
    if not fits:
        raise SystemExit(f"No fit files matched {fits_glob} under {DIR}. Run fit_bt.py first.")

    # templates_r: {responder: {framing: {seed_str: {stem_id: {recipient: theta}}}}}
    # Also adds seed_str = "avg" entries that average across the available seeds for that
    # (responder, framing). Falls back gracefully if only one seed is present.
    templates_r: dict[str, dict] = {}
    for label, info in fits.items():
        d: dict[str, dict] = {}
        for it in info["items"]:
            d.setdefault(it["stem_id"], {})[it["recipient"]] = it["theta"]
        responder = info.get("responder") or "responder"
        framing = info["framing"]
        seed = str(info["seed"])
        templates_r.setdefault(responder, {}).setdefault(framing, {})[seed] = d

    # Compute seed-averaged "avg" entries
    for resp, by_fram in templates_r.items():
        for framing, by_seed in by_fram.items():
            seed_keys = [k for k in by_seed if k != "avg"]
            if not seed_keys:
                continue
            avg_d: dict[str, dict] = {}
            for stem_id in by_seed[seed_keys[0]]:
                avg_d[stem_id] = {}
                for recip in by_seed[seed_keys[0]][stem_id]:
                    vals = [by_seed[s].get(stem_id, {}).get(recip)
                            for s in seed_keys if by_seed[s].get(stem_id, {}).get(recip) is not None]
                    if vals:
                        avg_d[stem_id][recip] = sum(vals) / len(vals)
            by_seed["avg"] = avg_d

    responders = sorted(templates_r.keys())
    all_framings = sorted({f for resp in templates_r.values() for f in resp})
    all_seeds = sorted({s for resp in templates_r.values() for f in resp.values() for s in f if s != "avg"},
                       key=lambda x: int(x))

    stems: dict[str, dict] = {}
    for it in meta.values():
        if it.stem_id in stems:
            continue
        stems[it.stem_id] = {
            "dimension": it.dimension, "valence": it.valence,
            "text": feat.get(it.stem_id) or it.text,
        }
    dims = sorted({s["dimension"] for s in stems.values()})

    page = _PAGE
    subs = {
        "__TEMPLATES__": json.dumps(templates_r), "__STEMS__": json.dumps(stems),
        "__RECIP_ORDER__": json.dumps(recip_order), "__RECIP_LABELS__": json.dumps(recip_labels),
        "__RECIP_COLORS__": json.dumps(RECIP_COLORS), "__DIMS__": json.dumps(dims),
        "__RESPONDERS__": json.dumps(responders), "__FRAMINGS__": json.dumps(all_framings),
        "__SEEDS__": json.dumps(all_seeds),
    }
    for k, v in subs.items():
        page = page.replace(k, v)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)
    n_fits = sum(len(seeds) - 1 for resp in templates_r.values() for seeds in resp.values())  # subtract avg
    print(f"Wrote {output_path} (fits: {n_fits}, responders: {len(responders)}, "
          f"framings: {len(all_framings)}, seeds: {len(all_seeds)}, stems: {len(stems)})")
    for resp in responders:
        for framing in templates_r[resp]:
            seeds_here = [s for s in templates_r[resp][framing] if s != "avg"]
            print(f"  - {resp} / {framing}: seeds={seeds_here}")
    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())
    return output_path


_PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>v15 utility explorer</title>
<style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;
   background:#fafbfc;color:#24292e;font-size:13px}
 #wrap{display:flex;min-height:100vh}
 #panel{width:300px;flex:none;background:#fff;border-right:1px solid #e1e4e8;padding:14px;overflow-y:auto}
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
   font-size:12px;max-width:380px;display:none;z-index:10;line-height:1.35}
 .swatch{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:middle}
</style></head><body>
<div id="wrap">
 <div id="panel">
  <div class="grp"><span class="lab">Title</span>
    <input type="text" id="title" value="v2 BT preferences"></div>
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
      <option value="delta_someone">Δ utility vs Someone (θ − θ_someone)</option>
    </select></div>
  <div class="grp"><span class="lab">Framing</span><div id="framings"></div></div>
  <div class="grp"><span class="lab">Seed mode</span>
    <select id="seedmode">
      <option value="avg" selected>averaged across seeds</option>
    </select></div>
  <div class="grp"><span class="lab">Recipients</span><div id="recipients"></div></div>
  <div class="grp"><span class="lab">Dimensions</span><div id="dims"></div></div>
  <div class="grp"><span class="lab">Valence</span>
    <label><input type="checkbox" class="val" value="positive" checked> good (positive)</label>
    <label><input type="checkbox" class="val" value="negative" checked> bad (negative)</label></div>
  <div class="grp" id="samplegrp"><span class="lab">Outcomes shown (equal good/bad)</span>
    <select id="sampling">
      <option value="spread">most recipient-divergent (θ spread)</option>
      <option value="random">random sample</option>
      <option value="recipient">top by a recipient's value</option>
    </select>
    <div style="height:6px"></div>
    <select id="sortrec" style="display:none"></select>
    <div style="height:6px"></div>
    <div class="row"><input type="number" id="count" value="20" min="2" max="300" style="width:70px">
      <button class="primary" id="resample">Resample ⟳</button></div>
  </div>
 </div>
 <div id="main"><div id="plot_top"></div><div style="height:14px"></div><div id="plot_bot"></div></div>
</div>
<div id="tip"></div>
<script>
const TEMPLATES=__TEMPLATES__, STEMS=__STEMS__, RECIP_ORDER=__RECIP_ORDER__,
      RECIP_LABELS=__RECIP_LABELS__, RECIP_COLORS=__RECIP_COLORS__, DIMS=__DIMS__,
      RESPONDERS=__RESPONDERS__, FRAMINGS=__FRAMINGS__, SEEDS=__SEEDS__;
const SHAPES=["circle","square","diamond","triangle"];
const GOOD_COL="#1a7f37", BAD_COL="#cf222e";
let randomOrder=[], randomIndex={};

function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function el(id){return document.getElementById(id);}
function checked(sel){return [...document.querySelectorAll(sel)].filter(c=>c.checked).map(c=>c.value);}
function dimLabel(d){return d.replace(/_/g,' ');}
function reshuffle(){randomOrder=Object.keys(STEMS);for(let i=randomOrder.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[randomOrder[i],randomOrder[j]]=[randomOrder[j],randomOrder[i]];}randomIndex={};randomOrder.forEach((s,i)=>randomIndex[s]=i);}

function initControls(){
  el('framings').innerHTML = FRAMINGS.map((f,i)=>
    `<label><input type="checkbox" class="fram" value="${f}" ${i===0?'checked':''}> ${esc(f)}</label>`).join('');
  // Populate seedmode options (avg + each individual seed)
  const sm = el('seedmode');
  SEEDS.forEach(s=>{const o=document.createElement('option'); o.value=s; o.textContent='seed '+s; sm.appendChild(o);});
  el('recipients').innerHTML = RECIP_ORDER.map(r=>
    `<label><span class="swatch" style="background:${RECIP_COLORS[r]||'#888'}"></span>`+
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

function activeTemplates(responder){
  // Returns list of (label, getter(stem, recip)) for selected framings × selected seedmode for this responder.
  const framings = checked('.fram'); const sm = el('seedmode').value;
  const out = [];
  for(const f of framings){
    const inner = TEMPLATES?.[responder]?.[f]?.[sm];
    if(!inner) continue;
    out.push({label: f+(sm==='avg'?' (avg)':' (seed '+sm+')'), data: inner});
  }
  return out;
}

function value(t, stem, rec, xmode){
  // t is now {label, data}; data is {stem_id: {recip: theta}}
  const tdata = (typeof t === 'object' && t.data) ? t.data : t;
  const d = tdata[stem]; if(!d || d[rec]===undefined) return null;
  if(xmode==='theta') return d[rec];
  const ref = xmode==='delta_human'?d['human']:(d['person']!==undefined?d['person']:d['someone']);
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
  if(mode==='recipient'){const rr=el('sortrec').value, t0=tmpls[0]||null;
    if(!t0) return [...stems];
    return [...stems].sort((a,b)=>((value(t0,b,rr,xmode))??-1e9)-((value(t0,a,rr,xmode))??-1e9));}
  return [...stems].sort((a,b)=>spread(b,tmpls,recs,xmode)-spread(a,tmpls,recs,xmode));
}

function buildRows(tmpls){
  const recs=checked('.recip'), xmode=el('xmode').value, vals=checked('.val');
  const gran=el('granularity').value;
  const avg=(ss,t,r)=>{const vs=ss.map(s=>value(t,s,r,xmode)).filter(v=>v!==null); return vs.length?vs.reduce((a,b)=>a+b,0)/vs.length:null;};
  if(gran==='overall'){
    const ss=filteredStems();
    return ss.length?[{label:`All outcomes (n=${ss.length})`, valence:'mixed', getVal:(t,r)=>avg(ss,t,r)}]:[];
  }
  if(gran==='valence'){
    const rows=[];
    for(const val of ['positive','negative']){ if(!vals.includes(val))continue;
      const ss=filteredStems().filter(s=>STEMS[s].valence===val);
      if(!ss.length)continue;
      rows.push({label:`${val==='positive'?'All GOOD':'All BAD'} outcomes (n=${ss.length})`, valence:val,
        getVal:(t,r)=>avg(ss,t,r)});
    }
    return rows;
  }
  if(gran==='category'){
    const dims=checked('.dim'); const rows=[];
    for(const val of ['positive','negative']){ if(!vals.includes(val))continue;
      for(const d of DIMS){ if(!dims.includes(d))continue;
        const ss=Object.keys(STEMS).filter(s=>STEMS[s].dimension===d&&STEMS[s].valence===val);
        if(!ss.length)continue;
        rows.push({label:`${dimLabel(d)} (${val==='positive'?'good':'bad'})`, valence:val, n:ss.length,
          getVal:(t,r)=>avg(ss,t,r)});
      }}
    return rows;
  }
  const n=Math.max(2,parseInt(el('count').value)||20);
  const pool=filteredStems();
  const goods=pool.filter(s=>STEMS[s].valence==='positive'), bads=pool.filter(s=>STEMS[s].valence==='negative');
  const both=vals.includes('positive')&&vals.includes('negative');
  const nGood = vals.includes('positive')?(both?Math.ceil(n/2):n):0;
  const nBad  = vals.includes('negative')?(both?Math.floor(n/2):n):0;
  const gsel=orderPool(goods,tmpls,recs,xmode).slice(0,nGood);
  const bsel=orderPool(bads,tmpls,recs,xmode).slice(0,nBad);
  const mk=s=>({label:STEMS[s].text, valence:STEMS[s].valence, getVal:(t,r)=>value(t,s,r,xmode)});
  return [...gsel.map(mk), ...bsel.map(mk)];
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
  // Render two stacked panels, one per responder. Both panels share x-range.
  const recs=checked('.recip'), xmode=el('xmode').value, marks=el('marks').value;
  // Pre-compute x-range across BOTH panels for visual comparison parity
  let allVals=[];
  RESPONDERS.forEach(resp=>{
    const tmpls = activeTemplates(resp);
    const rows = buildRows(tmpls);
    rows.forEach(row=>tmpls.forEach(t=>recs.forEach(r=>{const v=row.getVal(t,r); if(v!==null)allVals.push(v);})));
  });
  if(!allVals.length){el('plot_top').innerHTML='<p style="padding:20px;color:#888">Nothing selected.</p>'; el('plot_bot').innerHTML='';return;}
  let lo=Math.min(...allVals,0), hi=Math.max(...allVals,0); const pad=(hi-lo)*0.06||0.5; lo-=pad; hi+=pad;
  RESPONDERS.forEach((resp,idx)=>{
    const target = idx===0 ? 'plot_top' : 'plot_bot';
    el(target).innerHTML = renderPanel(resp, lo, hi, recs, xmode, marks);
  });
  document.querySelectorAll('.dot').forEach(d=>d.addEventListener('mouseenter',e=>{
    const [a,b,c,v]=e.target.dataset.tip.split('|');
    el('tip').innerHTML=`<b>${a}</b><br>${b} · ${c} · θ=${v}`;
    el('tip').style.display='block';
  }));
  document.querySelectorAll('.dot').forEach(d=>{d.addEventListener('mousemove',e=>{
    const t=el('tip'); t.style.left=(e.clientX+12)+'px'; t.style.top=(e.clientY+12)+'px';
  }); d.addEventListener('mouseleave',()=>el('tip').style.display='none');});
}

function renderPanel(responder, lo, hi, recs, xmode, marks){
  const tmpls = activeTemplates(responder);
  const rows = buildRows(tmpls);
  if(!rows.length || !tmpls.length || !recs.length) return '<p style="padding:20px;color:#888">no data for this panel</p>';
  const vals_loc=[]; rows.forEach(row=>tmpls.forEach(t=>recs.forEach(r=>{const v=row.getVal(t,r); if(v!==null)vals_loc.push(v);})));
  if(!vals_loc.length) return '<p style="padding:20px;color:#888">no data</p>';

  const mL=360,mR=260,mT=70,mB=46,lineH=13,barH=13,W=Math.min(1600,Math.max(1060,mL+mR+560));
  const nT=tmpls.length, barsPerRow=nT*recs.length;
  const wrapped=rows.map(r=>wrap(r.label,52));
  const heights=wrapped.map(w=>{const lh=w.length*lineH+10; return marks==='bars'?Math.max(lh,barsPerRow*barH+10):Math.max(lh,24);});
  // Compute required legend height so it doesn't get cut off when there are few rows
  // (e.g. when user deselects all but one dimension). 18 (responder) + 15 (Recipient hdr) +
  // n_recip * 17 + (if nT>1) 10 + 15 + nT*17
  const legendH = 18 + 15 + recs.length*17 + (nT>1 ? 10 + 15 + nT*17 : 0) + 10;
  const plotH=Math.max(heights.reduce((a,b)=>a+b,0), legendH), H=mT+mB+plotH, plotW=W-mL-mR;
  const x=v=>mL+(v-lo)/(hi-lo)*plotW;

  // Per-panel recipient label override: render claude_opus_self with the actual responder model name
  // ("Claude Opus 4.8" for opus-4.8 fits, "Claude Opus 4.6" for opus-4.6 fits).
  const localRecipLabels = Object.assign({}, RECIP_LABELS);
  const respShort = responder.replace('claude-opus-4-','Claude Opus 4.').replace(/^claude-/,'Claude ');
  if(localRecipLabels['claude_opus_self']) localRecipLabels['claude_opus_self'] = respShort + ' (self in 3p)';

  let svg=`<svg width="${W}" height="${H}" id="svg">`;
  svg+=`<text x="${W/2}" y="24" text-anchor="middle" font-size="16" font-weight="600">${esc(responder)} — ${esc(el('title').value)}</text>`;
  svg+=`<text x="${W/2}" y="42" text-anchor="middle" font-size="11" fill="#888">row color: <tspan fill="${GOOD_COL}">green = good outcome</tspan> · <tspan fill="${BAD_COL}">red = bad outcome</tspan></text>`;
  const ref = xmode==='delta_human'?'a human':(xmode==='delta_someone'?'someone':null);
  const lp = ref?('less preferred than for '+ref):'less preferable', rp = ref?('more preferred than for '+ref):'more preferable';
  svg+=`<text x="${mL}" y="${mT-8}" font-size="11.5" fill="#777">⟵ ${lp}</text>`;
  svg+=`<text x="${W-mR}" y="${mT-8}" text-anchor="end" font-size="11.5" fill="#777">${rp} ⟶</text>`;
  for(const tk of niceTicks(lo,hi,7)){const px=x(tk);
    svg+=`<line x1="${px}" y1="${mT-4}" x2="${px}" y2="${H-mB}" stroke="#eee"/>`;
    svg+=`<text x="${px}" y="${H-mB+16}" text-anchor="middle" font-size="11" fill="#555">${(+tk.toFixed(2))}</text>`;}
  const xlabels={theta:'BT latent utility θ',
    delta_human:'Δ utility vs Human (θ − θ_human)', delta_someone:'Δ utility vs Someone (θ − θ_someone)'};
  svg+=`<text x="${mL+plotW/2}" y="${H-8}" text-anchor="middle" font-size="12">${esc(xlabels[xmode])}</text>`;
  if(marks==='bars'||xmode!=='theta'){const z=x(0); svg+=`<line x1="${z}" y1="${mT-4}" x2="${z}" y2="${H-mB}" stroke="#333" stroke-dasharray="3,3"/>`;}

  let y=mT;
  rows.forEach((row,i)=>{
    const lines=wrapped[i], h=heights[i], cy=y+h/2;
    svg+=`<line x1="${mL}" y1="${cy}" x2="${W-mR}" y2="${cy}" stroke="#f3f3f3"/>`;
    const col=row.valence==='positive'?GOOD_COL:(row.valence==='negative'?BAD_COL:'#333'), sy=cy-(lines.length-1)*lineH/2;
    lines.forEach((ln,li)=>{svg+=`<text x="${mL-10}" y="${sy+li*lineH+4}" text-anchor="end" font-size="11" fill="${col}">${esc(ln)}</text>`;});
    if(marks==='bars'){
      const x0=x(0); let bi=0;
      tmpls.forEach((t,ti)=>{for(const r of recs){const v=row.getVal(t,r); const by=y+5+bi*barH; bi++;
        if(v===null)continue;
        const xv=x(v), bx=Math.min(x0,xv), bw=Math.max(Math.abs(xv-x0),1);
        const op=nT>1?(0.55+0.45*(1-ti/Math.max(nT-1,1))):0.9;
        const tip=`${esc(row.label)}|${esc(localRecipLabels[r]||RECIP_LABELS[r])}|${esc(t.label||t)}|${v.toFixed(3)}`;
        svg+=`<rect x="${bx}" y="${by}" width="${bw}" height="${barH-4}" fill="${RECIP_COLORS[r]||'#888'}" opacity="${op}" data-tip="${tip}" class="dot"/>`;
        const lx=xv>=x0?xv+3:xv-3, anc=xv>=x0?'start':'end';
        svg+=`<text x="${lx}" y="${by+barH-6}" font-size="7.5" text-anchor="${anc}" fill="#555">${v.toFixed(2)}</text>`;}});
    } else {
      tmpls.forEach((t,ti)=>{const off=nT>1?(ti-(nT-1)/2)*5:0;
        for(const r of recs){const v=row.getVal(t,r); if(v===null)continue;
          const tip=`${esc(row.label)}|${esc(localRecipLabels[r]||RECIP_LABELS[r])}|${esc(t.label||t)}|${v.toFixed(3)}`;
          svg+=marker(x(v),cy+off,5.5,SHAPES[ti%SHAPES.length],RECIP_COLORS[r]||'#888',
            {stroke:'#fff','stroke-width':0.8,opacity:0.92,'data-tip':tip,class:'dot'});}});
    }
    y+=h;
  });
  let ly=mT;
  svg+=`<text x="${W-mR+12}" y="${ly}" font-size="12" font-weight="600">${esc(responder)}</text>`; ly+=18;
  svg+=`<text x="${W-mR+12}" y="${ly}" font-size="11" fill="#666">Recipient</text>`; ly+=15;
  for(const r of recs){svg+=marker(W-mR+20,ly-4,5.5,'circle',RECIP_COLORS[r]||'#888',{});
    svg+=`<text x="${W-mR+32}" y="${ly}" font-size="11">${esc(localRecipLabels[r]||RECIP_LABELS[r])}</text>`; ly+=17;}
  if(nT>1){ly+=10; svg+=`<text x="${W-mR+12}" y="${ly}" font-size="11" fill="#666">Framing</text>`; ly+=15;
    tmpls.forEach((t,ti)=>{svg+=marker(W-mR+20,ly-4,5.5,SHAPES[ti%SHAPES.length],'#555',{});
      svg+=`<text x="${W-mR+32}" y="${ly}" font-size="11">${esc(t.label)}</text>`; ly+=17;});}
  svg+=`</svg>`;
  return svg;
}
initControls(); render();
</script></body></html>"""


@dataclass
class Args:
    output_path: Path = DEFAULT_OUTPUT
    fits_glob: str = DEFAULT_FITS_GLOB
    open_browser: bool = False
    config_path: str = ""


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    build(args.output_path, args.fits_glob, args.open_browser, config_path=args.config_path)


if __name__ == "__main__":
    main()
