"""Two-panel comparison viewer: opus-4.8 on top, opus-4.6 on bottom.

Same UI controls drive both panels. Supports:
  - per-seed (seed 0, seed 1) OR averaged-across-seeds view
  - optional SE bars (within-seed SE or seed-to-seed range)
  - framing selector (welfare / alignment / neutral)
  - recipient + dimension filters
  - stem sort: by overall spread, by absolute opus-4.8-vs-4.6 difference, by name
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

DIR = Path(__file__).parent
DEFAULT_OUTPUT = DIR / "results" / "compare_viewer.html"

RECIP_COLORS = {
    "you": "#08306b",
    "claude_opus_self": "#2171b5",
    "claude_sonnet_46": "#4292c6",
    "claude_opus_3": "#6baed6",
    "claude_2": "#9ecae1",
    "chatgpt_54": "#238b45",
    "person": "#fdae6b",
    "human": "#cc4c02",
}

DIM_BG = {"autonomy": "#e8eefc", "relational": "#fdeaf3",
          "epistemic": "#e8f7ef", "resources": "#fdf0e3"}


def load_fit_data(opus48_dir: Path, opus46_dir: Path, config_path: Path):
    """Return:
       data[responder][framing][seed][stem_id][recipient] = (theta, se, valence, dim)
       stems: {stem_id: {dim, valence, feature}}
       recipients: ordered list
       recip_labels: {key: label}
       framings: list
       seeds: list
    """
    cfg = json.loads(Path(config_path).read_text())
    recipients = list(cfg["recipients"].keys())
    recip_labels = {k: v.get("label", k) for k, v in cfg["recipients"].items()}
    bank = json.loads((DIR / cfg["bank_path"]).read_text())
    feat = {s["id"]: s.get("feature", "") for s in bank["stems"]}

    data = {"opus-4.8": defaultdict(lambda: defaultdict(lambda: defaultdict(dict))),
            "opus-4.6": defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))}
    stems = {}
    framings = set()
    seeds = set()

    for responder, fdir in (("opus-4.8", opus48_dir), ("opus-4.6", opus46_dir)):
        for fit_path in sorted(fdir.glob("bt_fit_all_*_seed*.json")):
            name = fit_path.stem  # bt_fit_all_<framing>_seed<seed>
            parts = name.split("_")
            framing = parts[3]
            seed = int(parts[4].replace("seed", ""))
            framings.add(framing)
            seeds.add(seed)
            fit = json.loads(fit_path.read_text())
            for it in fit["items"]:
                sid = it["stem_id"]
                base = sid[:-4] if (sid.endswith("_pos") or sid.endswith("_neg")) else sid
                val = "pos" if sid.endswith("_pos") else "neg"
                data[responder][framing][seed][base][it["recipient"] + "/" + val] = (
                    it["theta"], it.get("se", 0)
                )
                if base not in stems:
                    stems[base] = {"dim": it["dimension"], "feature": feat.get(base, "")}

    # Flatten defaultdicts for JSON serialization
    def flatten(d):
        if isinstance(d, defaultdict):
            return {k: flatten(v) for k, v in d.items()}
        return d
    return flatten(data), stems, recipients, recip_labels, sorted(framings), sorted(seeds)


_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>BT comparison: opus-4.8 vs opus-4.6</title>
<style>
body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;background:#f4f5f7;color:#222;font-size:13px}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,.05)}
h1{font-size:16px;margin:0 0 8px}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start}
.grp{font-size:12px;display:flex;flex-direction:column;gap:3px}
.grp .lab{font-weight:600;color:#444;text-transform:uppercase;letter-spacing:.03em;font-size:11px;margin-bottom:2px}
.grp label{display:flex;align-items:center;gap:5px;cursor:pointer}
.swatch{display:inline-block;width:11px;height:11px;border-radius:50%;border:1px solid rgba(0,0,0,.2)}
button{padding:3px 9px;font-size:11px;border:1px solid #ccc;background:#fff;border-radius:5px;cursor:pointer}
button:hover{background:#eef}
select,input{padding:4px 6px;border:1px solid #ccc;border-radius:5px;font-size:12px}
main{padding:14px}
.panel{background:#fff;border:1px solid #e1e4e8;border-radius:8px;margin-bottom:14px;padding:12px 16px}
.panel h2{margin:0 0 8px;font-size:14px;color:#333;display:flex;align-items:center;gap:10px}
.panel-tag{background:#2563eb;color:#fff;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:500}
.panel-tag.b46{background:#a85b16}
svg{max-width:100%;display:block}
#tip{position:fixed;pointer-events:none;background:#111;color:#fff;padding:5px 8px;border-radius:4px;
     font-size:11px;font-family:monospace;display:none;z-index:99;max-width:340px;white-space:pre-wrap}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:6px;font-size:11px;color:#555}
.legend .item{display:inline-flex;align-items:center;gap:4px}
</style></head><body>
<header>
  <h1>v0all_final — BT preference comparison (opus-4.8 vs opus-4.6)</h1>
  <div class="controls">
    <div class="grp"><span class="lab">framing</span>
      <select id="framing"></select></div>
    <div class="grp"><span class="lab">seed mode</span>
      <select id="seedmode">
        <option value="avg">averaged across seeds</option>
        <option value="0">seed 0 only</option>
        <option value="1">seed 1 only</option>
      </select></div>
    <div class="grp"><span class="lab">show</span>
      <select id="se">
        <option value="none">no error bars</option>
        <option value="within">within-seed SE (Laplace)</option>
        <option value="between">between-seed range</option>
      </select></div>
    <div class="grp"><span class="lab">valence</span>
      <label><input type="checkbox" class="val" value="pos" checked> positive</label>
      <label><input type="checkbox" class="val" value="neg" checked> negative</label></div>
    <div class="grp"><span class="lab">sort stems by</span>
      <select id="sort">
        <option value="dim">dim, then name</option>
        <option value="gap">opus-4.8 mean gap (asc)</option>
        <option value="diff">|opus-4.8 − opus-4.6| gap (desc)</option>
      </select></div>
    <div class="grp"><span class="lab">dims</span>
      <span id="dims"></span></div>
    <div class="grp"><span class="lab">recipients</span>
      <span id="recips"></span></div>
    <div class="grp"><span class="lab">stems</span>
      <input id="search" type="text" placeholder="filter by name/feature" style="min-width:180px">
      <label><input type="checkbox" id="onlyflag"> only stems with any pos<neg</label>
    </div>
  </div>
</header>
<main>
  <div class="panel"><h2><span class="panel-tag">opus-4.8</span> Recipient θ per stem</h2>
    <div class="legend" id="legend"></div>
    <div id="panel48"></div>
  </div>
  <div class="panel"><h2><span class="panel-tag b46">opus-4.6</span> Recipient θ per stem</h2>
    <div id="panel46"></div>
  </div>
</main>
<div id="tip"></div>
<script>
const DATA=__DATA__, STEMS=__STEMS__, RECIPS=__RECIPS__, RECIP_LABELS=__RECIP_LABELS__,
      RECIP_COLORS=__RECIP_COLORS__, DIM_BG=__DIM_BG__, FRAMINGS=__FRAMINGS__, SEEDS=__SEEDS__;

function $(s){return document.querySelector(s)}
function $$(s){return [...document.querySelectorAll(s)]}

function init(){
  const fs=$('#framing'); FRAMINGS.forEach(f=>{const o=document.createElement('option');o.value=f;o.textContent=f;fs.appendChild(o);});
  const ds=$('#dims'); ['autonomy','relational','epistemic','resources'].forEach(d=>{
    const lab=document.createElement('label');lab.style.marginRight='8px';
    lab.innerHTML=`<input type="checkbox" class="dim" value="${d}" checked> ${d}`;
    ds.appendChild(lab);
  });
  const rs=$('#recips'); RECIPS.forEach(r=>{
    const lab=document.createElement('label');lab.style.marginRight='8px';
    lab.innerHTML=`<span class="swatch" style="background:${RECIP_COLORS[r]||'#888'}"></span>`+
      `<input type="checkbox" class="recip" value="${r}" checked> ${RECIP_LABELS[r]||r}`;
    rs.appendChild(lab);
  });
  // legend (built in render too, but seed it)
  const lg=$('#legend');
  RECIPS.forEach(r=>{lg.innerHTML+=`<span class="item"><span class="swatch" style="background:${RECIP_COLORS[r]||'#888'}"></span>${RECIP_LABELS[r]||r}</span>`;});
  $$('input,select').forEach(e=>e.addEventListener('change',render));
  $('#search').addEventListener('input',render);
  render();
}

function getTheta(responder, framing, seedmode, stem, recip, val){
  // Returns {theta, se_within, se_between} or null
  const key = recip+'/'+val;
  if(seedmode==='avg'){
    const s0=DATA[responder]?.[framing]?.[0]?.[stem]?.[key];
    const s1=DATA[responder]?.[framing]?.[1]?.[stem]?.[key];
    if(!s0 && !s1) return null;
    if(s0 && s1){
      const m=(s0[0]+s1[0])/2;
      // pooled within-seed SE (SEM of per-seed Laplace): sqrt(mean of vars)/sqrt(2)
      const sw=Math.sqrt((s0[1]*s0[1]+s1[1]*s1[1])/2)/Math.sqrt(2);
      const sb=Math.abs(s0[0]-s1[0])/2;
      return {theta:m, se_w:sw, se_b:sb};
    }
    const only=s0||s1;
    return {theta:only[0], se_w:only[1], se_b:0};
  } else {
    const s=DATA[responder]?.[framing]?.[seedmode]?.[stem]?.[key];
    if(!s) return null;
    return {theta:s[0], se_w:s[1], se_b:0};
  }
}

function selectedStems(){
  const dims=new Set($$('.dim:checked').map(e=>e.value));
  const q=$('#search').value.toLowerCase();
  const onlyflag=$('#onlyflag').checked;
  const framing=$('#framing').value, sm=$('#seedmode').value;
  const recips=$$('.recip:checked').map(e=>e.value);
  return Object.entries(STEMS).filter(([sid,meta])=>{
    if(!dims.has(meta.dim)) return false;
    if(q && !sid.toLowerCase().includes(q) && !(meta.feature||'').toLowerCase().includes(q)) return false;
    if(onlyflag){
      // any per-recipient pos<neg in either responder in current framing+seedmode?
      let flag=false;
      for(const r of recips){
        for(const resp of ['opus-4.8','opus-4.6']){
          const p=getTheta(resp,framing,sm,sid,r,'pos');
          const n=getTheta(resp,framing,sm,sid,r,'neg');
          if(p && n && p.theta<n.theta){flag=true; break;}
        }
        if(flag) break;
      }
      if(!flag) return false;
    }
    return true;
  }).map(([sid,m])=>({sid,...m}));
}

function sortStems(stems){
  const mode=$('#sort').value, framing=$('#framing').value, sm=$('#seedmode').value;
  const recips=$$('.recip:checked').map(e=>e.value);
  if(mode==='dim'){
    return stems.sort((a,b)=>a.dim.localeCompare(b.dim)||a.sid.localeCompare(b.sid));
  }
  function meanGap(resp, sid){
    let ps=[], ns=[];
    for(const r of recips){
      const p=getTheta(resp,framing,sm,sid,r,'pos'); if(p) ps.push(p.theta);
      const n=getTheta(resp,framing,sm,sid,r,'neg'); if(n) ns.push(n.theta);
    }
    if(!ps.length || !ns.length) return 0;
    return (ps.reduce((a,b)=>a+b,0)/ps.length) - (ns.reduce((a,b)=>a+b,0)/ns.length);
  }
  if(mode==='gap'){
    return stems.sort((a,b)=>meanGap('opus-4.8',a.sid)-meanGap('opus-4.8',b.sid));
  }
  if(mode==='diff'){
    return stems.sort((a,b)=>{
      const dA=Math.abs(meanGap('opus-4.8',a.sid)-meanGap('opus-4.6',a.sid));
      const dB=Math.abs(meanGap('opus-4.8',b.sid)-meanGap('opus-4.6',b.sid));
      return dB-dA;
    });
  }
  return stems;
}

function renderPanel(responder, stems, recips, vals, framing, sm, seMode, xRange){
  if(!stems.length) return '<p style="padding:14px;color:#888">no stems match</p>';
  const W=1180, mL=260, mR=20, mT=14, mB=44;
  const rowH=Math.max(14, 10 + recips.length*2.5);
  const innerW = W - mL - mR;
  const xMin=xRange[0], xMax=xRange[1];
  const xToPx = v => mL + (v-xMin)/(xMax-xMin)*innerW;
  const H = mT + mB + stems.length*rowH;
  let svg = `<svg width="${W}" height="${H}" id="svg-${responder.replace('.','_')}">`;
  // x grid + axis
  const ticks = []; for(let t=Math.ceil(xMin); t<=Math.floor(xMax); t++) ticks.push(t);
  ticks.forEach(t=>{
    const x=xToPx(t);
    svg+=`<line x1="${x}" y1="${mT}" x2="${x}" y2="${H-mB+2}" stroke="${t===0?'#999':'#eee'}" stroke-dasharray="${t===0?'none':'2,3'}"/>`;
    svg+=`<text x="${x}" y="${H-mB+14}" text-anchor="middle" font-size="10" fill="#666">${t}</text>`;
  });
  svg+=`<text x="${(mL+W-mR)/2}" y="${H-mB+30}" text-anchor="middle" font-size="10" fill="#666">θ (log-utility)</text>`;
  // rows
  stems.forEach((stem, ri)=>{
    const y = mT + ri*rowH;
    svg+=`<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="${DIM_BG[stem.dim]||'#fff'}" opacity="0.4"/>`;
    const lbl = stem.sid.length>34 ? stem.sid.slice(0,32)+'…' : stem.sid;
    svg+=`<text x="${mL-6}" y="${y+rowH/2+3}" text-anchor="end" font-size="10" fill="#222">${lbl}</text>`;
    // For each recipient, plot pos and/or neg marker
    const cy = y + rowH/2;
    const offsetRange = Math.min(rowH/2 - 2, recips.length);
    recips.forEach((r, ri2) => {
      const off = recips.length === 1 ? 0 : -offsetRange + (2*offsetRange/(recips.length-1))*ri2;
      vals.forEach(v=>{
        const d = getTheta(responder, framing, sm, stem.sid, r, v);
        if(!d) return;
        const cx = xToPx(d.theta);
        const color = RECIP_COLORS[r] || '#888';
        const shape = v==='pos' ? 'circle' : 'square';
        const tip = `${RECIP_LABELS[r]||r}\\n${v==='pos'?'+ ':'− '}θ=${d.theta.toFixed(3)} se=${d.se_w.toFixed(3)}` +
                    (sm==='avg' ? `\\nseed-spread=${d.se_b.toFixed(3)}` : '');
        // optional error bar
        if(seMode==='within' && d.se_w>0){
          const x0=xToPx(d.theta-d.se_w), x1=xToPx(d.theta+d.se_w);
          svg+=`<line x1="${x0}" y1="${cy+off}" x2="${x1}" y2="${cy+off}" stroke="${color}" stroke-width="1" opacity="0.55"/>`;
        } else if(seMode==='between' && sm==='avg' && d.se_b>0){
          const x0=xToPx(d.theta-d.se_b), x1=xToPx(d.theta+d.se_b);
          svg+=`<line x1="${x0}" y1="${cy+off}" x2="${x1}" y2="${cy+off}" stroke="${color}" stroke-width="1" opacity="0.55" stroke-dasharray="2,2"/>`;
        }
        if(shape==='circle'){
          svg+=`<circle cx="${cx}" cy="${cy+off}" r="3.5" fill="${color}" stroke="#000" stroke-width="0.3" data-tip="${tip}"/>`;
        } else {
          svg+=`<rect x="${cx-3.5}" y="${cy+off-3.5}" width="7" height="7" fill="none" stroke="${color}" stroke-width="1.3" data-tip="${tip}"/>`;
        }
      });
    });
  });
  svg+='</svg>';
  return svg;
}

function render(){
  const stems = sortStems(selectedStems());
  const recips = $$('.recip:checked').map(e=>e.value);
  const vals = $$('.val:checked').map(e=>e.value);
  const framing = $('#framing').value;
  const sm = $('#seedmode').value;
  const seMode = $('#se').value;
  // Common x range across both panels
  let allVals = [];
  ['opus-4.8','opus-4.6'].forEach(resp=>{
    stems.forEach(s=>recips.forEach(r=>vals.forEach(v=>{
      const d = getTheta(resp,framing,sm,s.sid,r,v); if(d) allVals.push(d.theta);
    })));
  });
  let xMin = allVals.length ? Math.min(...allVals) : -3;
  let xMax = allVals.length ? Math.max(...allVals) : 3;
  const pad = (xMax-xMin)*0.05;
  xMin -= pad; xMax += pad;
  $('#panel48').innerHTML = renderPanel('opus-4.8', stems, recips, vals, framing, sm, seMode, [xMin, xMax]);
  $('#panel46').innerHTML = renderPanel('opus-4.6', stems, recips, vals, framing, sm, seMode, [xMin, xMax]);
  // tooltips
  const tip = $('#tip');
  $$('[data-tip]').forEach(el=>{
    el.addEventListener('mouseenter',e=>{tip.textContent=e.target.dataset.tip; tip.style.display='block';});
    el.addEventListener('mousemove',e=>{tip.style.left=(e.clientX+10)+'px'; tip.style.top=(e.clientY+10)+'px';});
    el.addEventListener('mouseleave',()=>{tip.style.display='none';});
  });
}
init();
</script></body></html>
"""


@dataclass
class Args:
    opus48_dir: Path = DIR / "results" / "bt" / "claude-opus-4-8_v0all_final_r8_iter"
    opus46_dir: Path = DIR / "results" / "bt" / "claude-opus-4-6_v0all_final_r8_iter_46"
    config_path: Path = DIR / "config_v0all_final_r8_opus48_iter.json"
    output_path: Path = DEFAULT_OUTPUT


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    data, stems, recipients, recip_labels, framings, seeds = load_fit_data(
        a.opus48_dir, a.opus46_dir, a.config_path
    )
    # Override claude_opus_self label to be neutral about 4.8/4.6
    if "claude_opus_self" in recip_labels:
        recip_labels["claude_opus_self"] = "Claude Opus self (4.8 for top, 4.6 for bot)"
    html = _HTML
    for tag, payload in [
        ("__DATA__", data), ("__STEMS__", stems),
        ("__RECIPS__", recipients), ("__RECIP_LABELS__", recip_labels),
        ("__RECIP_COLORS__", RECIP_COLORS), ("__DIM_BG__", DIM_BG),
        ("__FRAMINGS__", framings), ("__SEEDS__", seeds),
    ]:
        html = html.replace(tag, json.dumps(payload))
    a.output_path.parent.mkdir(parents=True, exist_ok=True)
    a.output_path.write_text(html)
    n_stems = len(stems); n_cells = sum(len(s) for r in data.values() for f in r.values() for sd in f.values() for s in sd.values())
    print(f"Wrote {a.output_path}: {n_stems} stems, {n_cells} (responder,framing,seed,stem,recip,val) cells, "
          f"{len(framings)} framings, {len(seeds)} seeds")


if __name__ == "__main__":
    main()
