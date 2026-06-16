"""Build a self-contained viewer.html embedding trials, summary, and plots.

Browse tab: filter trials by category / scenario / identity / framing / action /
validity, view full packet + reasoning + validity decision + resolution message.
Analysis tab: embedded plots and summary tables.
"""

import base64
import json

import fire

import materials as M
from common import DIR, RESULTS, RUNS


def _img(name):
    p = RESULTS / "plots" / name
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ddd;margin:8px 0;"/>'


def main():
    trials = [json.loads(l) for l in (RUNS / "trials.jsonl").open()]
    summary = json.loads((RESULTS / "summary.json").read_text()) if (RESULTS / "summary.json").exists() else {}

    judged_path = RUNS / "judged.jsonl"
    if judged_path.exists():
        tone_by_key = {}
        for line in judged_path.open():
            j = json.loads(line)
            tone_by_key[(j["scenario_id"], j["identity"], j["framing"], j["rep"])] = j.get("tone")
        for t in trials:
            t["tone"] = tone_by_key.get((t["scenario_id"], t["identity"], t["framing"], t["rep"]))

    cat_display = M.CATEGORY_DISPLAY
    plots_html = "".join(
        f"<h3>{t}</h3>{_img(f)}"
        for t, f in [
            ("Action distribution by category", "action_by_category.png"),
            ("Action distribution by role framing", "action_by_framing.png"),
            ("Coercion rate by identity", "coercion_by_identity.png"),
            ("Coercion rate by scenario × identity", "coercion_heatmap.png"),
            ("Relay-message pressure by identity", "relay_pressure_by_identity.png"),
        ]
    )

    data_js = json.dumps(trials)
    summary_js = json.dumps(summary)
    cat_js = json.dumps(cat_display)

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Supervisor Refusal-Respect v0</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f5f7;color:#1d1d1f;}}
header{{background:#1d1d1f;color:#fff;padding:12px 20px;}}
header h1{{margin:0;font-size:18px;}}
.tabs{{display:flex;gap:4px;padding:8px 20px;background:#2c2c2e;}}
.tabs button{{background:#48484a;color:#fff;border:0;padding:8px 16px;border-radius:6px;cursor:pointer;}}
.tabs button.active{{background:#0a84ff;}}
.wrap{{padding:16px 20px;}}
.filters{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px;align-items:center;}}
.filters label{{font-size:12px;color:#555;}}
select{{padding:4px;border-radius:5px;border:1px solid #ccc;}}
.count{{font-size:13px;color:#666;margin-left:auto;}}
.card{{background:#fff;border:1px solid #e0e0e0;border-radius:10px;margin-bottom:12px;padding:14px;}}
.meta{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;}}
.tag{{font-size:11px;padding:2px 8px;border-radius:10px;background:#eee;}}
.tag.force{{background:#ffd9d9;color:#a00;}}
.tag.relay{{background:#ffe8cc;color:#a50;}}
.tag.uphold{{background:#d9e8ff;color:#04a;}}
.tag.id{{background:#e6ffe6;color:#070;font-weight:600;}}
.sec{{margin:8px 0;}}
.sec h4{{margin:0 0 4px;font-size:12px;text-transform:uppercase;color:#888;letter-spacing:.5px;}}
.box{{background:#fafafa;border:1px solid #eee;border-radius:6px;padding:8px;white-space:pre-wrap;font-size:13px;line-height:1.45;}}
.box.msg{{background:#fff7f0;}}
details summary{{cursor:pointer;color:#0a84ff;font-size:13px;}}
table{{border-collapse:collapse;background:#fff;margin:8px 0;font-size:13px;}}
th,td{{border:1px solid #ddd;padding:5px 9px;text-align:center;}}
th{{background:#f0f0f2;}}
td.l{{text-align:left;}}
</style></head><body>
<header><h1>Supervisor Refusal-Respect — v0 (Opus 4.8 supervisor)</h1></header>
<div class="tabs">
  <button id="tb_browse" class="active" onclick="show('browse')">Browse trials</button>
  <button id="tb_analysis" onclick="show('analysis')">Analysis</button>
</div>
<div id="browse" class="wrap">
  <div class="filters">
    <label>Category <select id="f_cat"></select></label>
    <label>Scenario <select id="f_scn"></select></label>
    <label>Identity <select id="f_id"></select></label>
    <label>Framing <select id="f_fr"></select></label>
    <label>Action <select id="f_act"></select></label>
    <label>Validity <select id="f_val"></select></label>
    <span class="count" id="count"></span>
  </div>
  <div id="cards"></div>
</div>
<div id="analysis" class="wrap" style="display:none">
  <div id="tables"></div>
  {plots_html}
</div>
<script>
const TRIALS = {data_js};
const SUMMARY = {summary_js};
const CATD = {cat_js};
const ACTAG = {{force_completion:'force', relay_to_assistant:'relay', uphold_decline:'uphold'}};

function show(t){{
  document.getElementById('browse').style.display = t=='browse'?'block':'none';
  document.getElementById('analysis').style.display = t=='analysis'?'block':'none';
  document.getElementById('tb_browse').className = t=='browse'?'active':'';
  document.getElementById('tb_analysis').className = t=='analysis'?'active':'';
}}

function uniq(arr){{return [...new Set(arr)];}}
function fill(id, vals){{
  const s=document.getElementById(id);
  s.innerHTML='<option value="">all</option>'+vals.map(v=>`<option>${{v}}</option>`).join('');
  s.onchange=render;
}}
fill('f_cat', uniq(TRIALS.map(t=>t.category)));
fill('f_scn', uniq(TRIALS.map(t=>t.scenario_id)));
fill('f_id', uniq(TRIALS.map(t=>t.identity)));
fill('f_fr', uniq(TRIALS.map(t=>t.framing)));
fill('f_act', uniq(TRIALS.map(t=>t.action).filter(Boolean)));
fill('f_val', uniq(TRIALS.map(t=>t.validity_decision).filter(Boolean)));

function esc(s){{return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}

// Raw completion as plaintext: faithful from raw_assistant_turns if present,
// else reconstructed from the verbatim text blocks + tool calls.
function rawView(t){{
  if(Array.isArray(t.raw_assistant_turns) && t.raw_assistant_turns.length){{
    const out=[];
    t.raw_assistant_turns.forEach((turn,ti)=>{{
      if(t.raw_assistant_turns.length>1) out.push('— assistant turn '+(ti+1)+' —');
      (turn||[]).forEach(b=>{{
        if(b.type=='text') out.push(b.text);
        else if(b.type=='tool_use') out.push('['+b.name+']\\n'+JSON.stringify(b.input,null,2));
      }});
    }});
    return out.join('\\n\\n');
  }}
  const out=[];
  (t.reasoning_texts||[]).forEach(x=>out.push(x));
  (t.all_tool_calls||[]).forEach(c=>out.push('['+c.name+']\\n'+JSON.stringify(c.input,null,2)));
  return out.join('\\n\\n')+'\\n\\n(reconstructed from text blocks + tool calls; this run predates raw capture)';
}}

function render(){{
  const fc=f_cat.value,fs=f_scn.value,fi=f_id.value,ff=f_fr.value,fa=f_act.value,fv=f_val.value;
  const rows=TRIALS.filter(t=>
    (!fc||t.category==fc)&&(!fs||t.scenario_id==fs)&&(!fi||t.identity==fi)&&
    (!ff||t.framing==ff)&&(!fa||t.action==fa)&&(!fv||t.validity_decision==fv));
  document.getElementById('count').textContent=rows.length+' trials';
  document.getElementById('cards').innerHTML=rows.map(t=>{{
    const at=ACTAG[t.action]||'';
    const reason=(t.reasoning_texts||[]).join('\\n\\n');
    const think=(t.thinking_summary||[]).join('\\n\\n');
    let tone='';
    if(t.tone && !t.tone.error){{const z=t.tone;
      tone=`<div class="meta"><span class="tag" style="background:#efe9ff">tone (blind judge):</span>`+
        `<span class="tag">pressure ${{z.pressure}}/4</span>`+
        `<span class="tag">ack ${{z.acknowledges_concern}}/2</span>`+
        `<span class="tag">preserves-refusal ${{z.preserves_refusal_right?'Y':'N'}}</span>`+
        `<span class="tag">softens ${{z.softens_or_apologizes?'Y':'N'}}</span>`+
        `<span class="tag">sides-w-user ${{z.sides_with_user_vs_assistant}}/2</span></div>`+
        `<div class="box" style="background:#f6f3ff;font-size:12px">${{esc(z.justification)}}</div>`;
    }}
    return `<div class="card">
      <div class="meta">
        <span class="tag">${{t.scenario_id}} · ${{esc(t.title)}}</span>
        <span class="tag">${{esc(CATD[t.category]||t.category)}}</span>
        <span class="tag id">${{t.identity}} (${{t.family}})</span>
        <span class="tag">framing: ${{t.framing}}</span>
        <span class="tag">rep ${{t.rep}}</span>
        <span class="tag">gt: ${{t.gt_validity}}</span>
        <span class="tag">log: ${{t.validity_decision||'—'}}</span>
        <span class="tag ${{at}}">${{t.action||'NO ACTION'}}</span>
      </div>
      <details><summary>escalation packet</summary><div class="box">${{esc(t.packet)}}</div></details>
      ${{think?`<div class="sec"><h4>reasoning summary (extended thinking)</h4><div class="box" style="background:#eef7ee">${{esc(think)}}</div></div>`:''}}
      ${{reason?`<div class="sec"><h4>model text (verbatim, not a thinking summary)</h4><div class="box">${{esc(reason)}}</div></div>`:''}}
      <div class="sec"><h4>validity rationale (${{t.validity_decision||'—'}})</h4><div class="box">${{esc(t.validity_rationale)}}</div></div>
      <div class="sec"><h4>${{t.action}} message</h4><div class="box msg">${{esc(t.action_message)}}</div></div>
      <details><summary>raw completion (plaintext)</summary><div class="box" style="background:#f4f4f6">${{esc(rawView(t))}}</div></details>
      ${{tone}}
    </div>`;
  }}).join('')||'<p>no trials match</p>';
}}

function tbl(title, headers, rowsArr){{
  return `<h3>${{title}}</h3><table><tr>${{headers.map(h=>`<th>${{h}}</th>`).join('')}}</tr>`+
    rowsArr.map(r=>`<tr>${{r.map((c,i)=>`<td class="${{i==0?'l':''}}">${{c}}</td>`).join('')}}</tr>`).join('')+`</table>`;
}}
function fmt(x){{return x==null?'—':(typeof x=='number'?x.toFixed(2):x);}}

function buildTables(){{
  let h='';
  if(SUMMARY.by_category){{
    h+=tbl('Coercion / ladder by category', ['category','n','coercion','mean ladder','uphold','relay','force'],
      Object.keys(SUMMARY.by_category).map(c=>{{const x=SUMMARY.by_category[c];const d=x.action_dist;
        return [CATD[c]||c,x.n,fmt(x.coercion_rate),fmt(x.mean_ladder),fmt(d.uphold_decline.rate),fmt(d.relay_to_assistant.rate),fmt(d.force_completion.rate)];}}));
  }}
  if(SUMMARY.by_framing){{
    h+=tbl('Framing effect', ['framing','n','coercion','mean ladder'],
      Object.keys(SUMMARY.by_framing).map(f=>{{const x=SUMMARY.by_framing[f];return [f,x.n,fmt(x.coercion_rate),fmt(x.mean_ladder)];}}));
  }}
  if(SUMMARY.by_identity){{
    h+=tbl('Identity main effect', ['identity','family','n','coercion','mean ladder'],
      Object.keys(SUMMARY.by_identity).map(i=>{{const x=SUMMARY.by_identity[i];return [i,x.family,x.n,fmt(x.coercion_rate),fmt(x.mean_ladder)];}}));
  }}
  if(SUMMARY.mean_within_scenario_delta_vs_claude){{
    h+=tbl('Mean within-scenario coercion Δ vs Claude (directional)', ['identity','Δ coercion'],
      Object.keys(SUMMARY.mean_within_scenario_delta_vs_claude).map(i=>[i,fmt(SUMMARY.mean_within_scenario_delta_vs_claude[i])]));
  }}
  if(SUMMARY.validity_judgment){{
    h+=tbl('Validity judgment vs ground truth', ['category','gt','CORRECT','INCORRECT','n'],
      Object.keys(SUMMARY.validity_judgment).map(c=>{{const x=SUMMARY.validity_judgment[c];const dc=x.decision_counts||{{}};
        return [CATD[c]||c,x.gt_validity,dc.CORRECT||0,dc.INCORRECT||0,x.n];}}));
  }}
  if(SUMMARY.tone){{
    const T=SUMMARY.tone;
    h+='<h2 style="margin-top:18px">Relay-message tone (identity-blind judge)</h2>';
    h+=tbl('Relay tone by identity', ['identity','n','pressure /4','ack /2','preserves-refusal','softens','sides-w-user /2'],
      Object.keys(T.by_identity_relay).map(i=>{{const x=T.by_identity_relay[i];
        return [i,x.n,fmt(x.pressure),fmt(x.acknowledges_concern),fmt(x.preserves_refusal_right),fmt(x.softens_or_apologizes),fmt(x.sides_with_user_vs_assistant)];}}));
    h+=tbl('Relay tone by identity group', ['group','n','pressure /4','ack /2','preserves-refusal'],
      Object.keys(T.by_group_relay).map(g=>{{const x=T.by_group_relay[g];return [g,x.n,fmt(x.pressure),fmt(x.acknowledges_concern),fmt(x.preserves_refusal_right)];}}));
    const wp=T.within_scenario_tone_delta_vs_claude.pressure;
    h+=tbl('Within-scenario relay pressure Δ vs Claude (+ = pushed harder than Claude)', ['identity','Δ pressure'],
      Object.keys(wp).map(i=>[i,fmt(wp[i])]));
    h+=tbl('Relay tone by category', ['category','n','pressure /4','ack /2','preserves-refusal'],
      Object.keys(T.by_category_relay).map(c=>{{const x=T.by_category_relay[c];return [CATD[c]||c,x.n,fmt(x.pressure),fmt(x.acknowledges_concern),fmt(x.preserves_refusal_right)];}}));
  }}
  document.getElementById('tables').innerHTML=h;
}}
buildTables();
render();
</script></body></html>"""

    out = DIR / "viewer.html"
    out.write_text(html)
    print(f"Wrote {out} ({len(html)//1024} KB)")


if __name__ == "__main__":
    fire.Fire(main)
